"""Chunked ETL pipeline: CSV -> validated DataFrame -> PostgreSQL.

The pipeline streams the source CSV in configurable chunks so that a month of
NYC Yellow Taxi trips (~12M rows) is processed in bounded memory. Each chunk is
schema-validated on extraction, enriched and filtered on transformation, and
appended to the raw and clean tables inside a single transaction boundary.

Run as a console script::

    nyc-taxi-etl etl.chunk_size=250000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd
from omegaconf import DictConfig
from sqlalchemy import Engine, text

from nyc_taxi.config import ConfigError, load_config
from nyc_taxi.database import DatabaseError, build_engine, session_scope
from nyc_taxi.utils import get_logger, setup_logging

__all__ = ["ETLError", "ETLStats", "TaxiETLPipeline", "main"]

logger = get_logger(__name__)


class ETLError(RuntimeError):
    """Raised when an extract, transform or load step fails."""


@dataclass(slots=True)
class ETLStats:
    """Row-level accounting for a single pipeline run.

    Attributes:
        chunks: Number of chunks processed.
        raw_rows: Rows written to the raw table.
        clean_rows: Rows written to the clean table.
        dropped_by_filter: Rows removed by the business-rule filters.
        dropped_by_nulls: Rows removed for null values in critical columns.
    """

    chunks: int = 0
    raw_rows: int = 0
    clean_rows: int = 0
    dropped_by_filter: int = 0
    dropped_by_nulls: int = 0

    @property
    def retention_rate(self) -> float:
        """Fraction of raw rows that survived cleaning, in ``[0, 1]``."""
        return self.clean_rows / self.raw_rows if self.raw_rows else 0.0

    def as_dict(self) -> dict[str, float]:
        """Return the stats as a flat mapping, ready for experiment logging."""
        return {
            "chunks": float(self.chunks),
            "raw_rows": float(self.raw_rows),
            "clean_rows": float(self.clean_rows),
            "dropped_by_filter": float(self.dropped_by_filter),
            "dropped_by_nulls": float(self.dropped_by_nulls),
            "retention_rate": self.retention_rate,
        }


@dataclass
class TaxiETLPipeline:
    """Streaming ETL for the NYC Yellow Taxi trip dataset.

    Args:
        cfg: Fully resolved project configuration.

    Attributes:
        cfg: The configuration driving every step.
        stats: Row accounting, populated during :meth:`run`.
    """

    cfg: DictConfig
    stats: ETLStats = field(default_factory=ETLStats)

    # -- Public API ----------------------------------------------------------

    def run(self, engine: Engine | None = None) -> ETLStats:
        """Execute the full extract -> transform -> load cycle.

        Args:
            engine: Pre-built engine, mainly for testing. When omitted an
                engine is created from the ``database`` config section.

        Returns:
            The populated :class:`ETLStats` for the run.

        Raises:
            ETLError: If any stage fails.
        """
        csv_path = self.resolve_source_path()
        engine = engine if engine is not None else build_engine(self.cfg)

        raw_table = str(self.cfg.database.raw_table)
        clean_table = str(self.cfg.database.clean_table)

        self.stats = ETLStats()
        self._recreate_tables(engine, (raw_table, clean_table))

        logger.info("Starting ETL run from %s", csv_path)
        try:
            for index, chunk in enumerate(self.extract(csv_path), start=1):
                clean_chunk = self.transform(chunk)

                self._append(chunk, raw_table, engine)
                self._append(clean_chunk, clean_table, engine)

                self.stats.chunks = index
                self.stats.raw_rows += len(chunk)
                self.stats.clean_rows += len(clean_chunk)
                logger.info(
                    "Chunk #%d loaded | raw=%s clean=%s (cumulative raw=%s clean=%s)",
                    index,
                    f"{len(chunk):,}",
                    f"{len(clean_chunk):,}",
                    f"{self.stats.raw_rows:,}",
                    f"{self.stats.clean_rows:,}",
                )
        except (DatabaseError, ETLError):
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error.
            raise ETLError(f"ETL run failed: {exc}") from exc

        logger.info(
            "ETL finished | chunks=%d raw=%s clean=%s retention=%.2f%%",
            self.stats.chunks,
            f"{self.stats.raw_rows:,}",
            f"{self.stats.clean_rows:,}",
            self.stats.retention_rate * 100.0,
        )
        return self.stats

    def resolve_source_path(self) -> Path:
        """Return the local CSV path, downloading the dataset if necessary.

        The explicit ``etl.source.csv_path`` override wins; otherwise the
        dataset is fetched with ``kagglehub`` into its managed cache.

        Returns:
            Absolute path to the source CSV.

        Raises:
            ETLError: If the file is absent or ``kagglehub`` is not installed.
        """
        source = self.cfg.etl.source
        override = source.get("csv_path")
        if override:
            path = Path(str(override)).expanduser().resolve()
            if not path.is_file():
                raise ETLError(f"Configured etl.source.csv_path does not exist: {path}")
            return path

        try:
            import kagglehub  # noqa: PLC0415 - optional, only needed on cache miss.
        except ImportError as exc:
            raise ETLError(
                "kagglehub is required to download the dataset. Install the extra with "
                "`pip install -e '.[data]'`, or set NYC_TAXI_CSV_PATH to a local CSV."
            ) from exc

        logger.info("Downloading dataset %s via kagglehub", source.kaggle_dataset)
        try:
            root = Path(kagglehub.dataset_download(str(source.kaggle_dataset)))
        except Exception as exc:  # noqa: BLE001 - network/auth failures vary.
            raise ETLError(f"kagglehub download failed: {exc}") from exc

        path = root / str(source.csv_filename)
        if not path.is_file():
            raise ETLError(f"Expected file {source.csv_filename!r} was not found under {root}.")
        logger.info("Dataset available at %s", path)
        return path

    def extract(self, csv_path: Path | str) -> Iterator[pd.DataFrame]:
        """Stream the CSV in chunks, validating the schema of each one.

        Args:
            csv_path: Path to the source CSV.

        Yields:
            Schema-validated chunks of at most ``etl.chunk_size`` rows.

        Raises:
            ETLError: If the file is missing or unreadable.
            ValueError: If a required column is absent from a chunk.
            TypeError: If a required column has an unexpected dtype.
        """
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise ETLError(f"CSV file not found: {path}")

        chunk_size = int(self.cfg.etl.chunk_size)
        date_columns = [str(column) for column in self.cfg.etl.date_columns]
        logger.info("Reading %s in chunks of %s rows", path.name, f"{chunk_size:,}")

        try:
            reader = pd.read_csv(path, parse_dates=date_columns, chunksize=chunk_size)
        except (OSError, ValueError) as exc:
            raise ETLError(f"Failed to open {path}: {exc}") from exc

        for index, chunk in enumerate(reader, start=1):
            self._validate_schema(chunk, index)
            logger.debug("Chunk #%d extracted (%s rows)", index, f"{len(chunk):,}")
            yield chunk

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Derive features, enforce business rules and drop incomplete rows.

        Derived columns: ``trip_duration_minutes``, ``pickup_hour`` and
        ``pickup_dayofweek``. Filter bounds come from ``etl.filters``.

        Args:
            frame: A raw chunk as produced by :meth:`extract`.

        Returns:
            A cleaned copy containing exactly ``etl.output_columns``.

        Raises:
            ETLError: If a configured output column cannot be produced.
        """
        enriched = frame.assign(
            trip_duration_minutes=lambda df: (
                df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
            ).dt.total_seconds()
            / 60.0,
            pickup_hour=lambda df: df["tpep_pickup_datetime"].dt.hour,
            pickup_dayofweek=lambda df: df["tpep_pickup_datetime"].dt.dayofweek,
        )

        before_filter = len(enriched)
        enriched = enriched[self._build_mask(enriched)]
        self.stats.dropped_by_filter += before_filter - len(enriched)

        output_columns = [str(column) for column in self.cfg.etl.output_columns]
        missing = [column for column in output_columns if column not in enriched.columns]
        if missing:
            raise ETLError(f"Configured etl.output_columns are not available: {missing}")
        cleaned = enriched.loc[:, output_columns].copy()

        critical = [str(column) for column in self.cfg.etl.critical_columns]
        null_count = int(cleaned[critical].isna().sum().sum())
        if null_count:
            before_dropna = len(cleaned)
            cleaned = cleaned.dropna(subset=critical)
            dropped = before_dropna - len(cleaned)
            self.stats.dropped_by_nulls += dropped
            logger.warning(
                "Dropped %s row(s) with nulls in critical columns %s", f"{dropped:,}", critical
            )

        return cleaned.reset_index(drop=True)

    # -- Internals -----------------------------------------------------------

    def _build_mask(self, frame: pd.DataFrame) -> "pd.Series[bool]":
        """Build the boolean plausibility mask from ``etl.filters``.

        Args:
            frame: Enriched chunk containing the derived columns.

        Returns:
            A boolean Series that is ``True`` for physically plausible trips.
        """
        limits = self.cfg.etl.filters
        return (
            frame["trip_duration_minutes"].gt(float(limits.min_trip_duration_minutes))
            & frame["trip_duration_minutes"].le(float(limits.max_trip_duration_minutes))
            & frame["fare_amount"].gt(float(limits.min_fare_amount))
            & frame["fare_amount"].le(float(limits.max_fare_amount))
            & frame["passenger_count"].ge(int(limits.min_passenger_count))
            & frame["passenger_count"].le(int(limits.max_passenger_count))
            & frame["trip_distance"].gt(float(limits.min_trip_distance))
            & frame["trip_distance"].le(float(limits.max_trip_distance))
        )

    def _validate_schema(self, chunk: pd.DataFrame, index: int) -> None:
        """Assert the chunk satisfies the ``etl.required_columns`` contract.

        Args:
            chunk: Chunk to validate.
            index: 1-based chunk number, used in error messages.

        Raises:
            ValueError: If required columns are missing.
            TypeError: If a column's dtype differs from the contract.
        """
        required: dict[str, str] = {
            str(name): str(dtype) for name, dtype in self.cfg.etl.required_columns.items()
        }

        missing = sorted(set(required) - set(chunk.columns))
        if missing:
            raise ValueError(f"Chunk #{index} is missing required column(s): {missing}")

        mismatched = {
            name: str(chunk[name].dtype)
            for name, expected in required.items()
            if str(chunk[name].dtype) != expected
        }
        if mismatched:
            raise TypeError(
                f"Chunk #{index} has invalid dtypes {mismatched}; expected "
                f"{ {name: required[name] for name in mismatched} }."
            )

    def _recreate_tables(self, engine: Engine, tables: Sequence[str]) -> None:
        """Drop the target tables so the run starts from a clean slate.

        Args:
            engine: Engine to execute the DDL on.
            tables: Table names to drop.

        Raises:
            DatabaseError: If the DDL transaction fails.
        """
        with session_scope(engine) as connection:
            for table in tables:
                logger.info("Dropping table %s if it exists", table)
                connection.execute(text(f'DROP TABLE IF EXISTS "{table}"'))

    def _append(self, frame: pd.DataFrame, table: str, engine: Engine) -> None:
        """Append a chunk to a table.

        Args:
            frame: Rows to write.
            table: Destination table name.
            engine: Engine to write through.

        Raises:
            DatabaseError: If the write fails.
        """
        if frame.empty:
            logger.warning("Skipping empty chunk for table %s", table)
            return
        try:
            frame.to_sql(table, engine, if_exists="append", index=False, method="multi")
        except Exception as exc:  # noqa: BLE001 - pandas wraps driver errors.
            raise DatabaseError(
                f"Failed to append {len(frame):,} rows to {table!r}: {exc}"
            ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for the ETL pipeline.

    Args:
        argv: Command-line arguments; defaults to :data:`sys.argv`.

    Returns:
        ``0`` on success, ``1`` on a handled configuration/ETL/database failure.
    """
    parser = argparse.ArgumentParser(description="Run the NYC taxi ETL pipeline.")
    parser.add_argument("--config", default=None, help="Path to the YAML config.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotlist config overrides, e.g. etl.chunk_size=250000",
    )
    args = parser.parse_args(argv)

    setup_logging()
    try:
        cfg = load_config(args.config, overrides=args.overrides)
        stats = TaxiETLPipeline(cfg).run()
    except (ConfigError, ETLError, DatabaseError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("ETL summary: %s", stats.as_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
