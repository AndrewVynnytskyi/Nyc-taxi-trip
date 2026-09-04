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
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence
from urllib.error import URLError
from urllib.request import urlopen

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
        source_path = self.resolve_source_path()
        engine = engine if engine is not None else build_engine(self.cfg)

        raw_table = str(self.cfg.database.raw_table)
        clean_table = str(self.cfg.database.clean_table)

        self.stats = ETLStats()
        self._recreate_tables(engine, (raw_table, clean_table))

        logger.info("Starting ETL run from %s", source_path)
        try:
            for index, chunk in enumerate(self.extract(source_path), start=1):
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
        """Return a local path to the source file, downloading it if necessary.

        Resolution order:
            1. ``etl.source.local_path`` -- an explicit override.
            2. A previously downloaded file in ``etl.source.download_dir``.
            3. ``etl.source.url`` -- the NYC TLC public parquet (no credentials).
            4. ``etl.source.kaggle_dataset`` -- the Kaggle mirror, via kagglehub.

        Returns:
            Absolute path to the source file.

        Raises:
            ETLError: If no source can be resolved.
        """
        source = self.cfg.etl.source

        override = source.get("local_path")
        if override:
            path = Path(str(override)).expanduser().resolve()
            if not path.is_file():
                raise ETLError(f"Configured etl.source.local_path does not exist: {path}")
            return path

        url = source.get("url")
        if url:
            destination = (
                Path(str(source.download_dir)).expanduser().resolve() / Path(str(url)).name
            )
            if destination.is_file():
                logger.info("Using cached download %s", destination)
                return destination
            return self._download(str(url), destination)

        return self._download_from_kaggle()

    def _download(self, url: str, destination: Path) -> Path:
        """Stream a remote file to disk, writing atomically via a temp file.

        Args:
            url: HTTP(S) URL to fetch.
            destination: Final path for the downloaded file.

        Returns:
            The path the file was written to.

        Raises:
            ETLError: If the download fails or writes an empty file.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        logger.info("Downloading %s -> %s", url, destination)

        try:
            with urlopen(url) as response, temporary.open("wb") as handle:  # noqa: S310
                shutil.copyfileobj(response, handle, length=1 << 20)
        except (URLError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            raise ETLError(f"Failed to download {url}: {exc}") from exc

        if temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise ETLError(f"Download from {url} produced an empty file.")

        temporary.replace(destination)
        logger.info("Downloaded %.1f MB", destination.stat().st_size / 1e6)
        return destination

    def _download_from_kaggle(self) -> Path:
        """Fetch the dataset from the Kaggle mirror via ``kagglehub``.

        Returns:
            Absolute path to the expected file inside the kagglehub cache.

        Raises:
            ETLError: If ``kagglehub`` is missing, the download fails, or the
                expected file is not present in the downloaded dataset.
        """
        source = self.cfg.etl.source
        try:
            import kagglehub  # noqa: PLC0415 - optional, only needed on cache miss.
        except ImportError as exc:
            raise ETLError(
                "kagglehub is required for the Kaggle fallback. Install it with "
                "`pip install -e '.[data]'`, or set NYC_TAXI_SOURCE_PATH to a local file."
            ) from exc

        logger.info("Downloading dataset %s via kagglehub", source.kaggle_dataset)
        try:
            root = Path(kagglehub.dataset_download(str(source.kaggle_dataset)))
        except Exception as exc:  # noqa: BLE001 - network/auth failures vary.
            raise ETLError(f"kagglehub download failed: {exc}") from exc

        path = root / str(source.kaggle_filename)
        if not path.is_file():
            raise ETLError(f"Expected file {source.kaggle_filename!r} was not found under {root}.")
        logger.info("Dataset available at %s", path)
        return path

    def _resolve_format(self, path: Path) -> str:
        """Determine whether the source should be read as CSV or parquet.

        Args:
            path: Source file path.

        Returns:
            Either ``"csv"`` or ``"parquet"``.

        Raises:
            ETLError: If the configured format is unsupported, or ``auto`` is
                used with an unrecognised file suffix.
        """
        declared = str(self.cfg.etl.source.format).strip().lower()
        if declared in {"csv", "parquet"}:
            return declared
        if declared != "auto":
            raise ETLError(
                f"Unsupported etl.source.format {declared!r}; expected auto, csv or parquet."
            )

        suffix = path.suffix.lower()
        if suffix in {".csv", ".gz", ".zip", ".bz2"}:
            return "csv"
        if suffix in {".parquet", ".pq"}:
            return "parquet"
        raise ETLError(
            f"Cannot infer a format for {path.name!r}; set etl.source.format explicitly."
        )

    def extract(self, source_path: Path | str) -> Iterator[pd.DataFrame]:
        """Stream the source file in chunks, validating the schema of each one.

        Honours ``etl.max_rows``: the final chunk is truncated so that no more
        than that many rows are ever yielded.

        Args:
            source_path: Path to the source CSV or parquet file.

        Yields:
            Schema-validated chunks of at most ``etl.chunk_size`` rows.

        Raises:
            ETLError: If the file is missing or unreadable.
            ValueError: If a required column is absent from a chunk.
            TypeError: If a required column has an unexpected dtype.
        """
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise ETLError(f"Source file not found: {path}")

        file_format = self._resolve_format(path)
        chunk_size = int(self.cfg.etl.chunk_size)
        max_rows_cfg = self.cfg.etl.get("max_rows")
        max_rows = int(max_rows_cfg) if max_rows_cfg is not None else None

        logger.info(
            "Reading %s as %s in chunks of %s rows%s",
            path.name,
            file_format,
            f"{chunk_size:,}",
            f" (capped at {max_rows:,} rows)" if max_rows else "",
        )

        reader = (
            self._iter_csv_chunks(path, chunk_size)
            if file_format == "csv"
            else self._iter_parquet_chunks(path, chunk_size)
        )

        emitted = 0
        for index, chunk in enumerate(reader, start=1):
            if max_rows is not None:
                remaining = max_rows - emitted
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]

            chunk = self._normalise_dtypes(chunk)
            self._validate_schema(chunk, index)
            emitted += len(chunk)
            logger.debug("Chunk #%d extracted (%s rows)", index, f"{len(chunk):,}")
            yield chunk

    def _normalise_dtypes(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Coerce a chunk to the dtypes the schema contract expects.

        Sources disagree on timestamp precision -- the NYC TLC parquet files use
        microseconds while ``pandas.read_csv`` produces nanoseconds -- so the
        date columns are normalised here rather than loosening the contract.

        Args:
            chunk: Raw chunk as read from the source.

        Returns:
            The chunk with its date columns cast to ``datetime64[ns]``.

        Raises:
            ETLError: If a date column cannot be converted.
        """
        date_columns = [
            str(column) for column in self.cfg.etl.date_columns if str(column) in chunk.columns
        ]
        if not date_columns:
            return chunk

        try:
            return chunk.astype({column: "datetime64[ns]" for column in date_columns})
        except (TypeError, ValueError) as exc:
            raise ETLError(f"Failed to normalise date columns {date_columns}: {exc}") from exc

    def _iter_csv_chunks(self, path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Yield chunks of a CSV file.

        Args:
            path: CSV file to read.
            chunk_size: Maximum rows per chunk.

        Yields:
            Raw chunks, with the configured date columns already parsed.

        Raises:
            ETLError: If the file cannot be opened or parsed.
        """
        date_columns = [str(column) for column in self.cfg.etl.date_columns]
        try:
            reader = pd.read_csv(path, parse_dates=date_columns, chunksize=chunk_size)
            yield from reader
        except (OSError, ValueError) as exc:
            raise ETLError(f"Failed to read CSV {path}: {exc}") from exc

    def _iter_parquet_chunks(self, path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
        """Yield chunks of a parquet file, reading only the required columns.

        Every column is read, not just the ones the model consumes: the raw
        landing table is meant to be a faithful copy of the source, and columns
        outside the feature set are exactly what future experiments need.

        Args:
            path: Parquet file to read.
            chunk_size: Maximum rows per record batch.

        Yields:
            Raw chunks as pandas DataFrames.

        Raises:
            ETLError: If the file cannot be opened, or a required column is
                absent from the parquet schema.
        """
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415 - heavy optional import.

            parquet_file = pq.ParquetFile(path)
        except Exception as exc:  # noqa: BLE001 - pyarrow raises a variety of errors.
            raise ETLError(f"Failed to open parquet file {path}: {exc}") from exc

        available = set(parquet_file.schema_arrow.names)
        wanted = {str(name) for name in self.cfg.etl.required_columns}
        missing = sorted(wanted - available)
        if missing:
            raise ETLError(f"Parquet file {path.name} is missing required column(s): {missing}")

        try:
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                yield batch.to_pandas()
        except Exception as exc:  # noqa: BLE001 - pyarrow raises a variety of errors.
            raise ETLError(f"Failed to read parquet {path}: {exc}") from exc

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
