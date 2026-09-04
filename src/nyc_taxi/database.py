"""PostgreSQL connectivity.

Credentials never appear in source or config literals: they are interpolated
from the environment by ``configs/train.yaml`` and assembled here into a
SQLAlchemy URL whose password is masked in every log line and ``repr``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd
from omegaconf import DictConfig
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.exc import SQLAlchemyError

from nyc_taxi.utils import get_logger

__all__ = ["DatabaseError", "build_engine", "build_url", "read_table", "session_scope"]

logger = get_logger(__name__)


class DatabaseError(RuntimeError):
    """Raised when a database connection or statement fails."""


def build_url(cfg: DictConfig) -> URL:
    """Assemble a SQLAlchemy URL from the ``database`` config section.

    Args:
        cfg: Full project config (the ``database`` section is read).

    Returns:
        A :class:`sqlalchemy.engine.URL`; rendering it hides the password.

    Raises:
        DatabaseError: If the password is still the mandatory-missing sentinel,
            i.e. ``POSTGRES_PASSWORD`` was not exported.
    """
    db = cfg.database
    raw_password = db.get("password")
    password = "" if raw_password is None else str(raw_password)
    if not password:
        raise DatabaseError(
            "Database password is not set. Export POSTGRES_PASSWORD (see .env.example) "
            "before running the ETL."
        )

    return URL.create(
        drivername=str(db.driver),
        username=str(db.user),
        password=password,
        host=str(db.host),
        port=int(db.port),
        database=str(db.database),
    )


def build_engine(cfg: DictConfig) -> Engine:
    """Create a pooled SQLAlchemy engine and verify connectivity eagerly.

    Args:
        cfg: Full project config.

    Returns:
        A ready-to-use :class:`sqlalchemy.Engine`.

    Raises:
        DatabaseError: If the engine cannot be created or the smoke-test
            ``SELECT 1`` fails.
    """
    db = cfg.database
    url = build_url(cfg)

    try:
        engine = create_engine(
            url,
            pool_size=int(db.pool_size),
            max_overflow=int(db.max_overflow),
            pool_pre_ping=bool(db.pool_pre_ping),
            echo=bool(db.echo),
            future=True,
        )
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not connect to {url.render_as_string()}: {exc}") from exc

    logger.info("Connected to %s", url.render_as_string())
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Connection]:
    """Provide a transactional connection that commits or rolls back atomically.

    Args:
        engine: Engine to open the connection on.

    Yields:
        An open :class:`sqlalchemy.engine.Connection` inside a transaction.

    Raises:
        DatabaseError: If the transaction fails; the transaction is rolled back
            by the underlying context manager before the error is re-raised.
    """
    try:
        with engine.begin() as connection:
            yield connection
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Transaction failed and was rolled back: {exc}") from exc


def read_table(
    engine: Engine,
    table: str,
    columns: list[str] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Read a table into a DataFrame with an optional column and row projection.

    Args:
        engine: Engine to query.
        table: Table name (validated as a bare SQL identifier).
        columns: Columns to select; ``None`` selects all.
        limit: Optional ``LIMIT`` for fast experimentation.

    Returns:
        The query result as a :class:`pandas.DataFrame`.

    Raises:
        ValueError: If ``table``, a column name or ``limit`` is invalid.
        DatabaseError: If the query fails.
    """
    _assert_identifier(table)
    projection = "*"
    if columns:
        for column in columns:
            _assert_identifier(column)
        projection = ", ".join(f'"{column}"' for column in columns)

    query = f'SELECT {projection} FROM "{table}"'
    if limit is not None:
        if int(limit) < 1:
            raise ValueError(f"limit must be >= 1, got {limit}.")
        query += f" LIMIT {int(limit)}"

    logger.info("Executing: %s", query)
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(text(query), connection)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Failed to read table {table!r}: {exc}") from exc

    logger.info("Fetched %s rows x %s columns from %s", f"{len(frame):,}", frame.shape[1], table)
    return frame


def _assert_identifier(name: str) -> None:
    """Reject anything that is not a plain SQL identifier.

    Table and column names cannot be bound as query parameters, so they are
    interpolated into the statement; this check keeps that interpolation safe.

    Args:
        name: Candidate identifier.

    Raises:
        ValueError: If ``name`` is empty or contains characters outside
            ``[A-Za-z0-9_]`` or starts with a digit.
    """
    if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(f"Unsafe SQL identifier: {name!r}.")
