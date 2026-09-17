from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import duckdb
import polars as pl

from bootstrap import SRC_DIR  # noqa: F401
from config import DUCKDB_PATH

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SCHEMA_SQL = [
    """
    CREATE TABLE IF NOT EXISTS historical_market_data (
        date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        indiavix DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sentiment_events (
        date DATE,
        sentiment_score DOUBLE,
        reasoning VARCHAR,
        source VARCHAR,
        created_at TIMESTAMP DEFAULT current_timestamp
    )
    """,
]


def quote_ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


class DatabaseManager:
    def __init__(self, db_path: Path = DUCKDB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        conn = duckdb.connect(str(self.db_path), read_only=read_only)
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connection() as conn:
            for stmt in SCHEMA_SQL:
                conn.execute(stmt)

    def _table_columns(self, conn, table_name: str) -> list[str]:
        rows = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
            """,
            [table_name],
        ).fetchall()
        return [r[0] for r in rows]

    def save_polars_df(self, df: pl.DataFrame, table_name: str, mode: str = "overwrite") -> None:
        table = quote_ident(table_name)
        with self.connection() as conn:
            conn.register("incoming_df", df)
            exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
            if mode == "overwrite":
                conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM incoming_df")
            elif mode == "append":
                if exists:
                    table_cols = self._table_columns(conn, table_name)
                    incoming = [c for c in df.columns if c in table_cols]
                    if not incoming:
                        raise ValueError(f"No overlapping columns to insert into {table_name}")
                    col_sql = ", ".join(quote_ident(c) for c in incoming)
                    conn.execute(f"INSERT INTO {table} ({col_sql}) SELECT {col_sql} FROM incoming_df")
                else:
                    conn.execute(f"CREATE TABLE {table} AS SELECT * FROM incoming_df")
            else:
                raise ValueError(f"Unknown mode: {mode}")
            conn.unregister("incoming_df")

    def load_table(self, table_name: str) -> pl.DataFrame:
        table = quote_ident(table_name)
        with self.connection(read_only=True) as conn:
            return conn.execute(f"SELECT * FROM {table}").pl()

    def table_exists(self, table_name: str) -> bool:
        try:
            with self.connection(read_only=True) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                    [table_name],
                ).fetchone()
                return bool(row and row[0])
        except Exception:
            return False

    def execute(self, sql: str, params: Optional[list] = None, read_only: bool = False) -> pl.DataFrame:
        with self.connection(read_only=read_only) as conn:
            cur = conn.execute(sql, params or [])
            try:
                return cur.pl()
            except Exception:
                return pl.DataFrame()
