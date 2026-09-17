from __future__ import annotations

import logging
from typing import Iterable, Optional

from config import PROCESSED_DATA_PATH
from data_pipeline.db_utils import DatabaseManager

logger = logging.getLogger(__name__)

DEFAULT_TABLES = (
    "historical_market_data",
    "market_features",
    "fused_features",
    "regime_predictions",
    "sentiment_events",
    "option_trades",
    "backtest_results",
)


def export_tables(tables: Optional[Iterable[str]] = None, db: Optional[DatabaseManager] = None) -> list[str]:
    """Write DuckDB tables to data/processed as parquet (csv fallback)."""
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    db = db or DatabaseManager()
    written: list[str] = []
    for name in tables or DEFAULT_TABLES:
        try:
            if not db.table_exists(name):
                continue
            df = db.load_table(name)
            if df is None or df.height == 0:
                continue
            dest = PROCESSED_DATA_PATH / f"{name}.parquet"
            try:
                df.write_parquet(dest)
            except Exception:
                dest = PROCESSED_DATA_PATH / f"{name}.csv"
                df.write_csv(dest)
            written.append(str(dest))
            logger.info("Exported %s -> %s (%s rows)", name, dest.name, df.height)
        except Exception as exc:
            logger.warning("Skip export of %s: %s", name, exc)
    if not written:
        logger.info("No processed snapshots written (tables empty or missing)")
    return written
