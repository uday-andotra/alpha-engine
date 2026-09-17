from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional, Union

import polars as pl

from bootstrap import SRC_DIR  # noqa: F401
from config import DUCKDB_PATH
from data_pipeline.db_utils import DatabaseManager

logger = logging.getLogger(__name__)


def _as_date(value: Union[str, date, datetime, None]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


class SignalAggregator:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(DUCKDB_PATH)
        self.db.init_schema()

    def record_sentiment_event(
        self,
        score: float,
        statement_date: Union[str, date, datetime],
        reasoning: str = "",
        source: str = "",
    ) -> None:
        dt = _as_date(statement_date)
        if dt is None:
            raise ValueError("statement_date is required to persist a sentiment event")
        event = pl.DataFrame(
            {
                "date": [dt],
                "sentiment_score": [float(score)],
                "reasoning": [reasoning or ""],
                "source": [source or ""],
            }
        ).with_columns(pl.col("date").cast(pl.Date))
        # Replace any existing row for the same date+source
        if self.db.table_exists("sentiment_events"):
            try:
                existing = self.db.load_table("sentiment_events")
                if existing.height:
                    existing = existing.with_columns(pl.col("date").cast(pl.Date))
                    filtered = existing.filter(
                        ~((pl.col("date") == dt) & (pl.col("source") == (source or "")))
                    )
                    combined = pl.concat([filtered.select(event.columns), event], how="vertical")
                    self.db.save_polars_df(combined, "sentiment_events", mode="overwrite")
                    return
            except Exception as exc:
                logger.warning("Could not merge sentiment events, appending: %s", exc)
        self.db.save_polars_df(event, "sentiment_events", mode="append")

    def fuse_signals(
        self,
        latest_sentiment_score: Optional[float] = None,
        statement_date: Union[str, date, datetime, None] = None,
        override_date: Union[str, date, datetime, None] = None,
        source: str = "",
        reasoning: str = "",
        persist_event: bool = True,
    ) -> pl.DataFrame:
        """
        Merge daily quant features with event-driven sentiment.

        `override_date` is accepted as an alias of `statement_date` so older scripts work.
        If a score + date are provided they are stored in `sentiment_events`.
        The panel is then built by forward-filling all known events.
        """
        statement_date = statement_date or override_date
        if latest_sentiment_score is not None and statement_date is not None and persist_event:
            self.record_sentiment_event(
                score=latest_sentiment_score,
                statement_date=statement_date,
                reasoning=reasoning,
                source=source,
            )
        elif latest_sentiment_score is not None and statement_date is None:
            logger.warning(
                "Score %.3f provided without statement_date; not written as an event. "
                "Existing sentiment_events will still be forward-filled.",
                latest_sentiment_score,
            )

        df = self.db.load_table("market_features").sort("date")
        df = df.with_columns(pl.col("date").cast(pl.Datetime).dt.date().alias("date"))

        events = pl.DataFrame({"date": [], "sentiment_score": []}, schema={"date": pl.Date, "sentiment_score": pl.Float64})
        if self.db.table_exists("sentiment_events"):
            try:
                raw_events = self.db.load_table("sentiment_events")
                if raw_events.height:
                    events = (
                        raw_events.with_columns(pl.col("date").cast(pl.Date))
                        .group_by("date")
                        .agg(pl.col("sentiment_score").mean())
                        .sort("date")
                    )
            except Exception as exc:
                logger.warning("Could not load sentiment_events: %s", exc)

        if events.height:
            df = df.join(events.rename({"sentiment_score": "raw_sentiment"}), on="date", how="left")
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("raw_sentiment"))

        df = df.with_columns(
            pl.col("raw_sentiment").forward_fill().fill_null(0.0).alias("macro_sentiment")
        ).drop("raw_sentiment")

        self.db.save_polars_df(df, "fused_features", mode="overwrite")
        logger.info("Wrote fused_features (%s rows, %s sentiment events)", df.height, events.height)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(SignalAggregator().fuse_signals().tail(5))
