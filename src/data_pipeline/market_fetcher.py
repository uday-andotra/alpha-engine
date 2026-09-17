from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from bootstrap import SRC_DIR  # noqa: F401
from config import DUCKDB_PATH, MARKET_START_DATE, NIFTY_TICKER, VIX_TICKER
from data_pipeline.db_utils import DatabaseManager

logger = logging.getLogger(__name__)

NIFTY_CANDIDATES = [NIFTY_TICKER, "^NSEI", "NSEI.NS", "^NSEBANK"]
VIX_CANDIDATES = [VIX_TICKER, "^INDIAVIX", "INDIAVIX.BO", "INDIAVIX.NS"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df


def _pick_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    mapping = {}
    lower = {c.lower(): c for c in df.columns}
    for target in ("open", "high", "low", "close"):
        if target in lower:
            mapping[lower[target]] = target
        elif target.capitalize() in df.columns:
            mapping[target.capitalize()] = target
        elif target.title() in df.columns:
            mapping[target.title()] = target
    if "close" not in mapping.values():
        raise KeyError(f"No close column in {list(df.columns)}")
    out = df.rename(columns=mapping)
    keep = [c for c in ("open", "high", "low", "close") if c in out.columns]
    return out[keep]


def _download_yahoo(ticker: str, start_date: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.download(
        ticker,
        start=start_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        hist = yf.Ticker(ticker).history(start=start_date, auto_adjust=False)
        frame = hist
    if frame is None or frame.empty:
        return pd.DataFrame()
    if getattr(frame.index, "tz", None) is not None:
        frame = frame.copy()
        frame.index = frame.index.tz_localize(None)
    return frame


def _first_working_series(tickers: List[str], start_date: str, label: str) -> Tuple[pd.DataFrame, str]:
    last_err = None
    for ticker in tickers:
        try:
            raw = _download_yahoo(ticker, start_date)
            if raw.empty:
                logger.warning("%s ticker %s returned no rows", label, ticker)
                continue
            parsed = _pick_ohlc(raw)
            if parsed.empty:
                continue
            logger.info("%s loaded from %s (%s rows)", label, ticker, len(parsed))
            return parsed, ticker
        except Exception as exc:
            last_err = exc
            logger.warning("%s ticker %s failed: %s", label, ticker, exc)
    raise RuntimeError(f"Could not download {label}. Last error: {last_err}")


def _proxy_vix_from_nifty(nifty: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised 21-day realized vol in VIX-like percent units."""
    close = nifty["close"].astype(float)
    logret = np.log(close / close.shift(1))
    realized = logret.rolling(window).std() * np.sqrt(252) * 100.0
    proxy = pd.DataFrame({"indiavix": realized}, index=nifty.index)
    logger.warning(
        "INDIAVIX is unavailable from Yahoo. Using NIFTY 21-day realized vol * 100 as a proxy. "
        "VRP will be near zero by construction and should not be treated as a real vol-risk premium."
    )
    return proxy.dropna()


class MarketDataPipeline:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(DUCKDB_PATH)
        self.db.init_schema()
        self.vix_source = "missing"

    def fetch_historical_base(self, start_date: str = MARKET_START_DATE) -> pl.DataFrame:
        logger.info("Fetching NIFTY / INDIAVIX from %s", start_date)
        nifty, nifty_ticker = _first_working_series(NIFTY_CANDIDATES, start_date, "NIFTY")
        nifty = nifty.rename(columns=lambda c: c.lower())
        for col in ("open", "high", "low", "close"):
            if col not in nifty.columns:
                nifty[col] = nifty.get("close")

        try:
            vix, vix_ticker = _first_working_series(VIX_CANDIDATES, start_date, "INDIAVIX")
            vix = vix.rename(columns={"close": "indiavix"})[["indiavix"]]
            self.vix_source = vix_ticker
        except RuntimeError as exc:
            logger.warning("%s", exc)
            vix = _proxy_vix_from_nifty(nifty)
            self.vix_source = "nifty_realized_proxy"

        df = nifty.join(vix, how="inner").dropna()
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        pl_df = pl.from_pandas(df).with_columns(pl.col("date").cast(pl.Date))
        if pl_df.height < 50:
            raise RuntimeError(f"Insufficient market history after join: {pl_df.height} rows")

        self.db.save_polars_df(pl_df, "historical_market_data", mode="overwrite")
        logger.info(
            "Wrote %s rows to historical_market_data (nifty=%s, vix=%s)",
            pl_df.height,
            nifty_ticker,
            self.vix_source,
        )
        return pl_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = MarketDataPipeline().fetch_historical_base()
    print(raw.tail(3))
