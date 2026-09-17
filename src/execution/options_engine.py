from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import requests

from bootstrap import SRC_DIR  # noqa: F401
from config import HAWKISH_SENTIMENT_THRESHOLD, HIGH_VOL_THRESHOLD, OPTION_SYMBOL, OTM_PCT
from models.vol_surface import SABRVolSurface

logger = logging.getLogger(__name__)


def route_signal(prob_high_vol: float, sentiment_score: float) -> Tuple[str, str, str]:
    """Return (side, option_type, action_label). side is BUY or SELL."""
    if prob_high_vol >= HIGH_VOL_THRESHOLD:
        return "BUY", "Put", "BUY PUT (Crash Protection)"
    if sentiment_score > HAWKISH_SENTIMENT_THRESHOLD:
        return "SELL", "Call", "SELL CALL (Harvest Premium)"
    return "SELL", "Put", "SELL PUT (Harvest Premium)"


class LiveOptionChainFetcher:
    def __init__(self, symbol: str = OPTION_SYMBOL):
        self.symbol = symbol
        self.base_url = "https://www.nseindia.com"
        self.api_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={self.symbol}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/option-chain",
        }
        self.session = requests.Session()

    def fetch_nearest_chain(self) -> Tuple[pd.DataFrame, float, str]:
        logger.info("Connecting to NSE for %s", self.symbol)
        self.session.get(self.base_url, headers=self.headers, timeout=10)
        res = self.session.get(self.api_url, headers=self.headers, timeout=10)
        if res.status_code != 200:
            raise ConnectionError(f"NSE status {res.status_code}")
        data = res.json()
        spot = float(data["records"]["underlyingValue"])
        nearest_expiry = data["records"]["expiryDates"][0]
        records = []
        for item in data["records"]["data"]:
            if item.get("expiryDate") != nearest_expiry:
                continue
            strike = item.get("strikePrice")
            if "CE" in item:
                records.append(
                    {"Type": "Call", "Strike": strike, "LTP": item["CE"].get("lastPrice"), "IV": item["CE"].get("impliedVolatility")}
                )
            if "PE" in item:
                records.append(
                    {"Type": "Put", "Strike": strike, "LTP": item["PE"].get("lastPrice"), "IV": item["PE"].get("impliedVolatility")}
                )
        return pd.DataFrame(records), spot, nearest_expiry


def _years_to_expiry(expiry_label: str) -> float:
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
        try:
            exp = datetime.strptime(expiry_label, fmt)
            return max((exp - datetime.utcnow()).days, 1) / 365.0
        except ValueError:
            continue
    return 30.0 / 365.0


def calibrate_sabr_from_chain(chain: pd.DataFrame, spot: float, expiry_label: str) -> Optional[dict]:
    work = chain.dropna(subset=["Strike", "IV"]).copy()
    work = work[work["IV"] > 0]
    if work.empty:
        return None
    grouped = work.groupby("Strike", as_index=False)["IV"].mean()
    vols = grouped["IV"].to_numpy(dtype=float)
    if vols.mean() > 1.5:
        grouped["IV"] = vols / 100.0
    sabr = SABRVolSurface(forward_price=spot, time_to_maturity=_years_to_expiry(expiry_label))
    try:
        return sabr.calibrate(grouped["Strike"].tolist(), grouped["IV"].tolist())
    except Exception as exc:
        logger.warning("SABR calibration failed: %s", exc)
        return None


def nearest_strike(frame: pd.DataFrame, target: float) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("No option rows available")
    idx = (frame["Strike"] - target).abs().idxmin()
    return frame.loc[[idx]]


class DynamicExecutionRouter:
    @staticmethod
    def route_execution(chain_df, spot_price: float, prob_high_vol: float, sentiment_score: float, otm_pct: float = OTM_PCT):
        side, opt, action = route_signal(prob_high_vol, sentiment_score)
        target = spot_price * (1.0 - otm_pct) if opt == "Put" else spot_price * (1.0 + otm_pct)
        selected = nearest_strike(chain_df[chain_df["Type"] == opt], target)
        logger.info("Signal routed: %s | strike=%s | ltp=%s", action, selected["Strike"].iloc[0], selected["LTP"].iloc[0])
        return selected, action


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(route_signal(0.75, -0.2))
    try:
        chain, spot, expiry = LiveOptionChainFetcher().fetch_nearest_chain()
        print(f"Fetched {len(chain)} options expiry={expiry} spot={spot}")
        print(calibrate_sabr_from_chain(chain, spot, expiry))
    except Exception as exc:
        print(f"Live chain unavailable: {exc}")
