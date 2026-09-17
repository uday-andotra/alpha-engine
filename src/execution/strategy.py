from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["indiavix"] = pd.to_numeric(out["indiavix"], errors="coerce")
    out["vix_ma21"] = out["indiavix"].rolling(21, min_periods=10).mean()
    out["vix_chg_5d"] = out["indiavix"].pct_change(5)
    out["vix_pctile"] = out["indiavix"].rolling(252, min_periods=60).apply(lambda x: float((x[-1] >= x).mean()), raw=True)
    out["ou_z_score"] = pd.to_numeric(out.get("ou_z_score"), errors="coerce")
    out["vrp"] = pd.to_numeric(out.get("vrp"), errors="coerce")
    out["macro_sentiment"] = pd.to_numeric(out.get("macro_sentiment"), errors="coerce").fillna(0.0)
    out["prob_regime_high_vol"] = pd.to_numeric(out["prob_regime_high_vol"], errors="coerce")
    changed = out["macro_sentiment"].diff().abs().fillna(0) > 1e-9
    event_date = out["date"].where(changed)
    if "date" in out.columns:
        event_date = event_date.ffill()
        try:
            out["days_since_event"] = (pd.to_datetime(out["date"]) - pd.to_datetime(event_date)).dt.days
        except Exception:
            out["days_since_event"] = 999
    else:
        out["days_since_event"] = 999
    out["days_since_event"] = out["days_since_event"].fillna(999)
    try:
        from features.vix_forecast import add_vix_forecast
        out = add_vix_forecast(out)
    except Exception:
        out["vix_forecast_5d"] = 0.0
    return out


def _n(row: pd.Series, key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


@dataclass
class Decision:
    side: Optional[str]
    option_type: Optional[str]
    label: str
    conviction: float
    vol_regime: str
    sent_regime: str
    reasons: List[str] = field(default_factory=list)

    def as_route(self) -> Tuple[Optional[str], Optional[str], str]:
        return self.side, self.option_type, self.label


def decide(row: pd.Series) -> Decision:
    """
    One object every day: vol state + RBI state + event freshness → trade + size.
    """
    vix = _n(row, "indiavix")
    z = _n(row, "ou_z_score")
    vrp = _n(row, "vrp")
    p_high = _n(row, "prob_regime_high_vol")
    chg5 = _n(row, "vix_chg_5d")
    sent = _n(row, "macro_sentiment")
    days = _n(row, "days_since_event", 999)

    reasons: List[str] = []

    if sent > 0.15:
        sent_regime = "hawkish"
    elif sent < -0.15:
        sent_regime = "dovish"
    else:
        sent_regime = "neutral"

    breaking = (vix >= 24.0 and chg5 > 0) or (chg5 >= 0.18 and vix >= 20.0)
    fc = _n(row, "vix_forecast_5d", 0.0)
    pct = _n(row, "vix_pctile", 0.5)
    # Side comes from where VIX sits in its own 1-year history.
    # Top third = rich (sell). Bottom third = cheap (buy).
    if breaking:
        vol_regime, vol_score = "breaking", -1.0
        reasons.append("vol break")
    elif pct >= 0.65:
        vol_regime, vol_score = "rich", 0.6 + 0.4 * (pct - 0.65) / 0.35
        reasons.append(f"VIX high vs 1y ({pct:.0%})")
    elif pct <= 0.35:
        vol_regime, vol_score = "cheap", -0.6 - 0.4 * (0.35 - pct) / 0.35
        reasons.append(f"VIX low vs 1y ({pct:.0%})")
    else:
        vol_regime, vol_score = "dead", 0.0
        reasons.append(f"VIX mid vs 1y ({pct:.0%})")

    fresh = days <= 15
    if fresh and abs(sent) >= 0.1:
        reasons.append(f"fresh RBI ({int(days)}d, {sent:+.2f})")

    # Sentiment shifts the wing and adds conviction; it can also veto a dangerous short.
    conviction = min(1.0, abs(vol_score))
    if fresh:
        conviction = min(1.0, conviction + 0.25 * abs(sent))

    side = None
    opt = None
    label = "FLAT"

    if vol_regime == "dead" and not (fresh and abs(sent) >= 0.25):
        label = "FLAT (no edge)"
        return Decision(None, None, label, 0.0, vol_regime, sent_regime, reasons)

    if vol_regime in ("breaking", "cheap"):
        # Long vol only in the cheap/break buckets, not every negative tick.
        side = "BUY"
        if sent_regime == "dovish" and p_high < 0.4 and chg5 <= 0:
            opt = "Call"
            label = "BUY CALL (cheap vol, dovish/risk-on)"
        else:
            opt = "Put"
            label = "BUY PUT (cheap/breaking vol)"
        if sent_regime == "hawkish":
            conviction = min(1.0, conviction + 0.15)
            reasons.append("hawkish adds to put")
    else:
        # Short vol. Hawkish → call wing. Dovish + jumpy → do not sell put.
        if sent_regime == "dovish" and (p_high >= 0.4 or z >= 0.5):
            reasons.append("dovish veto on short put")
            return Decision(None, None, "FLAT (dovish + nervous tape)", 0.0, vol_regime, sent_regime, reasons)
        side = "SELL"
        if sent_regime == "hawkish":
            opt = "Call"
            label = "SELL CALL (rich vol, hawkish cap)"
        else:
            opt = "Put"
            label = "SELL PUT (rich vol)"

    fc = _n(row, "vix_forecast_5d", 0.0)
    if pd.notna(row.get("vix_forecast_5d")):
        reasons.append(f"vix_fc_5d={fc:+.3f}")
        if side == "SELL" and fc > 0.025:
            return Decision(None, None, "FLAT (forecast VIX up, no short vol)", 0.0, vol_regime, sent_regime, reasons)
        if side == "BUY" and vol_regime != "breaking" and fc < -0.025:
            return Decision(None, None, "FLAT (forecast VIX down, no long vol)", 0.0, vol_regime, sent_regime, reasons)
        if (side == "SELL" and fc <= 0) or (side == "BUY" and fc >= 0):
            conviction = min(1.0, conviction + 0.15)
            reasons.append("forecast agrees")

    if conviction < 0.20:
        return Decision(None, None, "FLAT (weak conviction)", 0.0, vol_regime, sent_regime, reasons)

    return Decision(side, opt, label, float(conviction), vol_regime, sent_regime, reasons)


def route_row(row: pd.Series) -> Tuple[Optional[str], Optional[str], str]:
    return decide(row).as_route()
