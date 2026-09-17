from __future__ import annotations

import pandas as pd

from execution.strategy import decide, route_row


def _row(**kwargs) -> pd.Series:
    base = {
        "prob_regime_high_vol": 0.20,
        "ou_z_score": 0.0,
        "vrp": 0.01,
        "macro_sentiment": 0.0,
        "indiavix": 15.0,
        "vix_ma21": 15.0,
        "vix_chg_5d": 0.0,
        "days_since_event": 80,
        "vix_pctile": 0.50,
        "vix_forecast_5d": 0.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_breakout_buys_put():
    d = decide(_row(
        indiavix=26, vix_ma21=16, vix_chg_5d=0.20,
        vix_pctile=0.95, vix_forecast_5d=0.05, prob_regime_high_vol=0.8, ou_z_score=2.0,
    ))
    assert d.side == "BUY" and d.option_type == "Put"
    assert d.vol_regime == "breaking"


def test_rich_hawkish_sells_call():
    d = decide(_row(
        vix_pctile=0.80, vix_forecast_5d=-0.01,
        macro_sentiment=0.40, days_since_event=3, indiavix=18,
    ))
    assert d.side == "SELL" and d.option_type == "Call"
    assert d.vol_regime == "rich"


def test_rich_neutral_sells_put():
    d = decide(_row(vix_pctile=0.80, vix_forecast_5d=-0.01, indiavix=18))
    assert d.side == "SELL" and d.option_type == "Put"


def test_cheap_buys_put():
    d = decide(_row(vix_pctile=0.20, vix_forecast_5d=0.01, indiavix=12))
    assert d.side == "BUY" and d.option_type == "Put"
    assert d.vol_regime == "cheap"


def test_mid_percentile_is_flat():
    d = decide(_row(vix_pctile=0.50))
    assert d.side is None
    assert d.vol_regime == "dead"


def test_forecast_blocks_short_when_vix_up():
    d = decide(_row(vix_pctile=0.80, vix_forecast_5d=0.04, indiavix=18))
    assert d.side is None
    assert "forecast" in d.label.lower()


def test_forecast_blocks_long_when_vix_down():
    d = decide(_row(vix_pctile=0.20, vix_forecast_5d=-0.04, indiavix=12))
    assert d.side is None
    assert "forecast" in d.label.lower()


def test_fresh_hawkish_raises_conviction():
    fresh = decide(_row(
        vix_pctile=0.80, vix_forecast_5d=-0.01, indiavix=18,
        macro_sentiment=0.40, days_since_event=3,
    ))
    stale = decide(_row(
        vix_pctile=0.80, vix_forecast_5d=-0.01, indiavix=18,
        macro_sentiment=0.0, days_since_event=90,
    ))
    assert fresh.conviction >= stale.conviction


def test_route_row_matches_decide():
    row = _row(vix_pctile=0.20, vix_forecast_5d=0.01, indiavix=12)
    assert route_row(row) == decide(row).as_route()
