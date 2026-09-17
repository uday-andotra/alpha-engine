from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from execution.position_sizer import KellyPositionSizer
from models.meta_labeler import triple_barrier_labels
from models.regime_utils import identify_high_vol_state
from models.vol_surface import SABRVolSurface


def test_half_kelly_bounds():
    sizer = KellyPositionSizer(half_kelly=True, max_leverage=1.0)
    alloc = sizer.calculate_capital_allocation(win_prob=0.8, win_loss_ratio=1.0)
    assert pytest.approx(alloc, 0.01) == 0.30
    assert sizer.calculate_capital_allocation(win_prob=0.3, win_loss_ratio=1.0) == 0.0


def test_full_kelly_and_clip():
    sizer = KellyPositionSizer(half_kelly=False, max_leverage=0.25)
    # p=0.8, b=1 -> f*=0.6 clipped to 0.25
    assert pytest.approx(sizer.calculate_capital_allocation(0.8, 1.0), 0.01) == 0.25


def test_identify_high_vol_state_by_variance():
    class Dummy:
        def __init__(self):
            self.params = pd.Series({"const[0]": 0.1, "const[1]": 0.4, "sigma2[0]": 0.05, "sigma2[1]": 0.9})

    assert identify_high_vol_state(Dummy()) == 1


def test_sabr_atm_limit_finite():
    sabr = SABRVolSurface(forward_price=22000, time_to_maturity=30 / 365, beta=1.0)
    atm = sabr._hagan_vol(22000, 0.12, -0.3, 0.5)
    near = sabr._hagan_vol(22000.01, 0.12, -0.3, 0.5)
    assert np.isfinite(atm) and atm > 0
    assert abs(atm - near) < 1e-3


def test_sabr_calibrate_mock_smile():
    sabr = SABRVolSurface(forward_price=22000, time_to_maturity=30 / 365)
    params = sabr.calibrate(
        [21000, 21500, 22000, 22500, 23000],
        [0.18, 0.15, 0.12, 0.13, 0.16],
    )
    assert params["mse"] < 0.01
    assert -1 < params["rho"] < 1
    assert params["alpha"] > 0


def test_triple_barrier_long_hits_tp():
    close = np.array([100.0, 100.5, 102.0, 101.0])
    side = np.array([1.0, 0.0, 0.0, 0.0])
    y = triple_barrier_labels(close, side, profit_take=0.015, stop_loss=0.015, horizon=3)
    assert y[0] == 1.0


def test_identify_high_vol_from_param_names_array():
    class Dummy:
        params = np.array([0.1, 0.4, 0.05, 0.9])
        param_names = ["const[0]", "const[1]", "sigma2[0]", "sigma2[1]"]

    assert identify_high_vol_state(Dummy()) == 1


def test_identify_high_vol_from_weighted_series():
    class Dummy:
        params = np.array([])

    endog = np.array([1.0, 1.1, 1.0, 5.0, 6.0, 5.5])
    probs = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
    ])
    assert identify_high_vol_state(Dummy(), endog=endog, probs=probs) == 1


def test_black_scholes_put_call_parity():
    from execution.black_scholes import black_scholes_price
    s, k, tau, r, vol = 22000.0, 22000.0, 30 / 365, 0.065, 0.15
    call = black_scholes_price(s, k, tau, r, vol, "call")
    put = black_scholes_price(s, k, tau, r, vol, "put")
    import math
    parity = call - put - (s - k * math.exp(-r * tau))
    assert abs(parity) < 1e-4
    assert call > 0 and put > 0


def test_connected_decision():
    """Kept for older checkouts; full matrix lives in test_strategy.py."""
    from execution.strategy import decide
    import pandas as pd

    breakout = pd.Series({
        "prob_regime_high_vol": 0.80, "ou_z_score": 2.0, "vrp": 0.01,
        "macro_sentiment": 0.0, "indiavix": 26, "vix_ma21": 16, "vix_chg_5d": 0.20,
        "days_since_event": 80, "vix_pctile": 0.95, "vix_forecast_5d": 0.05,
    })
    rich_hawk = pd.Series({
        "prob_regime_high_vol": 0.20, "ou_z_score": 0.3, "vrp": 0.03,
        "macro_sentiment": 0.4, "indiavix": 18, "vix_ma21": 16.0, "vix_chg_5d": 0.0,
        "days_since_event": 3, "vix_pctile": 0.80, "vix_forecast_5d": -0.01,
    })
    d = decide(breakout)
    assert d.side == "BUY" and d.option_type == "Put"
    h = decide(rich_hawk)
    assert h.side == "SELL" and h.option_type == "Call"
