from __future__ import annotations

import math

from scipy.stats import norm


def _d1_d2(spot: float, strike: float, tenor: float, rate: float, vol: float):
    tenor = max(float(tenor), 1.0 / 365.0)
    vol = max(float(vol), 1e-4)
    spot = max(float(spot), 1e-8)
    strike = max(float(strike), 1e-8)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * tenor) / (vol * math.sqrt(tenor))
    d2 = d1 - vol * math.sqrt(tenor)
    return d1, d2, tenor


def black_scholes_price(
    spot: float,
    strike: float,
    tenor: float,
    rate: float,
    vol: float,
    option_type: str,
) -> float:
    """European option price in index points."""
    option_type = option_type.lower()
    d1, d2, tenor = _d1_d2(spot, strike, tenor, rate, vol)
    df = math.exp(-rate * tenor)
    if option_type in ("call", "ce"):
        return float(spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
    if option_type in ("put", "pe"):
        return float(strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
    raise ValueError(f"Unknown option_type {option_type}")


def round_strike(spot: float, otm_pct: float, option_type: str, step: int = 50) -> float:
    raw = spot * (1.0 - otm_pct) if option_type.lower() in ("put", "pe") else spot * (1.0 + otm_pct)
    return float(max(step, int(round(raw / step) * step)))
