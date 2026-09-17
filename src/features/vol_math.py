from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import statsmodels.api as sm

TRADING_DAYS = 252


def garman_klass_variance(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Daily GK variance. Negative values (rare) are clipped at 0."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    var = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.clip(var, 0.0, None)


def calibrate_ou_ols(series: np.ndarray, dt: float = 1.0 / TRADING_DAYS) -> Dict[str, float]:
    """OLS discretisation of dS = theta*(mu-S)*dt + sigma*dW."""
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size < 10:
        raise ValueError("Need at least 10 observations to calibrate OU")
    s = series[:-1]
    ds = series[1:] - s
    x = sm.add_constant(s)
    model = sm.OLS(ds, x).fit()
    a, b = model.params
    if abs(b) < 1e-12:
        raise ValueError("OU slope is zero; series is not mean-reverting")
    theta = float(-b / dt)
    mu = float(-a / b)
    sigma = float(np.std(model.resid, ddof=1) / np.sqrt(dt))
    theta_eff = max(theta, 1e-8)
    stat_std = float(sigma / np.sqrt(2.0 * theta_eff))
    half_life = float(np.log(2.0) / theta_eff)
    return {"theta": theta, "mu": mu, "sigma": sigma, "stat_std": stat_std, "half_life": half_life}


def expanding_ou_zscore(vix: np.ndarray, min_obs: int = 252, dt: float = 1.0 / TRADING_DAYS) -> Tuple[np.ndarray, Dict[str, float]]:
    """Causal z-score: parameters at t use only observations up to t."""
    n = len(vix)
    z = np.full(n, np.nan)
    last_params: Dict[str, float] = {"theta": np.nan, "mu": np.nan, "sigma": np.nan, "stat_std": np.nan, "half_life": np.nan}
    refit_every = 21
    for t in range(min_obs - 1, n):
        should_fit = (t == min_obs - 1) or ((t - (min_obs - 1)) % refit_every == 0)
        if should_fit or not np.isfinite(last_params.get("mu", np.nan)):
            try:
                last_params = calibrate_ou_ols(vix[: t + 1], dt=dt)
            except ValueError:
                continue
        scale = last_params["stat_std"] if last_params["stat_std"] > 1e-8 else last_params["sigma"]
        z[t] = (vix[t] - last_params["mu"]) / scale
    return z, last_params
