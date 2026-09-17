from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.optimize import minimize


class SABRVolSurface:
    def __init__(self, forward_price: float, time_to_maturity: float, beta: float = 1.0):
        if forward_price <= 0:
            raise ValueError("forward_price must be positive")
        if time_to_maturity <= 0:
            raise ValueError("time_to_maturity must be positive")
        self.f = float(forward_price)
        self.T = float(time_to_maturity)
        self.beta = float(beta)

    def _atm_vol(self, alpha: float, rho: float, nu: float) -> float:
        f, t, beta = self.f, self.T, self.beta
        fwd_pow = f ** (1.0 - beta)
        term = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / (f ** (2.0 - 2.0 * beta)))
            + (rho * beta * nu * alpha) / (4.0 * fwd_pow)
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
        ) * t
        return (alpha / fwd_pow) * term

    def _hagan_vol(self, k: float, alpha: float, rho: float, nu: float) -> float:
        f, t, beta = self.f, self.T, self.beta
        k = max(float(k), 1e-8)
        if abs(np.log(f / k)) < 1e-8:
            return self._atm_vol(alpha, rho, nu)

        log_fk = np.log(f / k)
        fk_beta = (f * k) ** ((1.0 - beta) / 2.0)
        z = (nu / max(alpha, 1e-12)) * fk_beta * log_fk
        inner = max(1.0 - 2.0 * rho * z + z ** 2, 1e-16)
        denom_x = max(1.0 - rho, 1e-12)
        x = np.log((np.sqrt(inner) + z - rho) / denom_x)
        denom = fk_beta * (
            1.0
            + ((1.0 - beta) ** 2 / 24.0) * log_fk ** 2
            + ((1.0 - beta) ** 4 / 1920.0) * log_fk ** 4
        )
        term1 = alpha / max(denom, 1e-16)
        z_over_x = 1.0 if abs(z) < 1e-8 or abs(x) < 1e-12 else z / x
        term2 = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / (f * k) ** (1.0 - beta))
            + (rho * beta * nu * alpha) / (4.0 * fk_beta)
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
        ) * t
        return float(term1 * z_over_x * term2)

    def implied_vols(self, strikes: Iterable[float], alpha: float, rho: float, nu: float) -> np.ndarray:
        return np.array([self._hagan_vol(k, alpha, rho, nu) for k in strikes], dtype=float)

    def _objective(self, params, strikes, market_vols):
        alpha, rho, nu = params
        model = self.implied_vols(strikes, alpha, rho, nu)
        return float(np.mean((model - np.asarray(market_vols, dtype=float)) ** 2))

    def calibrate(self, strikes: List[float], market_vols: List[float]) -> Dict[str, float]:
        strikes = list(strikes)
        market_vols = list(market_vols)
        if len(strikes) != len(market_vols) or len(strikes) < 3:
            raise ValueError("Need at least 3 strike/vol pairs")
        # Seed alpha from closest-to-ATM market vol
        atm_idx = int(np.argmin(np.abs(np.asarray(strikes) - self.f)))
        alpha0 = max(float(market_vols[atm_idx]) * (self.f ** (1.0 - self.beta)), 1e-4)
        guess = [alpha0, -0.3, 0.5]
        bounds = ((1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0))
        result = minimize(self._objective, guess, args=(strikes, market_vols), bounds=bounds, method="L-BFGS-B")
        alpha, rho, nu = result.x
        return {"alpha": float(alpha), "beta": self.beta, "rho": float(rho), "nu": float(nu), "mse": float(result.fun), "success": bool(result.success)}


if __name__ == "__main__":
    sabr = SABRVolSurface(forward_price=22000, time_to_maturity=30 / 365)
    params = sabr.calibrate(
        [21000, 21500, 22000, 22500, 23000],
        [0.18, 0.15, 0.12, 0.13, 0.16],
    )
    print(params)
    print("ATM", sabr._hagan_vol(22000, params["alpha"], params["rho"], params["nu"]))



def sabr_iv_from_vix(spot: float, strike: float, tenor: float, vix_pct: float, rho: float = -0.35, nu: float = 0.75) -> float:
    """ATM = INDIAVIX, smile from a default equity-index SABR (beta=1)."""
    alpha = max(float(vix_pct) / 100.0, 0.05)
    sabr = SABRVolSurface(forward_price=max(spot, 1.0), time_to_maturity=max(tenor, 1.0 / 365.0), beta=1.0)
    iv = sabr._hagan_vol(strike, alpha, rho, nu)
    return float(max(iv, 0.05))
