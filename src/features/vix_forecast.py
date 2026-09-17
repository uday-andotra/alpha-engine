from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["ou_z_score", "vrp", "vix_chg_5d", "prob_regime_high_vol"]


def add_vix_forecast(df: pd.DataFrame, horizon: int = 5, min_train: int = 252, refit_every: int = 21) -> pd.DataFrame:
    """Causal 5-day VIX return forecast. Fitted only on rows whose label is already known."""
    out = df.copy()
    vix = pd.to_numeric(out["indiavix"], errors="coerce")
    y = (vix.shift(-horizon) / vix - 1.0).to_numpy()
    X = out.reindex(columns=FEATURE_COLS).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pred = np.full(len(out), np.nan)
    model = None
    scaler = None
    for i in range(min_train, len(out)):
        if model is None or (i - min_train) % refit_every == 0:
            end = i - horizon
            if end <= min_train // 2:
                continue
            Xi, yi = X[:end], y[:end]
            mask = np.isfinite(yi) & np.isfinite(Xi).all(axis=1)
            if mask.sum() < min_train // 2:
                continue
            scaler = StandardScaler()
            Xs = scaler.fit_transform(Xi[mask])
            model = Ridge(alpha=1.0)
            model.fit(Xs, yi[mask])
        if model is None:
            continue
        pred[i] = float(model.predict(scaler.transform(X[i : i + 1]))[0])
    out["vix_forecast_5d"] = pred
    return out
