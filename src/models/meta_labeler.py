from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from bootstrap import SRC_DIR  # noqa: F401

logger = logging.getLogger(__name__)


def triple_barrier_labels(
    close: np.ndarray,
    side: np.ndarray,
    profit_take: float = 0.015,
    stop_loss: float = 0.015,
    horizon: int = 5,
) -> np.ndarray:
    """
    López de Prado-style triple barrier on a primary side signal.
    side > 0 long, side < 0 short, side == 0 skipped (label = nan).
    Label is 1 if the take-profit is touched first (or horizon move is in-signal),
    else 0.
    """
    close = np.asarray(close, dtype=float)
    side = np.asarray(side, dtype=float)
    n = len(close)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if side[i] == 0 or not np.isfinite(side[i]):
            continue
        direction = 1.0 if side[i] > 0 else -1.0
        entry = close[i]
        end = min(n, i + 1 + horizon)
        path = (close[i + 1 : end] / entry - 1.0) * direction
        if path.size == 0:
            continue
        hit_tp = np.where(path >= profit_take)[0]
        hit_sl = np.where(path <= -stop_loss)[0]
        first_tp = hit_tp[0] if hit_tp.size else None
        first_sl = hit_sl[0] if hit_sl.size else None
        if first_tp is None and first_sl is None:
            y[i] = 1.0 if path[-1] > 0 else 0.0
        elif first_sl is None or (first_tp is not None and first_tp < first_sl):
            y[i] = 1.0
        else:
            y[i] = 0.0
    return y


class TradeMetaLabeler:
    """Secondary filter: P(primary trade is profitable)."""

    def __init__(self, max_depth: int = 3, n_estimators: int = 80, min_samples: int = 50):
        import xgboost as xgb

        self.min_samples = min_samples
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=0.05,
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self.explainer = None
        self.feature_names = None
        self.is_fitted = False

    def train(self, x: pd.DataFrame, y: pd.Series) -> None:
        mask = y.notna()
        x_fit = x.loc[mask]
        y_fit = y.loc[mask].astype(int)
        if len(y_fit) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} labeled rows, got {len(y_fit)}")
        if y_fit.nunique() < 2:
            raise ValueError("Meta-label target has a single class")
        logger.info("Training meta-labeler on %s rows", len(y_fit))
        self.model.fit(x_fit, y_fit)
        self.feature_names = list(x_fit.columns)
        self.is_fitted = True
        try:
            import shap

            self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            logger.warning("SHAP explainer unavailable: %s", exc)
            self.explainer = None

    def predict_success_probability(self, x: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Meta-labeler is not fitted")
        cols = self.feature_names or list(x.columns)
        return self.model.predict_proba(x[cols])[:, 1]

    def get_feature_contributions(self, x: pd.DataFrame):
        if self.explainer is None:
            raise ValueError("Fit the model (with SHAP installed) before explaining")
        return self.explainer(x[self.feature_names])


def build_training_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    work = df.copy()
    if "nifty_close" not in work.columns and "close" in work.columns:
        work["nifty_close"] = work["close"]
    side = np.where(work["prob_regime_high_vol"] >= 0.5, -1.0, 1.0)
    work["primary_side"] = side
    labels = triple_barrier_labels(work["nifty_close"].to_numpy(), side)
    feature_cols = [c for c in ("vrp", "ou_z_score", "macro_sentiment", "prob_regime_high_vol") if c in work.columns]
    x = work[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.Series(labels, index=work.index, name="meta_label")
    return x, y, work["primary_side"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    close = 22000 * np.cumprod(1 + rng.normal(0.0003, 0.01, 400))
    side = np.where(rng.random(400) > 0.45, 1.0, -1.0)
    y = triple_barrier_labels(close, side)
    print("label rate", np.nanmean(y), "n", np.isfinite(y).sum())
