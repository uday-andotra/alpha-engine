from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from bootstrap import SRC_DIR  # noqa: F401
from config import DUCKDB_PATH, REGIME_K, REGIME_REFIT_EVERY
from data_pipeline.db_utils import DatabaseManager
from models.regime_utils import as_prob_array, identify_high_vol_state

logger = logging.getLogger(__name__)


def _filtered_probs(result) -> np.ndarray:
    probs = getattr(result, "filtered_marginal_probabilities", None)
    if probs is None:
        probs = result.smoothed_marginal_probabilities
    return as_prob_array(probs)


class RegimeClassifier:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(DUCKDB_PATH)

    def _fit_one(self, endog: np.ndarray, k_regimes: int = REGIME_K):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(endog.reshape(-1, 1)).flatten()
        model = MarkovRegression(
            endog=scaled,
            k_regimes=k_regimes,
            trend="c",
            switching_trend=True,
            switching_variance=True,
        )
        result = model.fit(disp=False, maxiter=250)
        return result, scaler

    def fit_markov_switching(
        self,
        k_regimes: int = REGIME_K,
        causal: bool = True,
        refit_every: int = REGIME_REFIT_EVERY,
        min_obs: int = 504,
    ) -> Tuple[object, pl.DataFrame]:
        """
        2-state Markov switching on INDIAVIX.

        High-vol state is identified by estimated variance, not column order.
        When causal=True, filtered probabilities are used and the model is
        refit on an expanding window every `refit_every` sessions.
        """
        df = self.db.load_table("fused_features").drop_nulls(subset=["indiavix"]).sort("date")
        vix = df["indiavix"].to_numpy()
        n = len(vix)
        high_vol = np.full(n, np.nan)
        low_vol = np.full(n, np.nan)
        last_res = None

        if not causal:
            last_res, _ = self._fit_one(vix, k_regimes=k_regimes)
            probs = as_prob_array(last_res.smoothed_marginal_probabilities)
            if probs.ndim == 1:
                probs = np.column_stack([1.0 - probs, probs])
            high_idx = identify_high_vol_state(last_res, endog=vix, probs=probs)
            low_idx = 1 - high_idx if probs.shape[1] == 2 else int(np.argmin([probs[:, i].mean() for i in range(probs.shape[1])]))
            high_vol = probs[:, high_idx]
            low_vol = probs[:, low_idx]
        else:
            start = min(max(min_obs, 100), n)
            cursor = start
            while cursor <= n:
                try:
                    last_res, _ = self._fit_one(vix[:cursor], k_regimes=k_regimes)
                except Exception as exc:
                    logger.warning("Markov fit failed at n=%s: %s", cursor, exc)
                    cursor = min(n, cursor + refit_every)
                    if cursor == n:
                        break
                    continue
                probs = _filtered_probs(last_res)
                if probs.ndim == 1:
                    probs = np.column_stack([1.0 - probs, probs])
                high_idx = identify_high_vol_state(last_res, endog=vix[:cursor], probs=probs)
                low_idx = 0 if high_idx == 1 else 1
                end = min(n, cursor + refit_every)
                # Use the last filtered prob as the live estimate for the next block
                live_high = float(probs[-1, high_idx])
                live_low = float(probs[-1, low_idx])
                # For dates already in the fit window that are still unlabeled, write filtered path
                path_high = probs[:, high_idx]
                path_low = probs[:, low_idx]
                fill_end = cursor
                sl = slice(0, fill_end)
                missing = np.isnan(high_vol[sl])
                high_vol[:fill_end][missing] = path_high[:fill_end][missing]
                low_vol[:fill_end][missing] = path_low[:fill_end][missing]
                if end > cursor:
                    high_vol[cursor:end] = live_high
                    low_vol[cursor:end] = live_low
                if end <= cursor:
                    break
                cursor = end

        df = df.with_columns(
            [
                pl.Series("prob_regime_high_vol", high_vol),
                pl.Series("prob_regime_low_vol", low_vol),
            ]
        )
        self.db.save_polars_df(df, "regime_predictions", mode="overwrite")
        logger.info("Wrote regime_predictions (%s rows, causal=%s)", df.height, causal)
        return last_res, df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res, pred = RegimeClassifier().fit_markov_switching()
    print(pred.select(["date", "indiavix", "prob_regime_high_vol", "macro_sentiment"]).tail(5))
