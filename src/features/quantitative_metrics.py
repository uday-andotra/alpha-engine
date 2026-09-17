from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import polars as pl

from bootstrap import SRC_DIR  # noqa: F401
from config import DUCKDB_PATH, GK_WINDOW, OU_MIN_OBS, TRADING_DAYS
from data_pipeline.db_utils import DatabaseManager
from features.vol_math import calibrate_ou_ols, expanding_ou_zscore, garman_klass_variance

logger = logging.getLogger(__name__)


class VolatilityEngine:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager(DUCKDB_PATH)

    def calculate_garman_klass(self, window: int = GK_WINDOW) -> pl.DataFrame:
        df = self.db.load_table("historical_market_data").sort("date")
        gk = garman_klass_variance(
            df["open"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
        )
        df = df.with_columns(pl.Series("gk_estimator", gk))
        df = df.with_columns(
            (pl.col("gk_estimator").rolling_mean(window_size=window) * TRADING_DAYS).sqrt().alias("realized_vol_21d")
        )
        # INDIAVIX is quoted in percent; realized vol is a decimal.
        df = df.with_columns(
            (pl.col("indiavix") - pl.col("realized_vol_21d") * 100.0).alias("vrp")
        )
        self.db.save_polars_df(df, "market_features", mode="overwrite")
        logger.info("Wrote market_features (%s rows) with GK / VRP", df.height)
        return df

    def calibrate_ou_process(self, min_obs: int = OU_MIN_OBS, expanding: bool = True) -> Dict[str, float]:
        df = self.db.load_table("market_features").sort("date")
        vix = df["indiavix"].to_numpy()
        if expanding:
            z, params = expanding_ou_zscore(vix, min_obs=min_obs)
            df = df.with_columns(pl.Series("ou_z_score", z))
        else:
            params = calibrate_ou_ols(vix)
            scale = params["stat_std"] if params["stat_std"] > 1e-8 else params["sigma"]
            z = (vix - params["mu"]) / scale
            df = df.with_columns(pl.Series("ou_z_score", z))
        self.db.save_polars_df(df, "market_features", mode="overwrite")
        logger.info("OU params: %s", {k: round(v, 4) for k, v in params.items() if np.isfinite(v)})
        return params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = VolatilityEngine()
    engine.calculate_garman_klass()
    print(engine.calibrate_ou_process())
