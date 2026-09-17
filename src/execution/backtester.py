from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import polars as pl

from bootstrap import SRC_DIR  # noqa: F401
from config import (
    DUCKDB_PATH,
    HAWKISH_SENTIMENT_THRESHOLD,
    HIGH_VOL_THRESHOLD,
    INITIAL_CAPITAL,
    ONE_WAY_COST,
    RISK_FREE_RATE,
    TRADING_DAYS,
    WIN_LOSS_RATIO,
)
from data_pipeline.db_utils import DatabaseManager
from execution.position_sizer import KellyPositionSizer

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Causal overlay on NIFTY close-to-close returns.

    This is an *index overlay*, not a full options P&L engine. Option routing
    lives in options_engine.py. Costs are charged on position changes.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, db: Optional[DatabaseManager] = None):
        self.initial_capital = initial_capital
        self.db = db or DatabaseManager(DUCKDB_PATH)
        self.sizer = KellyPositionSizer(half_kelly=True)

    def _load_panel(self) -> pl.DataFrame:
        return self.db.execute(
            """
            SELECT
                r.date,
                h.close AS nifty_close,
                r.prob_regime_high_vol,
                r.macro_sentiment,
                r.vrp,
                r.ou_z_score
            FROM regime_predictions r
            JOIN historical_market_data h ON CAST(r.date AS DATE) = CAST(h.date AS DATE)
            ORDER BY r.date ASC
            """
        )

    def run_backtest(self, use_kelly: bool = True) -> pl.DataFrame:
        df = self._load_panel().drop_nulls(subset=["nifty_close", "prob_regime_high_vol"])
        df = df.with_columns(pl.col("nifty_close").pct_change().alias("benchmark_return")).drop_nulls()

        # Target exposure decided on day t, executed on t+1 return
        target = []
        for row in df.iter_rows(named=True):
            p_high = float(row["prob_regime_high_vol"] or 0.0)
            sent = float(row["macro_sentiment"] or 0.0)
            if p_high >= HIGH_VOL_THRESHOLD:
                side = -1.0
                p_win = p_high
            elif sent > HAWKISH_SENTIMENT_THRESHOLD:
                side = 0.5
                p_win = 0.5 + min(sent, 0.5) / 2.0
            else:
                side = 1.0
                p_win = max(1.0 - p_high, 0.05)
            weight = self.sizer.calculate_capital_allocation(p_win, WIN_LOSS_RATIO) if use_kelly else 1.0
            target.append(side * weight)

        df = df.with_columns(pl.Series("target_exposure", target))
        df = df.with_columns(pl.col("target_exposure").shift(1).alias("executed_exposure")).drop_nulls()
        turnover = (pl.col("executed_exposure") - pl.col("executed_exposure").shift(1).fill_null(0.0)).abs()
        df = df.with_columns(
            [
                turnover.alias("turnover"),
                (pl.col("executed_exposure") * pl.col("benchmark_return") - turnover * ONE_WAY_COST).alias(
                    "executed_return"
                ),
            ]
        )
        df = df.with_columns(
            [
                (1.0 + pl.col("benchmark_return")).cum_prod().alias("benchmark_equity"),
                (1.0 + pl.col("executed_return")).cum_prod().alias("alpha_engine_equity"),
            ]
        )
        self.db.save_polars_df(df, "backtest_results", mode="overwrite")
        return df

    def performance_stats(self, df: pl.DataFrame) -> Dict[str, float]:
        rets = df["executed_return"].to_numpy()
        equity = df["alpha_engine_equity"].to_numpy()
        n = max(len(df), 1)
        cagr = float(equity[-1] ** (TRADING_DAYS / n) - 1.0) if equity[-1] > 0 else -1.0
        vol = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
        excess = float(np.mean(rets) * TRADING_DAYS - RISK_FREE_RATE)
        sharpe = excess / vol if vol > 0 else 0.0
        downside = rets[rets < 0]
        down_vol = float(np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
        sortino = excess / down_vol if down_vol > 0 else 0.0
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - peak) / np.where(peak == 0, 1.0, peak)))
        hit = float(np.mean(rets > 0)) if len(rets) else 0.0
        return {
            "cagr": cagr,
            "vol": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "hit_rate": hit,
            "final_equity": float(equity[-1]),
        }

    def print_tearsheet(self, df: pl.DataFrame) -> Dict[str, float]:
        stats = self.performance_stats(df)
        print("\n=== SYSTEMATIC ENGINE PERFORMANCE TEARSHEET ===")
        print("Note: NIFTY overlay with costs; not option-premium P&L.")
        print(f"CAGR:               {stats['cagr']:.2%}")
        print(f"Annualized Vol:     {stats['vol']:.2%}")
        print(f"Sharpe (ex-RF):     {stats['sharpe']:.2f}")
        print(f"Sortino:            {stats['sortino']:.2f}")
        print(f"Max Drawdown:       {stats['max_drawdown']:.2%}")
        print(f"Hit Rate:           {stats['hit_rate']:.1%}")
        print("===============================================")
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bt = StrategyBacktester()
    try:
        results = bt.run_backtest()
        bt.print_tearsheet(results)
    except Exception as exc:
        print(f"Run the market + regime pipeline before backtesting: {exc}")
