from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from bootstrap import SRC_DIR  # noqa: F401
from config import (
    DUCKDB_PATH,
    INITIAL_CAPITAL,
    MAX_OPTION_LOTS,
    MIN_EQUITY_FRAC,
    NIFTY_LOT_SIZE,
    OPTION_COST_POINTS,
    OPTION_HOLD_DAYS,
    OPTION_TENOR_DAYS,
    OTM_PCT,
    RISK_FREE_RATE,
    RISK_PER_TRADE,
    STRIKE_STEP,
)
from data_pipeline.db_utils import DatabaseManager
from execution.black_scholes import black_scholes_price, round_strike
from models.vol_surface import sabr_iv_from_vix
from models.meta_labeler import TradeMetaLabeler
from execution.strategy import decide, prepare_features, route_row

logger = logging.getLogger(__name__)


class OptionsPnLBacktester:
    """Quiet-harvest shorts + early put hedges, 1.5% equity risk cap."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, db: Optional[DatabaseManager] = None):
        self.initial_capital = initial_capital
        self.db = db or DatabaseManager(DUCKDB_PATH)

    def _panel(self) -> pd.DataFrame:
        df = self.db.execute(
            """
            SELECT
                r.date,
                h.close AS spot,
                r.indiavix,
                r.prob_regime_high_vol,
                r.macro_sentiment,
                r.vrp,
                r.ou_z_score
            FROM regime_predictions r
            JOIN historical_market_data h ON CAST(r.date AS DATE) = CAST(h.date AS DATE)
            ORDER BY r.date ASC
            """
        ).to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["spot", "indiavix", "prob_regime_high_vol"]).reset_index(drop=True)
        return prepare_features(df)

    def _price(self, spot: float, strike: float, days_left: float, iv_pct: float, opt: str) -> float:
        tenor = max(days_left, 1.0) / 365.0
        try:
            vol = sabr_iv_from_vix(spot, strike, tenor, iv_pct)
        except Exception:
            vol = max(float(iv_pct), 1.0) / 100.0
        return black_scholes_price(
            spot=spot,
            strike=strike,
            tenor=tenor,
            rate=RISK_FREE_RATE,
            vol=vol,
            option_type=opt,
        )

    def _lots(self, equity: float, premium: float, side: str, conviction: float = 1.0) -> int:
        risk_budget = max(equity, 0.0) * RISK_PER_TRADE * max(min(conviction, 1.0), 0.25)
        # Long debit risk ≈ premium. Short risk approximated as 3x credit.
        unit_risk = max(premium, 1.0) * NIFTY_LOT_SIZE * (1.0 if side == "BUY" else 3.0)
        lots = int(risk_budget // unit_risk)
        return int(min(max(lots, 0), MAX_OPTION_LOTS))

    def run(self, use_meta_kelly: bool = False) -> pd.DataFrame:
        df = self._panel()
        trades: List[dict] = []
        open_pos = None
        equity = float(self.initial_capital)
        halt = self.initial_capital * MIN_EQUITY_FRAC
        hedge_cooldown = 0
        meta = None
        closed_X = []
        closed_y = []
        META_COLS = ["vrp", "ou_z_score", "prob_regime_high_vol", "macro_sentiment", "conviction", "vix_forecast_5d"]

        def close_at(i: int, reason: str) -> None:
            nonlocal open_pos, equity, hedge_cooldown, meta
            row = df.iloc[i]
            days_left = max(OPTION_TENOR_DAYS - open_pos["age"], 1)
            exit_px = max(self._price(row["spot"], open_pos["strike"], days_left, row["indiavix"], open_pos["opt"]), 0.0)
            signed = 1.0 if open_pos["side"] == "BUY" else -1.0
            pnl_pts = signed * (exit_px - open_pos["entry_px"]) - 2.0 * OPTION_COST_POINTS
            pnl_inr = pnl_pts * open_pos["lots"] * NIFTY_LOT_SIZE
            equity = max(equity + pnl_inr, 0.0)
            trades.append(
                {
                    "date_in": open_pos["date_in"],
                    "date_out": row["date"],
                    "action": open_pos["action"],
                    "side": open_pos["side"],
                    "option_type": open_pos["opt"],
                    "strike": open_pos["strike"],
                    "lots": open_pos["lots"],
                    "spot_in": open_pos["spot_in"],
                    "spot_out": float(row["spot"]),
                    "iv_in": open_pos["iv_in"],
                    "iv_out": float(row["indiavix"]),
                    "entry_px": open_pos["entry_px"],
                    "exit_px": exit_px,
                    "hold_days": open_pos["age"],
                    "pnl_points": pnl_pts,
                    "pnl_inr": pnl_inr,
                    "equity_after": equity,
                    "win": 1 if pnl_inr > 0 else 0,
                    "vrp": open_pos["vrp"],
                    "ou_z_score": open_pos["ou_z"],
                    "macro_sentiment": open_pos["sent"],
                    "prob_regime_high_vol": open_pos["p_high"],
                    "exit_reason": reason,
                }
            )
            if open_pos and open_pos["side"] == "BUY":
                hedge_cooldown = 8
            elif open_pos:
                hedge_cooldown = max(hedge_cooldown, 3)
            feat = open_pos.get("feat") if open_pos else None
            y = 1 if pnl_inr > 0 else 0
            open_pos = None
            if feat:
                closed_X.append(feat)
                closed_y.append(y)
                if len(closed_y) >= 40 and len(closed_y) % 8 == 0:
                    try:
                        meta = TradeMetaLabeler(min_samples=40)
                        meta.train(pd.DataFrame(closed_X), pd.Series(closed_y))
                    except Exception:
                        pass

        for i, row in df.iterrows():
            if equity < halt:
                if open_pos is not None:
                    close_at(i, "equity_halt")
                break

            if hedge_cooldown > 0:
                hedge_cooldown -= 1
            decision = decide(row)
            side, opt, action = decision.side, decision.option_type, decision.label
            if side == "BUY" and hedge_cooldown > 0:
                side, opt, action = None, None, "FLAT (hedge cooldown)"
            if open_pos is not None:
                open_pos["age"] += 1
                changed = (side, opt) != (open_pos["side"], open_pos["opt"])
                if changed or open_pos["age"] >= OPTION_HOLD_DAYS:
                    close_at(i, "signal_change" if changed else "max_hold")

            if open_pos is None and side and opt:
                feat = {
                    "vrp": float(row["vrp"]) if pd.notna(row.get("vrp")) else 0.0,
                    "ou_z_score": float(row["ou_z_score"]) if pd.notna(row.get("ou_z_score")) else 0.0,
                    "prob_regime_high_vol": float(row["prob_regime_high_vol"]),
                    "macro_sentiment": float(row["macro_sentiment"] or 0.0),
                    "conviction": float(decision.conviction),
                    "vix_forecast_5d": float(row["vix_forecast_5d"]) if pd.notna(row.get("vix_forecast_5d")) else 0.0,
                }
                p_win = None
                if meta is not None and meta.is_fitted:
                    try:
                        p_win = float(meta.predict_success_probability(pd.DataFrame([feat]))[0])
                    except Exception:
                        p_win = None
                if p_win is not None and p_win < 0.48:
                    continue
                if p_win is not None:
                    decision.conviction = min(1.0, decision.conviction * (0.5 + p_win))
                strike = round_strike(float(row["spot"]), OTM_PCT, opt, STRIKE_STEP)
                entry_px = max(self._price(float(row["spot"]), strike, OPTION_TENOR_DAYS, float(row["indiavix"]), opt), 0.05)
                lots = self._lots(equity, entry_px, side, decision.conviction)
                if lots <= 0:
                    continue
                open_pos = {
                    "date_in": row["date"],
                    "side": side,
                    "opt": opt,
                    "action": action,
                    "strike": strike,
                    "lots": lots,
                    "entry_px": entry_px,
                    "spot_in": float(row["spot"]),
                    "iv_in": float(row["indiavix"]),
                    "age": 0,
                    "p_high": float(row["prob_regime_high_vol"]),
                    "conviction": float(decision.conviction),
                    "feat": feat,
                    "sent": float(row["macro_sentiment"] or 0.0),
                    "vrp": float(row["vrp"]) if pd.notna(row.get("vrp")) else 0.0,
                    "ou_z": float(row["ou_z_score"]) if pd.notna(row.get("ou_z_score")) else 0.0,
                }

        if open_pos is not None and len(df):
            close_at(len(df) - 1, "end_of_sample")

        blotter = pd.DataFrame(trades)
        if blotter.empty:
            logger.warning("No option trades generated")
            return blotter
        self.db.save_polars_df(pl.from_pandas(blotter), "option_trades", mode="overwrite")
        logger.info("Wrote %s option trades to option_trades", len(blotter))
        return blotter

    def print_tearsheet(self, blotter: pd.DataFrame) -> Dict[str, float]:
        if blotter is None or blotter.empty:
            print("No option trades to summarise.")
            return {}
        pnl = blotter["pnl_inr"].to_numpy()
        equity = blotter["equity_after"].to_numpy() if "equity_after" in blotter.columns else self.initial_capital + np.cumsum(pnl)
        n_days = max((pd.to_datetime(blotter["date_out"]).max() - pd.to_datetime(blotter["date_in"]).min()).days, 1)
        ret = float(equity[-1] / self.initial_capital - 1.0)
        cagr = float((equity[-1] / self.initial_capital) ** (365.0 / n_days) - 1.0) if equity[-1] > 0 else -1.0
        peak = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - peak) / np.where(peak == 0, 1.0, peak)))
        losses = pnl[pnl < 0]
        gains = pnl[pnl > 0]
        stats = {
            "trades": int(len(blotter)),
            "win_rate": float(blotter["win"].mean()),
            "total_pnl": float(pnl.sum()),
            "avg_pnl": float(pnl.mean()),
            "final_equity": float(equity[-1]),
            "return": ret,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "profit_factor": float(gains.sum() / abs(losses.sum())) if len(losses) else float("inf"),
            "flat_skipped": "quiet-harvest: most days are FLAT",
        }
        print("\n=== OPTIONS P&L (connected vol book, SABR marks) ===")
        print("Risk cap on live equity. Long vol only on cheap/break; short vol when z/forecast agree.")
        print(f"Trades:             {stats['trades']}")
        print(f"Win rate:           {stats['win_rate']:.1%}")
        print(f"Total P&L:          ₹{stats['total_pnl']:,.0f}")
        print(f"Avg P&L / trade:    ₹{stats['avg_pnl']:,.0f}")
        print(f"End equity:         ₹{stats['final_equity']:,.0f}")
        print(f"Total return:       {stats['return']:.2%}")
        print(f"CAGR:               {stats['cagr']:.2%}")
        print(f"Max drawdown:       {stats['max_drawdown']:.2%}")
        print(f"Profit factor:      {stats['profit_factor']:.2f}")
        if "action" in blotter.columns:
            print("\nBy action:")
            print(blotter.groupby("action").agg(n=("win", "size"), win=("win", "mean"), pnl=("pnl_inr", "sum")).to_string())
        print("================================================")
        cols = [c for c in ("date_in", "date_out", "action", "lots", "entry_px", "exit_px", "pnl_inr", "equity_after", "win") if c in blotter.columns]
        print(blotter[cols].tail(8).to_string(index=False))
        return stats
