#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from execution.backtester import StrategyBacktester
from execution.options_pnl import OptionsPnLBacktester
from data_pipeline.exports import export_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    print("\n--- Index overlay (reference) ---")
    overlay = StrategyBacktester()
    try:
        results = overlay.run_backtest()
        overlay.print_tearsheet(results)
    except Exception as exc:
        print(f"Overlay backtest skipped: {exc}")

    print("\n--- Options P&L (router + Black-Scholes marks) ---")
    opt = OptionsPnLBacktester()
    blotter = opt.run(use_meta_kelly=True)
    opt.print_tearsheet(blotter)
    export_tables(("option_trades", "backtest_results", "regime_predictions"))


if __name__ == "__main__":
    main()
