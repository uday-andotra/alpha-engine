#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import DUCKDB_PATH, MPC_ARCHIVE_PATH
from data_pipeline.market_fetcher import MarketDataPipeline
from features.quantitative_metrics import VolatilityEngine
from features.signal_aggregator import SignalAggregator
from models.regime_classifier import RegimeClassifier
from data_pipeline.exports import export_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("daily_market")


def main() -> None:
    logger.info("DuckDB file: %s", Path(DUCKDB_PATH).resolve())
    logger.info("MPC archive: %s", Path(MPC_ARCHIVE_PATH).resolve())
    logger.info("Starting daily market pipeline")
    pipe = MarketDataPipeline()
    pipe.fetch_historical_base()
    vol = VolatilityEngine()
    vol.calculate_garman_klass()
    vol.calibrate_ou_process(expanding=True)
    SignalAggregator().fuse_signals()
    RegimeClassifier().fit_markov_switching(causal=True)
    paths = export_tables(("historical_market_data", "market_features", "fused_features", "regime_predictions", "sentiment_events"))
    logger.info("Daily market pipeline finished. VIX source=%s processed=%s", pipe.vix_source, len(paths))


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(
            "\nMissing package:",
            exc,
            "\nInstall project deps first:\n  pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        raise
