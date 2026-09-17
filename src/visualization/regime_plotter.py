from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from bootstrap import SRC_DIR  # noqa: F401
from config import DUCKDB_PATH
from data_pipeline.db_utils import DatabaseManager

logger = logging.getLogger(__name__)


class RegimeVisualizer:
    def __init__(self):
        self.db = DatabaseManager(DUCKDB_PATH)

    def plot_regimes(self, show: bool = True):
        df = self.db.execute(
            """
            SELECT date, indiavix, prob_regime_high_vol, macro_sentiment
            FROM regime_predictions
            ORDER BY date ASC
            """,
            read_only=True,
        ).to_pandas()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(df["date"], df["indiavix"], color="black", linewidth=1.5, label="INDIAVIX")
        ax1.set_title("INDIAVIX vs Causal Markov Regimes & Sentiment", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Implied Volatility")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ymin = float(np.nanmin(df["indiavix"]))
        ymax = float(np.nanmax(df["indiavix"]))
        pad = 0.05 * (ymax - ymin or 1.0)
        ax1.fill_between(
            df["date"],
            ymin - pad,
            ymax + pad,
            where=(df["prob_regime_high_vol"] > 0.5),
            color="red",
            alpha=0.15,
            label="High Vol Regime Detected",
        )
        ax1.legend(loc="upper left")

        ax2.plot(df["date"], df["prob_regime_high_vol"], color="red", linewidth=1.5, label="P(High Vol Regime)")
        ax2.bar(df["date"], df["macro_sentiment"], color="blue", alpha=0.3, width=2, label="GenAI Macro Sentiment")
        ax2.set_ylabel("Probability / Score")
        ax2.set_xlabel("Date")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        if show:
            plt.show()
        return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    RegimeVisualizer().plot_regimes()
