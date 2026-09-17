from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config import DUCKDB_PATH
from execution.strategy import prepare_features, route_row
from dashboard.components import render_regime_dashboard_chart
from data_pipeline.db_utils import DatabaseManager

st.set_page_config(page_title="Alpha Engine Terminal", layout="wide", page_icon="⚡")
st.title("⚡ Systematic INDIAVIX & GenAI Sentiment Engine")
st.markdown("---")


@st.cache_data(ttl=60)
def load_prediction_data():
    db = DatabaseManager(DUCKDB_PATH)
    if not db.table_exists("regime_predictions"):
        raise FileNotFoundError("regime_predictions table is missing")
    df = db.execute(
        """
        SELECT date, indiavix, prob_regime_high_vol, macro_sentiment, vrp, ou_z_score
        FROM regime_predictions
        ORDER BY date ASC
        """,
        read_only=True,
    ).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df


def router_action(row) -> str:
    _, _, label = route_row(row)
    return label


try:
    raw_df = prepare_features(load_prediction_data())
    st.sidebar.header("Dashboard Controls")
    min_date = raw_df["date"].min().date()
    max_date = raw_df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (raw_df["date"].dt.date >= start_date) & (raw_df["date"].dt.date <= end_date)
        df = raw_df.loc[mask]
    else:
        df = raw_df

    if df.empty:
        st.warning("No data available for the selected date range.")
    else:
        latest_vix = df["indiavix"].iloc[-1]
        latest_prob = df["prob_regime_high_vol"].iloc[-1]
        latest_sentiment = df["macro_sentiment"].iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("INDIAVIX Index", f"{latest_vix:.2f}")
        c2.metric("High-Vol Regime Prob", f"{latest_prob:.1%}")
        c3.metric("GenAI Sentiment Stance", f"{latest_sentiment:+.2f}")
        c4.metric("Options Router Action", router_action(df.iloc[-1]))
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Interactive Chart", "🗄️ Raw Data Feed", "📒 Option P&L"])
        with tab1:
            st.plotly_chart(render_regime_dashboard_chart(df), use_container_width=True)
        with tab2:
            st.markdown("### Historical Regime & Sentiment Ledger")
            st.dataframe(
                df.sort_values(by="date", ascending=False).style.format(
                    {
                        "indiavix": "{:.2f}",
                        "prob_regime_high_vol": "{:.2%}",
                        "macro_sentiment": "{:+.2f}",
                    }
                ),
                use_container_width=True,
                height=400,
            )
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="alpha_engine_telemetry.csv",
                mime="text/csv",
            )


        with tab3:
            db = DatabaseManager(DUCKDB_PATH)
            if db.table_exists("option_trades"):
                trades = db.execute("SELECT * FROM option_trades ORDER BY date_in DESC", read_only=True).to_pandas()
                st.metric("Closed trades", len(trades))
                if "pnl_inr" in trades.columns and len(trades):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total P&L (₹)", f"{trades['pnl_inr'].sum():,.0f}")
                    c2.metric("Win rate", f"{trades['win'].mean():.1%}" if "win" in trades.columns else "n/a")
                    c3.metric("Avg P&L (₹)", f"{trades['pnl_inr'].mean():,.0f}")
                st.dataframe(trades, use_container_width=True, height=420)
            else:
                st.info("No option_trades table yet. Run `python scripts/run_backtest.py`.")

except Exception as exc:
    st.error(f"Could not load predictions database from {DUCKDB_PATH}: {exc}")
    st.info("Run `python scripts/run_daily_market_update.py` first to build alpha_engine_data.db.")
