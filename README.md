# Alpha Engine

A local research stack that fuses India VIX state with Reserve Bank of India (RBI) statement language to paper-trade two-percent out-of-the-money (OTM) NIFTY options.

## Overview

Alpha Engine is designed to identify when India variance is cheap or rich conditional on the policy path, global volatility, and the calendar. It builds a mark-to-model from VIX and a default SABR smile, merging quantitative volatility thermometers with a retrieval-augmented local LLM pipeline that extracts hawkish or dovish stances from RBI monetary-policy PDFs.

## Mathematical Framework & Core Logic

The engine evaluates the market through four primary lenses before a daily decision is routed:

* **Garman-Klass Realized Variance:** Captures intra-day diffusion by factoring in open, high, low, and close prices to account for trend and noise.



$$\hat{\sigma}_{GK,t}^{2}=\frac{1}{2}(h_{t}-l_{t})^{2}-(2 \ln 2-1)(c_{t}-o_{t})^{2}$$



* **Variance Risk Premium (VRP):** Calculates the spread between implied volatility (India VIX) and annualized 21-day realized variance to identify if selling insurance is adequately compensated.



$$VRP_{t}=V_{t}-100\cdot\sigma_{t}^{rv}$$



* **Ornstein-Uhlenbeck Mean Reversion:** Models VIX as a mean-reverting stochastic process pulled toward a long-run level, generating an expanding z-score to quantify how overextended fear levels are.



$$z_{t}=\frac{V_{t}-\hat{\mu}_{t}}{\sigma_{t}/\sqrt{2\hat{\theta}_{t}}}$$



* **SABR & Black-Scholes Pricing:** Translates volatility into a rupee price using Black-Scholes as the cash register, draped with a SABR smile expansion to adjust for the richness of OTM downside strikes.



## Repository Architecture

```text
alpha-engine/
├── scripts/
│   ├── run_daily_market_update.py  # Fetches OHLC/VIX, computes GK, VRP, OU z-score, Markov states
│   ├── run_batch_nlp_pipeline.py   # Chunks PDFs, embeds, scores via Ollama, and paints sentiment
│   └── run_backtest.py             # Executes the options blotter and index overlay
├── src/
│   ├── data_pipeline/              # Market fetchers, DuckDB utilities, and PDF scrapers
│   ├── features/                   # Math implementations for GK, VRP, VIX ridge forecast, embeddings
│   ├── models/                     # Markov-switching regimes, local LLM scoring, SABR volatility surface
│   ├── execution/                  # The strategy router, options P&L walk-forward, Half-Kelly sizing
│   └── dashboard/                  # Streamlit app and Plotly visualizations
├── tests/                          # 25-test aligned suite (Pytest) locking math and boundaries
├── notebooks/                      # Jupyter exploratory environments for regimes, sentiment, and P&L
└── data/                           # DuckDB database and exported parquet snapshots

```

## The Daily Decision

The core router (`decide(row)`) executes a strict law based on the computed indicators:

* **Vol Weather:** If VIX is in a shock (≥ 24 and rising) or a violent 5-day jump, weather is breaking (buy puts). If VIX is in its top annual tercile, weather is rich (sell premium, unless vetoed). If VIX is in its bottom tercile, weather is cheap (buy premium, unless vetoed).


* **Wing & Sentiment:** A hawkish RBI stance (positive score) steers shorts toward calls and adds weight to puts. A dovish stance (negative score) steers cheap-vol buys toward calls if the tape is calm, and forbids short puts in a nervous tape.


* **Forecast Veto:** A causal 5-day ridge regression cancels short options if it predicts a VIX rise of > 2.5%, and cancels non-shock longs if it predicts a drop of the same size.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.1:8b    # only needed to score PDFs
```

## 1. Market data

```bash
python scripts/run_daily_market_update.py
```

## 2. Policy PDFs (manual)
**Automating MPC pdf downloads are a Work in Progress via `download_rbi_pdfs.py`**

Copy MPC / Governor statement PDFs into:

```
data/raw/mpc_archive/
```

Name files with a date if you can, e.g. `MPC_2026-08-15.pdf`.
Then score whatever is in that folder:

```bash
python scripts/run_batch_nlp_pipeline.py
```

That reads **only** the archive folder. It writes scores into DuckDB
`sentiment_events` and forward-fills `macro_sentiment`.

## 3. Backtest / dashboard

```bash
python scripts/run_backtest.py
streamlit run src/dashboard/app.py
```

## Tests

```bash
PYTHONPATH=src python -m pytest -x --tb=short
```


## Options P&L

After market + NLP:

```bash
python scripts/run_backtest.py
```

Prints the index overlay *and* an options blotter. Each trade is the original
router (buy 2% OTM put / sell 2% OTM call or put), marked with Black-Scholes
using INDIAVIX as IV. Results land in DuckDB `option_trades` and the dashboard
"Option P&L" tab.

This is a research mark, not an NSE fill.
