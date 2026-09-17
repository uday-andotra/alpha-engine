# Alpha Engine

NIFTY / INDIAVIX research stack with optional RBI statement scores.

RBI PDFs are **manual**. Put them in `data/raw/mpc_archive/` yourself.
Nothing is downloaded from the internet for policy documents.

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

Copy MPC / Governor statement PDFs into:

```
data/raw/mpc_archive/
```

Name files with a date if you can, e.g. `MPC_2026-08-05.pdf`.
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
pytest -q
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
