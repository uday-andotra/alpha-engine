from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
DUCKDB_PATH = Path(os.getenv("ALPHA_DUCKDB_PATH", DATA_DIR / "duckdb" / "alpha_engine_data.db"))
VECTOR_STORE_PATH = Path(os.getenv("ALPHA_VECTOR_STORE_PATH", DATA_DIR / "vector_store"))
MPC_ARCHIVE_PATH = RAW_DATA_PATH / "mpc_archive"

NIFTY_TICKER = os.getenv("ALPHA_NIFTY_TICKER", "^NSEI")
VIX_TICKER = os.getenv("ALPHA_VIX_TICKER", "^INDIAVIX")
MARKET_START_DATE = os.getenv("ALPHA_MARKET_START", "2015-01-01")

GK_WINDOW = int(os.getenv("ALPHA_GK_WINDOW", "21"))
OU_MIN_OBS = int(os.getenv("ALPHA_OU_MIN_OBS", "252"))
TRADING_DAYS = 252

EMBEDDING_MODEL = os.getenv("ALPHA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("ALPHA_OLLAMA_MODEL", "llama3.1:8b")
VECTOR_COLLECTION = os.getenv("ALPHA_VECTOR_COLLECTION", "rbi_mpc_statements")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

REGIME_K = 2
REGIME_REFIT_EVERY = int(os.getenv("ALPHA_REGIME_REFIT_EVERY", "21"))
HIGH_VOL_THRESHOLD = 0.5

INITIAL_CAPITAL = float(os.getenv("ALPHA_INITIAL_CAPITAL", "1000000"))
RISK_FREE_RATE = float(os.getenv("ALPHA_RF", "0.065"))
ONE_WAY_COST = float(os.getenv("ALPHA_ONE_WAY_COST", "0.0005"))
WIN_LOSS_RATIO = float(os.getenv("ALPHA_PAYOFF_RATIO", "1.5"))
HAWKISH_SENTIMENT_THRESHOLD = 0.1

OPTION_SYMBOL = os.getenv("ALPHA_OPTION_SYMBOL", "NIFTY")
OTM_PCT = float(os.getenv("ALPHA_OTM_PCT", "0.02"))
NIFTY_LOT_SIZE = int(os.getenv("ALPHA_NIFTY_LOT", "65"))
STRIKE_STEP = int(os.getenv("ALPHA_STRIKE_STEP", "50"))
OPTION_TENOR_DAYS = int(os.getenv("ALPHA_OPTION_TENOR", "30"))
OPTION_HOLD_DAYS = int(os.getenv("ALPHA_OPTION_HOLD", "6"))
OPTION_COST_POINTS = float(os.getenv("ALPHA_OPTION_COST_PTS", "0.5"))
RISK_PER_TRADE = float(os.getenv("ALPHA_RISK_PER_TRADE", "0.012"))
MAX_OPTION_LOTS = int(os.getenv("ALPHA_MAX_LOTS", "15"))
MIN_EQUITY_FRAC = float(os.getenv("ALPHA_MIN_EQUITY_FRAC", "0.35"))

for path in (RAW_DATA_PATH, PROCESSED_DATA_PATH, DUCKDB_PATH.parent, VECTOR_STORE_PATH, MPC_ARCHIVE_PATH):
    path.mkdir(parents=True, exist_ok=True)
