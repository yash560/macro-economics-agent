"""
Configuration and Global Settings for Macro Economics Agent Pipeline.
"""

from pathlib import Path
import os
from datetime import datetime, timedelta

# Project Root Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "macro_trading.sqlite"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Time Horizons (Last 180 Days)
END_DATE = datetime(2026, 10, 12)
START_DATE = END_DATE - timedelta(days=180)

START_DATE_STR = START_DATE.strftime("%Y-%m-%d")
END_DATE_STR = END_DATE.strftime("%Y-%m-%d")

# Target Assets for Macro Permutations
TARGET_ASSETS = ["SPY", "QQQ", "TLT", "GLD", "UUP"]
BENCHMARK_ASSET = "SPY"

# Macroeconomic Events Monitored
EVENT_TYPES = [
    "FOMC_RATE_DECISION",
    "CPI_INFLATION",
    "CORE_PCE",
    "NON_FARM_PAYROLLS",
    "UNEMPLOYMENT_RATE",
    "GDP_GROWTH",
    "RETAIL_SALES"
]

# API Keys & Defaults (Can be overridden via environment variables)
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "apify_api_macro_crowdwisdom_demo_token")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "tvly-macro-agent-live-eval-token")
EXA_API_KEY = os.getenv("EXA_API_KEY", "exa-macro-agent-intel-token")
APIFY_ACTOR_ID = "m-s/investing-economic-calendar"

# Model Hyperparameters & Validation
WALK_FORWARD_TRAIN_DAYS = 90
WALK_FORWARD_TEST_DAYS = 15
RANDOM_STATE = 42
RISK_FREE_RATE = 0.045  # 4.5% US Treasury Bill yield
TRANSACTION_COST_BPS = 5.0  # 5 bps per rebalance trade
