"""
AdvisorIQ — Central Configuration
Single source of truth for all parameters. Change here, nowhere else.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────
ASSET_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "BAC", "V",
    "TSLA", "META",
    "GLD", "TLT",
]

MACRO_TICKERS: List[str] = ["^VIX", "^VIX3M", "SPY", "HYG", "TLT"]

# ─────────────────────────────────────────────────────────────────────
# Date Boundaries (temporal integrity)
# ─────────────────────────────────────────────────────────────────────
DATA_START = os.getenv("DATA_START", "2021-01-01")
DATA_END = os.getenv("DATA_END", "2025-01-01")

TRAIN_END = "2022-12-31"       # Train: 2021-01-01 → 2022-12-31
VAL_END = "2023-12-31"         # Val:   2023-01-01 → 2023-12-31
# Test:  2024-01-01 → end      (never touched until final eval)

# HMM uses same boundary
HMM_TRAIN_CUTOFF = "2023-01-01"

TRADING_DAYS_PER_YEAR = 252

# ─────────────────────────────────────────────────────────────────────
# Model A — VolSSM hyperparameters
# ─────────────────────────────────────────────────────────────────────
VOLSSM_INPUT_CHANNELS = 8
VOLSSM_D_MODEL = 128
VOLSSM_STATE_SIZE = 64
VOLSSM_SEQ_LEN = 252
VOLSSM_N_BLOCKS = 3
VOLSSM_DROPOUT = 0.1

VOLSSM_SSM_LR = 1e-4
VOLSSM_OTHER_LR = 1e-3
VOLSSM_WEIGHT_DECAY = 1e-4
VOLSSM_EPOCHS = 200
VOLSSM_BATCH_SIZE = 32
VOLSSM_GRAD_CLIP = 1.0

# ─────────────────────────────────────────────────────────────────────
# Model A — Feature schema (8 channels, strict contract)
# ─────────────────────────────────────────────────────────────────────
VOLSSM_FEATURE_COLS = [
    "ret_1d",        # 1-day log return
    "ret_5d",        # 5-day log return
    "hv_10d",        # 10-day annualised HV
    "hv_21d",        # 21-day annualised HV
    "hv_63d",        # 63-day annualised HV
    "vol_z_63d",     # Volume z-score (63d trailing)
    "vix_level",     # VIX close (replicated across tickers)
    "ivr_lag_5d",    # (ATM IV / HV_21d) lagged 5 days
]

FORWARD_RV_DAYS = 30            # Target: 30-day forward realised vol

# ─────────────────────────────────────────────────────────────────────
# Model B — HMM hyperparameters
# ─────────────────────────────────────────────────────────────────────
HMM_N_STATES = 3
HMM_N_INIT = 20
HMM_N_ITER = 500
HMM_COVARIANCE_TYPE = "full"

HMM_FEATURE_COLS = [
    "f1_vix",
    "f2_vix_chg_20d",
    "f3_slope",
    "f4_spy_ret_20d",
    "f5_hyg_ret_20d",
    "f6_corr_spy_tlt",
]

# ─────────────────────────────────────────────────────────────────────
# IVR Signal Engine — Regime-dependent thresholds
# ─────────────────────────────────────────────────────────────────────
IVR_THRESHOLDS: Dict[str, float] = {
    "LOW_VOL": 1.2,
    "NORMAL":  1.5,
    "STRESS":  1.8,
    "CRISIS":  2.0,
}

IVR_HIGH_FEAR_MULTIPLIER = 1.3   # HIGH_FEAR = threshold * 1.3
IV_PERCENTILE_THRESHOLD = 0.80   # Must be above 80th percentile of trailing 252d IV
IV_TRAILING_WINDOW = 252

# ─────────────────────────────────────────────────────────────────────
# Stress Test Scenarios (historical drawdowns per asset class)
# ─────────────────────────────────────────────────────────────────────
STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008_GFC": {
        "AAPL": -0.57, "MSFT": -0.44, "GOOGL": -0.56, "AMZN": -0.45, "NVDA": -0.76,
        "JPM": -0.69, "BAC": -0.80, "V": -0.30,
        "TSLA": -0.50, "META": -0.50,  # proxied
        "GLD": 0.05, "TLT": 0.33,
    },
    "2020_COVID": {
        "AAPL": -0.31, "MSFT": -0.30, "GOOGL": -0.34, "AMZN": -0.23, "NVDA": -0.34,
        "JPM": -0.43, "BAC": -0.48, "V": -0.33,
        "TSLA": -0.60, "META": -0.35,
        "GLD": -0.03, "TLT": 0.21,
    },
    "2022_RATE_SHOCK": {
        "AAPL": -0.27, "MSFT": -0.28, "GOOGL": -0.39, "AMZN": -0.50, "NVDA": -0.50,
        "JPM": -0.17, "BAC": -0.26, "V": -0.07,
        "TSLA": -0.65, "META": -0.64,
        "GLD": -0.01, "TLT": -0.41,
    },
}

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
CACHE_DIR = os.path.join(ARTIFACTS_DIR, "cache")
FEATURES_DIR = os.path.join(ARTIFACTS_DIR, "features")

# ─────────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
