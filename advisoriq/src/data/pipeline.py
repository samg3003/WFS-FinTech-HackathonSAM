"""
AdvisorIQ — Data Pipeline
Institutional quant preprocessing (all 8 steps from the system guide).

Steps 1-4: Missing values, timestamp alignment, log returns, annualisation
Steps 7-8: Temporal integrity, caching
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    TRADING_DAYS_PER_YEAR, CACHE_DIR, DATA_START, DATA_END,
    TRAIN_END, VAL_END, ASSET_TICKERS, MACRO_TICKERS,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Step 1: Missing Values — forward-fill only
# ─────────────────────────────────────────────────────────────────────

def forward_fill_prices(raw_data: pd.DataFrame) -> pd.DataFrame:
    return raw_data.ffill()


# ─────────────────────────────────────────────────────────────────────
# Step 2: Aligning Timestamps — strict inner join
# ─────────────────────────────────────────────────────────────────────

def align_prices_to_common_dates(prices_df: pd.DataFrame) -> pd.DataFrame:
    data = prices_df.dropna(how="any")
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="first")]
    return data


# ─────────────────────────────────────────────────────────────────────
# Step 3: Log returns
# ─────────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


# ─────────────────────────────────────────────────────────────────────
# Step 4: Annualising
# ─────────────────────────────────────────────────────────────────────

def annualise_volatility(daily_vol):
    return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)


# ─────────────────────────────────────────────────────────────────────
# Step 7: Temporal splits
# ─────────────────────────────────────────────────────────────────────

def temporal_split(data: pd.DataFrame, train_end: str = TRAIN_END, val_end: str = VAL_END):
    train = data[data.index <= train_end]
    val = data[(data.index > train_end) & (data.index <= val_end)]
    test = data[data.index > val_end]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────
# Step 8: Caching
# ─────────────────────────────────────────────────────────────────────

def _cache_key(tickers, start, end, prefix=""):
    s = prefix + "_".join(sorted(tickers)) + f"_{start}_{end}".replace("-", "")
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _cache_path(key, suffix):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}_{suffix}.parquet")


# ─────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: List[str], start: str = DATA_START,
                 end: str = DATA_END) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers."""
    logger.info("Fetching prices for %s (%s → %s)", tickers, start, end)
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.iloc[:, :len(tickers)].copy()
    else:
        prices = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Ensure columns match tickers
    if len(tickers) == 1 and prices.shape[1] == 1:
        prices.columns = tickers

    prices.index = pd.DatetimeIndex(prices.index)
    logger.info("Raw prices shape: %s", prices.shape)
    return prices


def fetch_volume(tickers: List[str], start: str = DATA_START,
                 end: str = DATA_END) -> pd.DataFrame:
    """Fetch daily volume for a list of tickers."""
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Volume" in raw.columns.get_level_values(0):
            volume = raw["Volume"].copy()
        else:
            volume = pd.DataFrame(index=raw.index)
    else:
        volume = raw[["Volume"]].copy() if "Volume" in raw.columns else pd.DataFrame(index=raw.index)

    if isinstance(volume, pd.Series):
        volume = volume.to_frame()

    if len(tickers) == 1 and volume.shape[1] == 1:
        volume.columns = tickers

    volume.index = pd.DatetimeIndex(volume.index)
    return volume


def fetch_macro_prices(start: str = DATA_START, end: str = DATA_END) -> pd.DataFrame:
    """Fetch macro tickers: ^VIX, ^VIX3M, SPY, HYG, TLT."""
    logger.info("Fetching macro data: %s", MACRO_TICKERS)
    raw = yf.download(MACRO_TICKERS, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.iloc[:, :len(MACRO_TICKERS)].copy()
    else:
        prices = raw.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices.index = pd.DatetimeIndex(prices.index)
    prices = forward_fill_prices(prices)
    logger.info("Macro prices shape: %s, columns: %s", prices.shape, list(prices.columns))
    return prices


# ─────────────────────────────────────────────────────────────────────
# IV Fetching (per-ticker, live snapshot)
# ─────────────────────────────────────────────────────────────────────

def fetch_current_iv(ticker: str) -> Optional[float]:
    """
    Get ATM implied vol from the nearest-to-30d expiry.
    Returns annualised IV as a decimal, or None on failure.
    """
    try:
        t = yf.Ticker(ticker)
        S = t.info.get("currentPrice")
        if not S:
            hist = t.history(period="5d")
            if hist.empty:
                return None
            S = float(hist["Close"].iloc[-1])

        exps = t.options
        if not exps:
            return None

        # Pick expiry closest to 30 days
        from datetime import datetime
        today = datetime.now()
        best_exp, best_diff = None, 999
        for exp in exps:
            diff = abs((datetime.strptime(exp, "%Y-%m-%d") - today).days - 30)
            if 15 <= (datetime.strptime(exp, "%Y-%m-%d") - today).days <= 60 and diff < best_diff:
                best_diff = diff
                best_exp = exp

        if best_exp is None and exps:
            best_exp = exps[0]

        if best_exp is None:
            return None

        chain = t.option_chain(best_exp)
        calls = chain.calls
        if calls.empty:
            return None

        atm = calls.loc[(calls["strike"] - S).abs().idxmin()]
        if "impliedVolatility" in atm.index and pd.notna(atm["impliedVolatility"]):
            return float(atm["impliedVolatility"])

        # Fallback: compute from mid price (requires py_vollib)
        return None

    except Exception as e:
        logger.warning("IV fetch failed for %s: %s", ticker, e)
        return None


def fetch_all_ivs(tickers: List[str] = None) -> Dict[str, Optional[float]]:
    """Fetch current IV for all asset tickers."""
    if tickers is None:
        tickers = ASSET_TICKERS
    ivs = {}
    for t in tickers:
        iv = fetch_current_iv(t)
        ivs[t] = iv
        logger.info("  %s IV: %s", t, f"{iv:.4f}" if iv else "N/A")
    return ivs


# ─────────────────────────────────────────────────────────────────────
# Full Pipeline: fetch → clean → cache
# ─────────────────────────────────────────────────────────────────────

def get_clean_data(
    tickers: List[str] = None,
    start: str = DATA_START,
    end: str = DATA_END,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: fetch → clean (Steps 1-2) → log returns (Step 3) → cache (Step 8).

    Returns: (prices, returns, volume) — all aligned to common dates.
    """
    if tickers is None:
        tickers = ASSET_TICKERS

    key = _cache_key(tickers, start, end, prefix="asset_")
    prices_path = _cache_path(key, "prices")
    returns_path = _cache_path(key, "returns")
    volume_path = _cache_path(key, "volume")

    if use_cache and os.path.exists(prices_path) and os.path.exists(returns_path):
        logger.info("Loading from cache: %s", CACHE_DIR)
        prices = pd.read_parquet(prices_path)
        returns = pd.read_parquet(returns_path)
        volume = pd.read_parquet(volume_path) if os.path.exists(volume_path) else pd.DataFrame()
        return prices, returns, volume

    # Fetch
    prices = fetch_prices(tickers, start, end)
    volume = fetch_volume(tickers, start, end)

    # Step 1: Forward fill
    prices = forward_fill_prices(prices)
    volume = forward_fill_prices(volume)

    # Step 2: Align
    prices = align_prices_to_common_dates(prices)
    volume = volume.reindex(prices.index).ffill().fillna(0)

    # Validate
    assert not prices.isnull().any().any(), "Prices contain NaN after cleaning"
    assert (prices > 0).all().all(), "Prices contain non-positive values"
    logger.info("Cleaned data: %d days, %d tickers", len(prices), len(prices.columns))

    # Step 3: Log returns
    returns = compute_log_returns(prices)
    volume = volume.reindex(returns.index)

    # Step 8: Cache
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        prices.to_parquet(prices_path)
        returns.to_parquet(returns_path)
        volume.to_parquet(volume_path)
        logger.info("Cached to %s", CACHE_DIR)

    return prices, returns, volume


def get_macro_data(
    start: str = DATA_START,
    end: str = DATA_END,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch and cache macro data."""
    key = _cache_key(MACRO_TICKERS, start, end, prefix="macro_")
    path = _cache_path(key, "macro")

    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)

    macro = fetch_macro_prices(start, end)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        macro.to_parquet(path)

    return macro
