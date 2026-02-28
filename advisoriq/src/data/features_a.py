"""
AdvisorIQ — Feature Engineering for Layer A (VolSSM)

Produces the 8-channel feature matrix and 30-day forward RV target per ticker.
Strict contract: [ret_1d, ret_5d, hv_10d, hv_21d, hv_63d, vol_z_63d, vix_level, ivr_lag_5d]
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    TRADING_DAYS_PER_YEAR, VOLSSM_FEATURE_COLS, VOLSSM_SEQ_LEN,
    FORWARD_RV_DAYS, TRAIN_END, VAL_END,
)
from src.data.pipeline import annualise_volatility

logger = logging.getLogger(__name__)


def build_ticker_features(
    prices: pd.Series,
    returns: pd.Series,
    volume: pd.Series,
    vix: pd.Series,
    atm_iv: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build the 8-channel daily feature vector for a single ticker.
    All features computed "as of" day D using only data up to D.

    Parameters
    ----------
    prices  : Adjusted close for the ticker (DatetimeIndex)
    returns : Daily log returns for the ticker
    volume  : Daily trading volume
    vix     : VIX close series (same date index)
    atm_iv  : Optional ATM 30d IV series; if None, ivr_lag_5d = NaN

    Returns
    -------
    DataFrame with columns matching VOLSSM_FEATURE_COLS, indexed by date.
    """
    idx = returns.index

    # 1) Lagged returns
    ret_1d = returns.copy().rename("ret_1d")
    ret_5d = returns.rolling(5).sum().rename("ret_5d")

    # 2) Realised volatility (annualised)
    hv_10d = annualise_volatility(returns.rolling(10).std()).rename("hv_10d")
    hv_21d = annualise_volatility(returns.rolling(21).std()).rename("hv_21d")
    hv_63d = annualise_volatility(returns.rolling(63).std()).rename("hv_63d")

    # 3) Volume z-score (63d trailing)
    vol_mean = volume.rolling(63).mean()
    vol_std = volume.rolling(63).std().replace(0, np.nan)
    vol_z_63d = ((volume - vol_mean) / vol_std).rename("vol_z_63d")

    # 4) VIX level (aligned to ticker index)
    vix_aligned = vix.reindex(idx).ffill().rename("vix_level")

    # 5) IV/HV ratio lagged 5 days
    if atm_iv is not None and len(atm_iv.dropna()) > 0:
        atm_iv_aligned = atm_iv.reindex(idx).ffill()
        iv_over_hv = atm_iv_aligned / hv_21d
        ivr_lag_5d = iv_over_hv.shift(5).rename("ivr_lag_5d")
    else:
        # When IV not available, fill with 1.0 (neutral ratio)
        ivr_lag_5d = pd.Series(1.0, index=idx, name="ivr_lag_5d")

    features = pd.concat([ret_1d, ret_5d, hv_10d, hv_21d, hv_63d,
                          vol_z_63d, vix_aligned, ivr_lag_5d], axis=1)
    features.columns = VOLSSM_FEATURE_COLS
    return features


def compute_forward_rv(returns: pd.Series, horizon: int = FORWARD_RV_DAYS) -> pd.Series:
    """
    30-day forward realised vol: annualised std of the next `horizon` daily
    log-returns. Label for date D uses returns D+1 ... D+horizon.
    """
    return (
        returns.shift(-1)
        .rolling(horizon)
        .std()
        .shift(-(horizon - 1))
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).rename("y_hv_fwd_30d")


def build_training_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    seq_len: int = VOLSSM_SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Build supervised (X, y) pairs for training.

    For each eligible anchor date D:
        X[i] = features[D-251 : D] → shape (252, 8)
        y[i] = forward_rv at D     → scalar

    Returns
    -------
    X      : (N, 252, 8) float32
    y      : (N,) float32
    dates  : list of anchor dates
    """
    dataset = features.join(target, how="inner").dropna()
    feat_vals = dataset[VOLSSM_FEATURE_COLS].values.astype(np.float32)
    label_vals = dataset["y_hv_fwd_30d"].values.astype(np.float32)
    date_vals = dataset.index

    X_list, y_list, d_list = [], [], []

    for i in range(seq_len - 1, len(dataset)):
        X_window = feat_vals[i - seq_len + 1: i + 1]
        if np.isnan(X_window).any():
            continue
        X_list.append(X_window)
        y_list.append(label_vals[i])
        d_list.append(date_vals[i])

    X = np.stack(X_list) if X_list else np.zeros((0, seq_len, len(VOLSSM_FEATURE_COLS)))
    y = np.array(y_list) if y_list else np.zeros(0)

    logger.info("Built %d training sequences (seq_len=%d)", len(X), seq_len)
    return X, y, d_list


def build_all_ticker_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame,
    vix: pd.Series,
    ivs: Optional[Dict[str, pd.Series]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]]:
    """
    Build training data for all tickers.

    Returns dict: ticker -> (X, y, dates)
    """
    all_data = {}

    for ticker in prices.columns:
        logger.info("Building features for %s...", ticker)

        atm_iv = ivs.get(ticker) if ivs else None

        features = build_ticker_features(
            prices=prices[ticker],
            returns=returns[ticker],
            volume=volume[ticker] if ticker in volume.columns else pd.Series(0, index=returns.index),
            vix=vix,
            atm_iv=atm_iv,
        )

        target = compute_forward_rv(returns[ticker])
        X, y, dates = build_training_sequences(features, target)
        all_data[ticker] = (X, y, dates)
        logger.info("  %s: %d samples", ticker, len(X))

    return all_data


def split_by_date(
    X: np.ndarray,
    y: np.ndarray,
    dates: List[pd.Timestamp],
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Temporal split into train/val/test. No leakage.
    """
    dates_arr = pd.DatetimeIndex(dates)
    train_mask = dates_arr <= train_end
    val_mask = (dates_arr > train_end) & (dates_arr <= val_end)
    test_mask = dates_arr > val_end

    return {
        "train": (X[train_mask], y[train_mask]),
        "val": (X[val_mask], y[val_mask]),
        "test": (X[test_mask], y[test_mask]),
    }
