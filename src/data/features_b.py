"""
AdvisorIQ — Feature Engineering for Layer B (HMM Regime Classifier)

Produces the 6-feature macro matrix:
    [f1_vix, f2_vix_chg_20d, f3_slope, f4_spy_ret_20d, f5_hyg_ret_20d, f6_corr_spy_tlt]
"""

import logging
import numpy as np
import pandas as pd

from config.settings import HMM_FEATURE_COLS

logger = logging.getLogger(__name__)


def build_hmm_features(macro_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build 6 macro features for the HMM regime classifier.

    Parameters
    ----------
    macro_prices : DataFrame with columns including:
                   '^VIX', '^VIX3M' (optional), 'SPY', 'HYG', 'TLT'

    Returns
    -------
    DataFrame with 6 columns matching HMM_FEATURE_COLS, indexed by date.
    NaN burn-in rows (~20 trading days) are dropped.
    """
    # Resolve column names (handle yfinance quirks)
    cols = macro_prices.columns.tolist()

    def _find_col(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    vix_col = _find_col(["^VIX", "VIX", "CBOE Volatility Index"])
    vix3m_col = _find_col(["^VIX3M", "VIX3M"])
    spy_col = _find_col(["SPY"])
    hyg_col = _find_col(["HYG"])
    tlt_col = _find_col(["TLT"])

    if vix_col is None:
        raise ValueError(f"VIX column not found. Available: {cols}")
    if spy_col is None or hyg_col is None or tlt_col is None:
        raise ValueError(f"Missing macro columns. Available: {cols}")

    vix = macro_prices[vix_col]
    spy = macro_prices[spy_col]
    hyg = macro_prices[hyg_col]
    tlt = macro_prices[tlt_col]

    # Feature 1: VIX level
    f1_vix = vix.rename("f1_vix")

    # Feature 2: VIX 20d change
    f2_vix_chg_20d = vix.diff(20).rename("f2_vix_chg_20d")

    # Feature 3: VIX term structure slope (VIX3M - VIX)
    if vix3m_col is not None:
        f3_slope = (macro_prices[vix3m_col] - vix).rename("f3_slope")
    else:
        # Fallback: use a constant or zero when VIX3M unavailable
        logger.warning("VIX3M not available — using zero for f3_slope")
        f3_slope = pd.Series(0.0, index=vix.index, name="f3_slope")

    # Feature 4: SPY 20d log return
    f4_spy_ret_20d = np.log(spy / spy.shift(20)).rename("f4_spy_ret_20d")

    # Feature 5: HYG 20d log return
    f5_hyg_ret_20d = np.log(hyg / hyg.shift(20)).rename("f5_hyg_ret_20d")

    # Feature 6: SPY/TLT 20d rolling correlation
    spy_lr = np.log(spy / spy.shift(1))
    tlt_lr = np.log(tlt / tlt.shift(1))
    f6_corr_spy_tlt = spy_lr.rolling(20).corr(tlt_lr).rename("f6_corr_spy_tlt")

    # Combine and drop burn-in
    features = pd.concat([f1_vix, f2_vix_chg_20d, f3_slope,
                          f4_spy_ret_20d, f5_hyg_ret_20d, f6_corr_spy_tlt], axis=1)
    features.columns = HMM_FEATURE_COLS
    features = features.dropna()

    # Clip correlation to [-1, 1] for safety
    features["f6_corr_spy_tlt"] = features["f6_corr_spy_tlt"].clip(-1.0, 1.0)

    logger.info("HMM features: %d rows, date range %s → %s",
                len(features), features.index[0].date(), features.index[-1].date())
    return features
