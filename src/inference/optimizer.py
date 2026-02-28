"""
AdvisorIQ — IV-Adjusted Portfolio Optimisation Layer

Standard mean-variance optimisation with a twist:
    - Replace diagonal (each asset variance) with IV-implied variance
    - Keep off-diagonals (correlations) historically estimated
    - Run in parallel: historical vs IV-adjusted, show the delta

Implements per-client optimisation targeting specific portfolio volatilities.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import ASSET_TICKERS, STRESS_SCENARIOS, TRADING_DAYS_PER_YEAR
from config.clients import ClientProfile, CLIENTS

logger = logging.getLogger(__name__)


def compute_historical_cov(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    """Compute historical covariance matrix from daily log returns."""
    cov = returns.cov()
    if annualise:
        cov = cov * TRADING_DAYS_PER_YEAR
    return cov


def compute_iv_adjusted_cov(
    hist_cov: pd.DataFrame,
    iv_implied_variances: Dict[str, float],
) -> pd.DataFrame:
    """
    Replace diagonal of covariance matrix with IV-implied variances.
    Off-diagonals (correlations) remain historically estimated.

    Parameters
    ----------
    hist_cov             : historical annualised covariance matrix
    iv_implied_variances : dict ticker -> (ATM_IV)^2
    """
    tickers = hist_cov.columns.tolist()
    adjusted = hist_cov.copy()

    # Extract correlation matrix from historical cov
    hist_std = np.sqrt(np.diag(hist_cov.values))
    hist_std[hist_std == 0] = 1e-8
    corr = hist_cov.values / np.outer(hist_std, hist_std)

    # Build new std vector: use IV-implied where available, else historical
    new_std = hist_std.copy()
    for i, ticker in enumerate(tickers):
        if ticker in iv_implied_variances:
            new_std[i] = np.sqrt(iv_implied_variances[ticker])

    # Reconstruct covariance: C_ij = corr_ij * std_i * std_j
    adjusted_vals = corr * np.outer(new_std, new_std)
    adjusted = pd.DataFrame(adjusted_vals, index=tickers, columns=tickers)

    return adjusted


def expected_returns_from_prices(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute annualised expected returns from trailing price data."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mean_daily = log_ret.tail(window).mean()
    return mean_daily * TRADING_DAYS_PER_YEAR


def optimize_portfolio(
    expected_ret: pd.Series,
    cov_matrix: pd.DataFrame,
    target_vol: float = None,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
    risk_free_rate: float = 0.04,
) -> Dict[str, float]:
    """
    Mean-variance optimisation.

    If target_vol is specified, finds the portfolio on the efficient frontier
    closest to that volatility. Otherwise, maximises Sharpe ratio.

    Uses simple analytical approach (no external optimiser dependency).
    Falls back to equal-weight if numerical issues arise.
    """
    tickers = list(cov_matrix.columns)
    n = len(tickers)

    # Ensure alignment
    mu = expected_ret.reindex(tickers).fillna(0.0).values
    sigma = cov_matrix.values

    # Simple constrained optimisation via grid search on risk aversion
    best_sharpe = -np.inf
    best_weights = np.ones(n) / n
    best_vol_diff = np.inf

    for gamma in np.logspace(-2, 3, 200):
        # Unconstrained weights: w = (1/gamma) * Sigma^-1 * mu
        try:
            w = np.linalg.solve(gamma * sigma + 1e-8 * np.eye(n), mu)
        except np.linalg.LinAlgError:
            continue

        # Apply constraints
        w = np.clip(w, min_weight, max_weight)
        w_sum = w.sum()
        if w_sum <= 0:
            w = np.ones(n) / n
        else:
            w = w / w_sum

        # Portfolio stats
        port_ret = w @ mu
        port_var = w @ sigma @ w
        port_vol = np.sqrt(max(port_var, 1e-12))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

        if target_vol is not None:
            vol_diff = abs(port_vol - target_vol)
            if vol_diff < best_vol_diff:
                best_vol_diff = vol_diff
                best_weights = w.copy()
        else:
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w.copy()

    return {ticker: round(float(best_weights[i]), 4) for i, ticker in enumerate(tickers)}


def compute_portfolio_stats(
    weights: Dict[str, float],
    expected_ret: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> dict:
    """Compute portfolio return, vol, Sharpe."""
    tickers = list(cov_matrix.columns)
    w = np.array([weights.get(t, 0) for t in tickers])
    mu = expected_ret.reindex(tickers).fillna(0).values
    sigma = cov_matrix.values

    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(max(w @ sigma @ w, 1e-12)))
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

    return {
        "expected_return": round(port_ret, 4),
        "annual_vol": round(port_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
    }


def stress_test_portfolio(
    weights: Dict[str, float],
    scenario_name: str,
) -> float:
    """Apply a stress scenario to a portfolio. Returns expected loss."""
    scenario = STRESS_SCENARIOS.get(scenario_name, {})
    loss = sum(weights.get(t, 0) * scenario.get(t, 0) for t in weights)
    return round(loss, 4)


def vix_doubling_scenario(
    weights_current: Dict[str, float],
    expected_ret: pd.Series,
    hist_cov: pd.DataFrame,
    current_ivs: Dict[str, float],
    target_vol: float = None,
    max_weight: float = 0.25,
) -> Dict[str, float]:
    """
    Dynamic VIX-doubling scenario: double all IVs, rebuild cov, reoptimise.
    Returns the new optimal weights.
    """
    doubled_iv_vars = {t: (iv * 2) ** 2 for t, iv in current_ivs.items() if iv is not None}
    doubled_cov = compute_iv_adjusted_cov(hist_cov, doubled_iv_vars)
    return optimize_portfolio(expected_ret, doubled_cov, target_vol=target_vol, max_weight=max_weight)


# ─────────────────────────────────────────────────────────────────────
# Per-Client Optimisation
# ─────────────────────────────────────────────────────────────────────

def optimise_for_client(
    client: ClientProfile,
    expected_ret: pd.Series,
    hist_cov: pd.DataFrame,
    iv_adjusted_cov: pd.DataFrame,
    current_ivs: Dict[str, float],
) -> dict:
    """
    Run full optimisation pipeline for a single client.

    Returns dict with: historical_weights, iv_adjusted_weights, deltas,
    portfolio stats for both, stress tests, drift from current.
    """
    # Historical optimisation
    hist_weights = optimize_portfolio(
        expected_ret, hist_cov,
        target_vol=client.target_annual_vol,
        max_weight=client.max_single_position,
    )

    # IV-adjusted optimisation
    iv_weights = optimize_portfolio(
        expected_ret, iv_adjusted_cov,
        target_vol=client.target_annual_vol,
        max_weight=client.max_single_position,
    )

    # Weight deltas
    deltas = {t: round(iv_weights.get(t, 0) - hist_weights.get(t, 0), 4) for t in ASSET_TICKERS}

    # Portfolio stats
    hist_stats = compute_portfolio_stats(hist_weights, expected_ret, hist_cov)
    iv_stats = compute_portfolio_stats(iv_weights, expected_ret, iv_adjusted_cov)
    current_stats = compute_portfolio_stats(client.current_weights, expected_ret, iv_adjusted_cov)

    # Drift: how far current portfolio is from IV-adjusted optimal
    drift = {t: round(client.current_weights.get(t, 0) - iv_weights.get(t, 0), 4) for t in ASSET_TICKERS}
    total_drift = round(sum(abs(v) for v in drift.values()) / 2, 4)

    # Stress tests
    stress_results = {}
    for scenario in STRESS_SCENARIOS:
        stress_results[scenario] = {
            "current": stress_test_portfolio(client.current_weights, scenario),
            "iv_adjusted": stress_test_portfolio(iv_weights, scenario),
        }

    # VIX doubling
    vix_double_weights = vix_doubling_scenario(
        iv_weights, expected_ret, hist_cov, current_ivs,
        target_vol=client.target_annual_vol,
        max_weight=client.max_single_position,
    )

    # Vol alignment check
    vol_misaligned = abs(current_stats["annual_vol"] - client.target_annual_vol) > 0.03

    return {
        "client_id": client.client_id,
        "name": client.name,
        "risk_profile": client.risk_profile,
        "target_vol": client.target_annual_vol,
        "target_return": client.target_return,
        "goals": client.goals,
        "constraints": client.constraints,
        "current_weights": client.current_weights,
        "historical_optimal": hist_weights,
        "iv_adjusted_optimal": iv_weights,
        "weight_deltas": deltas,
        "vix_doubling_weights": vix_double_weights,
        "historical_stats": hist_stats,
        "iv_adjusted_stats": iv_stats,
        "current_stats": current_stats,
        "drift": drift,
        "total_drift": total_drift,
        "vol_misaligned": vol_misaligned,
        "stress_tests": stress_results,
    }


def optimise_all_clients(
    expected_ret: pd.Series,
    hist_cov: pd.DataFrame,
    iv_adjusted_cov: pd.DataFrame,
    current_ivs: Dict[str, float],
) -> Dict[str, dict]:
    """Run optimisation for all 5 clients."""
    results = {}
    for client_id, client in CLIENTS.items():
        logger.info("Optimising for %s (%s)", client.name, client.risk_profile)
        results[client_id] = optimise_for_client(
            client, expected_ret, hist_cov, iv_adjusted_cov, current_ivs,
        )
    return results
