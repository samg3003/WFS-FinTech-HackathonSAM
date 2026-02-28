"""
AdvisorIQ — Layer B: HMM Regime Classifier

Unsupervised Hidden Markov Model that classifies market regimes.
States are labelled by sorting on mean VIX: LOW_VOL < NORMAL < STRESS [< CRISIS].
Supports causal forward-filter (for live/backtest) and non-causal predict_proba.

Unchanged from validated notebook.
"""

from __future__ import annotations

import json
import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HMMConfig:
    n_states: int = 3
    do_model_selection: bool = False
    selection_ks: Tuple[int, ...] = (2, 3, 4, 5)
    n_init: int = 20
    n_iter: int = 500
    tol: float = 1e-5
    covariance_type: str = "full"
    random_state: int = 42
    train_cutoff_date: str = "2023-01-01"
    min_occupancy_pct: float = 5.0
    min_diagonal_persistence: float = 0.90
    feature_names: Tuple[str, ...] = (
        "f1_vix", "f2_vix_chg_20d", "f3_slope",
        "f4_spy_ret_20d", "f5_hyg_ret_20d", "f6_corr_spy_tlt",
    )
    n_features: int = 6

    @staticmethod
    def compute_n_params(k: int, d: int = 6) -> int:
        return k * d + k * d * (d + 1) // 2 + k * (k - 1) + (k - 1)

    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "do_model_selection": self.do_model_selection,
            "selection_ks": list(self.selection_ks),
            "n_init": self.n_init, "n_iter": self.n_iter, "tol": self.tol,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
            "train_cutoff_date": self.train_cutoff_date,
            "min_occupancy_pct": self.min_occupancy_pct,
            "min_diagonal_persistence": self.min_diagonal_persistence,
            "feature_names": list(self.feature_names),
            "n_features": self.n_features,
        }


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

EXPECTED_COLS = (
    "f1_vix", "f2_vix_chg_20d", "f3_slope",
    "f4_spy_ret_20d", "f5_hyg_ret_20d", "f6_corr_spy_tlt",
)


class FeatureContractError(Exception):
    pass


def validate_features(features, config=None, allow_reorder=True):
    """Unified: DataFrame or ndarray -> (np.ndarray[T,6], dates_or_None)."""
    if isinstance(features, pd.DataFrame):
        expected = list(EXPECTED_COLS) if config is None else list(config.feature_names)
        missing = set(expected) - set(features.columns)
        if missing:
            raise FeatureContractError(f"Missing columns: {sorted(missing)}")
        df = features[expected].copy()
        nan_counts = df.isna().sum()
        if nan_counts.any():
            raise FeatureContractError(f"NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        return df.values.astype(np.float64), dates
    elif isinstance(features, np.ndarray):
        if features.ndim != 2 or features.shape[1] != 6:
            raise FeatureContractError(f"Expected [T,6], got {features.shape}")
        return features.astype(np.float64), None
    else:
        raise FeatureContractError(f"Expected DataFrame or ndarray, got {type(features)}")


# ─────────────────────────────────────────────────────────────────────
# Forward Filter (causal probabilities)
# ─────────────────────────────────────────────────────────────────────

def forward_filter(model, X: np.ndarray) -> np.ndarray:
    """P(S_t=k | x_{1:t}) — causal, forward-pass only. Shape [T, K], rows sum to 1."""
    K = model.n_components
    T = X.shape[0]

    filtered = np.zeros((T, K), dtype=np.float64)
    log_emit = model._compute_log_likelihood(X)
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))

    log_alpha = log_startprob + log_emit[0]
    log_alpha -= logsumexp(log_alpha)
    filtered[0] = np.exp(log_alpha)

    for t in range(1, T):
        log_predict = logsumexp(log_alpha[:, None] + log_transmat, axis=0)
        log_alpha = log_predict + log_emit[t]
        log_alpha -= logsumexp(log_alpha)
        filtered[t] = np.exp(log_alpha)

    return filtered


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

_LABEL_TEMPLATES = {
    2: ["LOW_VOL", "STRESS"],
    3: ["LOW_VOL", "NORMAL", "STRESS"],
    4: ["LOW_VOL", "NORMAL", "STRESS", "CRISIS"],
    5: ["LOW_VOL", "NORMAL", "MILD_STRESS", "STRESS", "CRISIS"],
}


def _assign_labels(means_original, k):
    vix_order = np.argsort(means_original[:, 0])
    templates = _LABEL_TEMPLATES.get(k, [f"STATE_{i}" for i in range(k)])
    return {int(vix_order[i]): templates[i] for i in range(k)}


def _fit_single_k(X_train, k, config):
    best_model, best_score = None, -np.inf
    for init_seed in range(config.n_init):
        m = GaussianHMM(
            n_components=k, covariance_type=config.covariance_type,
            n_iter=config.n_iter, tol=config.tol,
            random_state=config.random_state + init_seed,
            init_params="stmc", params="stmc",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_train)
        s = m.score(X_train)
        if s > best_score:
            best_score, best_model = s, m

    logger.info("K=%d: best LL=%.2f", k, best_score)
    return best_model, best_score


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────
# HMM Regime Classifier
# ─────────────────────────────────────────────────────────────────────

class HMMRegimeClassifier:
    """Full Model B: fit -> predict_regimes -> save/load."""

    def __init__(self, config: Optional[HMMConfig] = None):
        self.config = config or HMMConfig()
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[GaussianHMM] = None
        self._label_map: Optional[Dict[int, str]] = None
        self._means_original: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, features, dates=None, train_mask=None):
        seed_all(self.config.random_state)
        X_raw, inferred_dates = validate_features(features, config=self.config)
        if dates is None:
            dates = inferred_dates

        if train_mask is not None:
            train_mask = np.asarray(train_mask, dtype=bool)
        elif dates is not None:
            cutoff = pd.Timestamp(self.config.train_cutoff_date)
            train_mask = dates < cutoff
            logger.info("Train cutoff %s: %d train / %d total",
                        self.config.train_cutoff_date, train_mask.sum(), len(train_mask))
        else:
            logger.warning("No dates/train_mask — using ALL data for train.")
            train_mask = np.ones(X_raw.shape[0], dtype=bool)

        self._scaler = StandardScaler()
        self._scaler.fit(X_raw[train_mask])
        X_scaled = self._scaler.transform(X_raw)
        X_train = X_scaled[train_mask]

        self._model, _ = _fit_single_k(X_train, self.config.n_states, self.config)

        self._means_original = self._scaler.inverse_transform(self._model.means_)
        self._label_map = _assign_labels(self._means_original, self.config.n_states)
        self._fitted = True
        logger.info("HMM fitted: K=%d, labels=%s", self.config.n_states, self._label_map)
        return self

    def predict_regimes(self, features, dates=None, method="forward_filter"):
        self._check_fitted()
        X_raw, inferred_dates = validate_features(features, config=self.config)
        if dates is None:
            dates = inferred_dates
        X_scaled = self._scaler.transform(X_raw)

        state_ids = self._model.predict(X_scaled)

        if method == "forward_filter":
            probs = forward_filter(self._model, X_scaled)
        else:
            probs = self._model.predict_proba(X_scaled)

        K = self.config.n_states
        regime_labels = np.array([self._label_map[s] for s in state_ids])
        prob_col_names = [self._label_map[k] for k in range(K)]

        if dates is not None:
            probs_df = pd.DataFrame(probs, index=dates, columns=prob_col_names)
            regimes_out = pd.Series(regime_labels, index=dates, name="regime")
        else:
            probs_df = pd.DataFrame(probs, columns=prob_col_names)
            regimes_out = regime_labels

        vix_order = np.argsort(self._means_original[:, 0])
        A_ordered = self._model.transmat_[vix_order][:, vix_order]
        ordered_labels = [self._label_map[vix_order[i]] for i in range(K)]
        occupancy = {self._label_map[k]: float(np.mean(state_ids == k) * 100) for k in range(K)}

        return {
            "regimes": regimes_out,
            "state_ids": state_ids,
            "probs": probs_df,
            "diagnostics": {
                "transition_matrix": pd.DataFrame(A_ordered, index=ordered_labels, columns=ordered_labels),
                "state_occupancy_pct": occupancy,
                "emission_means_original": pd.DataFrame(
                    self._means_original,
                    index=[self._label_map[k] for k in range(K)],
                    columns=list(self.config.feature_names),
                ),
            },
        }

    def save(self, path):
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "hmm_model.pkl")
        joblib.dump(self._scaler, path / "scaler.pkl")
        meta = {
            "config": self.config.to_dict(),
            "label_map": {str(k): v for k, v in self._label_map.items()},
            "emission_means_original": self._means_original.tolist(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved to %s", path)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        cfg_dict = meta["config"]
        for k in ("selection_ks", "feature_names"):
            if k in cfg_dict and isinstance(cfg_dict[k], list):
                cfg_dict[k] = tuple(cfg_dict[k])
        config = HMMConfig(**cfg_dict)
        inst = cls(config=config)
        inst._model = joblib.load(path / "hmm_model.pkl")
        inst._scaler = joblib.load(path / "scaler.pkl")
        inst._label_map = {int(k): v for k, v in meta["label_map"].items()}
        inst._means_original = np.array(meta["emission_means_original"])
        inst._fitted = True
        return inst

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
