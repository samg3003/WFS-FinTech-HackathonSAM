"""
AdvisorIQ — Training Pipeline for Layer B (HMM Regime Classifier)

Trains the HMM on macro features (VIX, VIX3M, SPY, HYG, TLT derived).
Single model for the entire market — not per-ticker.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    MODELS_DIR, HMM_N_STATES, HMM_N_INIT, HMM_N_ITER,
    HMM_COVARIANCE_TYPE, HMM_TRAIN_CUTOFF,
)
from src.models.hmm_regime import HMMRegimeClassifier, HMMConfig
from src.data.pipeline import get_macro_data
from src.data.features_b import build_hmm_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_hmm_training():
    """Train HMM regime classifier on macro data. Main entry point."""
    logger.info("=" * 60)
    logger.info("LAYER B TRAINING: HMM Regime Classifier")
    logger.info("=" * 60)

    # 1) Fetch macro data
    macro = get_macro_data()
    logger.info("Macro data shape: %s", macro.shape)
    logger.info("Macro columns: %s", list(macro.columns))

    # 2) Build features
    features = build_hmm_features(macro)
    logger.info("HMM features: %d rows", len(features))

    # 3) Configure and train
    config = HMMConfig(
        n_states=HMM_N_STATES,
        n_init=HMM_N_INIT,
        n_iter=HMM_N_ITER,
        covariance_type=HMM_COVARIANCE_TYPE,
        train_cutoff_date=HMM_TRAIN_CUTOFF,
    )

    clf = HMMRegimeClassifier(config=config)
    clf.fit(features)

    # 4) Decode and inspect
    result = clf.predict_regimes(features, method="forward_filter")

    logger.info("\nTransition Matrix:")
    logger.info("\n%s", result["diagnostics"]["transition_matrix"].round(3))

    logger.info("\nEmission Means (original units):")
    logger.info("\n%s", result["diagnostics"]["emission_means_original"].round(3))

    logger.info("\nState Occupancy:")
    for label, pct in result["diagnostics"]["state_occupancy_pct"].items():
        logger.info("  %s: %.1f%%", label, pct)

    # 5) Save
    save_dir = os.path.join(MODELS_DIR, "hmm")
    clf.save(save_dir)
    logger.info("HMM model saved to %s", save_dir)

    # Also save the features for inference use
    features.to_parquet(os.path.join(save_dir, "hmm_features.parquet"))

    return clf, result


if __name__ == "__main__":
    run_hmm_training()
