#!/usr/bin/env python3
"""
AdvisorIQ — Train all models (Layer A + Layer B).
Usage: python scripts/train.py [--epochs 200] [--tickers AAPL MSFT ...]
"""
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import VOLSSM_EPOCHS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train AdvisorIQ models")
    parser.add_argument("--epochs", type=int, default=VOLSSM_EPOCHS, help="Training epochs for VolSSM")
    parser.add_argument("--tickers", nargs="+", default=None, help="Override ticker list")
    parser.add_argument("--skip-volssm", action="store_true", help="Skip Layer A training")
    parser.add_argument("--skip-hmm", action="store_true", help="Skip Layer B training")
    args = parser.parse_args()

    # Train Layer B (HMM) first — faster, no GPU needed
    if not args.skip_hmm:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Training Layer B (HMM Regime Classifier)")
        logger.info("=" * 60)
        from src.training.train_hmm import run_hmm_training
        run_hmm_training()

    # Train Layer A (VolSSM) — per-ticker, GPU-accelerated
    if not args.skip_volssm:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Training Layer A (VolSSM per-ticker)")
        logger.info("=" * 60)
        from src.training.train_vol_ssm import run_full_training
        run_full_training(tickers=args.tickers, n_epochs=args.epochs)

    logger.info("\n" + "=" * 60)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
