"""
AdvisorIQ — Training Pipeline for Layer A (VolSSM)

Trains one VolSSM model per ticker on real market data.
Handles: data loading, feature/label creation, temporal splits, training, artifact saving.
"""

import logging
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    ASSET_TICKERS, MODELS_DIR, CACHE_DIR,
    VOLSSM_INPUT_CHANNELS, VOLSSM_D_MODEL, VOLSSM_STATE_SIZE,
    VOLSSM_SEQ_LEN, VOLSSM_N_BLOCKS, VOLSSM_DROPOUT,
    VOLSSM_SSM_LR, VOLSSM_OTHER_LR, VOLSSM_WEIGHT_DECAY,
    VOLSSM_EPOCHS, VOLSSM_BATCH_SIZE, VOLSSM_GRAD_CLIP,
)
from src.models.vol_ssm import VolSSM, seed_all, make_optimizer
from src.data.pipeline import get_clean_data, get_macro_data
from src.data.features_a import build_ticker_features, compute_forward_rv, build_training_sequences, split_by_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VolDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X, mode='conv')
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=VOLSSM_GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X, mode='conv')
        loss = F.mse_loss(pred, y)
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / max(n, 1)


def train_ticker(
    ticker: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_epochs: int = VOLSSM_EPOCHS,
    save_dir: str = None,
) -> dict:
    """Train a VolSSM model for a single ticker. Returns training history."""

    if len(X_train) < 10:
        logger.warning("  %s: Only %d training samples — skipping", ticker, len(X_train))
        return {"ticker": ticker, "status": "skipped", "reason": "insufficient_data"}

    seed_all(42)
    model = VolSSM(
        input_channels=VOLSSM_INPUT_CHANNELS,
        d_model=VOLSSM_D_MODEL,
        state_size=VOLSSM_STATE_SIZE,
        seq_len=VOLSSM_SEQ_LEN,
        n_blocks=VOLSSM_N_BLOCKS,
        dropout=VOLSSM_DROPOUT,
    ).to(DEVICE)

    optimizer = make_optimizer(model, ssm_lr=VOLSSM_SSM_LR,
                               other_lr=VOLSSM_OTHER_LR, weight_decay=VOLSSM_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    train_loader = DataLoader(VolDataset(X_train, y_train), batch_size=VOLSSM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(VolDataset(X_val, y_val), batch_size=len(X_val), shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_loss = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            logger.info("  %s Epoch %3d: train=%.6f val=%.6f", ticker, epoch, train_loss, val_loss)

    # Restore best and save
    model.load_state_dict(best_state)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(save_dir, f"volssm_{ticker}.pt"))
        with open(os.path.join(save_dir, f"volssm_{ticker}_history.json"), "w") as f:
            json.dump(history, f)

    logger.info("  %s: best val MSE = %.6f (saved)", ticker, best_val_loss)
    return {
        "ticker": ticker,
        "status": "trained",
        "best_val_mse": best_val_loss,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }


def run_full_training(tickers=None, n_epochs=None):
    """Train VolSSM for all tickers. Main entry point."""
    if tickers is None:
        tickers = ASSET_TICKERS
    if n_epochs is None:
        n_epochs = VOLSSM_EPOCHS

    logger.info("=" * 60)
    logger.info("LAYER A TRAINING: VolSSM on %d tickers", len(tickers))
    logger.info("Device: %s, Epochs: %d", DEVICE, n_epochs)
    logger.info("=" * 60)

    # 1) Fetch data
    prices, returns, volume = get_clean_data(tickers)
    macro = get_macro_data()

    # Extract VIX series
    vix_col = "^VIX" if "^VIX" in macro.columns else macro.columns[0]
    vix = macro[vix_col].reindex(returns.index).ffill()

    # 2) Train per-ticker
    save_dir = os.path.join(MODELS_DIR, "volssm")
    results = []

    for ticker in tickers:
        if ticker not in prices.columns:
            logger.warning("  %s not found in price data — skipping", ticker)
            continue

        logger.info("\n--- Training %s ---", ticker)

        features = build_ticker_features(
            prices=prices[ticker],
            returns=returns[ticker],
            volume=volume[ticker] if ticker in volume.columns else pd.Series(0, index=returns.index),
            vix=vix,
            atm_iv=None,  # IV not available for historical training; ivr_lag_5d = 1.0
        )

        target = compute_forward_rv(returns[ticker])
        X, y, dates = build_training_sequences(features, target)

        if len(X) == 0:
            logger.warning("  %s: No valid sequences — skipping", ticker)
            continue

        splits = split_by_date(X, y, dates)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]

        logger.info("  %s splits: train=%d, val=%d, test=%d",
                    ticker, len(X_train), len(X_val), len(splits["test"][0]))

        result = train_ticker(ticker, X_train, y_train, X_val, y_val,
                             n_epochs=n_epochs, save_dir=save_dir)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    for r in results:
        if r["status"] == "trained":
            logger.info("  %s: val MSE = %.6f (%d train, %d val)",
                       r["ticker"], r["best_val_mse"], r["train_samples"], r["val_samples"])
        else:
            logger.info("  %s: %s — %s", r["ticker"], r["status"], r.get("reason", ""))

    # Save summary
    with open(os.path.join(save_dir, "training_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VolSSM models")
    parser.add_argument("--epochs", type=int, default=VOLSSM_EPOCHS)
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()
    run_full_training(tickers=args.tickers, n_epochs=args.epochs)
