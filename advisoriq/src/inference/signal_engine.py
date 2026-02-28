"""
AdvisorIQ — IVR Signal Engine

Combines Layer A (predicted vol) + Layer B (regime) + live IV into actionable signals.

For each ticker, each day:
    IVR = ATM_IV / predicted_HV
    Fear flag depends on regime-specific threshold + IV percentile condition.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config.settings import (
    ASSET_TICKERS, IVR_THRESHOLDS, IVR_HIGH_FEAR_MULTIPLIER,
    IV_PERCENTILE_THRESHOLD, IV_TRAILING_WINDOW,
    VOLSSM_INPUT_CHANNELS, VOLSSM_D_MODEL, VOLSSM_STATE_SIZE,
    VOLSSM_SEQ_LEN, VOLSSM_N_BLOCKS, VOLSSM_DROPOUT,
    MODELS_DIR,
)
from src.models.vol_ssm import VolSSM

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TickerSignal:
    """Signal output for a single ticker on a single day."""
    ticker: str
    date: str
    atm_iv: float                    # Current ATM implied vol
    predicted_hv: float              # ML-predicted 30d realised vol
    ivr: float                       # IV / predicted HV
    regime: str                      # Current market regime
    ivr_threshold: float             # Regime-dependent threshold
    iv_percentile: float             # Percentile of IV in trailing 252d
    fear_level: str                  # NONE / ELEVATED_FEAR / HIGH_FEAR
    recommended_action: str          # hold / reduce_moderate / reduce_significant
    iv_implied_variance: float       # (ATM_IV)^2 — for covariance matrix


@dataclass
class SignalOutput:
    """Complete signal output for all tickers."""
    date: str
    regime: str
    regime_probs: Dict[str, float]
    ticker_signals: Dict[str, TickerSignal]
    alerts: List[str] = field(default_factory=list)


class SignalEngine:
    """
    Combines VolSSM predictions + HMM regime + live IV into portfolio signals.
    """

    def __init__(self):
        self._vol_models: Dict[str, VolSSM] = {}
        self._hmm = None

    def load_models(self, vol_model_dir: str = None, hmm_model_dir: str = None):
        """Load all trained models from disk."""
        import os
        from src.models.hmm_regime import HMMRegimeClassifier

        if vol_model_dir is None:
            vol_model_dir = os.path.join(MODELS_DIR, "volssm")
        if hmm_model_dir is None:
            hmm_model_dir = os.path.join(MODELS_DIR, "hmm")

        # Load VolSSM models
        for ticker in ASSET_TICKERS:
            path = os.path.join(vol_model_dir, f"volssm_{ticker}.pt")
            if os.path.exists(path):
                model = VolSSM(
                    input_channels=VOLSSM_INPUT_CHANNELS,
                    d_model=VOLSSM_D_MODEL,
                    state_size=VOLSSM_STATE_SIZE,
                    seq_len=VOLSSM_SEQ_LEN,
                    n_blocks=VOLSSM_N_BLOCKS,
                    dropout=VOLSSM_DROPOUT,
                ).to(DEVICE)
                state = torch.load(path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state)
                model.eval()
                self._vol_models[ticker] = model
                logger.info("Loaded VolSSM for %s", ticker)

        # Load HMM
        if os.path.exists(os.path.join(hmm_model_dir, "metadata.json")):
            self._hmm = HMMRegimeClassifier.load(hmm_model_dir)
            logger.info("Loaded HMM regime classifier")

        logger.info("Signal engine: %d vol models, HMM=%s",
                    len(self._vol_models), "loaded" if self._hmm else "missing")

    def predict_vol(self, ticker: str, features: np.ndarray) -> float:
        """
        Predict 30d forward realised vol for a single ticker.

        Parameters
        ----------
        ticker   : asset ticker
        features : (252, 8) numpy array of daily features

        Returns
        -------
        Predicted annualised HV as a float.
        """
        if ticker not in self._vol_models:
            # Fallback: use last known HV from features (hv_21d column, index 3)
            return float(features[-1, 3]) if features.shape[1] > 3 else 0.20

        model = self._vol_models[ticker]
        x = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x, mode='conv')
        return float(pred.cpu().item())

    def get_regime(self, hmm_features: pd.DataFrame) -> tuple:
        """
        Get current market regime from HMM.

        Returns (regime_label, regime_probs_dict)
        """
        if self._hmm is None:
            return "NORMAL", {"LOW_VOL": 0.1, "NORMAL": 0.8, "STRESS": 0.1}

        result = self._hmm.predict_regimes(hmm_features, method="forward_filter")
        current_regime = result["regimes"].iloc[-1]
        current_probs = result["probs"].iloc[-1].to_dict()
        return current_regime, current_probs

    def compute_signals(
        self,
        ticker_features: Dict[str, np.ndarray],  # ticker -> (252, 8) array
        current_ivs: Dict[str, float],            # ticker -> ATM IV
        hmm_features: pd.DataFrame,               # macro features for HMM
        iv_history: Optional[Dict[str, pd.Series]] = None,  # ticker -> historical IV
        as_of_date: str = None,
    ) -> SignalOutput:
        """
        Run the full signal pipeline.

        Parameters
        ----------
        ticker_features : dict mapping ticker -> (252, 8) feature array
        current_ivs     : dict mapping ticker -> current ATM IV (annualised decimal)
        hmm_features    : DataFrame of HMM features (at least last row is "today")
        iv_history      : optional dict mapping ticker -> Series of historical IVs
        as_of_date      : date string for the signal

        Returns
        -------
        SignalOutput with all ticker signals and alerts.
        """
        if as_of_date is None:
            as_of_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        # 1) Get regime
        regime, regime_probs = self.get_regime(hmm_features)

        # 2) Regime-dependent IVR threshold
        ivr_threshold = IVR_THRESHOLDS.get(regime, 1.5)
        high_fear_threshold = ivr_threshold * IVR_HIGH_FEAR_MULTIPLIER

        # 3) Process each ticker
        signals = {}
        alerts = []

        for ticker in ASSET_TICKERS:
            atm_iv = current_ivs.get(ticker)
            features = ticker_features.get(ticker)

            if atm_iv is None or features is None:
                continue

            # Predict vol
            predicted_hv = self.predict_vol(ticker, features)
            predicted_hv = max(predicted_hv, 0.01)  # floor to avoid division by zero

            # Compute IVR
            ivr = atm_iv / predicted_hv

            # IV percentile (trailing 252d)
            if iv_history and ticker in iv_history:
                hist = iv_history[ticker].dropna()
                if len(hist) >= 20:
                    iv_percentile = (hist < atm_iv).mean()
                else:
                    iv_percentile = 0.5
            else:
                iv_percentile = 0.5  # neutral when unknown

            # Fear classification
            dual_condition = (iv_percentile >= IV_PERCENTILE_THRESHOLD)

            if ivr >= high_fear_threshold and dual_condition:
                fear_level = "HIGH_FEAR"
                action = "reduce_significant"
            elif ivr >= ivr_threshold and dual_condition:
                fear_level = "ELEVATED_FEAR"
                action = "reduce_moderate"
            else:
                fear_level = "NONE"
                action = "hold"

            signal = TickerSignal(
                ticker=ticker,
                date=as_of_date,
                atm_iv=atm_iv,
                predicted_hv=predicted_hv,
                ivr=ivr,
                regime=regime,
                ivr_threshold=ivr_threshold,
                iv_percentile=iv_percentile,
                fear_level=fear_level,
                recommended_action=action,
                iv_implied_variance=atm_iv ** 2,
            )
            signals[ticker] = signal

            if fear_level != "NONE":
                alerts.append(f"{ticker}: {fear_level} (IVR={ivr:.2f}, regime={regime})")

        return SignalOutput(
            date=as_of_date,
            regime=regime,
            regime_probs=regime_probs,
            ticker_signals=signals,
            alerts=alerts,
        )

    def signals_to_dict(self, output: SignalOutput) -> dict:
        """Convert SignalOutput to a JSON-serialisable dict for the API."""
        return {
            "date": output.date,
            "regime": output.regime,
            "regime_probs": output.regime_probs,
            "alerts": output.alerts,
            "tickers": {
                ticker: {
                    "ticker": s.ticker,
                    "atm_iv": round(s.atm_iv, 4),
                    "predicted_hv": round(s.predicted_hv, 4),
                    "ivr": round(s.ivr, 3),
                    "regime": s.regime,
                    "ivr_threshold": s.ivr_threshold,
                    "iv_percentile": round(s.iv_percentile, 3),
                    "fear_level": s.fear_level,
                    "recommended_action": s.recommended_action,
                    "iv_implied_variance": round(s.iv_implied_variance, 6),
                }
                for ticker, s in output.ticker_signals.items()
            },
        }
