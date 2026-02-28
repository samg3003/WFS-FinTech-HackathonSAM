"""
AdvisorIQ — FastAPI Backend

Serves all model outputs, client optimisations, and LLM narratives to the dashboard UI.
Loads models once at startup; all subsequent requests are fast inference.
"""

import logging
import os
import sys
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import (
    ASSET_TICKERS, MODELS_DIR, CACHE_DIR, ANTHROPIC_API_KEY,
    VOLSSM_FEATURE_COLS, VOLSSM_SEQ_LEN,
)
from config.clients import CLIENTS
from src.inference.signal_engine import SignalEngine
from src.inference.optimizer import (
    compute_historical_cov, compute_iv_adjusted_cov,
    expected_returns_from_prices, optimise_all_clients,
)
from src.llm.narrator import generate_narrative, generate_chatbot_response
from src.data.pipeline import get_clean_data, get_macro_data, fetch_all_ivs
from src.data.features_a import build_ticker_features
from src.data.features_b import build_hmm_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AdvisorIQ", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────
# Global State (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────

STATE = {
    "signal_engine": None,
    "signal_output": None,
    "client_results": None,
    "prices": None,
    "returns": None,
    "macro": None,
    "current_ivs": None,
    "narratives": {},
    "ready": False,
}


# ─────────────────────────────────────────────────────────────────────
# Startup: Load models + compute signals
# ─────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load models and pre-compute all signals for the demo."""
    logger.info("=" * 60)
    logger.info("AdvisorIQ starting up...")
    logger.info("=" * 60)

    try:
        # 1) Load data from cache
        logger.info("Loading market data...")
        prices, returns, volume = get_clean_data(use_cache=True)
        macro = get_macro_data(use_cache=True)

        STATE["prices"] = prices
        STATE["returns"] = returns
        STATE["macro"] = macro

        # 2) Load signal engine
        logger.info("Loading models...")
        engine = SignalEngine()
        engine.load_models()
        STATE["signal_engine"] = engine

        # 3) Build current features for each ticker
        logger.info("Building features...")
        vix_col = "^VIX" if "^VIX" in macro.columns else macro.columns[0]
        vix = macro[vix_col].reindex(returns.index).ffill()

        ticker_features = {}
        for ticker in ASSET_TICKERS:
            if ticker not in returns.columns:
                continue
            features = build_ticker_features(
                prices=prices[ticker],
                returns=returns[ticker],
                volume=volume[ticker] if ticker in volume.columns else pd.Series(0, index=returns.index),
                vix=vix,
            )
            # Take last 252 rows as the feature window
            feat_vals = features[VOLSSM_FEATURE_COLS].values.astype(np.float32)
            if len(feat_vals) >= VOLSSM_SEQ_LEN:
                ticker_features[ticker] = feat_vals[-VOLSSM_SEQ_LEN:]

        # 4) Get current IVs (try live, fallback to HV_21d estimate)
        logger.info("Fetching implied volatilities...")
        try:
            current_ivs = fetch_all_ivs()
        except Exception as e:
            logger.warning("Live IV fetch failed: %s — using HV estimates", e)
            current_ivs = {}

        # Fill missing IVs with last known HV_21d * 1.1 (slight IV premium)
        for ticker in ASSET_TICKERS:
            if current_ivs.get(ticker) is None and ticker in ticker_features:
                hv_21d = float(ticker_features[ticker][-1, 3])  # hv_21d is index 3
                current_ivs[ticker] = hv_21d * 1.1 if hv_21d > 0 else 0.25

        STATE["current_ivs"] = current_ivs

        # 5) Build HMM features and run signal engine
        logger.info("Computing signals...")
        hmm_features = build_hmm_features(macro)

        signal_output = engine.compute_signals(
            ticker_features=ticker_features,
            current_ivs=current_ivs,
            hmm_features=hmm_features,
        )
        STATE["signal_output"] = signal_output
        signal_data = engine.signals_to_dict(signal_output)

        # 6) Run portfolio optimisation for all clients
        logger.info("Running portfolio optimisation...")
        expected_ret = expected_returns_from_prices(prices)
        hist_cov = compute_historical_cov(returns)

        iv_variances = {t: s.iv_implied_variance for t, s in signal_output.ticker_signals.items()}
        iv_adjusted_cov = compute_iv_adjusted_cov(hist_cov, iv_variances)

        client_results = optimise_all_clients(
            expected_ret, hist_cov, iv_adjusted_cov, current_ivs,
        )
        STATE["client_results"] = client_results

        # 7) Pre-generate narratives
        logger.info("Generating narratives...")
        for client_id, result in client_results.items():
            narrative = generate_narrative(result, signal_data, api_key=ANTHROPIC_API_KEY)
            STATE["narratives"][client_id] = narrative

        STATE["ready"] = True
        logger.info("=" * 60)
        logger.info("AdvisorIQ ready! Serving %d clients, %d tickers",
                    len(client_results), len(signal_output.ticker_signals))
        logger.info("Alerts: %s", signal_output.alerts or "None")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Startup failed: %s", e, exc_info=True)
        # Allow server to start even if models aren't loaded
        STATE["ready"] = False


# ─────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ready" if STATE["ready"] else "loading", "alerts": len(STATE.get("signal_output", {}) and STATE["signal_output"].alerts or [])}


@app.get("/api/dashboard")
def dashboard():
    """Main dashboard data: regime, alerts, all client summaries."""
    if not STATE["ready"]:
        raise HTTPException(503, "System still loading")

    engine = STATE["signal_engine"]
    signal_data = engine.signals_to_dict(STATE["signal_output"])

    client_summaries = []
    for cid, result in STATE["client_results"].items():
        client_summaries.append({
            "client_id": cid,
            "name": result["name"],
            "risk_profile": result["risk_profile"],
            "target_vol": result["target_vol"],
            "current_vol": result["current_stats"]["annual_vol"],
            "vol_misaligned": result["vol_misaligned"],
            "total_drift": result["total_drift"],
            "iv_adjusted_sharpe": result["iv_adjusted_stats"]["sharpe_ratio"],
            "n_alerts": sum(1 for t in signal_data["tickers"].values()
                          if t["fear_level"] != "NONE"),
        })

    return {
        "regime": signal_data["regime"],
        "regime_probs": signal_data["regime_probs"],
        "alerts": signal_data["alerts"],
        "clients": client_summaries,
    }


@app.get("/api/signals")
def signals():
    """Full signal data for all tickers."""
    if not STATE["ready"]:
        raise HTTPException(503, "System still loading")
    return STATE["signal_engine"].signals_to_dict(STATE["signal_output"])


@app.get("/api/client/{client_id}")
def client_detail(client_id: str):
    """Detailed view for a single client."""
    if not STATE["ready"]:
        raise HTTPException(503, "System still loading")

    if client_id not in STATE["client_results"]:
        raise HTTPException(404, f"Client '{client_id}' not found")

    result = STATE["client_results"][client_id]
    signal_data = STATE["signal_engine"].signals_to_dict(STATE["signal_output"])

    return {
        **result,
        "narrative": STATE["narratives"].get(client_id, ""),
        "signals": signal_data,
    }


class ChatRequest(BaseModel):
    client_id: str
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Chatbot endpoint for follow-up questions."""
    if not STATE["ready"]:
        raise HTTPException(503, "System still loading")

    if req.client_id not in STATE["client_results"]:
        raise HTTPException(404, f"Client '{req.client_id}' not found")

    signal_data = STATE["signal_engine"].signals_to_dict(STATE["signal_output"])
    client_result = STATE["client_results"][req.client_id]

    response = generate_chatbot_response(
        question=req.question,
        client_result=client_result,
        signal_data=signal_data,
        api_key=ANTHROPIC_API_KEY,
    )
    return {"response": response}


# ─────────────────────────────────────────────────────────────────────
# Serve UI
# ─────────────────────────────────────────────────────────────────────

UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ui")

@app.get("/")
def serve_ui():
    index = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "AdvisorIQ API running. UI not found at /ui/index.html"}


if os.path.exists(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
