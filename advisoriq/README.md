# AdvisorIQ — Options-Informed Portfolio Intelligence

**Hackathon submission: "Data in Finance"**

AdvisorIQ listens to what sophisticated market participants are whispering in the options market and acts on it before it becomes a scream in the price data.

## Architecture

```
Options Market Fear Signal → ML Layer A (VolSSM: predicted vol)
                          → ML Layer B (HMM: market regime)
                          → IVR Signal Engine (fear flags)
                          → IV-Adjusted Portfolio Optimiser
                          → Multi-Client Dashboard + LLM Narrator
```

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (downloads data + trains per-ticker VolSSM + HMM)
python scripts/train.py --epochs 50   # use 200 for full quality

# 3. Launch dashboard
python scripts/serve.py
```

Open `http://localhost:8000` for the dashboard.

## Repository Structure

```
advisoriq/
├── config/
│   ├── settings.py       # All parameters: tickers, dates, thresholds, model hyperparams
│   └── clients.py        # 5 client profiles with distinct risk tolerances
├── src/
│   ├── models/
│   │   ├── vol_ssm.py    # Layer A: S5-based volatility forecaster (unchanged from notebook)
│   │   └── hmm_regime.py # Layer B: HMM regime classifier (unchanged from notebook)
│   ├── data/
│   │   ├── pipeline.py   # Data ingestion, cleaning, caching (Steps 1-8)
│   │   ├── features_a.py # 8-channel feature engineering for VolSSM
│   │   └── features_b.py # 6-feature macro engineering for HMM
│   ├── training/
│   │   ├── train_vol_ssm.py  # Per-ticker VolSSM training pipeline
│   │   └── train_hmm.py      # HMM regime classifier training
│   ├── inference/
│   │   ├── signal_engine.py  # IVR computation + regime-dependent thresholds
│   │   └── optimizer.py      # IV-adjusted covariance + portfolio optimisation
│   ├── llm/
│   │   └── narrator.py       # LLM integration for client explanations + chatbot
│   └── app/
│       └── server.py         # FastAPI backend serving all endpoints
├── ui/
│   └── dashboard.jsx         # React dashboard with client cards + chatbot
├── scripts/
│   ├── train.py              # CLI: train all models
│   └── serve.py              # CLI: launch application
├── artifacts/                # Generated: model checkpoints, cached data
└── requirements.txt
```

## Key Innovation: IVR Signal

**IVR = Implied Vol / ML-Predicted Vol**

- IVR > 1 → options market prices more risk than our model predicts
- Regime-dependent thresholds prevent false alerts:
  - Low Vol regime: IVR > 1.2 triggers alert
  - Normal regime: IVR > 1.5
  - Stress regime: IVR > 1.8

## Evaluation

| Metric | Value |
|--------|-------|
| Model A (VolSSM) | Per-ticker MSE on out-of-sample 2023 data |
| Model B (HMM) | 3-state regime with >90% diagonal persistence |
| Signal quality | Dual condition: IVR threshold + IV 80th percentile |
| Portfolio improvement | Sharpe ratio delta: historical vs IV-adjusted |

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...   # For LLM narrator (optional — template fallback exists)
DATA_START=2021-01-01
DATA_END=2025-01-01
```
