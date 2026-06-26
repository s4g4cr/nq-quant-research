# NQ ORB Research — Quantitative Trading System

A full-stack quantitative research system for NQ (E-mini Nasdaq-100 futures) that develops, validates, and stress-tests a regime-filtered Opening Range Breakout strategy. Built across seven research phases, the system evolves from a simple ORB baseline through HMM-based regime detection, walk-forward optimization, and Monte Carlo FTMO challenge sizing — producing a documented, reproducible research pipeline with out-of-sample results.

---

## Research Summary

| Phase | Description | Key Result |
|-------|-------------|------------|
| 1 | ORB SHORT baseline | Established signal structure |
| 2–3 | ORB filter experiments (gap, OR position, trend) | Identified stable filter set |
| 4 | Mean Reversion LONG on OR extremes | PF 1.22 OOS |
| 5 | HMM 3-state regime filter (ranging / trending / volatile) | First regime-filtered combined system |
| 6 | Walk-Forward Optimization (5 anchored windows) | **Sharpe OOS +1.298 \| PF 1.223** |
| 7 | Monte Carlo FTMO sizing (10,000 sims × 7 levels) | **87.2% pass probability at 1 contract** |

**Final OOS period:** December 2024 – June 2026  
**OOS trades:** 194 (ORB SHORT: 99 \| MR: 95)  
**Win rate:** 54.6% \| **R/R:** 1.03:1 \| **Expectancy:** +$35.75/trade (1 contract)  
**FTMO max drawdown p95 (successful paths):** 6.4% — well within the 10% hard limit

---

## Data

The dataset is **not included** in this repository due to [Databento](https://databento.com) licensing terms.

To reproduce, purchase or download the exact dataset:

| Field | Value |
|-------|-------|
| Provider | Databento |
| Dataset | GLBX MDP3 (CME Globex) |
| Schema | OHLCV-1m (1-minute OHLCV bars) |
| Instrument | NQ front-month continuous (roll-adjusted) |
| Date range | 2021-06-25 to 2026-06-24 |
| Expected filename | `glbx-mdp3-20210625-20260624.ohlcv-1m.csv` |

Place the CSV file in the project root before running any script. The roll schedule and cache path are configured in `orb_system/config.py`.

---

## Installation

```bash
git clone https://github.com/s4g4cr/nq-or-backtest.git
cd nq-or-backtest
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

Run phases in order. Each script is self-contained and saves results to `results/`.

```bash
# Phase 1–3: ORB experiments
python run_backtest.py

# Phase 5: HMM regime-filtered combined system
python run_combined.py

# Phase 6: Walk-forward optimization
python run_wfo_v6.py

# Phase 7: Monte Carlo FTMO sizing
python run_monte_carlo.py
```

---

## Project Structure

```
nq-orb-backtest/
├── orb_system/                  # Core library (do not modify)
│   ├── backtester/engine.py     # ORB backtest engine
│   ├── data/loader.py           # Data loading & caching
│   ├── indicators/technical.py  # ATR, avg_vol, candle_rng
│   ├── strategy/
│   │   ├── orb.py               # ORB SHORT strategy wrapper
│   │   └── mean_reversion.py    # MR LONG/SHORT engine
│   ├── regime/
│   │   ├── features.py          # Daily HMM features (causal)
│   │   └── hmm.py               # GaussianHMM 3-state regime model
│   └── config.py                # Centralized configuration
├── run_backtest.py              # Phase 1–4: ORB + MR experiments
├── run_combined.py              # Phase 5: HMM combined system
├── run_wfo_v6.py                # Phase 6: Walk-forward optimization
├── run_monte_carlo.py           # Phase 7: Monte Carlo FTMO sizing
├── results/                     # Generated outputs (gitignored)
├── docs/
│   └── index.html               # Full research documentation (GitHub Pages)
├── requirements.txt
└── README.md
```

---

## Results

Interactive documentation with full phase-by-phase results, equity curves, and parameter tables:

**[View on GitHub Pages](https://s4g4cr.github.io/nq-or-backtest/)**

---

## Research Documents

| Document | Content |
|----------|---------|
| [NQ ORB Research (Phases 1–7)](docs/index.html) | ORB baseline, regime detection, walk-forward optimization, Monte Carlo FTMO sizing |
| [NQ VWAP Research (Phases 8–10)](docs/vwap_research.html) | VWAP Reversion, VWAP Breakout with trailing stop, SL calibration |

---

## Disclaimer

This project is for educational and research purposes only. Past backtest results do not guarantee future performance. Nothing here constitutes financial advice.
