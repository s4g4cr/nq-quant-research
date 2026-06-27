# NQ Quantitative Research

Systematic quantitative research on NQ E-mini Futures.
Three research cycles. Two hypotheses falsified. One confirmed
statistically. Built on a shared Python infrastructure:
event-driven backtester, walk-forward validation,
and bootstrap Monte Carlo sizing.

---

## Research Documents

| Research | Description | Result |
|---|---|---|
| [NQ ORB + HMM](docs/index.html) | Opening Range Breakout with HMM regime routing · Phases 1–7 | Edge confirmed · Sharpe +1.298 OOS |
| [NQ VWAP](docs/vwap_research.html) | VWAP as trading signal · two hypotheses falsified · Phases 8–10 | No edge · 90.7% of crossovers revert |
| [POC Reversion](docs/poc_research.html) | Volume Profile POC mean reversion · deterministic filters · Phases 11–16 | Edge confirmed · p=0.066 · bootstrap p5=0.999 |

Interactive documentation with full phase-by-phase results, equity curves, and parameter tables:

**[View on GitHub Pages](https://s4g4cr.github.io/nq-quant-research/)**

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

## Infrastructure

Custom Python backtesting system built across the research cycles:

```
orb_system/
├── data/loader.py           — Databento CSV · roll schedule · cache
├── indicators/technical.py  — ATR · VWAP · Volume Profile · POC
├── backtester/engine.py     — event-driven · SL/TP via High/Low
├── regime/hmm.py            — GaussianHMM regime classifier
├── strategy/                — ORB · VWAP · POC signal logic
└── results/                 — trade CSVs · walk-forward outputs
```

---

## Installation

```bash
git clone https://github.com/s4g4cr/nq-quant-research.git
cd nq-quant-research
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
# ORB research (Phases 1–7)
python run_backtest.py
python run_combined.py
python run_wfo_v6.py
python run_monte_carlo.py

# POC Reversion research (Phases 11–16)
python run_phase13.py
python run_phase14.py
python run_phase15.py
python run_phase15_B1.py
python run_phase16.py
```

---

## Disclaimer

This project is for educational and research purposes only. Past backtest results do not guarantee future performance. Nothing here constitutes financial advice.
