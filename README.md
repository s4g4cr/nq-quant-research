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
| [NQ ORB + HMM](docs/orb_research.html) | Opening Range Breakout with HMM regime routing · Phases 1–7 | Edge confirmed · Sharpe +1.298 OOS |
| [NQ VWAP](docs/vwap_research.html) | VWAP as trading signal · two hypotheses falsified · Phases 8–10 | No edge · 90.7% of crossovers revert |
| [POC Reversion](docs/poc_research.html) | Volume Profile POC mean reversion · deterministic filters · Phases 11–16 | Edge confirmed · p=0.066 · bootstrap p5=0.999 |

Interactive documentation with full phase-by-phase results, equity curves, and parameter tables.

All three are accessible from the research hub:

**[Research Hub — s4g4cr.github.io/nq-quant-research](https://s4g4cr.github.io/nq-quant-research/)**

| Direct links | |
|---|---|
| ORB + HMM (Phases 1–7) | [s4g4cr.github.io/nq-quant-research/orb_research.html](https://s4g4cr.github.io/nq-quant-research/orb_research.html) |
| VWAP Signals (Phases 8–10) | [s4g4cr.github.io/nq-quant-research/vwap_research.html](https://s4g4cr.github.io/nq-quant-research/vwap_research.html) |
| POC Reversion (Phases 11–16) | [s4g4cr.github.io/nq-quant-research/poc_research.html](https://s4g4cr.github.io/nq-quant-research/poc_research.html) |

---

## Confirmed Strategy — POC Reversion B1

The only strategy with statistically confirmed OOS edge:

- **Signal:** price ≥ 1.0 ATR from POC + exhaustion candle + volume > 1.3×
- **TP:** 67% of distance to target POC
- **SL:** 1.0 ATR from entry
- **F1:** prev_day_range / daily_atr < 1.2
- **F2:** poc_distance ≥ 3.0 ATR at signal
- **F3:** abs(trend_5d / daily_atr) < 1.5
- **Result:** PF 1.240 · SR 1.087 · p=0.066 · bootstrap p5=0.999 · 4/5 WFO windows

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

## Project Structure

```
nq-quant-research/
├── strategy/
│   ├── orb.py               — ORB breakout signal engine (Phases 1–7)
│   ├── vwap_reversion.py    — VWAP reversion engine (Phase 8, falsified)
│   ├── vwap_breakout.py     — VWAP breakout engine (Phase 9, falsified)
│   └── poc_reversion.py     — POC mean reversion engine (Phases 11–16, CONFIRMED)
├── indicators/
│   ├── technical.py         — ATR, VWAP, rolling indicators
│   └── volume_profile.py    — prev_poc and session_poc (strictly causal)
├── regime/
│   └── hmm.py               — GaussianHMM classifier (deprecated Phase 14)
├── orb_system/              — importable package used by all run scripts
│   ├── config.py
│   ├── data/loader.py
│   ├── indicators/
│   ├── backtester/engine.py
│   ├── strategy/
│   └── regime/
├── docs/
│   ├── index.html           — Research hub (landing page, links to all three)
│   ├── orb_research.html    — ORB + HMM research (Phases 1–7)
│   ├── vwap_research.html   — VWAP research (Phases 8–10)
│   └── poc_research.html    — POC Reversion research (Phases 11–16)
└── run_phase*.py            — phase entry-points
```

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

# VWAP research (Phases 8–10)
python run_vwap_research.py
python run_phase9.py

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
