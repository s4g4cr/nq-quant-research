# NQ Quantitative Research

Systematic quantitative research on NQ E-mini Futures.
Six research cycles. Five hypotheses falsified or not confirmed. One confirmed
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
| [Failed Spike Reversion](docs/failed_spike_research.html) | Opening 5-min spike failure fade · Phase 17 | No edge · reversion real (71.3%) but entry geometry structurally negative |
| [Spike Extreme S/R](docs/spike_extreme_research.html) | Spike extreme as support/resistance · Phase 18 | No edge · geometry fixed (ATR SL) but OOS win rate collapsed 12pp |
| [Intraday Momentum](docs/intraday_momentum_research.html) | Gao et al. (2018) replication · first 30-min return predicts last · Phase 19 | No edge · directional accuracy 48.6% · NQ shows reversal (p=0.026) not momentum |
| [Intraday Reversal](docs/intraday_reversal_research.html) | Inverse Gao et al. · SHORT when r1>0 · 17-experiment filter optimization · Phases 19B–19C | Not confirmed · SR=1.423 OOS · WFO 3/5 windows · p=0.110 (misses by 0.010) |

Interactive documentation with full phase-by-phase results, equity curves, and parameter tables.

All six are accessible from the research hub:

**[Research Hub — s4g4cr.github.io/nq-quant-research](https://s4g4cr.github.io/nq-quant-research/)**

| Direct links | |
|---|---|
| ORB + HMM (Phases 1–7) | [s4g4cr.github.io/nq-quant-research/orb_research.html](https://s4g4cr.github.io/nq-quant-research/orb_research.html) |
| VWAP Signals (Phases 8–10) | [s4g4cr.github.io/nq-quant-research/vwap_research.html](https://s4g4cr.github.io/nq-quant-research/vwap_research.html) |
| POC Reversion (Phases 11–16) | [s4g4cr.github.io/nq-quant-research/poc_research.html](https://s4g4cr.github.io/nq-quant-research/poc_research.html) |
| Failed Spike (Phase 17) | [s4g4cr.github.io/nq-quant-research/failed_spike_research.html](https://s4g4cr.github.io/nq-quant-research/failed_spike_research.html) |
| Spike Extreme (Phase 18) | [s4g4cr.github.io/nq-quant-research/spike_extreme_research.html](https://s4g4cr.github.io/nq-quant-research/spike_extreme_research.html) |
| Intraday Momentum (Phase 19) | [s4g4cr.github.io/nq-quant-research/intraday_momentum_research.html](https://s4g4cr.github.io/nq-quant-research/intraday_momentum_research.html) |
| Intraday Reversal (Phases 19B–19C) | [s4g4cr.github.io/nq-quant-research/intraday_reversal_research.html](https://s4g4cr.github.io/nq-quant-research/intraday_reversal_research.html) |

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
│   ├── poc_reversion.py     — POC mean reversion engine (Phases 11–16, CONFIRMED)
│   ├── failed_spike.py      — Failed spike reversion engine (Phase 17, falsified)
│   ├── spike_extreme_reversion.py — Spike extreme S/R engine (Phase 18, falsified)
│   ├── intraday_momentum.py       — Gao et al. momentum engine (Phase 19, falsified)
   └── intraday_reversal.py        — Inverse momentum reversal engine (Phases 19B–19C, not confirmed)
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
│   ├── index.html                        — Research hub (landing page, links to all six)
│   ├── orb_research.html                 — ORB + HMM research (Phases 1–7)
│   ├── vwap_research.html                — VWAP research (Phases 8–10)
│   ├── poc_research.html                 — POC Reversion research (Phases 11–16)
│   ├── failed_spike_research.html        — Failed Spike research (Phase 17)
│   ├── spike_extreme_research.html       — Spike Extreme S/R research (Phase 18)
│   ├── intraday_momentum_research.html   — Intraday Momentum research (Phase 19)
   └── intraday_reversal_research.html    — Intraday Reversal research (Phases 19B–19C)
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

# Failed Spike Reversion research (Phase 17)
python run_phase17.py

# Spike Extreme S/R research (Phase 18)
python run_phase18.py

# Intraday Momentum research (Phase 19)
python run_phase19.py

# Intraday Reversal research (Phases 19B–19C)
python run_phase19b.py
python run_phase19c.py
python run_phase19c_wfo.py
```

---

## Disclaimer

This project is for educational and research purposes only. Past backtest results do not guarantee future performance. Nothing here constitutes financial advice.
