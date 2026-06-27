"""
Phase 18 — Opening Spike Extreme as Support/Resistance.

HYPOTHESIS: The opening spike extreme acts as a structural S/R level for
the first 2 hours of the session. Price returning to within 0.5×ATR of
the spike extreme with rejection is a reversion entry toward session_open.

HYPOTHESIS FALSIFIED — Phase 18.

The ATR-based SL fix from Phase 17 resolved the geometry problem.
Median R/R improved to 1.78–3.09 depending on TP variant. Theoretical
expectancy was marginally positive (TP-A: +0.022, TP-B: +0.018).

The tradeable edge was not sustained out-of-sample:

  Diagnostic (full dataset):
    - 85.6% of sessions have spike > 1.5×ATR
    - 93.3% of qualifying sessions retest the spike extreme (09:35–11:30)
    - 81.7% rejection rate at spike extreme level
    - Theoretical WR 32.4% (TP-A, before costs)

  Train 2021–2024 (baseline): PF=1.044  WR=35.7%  SR=0.344
  Test  2025–2026 (baseline): PF=0.733  WR=23.4%  SR=-2.370
  Best combined (Exp 9):      PF=0.915  WR=21.1%  SR=-0.673

Win rate collapsed 12pp from train to test period. All 9 experiment
configurations showed PF < 1.0 in the OOS test window. WFO was not
executed (Exp 9 PF = 0.915 < 1.0; N = 90 < 80 threshold met, but PF not).

Key insight: The signal captures real market structure (spike extremes as
S/R magnets), but the entry bar is one step too late. By the time a
rejection bar confirms at the spike extreme level, the move is either
already exhausted or the adverse excursion (ATR-sized SL) overwhelms
the modest follow-through. In 2025, wider ATRs made SL distances larger
while TP-A (session_open) distance remained constant or shrank.

Parameters tested:
  TP variant: A (session_open), B (session_open ± 0.5×spike), C (opposite extreme)
  spike_mult: 1.5, 2.0, 2.5, 3.0
  Trailing stop: activate at 1.0×ATR, trail at 0.75×ATR
  Retest quality filter: bar range > 0.8×ATR, volume > 1.5×avg
  POC confluence: spike_extreme within 5 pts of prev_poc (N=5 — noise)
  min_rr: 1.0, 1.5, 2.0

Full implementation: orb_system/strategy/spike_extreme_reversion.py
Runner:             run_phase18.py
"""
