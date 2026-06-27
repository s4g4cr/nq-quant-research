"""
Failed Opening Spike Reversion strategy engine for NQ 1-min (Phase 17).

HYPOTHESIS (FALSIFIED):
  The first 5 minutes of NQ trading frequently produce a spike that sweeps
  stops before the real move begins. When that spike fails — price cannot
  hold the extreme and begins reversing — trapped participants fuel a fast,
  clean reversion. Entry on the first confirmation bar in the 09:35–09:54
  window after the spike extreme is set.

RESULT: Hypothesis rejected.
  The reversion behavior is real: 71.3% of sessions with a qualifying spike
  (>1.5×ATR) return to session_open within 60 bars (median 4 bars after signal).
  The tradeable edge is not: the entry sits between the spike extreme (SL) and
  session_open (TP-A), creating structural negative R/R regardless of parameter
  choices.

    TP-A (session_open):   WR 63.5%, avg R/R 0.39, expectancy -0.115
    TP-B (spike opposite): WR 52.2%, avg R/R 0.73, expectancy -0.099

  Key insight: to trade this pattern profitably, the entry would need to be
  BEFORE the spike forms (not after) — a prediction problem, not a confirmation
  problem. Or the SL would need to be inside the spike range, which creates a
  different and untested hypothesis.

  See run_phase17.py for the full diagnostic output.

Strategy parameters tested:
  Spike window:    09:30–09:34 (5 bars); entry window 09:35–09:54
  Spike filter:    spike_magnitude > spike_mult × atr_1min (1-min Wilder ATR)
  Volume filter:   bar volume > 1.3 × avg_volume(20)
  Position filter: close in upper/lower 40% of bar range
  SL:              spike_extreme ± 0.5 pts (fixed)
  Slippage:        0.5 pts per side; Commission: $4.00 RT/contract
  Experiments:     spike_mult sweep [1.5, 2.0, 2.5, 3.0], speed filter,
                   POC confluence, risk % sensitivity

Full implementation: orb_system/strategy/failed_spike.py
"""
