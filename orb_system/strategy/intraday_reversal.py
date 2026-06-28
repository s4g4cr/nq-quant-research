"""
Phase 19B strategy engine: Intraday Reversal in NQ (inverse Gao et al. 2018).

HYPOTHESIS: In NQ futures (2021–2026) the first half-hour return NEGATIVELY
predicts the last half-hour return. Institutional profit-taking at the close
drives reversal against the morning direction.

Direction: SHORT if r1 > 0 · LONG if r1 < 0   ← inverted vs Phase 19

Phase 19 diagnostic evidence:
  r1 > 0 → mean r17 = -0.0117%  (reversal, not momentum)
  r1 < 0 → mean r17 = +0.0227%  (reversal, not momentum)
  Two-sample t-test for reversal: t=-1.939, p=0.026

All session geometry, sizing, entry/exit, commission and slippage are
identical to Phase 19.  The ONLY change is the sign of direction.

Session detection and data structures are imported from intraday_momentum.
"""
from datetime import time as dt_time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from orb_system.strategy.intraday_momentum import (
    SessionInfo,
    MomentumTrade,
    detect_sessions,          # re-exported for callers
    SLIP, COMM, PV,
    _ENTRY_T, _EXIT_T,
)

__all__ = [
    "SessionInfo", "MomentumTrade", "detect_sessions",
    "run",
    "SLIP", "COMM", "PV",
]


def run(
    df: pd.DataFrame,
    session_infos: dict,
    atr_1min_series: pd.Series,
    entry_bar_time: dt_time = _ENTRY_T,
    r1_threshold: float = 0.0,
    high_vol_only: bool = False,
    rv_median_dict: Optional[dict] = None,
    vol_filter: bool = False,
    vol_median_dict: Optional[dict] = None,
    r16_agreement: bool = False,
    initial_capital: float = 100_000.0,
    risk_pct: float = 1.0,
) -> List[MomentumTrade]:
    """
    Simulate Phase 19B reversal strategy.
    Direction is INVERTED vs Phase 19: SHORT when r1 > 0, LONG when r1 < 0.
    Entry at close of entry_bar_time. Exit at 15:59 close. No SL, no TP.
    """
    trades: List[MomentumTrade] = []
    capital = float(initial_capital)

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    cl_v     = df["close"].values
    a1_v     = atr_1min_series.values

    for d in sorted(k for k, v in session_infos.items() if v is not None):
        si = session_infos[d]
        if np.isnan(si.r1) or si.r1 == 0.0:
            continue

        if abs(si.r1) <= r1_threshold:
            continue

        if high_vol_only and rv_median_dict is not None:
            rv_med = rv_median_dict.get(d)
            if rv_med is None or np.isnan(rv_med) or si.range_first30 <= rv_med:
                continue

        if vol_filter and vol_median_dict is not None:
            vol_med = vol_median_dict.get(d)
            if vol_med is None or np.isnan(vol_med) or si.vol_first30 <= vol_med:
                continue

        if r16_agreement:
            if np.isnan(si.r16):
                continue
            if np.sign(si.r1) != np.sign(si.r16):
                continue

        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]
        t2p   = {t: p for t, p in zip(times, idxs)}

        if entry_bar_time not in t2p or _EXIT_T not in t2p:
            continue

        entry_pos = t2p[entry_bar_time]
        exit_pos  = t2p[_EXIT_T]

        b_entry_cl = float(cl_v[entry_pos])
        b_exit_cl  = float(cl_v[exit_pos])
        b_atr      = float(a1_v[entry_pos])

        if np.isnan(b_atr) or b_atr <= 0:
            continue

        # ── INVERTED direction ──────────────────────────────────────────────
        direction = "short" if si.r1 > 0 else "long"

        if direction == "long":
            ep = b_entry_cl + SLIP
            xp = b_exit_cl  - SLIP
        else:
            ep = b_entry_cl - SLIP
            xp = b_exit_cl  + SLIP

        risk_usd = capital * risk_pct / 100.0
        n_c      = max(1, int(risk_usd / (b_atr * PV)))

        pnl_pts = (xp - ep) if direction == "long" else (ep - xp)
        pnl_net = pnl_pts * n_c * PV - COMM * n_c

        trade = MomentumTrade(
            date=d, direction=direction,
            r1=si.r1, r17=si.r17,
            range_first30=si.range_first30, r16=si.r16,
            vol_first30=si.vol_first30,
            entry_ts=df.index[entry_pos], entry_price=ep,
            exit_ts=df.index[exit_pos], exit_price=xp,
            n_contracts=n_c, capital_at_entry=capital,
            atr_at_entry=b_atr, pnl_pts=pnl_pts, pnl_net=pnl_net,
        )
        capital += pnl_net
        trades.append(trade)

    return trades
