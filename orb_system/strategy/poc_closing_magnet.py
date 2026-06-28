"""
Phase 20 — POC Closing Magnet strategy engine.

Hypothesis: In the final hour (15:00–16:00 NY), price gravitates toward
session volume equilibrium (POC). Three POC variants × three entry
conditions = 9 experiments.

POC-A  session_poc at 14:59 (rolling, 09:30–14:59)
POC-B  prev_poc (prior complete session)
POC-C  nearest_poc (whichever of A/B is closer to price at 14:59)

Condition 1  Distance + momentum: displacement ≥ 1.0 ATR + bar moving toward POC
Condition 2  VWAP cross toward POC between 15:00–15:30
Condition 3  Simple displacement at 15:00 (no confirmation needed)

TP = target_poc (fixed at entry). SL = entry ± 1.0 × ATR (fixed at entry).
Hard exit: close of 15:59. SL wins on same-bar conflict. Max 1 trade/session.
"""

from dataclasses import dataclass
from datetime import time as dt_time
from typing import List

import numpy as np
import pandas as pd

SLIP = 0.25
COMM = 4.0   # $4 RT per contract
PV   = 20.0

_SIG_T  = dt_time(14, 59)  # reference bar — close = "price at 15:00"
_WIN_E  = dt_time(15, 30)  # latest entry bar for Cond 1 & 2
_EXIT_T = dt_time(15, 59)  # hard exit bar


@dataclass
class ClosingMagnetTrade:
    date:             object
    poc_variant:      str
    condition:        int
    direction:        str
    entry_ts:         pd.Timestamp
    entry_px:         float
    exit_ts:          pd.Timestamp
    exit_px:          float
    exit_reason:      str
    sl_px:            float
    tp_px:            float
    target_poc:       float
    session_poc_15:   float
    prev_poc_15:      float
    nearest_poc_15:   float
    atr_at_entry:     float
    n_contracts:      int
    capital_at_entry: float
    pnl_pts:          float
    pnl_net:          float
    bars_held:        int
    tp_dist:          float
    sl_dist:          float


def _sim_exit(bar_iter, direction, entry_px, sl_px, tp_px):
    """Walk bars until SL, TP, or 15:59 hard exit. SL wins on same-bar conflict."""
    bars_held = 0
    for bar in bar_iter:
        bars_held += 1
        h  = float(bar.high)
        l  = float(bar.low)
        c  = float(bar.close)
        bt = bar.Index.time()
        sl_hit = (l <= sl_px) if direction == "long" else (h >= sl_px)
        tp_hit = (h >= tp_px) if direction == "long" else (l <= tp_px)
        if sl_hit:
            return sl_px, "sl", bar.Index, bars_held
        if tp_hit:
            return tp_px, "tp", bar.Index, bars_held
        if bt >= _EXIT_T:
            xp = (c - SLIP) if direction == "long" else (c + SLIP)
            return xp, "time", bar.Index, bars_held
    return entry_px, "time", pd.NaT, bars_held


def _record(trades, capital_before, d, poc_variant, condition, direction,
            entry_ts, ep, xt, xp, rsn, sl, tp, target, sp15, pp15, np15,
            cur_atr, n_c, bh):
    pnl_pts = (xp - ep) if direction == "long" else (ep - xp)
    pnl_net = pnl_pts * n_c * PV - COMM * n_c
    trades.append(ClosingMagnetTrade(
        date=d, poc_variant=poc_variant, condition=condition,
        direction=direction, entry_ts=entry_ts, entry_px=ep,
        exit_ts=xt, exit_px=xp, exit_reason=rsn,
        sl_px=sl, tp_px=tp, target_poc=target,
        session_poc_15=sp15, prev_poc_15=pp15, nearest_poc_15=np15,
        atr_at_entry=cur_atr, n_contracts=n_c, capital_at_entry=capital_before,
        pnl_pts=pnl_pts, pnl_net=pnl_net, bars_held=bh,
        tp_dist=abs(ep - tp), sl_dist=cur_atr,
    ))
    return pnl_net


def run(
    df: pd.DataFrame,
    poc_variant: str,
    condition: int,
    initial_capital: float = 100_000.0,
    risk_pct: float = 1.0,
) -> List[ClosingMagnetTrade]:
    """
    Run Phase 20 closing magnet simulation.

    df must have columns: session_poc, prev_poc, atr, vwap, high, low, close.
    poc_variant: "A" (session) | "B" (prev) | "C" (nearest)
    condition:   1 (dist+momentum) | 2 (VWAP cross) | 3 (simple displacement)
    """
    trades: List[ClosingMagnetTrade] = []
    capital = float(initial_capital)

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    u_dates  = np.unique(date_arr)

    cl_v   = df["close"].values
    atr_v  = df["atr"].values
    vwap_v = df["vwap"].values
    spoc_v = df["session_poc"].values
    ppoc_v = df["prev_poc"].values

    for d in u_dates:
        mask   = date_arr == d
        d_idx  = np.where(mask)[0]      # absolute positions in df for this date
        d_times = time_arr[d_idx]

        # Locate signal bar (14:59)
        sig_locs = np.where(d_times == _SIG_T)[0]
        if sig_locs.size == 0:
            continue
        sig_abs = int(d_idx[sig_locs[0]])

        sp15    = float(spoc_v[sig_abs])
        pp15    = float(ppoc_v[sig_abs])
        price15 = float(cl_v[sig_abs])
        atr15   = float(atr_v[sig_abs])

        if np.isnan(sp15) or np.isnan(pp15) or np.isnan(atr15) or atr15 <= 0:
            continue

        np15   = sp15 if abs(price15 - sp15) < abs(price15 - pp15) else pp15
        target = sp15 if poc_variant == "A" else (pp15 if poc_variant == "B" else np15)

        post_sig = d_idx[d_idx > sig_abs]   # absolute positions after signal bar

        # ── CONDITION 3: enter at close of 14:59 bar ─────────────────────
        if condition == 3:
            if price15 == target:
                continue
            direction = "long" if price15 < target else "short"
            ep = (price15 + SLIP) if direction == "long" else (price15 - SLIP)
            tp = target
            if (direction == "long" and tp <= ep) or (direction == "short" and tp >= ep):
                continue
            sl   = (ep - atr15) if direction == "long" else (ep + atr15)
            n_c  = max(1, int((capital * risk_pct / 100.0) / (atr15 * PV)))
            cap0 = capital
            bars = list(df.iloc[post_sig].itertuples())
            xp, rsn, xt, bh = _sim_exit(bars, direction, ep, sl, tp)
            pnl = _record(trades, cap0, d, poc_variant, condition, direction,
                          df.index[sig_abs], ep, xt, xp, rsn, sl, tp, target,
                          sp15, pp15, np15, atr15, n_c, bh)
            capital += pnl
            continue

        # ── CONDITIONS 1 & 2: scan 15:00–15:30 window ────────────────────
        below_poc = price15 < target - atr15   # potential long
        above_poc = price15 > target + atr15   # potential short

        vw15   = float(vwap_v[sig_abs])
        prev_cl = price15
        prev_vw = vw15

        win_abs = d_idx[(d_times >= dt_time(15, 0)) & (d_times <= _WIN_E)]

        for pos in win_abs:
            cur_cl   = float(cl_v[pos])
            cur_atr  = float(atr_v[pos])
            cur_vwap = float(vwap_v[pos])

            if np.isnan(cur_atr) or cur_atr <= 0:
                prev_cl = cur_cl; prev_vw = cur_vwap
                continue

            direction = None

            if condition == 1:
                if below_poc and cur_cl > prev_cl:
                    direction = "long"
                elif above_poc and cur_cl < prev_cl:
                    direction = "short"

            else:  # condition 2: VWAP cross toward POC
                if not (np.isnan(prev_vw) or np.isnan(cur_vwap)):
                    if prev_cl < prev_vw and cur_cl >= cur_vwap and target > cur_cl:
                        direction = "long"
                    elif prev_cl > prev_vw and cur_cl <= cur_vwap and target < cur_cl:
                        direction = "short"

            if direction is None:
                prev_cl = cur_cl; prev_vw = cur_vwap
                continue

            ep = (cur_cl + SLIP) if direction == "long" else (cur_cl - SLIP)
            tp = target
            if (direction == "long" and tp <= ep) or (direction == "short" and tp >= ep):
                prev_cl = cur_cl; prev_vw = cur_vwap
                continue

            sl   = (ep - cur_atr) if direction == "long" else (ep + cur_atr)
            n_c  = max(1, int((capital * risk_pct / 100.0) / (cur_atr * PV)))
            cap0 = capital
            post_entry = d_idx[d_idx > pos]
            bars = list(df.iloc[post_entry].itertuples())
            xp, rsn, xt, bh = _sim_exit(bars, direction, ep, sl, tp)
            pnl = _record(trades, cap0, d, poc_variant, condition, direction,
                          df.index[pos], ep, xt, xp, rsn, sl, tp, target,
                          sp15, pp15, np15, cur_atr, n_c, bh)
            capital += pnl
            break  # max 1 trade per session

    return trades
