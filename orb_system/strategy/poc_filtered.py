"""
Phase 14 POC Reversion engine with deterministic regime filters.

Three optional filters (all None = no filter, reproduces Phase 12 Exp A):
  filter1_thresh : prev_range_ratio < thresh  (day-level, OR_RANGE_FILTER)
  filter2_thresh : poc_distance >= thresh     (bar-level, DISTANCE_FILTER)
  filter3_thresh : abs(trend_5d) < thresh     (day-level, TREND_FILTER)

Entry signal and TP/SL logic unchanged from Phase 12.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from orb_system.strategy.poc_reversion import (
    POCTrade,
    POCResults,
    _exit,
    _parse_time,
    SLIP,
    COMM,
    PV,
    _UP_PCT,
    _LO_PCT,
)


class POCFilteredEngine:

    @staticmethod
    def run(
        df: pd.DataFrame,
        *,
        tp_frac:          float          = 0.67,
        filter1_thresh:   Optional[float] = None,
        filter2_thresh:   Optional[float] = None,
        filter3_thresh:   Optional[float] = None,
        deviation_mult:   float          = 1.0,
        sl_mult:          float          = 1.0,
        volume_mult:      float          = 1.3,
        exhaustion_mult:  float          = 1.2,
        max_bars:         int            = 120,
        time_start:       str            = "09:45",
        time_end:         str            = "14:30",
        label:            str            = "",
    ) -> POCResults:
        t_s = _parse_time(time_start)
        t_e = _parse_time(time_end)

        date_arr = np.array(df.index.date)
        u_dates, first_idx, counts = np.unique(
            date_arr, return_index=True, return_counts=True
        )
        trades: List[POCTrade] = []

        for ui in range(len(u_dates)):
            seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]

            # Filter 1: previous session range narrow enough
            if filter1_thresh is not None:
                prr = float(seg["prev_range_ratio"].iloc[0])
                if np.isnan(prr) or prr >= filter1_thresh:
                    continue

            # Filter 3: recent 5-day trend weak enough
            if filter3_thresh is not None:
                t5d = float(seg["trend_5d"].iloc[0])
                if np.isnan(t5d) or abs(t5d) >= filter3_thresh:
                    continue

            trades.extend(POCFilteredEngine._sim_session(
                seg, tp_frac, deviation_mult, sl_mult,
                volume_mult, exhaustion_mult, max_bars,
                t_s, t_e, filter2_thresh,
            ))

        return POCResults(trades, label)

    @staticmethod
    def _sim_session(seg, tp_frac, dev_mult, sl_mult, vol_mult, exh_mult,
                     max_bars, t_s, t_e, f2_thresh):
        trades: List[POCTrade] = []
        long_taken  = False
        short_taken = False
        bars        = list(seg.itertuples())
        n           = len(bars)
        i           = 0

        while i < n:
            bar = bars[i]
            bt  = bar.Index.time()

            if not (t_s <= bt <= t_e) or (long_taken and short_taken):
                i += 1
                continue

            atr_v  = float(bar.atr)
            avg_v  = float(bar.avg_vol)
            tp_poc = float(bar.target_poc)
            pp_v   = float(bar.prev_poc)

            if any(np.isnan([atr_v, avg_v, tp_poc, pp_v])) or atr_v <= 0 or avg_v <= 0:
                i += 1
                continue

            cl   = float(bar.close)
            hi   = float(bar.high)
            lo   = float(bar.low)
            op   = float(bar.open)
            crng = hi - lo

            long_sig = (
                not long_taken
                and cl < tp_poc - dev_mult * atr_v
                and crng > exh_mult * atr_v
                and cl > op
                and cl > lo + _UP_PCT * crng
                and float(bar.volume) > vol_mult * avg_v
            )
            short_sig = (
                not short_taken
                and cl > tp_poc + dev_mult * atr_v
                and crng > exh_mult * atr_v
                and cl < op
                and cl < lo + _LO_PCT * crng
                and float(bar.volume) > vol_mult * avg_v
            )

            if not (long_sig or short_sig):
                i += 1
                continue

            # Filter 2: bar-level distance filter (uses close, before slippage)
            if f2_thresh is not None:
                poc_dist = abs(cl - tp_poc) / atr_v
                if poc_dist < f2_thresh:
                    i += 1
                    continue

            dirn      = "long" if long_sig else "short"
            entry_px  = (cl + SLIP) if dirn == "long" else (cl - SLIP)
            full_dist = (tp_poc - entry_px) if dirn == "long" else (entry_px - tp_poc)

            if full_dist <= SLIP:
                i += 1
                continue

            tp_dist = tp_frac * full_dist
            tp_px   = (entry_px + tp_dist) if dirn == "long" else (entry_px - tp_dist)
            sl_dist = sl_mult * atr_v
            sl_px   = (entry_px - sl_dist) if dirn == "long" else (entry_px + sl_dist)

            if dirn == "long":
                long_taken = True
            else:
                short_taken = True

            post = bars[i + 1:]
            exit_px, reason, exit_ts, bars_held = _exit(
                post, dirn, entry_px, sl_px, tp_px, max_bars
            )

            pnl_pts = (exit_px - entry_px) if dirn == "long" else (entry_px - exit_px)
            pnl_net = pnl_pts * PV - COMM

            trades.append(POCTrade(
                entry_ts   = bar.Index,
                exit_ts    = exit_ts,
                direction  = dirn,
                entry_px   = entry_px,
                exit_px    = exit_px,
                sl_px      = sl_px,
                tp_px      = tp_px,
                exit_reason= reason,
                pnl_pts    = pnl_pts,
                pnl_net    = pnl_net,
                atr_entry  = atr_v,
                target_poc = tp_poc,
                prev_poc   = pp_v,
                session_poc= float(bar.session_poc),
                confluence = bool(bar.poc_confluence),
                tp_variant = f"frac{tp_frac:.2f}_f2{f2_thresh or 0:.1f}",
                tp_dist    = tp_dist,
                sl_dist    = sl_dist,
                bars_held  = bars_held,
            ))

            try:
                exit_pos = next(j for j, b in enumerate(post) if b.Index >= exit_ts)
                i += exit_pos + 2
            except StopIteration:
                break

        return trades
