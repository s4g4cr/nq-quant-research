"""
Phase 12 POC Mean Reversion engine.

Extends Phase 11 with:
  - Fractional TP: tp_px = entry + tp_frac * full_poc_distance
  - Optional HMM regime filter via regime_map

Entry signal conditions are unchanged from Phase 11.
"""

from typing import List

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


class POCRevFractEngine:

    @staticmethod
    def run(
        df: pd.DataFrame,
        *,
        tp_frac:         float = 1.0,
        regime_map:      dict  = None,
        allowed_regimes: tuple = ("ranging",),
        deviation_mult:  float = 1.0,
        sl_mult:         float = 1.0,
        volume_mult:     float = 1.3,
        exhaustion_mult: float = 1.2,
        max_bars:        int   = 120,
        time_start:      str   = "09:45",
        time_end:        str   = "14:30",
        label:           str   = "",
    ) -> POCResults:
        """
        tp_frac=1.0 reproduces Phase 11 Exp 6 baseline.
        regime_map=None trades all days regardless of regime.
        """
        t_s = _parse_time(time_start)
        t_e = _parse_time(time_end)

        date_arr = np.array(df.index.date)
        u_dates, first_idx, counts = np.unique(
            date_arr, return_index=True, return_counts=True
        )
        trades: List[POCTrade] = []

        for ui in range(len(u_dates)):
            d   = u_dates[ui]
            seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]

            if regime_map is not None:
                if regime_map.get(d, "other") not in allowed_regimes:
                    continue

            trades.extend(POCRevFractEngine._sim_session(
                seg, tp_frac, deviation_mult, sl_mult,
                volume_mult, exhaustion_mult, max_bars, t_s, t_e
            ))

        return POCResults(trades, label)

    @staticmethod
    def _sim_session(seg, tp_frac, dev_mult, sl_mult, vol_mult,
                     exh_mult, max_bars, t_s, t_e):
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

            long_sig  = (
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

            sp_v = float(bar.session_poc)
            conf = bool(bar.poc_confluence)

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
                session_poc= sp_v,
                confluence = conf,
                tp_variant = f"frac{tp_frac:.2f}",
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
