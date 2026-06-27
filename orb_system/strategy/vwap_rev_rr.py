"""
Phase 10: VWAP Reversion with R/R-derived SL.

Entry: identical to Phase 8 Exp 1
  - HMM state == "ranging"
  - close < VWAP - deviation_mult × ATR  (LONG)  /  >  (SHORT)
  - candle range > candle_mult × ATR (bullish/bearish rejection)
  - close in upper 30% of range (LONG)  /  lower 30% (SHORT)
  - volume > volume_mult × avg_vol
  - time 10:00–14:30, max 1 long + 1 short per session (sequential)

SL: derived from TP distance so trade has a target R/R ratio
  tp_distance = |vwap_at_entry - entry_price|
  sl_distance = tp_distance / target_rr
  sl_price    = entry_price - sl_distance (LONG)
               entry_price + sl_distance (SHORT)

TP: VWAP at time of entry (fixed, does not move).
Exit: SL/TP via H/L; SL wins same-bar; EOD 15:45; timeout max_bars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time as dt_time
from typing import List

import numpy as np
import pandas as pd

SLIP = 0.25     # 1 tick entry slippage (points)
COMM = 4.0      # round-trip commission ($2/side × 2)
PV   = 20.0     # NQ point value ($/point)

_EOD_T    = dt_time(15, 45)
_UPPER_P  = 0.70   # close ≥ low + 0.70×range  → bullish rejection (LONG)
_LOWER_P  = 0.30   # close ≤ low + 0.30×range  → bearish rejection (SHORT)

RR_LEVELS = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]


def _parse_time(s: str) -> dt_time:
    h, m = s.split(":")
    return dt_time(int(h), int(m))


# ── Trade + Results ────────────────────────────────────────────────────────────

@dataclass
class RevRRTrade:
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    direction:   str    # "long" | "short"
    entry_px:    float
    exit_px:     float
    sl_px:       float
    tp_px:       float
    exit_reason: str    # "sl" | "tp" | "eod" | "timeout"
    pnl_pts:     float
    pnl_net:     float
    atr_entry:   float
    vwap_entry:  float
    tp_dist:     float
    sl_dist:     float
    target_rr:   float
    bars_held:   int


class RevRRResults:
    def __init__(self, trades: List[RevRRTrade], label: str = ""):
        self.trades = trades
        self.label  = label

    def metrics(self) -> dict:
        if not self.trades:
            return {"n": 0, "wr": 0.0, "pf": 0.0, "sharpe": 0.0,
                    "ret_pct": 0.0, "max_dd_pct": 0.0, "avg_win": 0.0,
                    "avg_loss": 0.0, "trades_per_day": 0.0}
        pnls   = np.array([t.pnl_net for t in self.trades])
        wins   = pnls[pnls > 0];  losses = pnls[pnls <= 0]
        n      = len(pnls)
        wr     = float(len(wins)) / n
        gw     = float(wins.sum())  if len(wins)   else 0.0
        gl     = float(abs(losses.sum())) if len(losses) else 1e-9
        pf     = gw / max(gl, 1e-9)
        cap    = 100_000.0
        eq     = np.concatenate([[cap], cap + np.cumsum(pnls)])
        ret    = float((eq[-1] - cap) / cap * 100)
        peak   = np.maximum.accumulate(eq)
        dd     = (eq - peak) / peak * 100
        max_dd = float(abs(dd.min()))
        daily  = {}
        for t in self.trades:
            d = t.entry_ts.date();  daily[d] = daily.get(d, 0.0) + t.pnl_net
        dp   = np.array(list(daily.values()))
        sr   = float(dp.mean() / (dp.std(ddof=1) + 1e-9) * np.sqrt(252)) if len(dp) >= 2 else 0.0
        tpd  = n / max(len(daily), 1)
        return {"n": n, "wr": wr, "pf": pf, "sharpe": sr, "ret_pct": ret,
                "max_dd_pct": max_dd,
                "avg_win":  float(wins.mean())   if len(wins)   else 0.0,
                "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                "trades_per_day": tpd}

    def annual_breakdown(self) -> dict:
        by_yr: dict = {}
        for t in self.trades:
            by_yr.setdefault(t.entry_ts.year, []).append(t.pnl_net)
        out = {}
        for yr, plist in sorted(by_yr.items()):
            pnls = np.array(plist)
            wins = pnls[pnls > 0];  losses = pnls[pnls <= 0]
            gw = float(wins.sum()) if len(wins) else 0.0
            gl = float(abs(losses.sum())) if len(losses) else 1e-9
            cap = 100_000.0;  eq = np.concatenate([[cap], cap + np.cumsum(pnls)])
            dp = {};
            for t in (t for t in self.trades if t.entry_ts.year == yr):
                d = t.entry_ts.date();  dp[d] = dp.get(d, 0.0) + t.pnl_net
            dpa = np.array(list(dp.values()))
            sr = float(dpa.mean() / (dpa.std(ddof=1) + 1e-9) * np.sqrt(252)) if len(dpa) >= 2 else 0.0
            out[yr] = {"n": len(pnls), "wr": float(len(wins)) / len(pnls),
                       "pf": gw / max(gl, 1e-9),
                       "ret_pct": float((eq[-1] - cap) / cap * 100), "sharpe": sr}
        return out

    def exit_breakdown(self) -> dict:
        n = max(len(self.trades), 1)
        c = {"sl": 0, "tp": 0, "eod": 0, "timeout": 0}
        for t in self.trades:
            if t.exit_reason in c:
                c[t.exit_reason] += 1
        return {k: v / n for k, v in c.items()}

    def to_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "entry_ts": t.entry_ts, "exit_ts": t.exit_ts,
            "direction": t.direction, "entry_px": t.entry_px,
            "exit_px": t.exit_px, "sl_px": t.sl_px, "tp_px": t.tp_px,
            "exit_reason": t.exit_reason, "pnl_pts": t.pnl_pts,
            "pnl_net": t.pnl_net, "atr_entry": t.atr_entry,
            "vwap_entry": t.vwap_entry, "tp_dist": t.tp_dist,
            "sl_dist": t.sl_dist, "target_rr": t.target_rr,
            "bars_held": t.bars_held,
        } for t in self.trades])


# ── Exit simulation ────────────────────────────────────────────────────────────

def _simulate_exit(bars_after: list, entry_date, entry_px: float,
                   sl_px: float, tp_px: float, direction: str,
                   tp_dist: float, sl_dist: float, target_rr: float,
                   atr_entry: float, vwap_entry: float,
                   entry_ts: pd.Timestamp, max_bars: int) -> RevRRTrade:
    """
    Simulate exit for one trade. bars_after is a list of itertuples rows
    starting at the bar AFTER entry.
    SL wins when both SL and TP are touched on the same bar.
    """
    bars_held = 0
    for bar in bars_after:
        ts = bar.Index
        if ts.date() != entry_date:
            break
        if bars_held >= max_bars:
            break
        bars_held += 1
        h, l, c = float(bar.high), float(bar.low), float(bar.close)
        t = ts.time()

        if direction == "long":
            sl_hit = l <= sl_px
            tp_hit = h >= tp_px
        else:
            sl_hit = h >= sl_px
            tp_hit = l <= tp_px

        if sl_hit:
            exit_px = sl_px;  reason = "sl"
        elif tp_hit:
            exit_px = tp_px;  reason = "tp"
        elif t >= _EOD_T:
            exit_px = c - SLIP if direction == "long" else c + SLIP
            reason  = "eod"
        else:
            continue

        pnl_pts = (exit_px - entry_px) if direction == "long" else (entry_px - exit_px)
        return RevRRTrade(
            entry_ts=entry_ts, exit_ts=ts, direction=direction,
            entry_px=entry_px, exit_px=exit_px, sl_px=sl_px, tp_px=tp_px,
            exit_reason=reason, pnl_pts=pnl_pts, pnl_net=pnl_pts * PV - COMM,
            atr_entry=atr_entry, vwap_entry=vwap_entry,
            tp_dist=tp_dist, sl_dist=sl_dist, target_rr=target_rr,
            bars_held=bars_held,
        )

    # Timeout (or no bars / crossed session boundary)
    if bars_after:
        last = bars_after[min(bars_held, len(bars_after) - 1)]
        c    = float(last.close)
        ts   = last.Index
    else:
        c, ts = entry_px, entry_ts
    exit_px = c - SLIP if direction == "long" else c + SLIP
    pnl_pts = (exit_px - entry_px) if direction == "long" else (entry_px - exit_px)
    return RevRRTrade(
        entry_ts=entry_ts, exit_ts=ts, direction=direction,
        entry_px=entry_px, exit_px=exit_px, sl_px=sl_px, tp_px=tp_px,
        exit_reason="timeout", pnl_pts=pnl_pts, pnl_net=pnl_pts * PV - COMM,
        atr_entry=atr_entry, vwap_entry=vwap_entry,
        tp_dist=tp_dist, sl_dist=sl_dist, target_rr=target_rr,
        bars_held=bars_held,
    )


# ── Main engine ────────────────────────────────────────────────────────────────

class VWAPRevRREngine:
    """VWAP Reversion with R/R-derived SL (Phase 10)."""

    @staticmethod
    def run(
        df:            pd.DataFrame,
        regime_map:    dict,           # date -> "ranging"|"trending"|"volatile"
        target_rr:     float,
        *,
        deviation_mult: float = 1.5,
        candle_mult:    float = 0.8,
        volume_mult:    float = 1.2,
        max_bars:       int   = 90,
        time_start:     str   = "10:00",
        time_end:       str   = "14:30",
        label:          str   = "",
    ) -> RevRRResults:
        t_s = _parse_time(time_start)
        t_e = _parse_time(time_end)

        date_arr = np.array(df.index.date)
        u_dates, first_idx, counts = np.unique(
            date_arr, return_index=True, return_counts=True
        )
        trades: List[RevRRTrade] = []

        for ui in range(len(u_dates)):
            d = u_dates[ui]
            if regime_map.get(d, "") != "ranging":
                continue
            seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]
            day_trades = VWAPRevRREngine._sim_session(
                seg, d, target_rr, deviation_mult, candle_mult,
                volume_mult, max_bars, t_s, t_e
            )
            trades.extend(day_trades)

        return RevRRResults(trades, label)

    @staticmethod
    def _sim_session(seg, date, target_rr, dev_mult, candle_mult,
                     vol_mult, max_bars, t_start, t_end):
        trades: List[RevRRTrade] = []
        long_taken = short_taken = in_position = False

        bars = list(seg.itertuples())
        n    = len(bars)
        i    = 0

        while i < n:
            bar = bars[i]
            bt  = bar.Index.time()

            if not in_position and t_start <= bt < t_end:
                atr_v  = float(bar.atr)
                vwap_v = float(bar.vwap)
                avgv_v = float(bar.avg_vol)
                cl     = float(bar.close)
                hi     = float(bar.high)
                lo     = float(bar.low)
                crng   = hi - lo

                if atr_v <= 0 or avgv_v <= 0 or crng <= 0:
                    i += 1; continue

                vol_ok  = float(bar.volume) > vol_mult * avgv_v
                rng_ok  = crng > candle_mult * atr_v

                long_sig  = (vol_ok and rng_ok and not long_taken
                             and cl < vwap_v - dev_mult * atr_v
                             and cl >= lo + _UPPER_P * crng)
                short_sig = (vol_ok and rng_ok and not short_taken
                             and cl > vwap_v + dev_mult * atr_v
                             and cl <= lo + _LOWER_P * crng)

                entered_long  = long_sig
                entered_short = short_sig and not long_sig

                if entered_long or entered_short:
                    dirn = "long" if entered_long else "short"

                    if dirn == "long":
                        entry_px = cl + SLIP
                        tp_px    = vwap_v
                        tp_dist  = tp_px - entry_px
                    else:
                        entry_px = cl - SLIP
                        tp_px    = vwap_v
                        tp_dist  = entry_px - tp_px

                    if tp_dist <= 0:
                        i += 1; continue

                    sl_dist = tp_dist / target_rr
                    sl_px   = (entry_px - sl_dist) if dirn == "long" else (entry_px + sl_dist)

                    in_position = True
                    if dirn == "long":  long_taken  = True
                    else:               short_taken = True

                    post  = bars[i + 1:]
                    trade = _simulate_exit(
                        post, date, entry_px, sl_px, tp_px, dirn,
                        tp_dist, sl_dist, target_rr, atr_v, vwap_v,
                        bar.Index, max_bars
                    )
                    trades.append(trade)
                    in_position = False

                    # Fast-forward past exit bar
                    try:
                        exit_pos = next(
                            j for j, b in enumerate(post)
                            if b.Index >= trade.exit_ts
                        )
                        i += exit_pos + 2
                    except StopIteration:
                        i = n
                    continue

            if long_taken and short_taken:
                break
            i += 1

        return trades


# ── Pre-P&L diagnostic ─────────────────────────────────────────────────────────

def run_diagnostic_p10(
    df:            pd.DataFrame,
    regime_map:    dict,
    *,
    deviation_mult: float = 1.5,
    candle_mult:    float = 0.8,
    volume_mult:    float = 1.2,
    max_bars:       int   = 90,
    time_start:     str   = "10:00",
    time_end:       str   = "14:30",
) -> bool:
    """
    Pre-P&L diagnostic. For each signal (Phase 8 Exp 1 entry conditions on
    ranging HMM days), simulate outcome for ALL R/R levels simultaneously.
    Prints:
      1. TP distance distribution (ATR units)
      2. Theoretical WR per R/R level
      3. Theoretical expectancy table
    Returns True if at least one R/R level has positive theoretical expectancy.
    """
    t_s = _parse_time(time_start)
    t_e = _parse_time(time_end)

    date_arr = np.array(df.index.date)
    u_dates, first_idx, counts = np.unique(
        date_arr, return_index=True, return_counts=True
    )

    # Accumulate: tp distances, outcomes per R/R level
    tp_dists_atr = []
    rr_counts: dict = {rr: {"tp": 0, "ntp": 0} for rr in RR_LEVELS}
    total_signals = 0

    for ui in range(len(u_dates)):
        d = u_dates[ui]
        if regime_map.get(d, "") != "ranging":
            continue
        seg  = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]
        bars = list(seg.itertuples())
        n    = len(bars)

        long_done = short_done = False

        for i, bar in enumerate(bars):
            bt = bar.Index.time()
            if bt < t_s or bt >= t_e:
                continue
            if long_done and short_done:
                break

            atr_v  = float(bar.atr)
            vwap_v = float(bar.vwap)
            avgv_v = float(bar.avg_vol)
            cl     = float(bar.close)
            hi     = float(bar.high)
            lo     = float(bar.low)
            crng   = hi - lo

            if atr_v <= 0 or avgv_v <= 0 or crng <= 0:
                continue

            vol_ok = float(bar.volume) > volume_mult * avgv_v
            rng_ok = crng > candle_mult * atr_v

            for dirn, sig_cond, done_flag in [
                ("long",
                 vol_ok and rng_ok and not long_done
                 and cl < vwap_v - deviation_mult * atr_v
                 and cl >= lo + _UPPER_P * crng,
                 "long_done"),
                ("short",
                 vol_ok and rng_ok and not short_done
                 and cl > vwap_v + deviation_mult * atr_v
                 and cl <= lo + _LOWER_P * crng,
                 "short_done"),
            ]:
                if not sig_cond:
                    continue

                entry_px = (cl + SLIP) if dirn == "long" else (cl - SLIP)
                tp_px    = vwap_v
                tp_dist  = (tp_px - entry_px) if dirn == "long" else (entry_px - tp_px)

                if tp_dist <= 0:
                    continue

                if dirn == "long":  long_done  = True
                else:               short_done = True
                total_signals += 1
                tp_dists_atr.append(tp_dist / atr_v)

                # Simulate all R/R levels on post-entry bars
                post_bars = bars[i + 1:]
                pending   = set(RR_LEVELS)
                outcomes  = {}
                sl_prices = {
                    rr: (entry_px - tp_dist / rr) if dirn == "long"
                        else (entry_px + tp_dist / rr)
                    for rr in RR_LEVELS
                }

                for bar2 in post_bars:
                    ts2 = bar2.Index
                    if ts2.date() != d:
                        break
                    if not pending:
                        break
                    h2, l2 = float(bar2.high), float(bar2.low)
                    t2 = ts2.time()

                    for rr in list(pending):
                        sp = sl_prices[rr]
                        if dirn == "long":
                            sl_hit = l2 <= sp
                            tp_hit = h2 >= tp_px
                        else:
                            sl_hit = h2 >= sp
                            tp_hit = l2 <= tp_px

                        if sl_hit:
                            outcomes[rr] = "ntp"; pending.discard(rr)
                        elif tp_hit:
                            outcomes[rr] = "tp";  pending.discard(rr)
                        elif t2 >= _EOD_T:
                            outcomes[rr] = "ntp"; pending.discard(rr)

                for rr in pending:
                    outcomes[rr] = "ntp"

                for rr, oc in outcomes.items():
                    rr_counts[rr][oc] += 1

    # ── Print diagnostic ─────────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print("  PHASE 10 -- PRE-P&L DIAGNOSTIC")
    print("=" * W)
    print(f"  Ranging HMM sessions with qualifying signals: total={total_signals}")

    if total_signals == 0:
        print("  WARNING: zero signals found. Check HMM state labeling.")
        print("=" * W)
        return False

    arr = np.array(tp_dists_atr)
    print(f"\n  1. TP DISTANCE (ATR units)  [n={len(arr)} signals]")
    print(f"     p10={np.percentile(arr,10):.2f}  p25={np.percentile(arr,25):.2f}  "
          f"p50={np.percentile(arr,50):.2f}  p75={np.percentile(arr,75):.2f}  "
          f"p90={np.percentile(arr,90):.2f}")
    print(f"     Median SL at R/R=5.0: {np.percentile(arr,50)/5.0:.2f} ATR from entry")
    print(f"     Median SL at R/R=2.0: {np.percentile(arr,50)/2.0:.2f} ATR from entry")
    print(f"     Median SL at R/R=1.0: {np.percentile(arr,50)/1.0:.2f} ATR from entry")

    print(f"\n  2-3. THEORETICAL WIN RATE & EXPECTANCY")
    print(f"  {'R/R':>5}  {'WR%':>7}  {'Exp':>7}  {'Min WR%':>8}  {'Status':>12}")
    print("  " + "-" * 50)

    viable = False
    for rr in RR_LEVELS:
        tp_n  = rr_counts[rr]["tp"]
        tot   = tp_n + rr_counts[rr]["ntp"]
        wr    = tp_n / max(tot, 1)
        exp   = wr * rr - (1.0 - wr)
        min_wr = 1.0 / (rr + 1.0)
        status = "VIABLE" if exp > 0 else ("edge" if exp > -0.05 else "---")
        if exp > 0:
            viable = True
        print(f"  {rr:>5.1f}  {wr*100:>6.1f}%  {exp:>+7.3f}  "
              f"{min_wr*100:>7.1f}%   {status}")

    print("=" * W)
    if not viable:
        print("  RESULT: No R/R level produces positive theoretical expectancy.")
        print("  The 34% natural reversion rate is insufficient at any SL width.")
        print("  Recommend abandoning this hypothesis.")
    else:
        pos = [rr for rr in RR_LEVELS
               if rr_counts[rr]["tp"] / max(rr_counts[rr]["tp"] + rr_counts[rr]["ntp"], 1) * rr
                  - (1 - rr_counts[rr]["tp"] / max(rr_counts[rr]["tp"] + rr_counts[rr]["ntp"], 1)) > 0]
        print(f"  RESULT: Positive theoretical expectancy at R/R = {pos}")
        print("  Proceeding to full backtests.")
    print("=" * W)

    return viable
