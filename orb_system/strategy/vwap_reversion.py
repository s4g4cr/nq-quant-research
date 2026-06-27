"""
VWAP Reversion + Breakout strategy engine for NQ 1-min.

Two complementary modes driven by HMM regime:
  reversion — ranging days: fade deviation from VWAP back toward VWAP
  breakout  — trending days: trade VWAP break continuation

Key guarantees:
  - All signals strictly causal (uses only past + current bar data)
  - SL/TP checked on High/Low; SL wins when same bar hits both
  - Max 1 long + 1 short per session (sequential, no pyramiding)
  - EOD exit at 15:45, timeout at max_bars
"""

import math
from dataclasses import dataclass
from datetime import time as dt_time
from typing import List, Optional

import numpy as np
import pandas as pd

SLIP = 0.25    # 1 tick slippage (entry only)
COMM = 4.0     # commission round-trip per contract ($2/side)
PV   = 20.0    # NQ point value $/pt

_EOD_T   = dt_time(15, 45)
_OR_END  = dt_time(10, 0)   # earliest allowed entry

_UPPER_PCT = 0.70   # close must be in top 30% of range (bullish rejection)
_LOWER_PCT = 0.30   # close must be in bottom 30% of range (bearish rejection)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class VWAPTrade:
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    direction:   str        # "long" | "short"
    entry_px:    float
    exit_px:     float
    sl_px:       float
    tp_px:       float
    exit_reason: str        # "sl" | "tp" | "eod" | "timeout"
    pnl_pts:     float      # positive = profit
    pnl_net:     float      # pnl_pts * PV - COMM
    atr_entry:   float
    vwap_entry:  float
    dev_atr:     float      # |close - VWAP| / ATR at entry
    rr:          float      # TP distance / SL distance (geometric)
    strategy:    str        # "reversion" | "breakout"


# ── Results container ──────────────────────────────────────────────────────────

class VWAPResults:
    def __init__(self, trades: List[VWAPTrade], label: str = ""):
        self.trades = trades
        self.label  = label

    def metrics(self, capital: float = 100_000.0) -> dict:
        if not self.trades:
            return {"n": 0, "wr": 0.0, "pf": float("nan"), "sharpe": 0.0,
                    "ret_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                    "max_dd": 0.0, "trades_per_day": 0.0}
        net   = np.array([t.pnl_net for t in self.trades])
        wins  = net[net > 0];  losses = net[net <= 0]
        gross_w = float(wins.sum())  if wins.size  else 0.0
        gross_l = float(abs(losses.sum())) if losses.size else 0.0
        pf      = gross_w / gross_l if gross_l > 0 else float("inf")
        wr      = float(wins.size) / len(net)
        std     = float(net.std())
        sr      = float(net.mean() / std * math.sqrt(252)) if std > 0 else 0.0
        ret     = float(net.sum()) / capital * 100.0

        # Max drawdown (from peak capital)
        curve = np.concatenate([[0.0], np.cumsum(net)])
        peak  = np.maximum.accumulate(curve)
        dd    = (peak - curve) / capital * 100.0
        max_dd = float(dd.max())

        # Trades per day
        dates = {t.entry_ts.date() for t in self.trades}
        tpd   = len(self.trades) / max(len(dates), 1)

        return {
            "n": len(net), "wr": wr, "pf": pf, "sharpe": sr,
            "ret_pct": ret, "avg_win": float(wins.mean()) if wins.size else 0.0,
            "avg_loss": float(losses.mean()) if losses.size else 0.0,
            "max_dd": max_dd, "trades_per_day": tpd,
        }

    def annual_breakdown(self) -> dict:
        by_year = {}
        for t in self.trades:
            y = t.entry_ts.year
            by_year.setdefault(y, []).append(t.pnl_net)
        return {yr: _year_metrics(pnls) for yr, pnls in sorted(by_year.items())}

    def exit_breakdown(self) -> dict:
        reasons = [t.exit_reason for t in self.trades]
        total = max(len(reasons), 1)
        from collections import Counter
        cnt = Counter(reasons)
        return {r: cnt[r] / total for r in ["sl", "tp", "eod", "timeout"]}

    def to_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "entry_ts": t.entry_ts, "exit_ts": t.exit_ts,
            "direction": t.direction, "entry_px": t.entry_px,
            "exit_px": t.exit_px, "sl_px": t.sl_px, "tp_px": t.tp_px,
            "exit_reason": t.exit_reason, "pnl_pts": t.pnl_pts,
            "pnl_net": t.pnl_net, "atr_entry": t.atr_entry,
            "vwap_entry": t.vwap_entry, "dev_atr": t.dev_atr,
            "rr": t.rr, "strategy": t.strategy,
        } for t in self.trades])


def _year_metrics(pnls: list) -> dict:
    net = np.array(pnls)
    w   = net[net > 0]; l = net[net <= 0]
    gw  = float(w.sum())  if w.size  else 0.0
    gl  = float(abs(l.sum())) if l.size else 0.0
    pf  = gw / gl if gl > 0 else float("inf")
    std = float(net.std())
    sr  = float(net.mean() / std * math.sqrt(252)) if std > 0 else 0.0
    return {
        "n": len(net), "wr": float(w.size)/len(net) if len(net) else 0.0,
        "pf": pf, "sharpe": sr,
        "ret_pts": float(net.sum() / PV),
        "ret_usd": float(net.sum()),
    }


# ── Main engine ───────────────────────────────────────────────────────────────

class VWAPReversionEngine:
    """
    Simulate VWAP reversion and/or breakout trades on NQ 1-min data.

    Call run_reversion() or run_breakout() directly; the execution
    script combines them for Experiment 5.
    """

    @classmethod
    def run_reversion(
        cls,
        df: pd.DataFrame,
        regime_map: dict,        # date -> "ranging"|"trending"|"volatile"
        *,
        allowed_regimes: tuple   = ("ranging",),
        deviation_mult: float    = 1.5,
        candle_mult:    float    = 0.8,
        volume_mult:    float    = 1.2,
        sl_mult:        float    = 0.5,
        max_bars:       int      = 90,
        time_start:     str      = "10:00",
        time_end:       str      = "14:30",
        label:          str      = "",
    ) -> VWAPResults:
        """
        VWAP Reversion: fade price extended from VWAP back toward VWAP.
        TP = VWAP at entry (fixed).  SL = entry ± sl_mult × ATR.
        """
        t_s = _parse_time(time_start)
        t_e = _parse_time(time_end)
        trades = cls._run(df, regime_map, allowed_regimes, "reversion",
                          deviation_mult, candle_mult, volume_mult,
                          sl_mult, max_bars, t_s, t_e)
        return VWAPResults(trades, label)

    @classmethod
    def run_breakout(
        cls,
        df: pd.DataFrame,
        regime_map: dict,
        *,
        allowed_regimes: tuple   = ("trending",),
        breakout_mult:  float    = 0.5,   # entry beyond VWAP ± N × ATR
        candle_mult:    float    = 0.8,
        volume_mult:    float    = 1.2,
        tp_atr_mult:    float    = 2.0,   # TP = VWAP ± tp_atr_mult × ATR
        max_bars:       int      = 90,
        time_start:     str      = "10:00",
        time_end:       str      = "14:30",
        label:          str      = "",
    ) -> VWAPResults:
        """
        VWAP Breakout: ride continuation away from VWAP.
        TP = VWAP ± tp_atr_mult × ATR.  SL = VWAP at entry.
        """
        t_s = _parse_time(time_start)
        t_e = _parse_time(time_end)
        trades = cls._run(df, regime_map, allowed_regimes, "breakout",
                          breakout_mult, candle_mult, volume_mult,
                          tp_atr_mult, max_bars, t_s, t_e)
        return VWAPResults(trades, label)

    # ── Core simulation ───────────────────────────────────────────────────────

    @classmethod
    def _run(cls, df, regime_map, allowed_regimes, mode,
             dev_or_bo_mult, candle_mult, volume_mult, sl_or_tp_mult,
             max_bars, t_start, t_end):

        date_arr = np.array(df.index.date)
        u_dates, first_idx, counts = np.unique(
            date_arr, return_index=True, return_counts=True
        )
        trades: List[VWAPTrade] = []

        for ui in range(len(u_dates)):
            d   = u_dates[ui]
            seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]

            regime = regime_map.get(d, "volatile")
            if regime not in allowed_regimes:
                continue

            day_trades = cls._sim_session(
                seg, d, mode,
                dev_or_bo_mult, candle_mult, volume_mult, sl_or_tp_mult,
                max_bars, t_start, t_end
            )
            trades.extend(day_trades)

        return trades

    @classmethod
    def _sim_session(cls, seg, date, mode,
                     dev_mult, candle_mult, vol_mult, sl_or_tp_mult,
                     max_bars, t_start, t_end):

        trades: List[VWAPTrade] = []
        long_taken  = False
        short_taken = False
        in_position = False

        bars = list(seg.itertuples())
        n    = len(bars)

        i = 0
        while i < n:
            bar = bars[i]
            bt  = bar.Index.time()

            # ── Entry scan ────────────────────────────────────────────────────
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

                # Volume filter (common to both modes)
                vol_ok = float(bar.volume) > vol_mult * avgv_v
                # Candle size filter
                crng_ok = crng > candle_mult * atr_v

                if mode == "reversion":
                    # Long: price below VWAP by >= dev_mult × ATR, bullish rejection
                    long_sig = (
                        vol_ok and crng_ok
                        and not long_taken
                        and cl < vwap_v - dev_mult * atr_v
                        and cl >= lo + _UPPER_PCT * crng
                    )
                    # Short: price above VWAP by >= dev_mult × ATR, bearish rejection
                    short_sig = (
                        vol_ok and crng_ok
                        and not short_taken
                        and cl > vwap_v + dev_mult * atr_v
                        and cl <= lo + _LOWER_PCT * crng
                    )

                else:  # breakout
                    # Long: price breaks above VWAP + dev_mult × ATR, momentum up
                    long_sig = (
                        vol_ok and crng_ok
                        and not long_taken
                        and cl > vwap_v + dev_mult * atr_v
                        and cl >= lo + _UPPER_PCT * crng
                    )
                    # Short: price breaks below VWAP - dev_mult × ATR, momentum down
                    short_sig = (
                        vol_ok and crng_ok
                        and not short_taken
                        and cl < vwap_v - dev_mult * atr_v
                        and cl <= lo + _LOWER_PCT * crng
                    )

                # LONG priority on same bar (mirrors MR convention)
                entered_long  = long_sig
                entered_short = short_sig and not long_sig

                if entered_long or entered_short:
                    dirn = "long" if entered_long else "short"

                    if mode == "reversion":
                        entry_px, sl_px, tp_px, dev_atr, rr = cls._rev_levels(
                            dirn, cl, vwap_v, atr_v, sl_or_tp_mult
                        )
                    else:
                        entry_px, sl_px, tp_px, dev_atr, rr = cls._bo_levels(
                            dirn, cl, vwap_v, atr_v, sl_or_tp_mult
                        )

                    if rr <= 0 or (dirn == "long" and entry_px >= tp_px) \
                               or (dirn == "short" and entry_px <= tp_px):
                        # No room — skip
                        i += 1; continue

                    in_position = True
                    if dirn == "long":  long_taken  = True
                    else:               short_taken = True

                    # ── Exit simulation ───────────────────────────────────────
                    post = bars[i + 1:]
                    exit_px, exit_reason, exit_ts = cls._exit(
                        post, dirn, entry_px, sl_px, tp_px, max_bars
                    )
                    in_position = False

                    pnl_pts = exit_px - entry_px if dirn == "long" else entry_px - exit_px
                    pnl_net = pnl_pts * PV - COMM

                    trades.append(VWAPTrade(
                        entry_ts   = bar.Index,
                        exit_ts    = exit_ts,
                        direction  = dirn,
                        entry_px   = entry_px,
                        exit_px    = exit_px,
                        sl_px      = sl_px,
                        tp_px      = tp_px,
                        exit_reason= exit_reason,
                        pnl_pts    = pnl_pts,
                        pnl_net    = pnl_net,
                        atr_entry  = atr_v,
                        vwap_entry = vwap_v,
                        dev_atr    = dev_atr,
                        rr         = rr,
                        strategy   = mode,
                    ))

                    # Fast-forward i past exit bar
                    try:
                        exit_pos = next(
                            j for j, b in enumerate(post)
                            if b.Index >= exit_ts
                        )
                        i += exit_pos + 2  # resume bar after exit
                    except StopIteration:
                        break

                    # Stop if both directions taken
                    if long_taken and short_taken:
                        break
                    continue

            i += 1

        return trades

    # ── Entry level calculators ───────────────────────────────────────────────

    @staticmethod
    def _rev_levels(dirn, cl, vwap, atr, sl_mult):
        if dirn == "long":
            entry = cl + SLIP
            sl    = entry - sl_mult * atr     # below entry
            tp    = vwap                       # fixed VWAP
            dev   = (vwap - cl) / atr if atr > 0 else 0.0
            sl_d  = entry - sl
            tp_d  = tp - entry
        else:
            entry = cl - SLIP
            sl    = entry + sl_mult * atr     # above entry
            tp    = vwap
            dev   = (cl - vwap) / atr if atr > 0 else 0.0
            sl_d  = sl - entry
            tp_d  = entry - tp
        rr = tp_d / sl_d if sl_d > 0 else 0.0
        return entry, sl, tp, dev, rr

    @staticmethod
    def _bo_levels(dirn, cl, vwap, atr, tp_mult):
        if dirn == "long":
            entry = cl + SLIP
            sl    = vwap              # revert back to VWAP = stop
            tp    = vwap + tp_mult * atr
            dev   = (cl - vwap) / atr if atr > 0 else 0.0
            sl_d  = entry - sl
            tp_d  = tp - entry
        else:
            entry = cl - SLIP
            sl    = vwap
            tp    = vwap - tp_mult * atr
            dev   = (vwap - cl) / atr if atr > 0 else 0.0
            sl_d  = sl - entry
            tp_d  = entry - tp
        rr = tp_d / sl_d if sl_d > 0 else 0.0
        return entry, sl, tp, dev, rr

    # ── Exit simulation ────────────────────────────────────────────────────────

    @staticmethod
    def _exit(post_bars, dirn, entry_px, sl_px, tp_px, max_bars):
        bars_held = 0
        last_bar  = None

        for bar in post_bars:
            bars_held += 1
            bt = bar.Index.time()
            h  = float(bar.high)
            l  = float(bar.low)
            c  = float(bar.close)

            if dirn == "long":
                sl_hit = l <= sl_px
                tp_hit = h >= tp_px
            else:
                sl_hit = h >= sl_px
                tp_hit = l <= tp_px

            # SL wins when same bar hits both
            if sl_hit:
                return sl_px, "sl", bar.Index
            if tp_hit:
                return tp_px, "tp", bar.Index

            # EOD exit
            if bt >= _EOD_T:
                ep = (c - SLIP) if dirn == "long" else (c + SLIP)
                return ep, "eod", bar.Index

            # Timeout
            if bars_held >= max_bars:
                ep = (c - SLIP) if dirn == "long" else (c + SLIP)
                return ep, "timeout", bar.Index

            last_bar = bar

        # Session ended while still in position
        if last_bar is not None:
            c  = float(last_bar.close)
            ep = (c - SLIP) if dirn == "long" else (c + SLIP)
            return ep, "eod", last_bar.Index
        return entry_px, "eod", post_bars[0].Index if post_bars else pd.Timestamp.now()


# ── Pre-P&L diagnostic ────────────────────────────────────────────────────────

def run_diagnostic(df: pd.DataFrame, regime_map: dict,
                   deviation_mult: float = 1.5,
                   candle_mult:    float = 0.8,
                   volume_mult:    float = 1.2,
                   time_start: str = "10:00",
                   time_end:   str = "14:30",
                   max_bars:   int = 90) -> None:
    """
    Analyse VWAP deviation geometry before looking at any P&L.

    Section 1: deviation distribution (all ranging-day bars)
    Section 2: FILTERED signal frequency (deviation + candle_mult + volume_mult)
    Section 3: reversion speed for filtered signals
    Section 4: geometric R/R at filtered signal bars
    """
    t_s = _parse_time(time_start)
    t_e = _parse_time(time_end)

    date_arr = np.array(df.index.date)
    u_dates, first_idx, counts = np.unique(
        date_arr, return_index=True, return_counts=True
    )

    deviations_atr   = []   # |close - VWAP| / ATR (all ranging-day bars)
    signal_devs      = []   # dev at FILTERED signal bars
    rev_bars_success = []   # bars to VWAP after filtered signal (success)
    rev_bars_fail    = []   # max_bars or session end (failure)
    rr_ratios        = []   # R/R at filtered signal bars
    raw_spd          = []   # raw dev-only signals per ranging day
    filt_spd         = []   # filtered (dev+candle+vol) signals per ranging day

    for ui in range(len(u_dates)):
        d   = u_dates[ui]
        seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]

        regime = regime_map.get(d, "volatile")
        if regime != "ranging":
            continue

        bars         = list(seg.itertuples())
        day_raw      = 0
        day_filt     = 0

        for j, bar in enumerate(bars):
            bt    = bar.Index.time()
            atr_v = float(bar.atr)
            avgv  = float(bar.avg_vol)
            if atr_v <= 0 or avgv <= 0 or not (t_s <= bt < t_e):
                continue

            cl   = float(bar.close)
            hi   = float(bar.high)
            lo   = float(bar.low)
            crng = hi - lo
            vwap = float(bar.vwap)
            vol  = float(bar.volume)

            if crng <= 0:
                continue

            dev = abs(cl - vwap) / atr_v
            deviations_atr.append(dev)

            if dev < deviation_mult:
                continue

            day_raw += 1

            # Full candle + volume filter (same as engine)
            if crng <= candle_mult * atr_v:
                continue
            if vol <= volume_mult * avgv:
                continue
            # Directional rejection: close in upper or lower 30% of range
            is_bull = cl >= lo + _UPPER_PCT * crng
            is_bear = cl <= lo + _LOWER_PCT * crng
            if not (is_bull or is_bear):
                continue

            day_filt += 1
            signal_devs.append(dev)

            sl_d = 0.5 * atr_v
            tp_d = abs(vwap - cl) - SLIP
            rr   = tp_d / sl_d if sl_d > 0 else 0.0
            rr_ratios.append(rr)

            # Raw reversion: does price reach VWAP within max_bars?
            post    = bars[j + 1:]
            reached = False
            for k, pb in enumerate(post):
                if k >= max_bars:
                    rev_bars_fail.append(max_bars)
                    break
                if cl < vwap and float(pb.high) >= vwap:
                    rev_bars_success.append(k + 1)
                    reached = True; break
                if cl > vwap and float(pb.low) <= vwap:
                    rev_bars_success.append(k + 1)
                    reached = True; break
            else:
                if not reached:
                    rev_bars_fail.append(len(post))

        raw_spd.append(day_raw)
        filt_spd.append(day_filt)

    # ── Print diagnostic report ────────────────────────────────────────
    W = 68
    print("\n" + "=" * W)
    print("  PRE-PNL DIAGNOSTIC -- VWAP DEVIATION ANALYSIS")
    print("=" * W)

    devs = np.array(deviations_atr)
    pcts = np.percentile(devs, [25, 50, 75, 90, 95]) if devs.size else [0]*5
    pct_gt = float((devs >= deviation_mult).mean()) * 100 if devs.size else 0.0

    print(f"\n  1. VWAP DEVIATION DISTRIBUTION (ranging days, {time_start}-{time_end})")
    print(f"     Bars analysed           : {devs.size:,}")
    print(f"     Deviation (ATR units)   : p25={pcts[0]:.2f}  p50={pcts[1]:.2f}"
          f"  p75={pcts[2]:.2f}  p90={pcts[3]:.2f}  p95={pcts[4]:.2f}")
    print(f"     > {deviation_mult:.1f} ATR from VWAP  : {pct_gt:.1f}% of bars")

    n_ranging = len(filt_spd)
    raw_arr   = np.array(raw_spd)
    filt_arr  = np.array(filt_spd)
    pct_sig_day = float((filt_arr > 0).sum()) / max(n_ranging, 1) * 100

    print(f"\n  2. SIGNAL FREQUENCY (filtered: dev>{deviation_mult} + candle>{candle_mult}x + vol>{volume_mult}x)")
    print(f"     Ranging days analysed   : {n_ranging}")
    print(f"     Raw dev-only signals    : {int(raw_arr.sum())}  ({raw_arr.mean():.1f}/day)")
    print(f"     Filtered signals total  : {int(filt_arr.sum())}  ({filt_arr.mean():.2f}/day)")
    dist = {k: int((filt_arr == k).sum()) for k in range(5)}
    print(f"     Distribution: 0sig={dist[0]}d  1sig={dist[1]}d  2sig={dist[2]}d"
          f"  3sig={dist[3]}d  4+sig={int((filt_arr>=4).sum())}d")
    print(f"     Ranging days with >= 1 filtered signal: {pct_sig_day:.1f}%")

    n_succ  = len(rev_bars_success)
    n_fail  = len(rev_bars_fail)
    n_total = n_succ + n_fail
    pct_rev = n_succ / max(n_total, 1) * 100

    if n_total > 0:
        rb_arr = np.array(rev_bars_success) if n_succ else np.array([0])
        print(f"\n  3. REVERSION SPEED (filtered signals reaching VWAP within {max_bars} bars)")
        print(f"     Reach VWAP: {n_succ}/{n_total} = {pct_rev:.1f}%")
        if n_succ:
            print(f"     Bars to VWAP  : p25={np.percentile(rb_arr,25):.0f}"
                  f"  p50={np.percentile(rb_arr,50):.0f}"
                  f"  p75={np.percentile(rb_arr,75):.0f}"
                  f"  p95={np.percentile(rb_arr,95):.0f}")

    if rr_ratios:
        rr = np.array(rr_ratios)
        pct_2 = float((rr >= 2.0).mean()) * 100
        pct_3 = float((rr >= 3.0).mean()) * 100
        print(f"\n  4. GEOMETRIC R/R AT FILTERED SIGNAL BARS (sl=0.5xATR, tp=VWAP)")
        print(f"     R/R distribution    : p25={np.percentile(rr,25):.2f}"
              f"  p50={np.percentile(rr,50):.2f}"
              f"  p75={np.percentile(rr,75):.2f}")
        print(f"     Signals with R/R >= 2:1 : {pct_2:.1f}%")
        print(f"     Signals with R/R >= 3:1 : {pct_3:.1f}%")

    print("=" * W)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_time(t_str: str) -> dt_time:
    h, m = map(int, t_str.split(":"))
    return dt_time(h, m)
