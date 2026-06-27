"""
VWAP Breakout strategy engine for NQ 1-min (Phase 9).

Signal: VWAP crossover on trending days (OR/ATR hard threshold + HMM).
Exit:   Structure-based SL (VWAP at entry) + trailing stop that activates
        after trade is ATR-profitable, capturing the full trending move.

Key design choices:
  - SL = VWAP at time of entry (fixed -- if price crosses back, thesis broken)
  - Trailing stop activates after trail_activation × ATR in profit
  - trail_price = peak_favorable - trail_mult × ATR_entry  (LONG)
  - SL wins if same bar touches both SL and trail levels
  - Max 1 trade per session (first valid crossover only)
  - Entry at close + 1-tick slip; SL/trail exits at exact level (no slip)
  - EOD/timeout exits at close ± 1-tick slip
"""

import math
from collections import Counter
from dataclasses import dataclass
from datetime import time as dt_time
from typing import List, Optional

import numpy as np
import pandas as pd

SLIP = 0.25
COMM = 4.0     # round-trip ($2/side)
PV   = 20.0    # NQ point value $/pt

_EOD_T = dt_time(15, 45)
_OR_S  = dt_time(9, 30)
_OR_E  = dt_time(10, 30)   # exclusive

# Close in upper 40% of range = close >= low + 0.60 × range
_UPPER_40 = 0.60
# Close in lower 40% of range = close <= low + 0.40 × range
_LOWER_40 = 0.40


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class BreakoutTrade:
    entry_ts:       pd.Timestamp
    exit_ts:        pd.Timestamp
    direction:      str          # "long" | "short"
    entry_px:       float
    exit_px:        float
    sl_px:          float        # VWAP at entry
    exit_reason:    str          # "sl" | "trail" | "eod" | "timeout"
    pnl_pts:        float
    pnl_net:        float
    atr_entry:      float
    vwap_entry:     float
    or_range_atr:   float        # OR/ATR for this session
    peak_favorable: float        # best price seen (max H for long, min L for short)
    bars_held:      int
    trail_mult:     float


# ── Results container ─────────────────────────────────────────────────────────

class BreakoutResults:
    def __init__(self, trades: List[BreakoutTrade], label: str = ""):
        self.trades = trades
        self.label  = label

    def metrics(self, capital: float = 100_000.0) -> dict:
        if not self.trades:
            return {"n": 0, "wr": 0.0, "pf": float("nan"), "sharpe": 0.0,
                    "ret_pct": 0.0, "ret_usd": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0,
                    "max_dd_usd": 0.0, "max_dd_pct": 0.0,
                    "trades_per_day": 0.0}
        net  = np.array([t.pnl_net for t in self.trades])
        wins = net[net > 0]; losses = net[net <= 0]
        gw   = float(wins.sum())  if wins.size  else 0.0
        gl   = float(abs(losses.sum())) if losses.size else 0.0
        pf   = gw / gl if gl > 0 else float("inf")
        wr   = float(wins.size) / len(net)
        std  = float(net.std())
        sr   = float(net.mean() / std * math.sqrt(252)) if std > 0 else 0.0
        ret_usd = float(net.sum())
        ret_pct = ret_usd / capital * 100.0

        curve   = np.concatenate([[0.0], np.cumsum(net)])
        peak    = np.maximum.accumulate(curve)
        dd_usd  = float((peak - curve).max())
        dd_pct  = dd_usd / capital * 100.0

        dates   = {t.entry_ts.date() for t in self.trades}
        tpd     = len(self.trades) / max(len(dates), 1)

        return {
            "n": len(net), "wr": wr, "pf": pf, "sharpe": sr,
            "ret_usd": ret_usd, "ret_pct": ret_pct,
            "avg_win": float(wins.mean()) if wins.size else 0.0,
            "avg_loss": float(losses.mean()) if losses.size else 0.0,
            "max_dd_usd": dd_usd, "max_dd_pct": dd_pct,
            "trades_per_day": tpd,
        }

    def annual_breakdown(self) -> dict:
        by_year = {}
        for t in self.trades:
            by_year.setdefault(t.entry_ts.year, []).append(t.pnl_net)
        return {yr: _year_metrics(pnls) for yr, pnls in sorted(by_year.items())}

    def exit_breakdown(self) -> dict:
        cnt   = Counter(t.exit_reason for t in self.trades)
        total = max(sum(cnt.values()), 1)
        return {r: cnt.get(r, 0) / total for r in ["sl", "trail", "eod", "timeout"]}

    def to_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "entry_ts": t.entry_ts, "exit_ts": t.exit_ts,
            "direction": t.direction,
            "entry_px": t.entry_px, "exit_px": t.exit_px, "sl_px": t.sl_px,
            "exit_reason": t.exit_reason,
            "pnl_pts": t.pnl_pts, "pnl_net": t.pnl_net,
            "atr_entry": t.atr_entry, "vwap_entry": t.vwap_entry,
            "or_range_atr": t.or_range_atr,
            "peak_favorable": t.peak_favorable, "bars_held": t.bars_held,
            "trail_mult": t.trail_mult,
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
        "ret_usd": float(net.sum()),
    }


# ── Main engine ───────────────────────────────────────────────────────────────

class VWAPBreakoutEngine:
    """
    VWAP crossover + trailing stop breakout strategy.

    Day qualification:
      or_atr_map[date] >= or_atr_threshold   (hard OR-range filter)
      regime_map[date] == "trending"          (HMM secondary filter, optional)

    Signal: close crosses VWAP (prev_close on other side), with candle
            size, volume, and directional confirmation.
    """

    @classmethod
    def run(
        cls,
        df: pd.DataFrame,
        or_atr_map: dict,          # date -> or_range/ATR for each session
        regime_map: dict,          # date -> "ranging"|"trending"|"volatile"
        *,
        or_atr_threshold:   float = 1.2,
        candle_mult:        float = 0.8,
        volume_mult:        float = 1.3,
        trail_mult:         float = 1.5,
        trail_activation:   float = 1.0,   # ATR units of profit before trail fires
        max_bars:           int   = 90,
        time_start:         str   = "10:30",
        time_end:           str   = "14:00",
        require_hmm:        bool  = True,
        label:              str   = "",
    ) -> BreakoutResults:

        t_s = _parse_time(time_start)
        t_e = _parse_time(time_end)

        date_arr = np.array(df.index.date)
        u_dates, first_idx, counts = np.unique(
            date_arr, return_index=True, return_counts=True
        )

        trades: List[BreakoutTrade] = []

        for ui in range(len(u_dates)):
            d   = u_dates[ui]
            seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]

            # OR/ATR hard filter
            or_atr = or_atr_map.get(d, 0.0)
            if np.isnan(or_atr) or or_atr < or_atr_threshold:
                continue

            # HMM regime filter (optional)
            if require_hmm:
                regime = regime_map.get(d, "volatile")
                if regime != "trending":
                    continue

            trade = cls._sim_session(
                seg, d, or_atr,
                t_s, t_e, candle_mult, volume_mult,
                trail_mult, trail_activation, max_bars
            )
            if trade is not None:
                trades.append(trade)

        return BreakoutResults(trades, label)

    # ── Session simulation ────────────────────────────────────────────────────

    @classmethod
    def _sim_session(
        cls, seg, date, or_atr,
        t_start, t_end, candle_mult, volume_mult,
        trail_mult, trail_activation, max_bars
    ) -> Optional[BreakoutTrade]:

        bars       = list(seg.itertuples())
        n          = len(bars)
        prev_close = None
        prev_vwap  = None

        for i in range(n):
            bar = bars[i]
            bt  = bar.Index.time()
            cl  = float(bar.close)
            vwap = float(bar.vwap)

            if prev_close is not None and t_start <= bt < t_end:
                atr_v = float(bar.atr)
                avgv  = float(bar.avg_vol)
                hi    = float(bar.high)
                lo    = float(bar.low)
                crng  = hi - lo
                vol   = float(bar.volume)

                if atr_v > 0 and avgv > 0 and crng > 0:
                    # VWAP crossover
                    long_cross  = (prev_close < prev_vwap) and (cl > vwap)
                    short_cross = (prev_close > prev_vwap) and (cl < vwap)

                    # Quality filters
                    crng_ok = crng > candle_mult * atr_v
                    vol_ok  = vol  > volume_mult * avgv

                    long_sig  = long_cross  and crng_ok and vol_ok \
                                and cl >= lo + _UPPER_40 * crng
                    short_sig = short_cross and crng_ok and vol_ok \
                                and cl <= lo + _LOWER_40 * crng

                    dirn = "long" if long_sig else ("short" if short_sig else None)

                    if dirn is not None:
                        entry_px = cl + SLIP if dirn == "long" else cl - SLIP
                        sl_px    = vwap   # structure SL at VWAP at time of entry

                        # Exit simulation
                        post  = bars[i + 1:]
                        exit_px, exit_reason, exit_ts, peak, bh = cls._exit(
                            post, dirn, entry_px, sl_px, atr_v,
                            trail_mult, trail_activation, max_bars
                        )

                        pnl_pts = exit_px - entry_px if dirn == "long" else entry_px - exit_px
                        pnl_net = pnl_pts * PV - COMM

                        return BreakoutTrade(
                            entry_ts       = bar.Index,
                            exit_ts        = exit_ts,
                            direction      = dirn,
                            entry_px       = entry_px,
                            exit_px        = exit_px,
                            sl_px          = sl_px,
                            exit_reason    = exit_reason,
                            pnl_pts        = pnl_pts,
                            pnl_net        = pnl_net,
                            atr_entry      = atr_v,
                            vwap_entry     = vwap,
                            or_range_atr   = or_atr,
                            peak_favorable = peak,
                            bars_held      = bh,
                            trail_mult     = trail_mult,
                        )

            prev_close = cl
            prev_vwap  = vwap

        return None

    # ── Exit simulation ───────────────────────────────────────────────────────

    @staticmethod
    def _exit(post_bars, dirn, entry_px, sl_px, atr_entry,
              trail_mult, trail_activation, max_bars):
        peak           = entry_px
        bars_held      = 0
        trail_activated = False
        trail_price     = None
        last_bar        = None

        for bar in post_bars:
            bars_held += 1
            h  = float(bar.high)
            l  = float(bar.low)
            c  = float(bar.close)
            bt = bar.Index.time()

            # Update peak favorable
            if dirn == "long":
                peak = max(peak, h)
            else:
                peak = min(peak, l)

            # Check trail activation
            if not trail_activated:
                profit = (peak - entry_px) if dirn == "long" else (entry_px - peak)
                if profit >= trail_activation * atr_entry:
                    trail_activated = True

            # Current trail price
            if trail_activated:
                trail_price = (peak - trail_mult * atr_entry) if dirn == "long" \
                              else (peak + trail_mult * atr_entry)

            # 1. SL (VWAP cross-back) — checked first, wins on same-bar conflict
            if dirn == "long":
                sl_hit = l <= sl_px
            else:
                sl_hit = h >= sl_px

            if sl_hit:
                return sl_px, "sl", bar.Index, peak, bars_held

            # 2. Trailing stop
            if trail_activated and trail_price is not None:
                if dirn == "long" and l <= trail_price:
                    return trail_price, "trail", bar.Index, peak, bars_held
                elif dirn == "short" and h >= trail_price:
                    return trail_price, "trail", bar.Index, peak, bars_held

            # 3. EOD
            if bt >= _EOD_T:
                ep = (c - SLIP) if dirn == "long" else (c + SLIP)
                return ep, "eod", bar.Index, peak, bars_held

            # 4. Timeout
            if bars_held >= max_bars:
                ep = (c - SLIP) if dirn == "long" else (c + SLIP)
                return ep, "timeout", bar.Index, peak, bars_held

            last_bar = bar

        # Session ended with open position
        if last_bar is not None:
            c  = float(last_bar.close)
            ep = (c - SLIP) if dirn == "long" else (c + SLIP)
            return ep, "eod", last_bar.Index, peak, bars_held
        return entry_px, "eod", pd.Timestamp.now(), peak, bars_held


# ── Pre-P&L diagnostic ────────────────────────────────────────────────────────

def run_diagnostic_p9(
    df: pd.DataFrame,
    or_atr_map: dict,
    regime_map: dict,
    *,
    or_atr_threshold: float = 1.2,
    candle_mult:      float = 0.8,
    volume_mult:      float = 1.3,
    time_start:       str   = "10:30",
    time_end:         str   = "14:00",
    max_bars:         int   = 90,
) -> None:
    """
    Mandatory pre-P&L diagnostic on the TRAINING SET.
    Validates signal geometry before any P&L is examined.

    Reports:
      1. Qualifying day frequency (OR filter + HMM)
      2. Crossover funnel (raw -> candle -> volume -> direction filter)
      3. Natural TP distance + SL distance (in ATR)
      4. Geometric R/R estimate
    """
    t_s = _parse_time(time_start)
    t_e = _parse_time(time_end)

    date_arr = np.array(df.index.date)
    u_dates, first_idx, counts = np.unique(
        date_arr, return_index=True, return_counts=True
    )

    total_days          = 0
    hmm_trending_days   = 0
    qualifying_days     = 0   # OR/ATR > threshold AND HMM trending

    cross_raw           = 0   # all crossovers in entry window
    cross_candle        = 0   # + candle filter
    cross_vol           = 0   # + volume filter
    cross_all           = 0   # + direction filter

    sl_dists_atr        = []  # |entry - VWAP| / ATR for filtered signals
    nat_tp_atr          = []  # max favorable / ATR before VWAP return
    bars_to_vwap        = []  # bars until VWAP returned (for those that did)

    for ui in range(len(u_dates)):
        d   = u_dates[ui]
        seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]
        total_days += 1

        regime = regime_map.get(d, "volatile")
        or_atr = or_atr_map.get(d, 0.0)

        if regime == "trending":
            hmm_trending_days += 1

        qualified = (not np.isnan(or_atr)) and (or_atr >= or_atr_threshold) \
                    and (regime == "trending")
        if qualified:
            qualifying_days += 1

        # Only count crossovers on qualifying days for funnel stats
        if not qualified:
            continue

        bars     = list(seg.itertuples())
        n        = len(bars)
        prev_cl  = None
        prev_vwap = None

        for i in range(n):
            bar  = bars[i]
            bt   = bar.Index.time()
            cl   = float(bar.close)
            vwap = float(bar.vwap)

            if prev_cl is not None and t_s <= bt < t_e:
                atr_v = float(bar.atr)
                avgv  = float(bar.avg_vol)
                hi    = float(bar.high)
                lo    = float(bar.low)
                crng  = hi - lo
                vol   = float(bar.volume)

                if atr_v <= 0 or avgv <= 0 or crng <= 0:
                    prev_cl = cl; prev_vwap = vwap; continue

                long_cross  = (prev_cl < prev_vwap) and (cl > vwap)
                short_cross = (prev_cl > prev_vwap) and (cl < vwap)
                is_cross    = long_cross or short_cross

                if is_cross:
                    cross_raw += 1
                    dirn = "long" if long_cross else "short"

                    crng_ok = crng > candle_mult * atr_v
                    vol_ok  = vol  > volume_mult * avgv
                    dir_ok  = (dirn == "long"  and cl >= lo + _UPPER_40 * crng) \
                           or (dirn == "short" and cl <= lo + _LOWER_40 * crng)

                    if crng_ok:
                        cross_candle += 1
                        if vol_ok:
                            cross_vol += 1
                            if dir_ok:
                                cross_all += 1

                                # Measure SL distance and natural TP
                                entry_px = cl + SLIP if dirn == "long" else cl - SLIP
                                vwap_sl  = vwap
                                sl_d_atr = abs(entry_px - vwap_sl) / atr_v
                                sl_dists_atr.append(sl_d_atr)

                                # Natural TP: how far does price move before VWAP return?
                                post   = bars[i + 1:]
                                peak   = entry_px
                                reached = False
                                for k, pb in enumerate(post):
                                    if k >= max_bars:
                                        break
                                    ph = float(pb.high); pl = float(pb.low)
                                    if dirn == "long":
                                        peak = max(peak, ph)
                                        if pl <= vwap_sl:
                                            reached = True; bars_to_vwap.append(k + 1); break
                                    else:
                                        peak = min(peak, pl)
                                        if ph >= vwap_sl:
                                            reached = True; bars_to_vwap.append(k + 1); break
                                nat_move = abs(peak - entry_px) / atr_v
                                nat_tp_atr.append(nat_move)

            prev_cl   = cl
            prev_vwap = vwap

    # ── Print report ──────────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print("  PHASE 9 -- PRE-PNL DIAGNOSTIC (TRAINING SET)")
    print("=" * W)

    pct_hmm   = hmm_trending_days / max(total_days, 1) * 100
    pct_qual  = qualifying_days   / max(total_days, 1) * 100
    print(f"\n  1. QUALIFYING DAY FREQUENCY")
    print(f"     Training days total          : {total_days}")
    print(f"     HMM trending days            : {hmm_trending_days} ({pct_hmm:.1f}%)")
    print(f"     OR/ATR > {or_atr_threshold:.1f} AND trending  : {qualifying_days} ({pct_qual:.1f}%)")
    if qualifying_days > 0:
        spd = cross_all / qualifying_days
        print(f"     Avg filtered signals/session : {spd:.2f}")
    print(f"     Total test-set range: see experiment output")

    if cross_raw > 0:
        p_cng = cross_candle / cross_raw * 100
        p_vol = cross_vol    / cross_raw * 100
        p_all = cross_all    / cross_raw * 100
        print(f"\n  2. SIGNAL FILTER FUNNEL (on qualifying days, {time_start}-{time_end})")
        print(f"     Raw crossovers              : {cross_raw}")
        print(f"     + candle > {candle_mult}x ATR      : {cross_candle} ({p_cng:.1f}%)")
        print(f"     + volume > {volume_mult}x avg      : {cross_vol}  ({p_vol:.1f}%)")
        print(f"     + directional filter         : {cross_all}  ({p_all:.1f}%) -- TRADEABLE")

    if sl_dists_atr:
        sl_arr = np.array(sl_dists_atr)
        ps = np.percentile(sl_arr, [25, 50, 75, 95])
        print(f"\n  3. SL DISTANCE (entry to VWAP, ATR units)")
        print(f"     Signals measured      : {len(sl_arr)}")
        print(f"     p25={ps[0]:.3f}  p50={ps[1]:.3f}  p75={ps[2]:.3f}  p95={ps[3]:.3f} ATR")
        pct_small = float((sl_arr < 0.1).mean()) * 100
        print(f"     < 0.1 ATR (< 2 pts)  : {pct_small:.1f}%  (very tight SL)")

    if nat_tp_atr:
        tp_arr  = np.array(nat_tp_atr)
        ps2     = np.percentile(tp_arr, [25, 50, 75, 95])
        pct_ret = len(bars_to_vwap) / len(nat_tp_atr) * 100 if nat_tp_atr else 0
        print(f"\n  4. NATURAL TP DISTANCE (peak move before VWAP return, ATR units)")
        print(f"     p25={ps2[0]:.2f}  p50={ps2[1]:.2f}  p75={ps2[2]:.2f}  p95={ps2[3]:.2f} ATR")
        if bars_to_vwap:
            bt_arr = np.array(bars_to_vwap)
            print(f"     Returned to VWAP in {max_bars}b : {len(bars_to_vwap)}/{len(nat_tp_atr)} = {pct_ret:.1f}%")
            print(f"     Bars to VWAP: p25={np.percentile(bt_arr,25):.0f}"
                  f"  p50={np.percentile(bt_arr,50):.0f}"
                  f"  p75={np.percentile(bt_arr,75):.0f}")

    if sl_dists_atr and nat_tp_atr:
        sl_med = float(np.median(sl_dists_atr))
        tp_med = float(np.median(nat_tp_atr))
        rr_est = tp_med / sl_med if sl_med > 0 else float("inf")
        pct_rr2 = float(np.mean(
            np.array(nat_tp_atr) / np.maximum(np.array(sl_dists_atr), 1e-6) >= 2.0
        ) * 100)
        print(f"\n  5. GEOMETRIC R/R ESTIMATE")
        print(f"     Median TP / Median SL  : {tp_med:.2f} / {sl_med:.3f} = {rr_est:.1f}:1")
        print(f"     Signals with R/R >= 2:1: {pct_rr2:.1f}%")
        if rr_est >= 2.0:
            print(f"     HYPOTHESIS HAS GEOMETRIC BASIS -- natural R/R > 2:1")
        else:
            print(f"     WARNING: median R/R < 2:1 -- check signal quality")

    print("=" * W)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_time(t_str: str) -> dt_time:
    h, m = map(int, t_str.split(":"))
    return dt_time(h, m)
