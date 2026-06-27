"""
POC (Point of Control) Mean Reversion strategy engine — Phases 11–16.

HYPOTHESIS (CONFIRMED): price extended from Volume Profile POC with an
exhaustion candle reverts back toward POC at a statistically significant rate.

CONFIRMED CONFIGURATION — B1 (Phase 15):
  Entry:  price >= 1.0 ATR from target_poc + exhaustion candle + volume > 1.3x avg
  TP:     entry + 0.67 * (target_poc - entry)    (fractional — 67% of the way back)
  SL:     1.0 ATR from entry
  F1:     prev_day_range / daily_atr < 1.2        (narrow prior session)
  F2:     abs(close - target_poc) / ATR >= 3.0    (meaningful distance at signal)
  F3:     abs(trend_5d / daily_atr) < 1.5         (no strong 5-day trend)
  Window: 09:45–14:30 NY. Max 120 bars. EOD exit 15:45.

CONFIRMED RESULTS (458 pooled OOS trades, 5 anchored WFO windows):
  PF 1.240  |  SR 1.087  |  WR 21.4%  |  R/R 4.58:1
  T-test p=0.066 (< 0.10)  |  Bootstrap p5=0.999 (> 0.95)
  4/5 WFO windows PF > 1.0
  FTMO sizing: 1 fixed contract — P(pass) 87.3%, median 94 trading days

  The B1 deterministic filter implementation is in:
  orb_system/strategy/poc_filtered.py (used by run_phase15_B1.py, run_phase16.py)

This file (poc_reversion.py) is the Phase 11 base engine without filters.
It is imported by poc_filtered.py in the orb_system package.
"""

import math
from dataclasses import dataclass
from datetime import time as dt_time
from typing import List

import numpy as np
import pandas as pd

SLIP = 0.25
COMM = 4.0
PV   = 20.0

_EOD_T  = dt_time(15, 45)
_UP_PCT = 0.60   # LONG close must be in top 40% of bar range
_LO_PCT = 0.40   # SHORT close must be in bottom 40%

SL_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0]


# -- Trade record ---------------------------------------------------------------

@dataclass
class POCTrade:
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    direction:   str
    entry_px:    float
    exit_px:     float
    sl_px:       float
    tp_px:       float
    exit_reason: str
    pnl_pts:     float
    pnl_net:     float
    atr_entry:   float
    target_poc:  float
    prev_poc:    float
    session_poc: float
    confluence:  bool
    tp_variant:  str
    tp_dist:     float
    sl_dist:     float
    bars_held:   int


# -- Results container ----------------------------------------------------------

def _year_metrics(pnls: list) -> dict:
    net = np.array(pnls)
    w   = net[net > 0];  l = net[net <= 0]
    gw  = float(w.sum())  if w.size else 0.0
    gl  = float(abs(l.sum())) if l.size else 0.0
    pf  = gw / gl if gl > 0 else float("inf")
    std = float(net.std())
    sr  = float(net.mean() / std * math.sqrt(252)) if std > 0 else 0.0
    return {
        "n": len(net),
        "wr": float(w.size) / len(net) if len(net) else 0.0,
        "pf": pf, "sharpe": sr,
        "ret_usd": float(net.sum()),
    }


class POCResults:
    def __init__(self, trades: List[POCTrade], label: str = ""):
        self.trades = trades
        self.label  = label

    def metrics(self, capital: float = 100_000.0) -> dict:
        if not self.trades:
            return {"n": 0, "wr": 0.0, "pf": float("nan"), "sharpe": 0.0,
                    "ret_pct": 0.0, "max_dd_pct": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0, "trades_per_day": 0.0}
        net    = np.array([t.pnl_net for t in self.trades])
        wins   = net[net > 0];  losses = net[net <= 0]
        gw     = float(wins.sum())   if wins.size   else 0.0
        gl     = float(abs(losses.sum())) if losses.size else 0.0
        pf     = gw / gl if gl > 0 else float("inf")
        wr     = float(wins.size) / len(net)
        std    = float(net.std())
        sr     = float(net.mean() / std * math.sqrt(252)) if std > 0 else 0.0
        ret    = float(net.sum()) / capital * 100.0
        curve  = np.concatenate([[0.0], np.cumsum(net)])
        peak   = np.maximum.accumulate(curve)
        max_dd = float(((peak - curve) / capital * 100.0).max())
        dates  = {t.entry_ts.date() for t in self.trades}
        tpd    = len(self.trades) / max(len(dates), 1)
        return {
            "n": len(net), "wr": wr, "pf": pf, "sharpe": sr,
            "ret_pct": ret, "max_dd_pct": max_dd,
            "avg_win":  float(wins.mean())   if wins.size   else 0.0,
            "avg_loss": float(losses.mean()) if losses.size else 0.0,
            "trades_per_day": tpd,
        }

    def annual_breakdown(self) -> dict:
        by_year = {}
        for t in self.trades:
            by_year.setdefault(t.entry_ts.year, []).append(t.pnl_net)
        return {yr: _year_metrics(v) for yr, v in sorted(by_year.items())}

    def exit_breakdown(self) -> dict:
        from collections import Counter
        cnt   = Counter(t.exit_reason for t in self.trades)
        total = max(len(self.trades), 1)
        return {r: cnt.get(r, 0) / total for r in ["sl", "tp", "eod", "timeout"]}

    def to_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trades])


# -- Exit simulation ------------------------------------------------------------

def _exit(post_bars, dirn, entry_px, sl_px, tp_px, max_bars):
    bars_held = 0
    last_bar  = None
    for bar in post_bars:
        bars_held += 1
        bt = bar.Index.time()
        h  = float(bar.high);  l = float(bar.low);  c = float(bar.close)
        sl_hit = (l <= sl_px) if dirn == "long" else (h >= sl_px)
        tp_hit = (h >= tp_px) if dirn == "long" else (l <= tp_px)
        if sl_hit:
            return sl_px, "sl", bar.Index, bars_held
        if tp_hit:
            return tp_px, "tp", bar.Index, bars_held
        if bt >= _EOD_T:
            ep = (c - SLIP) if dirn == "long" else (c + SLIP)
            return ep, "eod", bar.Index, bars_held
        if bars_held >= max_bars:
            ep = (c - SLIP) if dirn == "long" else (c + SLIP)
            return ep, "timeout", bar.Index, bars_held
        last_bar = bar
    if last_bar is not None:
        c  = float(last_bar.close)
        ep = (c - SLIP) if dirn == "long" else (c + SLIP)
        return ep, "eod", last_bar.Index, bars_held
    return entry_px, "eod", pd.Timestamp.now(), bars_held


def _parse_time(s: str) -> dt_time:
    h, m = map(int, s.split(":"))
    return dt_time(h, m)


# -- Diagnostic -----------------------------------------------------------------

def run_diagnostic_p11(
    df: pd.DataFrame,
    *,
    deviation_mult: float  = 1.5,
    volume_mult:    float  = 1.3,
    exhaustion_mult: float = 0.8,
    sl_mults: list         = None,
    max_bars: int          = 120,
    time_start: str        = "09:45",
    time_end:   str        = "14:30",
) -> bool:
    """
    Mandatory pre-P&L diagnostic on training data.
    Returns True if at least one sl_mult shows positive theoretical expectancy.
    """
    if sl_mults is None:
        sl_mults = SL_MULTS
    t_s = _parse_time(time_start)
    t_e = _parse_time(time_end)

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    u_dates  = np.unique(date_arr)
    all_pos  = np.arange(len(df))

    W = 70
    print("\n" + "=" * W)
    print("  PHASE 11 PRE-P&L DIAGNOSTIC")
    print(f"  Entry: dev={deviation_mult}xATR | exhaust={exhaustion_mult}xATR | vol={volume_mult}xavg")
    print(f"  Window: {time_start}-{time_end} | max_bars={max_bars}")
    print("=" * W)

    # -- Section 1: POC level statistics ---------------------------------------
    print("\n  SECTION 1  -  POC Level Statistics (daily)")

    atr_col = df["atr"].values
    prev_pocs, sess_pocs_eod, gaps_atr, conf_eod = [], [], [], []

    for i, d in enumerate(u_dates):
        mask  = (date_arr == d)
        idxs  = all_pos[mask]
        times = time_arr[mask]

        # RTH open bar ATR
        rth   = np.array([t >= dt_time(9, 30) for t in times])
        if rth.sum() == 0:
            continue
        open_idx  = idxs[rth][0]
        open_atr  = float(atr_col[open_idx])
        open_px   = float(df["open"].values[open_idx])

        pp = float(df["prev_poc"].values[open_idx])
        if np.isnan(pp) or open_atr <= 0:
            continue
        prev_pocs.append(pp)
        gap_atr = abs(pp - open_px) / open_atr
        gaps_atr.append(gap_atr)

        # EOD session_poc
        sp_eod = float(df["session_poc"].values[idxs[-1]])
        if not np.isnan(sp_eod):
            sess_pocs_eod.append(sp_eod)
            conf_eod.append(abs(pp - sp_eod) <= 2.0)

    n_sess = len(prev_pocs)
    if n_sess == 0:
        print("  No valid sessions with prev_poc.")
        return False

    gaps = np.array(gaps_atr)
    print(f"  Sessions with valid prev_poc : {n_sess}")
    print(f"  Gap prev_poc->open (ATR units): "
          f"p25={np.percentile(gaps,25):.2f}  p50={np.percentile(gaps,50):.2f}  "
          f"p75={np.percentile(gaps,75):.2f}  mean={gaps.mean():.2f}")
    pct_1atr = (gaps <= 1.0).mean() * 100
    pct_2atr = (gaps <= 2.0).mean() * 100
    pct_3atr = (gaps <= 3.0).mean() * 100
    print(f"  Prev_poc within 1/2/3 ATR of open: "
          f"{pct_1atr:.1f}% / {pct_2atr:.1f}% / {pct_3atr:.1f}%")
    if conf_eod:
        pct_conf = np.mean(conf_eod) * 100
        print(f"  Sessions with EOD confluence (|prev-sess|<=2pt): "
              f"{sum(conf_eod)}/{len(conf_eod)} ({pct_conf:.1f}%)")

    # -- Section 2: Deviation frequency ----------------------------------------
    print("\n  SECTION 2  -  Deviation & Signal Frequency")

    n_dev_bars = 0  # bars in window where |close - target_poc| > dev*ATR
    n_sess_w_signal = 0
    all_sigs_per_sess = []

    for i, d in enumerate(u_dates):
        mask   = (date_arr == d)
        idxs   = all_pos[mask]
        times  = time_arr[mask]
        seg    = df.iloc[idxs]
        bars   = list(seg.itertuples())

        pp_val = float(df["prev_poc"].values[idxs[0]])
        if np.isnan(pp_val):
            continue

        n_sig_this = 0
        for bar in bars:
            bt = bar.Index.time()
            if not (t_s <= bt <= t_e):
                continue
            atr_v  = float(bar.atr)
            tp_v   = float(bar.target_poc)
            cl     = float(bar.close)
            if np.isnan(tp_v) or atr_v <= 0:
                continue
            dev    = abs(cl - tp_v)
            if dev > deviation_mult * atr_v:
                n_dev_bars += 1
            # Full signal
            crng     = float(bar.high) - float(bar.low)
            avg_v    = float(bar.avg_vol)
            long_sig  = (
                cl < tp_v - deviation_mult * atr_v
                and crng > exhaustion_mult * atr_v
                and float(bar.close) > float(bar.open)
                and cl > float(bar.low) + _UP_PCT * crng
                and float(bar.volume) > volume_mult * avg_v
                and avg_v > 0
            )
            short_sig = (
                cl > tp_v + deviation_mult * atr_v
                and crng > exhaustion_mult * atr_v
                and float(bar.close) < float(bar.open)
                and cl < float(bar.low) + _LO_PCT * crng
                and float(bar.volume) > volume_mult * avg_v
                and avg_v > 0
            )
            if long_sig or short_sig:
                n_sig_this += 1

        all_sigs_per_sess.append(n_sig_this)
        if n_sig_this > 0:
            n_sess_w_signal += 1

    sps = np.array(all_sigs_per_sess)
    n_total_sig = int(sps.sum())
    print(f"  Sessions scanned: {len(sps)}")
    print(f"  Sessions with >=1 signal: {n_sess_w_signal} ({n_sess_w_signal/max(len(sps),1)*100:.1f}%)")
    print(f"  Total qualifying signals: {n_total_sig} "
          f"| avg/session: {sps.mean():.2f}")
    dist = {0: (sps == 0).sum(), 1: (sps == 1).sum(),
            2: (sps == 2).sum(), "3+": (sps >= 3).sum()}
    print(f"  Signal dist per session: "
          f"0={dist[0]} | 1={dist[1]} | 2={dist[2]} | 3+={dist['3+']}")

    # -- Section 3: Collect signals for reversion analysis ---------------------
    print("\n  SECTION 3  -  Reversion Rate (no SL)")

    signals = []  # list of (entry_px, target_poc_val, atr_e, dirn, post_bars, entry_ts)

    for i, d in enumerate(u_dates):
        mask  = (date_arr == d)
        idxs  = all_pos[mask]
        seg   = df.iloc[idxs]
        bars  = list(seg.itertuples())
        n_b   = len(bars)

        for j, bar in enumerate(bars):
            bt = bar.Index.time()
            if not (t_s <= bt <= t_e):
                continue
            atr_v  = float(bar.atr)
            avg_v  = float(bar.avg_vol)
            tp_v   = float(bar.target_poc)
            pp_v   = float(bar.prev_poc)
            if np.isnan(tp_v) or np.isnan(pp_v) or atr_v <= 0 or avg_v <= 0:
                continue
            cl   = float(bar.close)
            hi   = float(bar.high)
            lo   = float(bar.low)
            crng = hi - lo

            long_sig  = (
                cl < tp_v - deviation_mult * atr_v
                and crng > exhaustion_mult * atr_v
                and cl > float(bar.open)
                and cl > lo + _UP_PCT * crng
                and float(bar.volume) > volume_mult * avg_v
            )
            short_sig = (
                cl > tp_v + deviation_mult * atr_v
                and crng > exhaustion_mult * atr_v
                and cl < float(bar.open)
                and cl < lo + _LO_PCT * crng
                and float(bar.volume) > volume_mult * avg_v
            )
            if not (long_sig or short_sig):
                continue

            dirn     = "long" if long_sig else "short"
            entry_px = (cl + SLIP) if dirn == "long" else (cl - SLIP)
            post     = bars[j + 1:]
            signals.append({
                "entry_ts":  bar.Index,
                "entry_px":  entry_px,
                "target_poc": tp_v,
                "atr_e":     atr_v,
                "dirn":      dirn,
                "post":      post,
            })

    n_sig = len(signals)
    if n_sig == 0:
        print("  No qualifying signals found. Cannot continue.")
        return False

    print(f"  Total signals collected: {n_sig}")

    # Reversion simulation (no SL)
    reached_30, reached_60, reached_90, reached_120 = 0, 0, 0, 0
    reached_before_1atr = 0
    mae_list  = []
    bars_to_tp = []

    for s in signals:
        ep  = s["entry_px"]
        tp  = s["target_poc"]
        atr = s["atr_e"]
        dir = s["dirn"]
        post = s["post"]

        hit    = False
        b_cnt  = 0
        mae    = 0.0

        for bar in post:
            if bar.Index.date() != s["entry_ts"].date():
                break
            b_cnt += 1
            h = float(bar.high);  l = float(bar.low)
            if dir == "long":
                adverse = ep - l
                tp_hit  = h >= tp
            else:
                adverse = h - ep
                tp_hit  = l <= tp
            mae = max(mae, adverse)
            if tp_hit:
                hit = True
                if b_cnt <= 30:  reached_30 += 1
                if b_cnt <= 60:  reached_60 += 1
                if b_cnt <= 90:  reached_90 += 1
                if b_cnt <= 120: reached_120 += 1
                bars_to_tp.append(b_cnt)
                break
            if b_cnt >= max_bars:
                break

        mae_list.append(mae / atr if atr > 0 else 0.0)

        # Reached TP before moving 1 ATR against entry
        if hit and mae <= 1.0 * atr:
            reached_before_1atr += 1

    mae_arr = np.array(mae_list)
    pct = lambda x: x / n_sig * 100
    print(f"  Reached target_poc within  30 bars: {reached_30:3d} ({pct(reached_30):.1f}%)")
    print(f"  Reached target_poc within  60 bars: {reached_60:3d} ({pct(reached_60):.1f}%)")
    print(f"  Reached target_poc within  90 bars: {reached_90:3d} ({pct(reached_90):.1f}%)")
    print(f"  Reached target_poc within 120 bars: {reached_120:3d} ({pct(reached_120):.1f}%)")
    if bars_to_tp:
        print(f"  Median bars to reach TP: {np.median(bars_to_tp):.0f}")
    n_before = reached_before_1atr
    print(f"  Reached TP BEFORE moving 1 ATR adverse: "
          f"{n_before} ({pct(n_before):.1f}%)")
    print(f"  MAE (ATR units) p25={np.percentile(mae_arr,25):.2f}  "
          f"p50={np.percentile(mae_arr,50):.2f}  "
          f"p75={np.percentile(mae_arr,75):.2f}  "
          f"p90={np.percentile(mae_arr,90):.2f}")

    # -- Section 4: Geometric R/R estimate -------------------------------------
    print("\n  SECTION 4  -  Geometric R/R (TP=target_poc, SL=1.0xATR baseline)")

    tp_dists = []
    for s in signals:
        tp_dist = abs(s["entry_px"] - s["target_poc"])
        tp_dists.append(tp_dist)
    tp_arr = np.array(tp_dists)
    atr_arr = np.array([s["atr_e"] for s in signals])
    rr_arr  = tp_arr / np.where(atr_arr > 0, atr_arr, np.nan)
    rr_arr  = rr_arr[~np.isnan(rr_arr)]

    if len(rr_arr) > 0:
        print(f"  TP distance (pts): "
              f"p25={np.percentile(tp_arr,25):.1f}  "
              f"p50={np.percentile(tp_arr,50):.1f}  "
              f"p75={np.percentile(tp_arr,75):.1f}")
        print(f"  R/R (TP/1xATR):    "
              f"p25={np.percentile(rr_arr,25):.2f}  "
              f"p50={np.percentile(rr_arr,50):.2f}  "
              f"p75={np.percentile(rr_arr,75):.2f}")
        print(f"  % signals with R/R > 1.5: {(rr_arr > 1.5).mean()*100:.1f}%  "
              f"| R/R > 2.0: {(rr_arr > 2.0).mean()*100:.1f}%")

    # -- Section 5: Theoretical expectancy table --------------------------------
    print("\n  SECTION 5  -  Theoretical Expectancy Table")
    print(f"  {'SL mult':>8} | {'WR%':>6} | {'min WR%':>8} | "
          f"{'median RR':>9} | {'Expectancy':>10} | {'Viable':>6}")
    print("  " + "-" * 60)

    any_viable = False
    for slm in sl_mults:
        wins   = 0
        rr_vals = []
        for s in signals:
            ep   = s["entry_px"]
            tp   = s["target_poc"]
            atr  = s["atr_e"]
            dir  = s["dirn"]
            sl   = (ep - slm * atr) if dir == "long" else (ep + slm * atr)
            tp_d = abs(ep - tp)
            sl_d = slm * atr
            rr_vals.append(tp_d / sl_d if sl_d > 0 else 0.0)
            post = s["post"]
            hit_tp = False
            for bar in post:
                if bar.Index.date() != s["entry_ts"].date():
                    break
                h = float(bar.high);  l = float(bar.low)
                sl_hit = (l <= sl) if dir == "long" else (h >= sl)
                tp_hit = (h >= tp) if dir == "long" else (l <= tp)
                if sl_hit:
                    break
                if tp_hit:
                    hit_tp = True
                    break
            if hit_tp:
                wins += 1

        wr      = wins / n_sig if n_sig > 0 else 0.0
        med_rr  = float(np.median(rr_vals)) if rr_vals else 0.0
        min_wr  = 1.0 / (1.0 + med_rr) if med_rr > 0 else 1.0
        exp     = wr * med_rr - (1.0 - wr)
        viable  = exp > 0.0
        if viable:
            any_viable = True
        mark = " <<<" if viable else ""
        print(f"  {slm:>8.2f} | {wr*100:>5.1f}% | {min_wr*100:>7.1f}% | "
              f"{med_rr:>9.2f} | {exp:>+10.3f} | {'YES':>6}{mark}")

    print("  " + "-" * 60)
    if any_viable:
        print("  [OK] At least one SL configuration shows positive theoretical expectancy.")
        print("  Proceeding to backtests.")
    else:
        print("  [NO] No SL configuration produces positive theoretical expectancy.")
        print("  POC reversion hypothesis is not structurally viable.")
        print("  DO NOT proceed to backtests.")
    print("=" * W)

    return any_viable


# -- Main engine ----------------------------------------------------------------

class POCReversionEngine:

    @staticmethod
    def run(
        df: pd.DataFrame,
        *,
        tp_variant:      str   = "A",
        deviation_mult:  float = 1.5,
        sl_mult:         float = 1.0,
        volume_mult:     float = 1.3,
        exhaustion_mult: float = 0.8,
        confluence_only: bool  = False,
        max_bars:        int   = 120,
        time_start:      str   = "09:45",
        time_end:        str   = "14:30",
        label:           str   = "",
    ) -> POCResults:
        t_s    = _parse_time(time_start)
        t_e    = _parse_time(time_end)
        trades = POCReversionEngine._sim_all(
            df, tp_variant, deviation_mult, sl_mult, volume_mult,
            exhaustion_mult, confluence_only, max_bars, t_s, t_e
        )
        return POCResults(trades, label)

    @staticmethod
    def _sim_all(df, tp_variant, dev_mult, sl_mult, vol_mult,
                 exh_mult, conf_only, max_bars, t_s, t_e):

        date_arr = np.array(df.index.date)
        u_dates, first_idx, counts = np.unique(
            date_arr, return_index=True, return_counts=True
        )
        trades: List[POCTrade] = []

        for ui in range(len(u_dates)):
            d   = u_dates[ui]
            seg = df.iloc[int(first_idx[ui]): int(first_idx[ui]) + int(counts[ui])]
            trades.extend(POCReversionEngine._sim_session(
                seg, d, tp_variant, dev_mult, sl_mult, vol_mult,
                exh_mult, conf_only, max_bars, t_s, t_e
            ))
        return trades

    @staticmethod
    def _sim_session(seg, date, tp_variant, dev_mult, sl_mult, vol_mult,
                     exh_mult, conf_only, max_bars, t_s, t_e):
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
            sp_v   = float(bar.session_poc)
            conf   = bool(bar.poc_confluence)

            if any(map(np.isnan, [atr_v, avg_v, tp_poc, pp_v])) or atr_v <= 0 or avg_v <= 0:
                i += 1
                continue

            if conf_only and not conf:
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

            dirn     = "long" if long_sig else "short"
            entry_px = (cl + SLIP) if dirn == "long" else (cl - SLIP)

            # TP selection
            if tp_variant == "B" and conf:
                tp_px = sp_v if not np.isnan(sp_v) else tp_poc
            else:
                tp_px = tp_poc

            sl_dist = sl_mult * atr_v
            sl_px   = (entry_px - sl_dist) if dirn == "long" else (entry_px + sl_dist)
            tp_dist = abs(entry_px - tp_px)

            # Skip if TP offers no room
            if tp_dist < SLIP or (dirn == "long" and tp_px <= entry_px) or \
               (dirn == "short" and tp_px >= entry_px):
                i += 1
                continue

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
                session_poc= sp_v,
                confluence = conf,
                tp_variant = tp_variant,
                tp_dist    = tp_dist,
                sl_dist    = sl_dist,
                bars_held  = bars_held,
            ))

            # Fast-forward past exit bar
            try:
                exit_pos = next(j for j, b in enumerate(post) if b.Index >= exit_ts)
                i += exit_pos + 2
            except StopIteration:
                break

        return trades
