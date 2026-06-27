#!/usr/bin/env python3
"""
Phase 15 — Window 1 failure investigation (H1 2023, B1 config).
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from datetime import time as dt_time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import add_indicators
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.indicators.det_regime_v2 import compute_det_regime_features_v2
from orb_system.strategy.poc_filtered import POCFilteredEngine

F1, F2, F3 = 1.2, 3.0, 1.5
FIXED = dict(tp_frac=0.67, deviation_mult=1.0, exhaustion_mult=1.2,
             volume_mult=1.3, sl_mult=1.0, max_bars=120,
             time_start="09:45", time_end="14:30")

_RTH_S = dt_time(9, 30)
_RTH_E = dt_time(15, 45)
MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

RESULTS_DIR = os.path.join(ROOT, "results")
W = 76

OTHER_WINDOWS = [
    ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2026-06-17"),
]


def _pf(v):
    a = np.array(v); w = a[a > 0]; l = a[a <= 0]
    gw = float(w.sum()) if w.size else 0.0
    gl = float(abs(l.sum())) if l.size else 0.0
    return gw / gl if gl > 0 else float("inf")

def _sr(v):
    a = np.array(v); s = float(a.std())
    return float(a.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _fmt(v, d=3):
    return "  inf" if (v != v or v == float("inf")) else f"{v:.{d}f}"

def _slice(df_ind, start_str, end_str):
    d = np.array(df_ind.index.date)
    s = pd.Timestamp(start_str).date()
    e = pd.Timestamp(end_str).date()
    return df_ind[(d >= s) & (d <= e)]


def _session_context(df_ind, start_str, end_str):
    """Per-session context stats for sessions passing F1 and F3."""
    df_w = _slice(df_ind, start_str, end_str)
    date_arr = np.array(df_w.index.date)
    time_arr = np.array(df_w.index.time)
    u_dates  = np.unique(date_arr)
    all_pos  = np.arange(len(df_w))
    rows = []
    for d in u_dates:
        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]
        rth   = np.array([_RTH_S <= t <= _RTH_E for t in times])
        if rth.sum() == 0:
            continue
        fi = idxs[rth][0]
        prr = float(df_w["prev_range_ratio"].values[fi])
        t5d = float(df_w["trend_5d"].values[fi])
        da  = float(df_w["daily_atr"].values[fi])
        if np.isnan(prr) or np.isnan(t5d) or np.isnan(da):
            continue
        f1_ok = prr < F1
        f3_ok = abs(t5d) < F3
        rows.append({"date": d, "month": d.month,
                     "prev_range_ratio": prr,
                     "abs_trend_5d":  abs(t5d),
                     "daily_atr": da,
                     "f1_pass": f1_ok, "f3_pass": f3_ok,
                     "both_pass": f1_ok and f3_ok})
    return pd.DataFrame(rows)


def _poc_dist(trades):
    return np.array([abs(t.target_poc - t.entry_px) / t.atr_entry for t in trades])


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    cfg = Config()
    df  = load_data(cfg)
    df_ind = add_indicators(df, cfg)
    df_poc = compute_poc_features(df_ind, confluence_threshold=2.0)
    df_ind["prev_poc"]       = df_poc["prev_poc"]
    df_ind["session_poc"]    = df_poc["session_poc"]
    df_ind["poc_confluence"] = df_poc["poc_confluence"]
    df_ind["target_poc"]     = df_poc["target_poc"]
    prr_s, t5d_s, da_s = compute_det_regime_features_v2(df_ind)
    df_ind["prev_range_ratio"] = prr_s
    df_ind["trend_5d"]         = t5d_s
    df_ind["daily_atr"]        = da_s

    # ── Window 1 backtest ─────────────────────────────────────────────────────
    df_w1 = _slice(df_ind, "2023-01-01", "2023-06-30")
    r_w1  = POCFilteredEngine.run(df_w1, filter1_thresh=F1, filter2_thresh=F2,
                                  filter3_thresh=F3, label="w1_invest", **FIXED)
    trades = r_w1.trades

    # ── Other windows for comparison ──────────────────────────────────────────
    other_trades = []
    for s, e in OTHER_WINDOWS:
        df_ow = _slice(df_ind, s, e)
        r_ow  = POCFilteredEngine.run(df_ow, filter1_thresh=F1, filter2_thresh=F2,
                                      filter3_thresh=F3, label=f"ow_{s}", **FIXED)
        other_trades.extend(r_ow.trades)

    print("=" * W)
    print("  WINDOW 1 FAILURE INVESTIGATION  (H1 2023, B1 config)")
    print(f"  F1={F1}  F2={F2}  F3={F3}  tp_frac=0.67")
    print("=" * W)

    # ── 1. Monthly breakdown ──────────────────────────────────────────────────
    print(f"\n  1. MONTHLY BREAKDOWN (Jan–Jun 2023)")
    print(f"  {'Month':>6} | {'N':>4} | {'WR%':>5} | {'PF':>6} | {'SR':>6} | "
          f"{'AvgWin':>7} | {'AvgLoss':>8}")
    print("  " + "-" * (W - 2))

    monthly_rows = []
    for mo in range(1, 7):
        v  = [t for t in trades if t.entry_ts.month == mo]
        pnl = [t.pnl_net for t in v]
        if not v:
            print(f"  {MONTHS[mo]:>6} | {'0':>4} |       |        |        |")
            continue
        pf_m = _pf(pnl); sr_m = _sr(pnl)
        wr_m = sum(1 for x in pnl if x > 0) / len(pnl)
        wins   = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x <= 0]
        aw = float(np.mean(wins))   if wins   else 0.0
        al = float(np.mean(losses)) if losses else 0.0
        print(f"  {MONTHS[mo]:>6} | {len(v):>4} | {wr_m*100:>5.1f}% | "
              f"{_fmt(pf_m):>6} | {sr_m:>6.2f} | ${aw:>6.0f} | ${al:>7.0f}")
        monthly_rows.append({"month": mo, "n": len(v), "wr": wr_m,
                              "pf": pf_m if pf_m != float("inf") else -1, "sr": sr_m})

    tot_pnl = [t.pnl_net for t in trades]
    wr_tot  = sum(1 for x in tot_pnl if x > 0) / max(len(tot_pnl), 1)
    print("  " + "-" * (W - 2))
    print(f"  {'H1 23':>6} | {len(trades):>4} | {wr_tot*100:>5.1f}% | "
          f"{_fmt(_pf(tot_pnl)):>6} | {_sr(tot_pnl):>6.2f}")

    # ── 2. Market context ─────────────────────────────────────────────────────
    ctx_w1 = _session_context(df_ind, "2023-01-01", "2023-06-30")
    ctx_all = pd.concat([
        _session_context(df_ind, s, e) for s, e in
        [("2023-01-01","2023-06-30")] + list(OTHER_WINDOWS)
    ])
    ctx_pass = ctx_w1[ctx_w1["both_pass"]]  # sessions that passed both filters

    print(f"\n  2. MARKET CONTEXT (sessions passing F1 AND F3)")
    print(f"  {'Month':>6} | {'N sess':>6} | {'avg daily_atr':>13} | "
          f"{'avg |trend_5d|':>14} | {'avg prev_rng_ratio':>18}")
    print("  " + "-" * (W - 2))

    ctx_other = ctx_all[ctx_all["both_pass"] & (ctx_all["date"] >= pd.Timestamp("2023-07-01").date())]

    context_save_rows = []
    for mo in range(1, 7):
        sub = ctx_pass[ctx_pass["month"] == mo]
        if sub.empty:
            print(f"  {MONTHS[mo]:>6} | {'0':>6} |")
            continue
        n_s  = len(sub)
        da   = float(sub["daily_atr"].mean())
        at5d = float(sub["abs_trend_5d"].mean())
        prr  = float(sub["prev_range_ratio"].mean())
        print(f"  {MONTHS[mo]:>6} | {n_s:>6} | {da:>13.2f} | {at5d:>14.3f} | {prr:>18.3f}")
        context_save_rows.append({"month": mo, "n_sessions": n_s,
                                  "avg_daily_atr": da, "avg_abs_trend5d": at5d,
                                  "avg_prev_range_ratio": prr})

    # Compare H1 2023 vs rest
    da_w1   = float(ctx_pass["daily_atr"].mean()) if not ctx_pass.empty else np.nan
    da_rest = float(ctx_other["daily_atr"].mean()) if not ctx_other.empty else np.nan
    at_w1   = float(ctx_pass["abs_trend_5d"].mean()) if not ctx_pass.empty else np.nan
    at_rest = float(ctx_other["abs_trend_5d"].mean()) if not ctx_other.empty else np.nan
    pr_w1   = float(ctx_pass["prev_range_ratio"].mean()) if not ctx_pass.empty else np.nan
    pr_rest = float(ctx_other["prev_range_ratio"].mean()) if not ctx_other.empty else np.nan

    print("  " + "-" * (W - 2))
    print(f"  {'W1 avg':>6} | {len(ctx_pass):>6} | {da_w1:>13.2f} | {at_w1:>14.3f} | {pr_w1:>18.3f}")
    print(f"  {'W2-5':>6} | {len(ctx_other):>6} | {da_rest:>13.2f} | {at_rest:>14.3f} | {pr_rest:>18.3f}")

    # ── 3. Exit breakdown ──────────────────────────────────────────────────────
    print(f"\n  3. EXIT BREAKDOWN")
    ex     = r_w1.exit_breakdown()
    bh     = [t.bars_held for t in trades]
    bh_oth = [t.bars_held for t in other_trades]

    print(f"  {'':>10} | {'SL%':>5} | {'TP%':>5} | {'Time%':>6} | {'EOD%':>5} | "
          f"{'MedBars':>7} | {'AvgBars':>7}")
    print("  " + "-" * (W - 2))
    print(f"  {'H1 2023':>10} | {ex['sl']*100:>5.0f}% | {ex['tp']*100:>5.0f}% | "
          f"{ex['timeout']*100:>6.0f}% | {ex['eod']*100:>5.0f}% | "
          f"{np.median(bh):>7.1f} | {np.mean(bh):>7.1f}")

    # Other windows
    ex_oth = r_w1.__class__(other_trades, "oth")  # piggyback POCResults
    ex_oth_bd = ex_oth.exit_breakdown()
    print(f"  {'W2-W5 avg':>10} | {ex_oth_bd['sl']*100:>5.0f}% | "
          f"{ex_oth_bd['tp']*100:>5.0f}% | "
          f"{ex_oth_bd['timeout']*100:>6.0f}% | "
          f"{ex_oth_bd['eod']*100:>5.0f}% | "
          f"{np.median(bh_oth):>7.1f} | {np.mean(bh_oth):>7.1f}")

    # ── 4. POC distance distribution ──────────────────────────────────────────
    poc_w1  = _poc_dist(trades)
    poc_oth = _poc_dist(other_trades)

    print(f"\n  4. POC DISTANCE AT ENTRY (|target_poc - entry_px| / atr_entry)")
    print(f"  {'Window':>10} | {'N':>4} | {'p25':>6} | {'p50':>6} | {'p75':>6} | "
          f"{'mean':>6} | {'>=3.0%':>7} | {'>=5.0%':>7}")
    print("  " + "-" * (W - 2))

    def _dist_row(lbl, d):
        p25, p50, p75 = np.percentile(d, [25, 50, 75])
        ge3 = (d >= 3.0).mean() * 100
        ge5 = (d >= 5.0).mean() * 100
        print(f"  {lbl:>10} | {len(d):>4} | {p25:>6.2f} | {p50:>6.2f} | {p75:>6.2f} | "
              f"{d.mean():>6.2f} | {ge3:>6.0f}% | {ge5:>6.0f}%")
        return {"window": lbl, "n": len(d), "p25": p25, "p50": p50, "p75": p75,
                "mean": d.mean(), "pct_ge3": ge3, "pct_ge5": ge5}

    dist_rows = []
    dist_rows.append(_dist_row("H1 2023", poc_w1))
    for (s, e), lbl in zip(OTHER_WINDOWS, ["H2 2023","H1 2024","H2 2024","2025-26"]):
        r_ow = POCFilteredEngine.run(
            _slice(df_ind, s, e), filter1_thresh=F1, filter2_thresh=F2,
            filter3_thresh=F3, label=f"dist_{s}", **FIXED)
        if r_ow.trades:
            dist_rows.append(_dist_row(lbl, _poc_dist(r_ow.trades)))

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  DIAGNOSIS")
    print(f"  {'='*W}")

    # Check A: signal quality via poc_distance
    poc_w1_med = float(np.median(poc_w1))
    poc_oth_med = float(np.median(poc_oth))
    qual_diff = (poc_w1_med - poc_oth_med) / poc_oth_med * 100

    # Check B: regime — compare daily_atr and trend strength
    da_ratio = (da_w1 - da_rest) / da_rest * 100 if da_rest > 0 else 0
    trend_ratio = (at_w1 - at_rest) / at_rest * 100 if at_rest > 0 else 0

    # Check C: variance — simple confidence interval
    from scipy import stats as sp_stats
    t_stat, p_two = sp_stats.ttest_1samp(tot_pnl, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    print(f"\n  A) Signal quality (poc_distance):")
    print(f"     H1 2023 median: {poc_w1_med:.2f} ATR")
    print(f"     W2-W5 median:   {poc_oth_med:.2f} ATR")
    print(f"     Difference: {qual_diff:+.1f}%  "
          f"-> {'signals are WEAKER in W1' if poc_w1_med < poc_oth_med else 'signals are STRONGER in W1'}")

    print(f"\n  B) Regime characteristics:")
    print(f"     Daily ATR  — H1 2023: {da_w1:.2f}  W2-W5: {da_rest:.2f}  "
          f"Diff: {da_ratio:+.1f}%")
    print(f"     Trend |5d| — H1 2023: {at_w1:.3f}  W2-W5: {at_rest:.3f}  "
          f"Diff: {trend_ratio:+.1f}%")
    if abs(da_ratio) > 20:
        print(f"     -> Volatility is {'HIGHER' if da_ratio > 0 else 'LOWER'} in H1 2023 "
              f"({abs(da_ratio):.0f}% different)")
    else:
        print(f"     -> Volatility is SIMILAR to other windows")

    print(f"\n  C) Variance / sample size:")
    print(f"     n=80  t={t_stat:.3f}  p(mean>0)={p_one:.3f}")
    if p_one > 0.25:
        print(f"     -> NOT distinguishable from zero at any standard threshold")
        print(f"        At p5 of the H1 2023 distribution, this window is consistent")
        print(f"        with random variance around a positive-mean process.")
    elif p_one > 0.10:
        print(f"     -> Marginal; could be random variance given n=80")
    else:
        print(f"     -> Statistically significant at p<0.10; not purely random variance")

    print(f"\n  CONCLUSION:")
    causes = []
    if poc_w1_med < poc_oth_med * 0.85:
        causes.append("A: weaker signal quality (lower poc_distance)")
    if abs(da_ratio) > 20:
        causes.append(f"B: {'higher' if da_ratio > 0 else 'lower'} volatility regime")
    if abs(trend_ratio) > 20:
        causes.append(f"B: {'stronger' if trend_ratio > 0 else 'weaker'} trend days passing F3")
    if p_one > 0.25:
        causes.append("C: consistent with random variance at n=80")

    if not causes:
        causes.append("No single dominant cause identified — combination of factors")
    for c in causes:
        print(f"     -> {c}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_df = r_w1.to_df().copy()
    if not save_df.empty:
        save_df["poc_distance"] = poc_w1
        save_df.to_csv(
            os.path.join(RESULTS_DIR, "p15_window1_investigation.csv"), index=False)
    print(f"\n  Saved: p15_window1_investigation.csv")
    print(f"  {'='*W}")


if __name__ == "__main__":
    main()
