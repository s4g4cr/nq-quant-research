#!/usr/bin/env python3
"""
Phase 15 Part C — B1 walk-forward.
Config: F1=1.2 + F2=3.0 + F3=1.5 + tp_frac=0.67
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import add_indicators
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.indicators.det_regime_v2 import compute_det_regime_features_v2
from orb_system.strategy.poc_filtered import POCFilteredEngine

F1 = 1.2
F2 = 3.0
F3 = 1.5

FIXED = dict(
    tp_frac=0.67, deviation_mult=1.0, exhaustion_mult=1.2,
    volume_mult=1.3, sl_mult=1.0, max_bars=120,
    time_start="09:45", time_end="14:30",
)

WINDOWS = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-17"),
]

MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

RESULTS_DIR = os.path.join(ROOT, "results")
W = 76
N_BOOT = 1000


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

def _max_cl(trades):
    best = cur = 0
    for t in trades:
        if t.pnl_net < 0: cur += 1; best = max(best, cur)
        else: cur = 0
    return best


def main():
    print("=" * W)
    print("  PHASE 15 PART C — B1 WALK-FORWARD")
    print(f"  Config: F1={F1}  F2={F2}  F3={F3}  tp_frac=0.67")
    print("=" * W)

    cfg    = Config()
    df     = load_data(cfg)
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

    date_arr = np.array(df_ind.index.date)
    all_pnl  = []
    wfo_rows = []
    oos_dfs  = []

    for vn, (tr_s, tr_e, te_s, te_e) in enumerate(WINDOWS, start=1):
        s = pd.Timestamp(te_s).date()
        e = pd.Timestamp(te_e).date()
        df_te = df_ind[(date_arr >= s) & (date_arr <= e)]

        r  = POCFilteredEngine.run(
            df_te, filter1_thresh=F1, filter2_thresh=F2, filter3_thresh=F3,
            label=f"p15_B1_v{vn}", **FIXED)
        m  = r.metrics()
        ex = r.exit_breakdown()
        bh = [t.bars_held for t in r.trades]

        y25 = {yr: [t.pnl_net for t in r.trades if t.entry_ts.year == yr]
               for yr in [2025, 2026]}

        fl = "*" if m["n"] < 30 else " "
        if m["n"] >= 30:
            all_pnl.extend([t.pnl_net for t in r.trades])

        print(f"\n  {'='*W}")
        print(f"  W{vn}: {te_s} -> {te_e}")
        print(f"  {'='*W}")
        print(f"  Trades: {m['n']}{fl}  TPD: {m['trades_per_day']:.2f}")
        print(f"  WR: {m['wr']*100:.1f}%  PF: {_fmt(m['pf'])}  SR: {m['sharpe']:.3f}  "
              f"Return: {m['ret_pct']:.2f}%  MaxDD: {m['max_dd_pct']:.2f}%")
        if r.trades:
            print(f"  Exits: SL={ex['sl']*100:.0f}%  TP={ex['tp']*100:.0f}%  "
                  f"Time={ex['timeout']*100:.0f}%  EOD={ex['eod']*100:.0f}%")
            print(f"  AvgWin=${m['avg_win']:.0f}  AvgLoss=${m['avg_loss']:.0f}  "
                  f"MaxConsec: {_max_cl(r.trades)}  MedianBars: {np.median(bh):.0f}")
        if vn == 5:
            for yr in [2025, 2026]:
                v = y25[yr]
                if v:
                    print(f"  {yr}: n={len(v)}  PF={_fmt(_pf(v))}  SR={_sr(v):.2f}")

        if not r.to_df().empty:
            df_out = r.to_df()
            df_out.to_csv(
                os.path.join(RESULTS_DIR, f"p15_partC_B1_wfo_window_{vn}.csv"), index=False)
            df_out.insert(0, "window", vn)
            oos_dfs.append(df_out)

        wfo_rows.append((vn, te_s, te_e, m))

    # Summary table
    print(f"\n  {'='*W}")
    print("  WALK-FORWARD SUMMARY  (B1: F1=1.2 + F2=3.0 + F3=1.5)")
    print(f"  {'='*W}")
    print(f"  {'V':>1} | {'Test period':<22} | {'PF':>6} | {'SR':>6} | "
          f"{'N':>4} | {'WR%':>5} | {'TPD':>4}")
    print("  " + "-" * (W - 4))

    pf_gt1 = sr_gt0 = 0
    for vn, te_s, te_e, m in wfo_rows:
        fl = "*" if m["n"] < 30 else " "
        if m["pf"] > 1.0:   pf_gt1 += 1
        if m["sharpe"] > 0: sr_gt0 += 1
        print(f"  {vn:>1} | {te_s} -> {te_e} | {_fmt(m['pf']):>6} | "
              f"{m['sharpe']:>6.3f} | {m['n']:>3}{fl} | "
              f"{m['wr']*100:>4.0f}% | {m['trades_per_day']:>4.2f}")

    valid = [m for *_, m in wfo_rows if m["n"] >= 30]
    if valid:
        mu_pf = np.mean([min(m["pf"], 5.0) for m in valid])
        mu_sr = np.mean([m["sharpe"] for m in valid])
        print("  " + "-" * (W - 4))
        print(f"  mu| (valid windows)                     | "
              f"{mu_pf:>6.3f} | {mu_sr:>6.3f}")

    print(f"\n  Windows PF > 1.0 : {pf_gt1}/5")
    print(f"  Windows SR > 0.0 : {sr_gt0}/5")

    # Statistical tests
    n = len(all_pnl)
    print(f"\n  {'='*W}")
    print(f"  STATISTICAL TESTS  (pooled OOS: {n} trades)")
    print(f"  {'='*W}")

    arr = np.array(all_pnl)
    t_stat, p_two = stats.ttest_1samp(arr, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    np.random.seed(42)
    pf_boots = []
    for _ in range(N_BOOT):
        s  = np.random.choice(arr, size=n, replace=True)
        w  = s[s > 0]; l = s[s <= 0]
        gw = float(w.sum()) if w.size else 0.0
        gl = float(abs(l.sum())) if l.size else 0.0
        if gl > 0: pf_boots.append(gw / gl)
    pb   = np.array(pf_boots)
    mn_pf = float(pb.mean())
    p5    = float(np.percentile(pb, 5))
    p95   = float(np.percentile(pb, 95))

    print(f"  T-test (H1: mean > 0):  t={t_stat:.3f}  p={p_one:.4f}  "
          f"-> {'Significant p<0.10' if p_one < 0.10 else 'Not significant p>0.10'}")
    print(f"  Bootstrap PF ({N_BOOT} iter): mean={mn_pf:.3f}  p5={p5:.3f}  p95={p95:.3f}  "
          f"-> {'Strong (p5>1.0)' if p5>1.0 else 'Moderate (p5>0.95)' if p5>0.95 else 'Insufficient (p5<0.95)'}")

    # 2026 monthly breakdown (Window 5)
    w5_trades = [t for vn, *_ in [(r,) for *_, r in [wfo_rows[-1]]]
                 for _ in [0]] if False else None
    # Direct access
    s5 = pd.Timestamp(WINDOWS[4][2]).date()
    e5 = pd.Timestamp(WINDOWS[4][3]).date()
    df_te5 = df_ind[(date_arr >= s5) & (date_arr <= e5)]
    r5 = POCFilteredEngine.run(
        df_te5, filter1_thresh=F1, filter2_thresh=F2, filter3_thresh=F3,
        label="p15_B1_v5_monthly", **FIXED)

    trades_26 = [t for t in r5.trades if t.entry_ts.year == 2026]
    if trades_26:
        by_mo: dict = {}
        for t in trades_26:
            by_mo.setdefault(t.entry_ts.month, []).append(t.pnl_net)

        print(f"\n  {'='*W}")
        print("  2026 MONTHLY BREAKDOWN (Window 5, B1 config)")
        print(f"  {'='*W}")
        print(f"  {'Month':>7} | {'N':>5} | {'WR%':>5} | {'PF':>6} | {'SR':>6} | "
              f"{'AvgWin':>7} | {'AvgLoss':>8}")
        print("  " + "-" * (W - 4))

        monthly_rows = []
        total_26 = []
        for mo in sorted(by_mo.keys()):
            v  = by_mo[mo]
            total_26.extend(v)
            pf_m = _pf(v); sr_m = _sr(v)
            wr_m = sum(1 for x in v if x > 0) / len(v)
            wins = [x for x in v if x > 0]; losses = [x for x in v if x <= 0]
            aw = float(np.mean(wins))   if wins   else 0.0
            al = float(np.mean(losses)) if losses else 0.0
            print(f"  {MONTHS[mo]:>3} 26 | {len(v):>5} | {wr_m*100:>5.1f}% | "
                  f"{_fmt(pf_m):>6} | {sr_m:>6.2f} | ${aw:>6.0f} | ${al:>7.0f}")
            monthly_rows.append({"month": mo, "n": len(v), "wr": wr_m,
                                  "pf": pf_m if pf_m != float("inf") else -1, "sr": sr_m})

        print("  " + "-" * (W - 4))
        print(f"  {'2026':>7} | {len(total_26):>5} | "
              f"{sum(1 for x in total_26 if x>0)/len(total_26)*100:>5.1f}% | "
              f"{_fmt(_pf(total_26)):>6} | {_sr(total_26):>6.2f}")

        pd.DataFrame(monthly_rows).to_csv(
            os.path.join(RESULTS_DIR, "p15_partC_B1_2026_monthly.csv"), index=False)

    # Save pooled OOS
    if oos_dfs:
        pd.concat(oos_dfs, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "p15_partC_B1_oos_pooled.csv"), index=False)

    print(f"\n  Saved: p15_partC_B1_wfo_window_[1-5].csv  p15_partC_B1_oos_pooled.csv")

    # Verdict
    print(f"\n  {'='*W}")
    confirmed = p_one < 0.10 and p5 > 0.95
    print(f"  VERDICT: {'Edge CONFIRMED' if confirmed else 'Insufficient statistical power'}")
    print(f"  p={p_one:.4f} (target <0.10)  p5={p5:.3f} (target >0.95)  "
          f"PF>1.0: {pf_gt1}/5")
    if confirmed:
        print(f"  Proceed to Phase 16: Monte Carlo FTMO sizing.")
    else:
        gap = max(0.10 - p_one, 0) + max(0.95 - p5, 0)
        print(f"  Gap to thresholds: p gap={max(0, p_one-0.10):.4f}  "
              f"p5 gap={max(0, 0.95-p5):.4f}")
    print(f"  {'='*W}")


if __name__ == "__main__":
    main()
