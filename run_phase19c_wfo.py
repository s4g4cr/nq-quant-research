#!/usr/bin/env python3
"""
Phase 19C — Walk-Forward Validation on Exp 12 configuration.

Config: |r1| > 0.3% AND sign(r1) == sign(r16) · inverted direction
Signal: SHORT if r1>0, LONG if r1<0
Entry: close of 15:29 · Exit: close of 15:59 · No SL/TP
5 anchored windows · $100k capital reset per window
"""
import math
import os
import sys
from datetime import time as dt_time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import atr as compute_atr
from orb_system.strategy.intraday_reversal import detect_sessions, run
from orb_system.strategy.intraday_momentum import _ENTRY_T

INITIAL_CAP = 100_000.0
RESULTS_DIR = os.path.join(ROOT, "results")
W           = 72

WFO_WINDOWS = [
    (1, "2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    (2, "2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    (3, "2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    (4, "2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    (5, "2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]

EXP12_KW = dict(
    entry_bar_time  = _ENTRY_T,
    r1_threshold    = 0.003,
    r16_agreement   = True,
    high_vol_only   = False,
    rv_median_dict  = None,
    vol_filter      = False,
    vol_median_dict = None,
    initial_capital = INITIAL_CAP,
    risk_pct        = 1.0,
)


# ── stat helpers ──────────────────────────────────────────────────────────────

def _pf(trades):
    v = np.array([t.pnl_net for t in trades])
    w = v[v > 0]; l = v[v <= 0]
    return float(w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")

def _sr(trades):
    if not trades: return 0.0
    by_date: dict = {}
    for t in trades:
        d = str(t.date)
        by_date[d] = by_date.get(d, 0.0) + t.pnl_net
    v = np.array(list(by_date.values()))
    s = float(v.std())
    return float(v.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _wr(trades):
    return sum(1 for t in trades if t.pnl_net > 0) / len(trades) if trades else 0.0

def _max_dd(trades):
    cap = peak = INITIAL_CAP; worst = 0.0
    for t in trades:
        cap += t.pnl_net; peak = max(peak, cap)
        worst = max(worst, (peak - cap) / peak)
    return worst

def _ret(trades):
    return sum(t.pnl_net for t in trades) / INITIAL_CAP * 100.0

def _fmt(v, d=3):
    if v != v: return "  nan"
    if v == float("inf"): return "  inf"
    return f"{v:.{d}f}"

def _save_csv(trades, fname):
    if not trades: return
    rows = [{"date": t.date, "direction": t.direction,
             "r1": t.r1, "r17": t.r17, "r16": t.r16,
             "entry_ts": t.entry_ts, "entry_price": t.entry_price,
             "exit_ts": t.exit_ts, "exit_price": t.exit_price,
             "n_contracts": t.n_contracts, "atr_at_entry": t.atr_at_entry,
             "pnl_pts": t.pnl_pts, "pnl_net": t.pnl_net}
            for t in trades]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, fname), index=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 19C — WALK-FORWARD VALIDATION")
    print("  Config: |r1|>0.3% + r16_agree · SHORT if r1>0 · LONG if r1<0")
    print("=" * W)

    cfg   = Config()
    df    = load_data(cfg)
    print(f"  Data: {df.index[0]}  →  {df.index[-1]}  ({len(df):,} bars)")

    atr1m = compute_atr(df, 20)
    print(f"  ATR(20) computed.")

    all_oos    = []
    window_rows = []

    for vnum, tr_s, tr_e, te_s, te_e in WFO_WINDOWS:
        df_te = df.loc[te_s:te_e]
        a1_te = atr1m[df_te.index]
        si_te = detect_sessions(df_te, a1_te)

        te_trades = run(df_te, si_te, a1_te, **EXP12_KW)
        all_oos.extend(te_trades)

        n   = len(te_trades)
        wr  = _wr(te_trades) * 100
        pf  = _pf(te_trades)
        sr  = _sr(te_trades)
        ret = _ret(te_trades)
        mdd = _max_dd(te_trades) * 100

        window_rows.append({
            "window": vnum, "test_start": te_s, "test_end": te_e,
            "N": n, "WR": round(wr, 1), "PF": round(pf, 3),
            "SR": round(sr, 3), "Return": round(ret, 1), "MaxDD": round(mdd, 1),
        })

        print(f"\n  V{vnum}  Train {tr_s}→{tr_e} | Test {te_s}→{te_e}")
        print(f"       N={n:3d}  WR={wr:.1f}%  PF={_fmt(pf)}  "
              f"SR={sr:.3f}  Ret={ret:.1f}%  MaxDD={mdd:.1f}%")

        _save_csv(te_trades, f"p19c_wfo_window_{vnum}.csv")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  WALK-FORWARD SUMMARY TABLE")
    print(f"  {'V':>2} | {'Test period':<22} | {'N':>4} | {'WR%':>5} | "
          f"{'PF':>6} | {'SR':>6} | {'Return%':>8}")
    print(f"  {'─'*65}")
    for r in window_rows:
        print(f"  {r['window']:>2} | {r['test_start']}→{r['test_end']} | "
              f"{r['N']:>4} | {r['WR']:>4.1f}% | "
              f"{_fmt(r['PF']):>6} | {r['SR']:>6.3f} | {r['Return']:>7.1f}%")

    # Mean row
    ns   = [r["N"]      for r in window_rows]
    wrs  = [r["WR"]     for r in window_rows]
    pfs  = [r["PF"]     for r in window_rows if r["PF"] != float("inf")]
    srs  = [r["SR"]     for r in window_rows]
    rets = [r["Return"] for r in window_rows]
    print(f"  {'μ':>2} | {'':22} | {sum(ns)//len(ns):>4} | "
          f"{np.mean(wrs):>4.1f}% | {_fmt(np.mean(pfs)):>6} | "
          f"{np.mean(srs):>6.3f} | {np.mean(rets):>7.1f}%")

    # ── Statistical tests on pooled OOS ───────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  STATISTICAL TESTS — POOLED OOS")
    print(f"  Total OOS trades: {len(all_oos)}")

    pnl_arr     = np.array([t.pnl_net for t in all_oos])
    t_stat, p_v = stats.ttest_1samp(pnl_arr, 0.0, alternative="greater")

    rng     = np.random.default_rng(42)
    boot_pf = []
    for _ in range(1000):
        s  = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
        gw = s[s > 0].sum(); gl = abs(s[s <= 0].sum())
        boot_pf.append(gw / gl if gl > 0 else float("inf"))
    finite = [x for x in boot_pf if x != float("inf")]
    bp5    = float(np.percentile(finite, 5))
    b50    = float(np.percentile(finite, 50))
    b95    = float(np.percentile(finite, 95))
    b_mean = float(np.mean(finite))

    sr_wins = sum(1 for r in window_rows if r["SR"] > 0)
    pf_wins = sum(1 for r in window_rows if r["PF"] > 1.0)

    print(f"\n  T-test (H0: mean pnl ≤ 0):")
    print(f"    t = {t_stat:.4f}   p = {p_v:.4f}")
    print(f"\n  Bootstrap PF (1000 iterations, seed=42):")
    print(f"    mean = {b_mean:.4f}   p5 = {bp5:.4f}   p50 = {b50:.4f}   p95 = {b95:.4f}")
    print(f"\n  Windows SR > 0:  {sr_wins} / 5")
    print(f"  Windows PF > 1.0: {pf_wins} / 5")

    # Pooled PF directly
    pool_pf = _pf(all_oos)
    pool_sr = _sr(all_oos)
    pool_wr = _wr(all_oos) * 100
    pool_ret = _ret(all_oos)
    print(f"\n  Pooled OOS:  N={len(all_oos)}  WR={pool_wr:.1f}%  "
          f"PF={_fmt(pool_pf)}  SR={pool_sr:.3f}  Ret={pool_ret:.1f}%")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    passes   = []
    failures = []

    if p_v < 0.10:    passes.append(f"p={p_v:.4f} < 0.10")
    else:             failures.append(f"p={p_v:.4f} ≥ 0.10")
    if bp5 > 0.95:    passes.append(f"bootstrap p5={bp5:.4f} > 0.95")
    else:             failures.append(f"bootstrap p5={bp5:.4f} ≤ 0.95")
    if sr_wins >= 3:  passes.append(f"{sr_wins}/5 windows SR>0")
    else:             failures.append(f"only {sr_wins}/5 windows SR>0")

    confirmed = len(failures) == 0
    print(f"  VERDICT: {'EDGE CONFIRMED' if confirmed else 'Edge NOT confirmed'}")
    if passes:    print(f"  Pass:  {' · '.join(passes)}")
    if failures:  print(f"  Fail:  {' · '.join(failures)}")
    if confirmed: print(f"  → Proceed to Monte Carlo FTMO sizing.")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    _save_csv(all_oos, "p19c_wfo_oos_pooled.csv")

    summary_rows = []
    for r in window_rows:
        summary_rows.append(r)
    summary_rows.append({
        "window": "pooled", "test_start": "all", "test_end": "all",
        "N": len(all_oos), "WR": round(pool_wr, 1),
        "PF": round(pool_pf, 3), "SR": round(pool_sr, 3),
        "Return": round(pool_ret, 1), "MaxDD": None,
    })
    summary_rows.append({
        "window": "stats", "test_start": "t_stat", "test_end": str(round(t_stat, 4)),
        "N": None, "WR": None, "PF": round(b_mean, 4),
        "SR": round(p_v, 4), "Return": round(bp5, 4), "MaxDD": round(b95, 4),
    })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "p19c_wfo_summary.csv"), index=False)

    print(f"\n  Saved: results/p19c_wfo_window_[1-5].csv")
    print(f"         results/p19c_wfo_oos_pooled.csv")
    print(f"         results/p19c_wfo_summary.csv")

    print(f"\n{'='*W}")
    print("  WALK-FORWARD COMPLETE")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
