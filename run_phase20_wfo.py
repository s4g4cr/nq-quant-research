#!/usr/bin/env python3
"""
Phase 20 — Walk-Forward Validation on Experiment 3 configuration.

Config: POC-A (session_poc) + Condition-3 (simple displacement)
  LONG if price_at_15 < session_poc · SHORT if price_at_15 > session_poc
  Entry: close of 14:59 · TP: session_poc (fixed) · SL: 1.0×ATR (fixed)
  Hard exit: close of 15:59 · risk_pct=1.0% · $100k reset per window

5 anchored windows (expanding train, fixed test blocks).
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
from orb_system.indicators.technical import atr as compute_atr, session_vwap
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.strategy.poc_closing_magnet import run

RESULTS = os.path.join(ROOT, "results")
W = 76

EXP3_KW = dict(poc_variant="A", condition=3, initial_capital=100_000.0, risk_pct=1.0)

WFO_WINDOWS = [
    (1, "2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    (2, "2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    (3, "2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    (4, "2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    (5, "2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _pf(trades):
    v = np.array([t.pnl_net for t in trades])
    w = v[v > 0]; l = v[v <= 0]
    return float(w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")

def _sr(trades):
    if not trades: return 0.0
    by_d = {}
    for t in trades:
        by_d[t.date] = by_d.get(t.date, 0.0) + t.pnl_net
    v = np.array(list(by_d.values()))
    s = float(v.std())
    return float(v.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _wr(trades):
    return sum(1 for t in trades if t.pnl_net > 0) / len(trades) if trades else 0.0

def _ret(trades, cap=100_000.0):
    return sum(t.pnl_net for t in trades) / cap * 100.0 if trades else 0.0

def _mdd(trades, cap=100_000.0):
    c = peak = cap; worst = 0.0
    for t in trades:
        c += t.pnl_net; peak = max(peak, c)
        worst = max(worst, (peak - c) / peak)
    return worst * 100.0

def _fmt(v, d=3):
    if v != v: return "  nan"
    if v == float("inf"): return "  inf"
    return f"{v:.{d}f}"

def _save(trades, fname):
    if not trades: return
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame([vars(t) for t in trades]).to_csv(
        os.path.join(RESULTS, fname), index=False)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 20 — WALK-FORWARD VALIDATION")
    print("  Config: POC-A (session_poc) + Condition-3 (simple displacement)")
    print("=" * W)

    cfg    = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]}  ->  {df_raw.index[-1]}  ({len(df_raw):,} bars)")

    # Build working df (ATR + POC + VWAP) once over the full dataset
    atr1m    = compute_atr(df_raw, 20)
    poc_feat = compute_poc_features(df_raw)
    df_v     = session_vwap(df_raw)

    df = df_raw.copy()
    df["atr"]         = atr1m
    df["session_poc"] = poc_feat["session_poc"]
    df["prev_poc"]    = poc_feat["prev_poc"]
    df["vwap"]        = df_v["vwap"]
    print(f"  Indicators ready.")

    all_oos    = []
    window_rows = []

    for vnum, tr_s, tr_e, te_s, te_e in WFO_WINDOWS:
        df_te    = df.loc[te_s:te_e]
        te_trades = run(df_te, **EXP3_KW)
        all_oos.extend(te_trades)

        n   = len(te_trades)
        wr  = _wr(te_trades) * 100
        pf  = _pf(te_trades)
        sr  = _sr(te_trades)
        ret = _ret(te_trades)
        mdd = _mdd(te_trades)

        window_rows.append({
            "window": vnum, "test_start": te_s, "test_end": te_e,
            "N": n, "WR": round(wr, 1), "PF": round(pf, 3),
            "SR": round(sr, 3), "Return": round(ret, 1), "MaxDD": round(mdd, 1),
        })

        print(f"\n  V{vnum}  Train {tr_s} -> {tr_e}  |  Test {te_s} -> {te_e}")
        print(f"       N={n:3d}  WR={wr:.1f}%  PF={_fmt(pf)}  "
              f"SR={sr:+.3f}  Ret={ret:+.1f}%  MaxDD={mdd:.1f}%")

        _save(te_trades, f"p20_wfo_window_{vnum}.csv")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  WALK-FORWARD SUMMARY TABLE")
    print(f"  {'V':>2} | {'Test period':<22} | {'N':>4} | {'WR%':>5} | "
          f"{'PF':>6} | {'SR':>7} | {'Return%':>8}")
    print(f"  {'-' * 65}")
    for r in window_rows:
        sr_sign = "+" if r["SR"] >= 0 else ""
        print(f"  {r['window']:>2} | {r['test_start']} -> {r['test_end']} | "
              f"{r['N']:>4} | {r['WR']:>4.1f}% | "
              f"{_fmt(r['PF']):>6} | {r['SR']:>+7.3f} | {r['Return']:>+7.1f}%")

    ns  = [r["N"]      for r in window_rows]
    wrs = [r["WR"]     for r in window_rows]
    pfs = [r["PF"]     for r in window_rows if r["PF"] != float("inf")]
    srs = [r["SR"]     for r in window_rows]
    rts = [r["Return"] for r in window_rows]
    print(f"  {'mu':>2} | {'':22} | {int(np.mean(ns)):>4} | "
          f"{np.mean(wrs):>4.1f}% | "
          f"{_fmt(np.mean(pfs)):>6} | {np.mean(srs):>+7.3f} | {np.mean(rts):>+7.1f}%")

    # ── Statistical tests on pooled OOS ───────────────────────────────────────
    print(f"\n{'-' * W}")
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

    pool_pf = _pf(all_oos)
    pool_sr = _sr(all_oos)
    pool_wr = _wr(all_oos) * 100
    pool_ret = _ret(all_oos)

    sr_wins = sum(1 for r in window_rows if r["SR"] > 0)
    pf_wins = sum(1 for r in window_rows if r["PF"] > 1.0)

    print(f"\n  Pooled OOS:  N={len(all_oos)}  WR={pool_wr:.1f}%  "
          f"PF={_fmt(pool_pf)}  SR={pool_sr:+.3f}  Ret={pool_ret:+.1f}%")
    print(f"\n  T-test (H0: mean pnl <= 0):")
    print(f"    t = {t_stat:.4f}   p = {p_v:.4f}")
    print(f"\n  Bootstrap PF (1000 iterations, seed=42):")
    print(f"    mean={b_mean:.4f}   p5={bp5:.4f}   p50={b50:.4f}   p95={b95:.4f}")
    print(f"\n  Windows SR > 0:   {sr_wins} / 5")
    print(f"  Windows PF > 1.0: {pf_wins} / 5")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'-' * W}")
    passes   = []
    failures = []
    if p_v < 0.10:   passes.append(f"p={p_v:.4f} < 0.10")
    else:            failures.append(f"p={p_v:.4f} >= 0.10")
    if bp5 > 0.95:   passes.append(f"bootstrap p5={bp5:.4f} > 0.95")
    else:            failures.append(f"bootstrap p5={bp5:.4f} <= 0.95")
    if sr_wins >= 3: passes.append(f"{sr_wins}/5 windows SR > 0")
    else:            failures.append(f"only {sr_wins}/5 windows SR > 0")

    confirmed = len(failures) == 0
    print(f"  VERDICT: {'EDGE CONFIRMED' if confirmed else 'Edge NOT confirmed'}")
    if passes:   print(f"  Pass:  {' | '.join(passes)}")
    if failures: print(f"  Fail:  {' | '.join(failures)}")
    if confirmed:
        print(f"  -> Proceed to Monte Carlo FTMO sizing.")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    _save(all_oos, "p20_wfo_oos_pooled.csv")

    summary_rows = list(window_rows)
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
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS, "p20_wfo_summary.csv"), index=False)

    print(f"\n  Saved: results/p20_wfo_window_[1-5].csv")
    print(f"         results/p20_wfo_oos_pooled.csv")
    print(f"         results/p20_wfo_summary.csv")
    print(f"\n{'=' * W}")
    print("  WALK-FORWARD COMPLETE")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
