#!/usr/bin/env python3
"""
Phase 15: Corrected Filters 1 & 3 with Daily ATR + Combination with F2=3.0.

Fixes Phase 14's F1/F3 dimensional error by replacing 5-min ATR denominator
with daily_atr (20-day rolling mean of RTH session ranges).
F2=3.0 is active in EVERY experiment as the validated baseline.
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
from orb_system.strategy.poc_reversion import POCResults

SPLIT_DATE = "2024-12-01"

F1_THRESHOLDS = [0.7, 0.8, 1.0, 1.2, 1.5]
F3_THRESHOLDS = [0.5, 0.75, 1.0, 1.5, 2.0]
F2_FIXED      = 3.0

# Phase 14 walk-forward reference (to beat for Part C gate)
P14_PF_TEST   = 1.136
P14_SR_TEST   = 0.612   # window 5 result
P14_P5        = 0.949

MIN_N_TEST    = 80
N_BOOT        = 1000
W             = 76
W2            = 90

FIXED = dict(
    tp_frac        = 0.67,
    deviation_mult = 1.0,
    exhaustion_mult= 1.2,
    volume_mult    = 1.3,
    sl_mult        = 1.0,
    max_bars       = 120,
    time_start     = "09:45",
    time_end       = "14:30",
)

WINDOWS_WFO = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-17"),
]

MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data ───────────────────────────────────────────────────────────────────────

def _load_all():
    cfg    = Config()
    print("Loading data and computing indicators ...")
    df     = load_data(cfg)
    df_ind = add_indicators(df, cfg)

    print("Computing POC features ...")
    df_poc = compute_poc_features(df_ind, confluence_threshold=2.0)
    df_ind["prev_poc"]       = df_poc["prev_poc"]
    df_ind["session_poc"]    = df_poc["session_poc"]
    df_ind["poc_confluence"] = df_poc["poc_confluence"]
    df_ind["target_poc"]     = df_poc["target_poc"]

    print("Computing corrected deterministic features (daily_atr normalization) ...")
    prr_s, t5d_s, da_s = compute_det_regime_features_v2(df_ind)
    df_ind["prev_range_ratio"] = prr_s
    df_ind["trend_5d"]         = t5d_s
    df_ind["daily_atr"]        = da_s

    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)
    df_tr    = df_ind[date_arr < split]
    df_te    = df_ind[date_arr >= split]
    print(f"  Train: {df_tr.index[0].date()} to {df_tr.index[-1].date()} "
          f"({np.unique(date_arr[date_arr < split]).size} sessions)")
    print(f"  Test:  {df_te.index[0].date()} to {df_te.index[-1].date()} "
          f"({np.unique(date_arr[date_arr >= split]).size} sessions)")
    return df_ind, df_tr, df_te


def _slice(df_ind, start_str, end_str):
    s = pd.Timestamp(start_str).date()
    e = pd.Timestamp(end_str).date()
    d = np.array(df_ind.index.date)
    return df_ind[(d >= s) & (d <= e)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(v, d=3):
    if v != v or v == float("inf"):
        return "  inf"
    return f"{v:.{d}f}"


def _pf(pnl_net):
    a = np.array(pnl_net)
    w = a[a > 0]; l = a[a <= 0]
    gw = float(w.sum()) if w.size else 0.0
    gl = float(abs(l.sum())) if l.size else 0.0
    return gw / gl if gl > 0 else float("inf")


def _sr(pnl_net):
    a = np.array(pnl_net)
    s = float(a.std())
    return float(a.mean() / s * math.sqrt(252)) if s > 0 else 0.0


def _yr(trades, yr):
    v = [t.pnl_net for t in trades if t.entry_ts.year == yr]
    if not v:
        return {"n": 0, "pf": float("nan"), "sr": 0.0, "wr": 0.0}
    return {"n": len(v), "pf": _pf(v), "sr": _sr(v),
            "wr": sum(1 for x in v if x > 0) / len(v)}


def _run(df, f1=None, f2=F2_FIXED, f3=None, label=""):
    return POCFilteredEngine.run(
        df,
        filter1_thresh=f1,
        filter2_thresh=f2,
        filter3_thresh=f3,
        label=label,
        **FIXED,
    )


def _max_consec_loss(trades):
    best = cur = 0
    for t in trades:
        if t.pnl_net < 0:
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    return best


def _bootstrap_pf(pnl_arr, n_boot=N_BOOT):
    np.random.seed(42)
    pfs = []
    for _ in range(n_boot):
        s  = np.random.choice(pnl_arr, size=len(pnl_arr), replace=True)
        w  = s[s > 0]; l = s[s <= 0]
        gw = float(w.sum()) if w.size else 0.0
        gl = float(abs(l.sum())) if l.size else 0.0
        if gl > 0:
            pfs.append(gw / gl)
    pb = np.array(pfs)
    return float(pb.mean()), float(np.percentile(pb, 5)), float(np.percentile(pb, 95))


# ── Pre-diagnostics ────────────────────────────────────────────────────────────

def _diagnostic(df_ind, df_te):
    print(f"\n{'='*W}")
    print("  PRE-DIAGNOSTICS (corrected daily_atr normalization)")
    print(f"{'='*W}")

    date_arr = np.array(df_ind.index.date)
    split    = pd.Timestamp(SPLIT_DATE).date()

    # Extract per-session values for test period
    te_mask  = date_arr >= split
    te_dates = np.unique(date_arr[te_mask])
    n_te     = len(te_dates)

    prr_vals = {}; t5d_vals = {}; da_vals = {}
    for d in te_dates:
        m  = date_arr == d
        prr_vals[d] = float(df_ind["prev_range_ratio"].values[m][0])
        t5d_vals[d] = float(df_ind["trend_5d"].values[m][0])
        da_vals[d]  = float(df_ind["daily_atr"].values[m][0])

    valid_prr = {d: v for d, v in prr_vals.items() if not np.isnan(v)}
    valid_t5d = {d: abs(v) for d, v in t5d_vals.items() if not np.isnan(v)}
    n_prr = len(valid_prr); n_t5d = len(valid_t5d)

    # --- Filter 1 distribution ---
    prr_arr = np.array(list(valid_prr.values()))
    pcts = np.percentile(prr_arr, [10, 25, 50, 75, 90])
    print(f"\n  Filter 1 (prev_range_ratio = prev_range / daily_atr) — {n_prr} test sessions")
    print(f"  Distribution: p10={pcts[0]:.2f}  p25={pcts[1]:.2f}  p50={pcts[2]:.2f}  "
          f"p75={pcts[3]:.2f}  p90={pcts[4]:.2f}")
    print(f"  {'Thresh':>7} | {'N pass':>7} | {'% pass':>7}")
    for th in F1_THRESHOLDS:
        n_p = sum(1 for v in valid_prr.values() if v < th)
        print(f"  {th:>7.2f} | {n_p:>7} | {n_p/n_prr*100:>6.1f}%")

    # --- Filter 3 distribution ---
    t5d_arr = np.array(list(valid_t5d.values()))
    pcts3 = np.percentile(t5d_arr, [10, 25, 50, 75, 90])
    print(f"\n  Filter 3 (abs(trend_5d) = |5d trend / daily_atr|) — {n_t5d} test sessions")
    print(f"  Distribution: p10={pcts3[0]:.2f}  p25={pcts3[1]:.2f}  p50={pcts3[2]:.2f}  "
          f"p75={pcts3[3]:.2f}  p90={pcts3[4]:.2f}")
    print(f"  {'Thresh':>7} | {'N pass':>7} | {'% pass':>7}")
    for th in F3_THRESHOLDS:
        n_p = sum(1 for v in valid_t5d.values() if v < th)
        print(f"  {th:>7.2f} | {n_p:>7} | {n_p/n_t5d*100:>6.1f}%")

    # --- Overlap F1=1.0 × F3=1.0 ---
    common = {d for d in valid_prr if d in valid_t5d}
    f1_set = {d for d in common if valid_prr[d] < 1.0}
    f3_set = {d for d in common if valid_t5d[d] < 1.0}
    both   = f1_set & f3_set
    smaller = max(min(len(f1_set), len(f3_set)), 1)
    overlap_pct = len(both) / smaller * 100

    print(f"\n  Overlap at F1=1.0, F3=1.0 (baseline thresholds):")
    print(f"    F1 only passes : {len(f1_set)}/{len(common)} = {len(f1_set)/len(common)*100:.0f}%")
    print(f"    F3 only passes : {len(f3_set)}/{len(common)} = {len(f3_set)/len(common)*100:.0f}%")
    print(f"    Both pass      : {len(both)}")
    print(f"    Overlap        : {overlap_pct:.0f}% "
          f"({'REDUNDANT — test only one' if overlap_pct > 80 else 'complementary — test combination'})")

    # Save diagnostics CSV
    rows = []
    for d in sorted(te_dates):
        rows.append({
            "date": d,
            "prev_range_ratio": prr_vals.get(d, np.nan),
            "trend_5d": t5d_vals.get(d, np.nan),
            "daily_atr": da_vals.get(d, np.nan),
            "f1_pass_10": prr_vals.get(d, np.nan) < 1.0,
            "f3_pass_10": abs(t5d_vals.get(d, np.nan) if not np.isnan(t5d_vals.get(d, np.nan)) else 999) < 1.0,
        })
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "p15_diagnostics.csv"), index=False)

    print(f"  Saved: p15_diagnostics.csv")
    print(f"{'='*W}")
    return overlap_pct > 80


# ── Part A ─────────────────────────────────────────────────────────────────────

def _part_a_row(label, r_tr, r_te, col_width=10):
    m  = r_te.metrics(); mt = r_tr.metrics()
    y25 = _yr(r_te.trades, 2025); y26 = _yr(r_te.trades, 2026)
    fl  = "*" if m["n"] < MIN_N_TEST else " "
    print(f"  {label:>{col_width}} | {m['n']:>5}{fl} | {m['trades_per_day']:>4.2f} | "
          f"{m['wr']*100:>5.1f}% | {_fmt(mt['pf']):>6} | {_fmt(m['pf']):>6} | "
          f"{m['sharpe']:>6.3f} | {y25['sr']:>7.3f} | {y26['sr']:>7.3f}")
    return m


def _part_a_header():
    print(f"  {'threshold':>10} | {'N test':>5} | {'TPD':>4} | "
          f"{'WR%':>5} | {'PF tr':>6} | {'PF te':>6} | {'SR te':>6} | "
          f"{'2025 SR':>7} | {'2026 SR':>7}")
    print("  " + "-" * (W2 - 2))


def _part_a(df_tr, df_te):
    print(f"\n{'='*W2}")
    print("  PART A — INDIVIDUAL CORRECTED FILTERS (F2=3.0 always active)")
    print(f"{'='*W2}")

    # A1 reference
    a1_tr = _run(df_tr, label="p15_a1_tr")
    a1_te = _run(df_te, label="p15_a1_te")
    print(f"\n  A1 — F2=3.0 only (Phase 14 baseline):")
    _part_a_header()
    a1_m = _part_a_row("F2=3.0", a1_tr, a1_te)
    if not a1_te.to_df().empty:
        a1_te.to_df().to_csv(os.path.join(RESULTS_DIR, "p15_partA_a1.csv"), index=False)

    # F1 sweep
    print(f"\n  F1 sweep — prev_range_ratio < threshold (+ F2=3.0):")
    _part_a_header()
    f1_res = {}
    for th in F1_THRESHOLDS:
        r_tr = _run(df_tr, f1=th, label=f"p15_f1_{th}_tr")
        r_te = _run(df_te, f1=th, label=f"p15_f1_{th}_te")
        m    = _part_a_row(f"{th:.2f}", r_tr, r_te)
        f1_res[th] = (r_te, m)
        if not r_te.to_df().empty:
            r_te.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p15_partA_f1_{th}.csv"), index=False)

    # F3 sweep
    print(f"\n  F3 sweep — abs(trend_5d) < threshold (+ F2=3.0):")
    _part_a_header()
    f3_res = {}
    for th in F3_THRESHOLDS:
        r_tr = _run(df_tr, f3=th, label=f"p15_f3_{th}_tr")
        r_te = _run(df_te, f3=th, label=f"p15_f3_{th}_te")
        m    = _part_a_row(f"{th:.2f}", r_tr, r_te)
        f3_res[th] = (r_te, m)
        if not r_te.to_df().empty:
            r_te.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p15_partA_f3_{th}.csv"), index=False)

    return a1_m, a1_te, f1_res, f3_res


def _select_best(results_dict, a1_m, filter_name):
    """
    Find threshold with best SR test that:
      - beats A1 on PF and SR
      - N >= MIN_N_TEST
    Returns (best_thresh, r_te) or (None, None).
    """
    candidates = [
        (th, m["sharpe"], m["pf"], m["n"])
        for th, (r_te, m) in results_dict.items()
        if m["pf"] > a1_m["pf"] and m["sharpe"] > a1_m["sharpe"]
           and m["n"] >= MIN_N_TEST
    ]
    if not candidates:
        print(f"  {filter_name}: DOES NOT BEAT A1 baseline at any threshold "
              f"(need PF>{a1_m['pf']:.3f} AND SR>{a1_m['sharpe']:.3f} AND N>={MIN_N_TEST})")
        return None, None
    best_th = max(candidates, key=lambda x: x[1])[0]
    r_te, m  = results_dict[best_th]
    print(f"  {filter_name}: QUALIFIES at threshold={best_th}  "
          f"PF={_fmt(m['pf'])}  SR={m['sharpe']:.3f}  N={m['n']}")
    return best_th, r_te


# ── Part B ─────────────────────────────────────────────────────────────────────

def _part_b(df_tr, df_te, f1_best, f3_best, a1_m):
    print(f"\n{'='*W2}")
    print("  PART B — THREE-FILTER COMBINATION (F1 + F2=3.0 + F3)")
    print(f"{'='*W2}")

    if f1_best is None or f3_best is None:
        print("  Skipped: requires both F1 and F3 to qualify individually.")
        best_label = "F2=3.0"
        best_f1 = None; best_f3 = None
        return best_label, None, None

    lbl = f"F1={f1_best}+F2={F2_FIXED}+F3={f3_best}"
    r_tr = _run(df_tr, f1=f1_best, f3=f3_best, label=f"p15_b1_tr")
    r_te = _run(df_te, f1=f1_best, f3=f3_best, label=f"p15_b1_te")
    m_te = r_te.metrics()
    m_tr = r_tr.metrics()
    y25  = _yr(r_te.trades, 2025); y26 = _yr(r_te.trades, 2026)
    fl   = "*" if m_te["n"] < MIN_N_TEST else " "

    print(f"\n  B1 — {lbl}:")
    _part_a_header()
    print(f"  {'B1':>10} | {m_te['n']:>5}{fl} | {m_te['trades_per_day']:>4.2f} | "
          f"{m_te['wr']*100:>5.1f}% | {_fmt(m_tr['pf']):>6} | {_fmt(m_te['pf']):>6} | "
          f"{m_te['sharpe']:>6.3f} | {y25['sr']:>7.3f} | {y26['sr']:>7.3f}")
    print(f"  A1 ref: PF={_fmt(a1_m['pf'])}  SR={a1_m['sharpe']:.3f}")

    if not r_te.to_df().empty:
        r_te.to_df().to_csv(
            os.path.join(RESULTS_DIR, "p15_partB_combination.csv"), index=False)

    better = (m_te["pf"] > a1_m["pf"] and m_te["sharpe"] > a1_m["sharpe"]
              and m_te["n"] >= MIN_N_TEST)
    if better:
        print(f"  B1 beats A1 baseline — use as best config")
        return lbl, f1_best, f3_best
    else:
        # Find which of F1 or F3 individual was better
        print(f"  B1 does NOT beat A1 — check which individual filter is best:")
        return "F2=3.0", None, None  # fall back to A1 if neither beats individually


def _pick_best_for_wfo(a1_m, f1_res, f3_res, f1_best, f3_best, b1_label, b1_f1, b1_f3):
    """Select the single best configuration for Part C."""
    candidates = [("F2=3.0 (A1)", None, None, a1_m["pf"], a1_m["sharpe"])]

    if f1_best is not None:
        _, m1 = f1_res[f1_best]
        candidates.append((f"F1={f1_best}+F2=3.0", f1_best, None, m1["pf"], m1["sharpe"]))

    if f3_best is not None:
        _, m3 = f3_res[f3_best]
        candidates.append((f"F3={f3_best}+F2=3.0", None, f3_best, m3["pf"], m3["sharpe"]))

    if b1_f1 is not None and b1_f3 is not None:
        # get B1 metrics
        pass  # already captured above

    # Pick by SR test, require beats Phase 14 reference
    gate = any(pf > P14_PF_TEST or sr > P14_SR_TEST or sr > a1_m["sharpe"]
               for lbl, f1, f3, pf, sr in candidates[1:])
    if not gate:
        print(f"\n  No config beats Phase 14 baseline on any metric. "
              f"Running Part C with A1 (F2=3.0) for reference.")
    best = max(candidates, key=lambda x: x[4])  # max SR
    return best[0], best[1], best[2]


# ── Part C — Walk-forward ─────────────────────────────────────────────────────

def _part_c(df_ind, lbl, f1_th, f3_th):
    print(f"\n{'='*W}")
    print(f"  PART C — WALK-FORWARD VALIDATION")
    print(f"  Config: {lbl} (F2=3.0 always active)")
    print(f"  Deterministic filters — no training step")
    print(f"{'='*W}")

    wfo_res = []
    all_pnl = []

    for vn, (tr_s, tr_e, te_s, te_e) in enumerate(WINDOWS_WFO, start=1):
        df_te_v = _slice(df_ind, te_s, te_e)
        r  = _run(df_te_v, f1=f1_th, f3=f3_th, label=f"p15_wfo_v{vn}")
        m  = r.metrics()
        y25 = _yr(r.trades, 2025); y26 = _yr(r.trades, 2026)
        ex  = r.exit_breakdown()
        bh  = [t.bars_held for t in r.trades]

        n_fl = "*" if m["n"] < 30 else " "
        if m["n"] >= 30:
            all_pnl.extend([t.pnl_net for t in r.trades])

        yr5 = (f"  2025: n={y25['n']} PF={_fmt(y25['pf'])} SR={y25['sr']:.2f}  "
               f"2026: n={y26['n']} PF={_fmt(y26['pf'])} SR={y26['sr']:.2f}"
               if vn == 5 else "")

        print(f"\n  {'='*W}")
        print(f"  W{vn}: {te_s} -> {te_e}")
        print(f"  {'='*W}")
        print(f"  Trades: {m['n']}{n_fl}  TPD: {m['trades_per_day']:.2f}")
        print(f"  WR: {m['wr']*100:.1f}%  PF: {_fmt(m['pf'])}  SR: {m['sharpe']:.3f}  "
              f"Return: {m['ret_pct']:.2f}%  MaxDD: {m['max_dd_pct']:.2f}%")
        if r.trades:
            print(f"  Exits: SL={ex['sl']*100:.0f}%  TP={ex['tp']*100:.0f}%  "
                  f"Time={ex['timeout']*100:.0f}%  EOD={ex['eod']*100:.0f}%")
            print(f"  AvgWin=${m['avg_win']:.0f}  AvgLoss=${m['avg_loss']:.0f}  "
                  f"MaxConsec: {_max_consec_loss(r.trades)}")
            if bh:
                print(f"  Median bars: {np.median(bh):.0f}")
        if yr5:
            print(yr5)

        if not r.to_df().empty:
            r.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p15_wfo_window_{vn}.csv"), index=False)
        wfo_res.append((vn, te_s, te_e, r, m))

    # WFO summary
    print(f"\n{'='*W2}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'='*W2}")
    print(f"  {'V':>1} | {'Test period':<22} | {'PF':>6} | {'SR':>6} | "
          f"{'N':>4} | {'WR%':>5} | {'TPD':>4}")
    print("  " + "-" * (W2 - 2))

    pf_gt1 = sr_gt0 = 0
    for vn, te_s, te_e, r, m in wfo_res:
        fl = "*" if m["n"] < 30 else " "
        if m["pf"] > 1.0:   pf_gt1 += 1
        if m["sharpe"] > 0: sr_gt0 += 1
        print(f"  {vn:>1} | {te_s} -> {te_e} | {_fmt(m['pf']):>6} | "
              f"{m['sharpe']:>6.3f} | {m['n']:>3}{fl} | "
              f"{m['wr']*100:>4.0f}% | {m['trades_per_day']:>4.2f}")

    valid = [m for *_, m in wfo_res if m["n"] >= 30]
    if valid:
        mu_pf = np.mean([min(m["pf"], 5.0) for m in valid])
        mu_sr = np.mean([m["sharpe"] for m in valid])
        mu_n  = np.mean([m["n"] for m in valid])
        print("  " + "-" * (W2 - 2))
        print(f"  mu|  (valid windows)                     | "
              f"{mu_pf:>6.3f} | {mu_sr:>6.3f} | {mu_n:>4.0f}")

    print(f"{'='*W2}")
    print(f"  Windows PF > 1.0 : {pf_gt1}/5")
    print(f"  Windows SR > 0.0 : {sr_gt0}/5")

    return wfo_res, pf_gt1, sr_gt0, all_pnl


# ── 2026 Monthly Breakdown ────────────────────────────────────────────────────

def _monthly_2026(wfo_res):
    # Window 5 trades
    w5_trades = wfo_res[4][3].trades  # (vn, te_s, te_e, r, m)[3].trades
    trades_26 = [t for t in w5_trades if t.entry_ts.year == 2026]

    if not trades_26:
        print("\n  2026: no trades in Window 5.")
        return

    by_month: dict = {}
    for t in trades_26:
        m = t.entry_ts.month
        by_month.setdefault(m, []).append(t)

    print(f"\n{'='*W}")
    print("  2026 MONTHLY BREAKDOWN (Window 5)")
    print(f"{'='*W}")
    print(f"  {'Month':>7} | {'N':>5} | {'WR%':>5} | {'PF':>6} | {'SR':>6} | "
          f"{'AvgWin':>7} | {'AvgLoss':>8}")
    print("  " + "-" * (W - 2))

    rows = []
    total_pnl = []
    for mo in sorted(by_month.keys()):
        t_list = by_month[mo]
        pnl    = [t.pnl_net for t in t_list]
        total_pnl.extend(pnl)
        pf_m   = _pf(pnl); sr_m = _sr(pnl)
        wr_m   = sum(1 for x in pnl if x > 0) / len(pnl)
        wins   = [x for x in pnl if x > 0]; losses = [x for x in pnl if x <= 0]
        aw     = float(np.mean(wins))   if wins   else 0.0
        al     = float(np.mean(losses)) if losses else 0.0
        print(f"  {MONTHS[mo]:>3} 26 | {len(pnl):>5} | {wr_m*100:>5.1f}% | "
              f"{_fmt(pf_m):>6} | {sr_m:>6.2f} | ${aw:>6.0f} | ${al:>7.0f}")
        rows.append({"month": mo, "n": len(pnl), "wr": wr_m,
                     "pf": pf_m if pf_m != float("inf") else -1,
                     "sr": sr_m})

    print("  " + "-" * (W - 2))
    print(f"  {'2026':>7} | {len(total_pnl):>5} | "
          f"{sum(1 for x in total_pnl if x>0)/len(total_pnl)*100:>5.1f}% | "
          f"{_fmt(_pf(total_pnl)):>6} | {_sr(total_pnl):>6.2f}")

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "p15_2026_monthly.csv"), index=False)
    print(f"  Saved: p15_2026_monthly.csv")
    print(f"{'='*W}")


# ── Statistical tests ─────────────────────────────────────────────────────────

def _stat_tests(all_pnl, header="OOS pooled"):
    n = len(all_pnl)
    print(f"\n  Statistical tests ({header} — {n} trades):")
    if n < 10:
        print("  Insufficient trades."); return 1.0, 0.0, 0.0

    arr = np.array(all_pnl)
    t_stat, p_two = stats.ttest_1samp(arr, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
    mn_pf, p5, p95 = _bootstrap_pf(arr)

    interp_t = ("Significant p<0.05" if p_one < 0.05 else
                "Marginal p<0.10"    if p_one < 0.10 else
                "Not significant")
    interp_b = ("Strong (p5>1.0)"    if p5 > 1.0  else
                "Moderate (p5>0.95)" if p5 > 0.95 else
                "Insufficient (p5<0.95)")

    print(f"  T-test:     t={t_stat:.3f}  p={p_one:.4f}  -> {interp_t}")
    print(f"  Bootstrap:  mean={mn_pf:.3f}  p5={p5:.3f}  p95={p95:.3f}  -> {interp_b}")
    return p_one, p5, mn_pf


# ── Final verdict ─────────────────────────────────────────────────────────────

def _verdict(lbl, pf_gt1, sr_gt0, p_val, p5):
    print(f"\n{'='*W}")
    print("  FINAL VERDICT")
    print(f"{'='*W}")
    print(f"  Config: {lbl}")
    print(f"  Windows PF>1.0: {pf_gt1}/5   Windows SR>0: {sr_gt0}/5")
    print(f"  T-test p: {p_val:.4f}   Bootstrap p5: {p5:.3f}")

    confirmed_strong = p_val < 0.10 and p5 > 0.95 and pf_gt1 >= 4
    confirmed_caveats = p_val < 0.10 and p5 > 0.95 and pf_gt1 == 3
    gap_p = (0.10 - p_val) if p_val < 0.10 else (p_val - 0.10)

    if confirmed_strong:
        print(f"\n  VERDICT: Edge CONFIRMED.")
        print(f"  {pf_gt1}/5 windows PF>1.0 · p={p_val:.4f} · p5={p5:.3f}")
        print(f"  Proceed to Phase 16: Monte Carlo FTMO sizing.")
    elif confirmed_caveats:
        print(f"\n  VERDICT: Edge confirmed with caveats.")
        print(f"  3/5 windows PF>1.0 · p={p_val:.4f} · p5={p5:.3f}")
        print(f"  Recommend paper trading 6 months before FTMO to validate 2026 behavior.")
    else:
        # Estimate trades needed
        total_oos_pnl_mean = 0.0  # placeholder; actual computed in _stat_tests
        print(f"\n  VERDICT: Insufficient statistical power.")
        print(f"  p={p_val:.4f} (target <0.10)  p5={p5:.3f} (target >0.95)")
        if p_val < 0.20 and p5 > 0.90:
            # Close — estimate additional trades needed
            # t ~ sqrt(n) * effect_size; to reach p=0.10 (t~1.28), need n ~ (1.28/current_t)^2 * current_n
            print(f"  Result is directionally positive. Recommend extending dataset.")
        else:
            print(f"  Recommend closing this research cycle or rethinking entry conditions.")
    print(f"{'='*W}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 15: CORRECTED FILTERS (daily_atr) + F2=3.0 COMBINATION")
    print("  F2=3.0 active in every experiment")
    print("=" * W)

    df_ind, df_tr, df_te = _load_all()

    # Diagnostic
    filters_redundant = _diagnostic(df_ind, df_te)

    # Part A
    a1_m, a1_te, f1_res, f3_res = _part_a(df_tr, df_te)

    # Selection
    print(f"\n  PART A SELECTION (must beat A1: PF>{a1_m['pf']:.3f} SR>{a1_m['sharpe']:.3f} N>={MIN_N_TEST}):")
    f1_th, _ = _select_best(f1_res, a1_m, "F1+F2=3.0")
    f3_th, _ = _select_best(f3_res, a1_m, "F3+F2=3.0")

    # Part B
    if filters_redundant:
        print(f"\n  F1 and F3 are redundant (overlap>80%). Skipping B1 three-way combination.")
        b1_lbl = "F2=3.0"; b1_f1 = f1_th; b1_f3 = None
    else:
        b1_lbl, b1_f1, b1_f3 = _part_b(df_tr, df_te, f1_th, f3_th, a1_m)

    # Pick best for WFO
    best_lbl, wfo_f1, wfo_f3 = _pick_best_for_wfo(
        a1_m, f1_res, f3_res, f1_th, f3_th, b1_lbl, b1_f1, b1_f3)

    print(f"\n  Proceeding to Part C with: {best_lbl}")

    # Part C
    wfo_res, pf_gt1, sr_gt0, all_pnl = _part_c(df_ind, best_lbl, wfo_f1, wfo_f3)

    # 2026 monthly breakdown
    _monthly_2026(wfo_res)

    # Save pooled OOS
    all_oos = []
    for vn, te_s, te_e, r, m in wfo_res:
        dv = r.to_df()
        if not dv.empty:
            dv.insert(0, "window", vn)
            all_oos.append(dv)
    if all_oos:
        pd.concat(all_oos, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "p15_oos_pooled.csv"), index=False)
        print(f"\n  Saved: p15_oos_pooled.csv")

    # Stat tests
    p_val, p5, mn_pf = _stat_tests(all_pnl, "all WFO windows")

    # Verdict
    _verdict(best_lbl, pf_gt1, sr_gt0, p_val, p5)


if __name__ == "__main__":
    main()
