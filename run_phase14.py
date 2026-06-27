#!/usr/bin/env python3
"""
Phase 14: Deterministic Regime Filters for POC Reversion.

Three filters tested individually then combined:
  F1 (OR_RANGE_FILTER)  : prev_day_range / ATR < threshold_range
  F2 (DISTANCE_FILTER)  : |close - target_poc| / ATR >= threshold_distance
  F3 (TREND_FILTER)     : |trend_5d| < threshold_trend

No HMM. All thresholds are fixed constants — no training, no fitting.
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
from orb_system.indicators.det_regime import compute_det_regime_features
from orb_system.strategy.poc_filtered import POCFilteredEngine
from orb_system.strategy.poc_reversion import POCResults, PV, SLIP

SPLIT_DATE = "2024-12-01"

F1_THRESHOLDS = [0.7, 0.8, 1.0, 1.2, 1.5]
F2_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
F3_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]

BASELINE_PF   = 1.078
BASELINE_SR   = 0.348
BASELINE_N    = 472
MIN_N_TEST    = 80
N_BOOT        = 1000

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

W  = 76
W2 = 88
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

    print("Computing deterministic regime features ...")
    prr_s, t5d_s = compute_det_regime_features(df_ind)
    df_ind["prev_range_ratio"] = prr_s
    df_ind["trend_5d"]         = t5d_s

    split = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)
    df_tr = df_ind[date_arr < split]
    df_te = df_ind[date_arr >= split]
    print(f"  Train: {df_tr.index[0].date()} to {df_tr.index[-1].date()} "
          f"({len(np.unique(date_arr[date_arr < split]))} sessions)")
    print(f"  Test:  {df_te.index[0].date()} to {df_te.index[-1].date()} "
          f"({len(np.unique(date_arr[date_arr >= split]))} sessions)")
    return df_ind, df_tr, df_te


def _slice(df_ind, start_str, end_str):
    s = pd.Timestamp(start_str).date()
    e = pd.Timestamp(end_str).date()
    d = np.array(df_ind.index.date)
    return df_ind[(d >= s) & (d <= e)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(v, dec=3):
    if v != v or v == float("inf"):
        return "  inf"
    return f"{v:.{dec}f}"


def _pf(pnl_net):
    arr = np.array(pnl_net)
    w = arr[arr > 0]; l = arr[arr <= 0]
    gw = float(w.sum()) if w.size else 0.0
    gl = float(abs(l.sum())) if l.size else 0.0
    return gw / gl if gl > 0 else float("inf")


def _sr(pnl_net):
    arr = np.array(pnl_net)
    s = float(arr.std())
    return float(arr.mean() / s * math.sqrt(252)) if s > 0 else 0.0


def _yr_metrics(trades, yr):
    v = [t.pnl_net for t in trades if t.entry_ts.year == yr]
    if not v:
        return {"n": 0, "pf": float("nan"), "sr": 0.0}
    return {"n": len(v), "pf": _pf(v), "sr": _sr(v)}


def _run(df, f1=None, f2=None, f3=None, label=""):
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


# ── Pre-Part-A Diagnostic ────────────────────────────────────────────────────

def _diagnostic(df_ind, df_te):
    print(f"\n{'='*W}")
    print("  PRE-PART-A DIAGNOSTIC")
    print(f"{'='*W}")

    date_arr = np.array(df_ind.index.date)
    u_all = np.unique(date_arr)

    # --- Filter 1 coverage ---
    prr_vals = {}
    for d in u_all:
        mask = date_arr == d
        v = df_ind["prev_range_ratio"].values[mask][0]
        if not np.isnan(v):
            prr_vals[d] = v

    split = pd.Timestamp(SPLIT_DATE).date()
    prr_test = {d: v for d, v in prr_vals.items() if d >= split}
    total_te = len(prr_test)

    print(f"\n  Filter 1 (prev_range_ratio) — {len(prr_vals)} total sessions with valid data")
    print(f"  Test sessions: {total_te}")
    print(f"  {'Thresh':>8} | {'N pass':>7} | {'% pass':>7} | {'Yr breakdown (pass%)': <40}")
    by_yr = {yr: sum(1 for d, v in prr_test.items()
                     if d.year == yr) for yr in [2024, 2025, 2026]}
    for th in F1_THRESHOLDS:
        n_pass = sum(1 for v in prr_test.values() if v < th)
        yr_str = "  ".join(
            f"{yr}:{sum(1 for d,v in prr_test.items() if d.year==yr and v<th)}"
            f"/{by_yr[yr]}"
            for yr in [2024, 2025, 2026] if by_yr.get(yr, 0) > 0
        )
        print(f"  {th:>8.1f} | {n_pass:>7} | {n_pass/total_te*100:>6.1f}% | {yr_str}")

    # --- Filter 3 coverage ---
    t5d_vals = {}
    for d in u_all:
        mask = date_arr == d
        v = df_ind["trend_5d"].values[mask][0]
        if not np.isnan(v):
            t5d_vals[d] = v

    t5d_test = {d: abs(v) for d, v in t5d_vals.items() if d >= split}
    total_t5 = len(t5d_test)

    abs_t5d = np.array(list(t5d_test.values()))
    p25, p50, p75 = np.percentile(abs_t5d, [25, 50, 75])
    print(f"\n  Filter 3 (abs trend_5d) — {total_t5} test sessions with valid data")
    print(f"  Distribution: p25={p25:.2f}  p50={p50:.2f}  p75={p75:.2f} "
          f"  >3.0: {(abs_t5d>3.0).mean()*100:.0f}%")
    print(f"  {'Thresh':>8} | {'N pass':>7} | {'% pass':>7}")
    for th in F3_THRESHOLDS:
        n_pass = sum(1 for v in t5d_test.values() if v < th)
        print(f"  {th:>8.1f} | {n_pass:>7} | {n_pass/total_t5*100:>6.1f}%")

    # --- Filter 2 coverage (based on base strategy signals in test period) ---
    print(f"\n  Filter 2 (poc_distance at signal bar) — running base signals on test ...")
    base_r = _run(df_te, label="diag_base")
    poc_dists = []
    for t in base_r.trades:
        d = abs(t.target_poc - t.entry_px) / t.atr_entry
        poc_dists.append(d)
    poc_dists = np.array(poc_dists) if poc_dists else np.array([0.0])
    n_sig = len(poc_dists)
    print(f"  Base signals (test): {n_sig} trades")
    print(f"  poc_dist distribution: p25={np.percentile(poc_dists,25):.2f}  "
          f"p50={np.percentile(poc_dists,50):.2f}  p75={np.percentile(poc_dists,75):.2f}")
    print(f"  {'Thresh':>8} | {'N pass':>7} | {'% pass':>7}")
    for th in F2_THRESHOLDS:
        n_pass = int((poc_dists >= th).sum())
        print(f"  {th:>8.1f} | {n_pass:>7} | {n_pass/max(n_sig,1)*100:>6.1f}%")

    # --- Overlap analysis F1 x F3 ---
    print(f"\n  Overlap analysis F1 x F3 (baseline thresholds 1.0, 1.5, 2.0):")
    print(f"  {'F1_th':>6} | {'F3_th':>6} | {'F1 only':>8} | {'F3 only':>8} | "
          f"{'Both':>8} | {'Overlap%':>9} | {'Redundant?':>10}")
    common = {d for d in prr_test if d in t5d_test}
    for f1_th, f3_th in [(1.0, 2.0), (0.8, 1.5), (1.2, 2.5)]:
        f1_pass = {d for d in common if prr_test[d] < f1_th}
        f3_pass = {d for d in common if t5d_test[d] < f3_th}
        both    = f1_pass & f3_pass
        overlap = len(both) / max(min(len(f1_pass), len(f3_pass)), 1) * 100
        redund  = "YES" if overlap > 80 else "no"
        print(f"  {f1_th:>6.1f} | {f3_th:>6.1f} | {len(f1_pass):>8} | "
              f"{len(f3_pass):>8} | {len(both):>8} | {overlap:>8.0f}% | {redund:>10}")

    print(f"\n{'='*W}")
    return base_r


# ── Part A — Individual filters ───────────────────────────────────────────────

def _row_str(label, m, y25, y26, n_suffix=""):
    n25 = y25["n"]; n26 = y26["n"]
    return (f"  {label:<8} | {m['n']:>6}{n_suffix} | {m['trades_per_day']:>4.2f} | "
            f"{m['wr']*100:>5.1f}% | {_fmt(m['pf']):>6} | {m['sharpe']:>6.3f} | "
            f"{n25:>4}/{_fmt(y25['pf']):>6}/{y25['sr']:>5.2f} | "
            f"{n26:>4}/{_fmt(y26['pf']):>6}/{y26['sr']:>5.2f}")


def _part_a_header(filter_name):
    print(f"\n  {'threshold':>10} | {'N test':>6} | {'TPD':>4} | "
          f"{'WR%':>5} | {'PF tr':>6} | {'PF te':>6} | {'SR te':>6} | "
          f"{'2025 N/PF/SR':>18} | {'2026 N/PF/SR':>18}")
    print("  " + "-" * (W2 - 2))


def _part_a_row(label, r_tr, r_te, df_te):
    m_te = r_te.metrics()
    m_tr = r_tr.metrics()
    y25  = _yr_metrics(r_te.trades, 2025)
    y26  = _yr_metrics(r_te.trades, 2026)
    n_flag = "*" if m_te["n"] < MIN_N_TEST else " "
    print(f"  {label:>10} | {m_te['n']:>5}{n_flag} | {m_te['trades_per_day']:>4.2f} | "
          f"{m_te['wr']*100:>5.1f}% | {_fmt(m_tr['pf']):>6} | {_fmt(m_te['pf']):>6} | "
          f"{m_te['sharpe']:>6.3f} | "
          f"{y25['n']:>3}/{_fmt(y25['pf']):>6}/{y25['sr']:>5.2f} | "
          f"{y26['n']:>3}/{_fmt(y26['pf']):>6}/{y26['sr']:>5.2f}")
    return m_te


def _part_a(df_tr, df_te):
    print(f"\n{'='*W2}")
    print("  PART A — INDIVIDUAL FILTERS")
    print(f"  Baseline (no filter): PF={BASELINE_PF}  SR={BASELINE_SR}  N={BASELINE_N}")
    print(f"{'='*W2}")

    results_f1 = {}; results_f2 = {}; results_f3 = {}

    # Filter 1
    print(f"\n  Filter 1: OR_RANGE_FILTER (prev_range_ratio < threshold)")
    _part_a_header("F1")
    for th in F1_THRESHOLDS:
        r_tr = _run(df_tr, f1=th, label=f"p14_f1_{th}_tr")
        r_te = _run(df_te, f1=th, label=f"p14_f1_{th}_te")
        m_te = _part_a_row(f"{th:.1f}", r_tr, r_te, df_te)
        results_f1[th] = (r_te, m_te)
        if not r_te.to_df().empty:
            r_te.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p14_filter1_{th}.csv"), index=False)

    # Filter 2
    print(f"\n  Filter 2: DISTANCE_FILTER (poc_distance >= threshold)")
    _part_a_header("F2")
    for th in F2_THRESHOLDS:
        r_tr = _run(df_tr, f2=th, label=f"p14_f2_{th}_tr")
        r_te = _run(df_te, f2=th, label=f"p14_f2_{th}_te")
        m_te = _part_a_row(f"{th:.1f}", r_tr, r_te, df_te)
        results_f2[th] = (r_te, m_te)
        if not r_te.to_df().empty:
            r_te.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p14_filter2_{th}.csv"), index=False)

    # Filter 3
    print(f"\n  Filter 3: TREND_FILTER (abs(trend_5d) < threshold)")
    _part_a_header("F3")
    for th in F3_THRESHOLDS:
        r_tr = _run(df_tr, f3=th, label=f"p14_f3_{th}_tr")
        r_te = _run(df_te, f3=th, label=f"p14_f3_{th}_te")
        m_te = _part_a_row(f"{th:.1f}", r_tr, r_te, df_te)
        results_f3[th] = (r_te, m_te)
        if not r_te.to_df().empty:
            r_te.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p14_filter3_{th}.csv"), index=False)

    return results_f1, results_f2, results_f3


def _select_qualifying(results_dict, df_tr, filter_name):
    """Find threshold with best SR test that beats baseline on all criteria."""
    qualifiers = []
    for th, (r_te, m) in results_dict.items():
        y26 = _yr_metrics(r_te.trades, 2026)
        passes = (
            m["pf"] > BASELINE_PF
            and m["sharpe"] > BASELINE_SR
            and m["n"] >= MIN_N_TEST
            and y26["sr"] > -1.59  # better than baseline 2026 SR
        )
        qualifiers.append((th, m["sharpe"], m["pf"], m["n"], passes))

    best = [(th, sr) for th, sr, pf, n, ok in qualifiers if ok]
    if best:
        best_th = max(best, key=lambda x: x[1])[0]
        r_te, m  = results_dict[best_th]
        print(f"  {filter_name}: QUALIFIES at threshold={best_th}  "
              f"PF={_fmt(m['pf'])}  SR={m['sharpe']:.3f}  N={m['n']}")
        return best_th, results_dict[best_th][0]
    else:
        print(f"  {filter_name}: DOES NOT QUALIFY at any threshold")
        return None, None


# ── Part B — Combinations ────────────────────────────────────────────────────

def _part_b(df_tr, df_te, qualifying):
    """qualifying: dict of {filter_id: (best_thresh, r_te)}"""
    print(f"\n{'='*W2}")
    print("  PART B — FILTER COMBINATIONS")
    print(f"{'='*W2}")

    if not qualifying:
        print("  No qualifying filters from Part A. Skipping Part B.")
        return None, None

    ids = sorted(qualifying.keys())
    print(f"  Qualifying filters: {ids}")

    combos = []
    # Singles already shown in Part A; include here for comparison
    for fid in ids:
        th, _ = qualifying[fid]
        combos.append(([fid], {fid: th}))

    # 2-way
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            combos.append(([ids[i], ids[j]], {ids[i]: qualifying[ids[i]][0],
                                              ids[j]: qualifying[ids[j]][0]}))
    # 3-way
    if len(ids) == 3:
        combos.append((ids, {fid: qualifying[fid][0] for fid in ids}))

    print(f"\n  {'Combination':<18} | {'N test':>6} | {'TPD':>4} | "
          f"{'WR%':>5} | {'PF tr':>6} | {'PF te':>6} | {'SR te':>6} | "
          f"{'2025 SR':>7} | {'2026 SR':>7}")
    print("  " + "-" * (W2 - 2))

    best_combo = None
    best_sr    = -999.0
    combo_results = {}

    for used_ids, thresholds in combos:
        f1 = thresholds.get("F1")
        f2 = thresholds.get("F2")
        f3 = thresholds.get("F3")
        lbl = "+".join(f"{k}={v:.1f}" for k, v in sorted(thresholds.items()))

        r_tr = _run(df_tr, f1=f1, f2=f2, f3=f3, label=f"p14_combo_{lbl}_tr")
        r_te = _run(df_te, f1=f1, f2=f2, f3=f3, label=f"p14_combo_{lbl}_te")
        m_tr = r_tr.metrics()
        m_te = r_te.metrics()
        y25  = _yr_metrics(r_te.trades, 2025)
        y26  = _yr_metrics(r_te.trades, 2026)
        n_fl = "*" if m_te["n"] < MIN_N_TEST else " "

        print(f"  {lbl:<18} | {m_te['n']:>5}{n_fl} | {m_te['trades_per_day']:>4.2f} | "
              f"{m_te['wr']*100:>5.1f}% | {_fmt(m_tr['pf']):>6} | "
              f"{_fmt(m_te['pf']):>6} | {m_te['sharpe']:>6.3f} | "
              f"{y25['sr']:>7.3f} | {y26['sr']:>7.3f}")

        if not r_te.to_df().empty:
            r_te.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p14_combination_{lbl}.csv"), index=False)

        combo_results[lbl] = (r_te, m_te, f1, f2, f3)
        if (m_te["pf"] > BASELINE_PF and m_te["n"] >= MIN_N_TEST
                and m_te["sharpe"] > best_sr):
            best_sr    = m_te["sharpe"]
            best_combo = (lbl, f1, f2, f3)

    if best_combo:
        lbl, f1, f2, f3 = best_combo
        print(f"\n  Best combination for Part C: {lbl}")
    else:
        print(f"\n  No combination beats baseline PF={BASELINE_PF}. "
              f"Stopping — Part C will not run.")

    return best_combo, combo_results


# ── Part C — Walk-forward ─────────────────────────────────────────────────────

def _part_c(df_ind, best_combo):
    if best_combo is None:
        print(f"\n{'='*W}")
        print("  PART C — SKIPPED (no combination beats baseline)")
        print(f"{'='*W}")
        return []

    lbl, f1_th, f2_th, f3_th = best_combo
    print(f"\n{'='*W}")
    print(f"  PART C — WALK-FORWARD VALIDATION")
    print(f"  Config: {lbl} | tp_frac=0.67 | No HMM — thresholds are fixed constants")
    print(f"{'='*W}")

    wfo_results = []
    all_pnl_net = []

    for vn, (tr_s, tr_e, te_s, te_e) in enumerate(WINDOWS_WFO, start=1):
        df_te_v = _slice(df_ind, te_s, te_e)
        r = _run(df_te_v, f1=f1_th, f2=f2_th, f3=f3_th, label=f"p14_wfo_v{vn}")
        m = r.metrics()
        y25 = _yr_metrics(r.trades, 2025)
        y26 = _yr_metrics(r.trades, 2026)

        n_flag = "*" if m["n"] < 30 else " "
        if m["n"] >= 30:
            all_pnl_net.extend([t.pnl_net for t in r.trades])

        ex   = r.exit_breakdown()
        bh   = [t.bars_held for t in r.trades]
        yr5_str = (f"  2025: n={y25['n']} PF={_fmt(y25['pf'])} SR={y25['sr']:.2f}  "
                   f"2026: n={y26['n']} PF={_fmt(y26['pf'])} SR={y26['sr']:.2f}"
                   if vn == 5 else "")

        print(f"\n  {'='*W}")
        print(f"  WINDOW {vn} | Train: {tr_s} -> {tr_e} | Test: {te_s} -> {te_e}")
        print(f"  {'='*W}")
        print(f"  Trades: {m['n']}{n_flag}  TPD: {m['trades_per_day']:.2f}")
        print(f"  WR: {m['wr']*100:.1f}%  PF: {_fmt(m['pf'])}  SR: {m['sharpe']:.3f}  "
              f"Return: {m['ret_pct']:.2f}%  MaxDD: {m['max_dd_pct']:.2f}%")
        if r.trades:
            print(f"  Exits: SL={ex['sl']*100:.0f}%  TP={ex['tp']*100:.0f}%  "
                  f"Time={ex['timeout']*100:.0f}%  EOD={ex['eod']*100:.0f}%")
            print(f"  AvgWin=${m['avg_win']:.0f}  AvgLoss=${m['avg_loss']:.0f}  "
                  f"MaxConsec: {_max_consec_loss(r.trades)}")
            if bh:
                print(f"  Median bars to exit: {np.median(bh):.0f}")
        if yr5_str:
            print(yr5_str)

        if not r.to_df().empty:
            r.to_df().to_csv(
                os.path.join(RESULTS_DIR, f"p14_wfo_window_{vn}.csv"), index=False)

        wfo_results.append((vn, tr_s, tr_e, te_s, te_e, r, m))

    # WFO summary table
    print(f"\n{'='*W2}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'='*W2}")
    print(f"  {'V':>1} | {'Test period':<22} | {'PF':>6} | {'SR':>6} | "
          f"{'N':>4} | {'WR%':>5} | {'TPD':>4}")
    print("  " + "-" * (W2 - 2))

    pf_gt1 = sr_gt0 = 0
    for vn, tr_s, tr_e, te_s, te_e, r, m in wfo_results:
        fl = "*" if m["n"] < 30 else " "
        if m["pf"] > 1.0:   pf_gt1 += 1
        if m["sharpe"] > 0: sr_gt0 += 1
        print(f"  {vn:>1} | {te_s} -> {te_e} | {_fmt(m['pf']):>6} | "
              f"{m['sharpe']:>6.3f} | {m['n']:>3}{fl} | "
              f"{m['wr']*100:>4.0f}% | {m['trades_per_day']:>4.2f}")

    valid_m = [m for *_, m in wfo_results if m["n"] >= 30]
    if valid_m:
        mu_pf = np.mean([min(m["pf"], 5.0) for m in valid_m])
        mu_sr = np.mean([m["sharpe"] for m in valid_m])
        mu_n  = np.mean([m["n"] for m in valid_m])
        print("  " + "-" * (W2 - 2))
        print(f"  mu|  (mean of valid windows)           | {mu_pf:>6.3f} | "
              f"{mu_sr:>6.3f} | {mu_n:>4.0f}")

    print(f"{'='*W2}")
    print(f"  Windows with PF > 1.0: {pf_gt1}/5")
    print(f"  Windows with SR > 0.0: {sr_gt0}/5")

    # Statistical tests
    _stat_tests(all_pnl_net)

    return wfo_results, pf_gt1, sr_gt0, all_pnl_net


def _stat_tests(all_pnl_net):
    n = len(all_pnl_net)
    print(f"\n  Statistical tests (OOS pooled — {n} trades):")
    if n < 10:
        print("  Insufficient trades.")
        return 1.0, 0.0

    arr = np.array(all_pnl_net)
    t_stat, p_two = stats.ttest_1samp(arr, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    np.random.seed(42)
    pf_boots = []
    for _ in range(N_BOOT):
        s  = np.random.choice(arr, size=n, replace=True)
        w  = s[s > 0]; l = s[s <= 0]
        gw = float(w.sum()) if w.size else 0.0
        gl = float(abs(l.sum())) if l.size else 0.0
        if gl > 0:
            pf_boots.append(gw / gl)
    pb   = np.array(pf_boots)
    p5   = np.percentile(pb, 5)
    p95  = np.percentile(pb, 95)

    print(f"  T-test: t={t_stat:.3f}  p={p_one:.4f} "
          f"-> {'Significant p<0.10' if p_one<0.10 else 'Not significant p>0.10'}")
    print(f"  Bootstrap PF: mean={pb.mean():.3f}  p5={p5:.3f}  p95={p95:.3f} "
          f"-> {'Strong (p5>1.0)' if p5>1.0 else 'Moderate (p5>0.95)' if p5>0.95 else 'Insufficient (p5<0.95)'}")
    return p_one, p5


def _verdict(best_combo, wfo_output):
    print(f"\n{'='*W}")
    print("  FINAL VERDICT")
    print(f"{'='*W}")

    if best_combo is None:
        print("  VERDICT: POC reversion edge not confirmable with deterministic filters.")
        print("  No combination in Part B beat baseline PF=1.078 with N>=80.")
        print("  The signal requires regime identification that static rules cannot capture.")
        print("  Consider closing this research cycle.")
        print(f"{'='*W}")
        return

    wfo_results, pf_gt1, sr_gt0, all_pnl = wfo_output
    lbl = best_combo[0]

    p_val, p5 = _stat_tests(all_pnl) if len(all_pnl) >= 10 else (1.0, 0.0)

    strong = p_val < 0.10 and p5 > 0.95
    ok_win = pf_gt1 >= 3

    if ok_win and strong:
        print(f"  Config: {lbl}")
        print(f"  VERDICT: Edge CONFIRMED with deterministic filters.")
        print(f"  {pf_gt1}/5 windows PF>1.0  p={p_val:.4f}  bootstrap p5={p5:.3f}")
        print(f"  Recommend Phase 15: Monte Carlo FTMO sizing.")
    elif ok_win:
        print(f"  Config: {lbl}")
        print(f"  VERDICT: Directionally positive ({pf_gt1}/5 windows PF>1.0)")
        print(f"  but insufficient statistical power (p={p_val:.3f}  p5={p5:.3f}).")
        print(f"  Recommend extending to longer dataset before FTMO preparation.")
    else:
        print(f"  Config: {lbl}")
        print(f"  VERDICT: POC reversion edge not confirmable with deterministic filters.")
        print(f"  Only {pf_gt1}/5 windows PF>1.0 — regime-specific or noise.")
        print(f"  The signal requires regime identification that static rules cannot capture.")
        print(f"  Consider closing this research cycle.")
    print(f"{'='*W}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 14: DETERMINISTIC REGIME FILTERS FOR POC REVERSION")
    print("  No HMM. Filters: OR_RANGE (F1) | DISTANCE (F2) | TREND (F3)")
    print("=" * W)

    df_ind, df_tr, df_te = _load_all()

    # Diagnostic
    _diagnostic(df_ind, df_te)

    # Part A
    results_f1, results_f2, results_f3 = _part_a(df_tr, df_te)

    # Selection
    print(f"\n  PART A SELECTION:")
    f1_best, f1_r = _select_qualifying(results_f1, df_tr, "F1")
    f2_best, f2_r = _select_qualifying(results_f2, df_tr, "F2")
    f3_best, f3_r = _select_qualifying(results_f3, df_tr, "F3")

    qualifying = {}
    if f1_best is not None: qualifying["F1"] = (f1_best, f1_r)
    if f2_best is not None: qualifying["F2"] = (f2_best, f2_r)
    if f3_best is not None: qualifying["F3"] = (f3_best, f3_r)

    # Part B
    best_combo, combo_results = _part_b(df_tr, df_te, qualifying)

    # Part C
    if best_combo is not None:
        wfo_output = _part_c(df_ind, best_combo)
        # Save pooled OOS
        all_oos = []
        for vn, *_, r, m in wfo_output[0]:
            dv = r.to_df()
            if not dv.empty:
                dv.insert(0, "window", vn)
                all_oos.append(dv)
        if all_oos:
            pd.concat(all_oos, ignore_index=True).to_csv(
                os.path.join(RESULTS_DIR, "p14_oos_pooled.csv"), index=False)
            print(f"\n  Saved: p14_oos_pooled.csv and per-window CSVs")
        _verdict(best_combo, wfo_output)
    else:
        _verdict(None, None)


if __name__ == "__main__":
    main()
