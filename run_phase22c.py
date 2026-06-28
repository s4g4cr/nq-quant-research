#!/usr/bin/env python3
"""
Phase 22C — Bearish Spike Fade: Frequency Optimization.

Context: Phase 22B best config (TP-D, P>0.60) achieved PF=1.648, SR=+0.629
but only N=22 test trades — below the 60-trade WFO threshold.
Two approaches: relax HMM threshold (Part A) and extend entry window (Part B).
Combined best + consecutive-bearish filter tested in Part C.
"""
import os
import sys
from collections import defaultdict
from datetime import time as dt_time

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.strategy.hmm_transition import (
    add_causal_features, extract_daily_features,
    label_states, train_hmm,
)
from orb_system.strategy.bearish_spike_fade import (
    compute_session_params, run_backtest,
)

RESULTS    = os.path.join(ROOT, "results")
TRAIN_END  = "2024-12-31"
INIT_CAP   = 100_000.0
BASE_RISK  = 1.0
W          = 72

TP_VARIANT = "D"    # prev_poc — fixed from Phase 22B
TP_X       = None
ENTRY_FRAC = 1 / 3

WFO_WINDOWS = [
    (1, "2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    (2, "2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    (3, "2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    (4, "2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    (5, "2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]

PART_A_CONFIGS = [
    ("A1", ">0.40",  0.40),
    ("A2", ">0.50",  0.50),
    ("A3", ">0.60",  0.60),   # Phase 22B baseline
    ("A4", ">0.70",  0.70),
    ("A5", ">0.743", 0.743),  # bearish diagonal
]

BASE_WINDOW = dt_time(10, 30)
PART_B_WINDOWS = [
    ("B1", "→10:00", dt_time(10,  0)),
    ("B2", "→10:30", dt_time(10, 30)),   # Phase 22B baseline
    ("B3", "→11:00", dt_time(11,  0)),
    ("B4", "→11:30", dt_time(11, 30)),
    ("B5", "→12:00", dt_time(12,  0)),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_csv(rows, fname):
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, fname), index=False)

def _build_date_index(df):
    da = np.array(df.index.date)
    m = {}
    for i, d in enumerate(da):
        m.setdefault(d, []).append(i)
    return {d: np.array(v) for d, v in m.items()}

def _pf(trades):
    w = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    l = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
    if l == 0: return np.inf
    if w == 0: return 0.0
    return w / l

def _wr(trades):
    if not trades: return 0.0
    return sum(1 for t in trades if t["pnl_usd"] > 0) / len(trades) * 100

def _sr(trades, fd, ld):
    if not trades: return np.nan
    dr = pd.date_range(fd, ld, freq="B")
    daily = defaultdict(float)
    for t in trades: daily[t["date"]] += t["pnl_usd"]
    s = np.array([daily.get(d.strftime("%Y-%m-%d"), 0.0) for d in dr])
    return float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else np.nan

def _maxdd(trades, cap=INIT_CAP):
    if not trades: return 0.0
    eq = [cap]
    for t in trades: eq.append(eq[-1] + t["pnl_usd"])
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / cap * 100).min())

def _ret(trades, cap=INIT_CAP):
    return sum(t["pnl_usd"] for t in trades) / cap * 100 if trades else 0.0

def _exits(trades):
    if not trades: return {r: 0.0 for r in ["SL","TP","TRAIL","TIME","EOD"]}
    n = len(trades)
    return {r: sum(1 for t in trades if t["exit_reason"] == r) / n * 100
            for r in ["SL","TP","TRAIL","TIME","EOD"]}

def _n_signal(session_params, dates):
    return sum(1 for d in dates if session_params.get(d, {}).get("signal_active"))

def _year_breakdown(trades, cap=INIT_CAP):
    by_yr = defaultdict(list)
    for t in trades: by_yr[t["year"]].append(t)
    return {yr: {"N": len(v), "WR": _wr(v), "PF": _pf(v), "ret": _ret(v, cap)}
            for yr, v in sorted(by_yr.items())}


def _stats_row(exp_id, label, trades_tr, trades_te, fd_tr, ld_tr,
               fd_te, ld_te, sig_tr, sig_te):
    return {
        "exp": exp_id, "label": label,
        "N_tr": len(trades_tr),  "sig_tr": sig_tr,
        "N_te": len(trades_te),  "sig_te": sig_te,
        "WR_tr": _wr(trades_tr),
        "PF_tr": _pf(trades_tr),
        "SR_tr": _sr(trades_tr, fd_tr, ld_tr),
        "WR_te": _wr(trades_te),
        "PF_te": _pf(trades_te),
        "SR_te": _sr(trades_te, fd_te, ld_te),
        "ret_te": _ret(trades_te),
        "maxdd_te": _maxdd(trades_te),
    }


def _print_full(name, trades_tr, trades_te, fd_tr, ld_tr, fd_te, ld_te,
                sig_tr=None, sig_te=None):
    def _blk(label, trades, fd, ld, sig):
        n = len(trades)
        if n == 0:
            print(f"  {label}: No trades")
            return
        e = _exits(trades)
        sr = _sr(trades, fd, ld)
        sig_str = f" | Sig={sig}" if sig is not None else ""
        print(f"  {label}: N={n}{sig_str} | WR={_wr(trades):.1f}% | "
              f"PF={_pf(trades):.3f} | SR={sr:+.3f} | "
              f"Ret={_ret(trades):+.1f}% | MaxDD={_maxdd(trades):.1f}%")
        print(f"    Exits: SL={e['SL']:.1f}% TP={e['TP']:.1f}% "
              f"Trail={e['TRAIL']:.1f}% Time={e['TIME']:.1f}% EOD={e['EOD']:.1f}%")
        by_yr = _year_breakdown(trades)
        if by_yr:
            yr_str = "  |  ".join(
                f"{yr}: N={v['N']} WR={v['WR']:.0f}% PF={v['PF']:.2f} "
                f"ret={v['ret']:+.1f}%"
                for yr, v in sorted(by_yr.items()))
            print(f"    Annual: {yr_str}")

    print(f"\n  ── {name} ──")
    _blk("TRAIN", trades_tr, fd_tr, ld_tr, sig_tr)
    _blk("TEST ", trades_te, fd_te, ld_te, sig_te)


# ── WFO ───────────────────────────────────────────────────────────────────────

def _bootstrap_pf(pnls, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    pf_s = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        w = s[s > 0].sum(); l = abs(s[s < 0].sum())
        pf_s.append(w / l if l > 0 else np.inf)
    pf_s = [x for x in pf_s if np.isfinite(x)]
    return np.mean(pf_s), np.percentile(pf_s, 5), np.percentile(pf_s, 95)


def run_wfo(df_raw, time_arr, date_idx_map, feat_all,
            poc_per_date, avg_vol_per_date,
            best_thresh, best_window, best_consec):
    print(f"\n{'=' * W}")
    print(f"  WALK-FORWARD VALIDATION (5 anchored windows)")
    print(f"  threshold={best_thresh} | window=→{best_window} | "
          f"consecutive={best_consec}")
    print(f"{'=' * W}")

    pooled = []; win_pf = []
    for win, tr_s, tr_e, te_s, te_e in WFO_WINDOWS:
        fa_w = (feat_all[(feat_all["date"].astype(str) >= tr_s) &
                         (feat_all["date"].astype(str) <= te_e)]
                .dropna(subset=["volume_ratio", "daily_atr"])
                .reset_index(drop=True))
        fa_tr_w = fa_w[fa_w["date"].astype(str) <= tr_e].reset_index(drop=True)
        X_tr_w  = fa_tr_w[["daily_return", "volume_ratio"]].values
        if len(X_tr_w) < 20:
            print(f"  Window {win}: insufficient data. Skip."); continue

        mod_w, st_w = train_hmm(X_tr_w, 3, seed=42)
        lmap_w = label_states(st_w, fa_tr_w["daily_return"].values, 3)
        sp_w = compute_session_params(
            fa_w, poc_per_date, avg_vol_per_date, mod_w, st_w, lmap_w, 3, tr_e,
            hmm_threshold=best_thresh, consecutive_filter=best_consec,
        )
        te_dates = fa_w[fa_w["date"].astype(str) > tr_e]["date"].values
        trades_w = run_backtest(df_raw, time_arr, date_idx_map, sp_w,
                                te_dates, TP_VARIANT, TP_X, ENTRY_FRAC,
                                INIT_CAP, BASE_RISK,
                                entry_window_end=best_window)
        pf_w = _pf(trades_w)
        sr_w = _sr(trades_w, te_s, te_e) if trades_w else np.nan
        pooled.extend(trades_w); win_pf.append(pf_w)
        print(f"  V{win} ({te_s}→{te_e}): N={len(trades_w):>3} | "
              f"PF={pf_w:.3f} | SR={sr_w:+.3f}")

    if not pooled:
        print("  No OOS trades."); return

    pnls = np.array([t["pnl_usd"] for t in pooled])
    pf_p = _pf(pooled)
    sr_p = _sr(pooled, WFO_WINDOWS[0][3], WFO_WINDOWS[-1][4])
    tstat, pval = ttest_1samp(pnls, 0.0)
    p1   = pval / 2 if tstat > 0 else 1.0 - pval / 2
    bm, b5, b95 = _bootstrap_pf(pnls)
    n_pos = sum(1 for pf in win_pf if pf > 1.0)

    print(f"\n  POOLED OOS ({len(pooled)} trades):")
    print(f"    PF={pf_p:.3f} | SR={sr_p:+.3f}")
    print(f"    T-test p (one-sided) = {p1:.4f}")
    print(f"    Bootstrap PF — mean={bm:.3f} | p5={b5:.3f} | p95={b95:.3f}")
    print(f"    Windows PF>1.0: {n_pos}/5")

    if p1 < 0.10 and b5 > 0.95 and n_pos >= 3:
        print(f"\n  ✓ EDGE CONFIRMED. Proceed to Monte Carlo FTMO sizing.")
    else:
        print(f"\n  ✗ Edge NOT confirmed.")
        reasons = []
        if p1 >= 0.10:    reasons.append(f"p={p1:.4f} ≥ 0.10")
        if b5 <= 0.95:    reasons.append(f"bootstrap p5={b5:.3f} ≤ 0.95")
        if n_pos < 3:     reasons.append(f"{n_pos}/5 windows PF>1.0")
        print(f"    Failed: {' | '.join(reasons)}")

    _save_csv(pooled, "p22c_wfo_oos_pooled.csv")
    print(f"  Saved: results/p22c_wfo_oos_pooled.csv ({len(pooled)} rows)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 22C — BEARISH SPIKE FADE: FREQUENCY OPTIMIZATION")
    print("=" * W)

    cfg    = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]} → {df_raw.index[-1]}")

    time_arr     = np.array(df_raw.index.time)
    date_idx_map = _build_date_index(df_raw)

    # ── features ──────────────────────────────────────────────────────────
    feat_all_raw = add_causal_features(extract_daily_features(df_raw))
    feat_all = (feat_all_raw.dropna(subset=["volume_ratio", "daily_atr"])
                .sort_values("date").reset_index(drop=True))
    feat_all = feat_all.copy()
    feat_all["avg_vol_20"] = (feat_all["volume"].shift(1)
                              .rolling(20, min_periods=5).mean())
    avg_vol_per_date = dict(zip(feat_all["date"].values,
                                feat_all["avg_vol_20"].values))

    # prev_poc per session date
    poc_df   = compute_poc_features(df_raw)
    da_arr   = np.array(df_raw.index.date)
    ta_arr   = np.array(df_raw.index.time)
    pv_arr   = poc_df["prev_poc"].values
    poc_per_date = {}
    for d in np.unique(da_arr):
        mask = (da_arr == d) & (ta_arr == dt_time(9, 30))
        if mask.any():
            v = float(pv_arr[np.where(mask)[0][0]])
            if not np.isnan(v):
                poc_per_date[d] = v

    # ── HMM (training set) ────────────────────────────────────────────────
    feat_tr = feat_all[feat_all["date"].astype(str) <= TRAIN_END].reset_index(drop=True)
    feat_te = feat_all[feat_all["date"].astype(str) >  TRAIN_END].reset_index(drop=True)
    X_tr    = feat_tr[["daily_return", "volume_ratio"]].values
    model, states_tr = train_hmm(X_tr, 3, seed=42)
    lmap    = label_states(states_tr, feat_tr["daily_return"].values, 3)

    tr_dates = feat_tr["date"].values
    te_dates = feat_te["date"].values
    fd_tr = str(tr_dates[0]);  ld_tr = TRAIN_END
    fd_te = str(te_dates[0]);  ld_te = str(te_dates[-1])

    print(f"  Train sessions: {len(feat_tr)} | Test sessions: {len(feat_te)}")
    print(f"  State map: {lmap}")

    # ══════════════════════════════════════════════════════════════════════
    #  PART A — HMM threshold scan  (entry window fixed at 10:30)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("  PART A — HMM THRESHOLD SCAN  (entry window 09:30–10:30)")
    print(f"{'=' * W}")

    part_a_rows = []
    for exp_id, label, thresh in PART_A_CONFIGS:
        sp = compute_session_params(
            feat_all, poc_per_date, avg_vol_per_date,
            model, states_tr, lmap, 3, TRAIN_END,
            hmm_threshold=thresh,
        )
        sig_tr = _n_signal(sp, tr_dates)
        sig_te = _n_signal(sp, te_dates)
        t_tr = run_backtest(df_raw, time_arr, date_idx_map, sp, tr_dates,
                            TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                            entry_window_end=BASE_WINDOW)
        t_te = run_backtest(df_raw, time_arr, date_idx_map, sp, te_dates,
                            TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                            entry_window_end=BASE_WINDOW)
        r = _stats_row(exp_id, label, t_tr, t_te, fd_tr, ld_tr,
                       fd_te, ld_te, sig_tr, sig_te)
        part_a_rows.append({**r, "_thresh": thresh, "_window": BASE_WINDOW,
                            "_consec": False})
        _print_full(f"{exp_id} threshold {label}", t_tr, t_te,
                    fd_tr, ld_tr, fd_te, ld_te, sig_tr, sig_te)

    _save_csv(part_a_rows, "p22c_partA.csv")

    # Part A summary table
    print(f"\n  PART A SUMMARY")
    print(f"  {'Thresh':>8} | {'N_tr':>6} | {'N_te':>6} | {'Sig_te':>7} | "
          f"{'WR%':>6} | {'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>7}")
    print(f"  {'-' * 64}")
    for r in part_a_rows:
        print(f"  {r['label']:>8} | {r['N_tr']:>6} | {r['N_te']:>6} | "
              f"{r['sig_te']:>7} | {r['WR_te']:>5.1f}% | "
              f"{r['PF_tr']:>6.3f} | {r['PF_te']:>6.3f} | {r['SR_te']:>+7.3f}")

    # Select best threshold for Part B
    cands_a = [r for r in part_a_rows if r["N_te"] >= 5]
    if not cands_a:
        cands_a = part_a_rows
    best_a = max(cands_a, key=lambda r: (r["SR_te"] if not np.isnan(r["SR_te"]) else -99))
    best_thresh = best_a["_thresh"]
    print(f"\n  → Best threshold for Part B: {best_a['label']} "
          f"(SR_te={best_a['SR_te']:+.3f}, N_te={best_a['N_te']})")

    # ══════════════════════════════════════════════════════════════════════
    #  PART B — Entry window scan  (best threshold from Part A)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print(f"  PART B — ENTRY WINDOW SCAN  (threshold {best_a['label']})")
    print(f"{'=' * W}")

    sp_best_a = compute_session_params(
        feat_all, poc_per_date, avg_vol_per_date,
        model, states_tr, lmap, 3, TRAIN_END,
        hmm_threshold=best_thresh,
    )
    sig_tr_b = _n_signal(sp_best_a, tr_dates)
    sig_te_b = _n_signal(sp_best_a, te_dates)

    part_b_rows = []
    for exp_id, label, wend in PART_B_WINDOWS:
        t_tr = run_backtest(df_raw, time_arr, date_idx_map, sp_best_a, tr_dates,
                            TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                            entry_window_end=wend)
        t_te = run_backtest(df_raw, time_arr, date_idx_map, sp_best_a, te_dates,
                            TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                            entry_window_end=wend)
        r = _stats_row(exp_id, label, t_tr, t_te, fd_tr, ld_tr,
                       fd_te, ld_te, sig_tr_b, sig_te_b)
        part_b_rows.append({**r, "_thresh": best_thresh, "_window": wend,
                            "_consec": False})
        _print_full(f"{exp_id} window {label}", t_tr, t_te,
                    fd_tr, ld_tr, fd_te, ld_te, sig_tr_b, sig_te_b)

    _save_csv(part_b_rows, "p22c_partB.csv")

    print(f"\n  PART B SUMMARY")
    print(f"  {'Window':>8} | {'N_tr':>6} | {'N_te':>6} | {'Sig_te':>7} | "
          f"{'WR%':>6} | {'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>7}")
    print(f"  {'-' * 64}")
    for r in part_b_rows:
        print(f"  {r['label']:>8} | {r['N_tr']:>6} | {r['N_te']:>6} | "
              f"{r['sig_te']:>7} | {r['WR_te']:>5.1f}% | "
              f"{r['PF_tr']:>6.3f} | {r['PF_te']:>6.3f} | {r['SR_te']:>+7.3f}")

    cands_b = [r for r in part_b_rows if r["N_te"] >= 5]
    if not cands_b: cands_b = part_b_rows
    best_b = max(cands_b, key=lambda r: (r["SR_te"] if not np.isnan(r["SR_te"]) else -99))
    best_window = best_b["_window"]
    print(f"\n  → Best window for Part C: {best_b['label']} "
          f"(SR_te={best_b['SR_te']:+.3f}, N_te={best_b['N_te']})")

    # ══════════════════════════════════════════════════════════════════════
    #  PART C — Combined best + consecutive filter
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print(f"  PART C — COMBINED BEST + CONSECUTIVE FILTER")
    print(f"  threshold={best_thresh} | window=→{best_window}")
    print(f"{'=' * W}")

    # C1: best combined, no consecutive filter
    sp_c1 = compute_session_params(
        feat_all, poc_per_date, avg_vol_per_date,
        model, states_tr, lmap, 3, TRAIN_END,
        hmm_threshold=best_thresh, consecutive_filter=False,
    )
    t_tr_c1 = run_backtest(df_raw, time_arr, date_idx_map, sp_c1, tr_dates,
                           TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                           entry_window_end=best_window)
    t_te_c1 = run_backtest(df_raw, time_arr, date_idx_map, sp_c1, te_dates,
                           TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                           entry_window_end=best_window)

    # C2: best combined + consecutive filter
    sp_c2 = compute_session_params(
        feat_all, poc_per_date, avg_vol_per_date,
        model, states_tr, lmap, 3, TRAIN_END,
        hmm_threshold=best_thresh, consecutive_filter=True,
    )
    t_tr_c2 = run_backtest(df_raw, time_arr, date_idx_map, sp_c2, tr_dates,
                           TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                           entry_window_end=best_window)
    t_te_c2 = run_backtest(df_raw, time_arr, date_idx_map, sp_c2, te_dates,
                           TP_VARIANT, TP_X, ENTRY_FRAC, INIT_CAP, BASE_RISK,
                           entry_window_end=best_window)

    sig_c1_te = _n_signal(sp_c1, te_dates)
    sig_c2_te = _n_signal(sp_c2, te_dates)

    _print_full("C1 — Best combined (no consecutive filter)",
                t_tr_c1, t_te_c1, fd_tr, ld_tr, fd_te, ld_te,
                _n_signal(sp_c1, tr_dates), sig_c1_te)
    _print_full("C2 — Best combined + consecutive bearish filter",
                t_tr_c2, t_te_c2, fd_tr, ld_tr, fd_te, ld_te,
                _n_signal(sp_c2, tr_dates), sig_c2_te)

    # Part C table with 2025/2026 breakdown
    def _yr_sr(trades, yr):
        yt = [t for t in trades if t["year"] == yr]
        if not yt: return np.nan
        # use full year range for SR
        return _sr(yt, f"{yr}-01-01", f"{yr}-12-31")

    print(f"\n  PART C SUMMARY")
    print(f"  {'Config':30} | {'N_te':>6} | {'WR%':>6} | {'PF_te':>6} | "
          f"{'SR_te':>7} | {'2025_SR':>8} | {'2026_SR':>8}")
    print(f"  {'-' * 74}")
    for cfg_lbl, t_te, sp_c in [
        ("C1 — best combined",        t_te_c1, sp_c1),
        ("C2 — + consecutive filter", t_te_c2, sp_c2),
    ]:
        sr25 = _yr_sr(t_te, 2025); sr26 = _yr_sr(t_te, 2026)
        sr_str = lambda s: f"{s:+.3f}" if not np.isnan(s) else "  n/a"
        print(f"  {cfg_lbl:30} | {len(t_te):>6} | {_wr(t_te):>5.1f}% | "
              f"{_pf(t_te):>6.3f} | {_sr(t_te, fd_te, ld_te):>+7.3f} | "
              f"{sr_str(sr25):>8} | {sr_str(sr26):>8}")

    r_c1 = _stats_row("C1", "best combined", t_tr_c1, t_te_c1,
                      fd_tr, ld_tr, fd_te, ld_te,
                      _n_signal(sp_c1, tr_dates), sig_c1_te)
    r_c2 = _stats_row("C2", "+consecutive", t_tr_c2, t_te_c2,
                      fd_tr, ld_tr, fd_te, ld_te,
                      _n_signal(sp_c2, tr_dates), sig_c2_te)
    _save_csv([{**r, "_thresh": best_thresh,
                "_window": str(best_window), "_consec": c}
               for r, c in [(r_c1, False), (r_c2, True)]],
              "p22c_partC.csv")

    # ── auto-selection ────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  AUTO-SELECTION")
    print(f"{'=' * W}")

    all_te = [
        ("C1", t_te_c1, best_thresh, best_window, False),
        ("C2", t_te_c2, best_thresh, best_window, True),
    ]
    # Also consider best Part A and B configs
    for r in part_a_rows + part_b_rows:
        # Reconstruct trades not stored; skip (use C1/C2 summary only)
        pass

    eligible = [(lbl, t, th, w, cf)
                for lbl, t, th, w, cf in all_te
                if len(t) >= 60 and _sr(t, fd_te, ld_te) > 0.5]

    if eligible:
        best_final = max(eligible,
                         key=lambda x: _sr(x[1], fd_te, ld_te))
        lbl, best_t, best_th, best_w, best_cf = best_final
        print(f"  ✓ Walk-forward candidate: {lbl}")
        print(f"    N test={len(best_t)} | "
              f"SR test={_sr(best_t, fd_te, ld_te):+.3f} | "
              f"PF test={_pf(best_t):.3f}")
        run_wfo(df_raw, time_arr, date_idx_map, feat_all,
                poc_per_date, avg_vol_per_date,
                best_th, best_w, best_cf)
    else:
        max_n_any = max(len(r["N_te"]) if isinstance(r["N_te"], list)
                        else r["N_te"]
                        for r in part_a_rows + part_b_rows + [r_c1, r_c2])
        # Find which got closest to 60
        all_n = {
            "Best Part A": best_a["N_te"],
            "Best Part B": best_b["N_te"],
            "C1":          len(t_te_c1),
            "C2":          len(t_te_c2),
        }
        print(f"  ✗ No config reaches N test >= 60 AND SR > 0.5.")
        print(f"  Maximum N achieved per candidate:")
        for k, v in sorted(all_n.items(), key=lambda x: -x[1]):
            # also report SR
            if k == "C1":
                sr_v = _sr(t_te_c1, fd_te, ld_te)
            elif k == "C2":
                sr_v = _sr(t_te_c2, fd_te, ld_te)
            elif k.startswith("Best Part A"):
                sr_v = best_a["SR_te"]
            else:
                sr_v = best_b["SR_te"]
            print(f"    {k:20}: N={v:>3} | SR={sr_v:+.3f}")

        print(f"\n  WFO skipped. To reach N>=60, consider further threshold "
              f"relaxation or abandoning the volume filter.")

    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    main()
