#!/usr/bin/env python3
"""
Phase 9: VWAP Breakout with Dynamic Trailing Stop

Experiments:
  Exp 1 -- Baseline VWAP Breakout (OR/ATR>15, HMM trending, trail=1.5)
  Exp 2 -- OR threshold sensitivity  [10, 13, 15, 18, 22]
  Exp 3 -- Trailing multiplier       [0.75, 1.0, 1.5, 2.0, 2.5]
  Exp 4 -- Time window               [A:10:00-13:00, B:10:30-14:00, C:11:00-15:00]
  Exp 5 -- Remove HMM (OR filter only)
  Exp 6 -- Best combined config from Exp 2-5

HMM: recalibrated with corrected features (removed or_range_normalized,
     added intraday_direction). OR/ATR handled as hard threshold.

Train: 2021-06-25 to 2023-06-30
Test:  2023-07-01 to 2026-06-24
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import add_indicators
from orb_system.regime.hmm import RegimeHMM, FEATURE_COLS as HMM_FEATURE_COLS
from orb_system.regime.features_p9 import (
    compute_daily_features_p9, zscore_normalize_p9, HMM_COLS
)
from orb_system.strategy.vwap_breakout import (
    VWAPBreakoutEngine, BreakoutResults, run_diagnostic_p9
)

SPLIT_DATE  = "2023-07-01"
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline params
# NOTE: or_atr_threshold is in 1-min ATR units (NQ typical: p25=11, p50=15, p75=21).
# The spec's "1.2" assumed daily ATR scale; the equivalent in 1-min ATR units is ~15.
BASE = dict(
    or_atr_threshold = 15,
    candle_mult      = 0.8,
    volume_mult      = 1.3,
    trail_mult       = 1.5,
    trail_activation = 1.0,
    max_bars         = 90,
    time_start       = "10:30",
    time_end         = "14:00",
    require_hmm      = True,
)


# ── Setup ─────────────────────────────────────────────────────────────────────

def _load_all():
    cfg    = Config()
    print("Loading data and computing indicators ...")
    df     = load_data(cfg)
    df_ind = add_indicators(df, cfg)
    print(f"  {len(df_ind):,} bars | "
          f"{df_ind.index[0].date()} to {df_ind.index[-1].date()}")
    return df_ind


def _setup_hmm(df_ind):
    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)

    print("\nComputing Phase 9 daily features ...")
    feat_raw  = compute_daily_features_p9(df_ind)
    feat_norm = zscore_normalize_p9(feat_raw)

    # Rename intraday_direction -> or_range_normalized so RegimeHMM is happy
    # (RegimeHMM expects FEATURE_COLS from hmm.py)
    feat_for_hmm = feat_norm.rename(
        columns={"intraday_direction": "or_range_normalized"}
    )

    tr_mask  = np.array([d < split for d in feat_for_hmm.index])
    feat_tr  = feat_for_hmm[tr_mask]
    print(f"  HMM training features: {feat_tr.dropna().shape[0]} valid days")
    print(f"  ({feat_tr.dropna().index[0]} to {feat_tr.dropna().index[-1]})")

    hmm = RegimeHMM(n_states=3, random_state=42)
    hmm.fit(feat_tr)

    regime_series = hmm.predict_regimes(feat_for_hmm)
    regime_map    = dict(zip(regime_series.index, regime_series.values))

    # print_diagnostics expects or_range_normalized; alias intraday_direction
    feat_diag = feat_raw.copy()
    feat_diag["or_range_normalized"] = feat_raw["intraday_direction"]
    hmm.print_diagnostics(feat_diag, regime_series, split)

    # OR/ATR map from raw (un-normalised) features
    or_atr_map = {
        d: float(v) for d, v in feat_raw["or_range_atr"].items()
        if not np.isnan(v)
    }

    return hmm, regime_map, or_atr_map, regime_series, feat_raw


def _split(df_ind):
    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)
    return df_ind[date_arr < split], df_ind[date_arr >= split]


# ── Printer helpers ───────────────────────────────────────────────────────────

def _fmt_pf(v):
    return f"{v:.3f}" if v != float("inf") else "  inf"


def _print_set(tag, r: BreakoutResults, min_trades=80, extra="") -> dict:
    m   = r.metrics()
    n   = m["n"]
    ok  = "OK" if n >= min_trades else "LOW"
    pf  = _fmt_pf(m["pf"])
    print(f"  {tag}: n={n}({ok}) WR={m['wr']*100:.1f}% PF={pf} "
          f"SR={m['sharpe']:.3f} Ret={m['ret_pct']:.2f}% "
          f"DD={m['max_dd_pct']:.2f}% TPD={m['trades_per_day']:.2f}{extra}")
    return m


def _print_exits(r: BreakoutResults):
    ex = r.exit_breakdown()
    print(f"       Exits: SL={ex['sl']*100:.0f}%  Trail={ex['trail']*100:.0f}%  "
          f"EOD={ex['eod']*100:.0f}%  Timeout={ex['timeout']*100:.0f}%")


def _print_annual(r: BreakoutResults, years_highlight=(2025, 2026)):
    ann = r.annual_breakdown()
    if not ann:
        return
    parts = []
    for yr, a in ann.items():
        tag = " **" if yr in years_highlight else ""
        parts.append(f"{yr}[n={a['n']} wr={a['wr']*100:.0f}% "
                     f"pf={_fmt_pf(a['pf'])} sr={a['sharpe']:.2f}]{tag}")
    print("       Annual: " + "  ".join(parts))


def _save(r: BreakoutResults, fname: str):
    df = r.to_df()
    if df.empty:
        return
    path = os.path.join(RESULTS_DIR, fname)
    df.to_csv(path, index=False)
    print(f"       Saved -> {fname}")


def _run(df, or_atr_map, regime_map, **kw):
    params = {**BASE, **kw}
    return VWAPBreakoutEngine.run(df, or_atr_map, regime_map, **params)


def _run_both(df_tr, df_te, or_atr_map, regime_map, **kw):
    r_tr = _run(df_tr, or_atr_map, regime_map, **kw)
    r_te = _run(df_te, or_atr_map, regime_map, **kw)
    return r_tr, r_te


# ── Experiments ───────────────────────────────────────────────────────────────

W = 72

def exp1_baseline(df_tr, df_te, or_atr_map, regime_map):
    print("\n" + "=" * W)
    print("  EXP 1 -- Baseline VWAP Breakout")
    print(f"  OR/ATR>{BASE['or_atr_threshold']} (1-min ATR units) | HMM=trending | "
          f"trail={BASE['trail_mult']} | window={BASE['time_start']}-{BASE['time_end']}")
    print("=" * W)

    r_tr, r_te = _run_both(df_tr, df_te, or_atr_map, regime_map, label="p9_exp1")
    _print_set("  TRAIN", r_tr, min_trades=30)
    _print_exits(r_tr)
    _print_set("  TEST ", r_te, min_trades=80)
    _print_exits(r_te)
    _print_annual(r_te)
    _save(r_te, "p9_exp1_baseline.csv")
    return r_tr, r_te


def exp2_or_threshold(df_tr, df_te, or_atr_map, regime_map):
    print("\n" + "=" * W)
    print("  EXP 2 -- OR/ATR Threshold Sensitivity")
    print("=" * W)

    # Thresholds in 1-min ATR units: NQ p25≈11, p50≈15, p75≈21
    print("  (OR/ATR in 1-min ATR units: NQ p25~11, p50~15, p75~21)")
    all_tr = {}; all_te = {}
    for thresh in [10, 13, 15, 18, 22]:
        r_tr, r_te = _run_both(df_tr, df_te, or_atr_map, regime_map,
                                or_atr_threshold=thresh)
        all_tr[thresh] = r_tr; all_te[thresh] = r_te
        lbl = f"  thresh={thresh:.1f}"
        _print_set(f"  Tr {lbl}", r_tr, min_trades=10)
        _print_set(f"  Te {lbl}", r_te, min_trades=30)

    # Save all
    frames = []
    for thresh, r in all_te.items():
        df2 = r.to_df()
        if not df2.empty:
            df2["or_atr_threshold"] = thresh
            frames.append(df2)
    if frames:
        pd.concat(frames).to_csv(
            os.path.join(RESULTS_DIR, "p9_exp2_or_thresh.csv"), index=False
        )
        print("       Saved -> p9_exp2_or_thresh.csv")

    return all_tr, all_te


def exp3_trailing(df_tr, df_te, or_atr_map, regime_map):
    print("\n" + "=" * W)
    print("  EXP 3 -- Trailing Multiplier Sensitivity")
    print("=" * W)

    all_tr = {}; all_te = {}
    for tm in [0.75, 1.0, 1.5, 2.0, 2.5]:
        r_tr, r_te = _run_both(df_tr, df_te, or_atr_map, regime_map,
                                trail_mult=tm)
        all_tr[tm] = r_tr; all_te[tm] = r_te
        m_tr = r_tr.metrics(); m_te = r_te.metrics()
        print(f"  trail={tm:.2f} | Tr: WR={m_tr['wr']*100:.0f}% "
              f"AvgW=${m_tr['avg_win']:.0f} AvgL=${m_tr['avg_loss']:.0f} "
              f"PF={_fmt_pf(m_tr['pf'])} SR={m_tr['sharpe']:.2f} | "
              f"Te: WR={m_te['wr']*100:.0f}% "
              f"AvgW=${m_te['avg_win']:.0f} AvgL=${m_te['avg_loss']:.0f} "
              f"PF={_fmt_pf(m_te['pf'])} SR={m_te['sharpe']:.2f}")

    frames = []
    for tm, r in all_te.items():
        df2 = r.to_df()
        if not df2.empty:
            df2["trail_mult_param"] = tm
            frames.append(df2)
    if frames:
        pd.concat(frames).to_csv(
            os.path.join(RESULTS_DIR, "p9_exp3_trail.csv"), index=False
        )
        print("       Saved -> p9_exp3_trail.csv")

    return all_tr, all_te


def exp4_time_window(df_tr, df_te, or_atr_map, regime_map):
    print("\n" + "=" * W)
    print("  EXP 4 -- Time Window Sensitivity")
    print("=" * W)

    windows = {
        "A(10:00-13:00)": ("10:00", "13:00"),
        "B(10:30-14:00)": ("10:30", "14:00"),   # baseline
        "C(11:00-15:00)": ("11:00", "15:00"),
    }
    all_tr = {}; all_te = {}
    for name, (ts, te) in windows.items():
        r_tr, r_te = _run_both(df_tr, df_te, or_atr_map, regime_map,
                                time_start=ts, time_end=te)
        all_tr[name] = r_tr; all_te[name] = r_te
        _print_set(f"  Tr {name}", r_tr, min_trades=10)
        _print_set(f"  Te {name}", r_te, min_trades=30)
        _print_annual(r_te)

    frames = []
    for name, r in all_te.items():
        df2 = r.to_df()
        if not df2.empty:
            df2["window"] = name
            frames.append(df2)
    if frames:
        pd.concat(frames).to_csv(
            os.path.join(RESULTS_DIR, "p9_exp4_window.csv"), index=False
        )
        print("       Saved -> p9_exp4_window.csv")

    return all_tr, all_te


def exp5_no_hmm(df_tr, df_te, or_atr_map, regime_map):
    print("\n" + "=" * W)
    print("  EXP 5 -- OR Filter Only (HMM removed)")
    print("=" * W)

    r_tr, r_te = _run_both(df_tr, df_te, or_atr_map, regime_map,
                            require_hmm=False, label="p9_exp5")

    tr_hmm_r, te_hmm_r = _run_both(df_tr, df_te, or_atr_map, regime_map,
                                     require_hmm=True)  # baseline for comparison

    _print_set("  Tr  OR-only   ", r_tr, min_trades=30)
    _print_set("  Tr  OR+HMM    ", tr_hmm_r, min_trades=30)
    _print_set("  Te  OR-only   ", r_te, min_trades=80)
    _print_exits(r_te)
    _print_set("  Te  OR+HMM    ", te_hmm_r, min_trades=80)

    te_sr_only = r_te.metrics()["sharpe"]
    te_sr_hmm  = te_hmm_r.metrics()["sharpe"]
    hmm_adds   = te_sr_hmm > te_sr_only
    print(f"\n  HMM vs no-HMM on test: {te_sr_hmm:.3f} vs {te_sr_only:.3f} "
          f"-- HMM {'ADDS value' if hmm_adds else 'does NOT add value'}")

    _save(r_te, "p9_exp5_nohmm.csv")
    return r_tr, r_te, hmm_adds


def exp6_best_combined(df_tr, df_te, or_atr_map, regime_map,
                        all_tr2, all_te2,
                        all_tr3, all_te3,
                        all_tr4, all_te4,
                        hmm_adds: bool):
    print("\n" + "=" * W)
    print("  EXP 6 -- Best Combined Configuration")
    print("=" * W)

    # Select best OR threshold (by train Sharpe, >= 30 train trades)
    best_or  = max(all_tr2.keys(),
                   key=lambda k: all_tr2[k].metrics()["sharpe"]
                   if all_tr2[k].metrics()["n"] >= 30 else -99)
    # Select best trail_mult (by train Sharpe)
    best_tm  = max(all_tr3.keys(),
                   key=lambda k: all_tr3[k].metrics()["sharpe"]
                   if all_tr3[k].metrics()["n"] >= 30 else -99)
    # Select best time window (by train Sharpe)
    best_win = max(all_tr4.keys(),
                   key=lambda k: all_tr4[k].metrics()["sharpe"]
                   if all_tr4[k].metrics()["n"] >= 10 else -99)

    # Window name -> (ts, te)
    win_map = {
        "A(10:00-13:00)": ("10:00", "13:00"),
        "B(10:30-14:00)": ("10:30", "14:00"),
        "C(11:00-15:00)": ("11:00", "15:00"),
    }
    ts, te = win_map[best_win]

    print(f"  Best params from train set:")
    best_or_str = f"{best_or:.1f}" if isinstance(best_or, float) else f"{best_or}"
    print(f"    OR/ATR threshold : {best_or_str}  (Exp 2, 1-min ATR units)")
    print(f"    trail_mult       : {best_tm:.2f}  (Exp 3)")
    print(f"    time window      : {best_win}  (Exp 4)")
    print(f"    require_hmm      : {hmm_adds}  (Exp 5)")
    print()

    r_tr, r_te = _run_both(
        df_tr, df_te, or_atr_map, regime_map,
        or_atr_threshold = best_or,
        trail_mult       = best_tm,
        time_start       = ts,
        time_end         = te,
        require_hmm      = hmm_adds,
        label            = "p9_exp6",
    )

    _print_set("  TRAIN", r_tr, min_trades=30)
    _print_exits(r_tr)
    _print_set("  TEST ", r_te, min_trades=80)
    _print_exits(r_te)
    _print_annual(r_te)
    _save(r_te, "p9_exp6_best.csv")
    return r_tr, r_te, best_or, best_tm, best_win


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(rows):
    W2 = 98
    print("\n" + "=" * W2)
    print("  PHASE 9 SUMMARY TABLE")
    print("=" * W2)
    hdr = (f"  {'Experiment':<38} {'OR':>5} {'Trail':>5} {'Window':<12} "
           f"{'PF_Tr':>6} {'PF_Te':>6} {'SR_Te':>7} {'N/d_Te':>6} {'WR_Te':>6}")
    print(hdr)
    print("  " + "-" * (W2 - 2))

    for label, r_tr, r_te, extra in rows:
        m_tr = r_tr.metrics()
        m_te = r_te.metrics()
        n_ok = " " if m_te["n"] >= 80 else "*"
        or_val = extra.get("or", BASE["or_atr_threshold"])
        or_str = f"{or_val:.1f}" if isinstance(or_val, float) else f"{or_val}"
        print(f"  {label:<38} "
              f"{or_str:>5} "
              f"{extra.get('trail', BASE['trail_mult']):>5.2f} "
              f"{extra.get('win', BASE['time_start']+'-'+BASE['time_end']):<12} "
              f"{_fmt_pf(m_tr['pf']):>6} "
              f"{_fmt_pf(m_te['pf']):>6} "
              f"{m_te['sharpe']:>7.3f} "
              f"{m_te['trades_per_day']:>5.2f}{n_ok} "
              f"{m_te['wr']*100:>5.1f}%")

    print("=" * W2)
    print("  * = < 80 test trades")
    print(f"  Test period: {SPLIT_DATE} to 2026-06-24")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 9: VWAP BREAKOUT + TRAILING STOP")
    print(f"  Train: 2021-06-25 to {SPLIT_DATE}  |  Test: {SPLIT_DATE}+")
    print("=" * W)

    df_ind = _load_all()
    hmm, regime_map, or_atr_map, regime_series, feat_raw = _setup_hmm(df_ind)
    df_tr, df_te = _split(df_ind)

    print(f"\nTrain: {df_tr.index[0].date()} to {df_tr.index[-1].date()} "
          f"| {len(df_tr):,} bars")
    print(f"Test:  {df_te.index[0].date()} to {df_te.index[-1].date()} "
          f"| {len(df_te):,} bars")

    # OR/ATR distribution on test set
    split = pd.Timestamp(SPLIT_DATE).date()
    or_te = [v for d, v in or_atr_map.items() if d >= split and not np.isnan(v)]
    if or_te:
        arr = np.array(or_te)
        pct12 = float((arr >= 1.2).mean()) * 100
        print(f"\nOR/ATR on test days (n={len(arr)}): "
              f"p25={np.percentile(arr,25):.2f}  p50={np.percentile(arr,50):.2f}  "
              f"p75={np.percentile(arr,75):.2f}  >= 1.2: {pct12:.1f}%")

    # ── Diagnostic (training set) ─────────────────────────────────────────────
    print("\nRunning pre-PnL diagnostic on training set ...")
    run_diagnostic_p9(
        df_tr, or_atr_map, regime_map,
        or_atr_threshold = 15,
        candle_mult      = 0.8,
        volume_mult      = 1.3,
        time_start       = "10:30",
        time_end         = "14:00",
        max_bars         = 90,
    )

    # ── Experiments ───────────────────────────────────────────────────────────
    e1_tr,  e1_te  = exp1_baseline(df_tr, df_te, or_atr_map, regime_map)
    e2_tr,  e2_te  = exp2_or_threshold(df_tr, df_te, or_atr_map, regime_map)
    e3_tr,  e3_te  = exp3_trailing(df_tr, df_te, or_atr_map, regime_map)
    e4_tr,  e4_te  = exp4_time_window(df_tr, df_te, or_atr_map, regime_map)
    e5_tr,  e5_te, hmm_adds = exp5_no_hmm(df_tr, df_te, or_atr_map, regime_map)
    e6_tr,  e6_te, b_or, b_tm, b_win = exp6_best_combined(
        df_tr, df_te, or_atr_map, regime_map,
        e2_tr, e2_te, e3_tr, e3_te, e4_tr, e4_te, hmm_adds
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    win_map = {
        "A(10:00-13:00)": "10:00-13:00",
        "B(10:30-14:00)": "10:30-14:00",
        "C(11:00-15:00)": "11:00-15:00",
    }
    b_win_short = win_map.get(b_win, b_win)

    summary_rows = [
        ("Exp1 Baseline (15/1.5/10:30-14:00/HMM)", e1_tr, e1_te,
         {"or": 15, "trail": 1.5, "win": "10:30-14:00"}),
    ]
    for thresh in [10, 13, 15, 18, 22]:
        summary_rows.append((
            f"Exp2 OR_thresh={thresh}", e2_tr[thresh], e2_te[thresh],
            {"or": thresh, "trail": 1.5, "win": "10:30-14:00"}
        ))
    for tm in [0.75, 1.0, 1.5, 2.0, 2.5]:
        summary_rows.append((
            f"Exp3 trail={tm:.2f}", e3_tr[tm], e3_te[tm],
            {"or": 15, "trail": tm, "win": "10:30-14:00"}
        ))
    for name in ["A(10:00-13:00)", "B(10:30-14:00)", "C(11:00-15:00)"]:
        summary_rows.append((
            f"Exp4 window={name}", e4_tr[name], e4_te[name],
            {"or": 15, "trail": 1.5, "win": win_map[name]}
        ))
    summary_rows.append(
        ("Exp5 OR-only (no HMM)", e5_tr, e5_te,
         {"or": 15, "trail": 1.5, "win": "10:30-14:00"})
    )
    b_or_str = f"{b_or:.1f}" if isinstance(b_or, float) else f"{b_or}"
    summary_rows.append(
        (f"Exp6 Best ({b_or_str}/{b_tm:.2f}/{b_win_short})", e6_tr, e6_te,
         {"or": b_or, "trail": b_tm, "win": b_win_short})
    )

    _print_summary(summary_rows)


if __name__ == "__main__":
    main()
