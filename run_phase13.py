#!/usr/bin/env python3
"""
Phase 13: Walk-Forward Validation of POC Reversion + HMM.

Fixed configuration (from Phase 12):
  deviation_mult=1.0  exhaustion_mult=1.2  volume_mult=1.3
  sl_mult=1.0xATR  tp_frac=0.67  max_bars=120  09:45-14:30 NY
  HMM: 3-state, ranging only, Viterbi (strict causal, no smoothing)

5 anchored expanding-train windows, ~6-month test each.
HMM re-trained from scratch per window.
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
from orb_system.regime.hmm import RegimeHMM, FEATURE_COLS
from orb_system.regime.features_p9 import compute_daily_features_p9, zscore_normalize_p9
from orb_system.strategy.poc_rev_fractional import POCRevFractEngine
from orb_system.strategy.poc_reversion import POCResults, PV

WINDOWS = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-17"),
]

FIXED = dict(
    tp_frac        = 0.67,
    deviation_mult = 1.0,
    exhaustion_mult= 1.2,
    volume_mult    = 1.3,
    sl_mult        = 1.0,
    max_bars       = 120,
    time_start     = "09:45",
    time_end       = "14:30",
    allowed_regimes= ("ranging",),
)

N_BOOT        = 1000
MIN_TRADES_WFO = 30
W             = 72
RESULTS_DIR   = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────────

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
    print(f"  {len(df_ind):,} bars | {df_ind.index[0].date()} to {df_ind.index[-1].date()}")

    print("Computing Phase 9 daily features ...")
    feat_raw  = compute_daily_features_p9(df_ind)
    feat_norm = zscore_normalize_p9(feat_raw)
    feat_hmm  = feat_norm.rename(columns={"intraday_direction": "or_range_normalized"})
    print(f"  {len(feat_hmm)} daily feature rows")

    return df_ind, feat_hmm


def _slice_df(df_ind, start_str, end_str):
    s = pd.Timestamp(start_str).date()
    e = pd.Timestamp(end_str).date()
    d = np.array(df_ind.index.date)
    return df_ind[(d >= s) & (d <= e)]


def _slice_feat(feat_hmm, end_str):
    e = pd.Timestamp(end_str).date()
    return feat_hmm[[d <= e for d in feat_hmm.index]]


# ── HMM per window ─────────────────────────────────────────────────────────────

def _train_hmm(feat_hmm, train_end_str):
    e = pd.Timestamp(train_end_str).date()
    feat_tr = feat_hmm[[d <= e for d in feat_hmm.index]].dropna()
    hmm = RegimeHMM(n_states=3, random_state=42)
    hmm.fit(feat_tr)
    regime_series = hmm.predict_regimes(feat_hmm.dropna())
    regime_map    = dict(zip(regime_series.index, regime_series.values))
    return hmm, regime_map, regime_series


def _regime_dist_in_window(regime_map, start_str, end_str):
    s = pd.Timestamp(start_str).date()
    e = pd.Timestamp(end_str).date()
    days = {d: r for d, r in regime_map.items() if s <= d <= e}
    total = max(len(days), 1)
    dist  = {lbl: sum(1 for r in days.values() if r == lbl) for lbl in
             ("ranging", "trending", "volatile")}
    pct   = {k: v / total * 100 for k, v in dist.items()}
    return dist, pct, total


# ── POC statistics ─────────────────────────────────────────────────────────────

def _poc_stats(df_te):
    from datetime import time as dt_time
    date_arr = np.array(df_te.index.date)
    time_arr = np.array(df_te.index.time)
    all_pos  = np.arange(len(df_te))
    gaps     = []
    for d in np.unique(date_arr):
        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]
        rth   = np.array([t >= dt_time(9, 30) for t in times])
        if rth.sum() == 0:
            continue
        oi    = idxs[rth][0]
        pp    = float(df_te["prev_poc"].values[oi])
        op    = float(df_te["open"].values[oi])
        atr   = float(df_te["atr"].values[oi])
        if np.isnan(pp) or atr <= 0:
            continue
        gaps.append(abs(pp - op) / atr)
    if not gaps:
        return np.nan, 0.0
    g = np.array(gaps)
    return float(np.median(g)), float((g <= 2.0).mean() * 100)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_pf(v):
    if v != v or v == float("inf"):
        return "  inf"
    return f"{v:.3f}"


def _max_consec_loss(trades):
    best = cur = 0
    for t in trades:
        if t.pnl_net < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _window_metrics(r: POCResults):
    m  = r.metrics()
    ex = r.exit_breakdown()
    bh = [t.bars_held for t in r.trades]
    return m, ex, bh


def _fmt_realized_rr(trades):
    wins   = [t.pnl_pts for t in trades if t.pnl_net > 0]
    losses = [abs(t.pnl_pts) for t in trades if t.pnl_net <= 0]
    if wins and losses:
        return float(np.mean(wins)) / float(np.mean(losses))
    return 0.0


# ── Print per-window block ────────────────────────────────────────────────────

def _print_window(vn, tr_s, tr_e, te_s, te_e, r, regime_map, df_te, hmm_n_tr):
    m, ex, bh = _window_metrics(r)
    poc_med, poc_pct = _poc_stats(df_te)
    dist, pct, n_days = _regime_dist_in_window(regime_map, te_s, te_e)
    ann = r.annual_breakdown()

    print(f"\n{'='*W}")
    print(f"  WINDOW {vn} · Train: {tr_s} -> {tr_e} | Test: {te_s} -> {te_e}")
    print(f"  HMM trained on {hmm_n_tr} days")
    print(f"{'='*W}")

    print(f"  HMM states (test, {n_days} sessions): "
          f"ranging {pct['ranging']:.0f}% / "
          f"trending {pct['trending']:.0f}% / "
          f"volatile {pct['volatile']:.0f}%")

    poc_str = (f"  POC stats: median gap {poc_med:.1f} ATR | "
               f"within 2 ATR: {poc_pct:.0f}%")
    print(poc_str)

    n     = m["n"]
    flag  = "" if n >= MIN_TRADES_WFO else " [LOW]"
    print(f"  Trades: {n}{flag} · TPD: {m['trades_per_day']:.2f}")
    print(f"  WR: {m['wr']*100:.1f}%  PF: {_fmt_pf(m['pf'])}  "
          f"SR: {m['sharpe']:.3f}  Return: {m['ret_pct']:.2f}%  "
          f"MaxDD: {m['max_dd_pct']:.2f}%")
    if n > 0:
        print(f"  Exits: SL={ex['sl']*100:.0f}%  TP={ex['tp']*100:.0f}%  "
              f"Time={ex['timeout']*100:.0f}%  EOD={ex['eod']*100:.0f}%")
        rr_r = _fmt_realized_rr(r.trades)
        print(f"  AvgWin=${m['avg_win']:.0f}  AvgLoss=${m['avg_loss']:.0f}  "
              f"R/R realized: {rr_r:.2f}")
        print(f"  Median bars to exit: {np.median(bh):.0f}  "
              f"Max consec losses: {_max_consec_loss(r.trades)}")
        if ann:
            yr_parts = []
            for yr, a in ann.items():
                yr_parts.append(f"{yr}[n={a['n']} wr={a['wr']*100:.0f}% "
                                 f"pf={_fmt_pf(a['pf'])} sr={a['sharpe']:.2f}]")
            print("  Annual: " + "  ".join(yr_parts))
    print(f"  {'='*W}")


# ── Statistical tests ────────────────────────────────────────────────────────

def _stat_tests(all_pnl_net):
    n = len(all_pnl_net)
    print(f"\n{'='*W}")
    print("  STATISTICAL SIGNIFICANCE (OOS trades pooled across all windows)")
    print(f"  Total OOS trades: {n}")
    print(f"{'='*W}")

    if n < 10:
        print("  Insufficient trades for statistical tests.")
        return

    arr = np.array(all_pnl_net)
    # T-test (one-sided: H1: mean > 0)
    t_stat, p_two = stats.ttest_1samp(arr, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
    print(f"\n  1. One-sample t-test (H1: mean pnl > 0):")
    print(f"     t = {t_stat:.3f}  p (one-sided) = {p_one:.4f}")
    if p_one < 0.01:
        interp = "Strong evidence (p<0.01)"
    elif p_one < 0.05:
        interp = "Significant (p<0.05)"
    elif p_one < 0.10:
        interp = "Marginal (p<0.10)"
    else:
        interp = "Not significant (p>0.10)"
    print(f"     -> {interp}")

    # Bootstrap PF
    np.random.seed(42)
    pf_boots = []
    for _ in range(N_BOOT):
        s   = np.random.choice(arr, size=n, replace=True)
        w   = s[s > 0];  l = s[s <= 0]
        gw  = float(w.sum())  if w.size else 0.0
        gl  = float(abs(l.sum())) if l.size else 0.0
        pf  = gw / gl if gl > 0 else np.nan
        if not np.isnan(pf):
            pf_boots.append(pf)

    pb = np.array(pf_boots)
    mn_pf = np.mean(pb)
    p5_pf = np.percentile(pb, 5)
    p95_pf= np.percentile(pb, 95)
    print(f"\n  2. Bootstrap PF ({N_BOOT} iterations, with replacement):")
    print(f"     mean={mn_pf:.3f}  p5={p5_pf:.3f}  p95={p95_pf:.3f}")
    if p5_pf > 1.0:
        boot_interp = "Strong evidence (p5>1.0)"
    elif p5_pf > 0.95:
        boot_interp = "Moderate evidence (p5>0.95)"
    else:
        boot_interp = "Insufficient evidence (p5<0.95)"
    print(f"     -> {boot_interp}")

    return p_one, mn_pf, p5_pf


# ── Summary table ─────────────────────────────────────────────────────────────

def _summary_table(window_results):
    W2 = 80
    print(f"\n{'='*W2}")
    print("  WALK-FORWARD SUMMARY TABLE")
    print(f"{'='*W2}")
    print(f"  {'V':>1} | {'Train period':<22} | {'Test period':<22} | "
          f"{'PF':>6} | {'SR':>6} | {'N':>4} | {'WR%':>5} | {'TPD':>4}")
    print("  " + "-" * (W2 - 2))

    pf_gt1 = 0;  sr_gt0 = 0
    for vn, tr_s, tr_e, te_s, te_e, r in window_results:
        m   = r.metrics()
        n   = m["n"]
        ok  = "" if n >= MIN_TRADES_WFO else "*"
        if m["pf"] > 1.0:  pf_gt1 += 1
        if m["sharpe"] > 0: sr_gt0 += 1
        print(f"  {vn:>1} | {tr_s} -> {tr_e} | {te_s} -> {te_e} | "
              f"{_fmt_pf(m['pf']):>6} | {m['sharpe']:>6.3f} | "
              f"{n:>3d}{ok} | {m['wr']*100:>4.0f}% | {m['trades_per_day']:>4.2f}")

    # Mean row
    all_m = [r.metrics() for *_, r in window_results]
    valid = [m for m in all_m if m["n"] >= MIN_TRADES_WFO]
    if valid:
        mean_pf = np.mean([m["pf"] if m["pf"] != float("inf") else 2.0 for m in valid])
        mean_sr = np.mean([m["sharpe"] for m in valid])
        mean_n  = np.mean([m["n"] for m in valid])
        mean_wr = np.mean([m["wr"] for m in valid])
        mean_tpd= np.mean([m["trades_per_day"] for m in valid])
        print("  " + "-" * (W2 - 2))
        print(f"  {'mu':>1} |                       (mean of valid windows)       | "
              f"{mean_pf:>6.3f} | {mean_sr:>6.3f} | "
              f"{mean_n:>4.0f} | {mean_wr*100:>4.0f}% | {mean_tpd:>4.2f}")

    print(f"{'='*W2}")
    print(f"  * = < {MIN_TRADES_WFO} OOS trades")
    print(f"  Windows with PF > 1.0: {pf_gt1}/5")
    print(f"  Windows with SR > 0.0: {sr_gt0}/5")

    return pf_gt1, sr_gt0


# ── Final verdict ─────────────────────────────────────────────────────────────

def _verdict(pf_gt1, sr_gt0, stat_results):
    print(f"\n{'='*W}")
    print("  FINAL VERDICT")
    print(f"{'='*W}")

    p_val, mean_pf, p5_pf = stat_results if stat_results else (1.0, 0.0, 0.0)

    print(f"  Windows PF>1.0: {pf_gt1}/5   Windows SR>0: {sr_gt0}/5")
    print(f"  T-test p-value (one-sided): {p_val:.4f}")
    print(f"  Bootstrap p5 PF: {p5_pf:.3f}")

    strong_stat  = p_val < 0.10 and p5_pf > 0.95
    ok_windows   = pf_gt1 >= 3

    if ok_windows and strong_stat:
        print(f"\n  VERDICT: Edge CONFIRMED across walk-forward windows.")
        print(f"  Recommend Phase 14: Monte Carlo sizing for FTMO.")
    elif ok_windows and not strong_stat:
        print(f"\n  VERDICT: Directionally positive ({pf_gt1}/5 windows PF>1.0)")
        print(f"  but insufficient statistical power (p={p_val:.3f}, p5={p5_pf:.3f}).")
        print(f"  Recommend extending dataset before FTMO preparation.")
    else:
        failing = 5 - pf_gt1
        print(f"\n  VERDICT: Edge NOT confirmed. {failing}/5 windows PF<1.0.")
        print(f"  Regime-specific or noise. Investigate failing windows before proceeding.")

    print(f"{'='*W}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 13: WALK-FORWARD VALIDATION")
    print("  Config: dev=1.0 exh=1.2 vol=1.3 sl=1.0xATR tp=0.67 HMM-ranging")
    print("  5 anchored windows · HMM re-trained per window")
    print("=" * W)

    df_ind, feat_hmm = _load_all()
    date_arr = np.array(df_ind.index.date)

    all_pnl_net   = []
    window_results = []

    for vn, (tr_s, tr_e, te_s, te_e) in enumerate(WINDOWS, start=1):
        print(f"\nWindow {vn}: training HMM on {tr_s} -> {tr_e} ...")

        # Re-train HMM on this window's training data
        feat_tr_slice = feat_hmm[[d <= pd.Timestamp(tr_e).date()
                                   for d in feat_hmm.index]].dropna()
        n_hmm_tr = len(feat_tr_slice)
        hmm = RegimeHMM(n_states=3, random_state=42)
        hmm.fit(feat_tr_slice)

        # Viterbi on full feature sequence
        regime_series = hmm.predict_regimes(feat_hmm.dropna())
        regime_map    = dict(zip(regime_series.index, regime_series.values))

        # Test data slice
        te_s_d = pd.Timestamp(te_s).date()
        te_e_d = pd.Timestamp(te_e).date()
        df_te  = df_ind[(date_arr >= te_s_d) & (date_arr <= te_e_d)]

        # Run backtest
        r = POCRevFractEngine.run(
            df_te,
            regime_map=regime_map,
            label=f"p13_v{vn}",
            **FIXED,
        )

        # Collect OOS trades
        if r.metrics()["n"] >= MIN_TRADES_WFO:
            all_pnl_net.extend([t.pnl_net for t in r.trades])

        # Save CSV
        df_out = r.to_df()
        if not df_out.empty:
            df_out.to_csv(os.path.join(RESULTS_DIR, f"p13_window_{vn}.csv"), index=False)

        # Print window block
        _print_window(vn, tr_s, tr_e, te_s, te_e, r,
                      regime_map, df_te, n_hmm_tr)

        window_results.append((vn, tr_s, tr_e, te_s, te_e, r))

    # Save pooled OOS trades
    all_oos_rows = []
    for vn, tr_s, tr_e, te_s, te_e, r in window_results:
        df_v = r.to_df()
        if not df_v.empty:
            df_v.insert(0, "window", vn)
            all_oos_rows.append(df_v)
    if all_oos_rows:
        pd.concat(all_oos_rows, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "p13_oos_all.csv"), index=False
        )

    # Summary table
    pf_gt1, sr_gt0 = _summary_table(window_results)

    # Summary CSV (window-level)
    summary_rows = []
    for vn, tr_s, tr_e, te_s, te_e, r in window_results:
        m = r.metrics()
        summary_rows.append({
            "window": vn, "train_start": tr_s, "train_end": tr_e,
            "test_start": te_s, "test_end": te_e,
            "n": m["n"], "wr": m["wr"], "pf": m["pf"] if m["pf"] != float("inf") else -1,
            "sharpe": m["sharpe"], "ret_pct": m["ret_pct"], "max_dd_pct": m["max_dd_pct"],
            "trades_per_day": m["trades_per_day"],
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "p13_summary.csv"), index=False
    )
    print(f"\n  Saved: p13_summary.csv, p13_oos_all.csv, p13_window_[1-5].csv")

    # Statistical tests
    stat_results = _stat_tests(all_pnl_net)

    # Final verdict
    _verdict(pf_gt1, sr_gt0, stat_results)


if __name__ == "__main__":
    main()
