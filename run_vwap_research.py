#!/usr/bin/env python3
"""
Phase 8: VWAP Reversion + Breakout Strategy Research

5 Experiments:
  Exp 1 --Baseline reversion (HMM ranging days only)
  Exp 2 --No HMM filter (all regimes, reversion)
  Exp 3 --Deviation threshold sensitivity: dev_mult in [1.0, 1.5, 2.0, 2.5]
  Exp 4 --Trending days: VWAP breakout continuation
  Exp 5 --Combined system (ranging=reversion, trending=breakout, volatile=skip)

Train: 2021-06-25 to 2023-06-30
Test:  2023-07-01 to 2026-06-24  (~3 years, well above 80-trade minimum)
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
from orb_system.regime.features import compute_daily_features, zscore_normalize
from orb_system.regime.hmm import RegimeHMM
from orb_system.strategy.vwap_reversion import (
    VWAPReversionEngine, VWAPResults, run_diagnostic
)

SPLIT_DATE  = "2023-07-01"
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def _load_all():
    cfg    = Config()
    print("Loading data and computing indicators ...")
    df     = load_data(cfg)
    df_ind = add_indicators(df, cfg)
    print(f"  {len(df_ind):,} bars | "
          f"{df_ind.index[0].date()} to {df_ind.index[-1].date()}")
    return df_ind, cfg


def _train_hmm(df_ind):
    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)

    # Causal features on ALL data, z-scored with rolling window
    feat_raw  = compute_daily_features(df_ind)
    feat_norm = zscore_normalize(feat_raw)

    # Restrict training to dates before split
    tr_mask   = np.array([d < split for d in feat_norm.index])
    feat_tr   = feat_norm[tr_mask]

    print(f"\nHMM training features: {feat_tr.dropna().shape[0]} valid days "
          f"({feat_norm.index[tr_mask][0]} to {feat_norm.index[tr_mask][-1]})")

    hmm = RegimeHMM(n_states=3, random_state=42)
    hmm.fit(feat_tr)

    # Predict ALL dates (model params fixed from train --causal approximation)
    regime_series = hmm.predict_regimes(feat_norm)
    regime_map    = dict(zip(regime_series.index, regime_series.values))

    hmm.print_diagnostics(feat_raw, regime_series, split)
    return hmm, regime_map


def _slice_test(df_ind):
    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)
    return df_ind[date_arr >= split]


# ── Result printer ────────────────────────────────────────────────────────────

def _print_result(r: VWAPResults, label: str, min_trades: int = 80) -> None:
    m   = r.metrics()
    n   = m["n"]
    tag = "OK" if n >= min_trades else "LOW"

    print(f"\n  {label}  [n={n} trades | {tag}]")
    if n == 0:
        print("    No trades generated.")
        return

    pf_s = f"{m['pf']:.3f}" if m["pf"] != float("inf") else "inf"
    print(f"    WR={m['wr']*100:.1f}%  PF={pf_s}  "
          f"Sharpe={m['sharpe']:.3f}  Ret={m['ret_pct']:.2f}%")
    print(f"    AvgWin=${m['avg_win']:.0f}  AvgLoss=${m['avg_loss']:.0f}  "
          f"MaxDD={m['max_dd']:.2f}%  TPD={m['trades_per_day']:.2f}")

    ex = r.exit_breakdown()
    print(f"    Exit: SL={ex['sl']*100:.0f}%  TP={ex['tp']*100:.0f}%  "
          f"EOD={ex['eod']*100:.0f}%  Timeout={ex['timeout']*100:.0f}%")

    ann = r.annual_breakdown()
    if ann:
        parts = []
        for yr, a in ann.items():
            pf_a = f"{a['pf']:.2f}" if a["pf"] != float("inf") else "inf"
            parts.append(f"{yr}[n={a['n']} wr={a['wr']*100:.0f}% "
                         f"pf={pf_a} sr={a['sharpe']:.2f}]")
        print("    Annual: " + "  ".join(parts))


def _save_csv(r: VWAPResults, fname: str) -> None:
    df = r.to_df()
    if df.empty:
        return
    path = os.path.join(RESULTS_DIR, fname)
    df.to_csv(path, index=False)
    print(f"    Saved -> {fname}")


# ── Experiments ───────────────────────────────────────────────────────────────

def _REV(df_te, regime_map, **kw):
    defaults = dict(deviation_mult=1.5, candle_mult=0.8, volume_mult=1.2,
                    sl_mult=0.5, max_bars=90, time_start="10:00", time_end="14:30")
    defaults.update(kw)
    return VWAPReversionEngine.run_reversion(df_te, regime_map, **defaults)


def _BO(df_te, regime_map, **kw):
    defaults = dict(breakout_mult=0.5, candle_mult=0.8, volume_mult=1.2,
                    tp_atr_mult=2.0, max_bars=90, time_start="10:00", time_end="14:30")
    defaults.update(kw)
    return VWAPReversionEngine.run_breakout(df_te, regime_map, **defaults)


def exp1_baseline(df_te, regime_map):
    W = 68
    print("\n" + "=" * W)
    print("  EXP 1 --Baseline: Reversion on ranging days only")
    print("=" * W)

    r = _REV(df_te, regime_map,
             allowed_regimes=("ranging",), label="Exp1_Baseline")

    _print_result(r, "Exp 1 --Baseline (ranging, dev=1.5, sl=0.5)")
    _save_csv(r, "vwap_exp1.csv")
    return r


def exp2_no_hmm(df_te, regime_map):
    W = 68
    print("\n" + "=" * W)
    print("  EXP 2 --No HMM filter (all regimes, reversion)")
    print("=" * W)

    r = _REV(df_te, regime_map,
             allowed_regimes=("ranging", "trending", "volatile"),
             label="Exp2_NoHMM")

    _print_result(r, "Exp 2 --No HMM (all regimes, dev=1.5)")
    _save_csv(r, "vwap_exp2.csv")
    return r


def exp3_sensitivity(df_te, regime_map):
    W = 68
    print("\n" + "=" * W)
    print("  EXP 3 --Deviation threshold sensitivity")
    print("=" * W)

    results = {}
    all_trades = []
    for dm in [1.0, 1.5, 2.0, 2.5]:
        r = _REV(df_te, regime_map,
                 allowed_regimes=("ranging",),
                 deviation_mult=dm, label=f"Exp3_dev{dm}")
        results[dm] = r
        _print_result(r, f"  dev_mult={dm:.1f}", min_trades=30)
        for t in r.trades:
            all_trades.append(t)

    r_all = VWAPResults(all_trades, "Exp3_Sensitivity_All")
    _save_csv(r_all, "vwap_exp3.csv")
    return results


def exp4_breakout(df_te, regime_map):
    W = 68
    print("\n" + "=" * W)
    print("  EXP 4 --VWAP Breakout on trending days")
    print("=" * W)

    r = _BO(df_te, regime_map,
            allowed_regimes=("trending",), label="Exp4_Breakout")

    _print_result(r, "Exp 4 --Breakout (trending, bo=0.5, tp=2xATR)")
    _save_csv(r, "vwap_exp4.csv")
    return r


def exp5_combined(df_te, regime_map):
    W = 68
    print("\n" + "=" * W)
    print("  EXP 5 --Combined: ranging=reversion, trending=breakout, volatile=skip")
    print("=" * W)

    r_rev = _REV(df_te, regime_map,
                 allowed_regimes=("ranging",), label="Exp5_rev")
    r_bo  = _BO(df_te, regime_map,
                allowed_regimes=("trending",), label="Exp5_bo")

    combined = sorted(r_rev.trades + r_bo.trades, key=lambda t: t.entry_ts)
    r = VWAPResults(combined, "Exp5_Combined")

    _print_result(r, "Exp 5 --Combined System")
    print(f"    Reversion: {len(r_rev.trades)} trades | "
          f"Breakout: {len(r_bo.trades)} trades")
    _save_csv(r, "vwap_exp5.csv")
    return r


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(rows):
    W = 84
    print("\n" + "=" * W)
    print("  VWAP STRATEGY RESEARCH --SUMMARY TABLE")
    print("=" * W)

    hdr = (f"  {'Experiment':<38} {'N':>5} {'WR%':>6} {'PF':>6} "
           f"{'Sharpe':>7} {'Ret%':>7} {'MaxDD%':>7}")
    print(hdr)
    print("  " + "-" * (W - 2))

    for label, r in rows:
        m = r.metrics()
        n = m["n"]
        if n == 0:
            print(f"  {label:<38} {'0':>5}")
            continue
        pf_s  = f"{m['pf']:.3f}" if m["pf"] != float("inf") else "  inf"
        star  = "*" if n < 80 else " "
        print(f"  {label:<38} {n:>4d}{star} {m['wr']*100:>5.1f}% "
              f"{pf_s:>6} {m['sharpe']:>7.3f} {m['ret_pct']:>7.2f}% "
              f"{m['max_dd']:>6.2f}%")

    print("=" * W)
    print("  * = fewer than 80 test trades (low statistical power)")
    print(f"  Test period: {SPLIT_DATE} to 2026-06-24")
    print(f"  Slippage: 1 tick ({0.25} pts) on entry, Commission: $4 rt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    W = 68
    print("=" * W)
    print("  PHASE 8: VWAP REVERSION + BREAKOUT STRATEGY RESEARCH")
    print(f"  Train: 2021-06-25 to {SPLIT_DATE}")
    print(f"  Test:  {SPLIT_DATE} to 2026-06-24")
    print("=" * W)

    df_ind, _   = _load_all()
    hmm, regime_map = _train_hmm(df_ind)
    df_te       = _slice_test(df_ind)

    print(f"\nTest set: {df_te.index[0].date()} to {df_te.index[-1].date()} "
          f"| {len(df_te):,} bars")

    # ── Pre-P&L diagnostic (must run before any experiment P&L) ──────────────
    print("\nRunning mandatory pre-P&L diagnostic on test set ...")
    run_diagnostic(df_te, regime_map,
                   deviation_mult=1.5, candle_mult=0.8, volume_mult=1.2,
                   time_start="10:00", time_end="14:30", max_bars=90)

    # ── Experiments ───────────────────────────────────────────────────────────
    r1       = exp1_baseline(df_te, regime_map)
    r2       = exp2_no_hmm(df_te, regime_map)
    r3_dict  = exp3_sensitivity(df_te, regime_map)
    r4       = exp4_breakout(df_te, regime_map)
    r5       = exp5_combined(df_te, regime_map)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_rows = [
        ("Exp1  Baseline (ranging, dev=1.5)",            r1),
        ("Exp2  No HMM filter (all, dev=1.5)",           r2),
    ]
    for dm, r in sorted(r3_dict.items()):
        summary_rows.append((f"Exp3  Ranging, dev_mult={dm:.1f}", r))
    summary_rows += [
        ("Exp4  Breakout (trending, bo=0.5)",            r4),
        ("Exp5  Combined (rev+bo)",                      r5),
    ]

    _print_summary(summary_rows)


if __name__ == "__main__":
    main()
