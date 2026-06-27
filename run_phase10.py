#!/usr/bin/env python3
"""
Phase 10: VWAP Reversion with R/R-derived SL calibration.

Tests whether VWAP Reversion has structural edge at wider SL widths.
Phase 8 Exp 1 used SL = 0.5×ATR ≈ R/R 5:1 → WR 11%, expectancy negative.
Hypothesis: wider SL (lower R/R target) improves WR enough to flip expectancy.

9 experiments: R/R = 5.0 / 4.5 / 4.0 / 3.5 / 3.0 / 2.5 / 2.0 / 1.5 / 1.0

Entry identical to Phase 8 Exp 1:
  HMM ranging | deviation_mult=1.5 | candle_mult=0.8 | volume_mult=1.2
  Time 10:00-14:30 | max 1L+1S per session | EOD 15:45 | max_bars=90

Train: 2021-06-25 to 2024-11-30
Test:  2024-12-01 to 2026-06-17

HMM: Phase 9 feature set (features_p9: intraday_direction replaces or_range_normalized).
     Retrained on the extended training set.
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
from orb_system.regime.hmm import RegimeHMM
from orb_system.regime.features_p9 import (
    compute_daily_features_p9, zscore_normalize_p9
)
from orb_system.strategy.vwap_rev_rr import (
    VWAPRevRREngine, RevRRResults, run_diagnostic_p10, RR_LEVELS
)

SPLIT_DATE  = "2024-12-01"
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

W = 72


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
    split = pd.Timestamp(SPLIT_DATE).date()

    print("\nComputing Phase 9 daily features (for HMM) ...")
    feat_raw  = compute_daily_features_p9(df_ind)
    feat_norm = zscore_normalize_p9(feat_raw)

    feat_for_hmm = feat_norm.rename(
        columns={"intraday_direction": "or_range_normalized"}
    )

    tr_mask  = np.array([d < split for d in feat_for_hmm.index])
    feat_tr  = feat_for_hmm[tr_mask].dropna()
    print(f"  HMM train features: {len(feat_tr)} valid days "
          f"({feat_tr.index[0]} to {feat_tr.index[-1]})")

    hmm = RegimeHMM(n_states=3, random_state=42)
    hmm.fit(feat_tr)

    regime_series = hmm.predict_regimes(feat_for_hmm)
    regime_map    = dict(zip(regime_series.index, regime_series.values))

    # Print HMM diagnostics
    feat_diag = feat_raw.copy()
    feat_diag["or_range_normalized"] = feat_raw["intraday_direction"]
    hmm.print_diagnostics(feat_diag, regime_series, split)

    return regime_map


def _split(df_ind):
    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)
    return df_ind[date_arr < split], df_ind[date_arr >= split]


# ── Formatter helpers ─────────────────────────────────────────────────────────

def _fmt_pf(v) -> str:
    if v == float("inf") or v != v:
        return "  inf"
    return f"{v:.3f}"


def _fmt_exp(rr, wr) -> str:
    exp = wr * rr - (1.0 - wr)
    return f"{exp:+.3f}"


# ── Experiment runner ─────────────────────────────────────────────────────────

ENTRY_PARAMS = dict(
    deviation_mult = 1.5,
    candle_mult    = 0.8,
    volume_mult    = 1.2,
    max_bars       = 90,
    time_start     = "10:00",
    time_end       = "14:30",
)


def _run(df, regime_map, target_rr, label=""):
    return VWAPRevRREngine.run(df, regime_map, target_rr, **ENTRY_PARAMS, label=label)


def _print_exp(r_tr: RevRRResults, r_te: RevRRResults, target_rr: float):
    W2 = 62
    print("\n" + "=" * W2)
    print(f"  R/R TARGET: {target_rr:.1f}:1  ->  SL = TP distance / {target_rr:.1f}")
    print("=" * W2)

    for tag, r, min_t in [("TRAIN", r_tr, 20), ("TEST ", r_te, 60)]:
        m  = r.metrics()
        n  = m["n"]
        ok = "OK" if n >= min_t else "LOW"
        pf = _fmt_pf(m["pf"])
        exp_str = _fmt_exp(target_rr, m["wr"])
        print(f"  {tag}: n={n}({ok}) WR={m['wr']*100:.1f}% PF={pf} "
              f"SR={m['sharpe']:.3f} Ret={m['ret_pct']:.2f}% "
              f"DD={m['max_dd_pct']:.2f}%")

        if n > 0:
            ex = r.exit_breakdown()
            print(f"         Exits: SL={ex['sl']*100:.0f}%  "
                  f"TP={ex['tp']*100:.0f}%  "
                  f"EOD={ex['eod']*100:.0f}%  "
                  f"Timeout={ex['timeout']*100:.0f}%")
            print(f"         AvgWin=${m['avg_win']:.0f}  AvgLoss=${m['avg_loss']:.0f}  "
                  f"Expectancy={exp_str}")

        if tag == "TEST " and n > 0:
            ann = r.annual_breakdown()
            parts = []
            for yr, a in ann.items():
                star = " **" if yr >= 2025 else ""
                parts.append(f"{yr}[n={a['n']} wr={a['wr']*100:.0f}% "
                              f"pf={_fmt_pf(a['pf'])} sr={a['sharpe']:.2f}]{star}")
            if parts:
                print("         Annual: " + "  ".join(parts))

    print("=" * W2)


# ── Interpretation ────────────────────────────────────────────────────────────

def _interpret(summary_rows: list):
    W2 = 78
    print("\n" + "=" * W2)
    print("  INTERPRETATION")
    print("=" * W2)

    profitable = [(rr, m_tr, m_te) for rr, m_tr, m_te in summary_rows
                  if m_te["pf"] > 1.0 and m_te["n"] >= 60]

    if not profitable:
        print("  VWAP reversion has no edge at any SL width tested.")
        print("  The natural reversion rate is insufficient to overcome")
        print("  noise at any R/R configuration.")
        print("  Recommend abandoning this hypothesis.")
        print("=" * W2)
        return

    best_rr, _, best_te = max(profitable, key=lambda x: x[2]["sharpe"])
    best_tr = next(m_tr for rr, m_tr, m_te in summary_rows if rr == best_rr)

    pf_gap = abs(best_tr["pf"] - best_te["pf"])
    gen_ok = pf_gap < 0.3

    print(f"  Edge found at R/R {best_rr:.1f}:1")
    print(f"    Test Sharpe: {best_te['sharpe']:.3f} | PF: {_fmt_pf(best_te['pf'])}")
    print(f"    Train-Test PF gap: {pf_gap:.3f} "
          f"({'< 0.3 -- generalises' if gen_ok else '>= 0.3 -- overfits'})")

    # Check if theoretical expectancy was positive but actual PF < 1
    diag_pass = [rr for rr, _, m_te in summary_rows if m_te["pf"] > 1.0 and m_te["n"] >= 60]
    if not diag_pass:
        print("  NOTE: Theoretical expectancy was positive but actual PF < 1.0")
        print("  Signal quality degrades after entry filters.")
        print("  Consider relaxing entry filters in Phase 11.")
    elif gen_ok:
        print(f"  Recommend Phase 11 walk-forward validation at R/R {best_rr:.1f}.")
    else:
        print("  Warning: high train-test gap suggests parameter sensitivity.")
        print("  Recommend testing with walk-forward before committing.")

    print("=" * W2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 10: VWAP REVERSION -- R/R SL CALIBRATION")
    print(f"  Train: 2021-06-25 to {SPLIT_DATE}")
    print(f"  Test:  {SPLIT_DATE} to 2026-06-17")
    print("=" * W)

    df_ind     = _load_all()
    regime_map = _setup_hmm(df_ind)
    df_tr, df_te = _split(df_ind)

    print(f"\nTrain: {df_tr.index[0].date()} to {df_tr.index[-1].date()} "
          f"| {len(df_tr):,} bars")
    print(f"Test:  {df_te.index[0].date()} to {df_te.index[-1].date()} "
          f"| {len(df_te):,} bars")

    # Regime distribution in test
    split_date  = pd.Timestamp(SPLIT_DATE).date()
    te_regimes  = {d: r for d, r in regime_map.items() if d >= split_date}
    te_ranging  = sum(1 for r in te_regimes.values() if r == "ranging")
    te_total    = len(te_regimes)
    print(f"\nTest regime distribution: "
          f"ranging={te_ranging}/{te_total} ({te_ranging/max(te_total,1)*100:.1f}%)")

    # ── Pre-P&L diagnostic (mandatory) ────────────────────────────────────────
    print("\nRunning mandatory pre-P&L diagnostic on TRAINING set ...")
    viable = run_diagnostic_p10(
        df_tr, regime_map, **ENTRY_PARAMS
    )

    if not viable:
        print("\nDiagnostic shows no viable R/R level. Skipping backtests.")
        return

    # ── 9 experiments ──────────────────────────────────────────────────────────
    summary_rows = []

    for rr in RR_LEVELS:
        r_tr = _run(df_tr, regime_map, rr, label=f"p10_rr{rr:.1f}_tr")
        r_te = _run(df_te, regime_map, rr, label=f"p10_rr{rr:.1f}_te")
        _print_exp(r_tr, r_te, rr)

        # Save test trades
        fname = f"p10_rr{rr:.1f}.csv".replace(".", "_").replace("_csv", ".csv")
        df_out = r_te.to_df()
        if not df_out.empty:
            df_out.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
            print(f"  Saved -> {fname}")

        summary_rows.append((rr, r_tr.metrics(), r_te.metrics()))

    # ── Summary table ──────────────────────────────────────────────────────────
    W2 = 100
    print("\n" + "=" * W2)
    print("  PHASE 10 SUMMARY TABLE  (SL calibration: R/R-derived from TP distance)")
    print("=" * W2)
    print(f"  {'R/R':>4} | {'WR%Tr':>6} | {'WR%Te':>6} | {'PF Tr':>6} | "
          f"{'PF Te':>6} | {'SR Te':>6} | {'N Te':>5} | "
          f"{'AvgWin':>7} | {'AvgLoss':>8} | {'Expectancy':>10}")
    print("  " + "-" * (W2 - 2))

    for rr, m_tr, m_te in summary_rows:
        n_ok  = " " if m_te["n"] >= 60 else "*"
        exp   = m_te["wr"] * rr - (1.0 - m_te["wr"])
        print(f"  {rr:>4.1f} | {m_tr['wr']*100:>5.1f}% | {m_te['wr']*100:>5.1f}% | "
              f"{_fmt_pf(m_tr['pf']):>6} | {_fmt_pf(m_te['pf']):>6} | "
              f"{m_te['sharpe']:>6.3f} | {m_te['n']:>4d}{n_ok} | "
              f"${m_te['avg_win']:>6.0f} | ${m_te['avg_loss']:>7.0f} | "
              f"{exp:>+10.3f}")

    print("=" * W2)
    print("  * = < 60 test trades")
    print(f"  Test period: {SPLIT_DATE} to 2026-06-17")
    print("  Entry: HMM=ranging | dev=1.5 | candle=0.8 | vol=1.2 | 10:00-14:30")
    print("  SL = TP_distance / R/R  (anchored to actual trade opportunity)")

    # ── Auto-interpret ─────────────────────────────────────────────────────────
    _interpret(summary_rows)


if __name__ == "__main__":
    main()
