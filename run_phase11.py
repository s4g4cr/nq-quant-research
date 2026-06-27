#!/usr/bin/env python3
"""
Phase 11: Volume Profile POC Mean Reversion.

Hypothesis: price extended from the Point of Control (POC) with an
exhaustion candle reverts back toward POC.

No regime filter  -  raw structural hypothesis validated first.
Two POC types tested:
  prev_poc     -  prior full-session POC (09:30-15:45)
  session_poc  -  rolling intraday POC

Train: 2021-06-25 to 2024-11-30
Test:  2024-12-01 to 2026-06-24

6 experiments if diagnostic shows edge.
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
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.strategy.poc_reversion import (
    POCReversionEngine, POCResults, run_diagnostic_p11
)

SPLIT_DATE  = "2024-12-01"
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

W  = 72
W2 = 70


# -- Data loading ---------------------------------------------------------------

def _load_all() -> pd.DataFrame:
    cfg    = Config()
    print("Loading data and computing indicators ...")
    df     = load_data(cfg)
    df_ind = add_indicators(df, cfg)
    print(f"  {len(df_ind):,} bars | "
          f"{df_ind.index[0].date()} to {df_ind.index[-1].date()}")

    print("Computing volume profile POC features (this may take 30-60s) ...")
    df_poc = compute_poc_features(df_ind, confluence_threshold=2.0)
    df_ind["prev_poc"]      = df_poc["prev_poc"]
    df_ind["session_poc"]   = df_poc["session_poc"]
    df_ind["poc_confluence"] = df_poc["poc_confluence"]
    df_ind["target_poc"]    = df_poc["target_poc"]
    n_valid = df_ind["prev_poc"].notna().sum()
    print(f"  POC features computed. Bars with valid prev_poc: {n_valid:,}")
    return df_ind


def _split(df_ind):
    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)
    return df_ind[date_arr < split], df_ind[date_arr >= split]


# -- Formatting helpers ---------------------------------------------------------

def _fmt_pf(v) -> str:
    if v != v or v == float("inf"):
        return "  inf"
    return f"{v:.3f}"


def _print_result(tag: str, r: POCResults, min_n: int = 20):
    m   = r.metrics()
    n   = m["n"]
    ok  = "" if n >= min_n else "*"
    ex  = r.exit_breakdown()
    ann = r.annual_breakdown()
    print(f"  {tag}: n={n}{ok}  WR={m['wr']*100:.1f}%  PF={_fmt_pf(m['pf'])}  "
          f"SR={m['sharpe']:.3f}  Ret={m['ret_pct']:.2f}%  DD={m['max_dd_pct']:.2f}%")
    if n > 0:
        print(f"    Exits: SL={ex['sl']*100:.0f}%  TP={ex['tp']*100:.0f}%  "
              f"EOD={ex['eod']*100:.0f}%  Time={ex['timeout']*100:.0f}%  "
              f"| AvgW=${m['avg_win']:.0f}  AvgL=${m['avg_loss']:.0f}  "
              f"TPD={m['trades_per_day']:.2f}")
        if ann:
            yr_parts = []
            for yr, a in ann.items():
                yr_parts.append(f"{yr}[n={a['n']} wr={a['wr']*100:.0f}% "
                                 f"pf={_fmt_pf(a['pf'])} sr={a['sharpe']:.2f}]")
            print("    Annual: " + "  ".join(yr_parts))


def _run_exp(df_tr, df_te, label_short: str, **kwargs) -> tuple:
    label_tr = f"p11_{label_short}_tr"
    label_te = f"p11_{label_short}_te"
    r_tr = POCReversionEngine.run(df_tr, label=label_tr, **kwargs)
    r_te = POCReversionEngine.run(df_te, label=label_te, **kwargs)

    fname = f"p11_{label_short}.csv"
    df_out = r_te.to_df()
    if not df_out.empty:
        df_out.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
    return r_tr, r_te


# -- Experiments ----------------------------------------------------------------

BASE = dict(
    tp_variant      = "A",
    deviation_mult  = 1.5,
    sl_mult         = 1.0,
    volume_mult     = 1.3,
    exhaustion_mult = 0.8,
    confluence_only = False,
    max_bars        = 120,
    time_start      = "09:45",
    time_end        = "14:30",
)


def run_experiment(n: int, name: str, df_tr, df_te, **override):
    params = {**BASE, **override}
    print(f"\n{'='*W2}")
    print(f"  EXPERIMENT {n}  -  {name}")
    kv = {k: v for k, v in params.items()
          if k not in ("max_bars", "time_start", "time_end")}
    print(f"  Params: {kv}")
    print(f"{'='*W2}")
    r_tr, r_te = _run_exp(df_tr, df_te, f"exp{n}", **params)
    _print_result("TRAIN", r_tr, min_n=20)
    _print_result("TEST ", r_te, min_n=80)
    return r_tr, r_te


def main():
    print("=" * W)
    print("  PHASE 11: VOLUME PROFILE POC MEAN REVERSION")
    print(f"  Train: 2021-06-25 to {SPLIT_DATE}")
    print(f"  Test:  {SPLIT_DATE} to 2026-06-24")
    print("  No regime filter  -  raw structural hypothesis")
    print("=" * W)

    df_ind       = _load_all()
    df_tr, df_te = _split(df_ind)

    print(f"\nTrain: {df_tr.index[0].date()} to {df_tr.index[-1].date()} "
          f"| {len(df_tr):,} bars")
    print(f"Test:  {df_te.index[0].date()} to {df_te.index[-1].date()} "
          f"| {len(df_te):,} bars")

    # -- Mandatory diagnostic on training set -----------------------------------
    print("\nRunning mandatory pre-P&L diagnostic on TRAINING set ...")
    viable = run_diagnostic_p11(
        df_tr,
        deviation_mult  = BASE["deviation_mult"],
        volume_mult     = BASE["volume_mult"],
        exhaustion_mult = BASE["exhaustion_mult"],
        max_bars        = BASE["max_bars"],
        time_start      = BASE["time_start"],
        time_end        = BASE["time_end"],
    )

    if not viable:
        print("\nDiagnostic shows no viable configuration. Skipping backtests.")
        print("Root cause: POC reversion rate insufficient to overcome noise.")
        return

    print("\nDiagnostic passed. Running 6 experiments.")

    # -- Experiment 1: Baseline, TP = target_poc (Variant A) -------------------
    r1_tr, r1_te = run_experiment(
        1, "Baseline . TP=target_poc (Variant A)",
        df_tr, df_te,
    )

    # -- Experiment 2: TP = confluence zone (Variant B) -------------------------
    r2_tr, r2_te = run_experiment(
        2, "TP = session_poc on confluence days (Variant B)",
        df_tr, df_te, tp_variant="B",
    )

    # -- Experiment 3: Deviation threshold sensitivity --------------------------
    print(f"\n{'='*W2}")
    print("  EXPERIMENT 3  -  Deviation Threshold Sensitivity")
    print(f"  Test deviation_mult = [1.0, 1.5, 2.0, 2.5, 3.0]")
    print(f"{'='*W2}")

    exp3_results = []
    for dm in [1.0, 1.5, 2.0, 2.5, 3.0]:
        r_tr, r_te = _run_exp(df_tr, df_te, f"exp3_dev{dm:.1f}".replace(".", ""),
                              **{**BASE, "deviation_mult": dm})
        m_tr = r_tr.metrics()
        m_te = r_te.metrics()
        print(f"  dev={dm:.1f}  Tr[n={m_tr['n']} WR={m_tr['wr']*100:.1f}% "
              f"SR={m_tr['sharpe']:.3f}]  "
              f"Te[n={m_te['n']} WR={m_te['wr']*100:.1f}% "
              f"PF={_fmt_pf(m_te['pf'])} SR={m_te['sharpe']:.3f}]")
        exp3_results.append((dm, m_tr, m_te))

    best_dev = max(exp3_results, key=lambda x: x[1]["sharpe"])[0]
    print(f"  Best dev_mult by Train Sharpe: {best_dev}")

    # -- Experiment 4: Confluence filter ----------------------------------------
    r4_tr, r4_te = run_experiment(
        4, "Confluence filter only (poc_confluence=True days)",
        df_tr, df_te, confluence_only=True,
    )

    exp4_better = (r4_tr.metrics()["pf"] > r1_tr.metrics()["pf"]
                   and r4_te.metrics()["n"] > 0)
    print(f"  Confluence filter vs baseline: "
          f"{'IMPROVES' if exp4_better else 'no improvement'} train PF")

    # -- Experiment 5: Exhaustion candle sensitivity -----------------------------
    print(f"\n{'='*W2}")
    print("  EXPERIMENT 5  -  Exhaustion Candle Sensitivity")
    print(f"  Test exhaustion_mult = [0.5, 0.8, 1.0, 1.2]")
    print(f"{'='*W2}")

    exp5_results = []
    for em in [0.5, 0.8, 1.0, 1.2]:
        r_tr, r_te = _run_exp(df_tr, df_te, f"exp5_exh{em:.1f}".replace(".", ""),
                              **{**BASE, "exhaustion_mult": em})
        m_tr = r_tr.metrics()
        m_te = r_te.metrics()
        print(f"  exh={em:.1f}  Tr[n={m_tr['n']} WR={m_tr['wr']*100:.1f}% "
              f"SR={m_tr['sharpe']:.3f}]  "
              f"Te[n={m_te['n']} WR={m_te['wr']*100:.1f}% "
              f"PF={_fmt_pf(m_te['pf'])} SR={m_te['sharpe']:.3f}]")
        exp5_results.append((em, m_tr, m_te))

    best_exh = max(exp5_results, key=lambda x: x[1]["sharpe"])[0]
    print(f"  Best exhaustion_mult by Train Sharpe: {best_exh}")

    # -- Experiment 6: Combined best ---------------------------------------------
    print(f"\n{'='*W2}")
    print("  EXPERIMENT 6  -  Combined Best Configuration")
    print(f"  dev={best_dev}  exh={best_exh}  "
          f"confluence_only={exp4_better}")
    print(f"{'='*W2}")

    best_params = {
        **BASE,
        "deviation_mult":  best_dev,
        "exhaustion_mult": best_exh,
        "confluence_only": exp4_better,
    }
    r6_tr, r6_te = _run_exp(df_tr, df_te, "exp6_combined", **best_params)
    _print_result("TRAIN", r6_tr, min_n=20)
    _print_result("TEST ", r6_te, min_n=80)

    # -- Summary table ----------------------------------------------------------
    W3 = 100
    print("\n" + "=" * W3)
    print("  PHASE 11 SUMMARY TABLE")
    print("=" * W3)
    print(f"  {'Exp':>3} | {'TP':>3} | {'Dev':>4} | {'Exh':>4} | {'Conf':>4} | "
          f"{'SL':>4} | {'N Tr':>5} | {'PF Tr':>6} | {'SR Tr':>6} | "
          f"{'N Te':>5} | {'PF Te':>6} | {'SR Te':>6} | {'WR% Te':>7}")
    print("  " + "-" * (W3 - 2))

    summary_rows = [
        (1,  "A",  1.5, 0.8, "N", 1.0, r1_tr,  r1_te),
        (2,  "B",  1.5, 0.8, "N", 1.0, r2_tr,  r2_te),
        (4,  "A",  1.5, 0.8, "Y", 1.0, r4_tr,  r4_te),
        (6,  "A",  best_dev, best_exh, "Y" if exp4_better else "N",
                                            1.0, r6_tr,  r6_te),
    ]

    for row in summary_rows:
        exp_n, tpv, dm, em, cf, slm, r_tr, r_te = row
        mt  = r_tr.metrics()
        me  = r_te.metrics()
        nok = "" if me["n"] >= 80 else "*"
        print(f"  {exp_n:>3} | {tpv:>3} | {dm:>4.1f} | {em:>4.1f} | {cf:>4} | "
              f"{slm:>4.1f} | {mt['n']:>5d} | {_fmt_pf(mt['pf']):>6} | "
              f"{mt['sharpe']:>6.3f} | {me['n']:>4d}{nok} | "
              f"{_fmt_pf(me['pf']):>6} | {me['sharpe']:>6.3f} | "
              f"{me['wr']*100:>6.1f}%")

    # Also add best dev/exh sweep rows
    print("  -- Exp 3 (deviation sweep):")
    for dm, mt, me in exp3_results:
        nok = "" if me["n"] >= 80 else "*"
        print(f"     dev={dm:.1f}  Tr[n={mt['n']} PF={_fmt_pf(mt['pf'])} "
              f"SR={mt['sharpe']:.3f}]  "
              f"Te[n={me['n']}{nok} PF={_fmt_pf(me['pf'])} SR={me['sharpe']:.3f}]")
    print("  -- Exp 5 (exhaustion sweep):")
    for em, mt, me in exp5_results:
        nok = "" if me["n"] >= 80 else "*"
        print(f"     exh={em:.1f}  Tr[n={mt['n']} PF={_fmt_pf(mt['pf'])} "
              f"SR={mt['sharpe']:.3f}]  "
              f"Te[n={me['n']}{nok} PF={_fmt_pf(me['pf'])} SR={me['sharpe']:.3f}]")

    print("=" * W3)
    print("  * = < 80 test trades")
    print(f"  Train: 2021-06-25 to {SPLIT_DATE} | Test: {SPLIT_DATE} to 2026-06-24")
    print("  SL = sl_mult x ATR(20) | TP = target_poc (A) or session_poc on conf days (B)")

    # -- Auto-interpretation ----------------------------------------------------
    _interpret(summary_rows, exp3_results, exp5_results)


def _interpret(summary_rows, exp3_results, exp5_results):
    print("\n" + "=" * W)
    print("  INTERPRETATION")
    print("=" * W)

    all_results = [(r[0], r[6].metrics(), r[7].metrics()) for r in summary_rows]

    profitable_te = [(n, mt, me) for n, mt, me in all_results
                     if me["pf"] > 1.0 and me["n"] >= 80]

    if not profitable_te:
        print("  No experiment produced PF > 1.0 in the test set (n>=80).")

        # Check if theoretical expectancy was positive
        print("  Diagnostic showed positive theoretical expectancy but")
        print("  entry conditions are filtering out the best setups.")

        # Which filter removes most quality?
        # Compare Exp 1 baseline vs sweeps
        exp3_best   = max(exp3_results, key=lambda x: x[2]["pf"])
        exp5_best   = max(exp5_results, key=lambda x: x[2]["pf"])
        print(f"  Best deviation_mult in test: {exp3_best[0]:.1f} "
              f"(PF={_fmt_pf(exp3_best[2]['pf'])} n={exp3_best[2]['n']})")
        print(f"  Best exhaustion_mult in test: {exp5_best[0]:.1f} "
              f"(PF={_fmt_pf(exp5_best[2]['pf'])} n={exp5_best[2]['n']})")
        print("  Recommend Phase 12: relax entry filters or test longer")
        print("  look-back window for POC stability.")
        print("=" * W)
        return

    best_n, best_tr, best_te = max(profitable_te, key=lambda x: x[2]["sharpe"])
    pf_gap = abs(best_tr["pf"] - best_te["pf"])
    gen_ok = pf_gap < 0.3

    print(f"  Edge found in Experiment {best_n}:")
    print(f"    Test PF={_fmt_pf(best_te['pf'])} | SR={best_te['sharpe']:.3f} "
          f"| WR={best_te['wr']*100:.1f}% | n={best_te['n']}")
    print(f"    Train-Test PF gap: {pf_gap:.3f} "
          f"({'< 0.3  -  generalises' if gen_ok else '>= 0.3  -  possible overfit'})")

    if gen_ok:
        print(f"  FLAG FOR PHASE 12: walk-forward validation at Exp {best_n} params.")
    else:
        print("  High train-test gap. Recommend walk-forward before committing.")
    print("=" * W)


if __name__ == "__main__":
    main()
