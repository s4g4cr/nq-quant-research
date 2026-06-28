#!/usr/bin/env python3
"""
Phase 21 — Intraday Return Seasonality (Heston, Korajczyk, Sadka 2010).

Tests whether the return of NQ futures in a specific 30-minute interval
positively predicts the return in that same interval on the following
N trading days (half-hour return continuation).

13 intervals I1–I13 (09:30–16:00). Parts A–E + mandatory pre-PnL diagnostic.
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
from orb_system.indicators.technical import atr as compute_atr
from orb_system.strategy.intraday_seasonality import (
    IV_NAMES,
    autocorrelation_table,
    compute_interval_returns,
    lookback_directional_accuracy,
    run_interval_backtest,
)

RESULTS    = os.path.join(ROOT, "results")
W          = 80
TRAIN_END  = "2024-12-31"
TEST_START = "2025-01-02"
LOOKBACKS  = [1, 3, 5, 10, 20]
RISK_PCTS  = [0.5, 1.0, 1.5, 2.0]

WFO_WINDOWS = [
    (1, "2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    (2, "2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    (3, "2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    (4, "2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    (5, "2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]


# ── stat helpers ───────────────────────────────────────────────────────────────

def _pf(trades):
    v = np.array([t.pnl_net for t in trades])
    w = v[v > 0].sum(); l = abs(v[v <= 0].sum())
    return float(w / l) if l > 0 else float("inf")

def _sr(trades):
    if not trades:
        return 0.0
    by_d = {}
    for t in trades:
        k = str(t.date)
        by_d[k] = by_d.get(k, 0.0) + t.pnl_net
    v = np.array(list(by_d.values()))
    s = float(v.std(ddof=1))
    return float(v.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _wr(trades):
    return sum(1 for t in trades if t.pnl_net > 0) / len(trades) * 100 if trades else 0.0

def _ret(trades, cap=100_000.0):
    return sum(t.pnl_net for t in trades) / cap * 100 if trades else 0.0

def _mdd(trades, cap=100_000.0):
    c = peak = cap; worst = 0.0
    for t in trades:
        c += t.pnl_net; peak = max(peak, c)
        worst = max(worst, (peak - c) / peak)
    return worst * 100

def _avg_c(trades):
    return float(np.mean([t.n_contracts for t in trades])) if trades else 0.0

def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and v != v):
        return "   nan"
    if v == float("inf"):
        return "   inf"
    return f"{v:.{d}f}"

def _save_csv(rows, fname):
    os.makedirs(RESULTS, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, fname), index=False)

def _save_trades(trades, fname):
    _save_csv([vars(t) for t in trades], fname)

def _run(df_slice, rets_full, iname, lookback, exit_method, tp_mult, rp=1.0):
    kw = {"lookback": lookback, "exit_method": exit_method, "risk_pct": rp,
          "initial_capital": 100_000.0}
    if exit_method != "A":
        kw["tp_mult"] = tp_mult
    return run_interval_backtest(df_slice, rets_full, iname, **kw)


# ── WFO helper ────────────────────────────────────────────────────────────────

def _select_best_interval(df_tr_w, rets_full):
    """Part A logic (N=1, exit=A) on training window. Returns interval with best PF (N>=20)."""
    best_iv = IV_NAMES[0]; best_pf = -1.0
    for iname in IV_NAMES:
        t = run_interval_backtest(df_tr_w, rets_full, iname, lookback=1, exit_method="A")
        pf = _pf(t)
        if len(t) >= 20 and pf > best_pf:
            best_pf = pf; best_iv = iname
    return best_iv


# ── bootstrap ─────────────────────────────────────────────────────────────────

def _bootstrap_pf(pnl_arr, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    bpf = []
    for _ in range(n):
        s  = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
        gw = s[s > 0].sum(); gl = abs(s[s <= 0].sum())
        bpf.append(gw / gl if gl > 0 else float("inf"))
    finite = [x for x in bpf if x != float("inf")]
    return (float(np.mean(finite)), float(np.percentile(finite, 5)),
            float(np.percentile(finite, 50)), float(np.percentile(finite, 95)))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 21 — INTRADAY RETURN SEASONALITY (Heston et al. 2010)")
    print("=" * W)

    cfg    = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]} -> {df_raw.index[-1]}  ({len(df_raw):,} bars)")

    atr1m    = compute_atr(df_raw, 20)
    df       = df_raw.copy()
    df["atr"] = atr1m

    print("  Computing interval returns (full dataset)...")
    rets_full = compute_interval_returns(df)
    print(f"  Done: {len(rets_full)} dates x 13 intervals")
    rets_full.index = pd.to_datetime(rets_full.index)

    df_tr   = df.loc[:TRAIN_END]
    df_te   = df.loc[TEST_START:]
    rets_tr = rets_full.loc[:TRAIN_END]
    print(f"  Train: thru {TRAIN_END}  ({len(rets_tr)} days)")
    print(f"  Test:  from {TEST_START}  ({len(rets_full.loc[TEST_START:]):} days)")

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  DIAGNOSTIC — AUTOCORRELATION ANALYSIS (training set)")
    print(f"\n  {'Intv':>5} | {'AC(1)':>7} | {'p1':>6} | {'AC(2)':>7} | {'p2':>6} | "
          f"{'AC(5)':>7} | {'p5':>6} | {'DirAcc%':>8}")
    print(f"  {'-' * 68}")

    ac_tbl = autocorrelation_table(rets_tr)
    _save_csv(ac_tbl.to_dict("records"), "p21_diagnostic.csv")

    sig_ivs = []
    for _, r in ac_tbl.iterrows():
        ps   = [v for v in [r.p1, r.p2, r.p5] if not np.isnan(v)]
        minp = min(ps) if ps else 1.0
        flag = " *" if minp < 0.10 else ("")
        if minp < 0.10:
            sig_ivs.append(r.interval)
        print(f"  {r.interval:>5} | {_fmt(r.ac1, 4):>7} | {_fmt(r.p1, 4):>6} | "
              f"{_fmt(r.ac2, 4):>7} | {_fmt(r.p2, 4):>6} | "
              f"{_fmt(r.ac5, 4):>7} | {_fmt(r.p5, 4):>6} | "
              f"{r.dacc_pct:>7.1f}%{flag}")

    print(f"\n  * p < 0.10  —  {len(sig_ivs)} interval(s) significant: "
          f"{', '.join(sig_ivs) if sig_ivs else 'none'}")

    # Gate
    all_ps = [v for _, r in ac_tbl.iterrows()
              for v in [r.p1, r.p2, r.p5] if not np.isnan(v)]
    if all(p > 0.10 for p in all_ps):
        print("\n  STOP: No interval has statistically significant autocorrelation.")
        print("  Intraday return seasonality does NOT exist in NQ at the half-hour level.")
        print(f"\n{'=' * W}\n")
        return

    # Diagnostic section 3: lookback DA for top 3 by p1
    top3 = ac_tbl.sort_values("p1").head(3)["interval"].tolist()
    print(f"\n  DIAGNOSTIC SECTION 3 — Lookback directional accuracy (top 3 by p1)")
    for iname in top3:
        lb_df = lookback_directional_accuracy(rets_tr, iname, LOOKBACKS)
        print(f"\n  {iname}:  {'N':>3} | {'DirAcc%':>8} | {'N_obs':>6}")
        for _, row in lb_df.iterrows():
            print(f"        {int(row.N):>3} | {row.dacc_pct:>7.1f}% | {int(row.n_obs):>6}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART A — Interval scan
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  PART A — INTERVAL SCAN  (N=1, hard exit at interval end)")
    print(f"\n  {'Intv':>5} | {'N_tr':>5} | {'WR_tr':>6} | {'PF_tr':>6} | {'SR_tr':>7} | "
          f"{'N_te':>5} | {'WR_te':>6} | {'PF_te':>6} | {'SR_te':>7} | Flag")
    print(f"  {'-' * 82}")

    partA_rows   = []
    flagged_ivs  = []

    for iname in IV_NAMES:
        tr_t = run_interval_backtest(df_tr, rets_full, iname, lookback=1, exit_method="A")
        te_t = run_interval_backtest(df_te, rets_full, iname, lookback=1, exit_method="A")

        n_tr  = len(tr_t); n_te  = len(te_t)
        wr_tr = _wr(tr_t); pf_tr = _pf(tr_t); sr_tr = _sr(tr_t)
        wr_te = _wr(te_t); pf_te = _pf(te_t); sr_te = _sr(te_t)

        flag = "***" if (pf_te > 1.0 and n_te >= 60) else ""
        if flag:
            flagged_ivs.append(iname)

        partA_rows.append({
            "interval": iname, "N_tr": n_tr, "WR_tr": round(wr_tr, 1),
            "PF_tr": round(pf_tr, 3), "SR_tr": round(sr_tr, 3),
            "N_te": n_te, "WR_te": round(wr_te, 1),
            "PF_te": round(pf_te, 3), "SR_te": round(sr_te, 3), "flagged": bool(flag),
        })
        print(f"  {iname:>5} | {n_tr:>5} | {wr_tr:>5.1f}% | {_fmt(pf_tr):>6} | "
              f"{sr_tr:>+7.3f} | {n_te:>5} | {wr_te:>5.1f}% | {_fmt(pf_te):>6} | "
              f"{sr_te:>+7.3f} | {flag}")

    _save_csv(partA_rows, "p21_partA_interval_scan.csv")
    print(f"\n  Flagged (PF_test > 1.0, N_test >= 60): {flagged_ivs or ['none']}")

    if not flagged_ivs:
        print("\n  No interval has PF_test > 1.0 with N >= 60.")
        print("  Seasonality not tradeable in NQ even if statistically present.")
        print(f"\n{'=' * W}\n")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # PART B — Lookback sensitivity
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  PART B — LOOKBACK SENSITIVITY (flagged intervals)")

    best_interval: str = flagged_ivs[0]
    best_N:        int = 1
    best_SR_te:  float = -999.0
    partB_all          = []

    for iname in flagged_ivs:
        print(f"\n  {iname}:")
        print(f"  {'N':>3} | {'WR_tr':>6} | {'PF_tr':>6} | {'SR_tr':>7} | "
              f"{'WR_te':>6} | {'PF_te':>6} | {'SR_te':>7}")
        print(f"  {'-' * 57}")

        for N in LOOKBACKS:
            tr_t = run_interval_backtest(df_tr, rets_full, iname, lookback=N, exit_method="A")
            te_t = run_interval_backtest(df_te, rets_full, iname, lookback=N, exit_method="A")
            pf_te_v = _pf(te_t); sr_te_v = _sr(te_t)

            star = " *" if pf_te_v > 1.0 else ""
            print(f"  {N:>3} | {_wr(tr_t):>5.1f}% | {_fmt(_pf(tr_t)):>6} | "
                  f"{_sr(tr_t):>+7.3f} | {_wr(te_t):>5.1f}% | {_fmt(pf_te_v):>6} | "
                  f"{sr_te_v:>+7.3f}{star}")

            partB_all.append({
                "interval": iname, "N": N,
                "WR_tr": round(_wr(tr_t), 1), "PF_tr": round(_pf(tr_t), 3),
                "SR_tr": round(_sr(tr_t), 3), "WR_te": round(_wr(te_t), 1),
                "PF_te": round(pf_te_v, 3), "SR_te": round(sr_te_v, 3),
            })
            if pf_te_v > 1.0 and sr_te_v > best_SR_te:
                best_SR_te = sr_te_v; best_interval = iname; best_N = N

        _save_csv([r for r in partB_all if r["interval"] == iname],
                  f"p21_partB_lookback_{iname}.csv")

    print(f"\n  Best: {best_interval}  N={best_N}  SR_te={best_SR_te:+.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART C — Exit comparison
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  PART C — EXIT METHOD COMPARISON  ({best_interval}, N={best_N})")
    print(f"\n  {'Cfg':>4} | {'Exit':>14} | {'N':>4} | {'WR%':>5} | {'PF':>6} | "
          f"{'SR':>7} | {'Ret%':>7} | {'MaxDD%':>7}")
    print(f"  {'-' * 67}")

    C_CONFIGS = [
        ("C1", "Hard",       "A", 1.5),
        ("C2", "SL+1.5xATR", "B", 1.5),
        ("C3", "SL+2.0xATR", "C", 2.0),
    ]
    partC_rows = []
    best_C = "C1"; best_C_SR = -999.0; best_exit = "A"; best_tp: float = 1.5

    for clabel, cdesc, cmethod, ctp in C_CONFIGS:
        te_t = _run(df_te, rets_full, best_interval, best_N, cmethod, ctp)
        n    = len(te_t); wr = _wr(te_t); pf = _pf(te_t); sr = _sr(te_t)
        ret  = _ret(te_t); mdd = _mdd(te_t)
        star = " *" if sr > best_C_SR else ""
        if sr > best_C_SR:
            best_C_SR = sr; best_C = clabel; best_exit = cmethod; best_tp = ctp
        print(f"  {clabel:>4} | {cdesc:>14} | {n:>4} | {wr:>4.1f}% | {_fmt(pf):>6} | "
              f"{sr:>+7.3f} | {ret:>+6.1f}% | {mdd:>6.1f}%{star}")
        partC_rows.append({
            "config": clabel, "exit": cdesc, "N": n, "WR": round(wr, 1),
            "PF": round(pf, 3), "SR": round(sr, 3),
            "Return": round(ret, 1), "MaxDD": round(mdd, 1),
        })

    _save_csv(partC_rows, "p21_partC_exits.csv")
    print(f"\n  Best exit: {best_C}  SR={best_C_SR:+.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART D — Risk sensitivity
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  PART D — RISK SENSITIVITY  ({best_interval}, N={best_N}, exit={best_C})")
    print(f"\n  {'risk%':>6} | {'avg_c':>6} | {'PF':>6} | {'SR':>7} | {'Ret%':>7} | {'MaxDD%':>7}")
    print(f"  {'-' * 52}")

    for rp in RISK_PCTS:
        te_t = _run(df_te, rets_full, best_interval, best_N, best_exit, best_tp, rp)
        print(f"  {rp:>5.1f}% | {_avg_c(te_t):>6.1f} | {_fmt(_pf(te_t)):>6} | "
              f"{_sr(te_t):>+7.3f} | {_ret(te_t):>+6.1f}% | {_mdd(te_t):>6.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # PART E — WFO (if triggered)
    # ─────────────────────────────────────────────────────────────────────────
    final_te = _run(df_te, rets_full, best_interval, best_N, best_exit, best_tp, 1.0)
    n_te_final = len(final_te)

    if best_C_SR < 0.5 or n_te_final < 80:
        print(f"\n{'=' * W}")
        print("  PART E — WFO NOT TRIGGERED")
        print(f"  SR_test={best_C_SR:.3f} (needs >0.5)  "
              f"N_test={n_te_final} (needs >=80)")
        _print_summary(best_interval, best_N, best_C, 1.0, final_te)
        return

    print(f"\n{'=' * W}")
    print("  PART E — WALK-FORWARD VALIDATION")
    print(f"  Interval re-selected per window via Part A logic (N=1, hard exit)")
    print(f"  Applied with: N={best_N}  exit={best_C}  risk=1.0%")

    all_oos     = []
    window_rows = []

    for vnum, tr_s, tr_e, te_s, te_e in WFO_WINDOWS:
        df_tr_w = df.loc[tr_s:tr_e]
        df_te_w = df.loc[te_s:te_e]

        win_iv = _select_best_interval(df_tr_w, rets_full)
        te_t   = _run(df_te_w, rets_full, win_iv, best_N, best_exit, best_tp, 1.0)
        all_oos.extend(te_t)

        n   = len(te_t); pf = _pf(te_t); sr = _sr(te_t)
        window_rows.append({
            "window": vnum, "test_start": te_s, "test_end": te_e,
            "interval": win_iv, "N": n, "WR": round(_wr(te_t), 1),
            "PF": round(pf, 3), "SR": round(sr, 3),
            "Return": round(_ret(te_t), 1), "MaxDD": round(_mdd(te_t), 1),
        })

        print(f"\n  V{vnum}  {tr_s} -> {tr_e}  |  Test {te_s} -> {te_e}")
        print(f"       Interval={win_iv}  N={n}  WR={_wr(te_t):.1f}%  "
              f"PF={_fmt(pf)}  SR={sr:+.3f}  Ret={_ret(te_t):+.1f}%")

        if vnum == 5:
            t25 = [t for t in te_t if t.date.year == 2025]
            t26 = [t for t in te_t if t.date.year == 2026]
            print(f"         2025: N={len(t25)}  PF={_fmt(_pf(t25))}  SR={_sr(t25):+.3f}")
            print(f"         2026: N={len(t26)}  PF={_fmt(_pf(t26))}  SR={_sr(t26):+.3f}")

    # WFO summary table
    print(f"\n{'=' * W}")
    print("  WFO SUMMARY TABLE")
    print(f"  {'V':>2} | {'Test period':<22} | {'Intv':>5} | {'N':>4} | "
          f"{'PF':>6} | {'SR':>7} | {'Ret%':>7}")
    print(f"  {'-' * 62}")
    for r in window_rows:
        print(f"  {r['window']:>2} | {r['test_start']} -> {r['test_end']} | "
              f"{r['interval']:>5} | {r['N']:>4} | "
              f"{_fmt(r['PF']):>6} | {r['SR']:>+7.3f} | {r['Return']:>+6.1f}%")

    # Statistical tests
    pnl_arr     = np.array([t.pnl_net for t in all_oos])
    t_stat, p_v = stats.ttest_1samp(pnl_arr, 0.0, alternative="greater")
    bm, bp5, bp50, bp95 = _bootstrap_pf(pnl_arr)

    sr_wins  = sum(1 for r in window_rows if r["SR"] > 0)
    pf_wins  = sum(1 for r in window_rows if r["PF"] > 1.0)
    pool_pf  = _pf(all_oos); pool_sr = _sr(all_oos)

    print(f"\n  Pooled OOS: N={len(all_oos)}  WR={_wr(all_oos):.1f}%  "
          f"PF={_fmt(pool_pf)}  SR={pool_sr:+.3f}  Ret={_ret(all_oos):+.1f}%")
    print(f"\n  T-test (H0: mean pnl <= 0):")
    print(f"    t={t_stat:.4f}  p={p_v:.4f}")
    print(f"\n  Bootstrap PF (1000 iter, seed=42):")
    print(f"    mean={bm:.4f}  p5={bp5:.4f}  p50={bp50:.4f}  p95={bp95:.4f}")
    print(f"\n  Windows SR>0: {sr_wins}/5  |  Windows PF>1.0: {pf_wins}/5")

    passes = []; failures = []
    if p_v < 0.10:   passes.append(f"p={p_v:.4f} < 0.10")
    else:            failures.append(f"p={p_v:.4f} >= 0.10")
    if bp5 > 0.95:   passes.append(f"bootstrap p5={bp5:.4f} > 0.95")
    else:            failures.append(f"bootstrap p5={bp5:.4f} <= 0.95")
    if sr_wins >= 3: passes.append(f"{sr_wins}/5 windows SR>0")
    else:            failures.append(f"only {sr_wins}/5 windows SR>0")

    confirmed = len(failures) == 0
    print(f"\n  VERDICT: {'EDGE CONFIRMED' if confirmed else 'Edge NOT confirmed'}")
    if passes:   print(f"  Pass:  {' | '.join(passes)}")
    if failures: print(f"  Fail:  {' | '.join(failures)}")
    if confirmed:
        print("  -> Proceed to Monte Carlo FTMO sizing.")

    # Save
    _save_trades(all_oos, "p21_wfo_oos_pooled.csv")
    summary = list(window_rows)
    summary.append({
        "window": "pooled", "test_start": "all", "test_end": "all",
        "interval": "mixed", "N": len(all_oos), "WR": round(_wr(all_oos), 1),
        "PF": round(pool_pf, 3), "SR": round(pool_sr, 3),
        "Return": round(_ret(all_oos), 1), "MaxDD": round(_mdd(all_oos), 1),
    })
    summary.append({
        "window": "stats", "test_start": "t_stat", "test_end": str(round(t_stat, 4)),
        "interval": "—", "N": None, "WR": None,
        "PF": round(bm, 4), "SR": round(p_v, 4),
        "Return": round(bp5, 4), "MaxDD": round(bp95, 4),
    })
    _save_csv(summary, "p21_wfo_summary.csv")
    print(f"\n  Saved: results/p21_wfo_oos_pooled.csv")
    print(f"         results/p21_wfo_summary.csv")

    _print_summary(best_interval, best_N, best_C, 1.0, final_te)


def _print_summary(iname, N, exit_label, risk_pct, te_trades):
    print(f"\n{'=' * W}")
    print("  SUMMARY")
    print(f"  Best interval:  {iname}")
    print(f"  Best lookback:  N={N}")
    print(f"  Best exit:      {exit_label}")
    print(f"  Risk:           {risk_pct}%")
    print(f"  PF test:        {_fmt(_pf(te_trades))}")
    print(f"  SR test:        {_sr(te_trades):+.3f}")
    print(f"  N test:         {len(te_trades)}")
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
