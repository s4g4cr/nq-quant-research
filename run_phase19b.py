#!/usr/bin/env python3
"""
Phase 19B — Intraday Reversal NQ (inverse Gao et al. 2018).

Direction: SHORT if r1 > 0, LONG if r1 < 0.
Diagnostic is informational only — all 7 experiments run regardless.
"""
import math
import os
import sys
from datetime import time as dt_time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import atr as compute_atr, avg_volume
from orb_system.strategy.intraday_reversal import (
    SessionInfo, MomentumTrade, detect_sessions, run,
    SLIP, COMM, PV,
)
from orb_system.strategy.intraday_momentum import _ENTRY_T, _EXIT_T

INITIAL_CAP = 100_000.0
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
W           = 72
RESULTS_DIR = os.path.join(ROOT, "results")

WFO_WINDOWS = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]


# ── stat helpers ──────────────────────────────────────────────────────────────

def _pf(trades):
    v = np.array([t.pnl_net for t in trades])
    w = v[v > 0]; l = v[v <= 0]
    gw = float(w.sum()) if w.size else 0.0
    gl = float(abs(l.sum())) if l.size else 0.0
    return gw / gl if gl > 0 else float("inf")

def _sr(trades):
    if not trades:
        return 0.0
    by_date: dict = {}
    for t in trades:
        d = str(t.date)
        by_date[d] = by_date.get(d, 0.0) + t.pnl_net
    v = np.array(list(by_date.values()))
    s = float(v.std())
    return float(v.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _wr(trades):
    return sum(1 for t in trades if t.pnl_net > 0) / len(trades) if trades else 0.0

def _max_dd(trades):
    cap = peak = INITIAL_CAP; worst = 0.0
    for t in trades:
        cap += t.pnl_net; peak = max(peak, cap)
        worst = max(worst, (peak - cap) / peak)
    return worst

def _tpd(trades, df):
    n_days = len(np.unique(np.array(df.index.date)))
    return len(trades) / n_days if n_days > 0 else 0.0

def _ret(trades):
    return sum(t.pnl_net for t in trades) / INITIAL_CAP * 100.0

def _avg_w_l(trades):
    wins  = [t.pnl_net for t in trades if t.pnl_net > 0]
    loses = [t.pnl_net for t in trades if t.pnl_net <= 0]
    aw  = float(np.mean(wins))  if wins  else 0.0
    al  = float(np.mean(loses)) if loses else 0.0
    rr  = abs(aw / al) if al != 0 else float("inf")
    awp = float(np.mean([t.pnl_pts for t in trades if t.pnl_net > 0])) if wins  else 0.0
    alp = float(np.mean([t.pnl_pts for t in trades if t.pnl_net <= 0])) if loses else 0.0
    return aw, awp, al, alp, rr

def _nc(trades):
    return float(np.mean([t.n_contracts for t in trades])) if trades else 0.0

def _annual(trades):
    by_year: dict = {}
    for t in trades:
        y = pd.Timestamp(str(t.date)).year
        by_year.setdefault(y, []).append(t)
    return [(y, len(tt), _wr(tt)*100, _pf(tt), _ret(tt))
            for y, tt in sorted(by_year.items())]

def _fmt(v, d=3):
    if v != v or v == float("inf"): return "  inf"
    return f"{v:.{d}f}"


# ── diagnostic (informational — never stops experiments) ──────────────────────

def run_diagnostic(session_infos: dict) -> None:
    valid = [si for si in session_infos.values()
             if si is not None and not np.isnan(si.r1)]
    n_all = len(valid)

    print(f"\n{'─'*W}")
    print("  DIAGNOSTIC — INFORMATIONAL (full dataset · no gate)")
    print(f"{'─'*W}")

    r1_vals = np.array([si.r1 for si in valid])
    n_pos   = int((r1_vals > 0).sum())
    n_neg   = int((r1_vals < 0).sum())
    n_zer   = int((r1_vals == 0).sum())
    p_r1    = np.percentile(r1_vals * 100, [10, 25, 50, 75, 90])

    print(f"\n  1. FIRST HALF-HOUR RETURN (r1)  [sessions: {n_all}]")
    print(f"     p10={p_r1[0]:.3f}%  p25={p_r1[1]:.3f}%  p50={p_r1[2]:.3f}%"
          f"  p75={p_r1[3]:.3f}%  p90={p_r1[4]:.3f}%")
    print(f"     r1 > 0 (signal: SHORT): {n_pos:4d} ({100*n_pos/n_all:.1f}%)")
    print(f"     r1 < 0 (signal: LONG):  {n_neg:4d} ({100*n_neg/n_all:.1f}%)")
    print(f"     r1 = 0 (skip):          {n_zer:4d} ({100*n_zer/n_all:.1f}%)")

    r17_vals = np.array([si.r17 for si in valid if not np.isnan(si.r17)])
    p_r17    = np.percentile(r17_vals * 100, [10, 25, 50, 75, 90])
    print(f"\n  2. LAST HALF-HOUR RETURN (r17)  [sessions: {len(r17_vals)}]")
    print(f"     p10={p_r17[0]:.3f}%  p25={p_r17[1]:.3f}%  p50={p_r17[2]:.3f}%"
          f"  p75={p_r17[3]:.3f}%  p90={p_r17[4]:.3f}%")
    print(f"     Unconditional mean r17: {np.mean(r17_vals)*100:.4f}%")

    print(f"\n  3. CONDITIONAL ANALYSIS (reversal direction)")
    long_r17  = np.array([si.r17 for si in valid if si.r1 < 0 and not np.isnan(si.r17)])
    short_r17 = np.array([si.r17 for si in valid if si.r1 > 0 and not np.isnan(si.r17)])

    # For reversal: LONG when r1<0 → correct if r17>0; SHORT when r1>0 → correct if r17<0
    long_corr  = float((long_r17 > 0).mean())  if len(long_r17)  > 0 else 0.0
    short_corr = float((short_r17 < 0).mean()) if len(short_r17) > 0 else 0.0
    total_corr = (long_r17 > 0).sum() + (short_r17 < 0).sum()
    total_n    = len(long_r17) + len(short_r17)
    dir_acc    = total_corr / total_n if total_n > 0 else 0.0

    t_l, p_l = stats.ttest_1samp(long_r17,  0.0, alternative="greater") if len(long_r17)  > 1 else (0., 1.)
    t_s, p_s = stats.ttest_1samp(short_r17, 0.0, alternative="less")    if len(short_r17) > 1 else (0., 1.)
    t_2, p_2 = stats.ttest_ind(short_r17, long_r17, alternative="less") if (len(long_r17)>1 and len(short_r17)>1) else (0., 1.)

    print(f"  {'Condition':<28} {'N':>5} {'Mean r17':>9} {'WR%':>7} {'t':>7} {'p':>7}")
    print(f"  {'─'*64}")
    print(f"  {'r1 > 0 → SHORT (correct<0)':<28} {len(short_r17):>5}"
          f" {short_r17.mean()*100:>8.4f}%"
          f" {short_corr*100:>6.1f}%"
          f" {t_s:>7.3f} {p_s:>7.4f}")
    print(f"  {'r1 < 0 → LONG  (correct>0)':<28} {len(long_r17):>5}"
          f" {long_r17.mean()*100:>8.4f}%"
          f" {long_corr*100:>6.1f}%"
          f" {t_l:>7.3f} {p_l:>7.4f}")
    print(f"  {'Two-sample reversal test':<28} {'':>5} {'':>9}"
          f" {'':>7} {t_2:>7.3f} {p_2:>7.4f}")
    print(f"\n  Directional accuracy (reversal): {dir_acc*100:.1f}%"
          f"  (SHORT side: {short_corr*100:.1f}%  LONG side: {long_corr*100:.1f}%)")

    if dir_acc >= 0.50:
        print(f"  → ≥50% directional accuracy — reversal signal present in data.")
    else:
        print(f"  → <50% directional accuracy — reversal weak on full dataset.")
    print(f"  Diagnostic complete. All 7 experiments will run regardless.")

    print(f"\n  4. VOLATILITY CONDITIONING (informational)")
    med_rv = float(np.median([si.range_first30 for si in valid]))
    for label, slist in [("High vol", [s for s in valid if s.range_first30 > med_rv]),
                          ("Low vol",  [s for s in valid if s.range_first30 <= med_rv])]:
        l17   = np.array([s.r17 for s in slist if s.r1 < 0 and not np.isnan(s.r17)])
        s17   = np.array([s.r17 for s in slist if s.r1 > 0 and not np.isnan(s.r17)])
        n_v   = len(l17) + len(s17)
        acc_v = ((l17 > 0).sum() + (s17 < 0).sum()) / n_v if n_v > 0 else 0.0
        ml17  = l17.mean()*100 if len(l17) > 0 else 0.0
        ms17  = s17.mean()*100 if len(s17) > 0 else 0.0
        print(f"     {label:<10}  N={n_v:4d}  acc={acc_v*100:.1f}%"
              f"  mean_r17|LONG={ml17:.4f}%  mean_r17|SHORT={ms17:.4f}%")

    print(f"\n  5. SIGNAL FREQUENCY")
    tradeable = sum(1 for si in valid if si.r1 != 0.0)
    print(f"     Tradeable days: {tradeable} / {n_all}"
          f" ({100*tradeable/max(1,n_all):.1f}%)  ≈ {tradeable/max(1,n_all):.3f} trades/day")


# ── metrics printer ───────────────────────────────────────────────────────────

def print_block(label: str, trades, df, show_annual=True, show_2025_2026=False):
    if not trades:
        print(f"  {label}: No trades")
        return
    n   = len(trades)
    tpd = _tpd(trades, df)
    wr  = _wr(trades) * 100
    pf  = _pf(trades)
    sr  = _sr(trades)
    ret = _ret(trades)
    mdd = _max_dd(trades) * 100
    aw, awp, al, alp, rr = _avg_w_l(trades)
    nc  = _nc(trades)

    print(f"  {label}: N={n}  TPD={tpd:.3f}  WR={wr:.1f}%  PF={_fmt(pf)}  "
          f"SR={sr:.3f}  Ret={ret:.1f}%  MaxDD={mdd:.1f}%")
    print(f"         AvgW=${aw:.0f}({awp:.1f}pt)  AvgL=${al:.0f}({alp:.1f}pt)"
          f"  R/R={_fmt(rr,2)}  AvgContr={nc:.1f}")

    r1_arr  = np.array([t.r1 for t in trades])
    r17_arr = np.array([t.r17 for t in trades if not np.isnan(t.r17)])
    if len(r17_arr) > 1:
        corr = np.corrcoef(r1_arr[:len(r17_arr)], r17_arr)[0, 1]
        print(f"         corr(r1,r17)={corr:.4f}"
              f"  (negative = reversal signal working)")

    if show_annual:
        for y, n_y, wr_y, pf_y, ret_y in _annual(trades):
            print(f"           {y}: N={n_y:3d}  WR={wr_y:.1f}%  "
                  f"PF={_fmt(pf_y)}  Ret={ret_y:.1f}%")

    if show_2025_2026:
        for yr in [2025, 2026]:
            tt = [t for t in trades if pd.Timestamp(str(t.date)).year == yr]
            if tt:
                print(f"         {yr}: N={len(tt):3d}  WR={_wr(tt)*100:.1f}%  "
                      f"PF={_fmt(_pf(tt))}  SR={_sr(tt):.3f}  Ret={_ret(tt):.1f}%")
            else:
                print(f"         {yr}: No trades")


def print_exp_header(num, name, params):
    print(f"\n{'═'*W}")
    print(f"  EXPERIMENT {num} — {name}")
    print(f"  {params}")
    print(f"{'═'*W}")


def _save_csv(trades, fname):
    if not trades:
        return
    rows = [{"date": t.date, "direction": t.direction,
             "r1": t.r1, "r17": t.r17, "r16": t.r16,
             "range_first30": t.range_first30, "vol_first30": t.vol_first30,
             "entry_ts": t.entry_ts, "entry_price": t.entry_price,
             "exit_ts": t.exit_ts, "exit_price": t.exit_price,
             "n_contracts": t.n_contracts, "atr_at_entry": t.atr_at_entry,
             "pnl_pts": t.pnl_pts, "pnl_net": t.pnl_net}
            for t in trades]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, fname), index=False)


def _split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te, kw, save_name=None):
    tr = run(df_tr, si_tr, a1_tr, **kw)
    te = run(df_te, si_te, a1_te, **kw)
    if save_name:
        _save_csv(tr + te, save_name)
    return tr, te


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_wfo(df, atr1m, best_cfg: dict):
    print(f"\n{'═'*W}")
    print("  WALK-FORWARD VALIDATION (5 anchored windows)")
    print(f"  Config: {best_cfg}")
    print(f"{'═'*W}")

    all_oos = []
    sr_wins = 0

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(WFO_WINDOWS, 1):
        df_te = df.loc[te_s:te_e]
        a1_te = atr1m[df_te.index]
        si_te = detect_sessions(df_te, a1_te)

        kw = {k: v for k, v in best_cfg.items()
              if k not in ("rv_median_dict", "vol_median_dict")}
        kw["rv_median_dict"]  = best_cfg.get("rv_median_dict")
        kw["vol_median_dict"] = best_cfg.get("vol_median_dict")
        kw["initial_capital"] = INITIAL_CAP

        te_trades = run(df_te, si_te, a1_te, **kw)
        all_oos.extend(te_trades)
        sr_v = _sr(te_trades)
        if sr_v > 0:
            sr_wins += 1
        print(f"  V{i}  Train {tr_s}–{tr_e} | Test {te_s}–{te_e}")
        print(f"       N={len(te_trades):3d}  WR={_wr(te_trades)*100:.1f}%  "
              f"PF={_fmt(_pf(te_trades))}  SR={sr_v:.3f}  "
              f"Ret={_ret(te_trades):.1f}%  MaxDD={_max_dd(te_trades)*100:.1f}%")

    print(f"\n  WFO Summary: Windows SR>0: {sr_wins}/5")

    if not all_oos:
        print("  No OOS trades.")
        return

    pnl_arr     = np.array([t.pnl_net for t in all_oos])
    t_stat, p_v = stats.ttest_1samp(pnl_arr, 0.0, alternative="greater")

    rng      = np.random.default_rng(42)
    boot_pf  = []
    for _ in range(1000):
        s  = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
        gw = s[s > 0].sum(); gl = abs(s[s <= 0].sum())
        boot_pf.append(gw / gl if gl > 0 else float("inf"))
    finite = [x for x in boot_pf if x != float("inf")]
    bp5    = float(np.percentile(finite, 5))
    bmn    = float(np.mean(finite))

    print(f"\n  POOLED OOS  ({len(all_oos)} trades)")
    print_block("OOS", all_oos, df, show_annual=False)
    print(f"  T-test:    t={t_stat:.3f}  p={p_v:.3f}")
    print(f"  Bootstrap: mean_PF={bmn:.3f}  p5={bp5:.3f}")

    edge = (p_v < 0.10 and bp5 > 0.95 and sr_wins >= 3)
    print(f"\n  VERDICT: {'EDGE CONFIRMED' if edge else 'Edge NOT confirmed'}")
    if not edge:
        fail = []
        if p_v >= 0.10:    fail.append(f"p={p_v:.3f}≥0.10")
        if bp5 <= 0.95:    fail.append(f"bootstrap p5={bp5:.3f}≤0.95")
        if sr_wins < 3:    fail.append(f"{sr_wins}/5 windows SR>0")
        print(f"  Failures: {', '.join(fail)}")
    else:
        print("  Proceed to Monte Carlo FTMO sizing (Phase 20).")

    _save_csv(all_oos, "p19b_wfo_oos_pooled.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 19B — INTRADAY REVERSAL NQ (INVERSE GAO ET AL. 2018)")
    print("  Direction: SHORT if r1>0 · LONG if r1<0")
    print("=" * W)

    cfg    = Config()
    df     = load_data(cfg)
    print(f"  Data: {df.index[0]}  →  {df.index[-1]}  ({len(df):,} bars)")

    print("  Computing ATR(20)...")
    atr1m   = compute_atr(df, 20)
    avg_vol = avg_volume(df, 20)

    print("  Detecting sessions...")
    si_full = detect_sessions(df, atr1m)
    valid   = [si for si in si_full.values() if si is not None]
    print(f"  Sessions with data: {len(valid)}")

    # Rolling medians (causal)
    si_sorted = sorted(valid, key=lambda s: s.date)
    dates_s   = [si.date for si in si_sorted]
    rv_arr    = np.array([si.range_first30 for si in si_sorted])
    vol_arr   = np.array([si.vol_first30   for si in si_sorted])

    rv_med_dict  = {}
    vol_med_dict = {}
    for i, d in enumerate(dates_s):
        if i >= 20:
            rv_med_dict[d]  = float(np.median(rv_arr[i-20:i]))
            vol_med_dict[d] = float(np.median(vol_arr[i-20:i]))

    run_diagnostic(si_full)

    # Split
    df_tr  = df.loc[:TRAIN_END];   df_te  = df.loc[TEST_START:]
    a1_tr  = atr1m[df_tr.index];  a1_te  = atr1m[df_te.index]
    si_tr  = detect_sessions(df_tr, a1_tr)
    si_te  = detect_sessions(df_te, a1_te)

    print(f"\n  Train: {df_tr.index[0].date()} → {df_tr.index[-1].date()}")
    print(f"  Test:  {df_te.index[0].date()} → {df_te.index[-1].date()}")

    # ── Experiment 1 — Baseline reversal ─────────────────────────────────────
    kw1 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
                initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(1, "Baseline reversal (strict inverse of Phase 19 Exp 1)",
                     "entry=15:30  exit=15:59  direction=INVERTED  no filters  risk=1.0%")
    tr1, te1 = _split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                      kw1, save_name="p19b_exp1_baseline.csv")
    print_block("TRAIN", tr1, df_tr)
    print_block("TEST ", te1, df_te, show_2025_2026=True)
    sr1_te = _sr(te1)

    # ── Experiment 2 — Magnitude filter ──────────────────────────────────────
    print_exp_header(2, "Magnitude filter  |r1| > threshold",
                     "thresholds: [0.1%, 0.2%, 0.3%, 0.5%]  entry=15:30")
    print(f"  {'thresh':>7} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>7}")
    print(f"  {'-'*62}")

    best2_thr = 0.0; best2_sr = sr1_te
    exp2_data = {}
    for thr_pct in [0.0, 0.1, 0.2, 0.3, 0.5]:
        thr = thr_pct / 100.0
        kw  = dict(entry_bar_time=_ENTRY_T, r1_threshold=thr,
                   initial_capital=INITIAL_CAP, risk_pct=1.0)
        tr_x = run(df_tr, si_tr, a1_tr, **kw)
        te_x = run(df_te, si_te, a1_te, **kw)
        sr_te = _sr(te_x)
        print(f"  {thr_pct:>6.1f}% | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x)):>6} | "
              f"{_fmt(_pf(te_x)):>6} | {sr_te:>7.3f}")
        exp2_data[thr_pct] = (tr_x, te_x, thr)
        if sr_te > best2_sr and len(te_x) >= 30:
            best2_sr = sr_te; best2_thr = thr
    use_thr  = best2_thr > 0
    best2_pct = best2_thr * 100
    print(f"  → Best threshold: {best2_pct:.1f}%"
          f"  ({'Improves over baseline' if use_thr else 'No improvement — use 0%'})")

    # ── Experiment 3 — Volatility conditioning ────────────────────────────────
    kw3 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
               high_vol_only=True, rv_median_dict=rv_med_dict,
               initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(3, "Volatility conditioning — high-vol days only",
                     "range_first30 > 20-session rolling median  entry=15:30")
    tr3, te3 = _split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                      kw3, save_name="p19b_exp3_hv.csv")
    print_block("TRAIN", tr3, df_tr)
    print_block("TEST ", te3, df_te, show_2025_2026=True)
    use_hv  = _sr(te3) > sr1_te and len(te3) >= 30
    print(f"\n  Vol cond SR={_sr(te3):.3f}  vs  baseline SR={sr1_te:.3f}"
          f"  → {'Include in combined' if use_hv else 'Skip in combined'}")

    # ── Experiment 4 — Entry time sensitivity ────────────────────────────────
    print_exp_header(4, "Entry time sensitivity",
                     "exit always 15:59  entry times: [14:30, 15:00, 15:30, 15:45]")
    print(f"  {'entry':>7} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>7}")
    print(f"  {'-'*62}")

    entry_times = {
        "14:30": dt_time(14, 29),
        "15:00": dt_time(14, 59),
        "15:30": dt_time(15, 29),
        "15:45": dt_time(15, 44),
    }
    best4_et = _ENTRY_T; best4_sr = sr1_te; best4_lbl = "15:30"
    for label, et in entry_times.items():
        kw    = dict(entry_bar_time=et, r1_threshold=0.0,
                     initial_capital=INITIAL_CAP, risk_pct=1.0)
        tr_x  = run(df_tr, si_tr, a1_tr, **kw)
        te_x  = run(df_te, si_te, a1_te, **kw)
        sr_te = _sr(te_x)
        print(f"  {label:>7} | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x)):>6} | "
              f"{_fmt(_pf(te_x)):>6} | {sr_te:>7.3f}")
        if sr_te > best4_sr and len(te_x) >= 30:
            best4_sr = sr_te; best4_et = et; best4_lbl = label
    print(f"  → Best entry time: {best4_lbl}"
          f"  ({'improved' if best4_lbl != '15:30' else 'no change from baseline'})")

    # ── Experiment 5 — r16 agreement filter ──────────────────────────────────
    kw5 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
               r16_agreement=True, initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(5, "r16 agreement filter  sign(r1) == sign(r16)",
                     "r16 = 14:30–15:00 return  confirms morning direction  entry=15:30")
    tr5, te5 = _split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                      kw5, save_name="p19b_exp5_r16.csv")
    print_block("TRAIN", tr5, df_tr)
    print_block("TEST ", te5, df_te, show_2025_2026=True)
    use_r16 = _sr(te5) > sr1_te and len(te5) >= 30
    print(f"\n  r16 agree SR={_sr(te5):.3f}  vs  baseline SR={sr1_te:.3f}"
          f"  → {'Include in combined' if use_r16 else 'Skip in combined'}")

    # ── Experiment 6 — Volume filter ─────────────────────────────────────────
    kw6 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
               vol_filter=True, vol_median_dict=vol_med_dict,
               initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(6, "Volume filter — first-30min volume > 20-session median",
                     "vol_first30 > rolling median  entry=15:30")
    tr6, te6 = _split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                      kw6, save_name="p19b_exp6_vol.csv")
    print_block("TRAIN", tr6, df_tr)
    print_block("TEST ", te6, df_te, show_2025_2026=True)
    use_vol = _sr(te6) > sr1_te and len(te6) >= 30
    print(f"\n  Vol filter SR={_sr(te6):.3f}  vs  baseline SR={sr1_te:.3f}"
          f"  → {'Include in combined' if use_vol else 'Skip in combined'}")

    # ── Experiment 7 — Combined best ─────────────────────────────────────────
    best_kw = dict(
        entry_bar_time  = best4_et,
        r1_threshold    = best2_thr,
        high_vol_only   = use_hv,
        rv_median_dict  = rv_med_dict if use_hv  else None,
        vol_filter      = use_vol,
        vol_median_dict = vol_med_dict if use_vol else None,
        r16_agreement   = use_r16,
        initial_capital = INITIAL_CAP,
        risk_pct        = 1.0,
    )
    filter_parts = []
    if best2_thr > 0:   filter_parts.append(f"|r1|>{best2_pct:.1f}%")
    if use_hv:          filter_parts.append("high_vol")
    if use_r16:         filter_parts.append("r16_agree")
    if use_vol:         filter_parts.append("high_vol30")
    filter_desc = "  ".join(filter_parts) if filter_parts else "none"

    print_exp_header(7, "Combined best configuration",
                     f"entry={best4_lbl}  filters={filter_desc}  risk=1.0%")
    tr7, te7 = _split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                      best_kw, save_name="p19b_exp7_combined.csv")
    print_block("TRAIN", tr7, df_tr)
    print_block("TEST ", te7, df_te, show_2025_2026=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  SUMMARY TABLE")
    print(f"  {'Exp':>3} | {'Entry':>6} | {'Filters':<18} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>7} | {'TPD':>5} | {'WR%':>5}")
    print(f"  {'-'*80}")

    rows = [
        (1, "15:30", "none",            tr1, te1),
        (3, "15:30", "high_vol",        tr3, te3),
        (5, "15:30", "r16_agree",       tr5, te5),
        (6, "15:30", "high_vol30",      tr6, te6),
        (7, best4_lbl, filter_desc[:18], tr7, te7),
    ]
    for (en, et, filt, trr, ter) in rows:
        print(f"  {en:>3} | {et:>6} | {filt:<18} | "
              f"{_fmt(_pf(trr)):>6} | {_fmt(_pf(ter)):>6} | "
              f"{_sr(ter):>7.3f} | {_tpd(ter, df_te):>5.3f} | "
              f"{_wr(ter)*100:>4.1f}%")

    # ── Auto-interpretation ───────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  AUTO-INTERPRETATION")
    sr7_te = _sr(te7); n7_te = len(te7)
    all_srs = [_sr(te1), _sr(te3), _sr(te5), _sr(te6), sr7_te]
    any_positive = any(s > 0 for s in all_srs)
    any_flagged  = any(s > 0.5 and n >= 80
                       for s, n in [(sr7_te, n7_te)])

    print(f"  Exp 7 test: SR={sr7_te:.3f}  N={n7_te}")
    if not any_positive:
        print("  → All SR_test < 0. Reversal not tradeable with these parameters.")
        print("  → Consider Hypothesis 3 (different signal or instrument).")
    elif any_flagged:
        print("  → SR>0.5 and N>=80 threshold met for Exp 7. Running walk-forward.")
    else:
        print("  → Positive SR in some experiments but Exp 7 below WFO threshold.")
        print(f"     (Requires SR>0.5 and N>=80 on Exp 7 — got SR={sr7_te:.3f}, N={n7_te})")

    if any_flagged:
        run_wfo(df, atr1m, {k: v for k, v in best_kw.items()
                             if k != "initial_capital"})

    print(f"\n{'='*W}")
    print("  PHASE 19B COMPLETE")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
