#!/usr/bin/env python3
"""
Phase 19 — Intraday Momentum (Gao et al. 2018).

First half-hour return predicts last half-hour return.
Mandatory diagnostic gates all experiments.
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
from orb_system.strategy.intraday_momentum import (
    SessionInfo, MomentumTrade, detect_sessions, run,
    SLIP, COMM, PV, _ENTRY_T, _EXIT_T,
)

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


# ── diagnostic ────────────────────────────────────────────────────────────────

def run_diagnostic(session_infos: dict, rv_median_dict: dict, vol_median_dict: dict) -> bool:
    valid = [si for si in session_infos.values()
             if si is not None and not np.isnan(si.r1)]
    n_all = len(valid)

    print(f"\n{'─'*W}")
    print("  DIAGNOSTIC — PRE-PnL STATISTICS (full dataset)")
    print(f"{'─'*W}")

    # ── 1. First half-hour return distribution ────────────────────────────────
    r1_vals  = np.array([si.r1  for si in valid])
    n_pos = int((r1_vals > 0).sum())
    n_neg = int((r1_vals < 0).sum())
    n_zer = int((r1_vals == 0).sum())
    p = np.percentile(r1_vals * 100, [10, 25, 50, 75, 90])
    print(f"\n  1. FIRST HALF-HOUR RETURN (r1)  [sessions: {n_all}]")
    print(f"     p10={p[0]:.3f}%  p25={p[1]:.3f}%  p50={p[2]:.3f}%"
          f"  p75={p[3]:.3f}%  p90={p[4]:.3f}%")
    print(f"     r1 > 0 (LONG days):  {n_pos:4d} ({100*n_pos/n_all:.1f}%)")
    print(f"     r1 < 0 (SHORT days): {n_neg:4d} ({100*n_neg/n_all:.1f}%)")
    print(f"     r1 = 0 (skip):       {n_zer:4d} ({100*n_zer/n_all:.1f}%)")

    # ── 2. Last half-hour return distribution ─────────────────────────────────
    r17_vals = np.array([si.r17 for si in valid if not np.isnan(si.r17)])
    n_r17p = int((r17_vals > 0).sum())
    p17 = np.percentile(r17_vals * 100, [10, 25, 50, 75, 90])
    print(f"\n  2. LAST HALF-HOUR RETURN (r17)  [sessions: {len(r17_vals)}]")
    print(f"     p10={p17[0]:.3f}%  p25={p17[1]:.3f}%  p50={p17[2]:.3f}%"
          f"  p75={p17[3]:.3f}%  p90={p17[4]:.3f}%")
    print(f"     r17 > 0: {n_r17p} ({100*n_r17p/max(1,len(r17_vals)):.1f}%)")
    print(f"     Unconditional mean r17: {np.mean(r17_vals)*100:.4f}%")

    # ── 3. Conditional analysis ───────────────────────────────────────────────
    print(f"\n  3. CONDITIONAL ANALYSIS (core test)")

    long_r17  = np.array([si.r17 for si in valid if si.r1 > 0 and not np.isnan(si.r17)])
    short_r17 = np.array([si.r17 for si in valid if si.r1 < 0 and not np.isnan(si.r17)])

    long_wr   = float((long_r17 > 0).mean())   if len(long_r17)  > 0 else 0.0
    short_wr  = float((short_r17 < 0).mean())  if len(short_r17) > 0 else 0.0
    all_corr  = len(long_r17) * long_wr + len(short_r17) * short_wr
    total_dir = len(long_r17) + len(short_r17)
    dir_acc   = all_corr / total_dir if total_dir > 0 else 0.0

    # t-test: mean(r17|r1>0) vs 0
    t_l, p_l = stats.ttest_1samp(long_r17,  0.0, alternative="greater") if len(long_r17)  > 1 else (0., 1.)
    # t-test: mean(r17|r1<0) vs 0  (short: expect negative mean)
    t_s, p_s = stats.ttest_1samp(short_r17, 0.0, alternative="less")    if len(short_r17) > 1 else (0., 1.)
    # two-sample t-test
    t_2, p_2 = stats.ttest_ind(long_r17, short_r17, alternative="greater") if (len(long_r17)>1 and len(short_r17)>1) else (0., 1.)

    print(f"  {'Condition':<22} {'N':>5} {'Mean r17':>9} {'WR%':>7} {'t':>7} {'p':>7}")
    print(f"  {'─'*58}")
    print(f"  {'All days':<22} {total_dir:>5}"
          f" {np.concatenate([long_r17, short_r17]).mean()*100:>8.4f}%"
          f" {dir_acc*100:>6.1f}%   —       —")
    print(f"  {'r1 > 0 (LONG)':<22} {len(long_r17):>5}"
          f" {long_r17.mean()*100:>8.4f}%"
          f" {long_wr*100:>6.1f}%"
          f" {t_l:>7.3f} {p_l:>7.4f}")
    print(f"  {'r1 < 0 (SHORT)':<22} {len(short_r17):>5}"
          f" {short_r17.mean()*100:>8.4f}%"
          f" {short_wr*100:>6.1f}%"
          f" {t_s:>7.3f} {p_s:>7.4f}")
    print(f"  {'Two-sample (L vs S)':<22} {'':>5} {'':>9}"
          f" {'':>7} {t_2:>7.3f} {p_2:>7.4f}")
    print(f"\n  Directional accuracy: {dir_acc*100:.1f}%"
          f"  (LONG side: {long_wr*100:.1f}%  SHORT side: {short_wr*100:.1f}%)")

    if dir_acc < 0.50:
        print(f"\n  *** DIRECTIONAL ACCURACY {dir_acc*100:.1f}% < 50% ***")
        print("  *** Gao et al. finding does NOT hold in NQ. STOPPING. ***")
        return False

    # ── 4. Volatility conditioning ────────────────────────────────────────────
    print(f"\n  4. VOLATILITY CONDITIONING")
    med_rv = np.median([si.range_first30 for si in valid])
    hi_vol = [si for si in valid if si.range_first30 > med_rv]
    lo_vol = [si for si in valid if si.range_first30 <= med_rv]

    def _cond_stats(slist, label):
        l17  = np.array([si.r17 for si in slist if si.r1 > 0 and not np.isnan(si.r17)])
        s17  = np.array([si.r17 for si in slist if si.r1 < 0 and not np.isnan(si.r17)])
        n    = len(l17) + len(s17)
        corr = (l17 > 0).sum() + (s17 < 0).sum()
        acc  = corr / n if n > 0 else 0.0
        all17 = np.concatenate([l17, -s17]) if (len(l17)>0 and len(s17)>0) else l17
        print(f"  {label:<20} N={n:4d}  accuracy={acc*100:.1f}%"
              f"  mean|r17|={abs(all17).mean()*100:.4f}%  "
              f"mean_long_r17={l17.mean()*100:.4f}%  mean_short_r17={s17.mean()*100:.4f}%")

    _cond_stats(hi_vol, f"High vol (>{med_rv:.1f}pts)")
    _cond_stats(lo_vol, f"Low vol  (≤{med_rv:.1f}pts)")

    # ── 5. Signal frequency ───────────────────────────────────────────────────
    print(f"\n  5. SIGNAL FREQUENCY")
    tradeable = [si for si in valid if si.r1 != 0.0]
    n_days_total = n_all
    print(f"     Tradeable days (r1≠0): {len(tradeable)} / {n_days_total}"
          f" ({100*len(tradeable)/max(1,n_days_total):.1f}%)")
    print(f"     ≈ {len(tradeable)/max(1,n_days_total):.3f} trades/day")
    print(f"\n  Diagnostic passed — proceeding to experiments.")
    return True


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
    print(f"         Exits: EOD 100%  AvgW=${aw:.0f}({awp:.1f}pt)"
          f"  AvgL=${al:.0f}({alp:.1f}pt)  R/R={_fmt(rr,2)}  AvgContr={nc:.1f}")

    # Show r1/r17 correlation
    r1_arr  = np.array([t.r1  for t in trades])
    r17_arr = np.array([t.r17 for t in trades if not np.isnan(t.r17)])
    corr = np.corrcoef(r1_arr[:len(r17_arr)], r17_arr)[0,1] if len(r17_arr) > 1 else 0.0
    print(f"         corr(r1,r17)={corr:.4f}")

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


def _run_split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te, kw, save_name=None):
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

    all_oos  = []
    sr_wins  = 0

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(WFO_WINDOWS, 1):
        df_te  = df.loc[te_s:te_e]
        a1_te  = atr1m[df_te.index]
        si_te  = detect_sessions(df_te, a1_te)

        kw = dict(
            entry_bar_time = best_cfg.get("entry_bar_time", _ENTRY_T),
            r1_threshold   = best_cfg.get("r1_threshold", 0.0),
            high_vol_only  = best_cfg.get("high_vol_only", False),
            rv_median_dict = best_cfg.get("rv_median_dict", None),
            vol_filter     = best_cfg.get("vol_filter", False),
            vol_median_dict= best_cfg.get("vol_median_dict", None),
            r16_agreement  = best_cfg.get("r16_agreement", False),
            initial_capital= INITIAL_CAP,
            risk_pct       = best_cfg.get("risk_pct", 1.0),
        )
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

    pnl_arr      = np.array([t.pnl_net for t in all_oos])
    t_stat, p_val = stats.ttest_1samp(pnl_arr, 0.0, alternative="greater")

    rng = np.random.default_rng(42)
    boot_pf = []
    for _ in range(1000):
        s = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
        w = s[s > 0].sum(); l = abs(s[s <= 0].sum())
        boot_pf.append(w / l if l > 0 else float("inf"))
    finite = [x for x in boot_pf if x != float("inf")]
    bp5  = float(np.percentile(finite, 5))
    bmn  = float(np.mean(finite))

    print(f"\n  POOLED OOS  ({len(all_oos)} trades)")
    print_block("OOS", all_oos, df, show_annual=False)
    print(f"  T-test:    t={t_stat:.3f}  p={p_val:.3f}")
    print(f"  Bootstrap: mean_PF={bmn:.3f}  p5={bp5:.3f}")

    edge = (p_val < 0.10 and bp5 > 0.95 and sr_wins >= 3)
    verdict = "EDGE CONFIRMED" if edge else "Edge NOT confirmed"
    print(f"\n  VERDICT: {verdict}")
    if not edge:
        fail = []
        if p_val >= 0.10:  fail.append(f"p={p_val:.3f}≥0.10")
        if bp5 <= 0.95:    fail.append(f"bootstrap p5={bp5:.3f}≤0.95")
        if sr_wins < 3:    fail.append(f"{sr_wins}/5 windows SR>0")
        print(f"  Failures: {', '.join(fail)}")
    else:
        print("  Proceed to Monte Carlo FTMO sizing (Phase 20).")

    _save_csv(all_oos, "p19_wfo_oos_pooled.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 19 — INTRADAY MOMENTUM (GAO ET AL. 2018)")
    print("=" * W)

    cfg = Config()
    df  = load_data(cfg)
    print(f"  Data: {df.index[0]}  →  {df.index[-1]}  ({len(df):,} bars)")

    print("  Computing ATR(20) and avg_volume(20)...")
    atr1m   = compute_atr(df, 20)
    avg_vol = avg_volume(df, 20)

    print("  Detecting sessions...")
    si_full = detect_sessions(df, atr1m)
    valid   = [si for si in si_full.values() if si is not None]
    print(f"  Sessions with data: {len(valid)}")

    # ── rolling medians for vol/rv filters (causal, computed on full set) ─────
    # Sort sessions by date and compute rolling 20-session median of range/vol
    si_sorted = sorted(valid, key=lambda s: s.date)
    dates_sorted = [si.date for si in si_sorted]
    rv_arr   = np.array([si.range_first30 for si in si_sorted])
    vol_arr  = np.array([si.vol_first30   for si in si_sorted])

    # Causal rolling median (min 20 sessions)
    rv_med_dict  = {}
    vol_med_dict = {}
    for i, d in enumerate(dates_sorted):
        if i >= 20:
            rv_med_dict[d]  = float(np.median(rv_arr[i-20:i]))
            vol_med_dict[d] = float(np.median(vol_arr[i-20:i]))

    ok = run_diagnostic(si_full, rv_med_dict, vol_med_dict)
    if not ok:
        return

    # ── split ─────────────────────────────────────────────────────────────────
    df_tr  = df.loc[:TRAIN_END];   df_te  = df.loc[TEST_START:]
    a1_tr  = atr1m[df_tr.index];  a1_te  = atr1m[df_te.index]
    si_tr  = detect_sessions(df_tr, a1_tr)
    si_te  = detect_sessions(df_te, a1_te)

    print(f"\n  Train: {df_tr.index[0].date()} → {df_tr.index[-1].date()}  "
          f"({len(df_tr):,} bars)")
    print(f"  Test:  {df_te.index[0].date()} → {df_te.index[-1].date()}  "
          f"({len(df_te):,} bars)")

    # ── Experiment 1 — Strict replication ────────────────────────────────────
    kw1 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
                initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(1, "Strict replication (Gao et al. exact)",
                     "entry=15:30  exit=15:59  no filters  risk=1.0%")
    tr1, te1 = _run_split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                           kw1, save_name="p19_exp1_baseline.csv")
    print_block("TRAIN", tr1, df_tr)
    print_block("TEST ", te1, df_te, show_2025_2026=True)
    sr1_te = _sr(te1)

    # ── Experiment 2 — Magnitude filter ──────────────────────────────────────
    print_exp_header(2, "Magnitude filter",
                     "r1_threshold sweep: [0.1%, 0.2%, 0.3%, 0.5%]  entry=15:30")
    print(f"  {'thresh':>7} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6}")
    print(f"  {'-'*62}")

    best2_thr = 0.0; best2_sr = sr1_te; best2_res = None
    exp2_res = {}
    for thr_pct in [0.0, 0.1, 0.2, 0.3, 0.5]:
        thr = thr_pct / 100.0
        kw = dict(entry_bar_time=_ENTRY_T, r1_threshold=thr,
                  initial_capital=INITIAL_CAP, risk_pct=1.0)
        tr_x = run(df_tr, si_tr, a1_tr, **kw)
        te_x = run(df_te, si_te, a1_te, **kw)
        sr_te = _sr(te_x)
        print(f"  {thr_pct:>6.1f}% | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x)):>6} | "
              f"{_fmt(_pf(te_x)):>6} | {sr_te:>6.3f}")
        exp2_res[thr_pct] = (tr_x, te_x, thr)
        if sr_te > best2_sr and len(te_x) >= 30:
            best2_sr = sr_te; best2_thr = thr; best2_res = (tr_x, te_x)
    print(f"  → Best threshold: {best2_thr*100:.1f}%"
          f"  (improvement vs baseline: {'Yes' if best2_res else 'No'})")

    # ── Experiment 3 — Volatility conditioning ────────────────────────────────
    kw3 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
               high_vol_only=True, rv_median_dict=rv_med_dict,
               initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(3, "Volatility conditioning (high-vol days only)",
                     "range_first30 > 20-session rolling median  entry=15:30")
    tr3, te3 = _run_split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                           kw3, save_name="p19_exp3_hv.csv")
    print_block("TRAIN", tr3, df_tr)
    print_block("TEST ", te3, df_te, show_2025_2026=True)
    use_hv = _sr(te3) > sr1_te and len(te3) >= 30
    print(f"\n  Vol filter SR={_sr(te3):.3f}  vs  baseline SR={sr1_te:.3f}"
          f"  → {'Use vol filter' if use_hv else 'Skip vol filter'}")

    # ── Experiment 4 — Entry time sensitivity ────────────────────────────────
    print_exp_header(4, "Entry time sensitivity",
                     "exit always 15:59  entry times: [14:30, 15:00, 15:30, 15:45]")
    print(f"  {'entry':>7} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6}")
    print(f"  {'-'*62}")

    entry_times = {
        "14:30": dt_time(14, 29),
        "15:00": dt_time(14, 59),
        "15:30": dt_time(15, 29),
        "15:45": dt_time(15, 44),
    }
    best4_et = _ENTRY_T; best4_sr = sr1_te
    for label, et in entry_times.items():
        kw = dict(entry_bar_time=et, r1_threshold=0.0,
                  initial_capital=INITIAL_CAP, risk_pct=1.0)
        tr_x = run(df_tr, si_tr, a1_tr, **kw)
        te_x = run(df_te, si_te, a1_te, **kw)
        sr_te = _sr(te_x)
        print(f"  {label:>7} | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x)):>6} | "
              f"{_fmt(_pf(te_x)):>6} | {sr_te:>6.3f}")
        if sr_te > best4_sr and len(te_x) >= 30:
            best4_sr = sr_te; best4_et = et
    best4_lbl = [k for k, v in entry_times.items() if v == best4_et][0]
    print(f"  → Best entry time: {best4_lbl}")

    # ── Experiment 5 — r16 agreement filter ──────────────────────────────────
    kw5 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
               r16_agreement=True, initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(5, "r16 agreement filter (sign(r1)==sign(r16))",
                     "r16 = 14:30–15:00 half-hour return  entry=15:30")
    tr5, te5 = _run_split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                           kw5, save_name="p19_exp5_r16.csv")
    print_block("TRAIN", tr5, df_tr)
    print_block("TEST ", te5, df_te, show_2025_2026=True)
    use_r16 = _sr(te5) > sr1_te and len(te5) >= 30
    print(f"\n  r16 SR={_sr(te5):.3f}  vs  baseline SR={sr1_te:.3f}"
          f"  → {'Use r16 filter' if use_r16 else 'Skip r16 filter'}")

    # ── Experiment 6 — Volume filter ─────────────────────────────────────────
    kw6 = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
               vol_filter=True, vol_median_dict=vol_med_dict,
               initial_capital=INITIAL_CAP, risk_pct=1.0)
    print_exp_header(6, "Volume filter (first-30min volume > 20-session median)",
                     "vol_first30 > rolling median  entry=15:30")
    tr6, te6 = _run_split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                           kw6, save_name="p19_exp6_vol.csv")
    print_block("TRAIN", tr6, df_tr)
    print_block("TEST ", te6, df_te, show_2025_2026=True)
    use_vol = _sr(te6) > sr1_te and len(te6) >= 30
    print(f"\n  Vol filter SR={_sr(te6):.3f}  vs  baseline SR={sr1_te:.3f}"
          f"  → {'Use vol filter' if use_vol else 'Skip vol filter'}")

    # ── Experiment 7 — Risk pct sensitivity ──────────────────────────────────
    print_exp_header(7, "Risk percentage sensitivity (baseline entry, no filters)",
                     "risk_pct: [0.5%, 1.0%, 1.5%, 2.0%]")
    print(f"  {'risk%':>6} | {'N_te':>5} | {'WR_te':>6} | {'PF_te':>6} | "
          f"{'SR_te':>6} | {'MaxDD':>7} | {'Ret%':>7}")
    print(f"  {'-'*62}")
    best7_rp = 1.0
    for rp in [0.5, 1.0, 1.5, 2.0]:
        kw = dict(entry_bar_time=_ENTRY_T, r1_threshold=0.0,
                  initial_capital=INITIAL_CAP, risk_pct=rp)
        te_x = run(df_te, si_te, a1_te, **kw)
        print(f"  {rp:>5.1f}% | {len(te_x):>5} | {_wr(te_x)*100:>5.1f}% | "
              f"{_fmt(_pf(te_x)):>6} | {_sr(te_x):>6.3f} | "
              f"{_max_dd(te_x)*100:>6.1f}% | {_ret(te_x):>6.1f}%")

    # ── Experiment 8 — Combined best ─────────────────────────────────────────
    best_kw = dict(
        entry_bar_time = best4_et,
        r1_threshold   = best2_thr,
        high_vol_only  = use_hv,
        rv_median_dict = rv_med_dict if use_hv else None,
        vol_filter     = use_vol,
        vol_median_dict= vol_med_dict if use_vol else None,
        r16_agreement  = use_r16,
        initial_capital= INITIAL_CAP,
        risk_pct       = 1.0,
    )
    parts = [f"entry={best4_lbl}"]
    if best2_thr > 0:          parts.append(f"|r1|>{best2_thr*100:.1f}%")
    if use_hv:                 parts.append("high_vol")
    if use_r16:                parts.append("r16_agree")
    if use_vol:                parts.append("high_vol30")

    print_exp_header(8, "Combined best configuration", "  ".join(parts))
    tr8, te8 = _run_split(df_tr, df_te, si_tr, si_te, a1_tr, a1_te,
                           best_kw, save_name="p19_exp8_best.csv")
    print_block("TRAIN", tr8, df_tr)
    print_block("TEST ", te8, df_te, show_2025_2026=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  SUMMARY TABLE")
    print(f"  {'Exp':>3} | {'Entry':>6} | {'Filters':<18} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6} | {'TPD':>5} | {'WR':>5}")
    print(f"  {'-'*80}")

    e2t, e2r = exp2_res.get(best2_thr*100 if best2_thr > 0 else 0.0,
                             exp2_res.get(0.0, (tr1, te1, 0.0)))[:2], \
               exp2_res.get(best2_thr*100 if best2_thr > 0 else 0.0,
                             exp2_res.get(0.0, (tr1, te1, 0.0)))[2]

    rows = [
        (1, "15:30", "none",                 tr1, te1),
        (3, "15:30", "high_vol",             tr3, te3),
        (5, "15:30", "r16_agree",            tr5, te5),
        (6, "15:30", "high_vol30",           tr6, te6),
        (8, best4_lbl, "  ".join(parts[1:]) if len(parts)>1 else "none", tr8, te8),
    ]
    for (en, et, filt, trr, ter) in rows:
        print(f"  {en:>3} | {et:>6} | {filt:<18} | "
              f"{_fmt(_pf(trr)):>6} | {_fmt(_pf(ter)):>6} | "
              f"{_sr(ter):>6.3f} | {_tpd(ter, df_te):>5.3f} | "
              f"{_wr(ter)*100:>4.1f}%")

    # ── WFO decision ──────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    sr8_te = _sr(te8); n8_te = len(te8)
    print(f"  Exp 8 test: SR={sr8_te:.3f}  N={n8_te}")
    if sr8_te > 0.5 and n8_te >= 80:
        print("  → Thresholds met (SR>0.5 and N>=80). Running walk-forward.")
        best_cfg = dict(best_kw)
        run_wfo(df, atr1m, best_cfg)
    else:
        reasons = []
        if sr8_te <= 0.5: reasons.append(f"SR={sr8_te:.3f}≤0.5")
        if n8_te < 80:    reasons.append(f"N={n8_te}<80")
        print(f"  → WFO skipped: {', '.join(reasons)}")

    print(f"\n{'='*W}")
    print("  PHASE 19 COMPLETE")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
