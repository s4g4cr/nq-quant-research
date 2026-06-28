#!/usr/bin/env python3
"""
Phase 19C — Intraday Reversal: filter optimization for N >= 80.

17 experiments systematically relaxing Phase 19B Exp 7's combined filter
to find the minimum set that keeps N_test >= 80 with SR_test > 1.0.
No hard gates — all experiments run always.
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
from orb_system.strategy.intraday_reversal import detect_sessions, run
from orb_system.strategy.intraday_momentum import _ENTRY_T, _EXIT_T

INITIAL_CAP = 100_000.0
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
W           = 76
RESULTS_DIR = os.path.join(ROOT, "results")


# ── stat helpers ──────────────────────────────────────────────────────────────

def _pf(trades):
    v = np.array([t.pnl_net for t in trades])
    w = v[v > 0]; l = v[v <= 0]
    return float(w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")

def _sr(trades):
    if not trades: return 0.0
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

def _avg_wl(trades):
    w = [t.pnl_net for t in trades if t.pnl_net > 0]
    l = [t.pnl_net for t in trades if t.pnl_net <= 0]
    wp = [t.pnl_pts for t in trades if t.pnl_net > 0]
    lp = [t.pnl_pts for t in trades if t.pnl_net <= 0]
    aw = float(np.mean(w))  if w else 0.0
    al = float(np.mean(l))  if l else 0.0
    awp= float(np.mean(wp)) if wp else 0.0
    alp= float(np.mean(lp)) if lp else 0.0
    rr = abs(aw / al) if al != 0 else float("inf")
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
    if v != v: return "  nan"
    if v == float("inf"): return "  inf"
    return f"{v:.{d}f}"

def _save_csv(trades, fname):
    if not trades: return
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


# ── per-experiment printer ────────────────────────────────────────────────────

def print_exp(num, name, tr, te, df_tr, df_te):
    def _block(label, trades, df, show_annual=False, show_split=False):
        if not trades:
            print(f"  {label}: No trades")
            return
        aw, awp, al, alp, rr = _avg_wl(trades)
        print(f"  {label}: N={len(trades)}  TPD={_tpd(trades,df):.3f}"
              f"  WR={_wr(trades)*100:.1f}%  PF={_fmt(_pf(trades))}"
              f"  SR={_sr(trades):.3f}  Ret={_ret(trades):.1f}%"
              f"  MaxDD={_max_dd(trades)*100:.1f}%  AvgC={_nc(trades):.1f}")
        print(f"         AvgW=${aw:.0f}({awp:.1f}pt)"
              f"  AvgL=${al:.0f}({alp:.1f}pt)  R/R={_fmt(rr,2)}")
        if show_annual:
            for y, n_y, wr_y, pf_y, ret_y in _annual(trades):
                print(f"           {y}: N={n_y:3d}  WR={wr_y:.1f}%"
                      f"  PF={_fmt(pf_y)}  Ret={ret_y:.1f}%")
        if show_split:
            for yr in [2025, 2026]:
                tt = [t for t in trades if pd.Timestamp(str(t.date)).year == yr]
                if tt:
                    print(f"         {yr}: N={len(tt):3d}  WR={_wr(tt)*100:.1f}%"
                          f"  PF={_fmt(_pf(tt))}  SR={_sr(tt):.3f}"
                          f"  Ret={_ret(tt):.1f}%")
                else:
                    print(f"         {yr}: No trades")

    print(f"\n{'═'*W}")
    print(f"  EXP {num:>2} — {name}")
    print(f"{'═'*W}")
    _block("TRAIN", tr, df_tr, show_annual=True)
    _block("TEST ", te, df_te, show_split=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 19C — INTRADAY REVERSAL: FILTER OPTIMIZATION FOR N >= 80")
    print("  Direction: SHORT if r1>0 · LONG if r1<0 · entry=15:30 · exit=15:59")
    print("=" * W)

    cfg   = Config()
    df    = load_data(cfg)
    print(f"  Data: {df.index[0]}  →  {df.index[-1]}  ({len(df):,} bars)")

    atr1m   = compute_atr(df, 20)
    avg_vol = avg_volume(df, 20)

    si_full = detect_sessions(df, atr1m)
    valid   = sorted([si for si in si_full.values() if si is not None],
                     key=lambda s: s.date)
    print(f"  Sessions with data: {len(valid)}")

    # Rolling 20-session medians (causal)
    rv_arr   = np.array([si.range_first30 for si in valid])
    vol_arr  = np.array([si.vol_first30   for si in valid])
    rv_med   = {}
    vol_med  = {}
    for i, si in enumerate(valid):
        if i >= 20:
            rv_med[si.date]  = float(np.median(rv_arr[i-20:i]))
            vol_med[si.date] = float(np.median(vol_arr[i-20:i]))

    # Split
    df_tr  = df.loc[:TRAIN_END];   df_te  = df.loc[TEST_START:]
    a1_tr  = atr1m[df_tr.index];  a1_te  = atr1m[df_te.index]
    si_tr  = detect_sessions(df_tr, a1_tr)
    si_te  = detect_sessions(df_te, a1_te)
    print(f"  Train: {df_tr.index[0].date()} → {df_tr.index[-1].date()}"
          f"  ({len(df_tr):,} bars)")
    print(f"  Test:  {df_te.index[0].date()} → {df_te.index[-1].date()}"
          f"  ({len(df_te):,} bars)")

    BASE = dict(
        entry_bar_time  = _ENTRY_T,
        r1_threshold    = 0.0,
        high_vol_only   = False,
        rv_median_dict  = rv_med,
        vol_filter      = False,
        vol_median_dict = vol_med,
        r16_agreement   = False,
        initial_capital = INITIAL_CAP,
        risk_pct        = 1.0,
    )

    EXPS = [
        (1,  "No filters (baseline)",
         dict()),
        (2,  "|r1| > 0.1%",
         dict(r1_threshold=0.001)),
        (3,  "|r1| > 0.2%",
         dict(r1_threshold=0.002)),
        (4,  "|r1| > 0.3%",
         dict(r1_threshold=0.003)),
        (5,  "high_vol",
         dict(high_vol_only=True)),
        (6,  "r16 agreement",
         dict(r16_agreement=True)),
        (7,  "high_volume",
         dict(vol_filter=True)),
        (8,  "|r1|>0.2% + high_vol",
         dict(r1_threshold=0.002, high_vol_only=True)),
        (9,  "|r1|>0.2% + r16_agree",
         dict(r1_threshold=0.002, r16_agreement=True)),
        (10, "|r1|>0.2% + high_volume",
         dict(r1_threshold=0.002, vol_filter=True)),
        (11, "|r1|>0.3% + high_vol",
         dict(r1_threshold=0.003, high_vol_only=True)),
        (12, "|r1|>0.3% + r16_agree",
         dict(r1_threshold=0.003, r16_agreement=True)),
        (13, "|r1|>0.3% + high_volume",
         dict(r1_threshold=0.003, vol_filter=True)),
        (14, "|r1|>0.2% + high_vol + r16_agree",
         dict(r1_threshold=0.002, high_vol_only=True, r16_agreement=True)),
        (15, "|r1|>0.2% + high_vol + high_volume",
         dict(r1_threshold=0.002, high_vol_only=True, vol_filter=True)),
        (16, "|r1|>0.3% + high_vol + r16_agree  [19B Exp7 minus vol filter]",
         dict(r1_threshold=0.003, high_vol_only=True, r16_agreement=True)),
        (17, "|r1|>0.3% + high_vol + high_volume",
         dict(r1_threshold=0.003, high_vol_only=True, vol_filter=True)),
    ]

    results = []

    for num, name, overrides in EXPS:
        kw_tr = {**BASE, **overrides}
        kw_te = {**BASE, **overrides}
        tr = run(df_tr, si_tr, a1_tr, **kw_tr)
        te = run(df_te, si_te, a1_te, **kw_te)
        print_exp(num, name, tr, te, df_tr, df_te)
        _save_csv(tr + te, f"p19c_exp{num:02d}.csv")

        sr_te   = _sr(te)
        pf_te   = _pf(te)
        pf_tr   = _pf(tr)
        n_te    = len(te)
        n_tr    = len(tr)
        wr_te   = _wr(te) * 100
        tpd_te  = _tpd(te, df_te)
        mdd_te  = _max_dd(te) * 100
        ret_te  = _ret(te)

        results.append({
            "exp": num, "name": name,
            "N_tr": n_tr, "N_te": n_te,
            "TPD": round(tpd_te, 3), "WR_te": round(wr_te, 1),
            "PF_tr": round(pf_tr, 3), "PF_te": round(pf_te, 3),
            "SR_te": round(sr_te, 3), "MaxDD_te": round(mdd_te, 1),
            "Ret_te": round(ret_te, 1),
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  SUMMARY TABLE")
    hdr = (f"  {'Exp':>3} | {'Filters':<35} | {'N Te':>5} | {'TPD':>5} | "
           f"{'WR%':>5} | {'PF Te':>6} | {'SR Te':>7} | {'MaxDD':>6}")
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for r in results:
        flag = ""
        if r["N_te"] >= 80 and r["SR_te"] > 1.0:
            flag = " ✓"
        elif r["N_te"] >= 60 and r["SR_te"] > 1.0:
            flag = " ~"
        print(f"  {r['exp']:>3} | {r['name']:<35} | {r['N_te']:>5} | "
              f"{r['TPD']:>5.3f} | {r['WR_te']:>4.1f}% | "
              f"{_fmt(r['PF_te']):>6} | {r['SR_te']:>7.3f} | "
              f"{r['MaxDD_te']:>5.1f}%{flag}")

    print(f"\n  ✓ = N≥80 and SR>1.0  · ~ = N≥60 and SR>1.0 (fallback)")

    # ── Auto-select best config ────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  AUTO-SELECTION")

    primary   = [r for r in results if r["N_te"] >= 80 and r["SR_te"] > 1.0]
    secondary = [r for r in results if r["N_te"] >= 60 and r["SR_te"] > 1.0
                 and r not in primary]

    if primary:
        best = max(primary, key=lambda r: r["SR_te"])
        print(f"  PRIMARY   — Exp {best['exp']:>2}: {best['name']}")
        print(f"              N_te={best['N_te']}  SR_te={best['SR_te']:.3f}"
              f"  PF_te={_fmt(best['PF_te'])}  WR={best['WR_te']:.1f}%")
        print(f"  → Meets both thresholds (N≥80, SR>1.0).")
        print(f"  → Candidate for walk-forward validation.")
    elif secondary:
        best = max(secondary, key=lambda r: r["SR_te"])
        print(f"  SECONDARY — Exp {best['exp']:>2}: {best['name']}")
        print(f"              N_te={best['N_te']}  SR_te={best['SR_te']:.3f}"
              f"  PF_te={_fmt(best['PF_te'])}  WR={best['WR_te']:.1f}%")
        print(f"  *** STATISTICAL CAVEAT: N={best['N_te']} < 80 ***")
        print(f"  → Below WFO gate — results should be treated with caution.")
    else:
        best = max(results, key=lambda r: r["SR_te"])
        print(f"  NO config meets N≥60 + SR>1.0. Best by SR:")
        print(f"  Exp {best['exp']:>2}: {best['name']}")
        print(f"  N_te={best['N_te']}  SR_te={best['SR_te']:.3f}"
              f"  PF_te={_fmt(best['PF_te'])}  WR={best['WR_te']:.1f}%")
        print(f"  → Reversal not tradeable with these filter combinations.")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(RESULTS_DIR, "p19c_summary.csv"), index=False)
    print(f"\n  Summary saved: results/p19c_summary.csv")

    print(f"\n{'='*W}")
    print("  PHASE 19C COMPLETE")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
