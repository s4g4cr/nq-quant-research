#!/usr/bin/env python3
"""
Phase 20 — POC Closing Magnet.

Hypothesis: In the last hour (15:00–16:00), institutional participants
rebalance toward the session's volume equilibrium (POC), creating a
directional signal. Tests 3 POC variants × 3 entry conditions = 9 experiments.

Train: 2021-06-25 → 2024-12-31
Test:  2025-01-02 → 2026-06-17
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
from orb_system.indicators.technical import atr as compute_atr, session_vwap
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.strategy.poc_closing_magnet import (
    run, ClosingMagnetTrade, SLIP, COMM, PV,
    _SIG_T, _WIN_E, _EXIT_T,
)

TRAIN_END  = "2024-12-31"
TEST_START = "2025-01-02"
RESULTS    = os.path.join(ROOT, "results")
W          = 76

EXPERIMENTS = [
    (1, "A", 1, "session", "dist+mom"),
    (2, "A", 2, "session", "vwap"),
    (3, "A", 3, "session", "simple"),
    (4, "B", 1, "prev",    "dist+mom"),
    (5, "B", 2, "prev",    "vwap"),
    (6, "B", 3, "prev",    "simple"),
    (7, "C", 1, "nearest", "dist+mom"),
    (8, "C", 2, "nearest", "vwap"),
    (9, "C", 3, "nearest", "simple"),
]


# ── metric helpers ─────────────────────────────────────────────────────────────

def _pf(trades):
    v = np.array([t.pnl_net for t in trades])
    w = v[v > 0]; l = v[v <= 0]
    return float(w.sum() / abs(l.sum())) if l.size and l.sum() != 0 else float("inf")

def _sr(trades, cap=100_000.0):
    if not trades: return 0.0
    by_d = {}
    for t in trades:
        by_d[t.date] = by_d.get(t.date, 0.0) + t.pnl_net
    v = np.array(list(by_d.values()))
    s = float(v.std())
    return float(v.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _wr(trades): return sum(1 for t in trades if t.pnl_net > 0) / len(trades) if trades else 0.0

def _ret(trades, cap=100_000.0):
    return sum(t.pnl_net for t in trades) / cap * 100.0 if trades else 0.0

def _mdd(trades, cap=100_000.0):
    c = peak = cap; worst = 0.0
    for t in trades:
        c += t.pnl_net; peak = max(peak, c)
        worst = max(worst, (peak - c) / peak)
    return worst * 100.0

def _exits(trades):
    n = max(len(trades), 1)
    sl  = sum(1 for t in trades if t.exit_reason == "sl")  / n * 100
    tp  = sum(1 for t in trades if t.exit_reason == "tp")  / n * 100
    tm  = sum(1 for t in trades if t.exit_reason == "time") / n * 100
    return sl, tp, tm

def _rr(trades):
    ws = [t.pnl_pts for t in trades if t.pnl_net > 0]
    ls = [abs(t.pnl_pts) for t in trades if t.pnl_net <= 0]
    aw = float(np.mean(ws)) if ws else 0.0
    al = float(np.mean(ls)) if ls else 0.0
    rr = aw / al if al > 0 else 0.0
    return aw, al, rr

def _annual(trades):
    by_yr = {}
    for t in trades:
        yr = t.entry_ts.year
        by_yr.setdefault(yr, []).append(t)
    return by_yr

def _fmt(v, d=3):
    if v != v: return "  nan"
    if v == float("inf"): return "  inf"
    return f"{v:+.{d}f}" if d > 0 else f"{v:.0f}"

def _save(trades, fname):
    if not trades: return
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame([vars(t) for t in trades]).to_csv(
        os.path.join(RESULTS, fname), index=False)


# ── data preparation ───────────────────────────────────────────────────────────

def build_df(cfg, df_raw):
    """Merge ATR, POC features, and session VWAP into one working DataFrame."""
    atr1m    = compute_atr(df_raw, 20)
    poc_feat = compute_poc_features(df_raw)
    df_v     = session_vwap(df_raw)

    out = df_raw.copy()
    out["atr"]         = atr1m
    out["session_poc"] = poc_feat["session_poc"]
    out["prev_poc"]    = poc_feat["prev_poc"]
    out["vwap"]        = df_v["vwap"]
    return out


# ── diagnostic ─────────────────────────────────────────────────────────────────

def diagnostic(df):
    print("=" * W)
    print("  PHASE 20 — PRE-P&L DIAGNOSTIC")
    print("=" * W)

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    u_dates  = np.unique(date_arr)

    cl_v   = df["close"].values
    atr_v  = df["atr"].values
    spoc_v = df["session_poc"].values
    ppoc_v = df["prev_poc"].values
    hi_v   = df["high"].values
    lo_v   = df["low"].values
    vwap_v = df["vwap"].values

    # Collect per-session snapshot at 14:59
    rows = []
    for d in u_dates:
        mask   = date_arr == d
        d_idx  = np.where(mask)[0]
        d_t    = time_arr[d_idx]

        sig_loc = np.where(d_t == _SIG_T)[0]
        if sig_loc.size == 0: continue
        sabs = int(d_idx[sig_loc[0]])

        sp = float(spoc_v[sabs]); pp = float(ppoc_v[sabs])
        px = float(cl_v[sabs]);   at = float(atr_v[sabs])
        if np.isnan(sp) or np.isnan(pp) or np.isnan(at) or at <= 0: continue

        np_val = sp if abs(px - sp) < abs(px - pp) else pp
        rows.append({"d": d, "sp": sp, "pp": pp, "np": np_val, "px": px, "atr": at,
                     "d_idx": d_idx, "d_t": d_t, "sabs": sabs})

    n = len(rows)
    print(f"\n  Sessions with valid POC data at 14:59: {n}")

    # ── SECTION 1: Displacement distribution ──────────────────────────────────
    print(f"\n  SECTION 1  —  POC Displacement at 14:59")

    for lbl, key in [("Session POC-A", "sp"), ("Prev POC-B", "pp"), ("Nearest POC-C", "np")]:
        dists = [abs(r["px"] - r[key]) / r["atr"] for r in rows]
        d_arr = np.array(dists)
        p10, p25, p50, p75, p90 = np.percentile(d_arr, [10, 25, 50, 75, 90])
        pct_gt1 = (d_arr > 1.0).mean() * 100
        print(f"\n  {lbl}:")
        print(f"    Dist (ATR units) p10={p10:.2f} p25={p25:.2f} p50={p50:.2f} "
              f"p75={p75:.2f} p90={p90:.2f}")
        print(f"    % sessions with distance > 1.0 ATR: {pct_gt1:.1f}%")

    # ── SECTION 2: Natural reversion rate (session_poc reference) ─────────────
    print(f"\n  SECTION 2  —  Natural Reversion Rate to Session POC (no SL)")

    n_disp = n_reach_15 = n_reach_30 = n_reach_45 = n_reach_60 = 0
    bars_to_poc = []

    for r in rows:
        sp   = r["sp"]; px = r["px"]; at = r["atr"]
        d_idx = r["d_idx"]; sabs = r["sabs"]
        if abs(px - sp) <= 1.0 * at:
            continue
        n_disp += 1
        dirn = "long" if px < sp else "short"

        post = d_idx[d_idx > sabs]
        hit = False
        for k, pos in enumerate(post):
            t = time_arr[pos]
            if t > _EXIT_T: break
            h = float(hi_v[pos]); l = float(lo_v[pos])
            reached = (h >= sp) if dirn == "long" else (l <= sp)
            if reached:
                hit = True
                b = k + 1
                if b <= 15: n_reach_15 += 1
                if b <= 30: n_reach_30 += 1
                if b <= 45: n_reach_45 += 1
                if b <= 60: n_reach_60 += 1
                bars_to_poc.append(b)
                break

    pct = lambda x: x / max(n_disp, 1) * 100
    print(f"  Sessions with displacement > 1.0 ATR: {n_disp}")
    print(f"  Reached session_poc within 15 bars:   {n_reach_15:3d} ({pct(n_reach_15):.1f}%)")
    print(f"  Reached session_poc within 30 bars:   {n_reach_30:3d} ({pct(n_reach_30):.1f}%)")
    print(f"  Reached session_poc within 45 bars:   {n_reach_45:3d} ({pct(n_reach_45):.1f}%)")
    print(f"  Reached session_poc within 60 bars:   {n_reach_60:3d} ({pct(n_reach_60):.1f}%)")
    if bars_to_poc:
        print(f"  Median bars to reach session_poc:      {np.median(bars_to_poc):.0f}")

    # ── SECTION 3: Signal frequency by condition ───────────────────────────────
    print(f"\n  SECTION 3  —  Signal Frequency by Entry Condition (15:00–15:30 window)")

    c1_sigs = c2_sigs = c3_sigs = 0

    for r in rows:
        sp   = r["sp"]; px = r["px"]; at = r["atr"]
        d_idx = r["d_idx"]; d_t_all = r["d_t"]; sabs = r["sabs"]

        # Condition 3: always if displaced
        if px != sp:
            c3_sigs += 1

        win_mask = (d_t_all >= dt_time(15, 0)) & (d_t_all <= _WIN_E)
        win_abs  = d_idx[win_mask]

        below = px < sp - at
        above = px > sp + at

        prev_cl = px
        prev_vw = float(vwap_v[sabs]) if not np.isnan(vwap_v[sabs]) else np.nan

        c1_found = c2_found = False
        for pos in win_abs:
            cur_cl   = float(cl_v[pos])
            cur_vwap = float(vwap_v[pos])
            # Condition 1
            if not c1_found:
                if (below and cur_cl > prev_cl) or (above and cur_cl < prev_cl):
                    c1_sigs += 1; c1_found = True
            # Condition 2
            if not c2_found and not (np.isnan(prev_vw) or np.isnan(cur_vwap)):
                if (prev_cl < prev_vw and cur_cl >= cur_vwap and sp > cur_cl) or \
                   (prev_cl > prev_vw and cur_cl <= cur_vwap and sp < cur_cl):
                    c2_sigs += 1; c2_found = True
            prev_cl = cur_cl; prev_vw = cur_vwap
            if c1_found and c2_found: break

    print(f"  Condition 1 (dist+momentum):   {c1_sigs} sessions ({c1_sigs/n*100:.1f}%)")
    print(f"  Condition 2 (VWAP cross→POC):  {c2_sigs} sessions ({c2_sigs/n*100:.1f}%)")
    print(f"  Condition 3 (simple displace):  {c3_sigs} sessions ({c3_sigs/n*100:.1f}%)")

    # ── SECTION 4: Geometric R/R ───────────────────────────────────────────────
    print(f"\n  SECTION 4  —  Geometric R/R (TP=session_poc, SL=1.0×ATR, Cond-3 style)")

    rr_vals = []
    for r in rows:
        sp = r["sp"]; px = r["px"]; at = r["atr"]
        if px == sp: continue
        ep = (px + SLIP) if px < sp else (px - SLIP)
        tp_d = abs(ep - sp)
        sl_d = at
        rr_vals.append(tp_d / sl_d if sl_d > 0 else 0.0)

    if rr_vals:
        rr_arr = np.array(rr_vals)
        print(f"  R/R p25={np.percentile(rr_arr,25):.2f}  "
              f"p50={np.percentile(rr_arr,50):.2f}  "
              f"p75={np.percentile(rr_arr,75):.2f}")
        print(f"  % with R/R > 1.5: {(rr_arr > 1.5).mean()*100:.1f}%  "
              f"| R/R > 2.0: {(rr_arr > 2.0).mean()*100:.1f}%")

    # ── SECTION 5: Theoretical expectancy per POC variant ─────────────────────
    print(f"\n  SECTION 5  —  Theoretical Expectancy (Cond-3 entry, SL=1.0×ATR)")
    print(f"  {'POC':>8} | {'N':>4} | {'WR%':>6} | {'med R/R':>7} | "
          f"{'min WR%':>7} | {'Expect':>7}")
    print(f"  {'─'*52}")

    for lbl, key in [("POC-A", "sp"), ("POC-B", "pp"), ("POC-C", "np")]:
        wins = 0; rr_v = []
        for r in rows:
            poc = r[key]; px = r["px"]; at = r["atr"]
            d_idx = r["d_idx"]; sabs = r["sabs"]
            if px == poc: continue
            dirn = "long" if px < poc else "short"
            ep   = (px + SLIP) if dirn == "long" else (px - SLIP)
            sl   = (ep - at)   if dirn == "long" else (ep + at)
            tp   = poc
            if (dirn == "long" and tp <= ep) or (dirn == "short" and tp >= ep): continue
            tp_d = abs(ep - tp); sl_d = at
            rr_v.append(tp_d / sl_d if sl_d > 0 else 0.0)

            post = d_idx[d_idx > sabs]
            hit_tp = False
            for pos in post:
                t = time_arr[pos]
                if t > _EXIT_T: break
                h = float(hi_v[pos]); l = float(lo_v[pos])
                sl_hit = (l <= sl) if dirn == "long" else (h >= sl)
                tp_hit = (h >= tp) if dirn == "long" else (l <= tp)
                if sl_hit: break
                if tp_hit: hit_tp = True; break
            if hit_tp: wins += 1

        nn = len(rr_v)
        if nn == 0: continue
        wr    = wins / nn
        med_rr = float(np.median(rr_v)) if rr_v else 0.0
        min_wr = 1.0 / (1.0 + med_rr) if med_rr > 0 else 1.0
        exp    = wr * med_rr - (1.0 - wr)
        mark   = " <<" if exp > 0 else ""
        print(f"  {lbl:>8} | {nn:>4} | {wr*100:>5.1f}% | {med_rr:>7.2f} | "
              f"{min_wr*100:>6.1f}% | {exp:>+7.3f}{mark}")

    print(f"\n  All 9 experiments proceed regardless of diagnostic results.")
    print("=" * W)


# ── experiment reporting ───────────────────────────────────────────────────────

def _print_exp(num, pv_lbl, cond_lbl, tr_trades, te_trades, te_25, te_26):
    cap = 100_000.0
    print(f"\n{'═'*W}")
    print(f"  EXPERIMENT {num} — POC-{pv_lbl.upper()} + Condition-{['','dist+mom','vwap','simple'][EXPERIMENTS[num-1][2]]}")
    print(f"{'═'*W}")

    for label, trd in [("TRAIN", tr_trades), ("TEST ", te_trades)]:
        if not trd:
            print(f"  {label}: N=0"); continue
        n  = len(trd)
        wr = _wr(trd)*100; pf = _pf(trd); sr = _sr(trd)
        rt = _ret(trd, cap); md = _mdd(trd, cap)
        sl_pct, tp_pct, tm_pct = _exits(trd)
        aw, al, rr = _rr(trd)
        tpd = n / max(len({t.date for t in trd}), 1)
        print(f"  {label}: N={n}  WR={wr:.1f}%  PF={_fmt(pf)}  SR={sr:.3f}  "
              f"Ret={rt:+.1f}%  MaxDD={md:.1f}%")
        print(f"         TPD={tpd:.2f}  Exits: SL={sl_pct:.0f}%  TP={tp_pct:.0f}%  "
              f"Time={tm_pct:.0f}%")
        print(f"         AvgWin={aw:+.1f}pt  AvgLoss={al:.1f}pt  RR={rr:.2f}")

        by_yr = _annual(trd)
        for yr, yt in sorted(by_yr.items()):
            yn = len(yt); ywr = _wr(yt)*100; ypf = _pf(yt); yret = _ret(yt, cap)
            print(f"           {yr}: N={yn:3d}  WR={ywr:.1f}%  PF={_fmt(ypf)}  Ret={yret:+.1f}%")

    # TEST: 2025 / 2026 separate
    if te_25 or te_26:
        for lbl, t2 in [("  2025", te_25), ("  2026", te_26)]:
            if not t2: print(f"    {lbl}: N=0"); continue
            print(f"    {lbl}: N={len(t2)}  WR={_wr(t2)*100:.1f}%  "
                  f"PF={_fmt(_pf(t2))}  SR={_sr(t2):.3f}  Ret={_ret(t2,cap):+.1f}%")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 20 — POC CLOSING MAGNET")
    print("=" * W)

    cfg = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]}  →  {df_raw.index[-1]}  ({len(df_raw):,} bars)")

    print("  Building working DataFrame (ATR + POC + VWAP)…")
    df = build_df(cfg, df_raw)
    print(f"  Indicators ready. Columns: {list(df.columns)}")

    df_tr = df.loc[:TRAIN_END]
    df_te = df.loc[TEST_START:]
    df_25 = df.loc["2025-01-02":"2025-12-31"]
    df_26 = df.loc["2026-01-02":]
    print(f"  Train: {df_tr.index[0].date()} → {df_tr.index[-1].date()}  ({len(df_tr):,} bars)")
    print(f"  Test:  {df_te.index[0].date()} → {df_te.index[-1].date()}  ({len(df_te):,} bars)")

    # ── Diagnostic ────────────────────────────────────────────────────────────
    diagnostic(df)

    # ── Experiments ───────────────────────────────────────────────────────────
    summary = []

    for num, pv, cond, pv_lbl, cond_lbl in EXPERIMENTS:
        kw = dict(poc_variant=pv, condition=cond, initial_capital=100_000.0, risk_pct=1.0)
        tr_trades = run(df_tr, **kw)
        te_trades = run(df_te, **kw)
        te_25     = run(df_25, **kw)
        te_26     = run(df_26, **kw)

        _print_exp(num, pv_lbl, cond_lbl, tr_trades, te_trades, te_25, te_26)

        n_te  = len(te_trades)
        wr_te = _wr(te_trades)*100
        pf_te = _pf(te_trades)
        sr_te = _sr(te_trades)
        rr_te = _rr(te_trades)[2]
        flag  = "✓" if pf_te > 1.0 and n_te >= 60 else ""
        summary.append((num, pv_lbl, cond_lbl, n_te, wr_te, pf_te, sr_te, rr_te, flag,
                        tr_trades, te_trades))

        name = f"p20_exp{num:02d}_{pv_lbl}_{cond_lbl.replace('+','_')}"
        _save(te_trades, f"{name}.csv")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  SUMMARY TABLE — TEST PERIOD")
    print(f"  {'Exp':>3} | {'POC':>7} | {'Cond':>9} | {'N Te':>5} | "
          f"{'WR%':>5} | {'PF Te':>6} | {'SR Te':>6} | {'R/R':>5} | {'Flag':>4}")
    print(f"  {'─'*70}")
    for num, pv, cond, n_te, wr, pf, sr, rr, flag, *_ in summary:
        print(f"  {num:>3} | {pv:>7} | {cond:>9} | {n_te:>5} | "
              f"{wr:>4.1f}% | {_fmt(pf):>6} | {sr:>+6.3f} | {rr:>5.2f} | {flag:>4}")

    # ── Auto-selection ─────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    candidates = [(num, pv, cond, pf, sr, n_te)
                  for num, pv, cond, n_te, wr, pf, sr, rr, flag, *_ in summary
                  if pf > 1.0 and n_te >= 60]
    if candidates:
        best = max(candidates, key=lambda x: x[4])  # rank by SR test
        print(f"  FLAGGED FOR WALK-FORWARD:")
        for c in sorted(candidates, key=lambda x: -x[4]):
            print(f"    Exp {c[0]} — POC-{c[1].upper()} + {c[2]}:  "
                  f"PF={c[3]:.3f}  SR={c[4]:+.3f}  N={c[5]}")
        print(f"\n  PRIMARY:  Exp {best[0]} — POC-{best[1].upper()} + {best[2]}")
    else:
        # Root cause from diagnostic
        print(f"  No experiment passes PF>1.0 with N≥60.")
        print(f"  Root cause: check Section 5 expectancy — if all variants negative,")
        print(f"  the closing magnet does not generate positive-expectancy R/R at 1.0×ATR SL.")

    # ── Save diagnostic summary ────────────────────────────────────────────────
    os.makedirs(RESULTS, exist_ok=True)
    diag_rows = []
    for num, pv, cond, n_te, wr, pf, sr, rr, flag, tr_trades, te_trades in summary:
        n_tr = len(tr_trades)
        diag_rows.append({
            "exp": num, "poc_variant": pv, "condition": cond,
            "N_train": n_tr, "WR_train": round(_wr(tr_trades)*100, 1) if tr_trades else 0,
            "PF_train": round(_pf(tr_trades), 3) if tr_trades else 0,
            "SR_train": round(_sr(tr_trades), 3) if tr_trades else 0,
            "N_test": n_te, "WR_test": round(wr, 1),
            "PF_test": round(pf, 3), "SR_test": round(sr, 3),
            "RR_test": round(rr, 2), "flag": flag,
        })
    pd.DataFrame(diag_rows).to_csv(os.path.join(RESULTS, "p20_summary.csv"), index=False)
    print(f"\n  Saved: results/p20_exp[1-9]_*.csv")
    print(f"         results/p20_summary.csv")
    print(f"\n{'='*W}")
    print("  PHASE 20 COMPLETE")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
