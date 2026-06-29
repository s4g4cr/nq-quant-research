#!/usr/bin/env python3
"""
Phase 23 — HMM Transition Signal as Filter/Scaler on POC Reversion B1.

Hypothesis: applying the HMM transition-matrix state as a hard filter or
position scaler on the confirmed B1 POC reversion strategy improves
risk-adjusted performance.

Approach A (hard filter)  — A1: ranging only; A2: ranging+bullish; A3: bearish only
Approach B (pos scaler)   — B1: 1.5/0.5/1.0; B2: 2.0/0.5/1.0;
                            B3: 1.5/skip/1.0; B4: 1.0/skip/1.0
"""

import math
import os
import sys
from datetime import date as dt_date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.det_regime_v2 import compute_det_regime_features_v2
from orb_system.indicators.technical import add_indicators
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.strategy.hmm_transition import (
    add_causal_features,
    extract_daily_features,
    label_states,
    predict_states,
    train_hmm,
)
from orb_system.strategy.poc_reversion import POCResults
from orb_system.strategy.poc_reversion_hmm import (
    CAPITAL,
    run_b1_baseline,
    run_filtered,
    run_scaled,
)

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_END  = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END   = "2026-06-17"

N_STATES = 3
N_BOOT   = 1000
W        = 76

WINDOWS: List[Tuple[str, str, str, str]] = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-17"),
]

RESULTS_DIR = os.path.join(ROOT, "results")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _pf(v):
    a = np.array(v)
    w = a[a > 0]; l = a[a <= 0]
    gw = float(w.sum()) if w.size else 0.0
    gl = float(abs(l.sum())) if l.size else 0.0
    return gw / gl if gl > 0 else float("inf")

def _sr(v):
    a = np.array(v); s = float(a.std())
    return float(a.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _fmt(v, d=3):
    if v != v or v == float("inf"): return "  inf"
    return f"{v:.{d}f}"

def _metrics(trades):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": float("nan"), "sharpe": 0.0,
                "max_dd_pct": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "max_consec_loss": 0}
    net  = np.array([t.pnl_net for t in trades])
    w    = net[net > 0]; l = net[net <= 0]
    gw   = float(w.sum()) if w.size else 0.0
    gl   = float(abs(l.sum())) if l.size else 0.0
    pf   = gw / gl if gl > 0 else float("inf")
    wr   = float(w.size) / len(net)
    std  = float(net.std())
    sr   = float(net.mean() / std * math.sqrt(252)) if std > 0 else 0.0
    curve = np.concatenate([[0.0], np.cumsum(net)])
    peak  = np.maximum.accumulate(curve)
    max_dd = float(((peak - curve) / CAPITAL * 100.0).max())
    best = cur = 0
    for t in trades:
        if t.pnl_net < 0: cur += 1; best = max(best, cur)
        else: cur = 0
    return {
        "n": len(net), "wr": wr, "pf": pf, "sharpe": sr,
        "max_dd_pct": max_dd,
        "avg_win":   float(w.mean()) if w.size else 0.0,
        "avg_loss":  float(l.mean()) if l.size else 0.0,
        "max_consec_loss": best,
    }

def _bootstrap_pf(pnls, n_boot=N_BOOT, seed=42):
    np.random.seed(seed)
    arr = np.array(pnls); n = len(arr); boots = []
    for _ in range(n_boot):
        s = np.random.choice(arr, size=n, replace=True)
        w = s[s > 0]; l = s[s <= 0]
        gw = float(w.sum()) if w.size else 0.0
        gl = float(abs(l.sum())) if l.size else 0.0
        if gl > 0: boots.append(gw / gl)
    return np.array(boots)

def _build_session_states(
    feat_valid: pd.DataFrame,
    states: np.ndarray,
    lmap: Dict[int, str],
) -> Dict[dt_date, str]:
    return {feat_valid.iloc[i]["date"]: lmap[int(states[i])]
            for i in range(len(feat_valid))}

def _train_and_predict(
    feat_all: pd.DataFrame,
    train_end_d: dt_date,
    test_start_d: dt_date,
    test_end_d: dt_date,
):
    """Train HMM on [start, train_end_d], predict on [test_start_d, test_end_d]."""
    feat_tr = feat_all[feat_all["date"] <= train_end_d]
    feat_tr_v = feat_tr.dropna(subset=["volume_ratio", "daily_atr"]).copy()

    X_tr = feat_tr_v[["daily_return", "volume_ratio"]].values
    model, states_tr = train_hmm(X_tr, n_states=N_STATES, seed=42)
    lmap = label_states(states_tr, feat_tr_v["daily_return"].values, N_STATES)

    feat_te = feat_all[
        (feat_all["date"] >= test_start_d) & (feat_all["date"] <= test_end_d)
    ]
    feat_te_v = feat_te.dropna(subset=["volume_ratio", "daily_atr"]).copy()

    if feat_te_v.empty:
        return {}, lmap, model, states_tr, feat_tr_v

    X_te = feat_te_v[["daily_return", "volume_ratio"]].values
    states_te = predict_states(model, X_te)
    sess_states = _build_session_states(feat_te_v, states_te, lmap)
    return sess_states, lmap, model, states_tr, feat_tr_v


def _state_dist_line(sess_states: Dict[dt_date, str]) -> str:
    total = max(len(sess_states), 1)
    nb  = sum(1 for s in sess_states.values() if s == "bearish")
    nr  = sum(1 for s in sess_states.values() if s == "ranging")
    nbu = sum(1 for s in sess_states.values() if s == "bullish")
    return (f"Bearish={nb} ({nb/total*100:.1f}%)  "
            f"Ranging={nr} ({nr/total*100:.1f}%)  "
            f"Bullish={nbu} ({nbu/total*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * W)
    print("  PHASE 23 — HMM AS FILTER/SCALER ON POC REVERSION B1")
    print("=" * W)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n  [Loading data and computing indicators...]")
    cfg    = Config()
    df     = load_data(cfg)
    df_ind = add_indicators(df, cfg)

    df_poc = compute_poc_features(df_ind, confluence_threshold=2.0)
    df_ind["prev_poc"]       = df_poc["prev_poc"]
    df_ind["session_poc"]    = df_poc["session_poc"]
    df_ind["poc_confluence"] = df_poc["poc_confluence"]
    df_ind["target_poc"]     = df_poc["target_poc"]

    prr_s, t5d_s, da_s = compute_det_regime_features_v2(df_ind)
    df_ind["prev_range_ratio"] = prr_s
    df_ind["trend_5d"]         = t5d_s
    df_ind["daily_atr"]        = da_s

    date_arr = np.array(df_ind.index.date)

    # ── 2. HMM features ──────────────────────────────────────────────────────
    feat_all = extract_daily_features(df_ind)
    feat_all = add_causal_features(feat_all)

    tr_end_d = pd.Timestamp(TRAIN_END).date()
    te_s_d   = pd.Timestamp(TEST_START).date()
    te_e_d   = pd.Timestamp(TEST_END).date()

    # ── 3. Train HMM and build session states ────────────────────────────────
    print(f"  [Training HMM on training data through {TRAIN_END}...]")
    sess_states_te, lmap, model, _, feat_tr_v = _train_and_predict(
        feat_all, tr_end_d, te_s_d, te_e_d
    )
    # Also get training-period states for diagnostic
    feat_tr_full = feat_all[feat_all["date"] <= tr_end_d].dropna(
        subset=["volume_ratio", "daily_atr"]
    ).copy()
    X_tr_all = feat_tr_full[["daily_return", "volume_ratio"]].values
    _, states_tr_all = train_hmm(X_tr_all, n_states=N_STATES, seed=42)
    lmap_tr = label_states(states_tr_all, feat_tr_full["daily_return"].values, N_STATES)
    sess_states_tr = _build_session_states(feat_tr_full, states_tr_all, lmap_tr)

    print(f"  Training state dist : {_state_dist_line(sess_states_tr)}")
    print(f"  Test state dist     : {_state_dist_line(sess_states_te)}")

    # ── 4. Test-period DataFrame ─────────────────────────────────────────────
    df_te = df_ind[(date_arr >= te_s_d) & (date_arr <= te_e_d)].copy()

    # ── 5. PRE-EXPERIMENT DIAGNOSTIC ─────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  PRE-EXPERIMENT DIAGNOSTIC")
    print(f"  {'='*W}")

    # Section 1 — State distribution by year (training)
    print(f"\n  SECTION 1 — HMM State Distribution by Year (Training Period through {TRAIN_END})")
    print(f"  {'Year':>4} | {'Bearish':>12} | {'Ranging':>12} | {'Bullish':>12} | {'Total':>5}")
    print("  " + "-" * 54)
    by_year: Dict[int, list] = {}
    for d, s in sess_states_tr.items():
        by_year.setdefault(d.year, []).append(s)
    for yr in sorted(by_year.keys()):
        v = by_year[yr]; n = len(v)
        nb  = sum(1 for s in v if s == "bearish")
        nr  = sum(1 for s in v if s == "ranging")
        nbu = sum(1 for s in v if s == "bullish")
        print(f"  {yr:>4} | {nb:>4} ({nb/n*100:>4.1f}%) | "
              f"{nr:>4} ({nr/n*100:>4.1f}%) | "
              f"{nbu:>4} ({nbu/n*100:>4.1f}%) | {n:>5}")

    # Test period summary
    nte: Dict[str, int] = {}
    for s in sess_states_te.values():
        nte[s] = nte.get(s, 0) + 1
    n_te_tot = max(sum(nte.values()), 1)
    print(f"\n  Test period ({TEST_START}–{TEST_END}): "
          f"Bearish={nte.get('bearish',0)} ({nte.get('bearish',0)/n_te_tot*100:.1f}%)  "
          f"Ranging={nte.get('ranging',0)} ({nte.get('ranging',0)/n_te_tot*100:.1f}%)  "
          f"Bullish={nte.get('bullish',0)} ({nte.get('bullish',0)/n_te_tot*100:.1f}%)")

    # Section 2 — B1 trades by state (test period)
    print(f"\n  SECTION 2 — B1 Trade Distribution by HMM State (Test Period)")
    r_base_full = run_b1_baseline(df_te, label="p23_base")
    by_state: Dict[str, list] = {"ranging": [], "bearish": [], "bullish": []}
    for t in r_base_full.trades:
        s = sess_states_te.get(t.entry_ts.date(), "unknown")
        by_state.setdefault(s, []).append(t)
    total_n = len(r_base_full.trades)
    print(f"  {'State':>10} | {'N':>4} | {'%tot':>5} | {'WR%':>5} | "
          f"{'PF':>6} | {'AvgWin':>7} | {'AvgLoss':>8}")
    print("  " + "-" * 60)
    diag_rows = []
    for s_name in ["ranging", "bearish", "bullish", "unknown"]:
        ts = by_state.get(s_name, [])
        if not ts:
            continue
        pnls = [t.pnl_net for t in ts]
        pf_s = _pf(pnls)
        wr_s = sum(1 for x in pnls if x > 0) / len(pnls)
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        aw = float(np.mean(wins))   if wins   else 0.0
        al = float(np.mean(losses)) if losses else 0.0
        pct = len(ts) / max(total_n, 1) * 100
        print(f"  {s_name:>10} | {len(ts):>4} | {pct:>5.1f}% | {wr_s*100:>5.1f}% | "
              f"{_fmt(pf_s):>6} | ${aw:>6.0f} | ${al:>7.0f}")
        if s_name != "unknown":
            diag_rows.append({
                "state": s_name, "n": len(ts),
                "pct_of_total": pct, "wr": wr_s,
                "pf": pf_s if pf_s != float("inf") else -1,
                "avg_win": aw, "avg_loss": al,
            })
    print("  " + "-" * 60)
    m_base_full = _metrics(r_base_full.trades)
    print(f"  {'TOTAL':>10} | {m_base_full['n']:>4} | {'100%':>5} | "
          f"{m_base_full['wr']*100:>5.1f}% | "
          f"{_fmt(m_base_full['pf']):>6} | ${m_base_full['avg_win']:>6.0f} | "
          f"${m_base_full['avg_loss']:>7.0f}")

    # Section 3 — Geometry detail
    print(f"\n  SECTION 3 — Per-State B1 Geometry")
    for s_name in ["ranging", "bearish", "bullish"]:
        ts = by_state.get(s_name, [])
        if not ts:
            print(f"    {s_name}: no trades"); continue
        pnls = [t.pnl_net for t in ts]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        aw = float(np.mean(wins))   if wins   else 0.0
        al = float(np.mean(losses)) if losses else 0.0
        ratio = abs(aw / al) if al != 0 else float("inf")
        pf = _pf(pnls); wr = sum(1 for x in pnls if x > 0) / len(pnls)
        avg_sl = float(np.mean([t.sl_dist for t in ts]))
        print(f"    {s_name:>10}: N={len(ts):>4}  WR={wr*100:>5.1f}%  PF={_fmt(pf)}  "
              f"AvgWin=${aw:>6.0f}  AvgLoss=${al:>7.0f}  "
              f"Win/Loss={ratio:.2f}x  AvgSL={avg_sl:.1f}pts")

    # Section 4 — Expected impact estimate
    print(f"\n  SECTION 4 — Expected Impact Estimate")
    n_r  = len(by_state.get("ranging",  []))
    n_b  = len(by_state.get("bearish",  []))
    n_bu = len(by_state.get("bullish",  []))
    pf_r  = _pf([t.pnl_net for t in by_state.get("ranging",  [])] or [0.0001])
    pf_b  = _pf([t.pnl_net for t in by_state.get("bearish",  [])] or [0.0001])
    pf_bu = _pf([t.pnl_net for t in by_state.get("bullish",  [])] or [0.0001])
    print(f"  Approach A:")
    print(f"    A1 (ranging only):     N~{n_r:>3}  per-state PF={_fmt(pf_r)}")
    print(f"    A2 (ranging+bullish):  N~{n_r+n_bu:>3}  ranging PF={_fmt(pf_r)}  "
          f"bullish PF={_fmt(pf_bu)}")
    print(f"    A3 (bearish only):     N~{n_b:>3}  per-state PF={_fmt(pf_b)}")
    print(f"  Approach B (over-weight ranging, under-weight/skip bearish):")
    print(f"    B1/B2: ranging ({n_r} trades) gets 1.5x/2.0x risk vs 1.0x baseline")
    print(f"    B3/B4: bearish ({n_b} trades) skipped entirely")

    pd.DataFrame(diag_rows).to_csv(
        os.path.join(RESULTS_DIR, "p23_diagnostic.csv"), index=False
    )
    print(f"\n  Saved: p23_diagnostic.csv")

    # ── 6. EXPERIMENTS ────────────────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  EXPERIMENTS  (test period: {0} to {1})".format(TEST_START, TEST_END))
    print(f"  {'='*W}")

    # Header row
    print(f"\n  {'Label':<14} | {'N':>4} | {'WR%':>5} | {'PF':>6} | "
          f"{'SR':>7} | {'MaxDD%':>6} | {'dSR':>7} | {'dPF':>7}")
    print("  " + "-" * (W - 2))

    def _pfmt_delta(d):
        if d != d: return "    nan"
        return f"{d:>+7.3f}"

    # Print and store one result row
    results_summary = []

    def _record(label, name, r: POCResults):
        m = _metrics(r.trades)
        d_sr = m["sharpe"] - m_base_full["sharpe"]
        d_pf = (
            m["pf"] - m_base_full["pf"]
            if m["pf"] != float("inf") and m_base_full["pf"] != float("inf")
            else float("nan")
        )
        print(f"  {label+' '+name:<14} | {m['n']:>4} | {m['wr']*100:>5.1f}% | "
              f"{_fmt(m['pf']):>6} | {m['sharpe']:>+7.3f} | {m['max_dd_pct']:>5.1f}% | "
              f"{_pfmt_delta(d_sr)} | {_pfmt_delta(d_pf)}")
        # Save OOS CSV
        df_out = r.to_df()
        if not df_out.empty:
            safe = name.replace(" ", "_").replace("+", "p").replace("/", "_")
            df_out.to_csv(
                os.path.join(RESULTS_DIR, f"p23_exp{label}_{safe}.csv"), index=False
            )
        results_summary.append({
            "label": label, "name": name,
            "n": m["n"], "wr": m["wr"], "pf": m["pf"], "sharpe": m["sharpe"],
            "max_dd_pct": m["max_dd_pct"],
            "avg_win": m["avg_win"], "avg_loss": m["avg_loss"],
            "d_sr": d_sr, "d_pf": d_pf,
        })
        return m

    # Exp 0 — Baseline
    m0 = _metrics(r_base_full.trades)
    print(f"  {'Exp0 Baseline':<14} | {m0['n']:>4} | {m0['wr']*100:>5.1f}% | "
          f"{_fmt(m0['pf']):>6} | {m0['sharpe']:>+7.3f} | {m0['max_dd_pct']:>5.1f}% | "
          f"{'  —':>7} | {'  —':>7}")
    results_summary.append({
        "label": "Exp0", "name": "Baseline",
        "n": m0["n"], "wr": m0["wr"], "pf": m0["pf"], "sharpe": m0["sharpe"],
        "max_dd_pct": m0["max_dd_pct"],
        "avg_win": m0["avg_win"], "avg_loss": m0["avg_loss"],
        "d_sr": 0.0, "d_pf": 0.0,
    })
    # Save baseline OOS CSV
    df_b = r_base_full.to_df()
    if not df_b.empty:
        df_b.to_csv(
            os.path.join(RESULTS_DIR, "p23_expExp0_Baseline.csv"), index=False
        )

    # Approach A — Hard Filter
    print(f"\n  -- Approach A: Hard Filter --")
    A_CONFIGS = [
        ("A1", "ranging only",        {"ranging"}),
        ("A2", "ranging+bullish",     {"ranging", "bullish"}),
        ("A3", "bearish only",        {"bearish"}),
    ]
    exp_metrics: Dict[str, dict] = {"Exp0": m0}
    for eid, name, allowed in A_CONFIGS:
        r = run_filtered(df_te, sess_states_te, allowed, label=f"p23_{eid}")
        exp_metrics[eid] = _record(eid, name, r)

    # Approach B — Position Scaler
    print(f"\n  -- Approach B: Position Scaler (capital={CAPITAL:,.0f}) --")
    B_CONFIGS = [
        ("B1", "1.5_0.5_1.0", {"ranging": 0.015, "bearish": 0.005, "bullish": 0.010}),
        ("B2", "2.0_0.5_1.0", {"ranging": 0.020, "bearish": 0.005, "bullish": 0.010}),
        ("B3", "1.5_skip_1.0",{"ranging": 0.015, "bearish": None,  "bullish": 0.010}),
        ("B4", "1.0_skip_1.0",{"ranging": 0.010, "bearish": None,  "bullish": 0.010}),
    ]
    for eid, name, risk_map in B_CONFIGS:
        r = run_scaled(df_te, sess_states_te, risk_map, label=f"p23_{eid}")
        exp_metrics[eid] = _record(eid, name, r)

    # ── 7. SUMMARY TABLE ─────────────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  SUMMARY TABLE")
    print(f"  {'='*W}")
    print(f"  {'Label':<14} | {'Name':<16} | {'N':>4} | {'PF':>6} | {'SR':>7} | "
          f"{'MaxDD%':>6} | {'dSR':>7} | {'dPF':>7} | {'Beat?':>5}")
    print("  " + "-" * (W - 2))
    for row in results_summary:
        beat = (
            row["sharpe"] > m0["sharpe"] and
            row["pf"] not in (float("nan"), float("inf")) and
            row["pf"] > 1.0
        ) if row["label"] != "Exp0" else None
        beat_str = "YES" if beat else ("—" if beat is None else "no")
        print(f"  {row['label']:<14} | {row['name']:<16} | {row['n']:>4} | "
              f"{_fmt(row['pf']):>6} | {row['sharpe']:>+7.3f} | "
              f"{row['max_dd_pct']:>5.1f}% | {_pfmt_delta(row['d_sr'])} | "
              f"{_pfmt_delta(row['d_pf'])} | {beat_str:>5}")

    # ── 8. AUTO-SELECTION ─────────────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print("  AUTO-SELECTION")
    print(f"  {'='*W}")

    all_exp_ids = [r["label"] for r in results_summary if r["label"] != "Exp0"]
    candidates = []
    for eid in all_exp_ids:
        m = exp_metrics[eid]
        if (m["sharpe"] > m0["sharpe"] and
                m["pf"] not in (float("nan"),) and m["pf"] > 1.0):
            candidates.append((eid, m))

    if not candidates:
        print("\n  No config beats baseline on both SR and PF > 1.0.")
        print("  HYPOTHESIS NOT SUPPORTED. Do not proceed to WFO.")
        print(f"  {'='*W}")
        return

    # Sort: primary SR desc, secondary MaxDD asc, tertiary N desc
    candidates.sort(key=lambda x: (-x[1]["sharpe"], x[1]["max_dd_pct"], -x[1]["n"]))
    best_label, best_m = candidates[0]
    print(f"\n  Candidates beating baseline (SR>{m0['sharpe']:.3f}, PF>1.0):")
    for eid, m in candidates:
        flag = " <<< BEST" if eid == best_label else ""
        print(f"    {eid}: SR={m['sharpe']:+.3f}  PF={_fmt(m['pf'])}  "
              f"N={m['n']}  MaxDD={m['max_dd_pct']:.1f}%{flag}")

    # Get best config parameters
    best_is_A = best_label.startswith("A")
    best_config_str = ""
    best_kwargs = {}
    if best_is_A:
        for eid, name, allowed in A_CONFIGS:
            if eid == best_label:
                best_config_str = f"Approach A — {name}"
                best_kwargs = {"allowed_states": allowed}
                break
    else:
        for eid, name, risk_map in B_CONFIGS:
            if eid == best_label:
                best_config_str = f"Approach B — {name}"
                best_kwargs = {"state_risk_map": risk_map}
                break

    print(f"\n  Best: {best_label}  ({best_config_str})")
    print(f"        SR={best_m['sharpe']:+.3f}  PF={_fmt(best_m['pf'])}  "
          f"N={best_m['n']}  MaxDD={best_m['max_dd_pct']:.1f}%")

    # WFO gate: SR > baseline AND PF > baseline AND N >= 80
    pf_base = m0["pf"] if m0["pf"] != float("inf") else 1.0
    wfo_eligible = (
        best_m["sharpe"] > m0["sharpe"] and
        best_m["pf"] > pf_base and
        best_m["n"] >= 80
    )
    print(f"\n  WFO gate: SR>{m0['sharpe']:.3f}? {best_m['sharpe']:.3f} [ok]  "
          f"PF>{pf_base:.3f}? {_fmt(best_m['pf'])} {'[ok]' if best_m['pf'] > pf_base else '[FAIL]'}  "
          f"N>=80? {best_m['n']} {'[ok]' if best_m['n'] >= 80 else '[FAIL]'}")

    if not wfo_eligible:
        print(f"\n  WFO SKIPPED — gate not passed "
              f"(N={best_m['n']} {'<' if best_m['n'] < 80 else '>='} 80, "
              f"PF {_fmt(best_m['pf'])} {'>' if best_m['pf'] > pf_base else '<='} {_fmt(pf_base)}).")
        print(f"  {'='*W}")
        return

    # ── 9. WALK-FORWARD VALIDATION ────────────────────────────────────────────
    print(f"\n  {'='*W}")
    print(f"  WALK-FORWARD VALIDATION  ({best_label}: {best_config_str})")
    print(f"  {'='*W}")
    print(f"\n  {'V':>1} | {'Test period':<22} | {'N':>4} | {'PF':>6} | "
          f"{'SR':>7} | {'WR%':>5} | {'MaxDD%':>6}")
    print("  " + "-" * (W - 4))

    all_oos_pnl = []
    wfo_rows = []
    oos_dfs = []

    for vn, (tr_s, tr_e, te_s, te_e) in enumerate(WINDOWS, start=1):
        tr_e_d_w = pd.Timestamp(tr_e).date()
        te_s_d_w = pd.Timestamp(te_s).date()
        te_e_d_w = pd.Timestamp(te_e).date()

        # Re-train HMM for this window
        sess_states_w, lmap_w, _, _, _ = _train_and_predict(
            feat_all, tr_e_d_w, te_s_d_w, te_e_d_w
        )

        df_te_w = df_ind[(date_arr >= te_s_d_w) & (date_arr <= te_e_d_w)].copy()

        if best_is_A:
            r_w = run_filtered(
                df_te_w, sess_states_w, best_kwargs["allowed_states"],
                label=f"p23_wfo_v{vn}",
            )
        else:
            r_w = run_scaled(
                df_te_w, sess_states_w, best_kwargs["state_risk_map"],
                label=f"p23_wfo_v{vn}",
            )

        m_w = _metrics(r_w.trades)
        fl  = "*" if m_w["n"] < 30 else " "
        if m_w["n"] >= 30:
            all_oos_pnl.extend([t.pnl_net for t in r_w.trades])

        print(f"  {vn:>1} | {te_s} -> {te_e} | {m_w['n']:>3}{fl} | "
              f"{_fmt(m_w['pf']):>6} | {m_w['sharpe']:>+7.3f} | "
              f"{m_w['wr']*100:>4.0f}% | {m_w['max_dd_pct']:>5.1f}%")

        if not r_w.to_df().empty:
            df_out = r_w.to_df()
            df_out.insert(0, "window", vn)
            oos_dfs.append(df_out)

        wfo_rows.append((vn, te_s, te_e, m_w))

    # WFO summary
    pf_gt1 = sum(1 for *_, m in wfo_rows if m["pf"] > 1.0)
    sr_gt0 = sum(1 for *_, m in wfo_rows if m["sharpe"] > 0)
    print("  " + "-" * (W - 4))

    valid_w = [m for *_, m in wfo_rows if m["n"] >= 30]
    if valid_w:
        mu_pf = np.mean([min(m["pf"], 5.0) for m in valid_w if m["pf"] == m["pf"]])
        mu_sr = np.mean([m["sharpe"] for m in valid_w])
        print(f"  mu| (valid windows >=30 trades)            | "
              f"{_fmt(mu_pf):>6} | {mu_sr:>+7.3f}")

    print(f"\n  Windows PF > 1.0 : {pf_gt1}/5")
    print(f"  Windows SR > 0.0 : {sr_gt0}/5")

    # Statistical tests on pooled OOS
    n_oos = len(all_oos_pnl)
    print(f"\n  {'='*W}")
    print(f"  STATISTICAL TESTS  (pooled OOS: {n_oos} trades)")
    print(f"  {'='*W}")

    if n_oos < 10:
        print("  Insufficient pooled OOS trades for statistical tests.")
    else:
        arr = np.array(all_oos_pnl)
        t_stat, p_two = stats.ttest_1samp(arr, 0.0)
        p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        pb = _bootstrap_pf(arr)
        mn_pf = float(pb.mean())
        p5    = float(np.percentile(pb, 5))
        p95   = float(np.percentile(pb, 95))
        print(f"  T-test (H1: mean > 0):  t={t_stat:.3f}  p={p_one:.4f}  "
              f"-> {'Significant p<0.10' if p_one < 0.10 else 'Not significant p>0.10'}")
        print(f"  Bootstrap PF ({N_BOOT} iter): mean={mn_pf:.3f}  "
              f"p5={p5:.3f}  p95={p95:.3f}  "
              f"-> {'Strong (p5>1.0)' if p5 > 1.0 else 'Moderate (p5>0.95)' if p5 > 0.95 else 'Insufficient (p5<0.95)'}")

        # Pooled OOS PF
        pf_pool = _pf(all_oos_pnl)
        sr_pool = _sr(all_oos_pnl)
        print(f"  Pooled OOS:  PF={_fmt(pf_pool)}  SR={sr_pool:+.3f}  N={n_oos}")

        confirmed = p_one < 0.10 and p5 > 0.95
        print(f"\n  {'='*W}")
        print(f"  VERDICT: {'EDGE CONFIRMED' if confirmed else 'Not sufficient to confirm edge'}")
        print(f"  p={p_one:.4f} (target <0.10)  p5={p5:.3f} (target >0.95)  "
              f"WFO: {pf_gt1}/5 windows PF>1.0")
        if confirmed:
            print(f"  HMM filter/scaler adds confirmed value over B1 baseline.")
        else:
            print(f"  HMM overlay does not pass confirmation threshold over B1 baseline.")
        print(f"  {'='*W}")

    # Save WFO pooled OOS
    if oos_dfs:
        pd.concat(oos_dfs, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "p23_wfo_oos_pooled.csv"), index=False
        )
        print(f"\n  Saved: p23_wfo_oos_pooled.csv")


if __name__ == "__main__":
    main()
