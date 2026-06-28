#!/usr/bin/env python3
"""
Phase 22B — Bearish Spike Fade: Entry and TP Calibration.

Hypothesis: On days where HMM transition matrix predicts P(bearish|yesterday) > 0.60,
NQ typically opens with an upward spike of ~0.221×prev_atr before reversing.
Entering SHORT when price reaches 1/3 of expected spike captures a favorable R/R.

Training: Jun 2021 → Dec 2024
Test:      Jan 2025 → Jun 2026
"""
import os
import sys
from collections import defaultdict
from datetime import time as dt_time
from datetime import date as dt_date

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.strategy.hmm_transition import (
    add_causal_features, extract_daily_features,
    label_states, train_hmm, compute_transition_matrix,
)
from orb_system.strategy.bearish_spike_fade import (
    compute_session_params, run_backtest,
    SPIKE_RATIO, SL_BUFFER, TRAIL_MULT, TRAIL_ACT,
    SLIP, PV, COMM_RT, SESS_OPEN, ENTRY_END, HARD_EXIT, MAX_BARS,
    P_BEAR_THRESH,
)

RESULTS    = os.path.join(ROOT, "results")
TRAIN_END  = "2024-12-31"
INIT_CAP   = 100_000.0
BASE_RISK  = 1.0
W          = 72

WFO_WINDOWS = [
    (1, "2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    (2, "2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    (3, "2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    (4, "2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    (5, "2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_date_index(df):
    da = np.array(df.index.date)
    m = {}
    for i, d in enumerate(da):
        m.setdefault(d, []).append(i)
    return {d: np.array(v) for d, v in m.items()}


def _save_csv(rows, fname):
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, fname), index=False)


def _pf(trades):
    wins = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    loss = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
    if loss == 0: return np.inf
    if wins == 0: return 0.0
    return wins / loss


def _wr(trades):
    if not trades: return 0.0
    return sum(1 for t in trades if t["pnl_usd"] > 0) / len(trades) * 100


def _sr(trades, first_date, last_date):
    if not trades: return np.nan
    dr = pd.date_range(first_date, last_date, freq="B")
    daily = defaultdict(float)
    for t in trades:
        daily[t["date"]] += t["pnl_usd"]
    pnl_d = np.array([daily.get(d.strftime("%Y-%m-%d"), 0.0) for d in dr])
    if pnl_d.std() == 0: return np.nan
    return float(pnl_d.mean() / pnl_d.std() * np.sqrt(252))


def _maxdd(trades, initial_capital):
    if not trades: return 0.0
    eq = [initial_capital]
    for t in trades:
        eq.append(eq[-1] + t["pnl_usd"])
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / initial_capital * 100).min())


def _ret_pct(trades, initial_capital):
    if not trades: return 0.0
    total = sum(t["pnl_usd"] for t in trades)
    return total / initial_capital * 100


def _exits(trades):
    if not trades: return {r: 0.0 for r in ["SL", "TP", "TRAIL", "TIME", "EOD"]}
    n = len(trades)
    return {r: sum(1 for t in trades if t["exit_reason"] == r) / n * 100
            for r in ["SL", "TP", "TRAIL", "TIME", "EOD"]}


def _year_rows(trades, initial_capital):
    rows = {}
    by_year = defaultdict(list)
    for t in trades:
        by_year[t["year"]].append(t)
    for yr, ytr in sorted(by_year.items()):
        rows[yr] = {
            "N": len(ytr),
            "WR": _wr(ytr),
            "PF": _pf(ytr),
            "ret": _ret_pct(ytr, initial_capital),
        }
    return rows


def _stats(trades, fd, ld, cap=INIT_CAP):
    n = len(trades)
    if n == 0:
        return dict(N=0, WR=0.0, PF=0.0, SR=np.nan, ret=0.0, maxdd=0.0,
                    exits={r: 0.0 for r in ["SL","TP","TRAIL","TIME","EOD"]},
                    win_pts=0.0, los_pts=0.0, rr=np.nan, avg_nc=0.0)
    winners = [t for t in trades if t["pnl_usd"] > 0]
    losers  = [t for t in trades if t["pnl_usd"] < 0]
    wpts = float(np.mean([t["pnl_pts"] for t in winners])) if winners else 0.0
    lpts = float(np.mean([t["pnl_pts"] for t in losers]))  if losers  else 0.0
    rr   = abs(wpts / lpts) if lpts < 0 else np.nan
    return dict(
        N=n, WR=_wr(trades), PF=_pf(trades), SR=_sr(trades, fd, ld),
        ret=_ret_pct(trades, cap), maxdd=_maxdd(trades, cap),
        exits=_exits(trades), win_pts=wpts, los_pts=lpts, rr=rr,
        avg_nc=float(np.mean([t["n_contracts"] for t in trades])),
    )


def _print_stats(label, s, year_rows, cap=INIT_CAP):
    n = s["N"]
    if n == 0:
        print(f"  {label}: No trades")
        return
    e = s["exits"]
    print(f"  {label}: N={n} | WR={s['WR']:.1f}% | PF={s['PF']:.3f} | "
          f"SR={s['SR']:+.3f} | Ret={s['ret']:+.1f}% | MaxDD={s['maxdd']:.1f}%")
    print(f"    Exits: SL={e['SL']:.1f}% TP={e['TP']:.1f}% "
          f"Trail={e['TRAIL']:.1f}% Time={e['TIME']:.1f}% EOD={e['EOD']:.1f}%")
    print(f"    Avg winner: {s['win_pts']:+.1f}pts | Avg loser: {s['los_pts']:+.1f}pts | "
          f"R/R realized: {s['rr']:.2f}x | Avg contracts: {s['avg_nc']:.1f}")
    if year_rows:
        print(f"    Annual: ", end="")
        parts = [f"{yr}: N={v['N']} WR={v['WR']:.0f}% PF={v['PF']:.2f} ret={v['ret']:+.1f}%"
                 for yr, v in sorted(year_rows.items())]
        print("  |  ".join(parts))


# ── diagnostic ────────────────────────────────────────────────────────────────

def run_diagnostic(df_raw, time_arr, date_idx_map, session_params,
                   feat_tr, tr_dates):
    W2 = 72
    print(f"\n{'=' * W2}")
    print("  PRE-PNL DIAGNOSTIC (training set only)")
    print(f"{'=' * W2}")

    high_v  = df_raw["high"].values
    low_v   = df_raw["low"].values
    close_v = df_raw["close"].values
    open_v  = df_raw["open"].values
    vol_v   = df_raw["volume"].values

    # § 1 — Signal frequency
    sig_days = [d for d in tr_dates if session_params.get(d, {}).get("signal_active")]
    by_year = defaultdict(list)
    for d in sig_days:
        by_year[d.year].append(d)

    print(f"\n  [1] SIGNAL FREQUENCY")
    print(f"  Total signal days (train): {len(sig_days)}  "
          f"| Monthly avg: {len(sig_days)/42:.1f}")
    print(f"  {'Year':>6} | {'Signal days':>12} | {'Pct of sessions':>16}")
    print(f"  {'-' * 38}")
    tr_by_year = defaultdict(list)
    for d in tr_dates:
        tr_by_year[d.year].append(d)
    for yr in sorted(by_year):
        pct = len(by_year[yr]) / len(tr_by_year[yr]) * 100
        print(f"  {yr:>6} | {len(by_year[yr]):>12} | {pct:>15.1f}%")

    # Scan entry fires on signal days
    entry_days  = []   # dates where entry fired
    entry_info  = []   # full info dicts

    from orb_system.strategy.bearish_spike_fade import BARS_PER_SESS, VOL_MULT

    for d in sig_days:
        p = session_params[d]
        if d not in date_idx_map:
            continue
        abs_idx = date_idx_map[d]; dt = time_arr[abs_idx]
        rth = (dt >= SESS_OPEN) & (dt <= HARD_EXIT)
        ra = abs_idx[rth]; rt = dt[rth]
        if len(ra) == 0:
            continue

        entry_trigger = p["open_930"] + (1/3) * p["expected_spike"]
        sl_price      = p["sl_price"]
        avg_vol       = p["avg_vol"]
        bvt = (avg_vol / BARS_PER_SESS * VOL_MULT
               if not np.isnan(avg_vol) and avg_vol > 0 else 0.0)

        fired = False; ep = None; ej = None
        for j, aj in enumerate(ra):
            bt = rt[j]
            if bt > ENTRY_END: break
            bh = high_v[aj]; bc = close_v[aj]; bo = open_v[aj]; bv = vol_v[aj]
            if (bh >= entry_trigger and bc < bo and bc < entry_trigger and bv > bvt):
                ep = bc; ej = j; fired = True; break

        if not fired:
            continue

        entry_days.append(d)
        # gather session extremes for diagnostics
        sh = float(high_v[ra].max()); sl_ = float(low_v[ra].min())
        open_930 = p["open_930"]

        # MFE from entry to EOD
        mfe_pts = max(0.0, ep - float(low_v[ra[ej+1:]].min())) if ej+1 < len(ra) else 0.0

        entry_info.append({
            "date": d,
            "open_930": open_930,
            "entry_price": ep,
            "entry_trigger": entry_trigger,
            "sl_price": sl_price,
            "session_high": sh,
            "session_low": sl_,
            "spike_drop": sl_ < open_930,          # price ended below open
            "sl_breached": sh > sl_price,           # spike exceeded SL level
            "mfe_from_entry": mfe_pts,
            "prev_atr": p["prev_atr"],
            "prev_poc": p["prev_poc"],
            "expected_spike": p["expected_spike"],
        })

    n_sig = len(sig_days); n_fire = len(entry_days)
    fire_pct = n_fire / n_sig * 100 if n_sig else 0.0
    print(f"\n  Entry fires (signal days where entry triggered): "
          f"{n_fire}/{n_sig}  ({fire_pct:.1f}%)")
    print(f"  Expected trades/month: {n_fire/42:.1f}")

    if not entry_info:
        print("\n  GATE: No entries fired on training set. STOP.")
        return None

    # § 2 — Entry quality
    n_sd = sum(1 for e in entry_info if e["spike_drop"])
    n_cont = n_fire - n_sd
    print(f"\n  [2] ENTRY QUALITY (of {n_fire} fired entries)")
    print(f"  Spike→Drop confirmed (price eventually below open): "
          f"{n_sd} ({n_sd/n_fire*100:.1f}%)")
    print(f"  Continuation (price stayed above open):             "
          f"{n_cont} ({n_cont/n_fire*100:.1f}%)")

    # § 3 — SL validation
    n_sl_breach = sum(1 for e in entry_info if e["sl_breached"])
    print(f"\n  [3] SL VALIDATION")
    print(f"  Sessions where session_high > sl_price (spike exceeded SL level):")
    print(f"    {n_sl_breach}/{n_fire} = {n_sl_breach/n_fire*100:.1f}%")
    if n_sl_breach / n_fire > 0.20:
        print(f"  WARNING: > 20% SL breach rate. Formula may be too tight.")
    else:
        print(f"  SL formula adequate (< 20% breach rate).")

    # § 4 — TP distance analysis
    print(f"\n  [4] TP DISTANCE ANALYSIS")

    sl_dists = [e["sl_price"] - e["entry_trigger"] for e in entry_info]
    tp_a_dists = [2 * sd for sd in sl_dists]
    print(f"  TP-A (2:1 fixed): tp_dist = 2×sl_dist")
    p25, p50, p75, p90 = np.percentile(tp_a_dists, [25, 50, 75, 90])
    print(f"    sl_dist  p25={np.percentile(sl_dists,25):.1f} p50={np.percentile(sl_dists,50):.1f} "
          f"p75={np.percentile(sl_dists,75):.1f}")
    print(f"    tp_dist  p25={p25:.1f} p50={p50:.1f} p75={p75:.1f} p90={p90:.1f} pts")

    for x_val in [1.0, 1.5, 2.0]:
        tp_b_prices = [e["open_930"] - x_val * e["prev_atr"] for e in entry_info]
        tp_b_dists  = [ep - tp for ep, tp in zip(
                       [e["entry_trigger"] for e in entry_info], tp_b_prices)]
        # % where price eventually reaches TP-B within 3h (approx from session_low proxy)
        # Check if session_low <= tp_price (imprecise but fast diagnostic)
        reached = sum(1 for e, tp in zip(entry_info, tp_b_prices)
                      if e["session_low"] <= tp)
        print(f"  TP-B X={x_val}: tp_dist p50={np.median(tp_b_dists):.1f}pts | "
              f"session_low reached TP level: {reached}/{n_fire} ({reached/n_fire*100:.1f}%)")

    mfe_arr = np.array([e["mfe_from_entry"] for e in entry_info])
    p25, p50, p75, p90 = np.percentile(mfe_arr, [25, 50, 75, 90])
    print(f"  TP-C (trailing): MFE from entry  "
          f"p25={p25:.1f} p50={p50:.1f} p75={p75:.1f} p90={p90:.1f} pts")

    valid_d = [e for e in entry_info if not np.isnan(e["prev_poc"])]
    poc_valid = [e for e in valid_d if e["prev_poc"] < e["entry_trigger"]]
    if valid_d:
        poc_dists = [e["entry_trigger"] - e["prev_poc"] for e in poc_valid]
        print(f"  TP-D (prev_poc): {len(poc_valid)}/{len(valid_d)} sessions have "
              f"poc < entry_trigger ({len(poc_valid)/len(valid_d)*100:.1f}%)")
        if poc_dists:
            print(f"    entry→poc dist  p25={np.percentile(poc_dists,25):.1f} "
                  f"p50={np.percentile(poc_dists,50):.1f} "
                  f"p75={np.percentile(poc_dists,75):.1f} pts")

    # § 5 — Theoretical expectancy (no costs)
    print(f"\n  [5] THEORETICAL EXPECTANCY (no costs, training set)")
    print(f"  {'TP variant':18} | {'N':>5} | {'WR%':>6} | {'avgWin R':>9} | "
          f"{'avgLoss R':>10} | {'Expectancy':>11} | {'Gate':>6}")
    print(f"  {'-' * 72}")

    theo_configs = [
        ("TP-A (2:1)",     "A", None,  1/3),
        ("TP-B X=1.0",     "B", 1.0,   1/3),
        ("TP-B X=1.5",     "B", 1.5,   1/3),
        ("TP-B X=2.0",     "B", 2.0,   1/3),
        ("TP-C (trail)",   "C", None,  1/3),
        ("TP-D (poc)",     "D", None,  1/3),
    ]

    any_positive = False
    for tlabel, tv, tx, ef in theo_configs:
        tt = run_backtest(df_raw, time_arr, date_idx_map, session_params,
                         tr_dates, tv, tx, ef, INIT_CAP, BASE_RISK,
                         theoretical=True)
        if not tt:
            print(f"  {tlabel:18} | {'0':>5} | {'—':>6} | {'—':>9} | "
                  f"{'—':>10} | {'—':>11} | {'—':>6}")
            continue
        rrs = [t["rr_realized"] for t in tt if not np.isnan(t.get("rr_realized", np.nan))]
        if not rrs:
            continue
        wins = [r for r in rrs if r > 0]; loss = [r for r in rrs if r <= 0]
        wr   = len(wins) / len(rrs) * 100
        avgW = float(np.mean(wins)) if wins else 0.0
        avgL = float(np.mean(loss)) if loss else 0.0
        exp  = wr/100 * avgW + (1 - wr/100) * avgL
        gate = "✓" if exp > 0 else "✗"
        if exp > 0: any_positive = True
        print(f"  {tlabel:18} | {len(rrs):>5} | {wr:>5.1f}% | {avgW:>+9.3f} | "
              f"{avgL:>+10.3f} | {exp:>+11.3f} | {gate:>6}")

    if not any_positive:
        print("\n  GATE FAILURE: All TP variants negative. STOP.")
        return None

    print(f"\n  GATE PASSED: At least one TP variant shows positive expectancy.")
    return {"n_signal_days": n_sig, "n_fired": n_fire, "n_spike_drop": n_sd,
            "n_sl_breach": n_sl_breach}


# ── experiment reporter ───────────────────────────────────────────────────────

def print_exp(exp_no, name, trades_tr, trades_te, fd_tr, ld_tr, fd_te, ld_te):
    W2 = 72
    print(f"\n{'═' * W2}")
    print(f"  EXPERIMENT {exp_no} — {name}")
    print(f"{'═' * W2}")
    str_tr = _stats(trades_tr, fd_tr, ld_tr, INIT_CAP)
    str_te = _stats(trades_te, fd_te, ld_te, INIT_CAP)
    yr_tr  = _year_rows(trades_tr, INIT_CAP)
    yr_te  = _year_rows(trades_te, INIT_CAP)
    _print_stats("TRAIN", str_tr, yr_tr)
    _print_stats("TEST ", str_te, yr_te)


# ── WFO ───────────────────────────────────────────────────────────────────────

def _bootstrap_pf(pnls, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    pf_s = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        w = s[s > 0].sum(); l = abs(s[s < 0].sum())
        pf_s.append(w / l if l > 0 else np.inf)
    pf_s = np.array([x for x in pf_s if np.isfinite(x)])
    return float(np.mean(pf_s)), float(np.percentile(pf_s, 5)), float(np.percentile(pf_s, 95))


def run_wfo(df_raw, time_arr, date_idx_map, feat_all_full,
            poc_per_date, avg_vol_per_date, best_tp, best_x, best_frac):
    print(f"\n{'=' * W}")
    print(f"  WALK-FORWARD VALIDATION (5 anchored windows)")
    print(f"  TP={best_tp} | X={best_x} | entry_frac={best_frac:.4f}")
    print(f"{'=' * W}")

    pooled = []
    win_pf = []

    for win, tr_s, tr_e, te_s, te_e in WFO_WINDOWS:
        fa_w = feat_all_full.copy()
        fa_w = fa_w[(fa_w["date"].astype(str) >= tr_s) &
                    (fa_w["date"].astype(str) <= te_e)]
        fa_w = fa_w.dropna(subset=["volume_ratio", "daily_atr"]).reset_index(drop=True)

        fa_tr_w = fa_w[fa_w["date"].astype(str) <= tr_e].reset_index(drop=True)
        X_tr_w  = fa_tr_w[["daily_return", "volume_ratio"]].values
        if len(X_tr_w) < 20:
            print(f"  Window {win}: insufficient training data. Skip.")
            continue

        mod_w, st_w = train_hmm(X_tr_w, 3, seed=42)
        lmap_w = label_states(st_w, fa_tr_w["daily_return"].values, 3)

        sp_w = compute_session_params(
            fa_w, poc_per_date, avg_vol_per_date,
            mod_w, st_w, lmap_w, 3, tr_e,
        )

        te_dates_w = fa_w[fa_w["date"].astype(str) > tr_e]["date"].values
        trades_w = run_backtest(df_raw, time_arr, date_idx_map, sp_w,
                                te_dates_w, best_tp, best_x, best_frac,
                                INIT_CAP, BASE_RISK)
        n_w   = len(trades_w)
        pf_w  = _pf(trades_w)
        sr_w  = _sr(trades_w, te_s, te_e) if trades_w else np.nan
        pooled.extend(trades_w)
        win_pf.append(pf_w)

        print(f"  V{win} ({te_s}→{te_e}): N={n_w:>3} | "
              f"PF={pf_w:.3f} | SR={sr_w:+.3f}")

    if not pooled:
        print("  No OOS trades. Cannot assess edge.")
        return

    pnls = np.array([t["pnl_usd"] for t in pooled])
    pf_pool = _pf(pooled)
    sr_pool = _sr(pooled, WFO_WINDOWS[0][3], WFO_WINDOWS[-1][4])
    tstat, pval = ttest_1samp(pnls, 0.0)
    p1 = pval / 2 if tstat > 0 else 1.0 - pval / 2
    bpf_mean, bpf_p5, bpf_p95 = _bootstrap_pf(pnls)
    n_pos = sum(1 for pf in win_pf if pf > 1.0)

    print(f"\n  POOLED OOS ({len(pooled)} trades):")
    print(f"    PF={pf_pool:.3f} | SR={sr_pool:+.3f}")
    print(f"    T-test p (one-sided)={p1:.4f}")
    print(f"    Bootstrap PF — mean={bpf_mean:.3f} | p5={bpf_p5:.3f} | p95={bpf_p95:.3f}")
    print(f"    Windows PF>1.0: {n_pos}/5")

    if p1 < 0.10 and bpf_p5 > 0.95 and n_pos >= 3:
        print(f"\n  ✓ EDGE CONFIRMED. Proceed to Monte Carlo FTMO sizing.")
    else:
        print(f"\n  ✗ Edge NOT confirmed.")
        reasons = []
        if p1 >= 0.10:   reasons.append(f"p={p1:.4f} ≥ 0.10")
        if bpf_p5 <= 0.95: reasons.append(f"bootstrap p5={bpf_p5:.3f} ≤ 0.95")
        if n_pos < 3:    reasons.append(f"{n_pos}/5 windows PF>1.0")
        print(f"    Failed: {' | '.join(reasons)}")

    _save_csv(pooled, "p22b_wfo_oos_pooled.csv")
    print(f"  Saved: results/p22b_wfo_oos_pooled.csv ({len(pooled)} rows)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 22B — BEARISH SPIKE FADE: ENTRY AND TP CALIBRATION")
    print("=" * W)

    cfg    = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]} → {df_raw.index[-1]}")

    time_arr     = np.array(df_raw.index.time)
    date_idx_map = _build_date_index(df_raw)

    # ── features ──────────────────────────────────────────────────────────
    feat_all_raw = add_causal_features(extract_daily_features(df_raw))
    feat_all = (feat_all_raw
                .dropna(subset=["volume_ratio", "daily_atr"])
                .sort_values("date").reset_index(drop=True))

    # avg session volume (20-day causal rolling)
    feat_all = feat_all.copy()
    feat_all["avg_vol_20"] = feat_all["volume"].shift(1).rolling(20, min_periods=5).mean()
    avg_vol_per_date = dict(zip(feat_all["date"].values, feat_all["avg_vol_20"].values))

    # prev_poc per session date
    poc_df = compute_poc_features(df_raw)
    # one prev_poc value per date = first bar of RTH for that date
    poc_per_date = {}
    date_arr = np.array(df_raw.index.date)
    ta       = np.array(df_raw.index.time)
    pv       = poc_df["prev_poc"].values
    for d in np.unique(date_arr):
        mask = (date_arr == d) & (ta == dt_time(9, 30))
        if mask.any():
            val = float(pv[np.where(mask)[0][0]])
            if not np.isnan(val):
                poc_per_date[d] = val

    # ── HMM training set ──────────────────────────────────────────────────
    feat_tr = feat_all[feat_all["date"].astype(str) <= TRAIN_END].reset_index(drop=True)
    feat_te = feat_all[feat_all["date"].astype(str) >  TRAIN_END].reset_index(drop=True)

    X_tr       = feat_tr[["daily_return", "volume_ratio"]].values
    model, states_tr = train_hmm(X_tr, 3, seed=42)
    lmap       = label_states(states_tr, feat_tr["daily_return"].values, 3)
    inv        = {v: k for k, v in lmap.items()}

    print(f"  Training sessions: {len(feat_tr)} | Test sessions: {len(feat_te)}")
    print(f"  State map: {lmap}")

    # ── session params (all dates) ────────────────────────────────────────
    session_params = compute_session_params(
        feat_all, poc_per_date, avg_vol_per_date,
        model, states_tr, lmap, 3, TRAIN_END,
    )

    tr_dates = feat_tr["date"].values
    te_dates = feat_te["date"].values
    fd_tr = str(tr_dates[0]); ld_tr = TRAIN_END
    fd_te = str(te_dates[0]); ld_te = str(te_dates[-1])

    # ── diagnostic ────────────────────────────────────────────────────────
    diag = run_diagnostic(df_raw, time_arr, date_idx_map, session_params,
                          tr_dates, tr_dates)
    if diag is None:
        return   # hard gate triggered

    # ── experiments ───────────────────────────────────────────────────────
    EXP_BASE = [
        (1, "TP-A (2:1 fixed)",         "A", None, 1/3,  BASE_RISK),
        (2, "TP-B X=1.0 (open-1.0×ATR)","B", 1.0,  1/3,  BASE_RISK),
        (3, "TP-B X=1.5 (open-1.5×ATR)","B", 1.5,  1/3,  BASE_RISK),
        (4, "TP-B X=2.0 (open-2.0×ATR)","B", 2.0,  1/3,  BASE_RISK),
        (5, "TP-C (trailing stop)",      "C", None, 1/3,  BASE_RISK),
        (6, "TP-D (prev_poc target)",    "D", None, 1/3,  BASE_RISK),
    ]

    results = []

    print(f"\n{'=' * W}")
    print("  EXPERIMENTS 1–6")
    print(f"{'=' * W}")

    all_trades_tr = {}; all_trades_te = {}

    for exp_no, name, tv, tx, ef, rp in EXP_BASE:
        trades_tr = run_backtest(df_raw, time_arr, date_idx_map, session_params,
                                 tr_dates, tv, tx, ef, INIT_CAP, rp)
        trades_te = run_backtest(df_raw, time_arr, date_idx_map, session_params,
                                 te_dates, tv, tx, ef, INIT_CAP, rp)
        all_trades_tr[exp_no] = trades_tr
        all_trades_te[exp_no] = trades_te

        print_exp(exp_no, name, trades_tr, trades_te, fd_tr, ld_tr, fd_te, ld_te)

        st = _stats(trades_te, fd_te, ld_te, INIT_CAP)
        results.append({
            "exp": exp_no, "name": name, "tp_variant": tv, "tp_x": tx,
            "entry_frac": ef, "risk_pct": rp,
            "pf_tr": _pf(trades_tr), "pf_te": st["PF"],
            "sr_te": st["SR"], "n_te": st["N"],
            "wr_te": st["WR"], "ret_te": st["ret"],
            "maxdd_te": st["maxdd"],
        })

        _save_csv(trades_tr, f"p22b_exp{exp_no}_train.csv")
        _save_csv(trades_te, f"p22b_exp{exp_no}_test.csv")

    # find best TP (SR on test, N >= 60)
    qualified = [r for r in results if r["n_te"] >= 60 and r["pf_te"] > 1.0]
    if qualified:
        best = max(qualified, key=lambda r: r["sr_te"])
    else:
        best = max(results, key=lambda r: r["pf_te"])  # fallback: best PF
    best_tp    = best["tp_variant"]
    best_x     = best["tp_x"]
    best_ef    = best["entry_frac"]
    best_label = best["name"]

    # ── experiments 7–9 ───────────────────────────────────────────────────
    EXP_EXT = [
        (7, f"Best TP risk=0.5% ({best_label})", best_tp, best_x, best_ef, 0.5),
        (8, f"Best TP risk=1.5% ({best_label})", best_tp, best_x, best_ef, 1.5),
    ]
    EXP_ENTRY = [
        (9,  f"Entry 1/4×spike  ({best_label})", best_tp, best_x, 1/4, BASE_RISK),
        (10, f"Entry 1/3×spike  ({best_label})", best_tp, best_x, 1/3, BASE_RISK),
        (11, f"Entry 1/2×spike  ({best_label})", best_tp, best_x, 1/2, BASE_RISK),
    ]

    print(f"\n{'=' * W}")
    print("  EXPERIMENTS 7–11 (sensitivity)")
    print(f"{'=' * W}")

    for exp_no, name, tv, tx, ef, rp in EXP_EXT + EXP_ENTRY:
        trades_tr = run_backtest(df_raw, time_arr, date_idx_map, session_params,
                                 tr_dates, tv, tx, ef, INIT_CAP, rp)
        trades_te = run_backtest(df_raw, time_arr, date_idx_map, session_params,
                                 te_dates, tv, tx, ef, INIT_CAP, rp)
        print_exp(exp_no, name, trades_tr, trades_te, fd_tr, ld_tr, fd_te, ld_te)
        st = _stats(trades_te, fd_te, ld_te, INIT_CAP)
        results.append({
            "exp": exp_no, "name": name, "tp_variant": tv, "tp_x": tx,
            "entry_frac": ef, "risk_pct": rp,
            "pf_tr": _pf(trades_tr), "pf_te": st["PF"],
            "sr_te": st["SR"], "n_te": st["N"],
            "wr_te": st["WR"], "ret_te": st["ret"],
            "maxdd_te": st["maxdd"],
        })
        _save_csv(trades_tr, f"p22b_exp{exp_no}_train.csv")
        _save_csv(trades_te, f"p22b_exp{exp_no}_test.csv")

    # ── summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SUMMARY TABLE")
    print(f"{'=' * W}")
    print(f"  {'Exp':>4} | {'TP variant':22} | {'entry':>7} | "
          f"{'PF Tr':>7} | {'PF Te':>7} | {'SR Te':>7} | "
          f"{'N Te':>6} | {'WR%':>6} | {'flag':>5}")
    print(f"  {'-' * 76}")
    for r in results:
        flag = "★" if (r["pf_te"] > 1.0 and r["n_te"] >= 60) else ""
        print(f"  {r['exp']:>4} | {r['name']:22} | {r['entry_frac']:>7.4f} | "
              f"{r['pf_tr']:>7.3f} | {r['pf_te']:>7.3f} | {r['sr_te']:>+7.3f} | "
              f"{r['n_te']:>6} | {r['wr_te']:>5.1f}% | {flag:>5}")

    _save_csv(results, "p22b_diagnostic.csv")
    print(f"\n  Best config: Exp {best['exp']} — {best_label}")
    print(f"  PF test={best['pf_te']:.3f} | SR test={best['sr_te']:+.3f} | "
          f"N test={best['n_te']}")

    # ── WFO gate ─────────────────────────────────────────────────────────
    if best["sr_te"] > 0.5 and best["n_te"] >= 60:
        run_wfo(df_raw, time_arr, date_idx_map, feat_all,
                poc_per_date, avg_vol_per_date,
                best_tp, best_x, best_ef)
    else:
        print(f"\n  WFO GATE: SR test={best['sr_te']:+.3f} / N test={best['n_te']} — "
              f"threshold not met (need SR>0.5 and N≥60). WFO skipped.")

    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    main()
