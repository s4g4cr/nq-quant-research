#!/usr/bin/env python3
"""
Phase 22 — Predictive HMM via Transition Matrix (Hamilton 1989).

DIAGNOSTIC ONLY — experiments are intentionally left blank pending
review of Sections 1-4 results.

Tests whether NQ daily states follow a Markov process with predictive
transition probabilities. Feature vector: (daily_return, volume_ratio).
"""
import math
import os
import sys
from datetime import time as dt_time

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.strategy.hmm_transition import (
    add_causal_features,
    chi_sq_row,
    compute_transition_matrix,
    extract_daily_features,
    label_states,
    predict_states,
    train_hmm,
)

RESULTS    = os.path.join(ROOT, "results")
W          = 80
TRAIN_END  = "2024-12-31"
TEST_START = "2025-01-02"
N_STATES   = [2, 3, 4]
PRIMARY_N  = 3
SESS_START = dt_time(9, 30)
SESS_END   = dt_time(15, 45)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and v != v):
        return "   nan"
    return f"{v:.{d}f}"

def _save_csv(rows, fname):
    os.makedirs(RESULTS, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, fname), index=False)

def _pct(arr, q):
    a = np.array(arr)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if len(a) > 0 else np.nan


# ── price behavior analysis ───────────────────────────────────────────────────

def _build_date_index(df_1min):
    """Pre-index df_1min by date for fast per-day lookup."""
    date_arr = np.array(df_1min.index.date)
    idx_map  = {}
    for i, d in enumerate(date_arr):
        if d not in idx_map:
            idx_map[d] = []
        idx_map[d].append(i)
    return {d: np.array(v) for d, v in idx_map.items()}


def analyze_price_behavior(dates, df_1min, feat_df, date_idx_map, direction, label):
    """
    Compute intraday price behavior stats (MFE, MAE, WR, time-of-extreme,
    drawdown-before-recovery) for a set of session dates.

    direction: 'long' (bullish), 'short' (bearish), 'neutral' (ranging)
    """
    feat_lookup = feat_df.set_index("date")
    high_v  = df_1min["high"].values
    low_v   = df_1min["low"].values
    time_arr = np.array(df_1min.index.time)

    mfe_pts = []; mae_pts = []; mfe_atr = []; mae_atr = []
    rets = []; extreme_hours = []; dip_below_n = 0; dip_depth_atr_v = []
    win_n = n_valid = 0

    for d in dates:
        if d not in feat_lookup.index or d not in date_idx_map:
            continue
        row      = feat_lookup.loc[d]
        open_930 = float(row["open_930"])
        cl_1545  = float(row["close_1545"])
        atr_d    = float(row.get("daily_atr", np.nan))
        if np.isnan(atr_d) or atr_d <= 0:
            continue

        abs_idx  = date_idx_map[d]
        d_times  = time_arr[abs_idx]
        sess_sel = (d_times >= SESS_START) & (d_times <= SESS_END)
        sess_abs = abs_idx[sess_sel]
        sess_t   = d_times[sess_sel]
        if len(sess_abs) == 0:
            continue

        sh = float(high_v[sess_abs].max())
        sl = float(low_v[sess_abs].min())

        if direction == "long":
            mfe = sh - open_930
            mae = max(0.0, open_930 - sl)
            win = cl_1545 > open_930
            ext_bar = int(np.argmax(high_v[sess_abs] == sh))
            extreme_hours.append(sess_t[ext_bar].hour)
        elif direction == "short":
            mfe = max(0.0, open_930 - sl)
            mae = sh - open_930
            win = cl_1545 < open_930
            ext_bar = int(np.argmax(low_v[sess_abs] == sl))
            extreme_hours.append(sess_t[ext_bar].hour)
        else:  # neutral
            mfe = max(sh - open_930, max(0.0, open_930 - sl))
            mae = min(abs(sh - open_930), abs(open_930 - sl))
            win = abs(cl_1545 - open_930) < atr_d * 0.5

        mfe_pts.append(mfe); mae_pts.append(mae)
        mfe_atr.append(mfe / atr_d); mae_atr.append(mae / atr_d)
        rets.append((cl_1545 - open_930) / open_930 * 100)
        if win:
            win_n += 1
        n_valid += 1

        # Drawdown before recovery
        if direction == "long":
            if sl < open_930:
                dip_below_n += 1
                if cl_1545 > open_930:
                    dip_depth_atr_v.append((open_930 - sl) / atr_d)
        elif direction == "short":
            if sh > open_930:
                dip_below_n += 1
                if cl_1545 < open_930:
                    dip_depth_atr_v.append((sh - open_930) / atr_d)

    if n_valid == 0:
        return None

    ma = np.array(mfe_atr); aa = np.array(mae_atr)
    mp = np.array(mfe_pts); ap = np.array(mae_pts)
    rr = np.array(rets)

    def q(arr, p): return round(float(np.percentile(arr, p)), 3)

    result = {
        "label":              label,
        "n":                  n_valid,
        "wr_pct":             round(win_n / n_valid * 100, 1),
        "ret_mean_pct":       round(float(rr.mean()), 4),
        "ret_std_pct":        round(float(rr.std(ddof=1)), 4),
        "ret_p10":            q(rr, 10), "ret_p25": q(rr, 25),
        "ret_p50":            q(rr, 50), "ret_p75": q(rr, 75),
        "ret_p90":            q(rr, 90),
        "mfe_p25_pts":        q(mp, 25), "mfe_p50_pts": q(mp, 50),
        "mfe_p75_pts":        q(mp, 75), "mfe_p90_pts": q(mp, 90),
        "mfe_p25_atr":        q(ma, 25), "mfe_p50_atr": q(ma, 50),
        "mfe_p75_atr":        q(ma, 75), "mfe_p90_atr": q(ma, 90),
        "mae_p25_pts":        q(ap, 25), "mae_p50_pts": q(ap, 50),
        "mae_p75_pts":        q(ap, 75), "mae_p90_pts": q(ap, 90),
        "mae_p25_atr":        q(aa, 25), "mae_p50_atr": q(aa, 50),
        "mae_p75_atr":        q(aa, 75), "mae_p90_atr": q(aa, 90),
        "mfe_p50_over_mae_p50": (
            round(float(np.median(mp)) / float(np.median(ap)), 3)
            if float(np.median(ap)) > 0 else float("inf")
        ),
        "dip_pct":            round(dip_below_n / n_valid * 100, 1),
        "dip_depth_p50_atr":  round(float(np.median(dip_depth_atr_v)), 3) if dip_depth_atr_v else np.nan,
        "dip_depth_p90_atr":  round(float(np.percentile(dip_depth_atr_v, 90)), 3) if dip_depth_atr_v else np.nan,
    }

    # Hour distribution of session extreme
    if extreme_hours:
        for h in range(9, 16):
            c = extreme_hours.count(h)
            result[f"extreme_h{h:02d}_pct"] = round(c / len(extreme_hours) * 100, 1)

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 22 — PREDICTIVE HMM VIA TRANSITION MATRIX (Hamilton 1989)")
    print("  DIAGNOSTIC ONLY")
    print("=" * W)

    cfg    = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]} -> {df_raw.index[-1]}  ({len(df_raw):,} bars)")

    print("  Extracting daily features...")
    feat_raw = extract_daily_features(df_raw)
    feat_all = add_causal_features(feat_raw)
    print(f"  Daily sessions: {len(feat_all)}")

    feat_tr = feat_all[feat_all["date"].astype(str) <= TRAIN_END].copy()
    feat_te = feat_all[feat_all["date"].astype(str) >= TEST_START].copy()

    # Drop rows with NaN features (first ~20 sessions)
    feat_tr_clean = feat_tr.dropna(subset=["volume_ratio", "daily_atr"]).reset_index(drop=True)
    feat_te_clean = feat_te.dropna(subset=["volume_ratio", "daily_atr"]).reset_index(drop=True)

    X_tr = feat_tr_clean[["daily_return", "volume_ratio"]].values
    X_te = feat_te_clean[["daily_return", "volume_ratio"]].values

    print(f"  Train sessions (after NaN drop): {len(feat_tr_clean)}")
    print(f"  Test sessions:  {len(feat_te_clean)}")

    # Pre-index 1-min data by date
    date_idx_map = _build_date_index(df_raw)

    # ─────────────────────────────────────────────────────────────────────────
    # N-STATE COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  N-STATE COMPARISON — State separation (F-statistic)")
    print(f"  {'N':>3} | {'F-stat':>8} | {'p-value':>8} | Description")
    print(f"  {'-' * 55}")

    models = {}
    states_tr_all = {}
    for N in N_STATES:
        m, st = train_hmm(X_tr, N)
        models[N] = m; states_tr_all[N] = st
        groups = [feat_tr_clean["daily_return"].values[st == i] for i in range(N)]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
        else:
            F, p = np.nan, np.nan
        desc = "BEST" if N == PRIMARY_N else ""
        print(f"  {N:>3} | {_fmt(F, 2):>8} | {_fmt(p, 4):>8} | N={N} states  {desc}")

    print(f"\n  Primary analysis uses N={PRIMARY_N}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIMARY MODEL (N=3)
    # ─────────────────────────────────────────────────────────────────────────
    model     = models[PRIMARY_N]
    states_tr = states_tr_all[PRIMARY_N]
    lmap      = label_states(states_tr, feat_tr_clean["daily_return"].values, PRIMARY_N)
    lmap_rev  = {v: k for k, v in lmap.items()}  # label → int

    # Apply to test set
    states_te = predict_states(model, X_te)

    state_labels = [lmap[i] for i in range(PRIMARY_N)]
    T_mat, T_counts = compute_transition_matrix(states_tr, PRIMARY_N)

    print(f"\n  State labels (N={PRIMARY_N}):")
    for s_int, s_lbl in sorted(lmap.items()):
        mask = states_tr == s_int
        n_s  = mask.sum()
        mr   = feat_tr_clean["daily_return"].values[mask].mean() * 100
        mv   = feat_tr_clean["volume_ratio"].values[mask].mean()
        print(f"    State {s_int} = {s_lbl:<14}  n={n_s:4d}  "
              f"mean_ret={mr:+.3f}%  mean_vol_ratio={mv:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — TRANSITION MATRIX
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SECTION 1 — TRANSITION MATRIX (N=3, empirical, training data)")

    ordered = sorted(lmap.keys(), key=lambda k: lmap[k])  # bearish→ranging→bullish
    col_labels = [lmap[s] for s in ordered]

    header = f"  {'State':<14} | " + " | ".join(f"{'P(' + lbl + ')':>12}" for lbl in col_labels)
    header += " | {'Strongest':>12} | {'chi2':>7} | {'p-value':>7}"
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")

    sig_rows = []
    tm_rows  = []
    for s_int in ordered:
        s_lbl   = lmap[s_int]
        row_T   = T_mat[s_int][ordered]  # reorder columns by label order
        row_cnt = T_counts[s_int][ordered]
        chi2, p = chi_sq_row(T_counts[s_int])  # use raw counts (unordered is fine)
        strongest = col_labels[int(np.argmax(row_T))]
        star      = " *" if (not np.isnan(p) and p < 0.10) else ""
        if not np.isnan(p) and p < 0.10:
            sig_rows.append(s_lbl)

        probs = " | ".join(f"{v:>12.4f}" for v in row_T)
        print(f"  {s_lbl:<14} | {probs} | {strongest:>12} | "
              f"{_fmt(chi2, 2):>7} | {_fmt(p, 4):>7}{star}")

        for j, s2_int in enumerate(ordered):
            tm_rows.append({
                "from_state": s_lbl, "to_state": col_labels[j],
                "probability": round(row_T[j], 4),
                "count": int(row_cnt[j]),
                "chi2": round(chi2, 4), "p_value": round(p, 4),
            })

    print(f"\n  * p < 0.10  —  {len(sig_rows)} row(s) significant: "
          f"{', '.join(sig_rows) if sig_rows else 'none'}")
    _save_csv(tm_rows, "p22_transition_matrix.csv")

    # GATE 1
    if not sig_rows:
        print("\n  STOP: All transition rows p > 0.10.")
        print("  Transitions are statistically indistinguishable from random.")
        print("  Markov predictive signal does NOT exist in NQ at daily level.")
        _finalize(tm_rows)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — DIRECTIONAL ACCURACY
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SECTION 2 — TRANSITION SIGNAL DIRECTIONAL ACCURACY (training data)")
    print(f"  Predicted state = argmax(T[state_yesterday])")

    n_states_primary = PRIMARY_N
    baseline = 1.0 / n_states_primary * 100

    pred_states = []; actual_states = []
    for t in range(1, len(states_tr)):
        prev_s = states_tr[t - 1]
        pred_s = int(np.argmax(T_mat[prev_s]))
        pred_states.append(pred_s)
        actual_states.append(states_tr[t])

    pred_arr   = np.array(pred_states)
    actual_arr = np.array(actual_states)

    sec2_rows = []
    print(f"\n  {'Predicted':<14} | {'N_pred':>7} | {'N_correct':>9} | "
          f"{'Accuracy%':>10} | {'Baseline%':>10}")
    print(f"  {'-' * 58}")

    for s_int in ordered:
        s_lbl  = lmap[s_int]
        mask_p = pred_arr == s_int
        if mask_p.sum() == 0:
            continue
        n_pred    = int(mask_p.sum())
        n_correct = int((actual_arr[mask_p] == s_int).sum())
        acc       = n_correct / n_pred * 100
        star      = " *" if acc > baseline + 5 else ""
        print(f"  {s_lbl:<14} | {n_pred:>7} | {n_correct:>9} | "
              f"{acc:>9.1f}% | {baseline:>9.1f}%{star}")
        sec2_rows.append({"predicted": s_lbl, "n_pred": n_pred,
                          "n_correct": n_correct, "accuracy": round(acc, 1),
                          "baseline": round(baseline, 1)})

    overall_n       = len(pred_arr)
    overall_correct = int((pred_arr == actual_arr).sum())
    overall_acc     = overall_correct / overall_n * 100
    print(f"  {'Overall':<14} | {overall_n:>7} | {overall_correct:>9} | "
          f"{overall_acc:>9.1f}% | {baseline:>9.1f}%")
    sec2_rows.append({"predicted": "overall", "n_pred": overall_n,
                      "n_correct": overall_correct, "accuracy": round(overall_acc, 1),
                      "baseline": round(baseline, 1)})

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — PRICE BEHAVIOR PER STATE
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SECTION 3 — PRICE BEHAVIOR PER STATE (training sessions)")

    behavior_rows = []
    behavior_by_label = {}

    for s_int in ordered:
        s_lbl  = lmap[s_int]
        mask   = states_tr == s_int
        dates_s = feat_tr_clean["date"].values[mask]
        direction = "long" if "bull" in s_lbl else ("short" if "bear" in s_lbl else "neutral")

        beh = analyze_price_behavior(
            dates_s, df_raw, feat_tr_clean, date_idx_map, direction, s_lbl
        )
        if beh is None:
            continue
        behavior_by_label[s_lbl] = beh
        behavior_rows.append(beh)

        print(f"\n  [{s_lbl.upper()}]  n={beh['n']}  WR={beh['wr_pct']}%  "
              f"mean_ret={beh['ret_mean_pct']:+.3f}%  std={beh['ret_std_pct']:.3f}%")

        print(f"  Return distribution (%):")
        print(f"    p10={beh['ret_p10']:+.3f}  p25={beh['ret_p25']:+.3f}  "
              f"p50={beh['ret_p50']:+.3f}  p75={beh['ret_p75']:+.3f}  p90={beh['ret_p90']:+.3f}")

        print(f"  MFE — p25={beh['mfe_p25_pts']:.1f}pts ({beh['mfe_p25_atr']:.3f}×ATR)  "
              f"p50={beh['mfe_p50_pts']:.1f}pts ({beh['mfe_p50_atr']:.3f}×ATR)  "
              f"p75={beh['mfe_p75_pts']:.1f}pts ({beh['mfe_p75_atr']:.3f}×ATR)  "
              f"p90={beh['mfe_p90_pts']:.1f}pts ({beh['mfe_p90_atr']:.3f}×ATR)")
        print(f"  MAE — p25={beh['mae_p25_pts']:.1f}pts ({beh['mae_p25_atr']:.3f}×ATR)  "
              f"p50={beh['mae_p50_pts']:.1f}pts ({beh['mae_p50_atr']:.3f}×ATR)  "
              f"p75={beh['mae_p75_pts']:.1f}pts ({beh['mae_p75_atr']:.3f}×ATR)  "
              f"p90={beh['mae_p90_pts']:.1f}pts ({beh['mae_p90_atr']:.3f}×ATR)")
        print(f"  MFE p50 / MAE p50 = {beh['mfe_p50_over_mae_p50']:.3f}x")

        # Time-of-extreme distribution
        hours = [h for h in range(9, 16) if f"extreme_h{h:02d}_pct" in beh]
        if hours:
            label_str = "high" if direction == "long" else ("low" if direction == "short" else "extreme")
            parts = [f"h{h}: {beh[f'extreme_h{h:02d}_pct']:>4.1f}%" for h in hours]
            print(f"  Time of session {label_str}: {' | '.join(parts)}")

        dip_label = "below open" if direction == "long" else "above open"
        print(f"  Price goes {dip_label} before close: {beh['dip_pct']:.1f}%  "
              f"depth p50={_fmt(beh['dip_depth_p50_atr'])}×ATR  "
              f"p90={_fmt(beh['dip_depth_p90_atr'])}×ATR")

    _save_csv(behavior_rows, "p22_price_behavior.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — COMBINED SIGNAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  SECTION 4 — TRANSITION SIGNAL + PRICE BEHAVIOR COMBINED")
    print(f"  On days where transition signal predicts a specific state:")

    gate2_passes = True
    diag_rows    = []
    mfe_mae_ratios = []

    for s_int in ordered:
        s_lbl     = lmap[s_int]
        mask_pred = pred_arr == s_int  # days where we predict s_lbl
        if mask_pred.sum() == 0:
            continue

        # Subset of feat_tr_clean days that were predicted as s_lbl
        # (these are training days T=1..end, indexed by pred/actual arrays)
        pred_day_indices = np.where(mask_pred)[0] + 1  # +1 because pred starts at t=1
        dates_pred = feat_tr_clean["date"].values[pred_day_indices]
        pred_rets  = feat_tr_clean["daily_return"].values[pred_day_indices]

        n_pred  = int(mask_pred.sum())
        n_corr  = int((actual_arr[mask_pred] == s_int).sum())
        acc     = n_corr / n_pred * 100
        mean_r  = float(pred_rets.mean() * 100)
        pos_r   = float((pred_rets > 0).mean() * 100)

        direction = "long" if "bull" in s_lbl else ("short" if "bear" in s_lbl else "neutral")
        beh = analyze_price_behavior(
            dates_pred, df_raw, feat_tr_clean, date_idx_map, direction, f"signal_{s_lbl}"
        )

        print(f"\n  Signal = '{s_lbl}'  (n_days={n_pred})")
        print(f"    State accuracy:  {acc:.1f}%  (baseline {baseline:.1f}%)")
        print(f"    Mean daily ret:  {mean_r:+.4f}%")
        print(f"    % positive ret:  {pos_r:.1f}%")

        if beh:
            mfe_med = beh["mfe_p50_pts"]; mae_med = beh["mae_p50_pts"]
            ratio   = beh["mfe_p50_over_mae_p50"]
            print(f"    MFE p50: {mfe_med:.1f}pts  MAE p50: {mae_med:.1f}pts  "
                  f"ratio: {_fmt(ratio, 3)}x")
            if "bull" in s_lbl or "bear" in s_lbl:
                mfe_mae_ratios.append(ratio)
        else:
            ratio = np.nan

        diag_rows.append({
            "signal": s_lbl, "n_days": n_pred, "state_accuracy": round(acc, 1),
            "mean_ret_pct": round(mean_r, 4), "pos_ret_pct": round(pos_r, 1),
            "mfe_p50_pts": beh["mfe_p50_pts"] if beh else np.nan,
            "mae_p50_pts": beh["mae_p50_pts"] if beh else np.nan,
            "mfe_mae_ratio": round(ratio, 3) if beh and not np.isnan(ratio) else np.nan,
        })

        if beh and ("bull" in s_lbl or "bear" in s_lbl):
            if not np.isnan(ratio) and ratio < 1.5:
                gate2_passes = False

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("  DIAGNOSTIC VERDICT")

    v_sig_exists   = len(sig_rows) > 0
    v_dir_any      = overall_acc > baseline
    v_mfe_mae_ok   = gate2_passes
    v_proceed      = v_sig_exists and v_mfe_mae_ok

    print(f"\n  Signal exists (Section 1):           {'YES' if v_sig_exists else 'NO'}  "
          f"  ({len(sig_rows)}/{PRIMARY_N} rows p<0.10)")
    print(f"  Directional accuracy > baseline:     {'YES' if v_dir_any else 'NO'}  "
          f"  ({overall_acc:.1f}% vs {baseline:.1f}% baseline)")
    mfe_mae_str = f"  (min ratio {min(mfe_mae_ratios):.3f}x)" if mfe_mae_ratios else ""
    print(f"  MFE >= 1.5 x MAE (Section 4 gate):  {'YES' if v_mfe_mae_ok else 'NO'}{mfe_mae_str}")
    print(f"\n  PROCEED TO EXPERIMENTS:  {'YES' if v_proceed else 'NO'}")
    if not v_proceed:
        if not v_sig_exists:
            print("  -> Transitions are random. No edge to exploit.")
        elif not v_mfe_mae_ok:
            print("  -> MFE < 1.5 x MAE. Risk/reward incompatible with viable strategy.")

    # Save diagnostics
    all_diag = sec2_rows + diag_rows
    _save_csv(all_diag, "p22_diagnostic.csv")
    print(f"\n  Saved: results/p22_transition_matrix.csv")
    print(f"         results/p22_price_behavior.csv")
    print(f"         results/p22_diagnostic.csv")
    print(f"\n{'=' * W}\n")


def _finalize(tm_rows):
    print(f"\n  Saved: results/p22_transition_matrix.csv")
    _save_csv(tm_rows, "p22_transition_matrix.csv")
    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    main()
