#!/usr/bin/env python3
"""
Phase 22 — Time-of-Day Analysis (bullish and bearish states).

Re-trains the Phase 22 HMM (same params: N=3, seed=42) on the training set,
then analyses session-low and session-high timing for bullish and bearish days.

Key questions:
  - What hour does the session low occur on bullish days?
  - What hour does the session high occur on bearish days?
  - How often does the "dip then rally" / "spike then drop" pattern appear?
  - What are the median magnitudes and timing of each leg?
"""
import os
import sys
from datetime import time as dt_time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.strategy.hmm_transition import (
    add_causal_features,
    extract_daily_features,
    label_states,
    predict_states,
    train_hmm,
)

RESULTS    = os.path.join(ROOT, "results")
W          = 72
TRAIN_END  = "2024-12-31"
SESS_START = dt_time(9, 30)
SESS_END   = dt_time(15, 45)


def _mins(t):
    return t.hour * 60 + t.minute

def _fmt_time(m):
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"

def _median_mins(times):
    return float(np.median([_mins(t) for t in times]))

def _save_csv(rows, fname):
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, fname), index=False)


def _build_date_index(df_1min):
    date_arr = np.array(df_1min.index.date)
    idx_map = {}
    for i, d in enumerate(date_arr):
        if d not in idx_map:
            idx_map[d] = []
        idx_map[d].append(i)
    return {d: np.array(v) for d, v in idx_map.items()}


def analyze_tod(dates, df_raw, feat_df, date_idx_map, primary_label):
    """
    For each date, extract:
      - Time of session low and session high (first occurrence of each)
      - Whether low occurs before or after high
      - Magnitudes of dip (open → low) and subsequent move (low → high)

    primary_label: 'bullish' or 'bearish' determines which extreme we focus on first.
    For bullish: primary extreme = session low (the dip we ride out before the rally)
    For bearish: primary extreme = session high (the spike we fade)
    """
    feat_lookup = feat_df.set_index("date")
    high_v  = df_raw["high"].values
    low_v   = df_raw["low"].values
    time_arr = np.array(df_raw.index.time)

    records = []

    for d in dates:
        if d not in feat_lookup.index or d not in date_idx_map:
            continue
        row      = feat_lookup.loc[d]
        open_930 = float(row["open_930"])

        abs_idx  = date_idx_map[d]
        d_times  = time_arr[abs_idx]
        sel      = (d_times >= SESS_START) & (d_times <= SESS_END)
        sess_abs = abs_idx[sel]
        sess_t   = d_times[sel]
        if len(sess_abs) == 0:
            continue

        h_vals = high_v[sess_abs]
        l_vals = low_v[sess_abs]

        sess_high = float(h_vals.max())
        sess_low  = float(l_vals.min())

        # First occurrence (time) of session high and low
        first_high_idx = int(np.argmax(h_vals == sess_high))
        first_low_idx  = int(np.argmax(l_vals == sess_low))
        high_time = sess_t[first_high_idx]
        low_time  = sess_t[first_low_idx]

        high_mins = _mins(high_time)
        low_mins  = _mins(low_time)

        if low_mins < high_mins:
            order = "low_first"
        elif high_mins < low_mins:
            order = "high_first"
        else:
            order = "same_bar"

        records.append({
            "date":        d,
            "open_930":    open_930,
            "sess_high":   sess_high,
            "sess_low":    sess_low,
            "high_time":   high_time,
            "low_time":    low_time,
            "high_mins":   high_mins,
            "low_mins":    low_mins,
            "high_hour":   high_time.hour,
            "low_hour":    low_time.hour,
            "order":       order,
            "dip_pts":     open_930 - sess_low,          # + = price went below open
            "spike_pts":   sess_high - open_930,         # + = price went above open
            "range_pts":   sess_high - sess_low,
        })

    return records


def print_hour_dist(records, field, label, n_total):
    counts = {}
    for r in records:
        h = r[field]
        counts[h] = counts.get(h, 0) + 1
    print(f"  {label}:")
    print(f"  {'Hour':>5} | {'Count':>6} | {'Pct%':>6}")
    print(f"  {'-' * 22}")
    for h in range(9, 16):
        c   = counts.get(h, 0)
        pct = c / n_total * 100
        bar = "#" * int(pct / 2)
        print(f"  {h:>5} | {c:>6} | {pct:>5.1f}%  {bar}")
    print()


def main():
    print("=" * W)
    print("  PHASE 22 — TIME-OF-DAY ANALYSIS")
    print("  Bullish and Bearish state session timing (training data)")
    print("=" * W)

    cfg    = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]} -> {df_raw.index[-1]}")

    feat_raw   = extract_daily_features(df_raw)
    feat_all   = add_causal_features(feat_raw)
    feat_tr    = feat_all[feat_all["date"].astype(str) <= TRAIN_END].copy()
    feat_clean = feat_tr.dropna(subset=["volume_ratio", "daily_atr"]).reset_index(drop=True)

    X_tr    = feat_clean[["daily_return", "volume_ratio"]].values
    model, states_tr = train_hmm(X_tr, 3, seed=42)
    lmap    = label_states(states_tr, feat_clean["daily_return"].values, 3)
    lmap_rev = {v: k for k, v in lmap.items()}

    print(f"  Training sessions: {len(feat_clean)}")
    print(f"  State map: {lmap}")

    bull_int = lmap_rev["bullish"]
    bear_int = lmap_rev["bearish"]

    bull_dates = feat_clean["date"].values[states_tr == bull_int]
    bear_dates = feat_clean["date"].values[states_tr == bear_int]

    date_idx_map = _build_date_index(df_raw)

    bull_recs = analyze_tod(bull_dates, df_raw, feat_clean, date_idx_map, "bullish")
    bear_recs = analyze_tod(bear_dates, df_raw, feat_clean, date_idx_map, "bearish")

    print(f"\n  Bullish days analyzed: {len(bull_recs)}  (state n={len(bull_dates)})")
    print(f"  Bearish days analyzed: {len(bear_recs)}  (state n={len(bear_dates)})")

    # ─────────────────────────────────────────────────────────────────────────
    # BULLISH DAYS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  BULLISH DAYS (n={len(bull_recs)})")
    print(f"  (Days where HMM assigns 'bullish' state)")
    print(f"  mean_ret=+0.222%  vol_ratio=0.489")

    print(f"\n  A) SESSION LOW HOUR DISTRIBUTION:")
    print_hour_dist(bull_recs, "low_hour", "Hour of first session low", len(bull_recs))

    print(f"  B) SESSION HIGH HOUR DISTRIBUTION:")
    print_hour_dist(bull_recs, "high_hour", "Hour of first session high", len(bull_recs))

    low_first  = [r for r in bull_recs if r["order"] == "low_first"]
    high_first = [r for r in bull_recs if r["order"] == "high_first"]
    same_bar   = [r for r in bull_recs if r["order"] == "same_bar"]
    n_b = len(bull_recs)

    print(f"  C) LOW vs HIGH TIMING:")
    print(f"     Low before high  (dip → rally):  {len(low_first):3d} days  "
          f"({len(low_first)/n_b*100:.1f}%)")
    print(f"     High before low  (rally → dip):  {len(high_first):3d} days  "
          f"({len(high_first)/n_b*100:.1f}%)")
    print(f"     Same bar:                         {len(same_bar):3d} days  "
          f"({len(same_bar)/n_b*100:.1f}%)")

    if low_first:
        n_lf = len(low_first)
        med_low_mins  = _median_mins([r["low_time"]  for r in low_first])
        med_high_mins = _median_mins([r["high_time"] for r in low_first])
        med_dip       = float(np.median([r["dip_pts"]   for r in low_first]))
        med_spike     = float(np.median([r["spike_pts"] for r in low_first]))
        med_range     = float(np.median([r["range_pts"] for r in low_first]))
        p25_dip  = float(np.percentile([r["dip_pts"]   for r in low_first], 25))
        p75_dip  = float(np.percentile([r["dip_pts"]   for r in low_first], 75))
        p25_rng  = float(np.percentile([r["range_pts"] for r in low_first], 25))
        p75_rng  = float(np.percentile([r["range_pts"] for r in low_first], 75))

        print(f"\n  D) DIP-THEN-RALLY PATTERN (low before high, n={n_lf}):")
        print(f"     Median time of session low:       {_fmt_time(med_low_mins)}")
        print(f"     Median time of session high:      {_fmt_time(med_high_mins)}")
        print(f"     Median dip magnitude (open→low):  {med_dip:.1f} pts  "
              f"[p25={p25_dip:.1f}  p75={p75_dip:.1f}]")
        print(f"     Median rally (low→high):          {med_range:.1f} pts  "
              f"[p25={p25_rng:.1f}  p75={p75_rng:.1f}]")
        print(f"     Median spike above open (open→high): {med_spike:.1f} pts")

    if high_first:
        n_hf = len(high_first)
        med_low_mins  = _median_mins([r["low_time"]  for r in high_first])
        med_high_mins = _median_mins([r["high_time"] for r in high_first])
        med_dip       = float(np.median([r["dip_pts"]   for r in high_first]))
        med_spike     = float(np.median([r["spike_pts"] for r in high_first]))
        med_range     = float(np.median([r["range_pts"] for r in high_first]))

        print(f"\n  E) RALLY-THEN-DIP PATTERN (high before low, n={n_hf}):")
        print(f"     Median time of session high:       {_fmt_time(med_high_mins)}")
        print(f"     Median time of session low:        {_fmt_time(med_low_mins)}")
        print(f"     Median spike above open (open→high): {med_spike:.1f} pts")
        print(f"     Median dip below open (open→low):    {med_dip:.1f} pts")
        print(f"     Median full range (low→high):        {med_range:.1f} pts")

    # ─────────────────────────────────────────────────────────────────────────
    # BEARISH DAYS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  BEARISH DAYS (n={len(bear_recs)})")
    print(f"  (Days where HMM assigns 'bearish' state)")
    print(f"  mean_ret=-0.415%  vol_ratio=1.273")

    print(f"\n  A) SESSION HIGH HOUR DISTRIBUTION:")
    print_hour_dist(bear_recs, "high_hour", "Hour of first session high", len(bear_recs))

    print(f"  B) SESSION LOW HOUR DISTRIBUTION:")
    print_hour_dist(bear_recs, "low_hour", "Hour of first session low", len(bear_recs))

    high_first_b = [r for r in bear_recs if r["order"] == "high_first"]
    low_first_b  = [r for r in bear_recs if r["order"] == "low_first"]
    same_bar_b   = [r for r in bear_recs if r["order"] == "same_bar"]
    n_br = len(bear_recs)

    print(f"  C) HIGH vs LOW TIMING:")
    print(f"     High before low  (spike → drop):  {len(high_first_b):3d} days  "
          f"({len(high_first_b)/n_br*100:.1f}%)")
    print(f"     Low before high  (drop → spike):  {len(low_first_b):3d} days  "
          f"({len(low_first_b)/n_br*100:.1f}%)")
    print(f"     Same bar:                          {len(same_bar_b):3d} days  "
          f"({len(same_bar_b)/n_br*100:.1f}%)")

    if high_first_b:
        n_hfb = len(high_first_b)
        med_high_mins = _median_mins([r["high_time"] for r in high_first_b])
        med_low_mins  = _median_mins([r["low_time"]  for r in high_first_b])
        med_spike     = float(np.median([r["spike_pts"] for r in high_first_b]))
        med_dip       = float(np.median([r["dip_pts"]   for r in high_first_b]))
        med_range     = float(np.median([r["range_pts"] for r in high_first_b]))
        p25_spk  = float(np.percentile([r["spike_pts"]  for r in high_first_b], 25))
        p75_spk  = float(np.percentile([r["spike_pts"]  for r in high_first_b], 75))
        p25_rng  = float(np.percentile([r["range_pts"]  for r in high_first_b], 25))
        p75_rng  = float(np.percentile([r["range_pts"]  for r in high_first_b], 75))

        print(f"\n  D) SPIKE-THEN-DROP PATTERN (high before low, n={n_hfb}):")
        print(f"     Median time of session high:       {_fmt_time(med_high_mins)}")
        print(f"     Median time of session low:        {_fmt_time(med_low_mins)}")
        print(f"     Median spike magnitude (open→high): {med_spike:.1f} pts  "
              f"[p25={p25_spk:.1f}  p75={p75_spk:.1f}]")
        print(f"     Median drop (high→low):             {med_range:.1f} pts  "
              f"[p25={p25_rng:.1f}  p75={p75_rng:.1f}]")
        print(f"     Median dip below open (open→low):   {med_dip:.1f} pts")

    if low_first_b:
        n_lfb = len(low_first_b)
        med_high_mins = _median_mins([r["high_time"] for r in low_first_b])
        med_low_mins  = _median_mins([r["low_time"]  for r in low_first_b])
        med_spike     = float(np.median([r["spike_pts"] for r in low_first_b]))
        med_dip       = float(np.median([r["dip_pts"]   for r in low_first_b]))
        med_range     = float(np.median([r["range_pts"] for r in low_first_b]))

        print(f"\n  E) DROP-THEN-SPIKE PATTERN (low before high, n={n_lfb}):")
        print(f"     Median time of session low:         {_fmt_time(med_low_mins)}")
        print(f"     Median time of session high:        {_fmt_time(med_high_mins)}")
        print(f"     Median dip below open (open→low):   {med_dip:.1f} pts")
        print(f"     Median spike above open (open→high): {med_spike:.1f} pts")
        print(f"     Median full range (low→high):        {med_range:.1f} pts")

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE CSV
    # ─────────────────────────────────────────────────────────────────────────
    all_rows = []
    for r in bull_recs:
        all_rows.append({
            "state": "bullish",
            "date":        str(r["date"]),
            "open_930":    r["open_930"],
            "sess_high":   r["sess_high"],
            "sess_low":    r["sess_low"],
            "high_time":   str(r["high_time"]),
            "low_time":    str(r["low_time"]),
            "high_hour":   r["high_hour"],
            "low_hour":    r["low_hour"],
            "order":       r["order"],
            "dip_pts":     round(r["dip_pts"], 2),
            "spike_pts":   round(r["spike_pts"], 2),
            "range_pts":   round(r["range_pts"], 2),
        })
    for r in bear_recs:
        all_rows.append({
            "state": "bearish",
            "date":        str(r["date"]),
            "open_930":    r["open_930"],
            "sess_high":   r["sess_high"],
            "sess_low":    r["sess_low"],
            "high_time":   str(r["high_time"]),
            "low_time":    str(r["low_time"]),
            "high_hour":   r["high_hour"],
            "low_hour":    r["low_hour"],
            "order":       r["order"],
            "dip_pts":     round(r["dip_pts"], 2),
            "spike_pts":   round(r["spike_pts"], 2),
            "range_pts":   round(r["range_pts"], 2),
        })

    _save_csv(all_rows, "p22_time_of_day.csv")
    print(f"\n  Saved: results/p22_time_of_day.csv  ({len(all_rows)} rows)")
    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    main()
