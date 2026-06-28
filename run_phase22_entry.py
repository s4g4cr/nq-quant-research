#!/usr/bin/env python3
"""
Phase 22 — Entry Timing Analysis.

Bearish spike->drop (n=148): track price decay after session_high forms.
Bullish dip->rally  (n=34):  track price rally after session_low forms.

For each pattern, computes price vs extreme at +15/+30/+45/+60/+90 min,
then evaluates SHORT/LONG entry R/R at +30/+45/+60 min.
"""
import os
import sys
from datetime import datetime, timedelta, time as dt_time

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
    train_hmm,
)

RESULTS    = os.path.join(ROOT, "results")
W          = 72
TRAIN_END  = "2024-12-31"
SESS_START = dt_time(9, 30)
SESS_END   = dt_time(15, 45)
INTERVALS  = [15, 30, 45, 60, 90]
SL_BUFFER  = 5.0   # points beyond spike high / below dip low


# ── helpers ───────────────────────────────────────────────────────────────────

def _mins(t):
    return t.hour * 60 + t.minute

def _fmt(m):
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"

def _add_mins(t, n):
    dt = datetime(2000, 1, 1, t.hour, t.minute) + timedelta(minutes=n)
    return dt.time()

def _save_csv(rows, fname):
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, fname), index=False)

def _build_date_index(df):
    da = np.array(df.index.date)
    m = {}
    for i, d in enumerate(da):
        m.setdefault(d, []).append(i)
    return {d: np.array(v) for d, v in m.items()}


# ── session record builder ────────────────────────────────────────────────────

def build_records(dates, feat_clean, df_raw, date_idx_map):
    """
    For each date extract: open, session high/low, timing of each,
    order (high_first / low_first), and raw price arrays for later tracking.
    """
    feat_idx = feat_clean.set_index("date")
    high_v   = df_raw["high"].values
    low_v    = df_raw["low"].values
    close_v  = df_raw["close"].values
    time_arr = np.array(df_raw.index.time)

    records = []
    for d in dates:
        if d not in feat_idx.index or d not in date_idx_map:
            continue
        row      = feat_idx.loc[d]
        open_930 = float(row["open_930"])

        abs_idx = date_idx_map[d]
        d_times = time_arr[abs_idx]
        sel     = (d_times >= SESS_START) & (d_times <= SESS_END)
        sess_a  = abs_idx[sel]
        sess_t  = d_times[sel]
        if len(sess_a) == 0:
            continue

        h_v = high_v[sess_a]
        l_v = low_v[sess_a]
        c_v = close_v[sess_a]

        sess_high = float(h_v.max())
        sess_low  = float(l_v.min())
        hi  = int(np.argmax(h_v == sess_high))
        li  = int(np.argmax(l_v == sess_low))
        ht  = sess_t[hi]
        lt  = sess_t[li]

        if _mins(ht) < _mins(lt):
            order = "high_first"
        elif _mins(lt) < _mins(ht):
            order = "low_first"
        else:
            order = "same_bar"

        records.append({
            "date":      d,
            "open_930":  open_930,
            "sess_high": sess_high,
            "sess_low":  sess_low,
            "high_t":    ht,
            "low_t":     lt,
            "order":     order,
            "_sess_t":   sess_t,
            "_c_v":      c_v,
        })
    return records


# ── price tracking ────────────────────────────────────────────────────────────

def compute_entry_rows(records, direction):
    """
    direction 'short': extreme = session_high (spike); track fall after it.
    direction 'long':  extreme = session_low  (dip);  track rally after it.

    For each record returns a dict with prices at +15/30/45/60/90 and
    entry R/R at +30/45/60.
    """
    rows = []
    for r in records:
        ext_t  = r["high_t"]      if direction == "short" else r["low_t"]
        ext_p  = r["sess_high"]   if direction == "short" else r["sess_low"]
        target = r["sess_low"]    if direction == "short" else r["sess_high"]
        open_p = r["open_930"]
        sess_t = r["_sess_t"]
        c_v    = r["_c_v"]

        row = {
            "date":      str(r["date"]),
            "open_930":  open_p,
            "sess_high": r["sess_high"],
            "sess_low":  r["sess_low"],
            "extreme_t": str(ext_t),
            "extreme_p": ext_p,
            "target_p":  target,
            "direction": direction,
        }

        for N in INTERVALS:
            tgt_t = _add_mins(ext_t, N)
            if tgt_t > SESS_END:
                row[f"price_+{N}"] = np.nan
                continue
            # First bar strictly after ext_t AND at or past tgt_t
            mask = (sess_t > ext_t) & (sess_t >= tgt_t)
            if not mask.any():
                row[f"price_+{N}"] = np.nan
                continue
            row[f"price_+{N}"] = float(c_v[np.argmax(mask)])

        # Derived: vs extreme, vs open
        for N in INTERVALS:
            p = row.get(f"price_+{N}", np.nan)
            row[f"vs_extreme_+{N}"] = round(p - ext_p, 2) if not np.isnan(p) else np.nan
            row[f"vs_open_+{N}"]    = round(p - open_p, 2) if not np.isnan(p) else np.nan

        # Entry R/R at +30/45/60
        for N in [30, 45, 60]:
            ep = row.get(f"price_+{N}", np.nan)
            if np.isnan(ep):
                row[f"sl_dist_+{N}"] = np.nan
                row[f"rem_+{N}"]     = np.nan
                row[f"rr_+{N}"]      = np.nan
                continue
            if direction == "short":
                sl   = ext_p + SL_BUFFER
                sl_d = sl - ep          # >0 when we're below the spike (healthy entry)
                rem  = ep - target      # >0 when still above session_low
            else:
                sl   = ext_p - SL_BUFFER
                sl_d = ep - sl          # >0 when above the dip
                rem  = target - ep      # >0 when still below session_high
            rr = rem / sl_d if sl_d > 0 else np.nan
            row[f"sl_dist_+{N}"] = round(sl_d, 2)
            row[f"rem_+{N}"]     = round(rem, 2)
            row[f"rr_+{N}"]      = round(rr, 3)

        rows.append(row)
    return rows


# ── print helpers ─────────────────────────────────────────────────────────────

def print_decay_table(rows, direction):
    extr_word = "spike" if direction == "short" else "dip"
    favor     = "fallen" if direction == "short" else "rallied"
    below_lbl = "% below open" if direction == "short" else "% above open"

    print(f"\n  Price progression after {extr_word} (first occurrence):")
    print(f"  {'Interval':>10} | {'N_valid':>7} | {'Med vs extreme':>14} | "
          f"{'Med vs open':>12} | {below_lbl:>13} | {'%>50 ' + favor:>12}")
    print(f"  {'-' * 74}")

    for N in INTERVALS:
        kp = f"price_+{N}"; ke = f"vs_extreme_+{N}"; ko = f"vs_open_+{N}"
        valid = [r for r in rows if not np.isnan(r.get(kp, np.nan))]
        nv    = len(valid)
        if nv == 0:
            print(f"  {f'+{N} min':>10} | {0:>7} | {'—':>14} | {'—':>12} | {'—':>13} | {'—':>12}")
            continue
        vs_e = np.array([r[ke] for r in valid])
        vs_o = np.array([r[ko] for r in valid])
        med_e = float(np.median(vs_e)); med_o = float(np.median(vs_o))
        if direction == "short":
            pct_side  = (vs_o < 0).mean() * 100
            pct_fifty = (vs_e < -50).mean() * 100
        else:
            pct_side  = (vs_o > 0).mean() * 100
            pct_fifty = (vs_e > 50).mean() * 100
        print(f"  {f'+{N} min':>10} | {nv:>7} | {med_e:>+14.1f} pts | "
              f"{med_o:>+11.1f} pts | {pct_side:>12.1f}% | {pct_fifty:>12.1f}%")


def print_additional_stats(rows, direction):
    for N in [30, 60]:
        kp = f"price_+{N}"; ko = f"vs_open_+{N}"; ke = f"vs_extreme_+{N}"
        valid = [r for r in rows if not np.isnan(r.get(kp, np.nan))]
        nv    = len(valid)
        if nv == 0:
            continue
        vs_o = np.array([r[ko] for r in valid])
        vs_e = np.array([r[ke] for r in valid])
        if direction == "short":
            pct_open = (vs_o < 0).mean() * 100
            pct_50   = (vs_e < -50).mean() * 100
            print(f"  +{N} min: {pct_open:.1f}% below open  |  {pct_50:.1f}% fallen >50 pts from spike")
        else:
            pct_open = (vs_o > 0).mean() * 100
            pct_50   = (vs_e > 50).mean() * 100
            print(f"  +{N} min: {pct_open:.1f}% above open  |  {pct_50:.1f}% rallied >50 pts from dip")


def print_entry_table(rows, direction):
    side_word = "SHORT" if direction == "short" else "LONG"
    ref_word  = "spike" if direction == "short" else "dip"
    print(f"\n  Entry R/R — enter {side_word} at +N min, SL = {ref_word} ± {SL_BUFFER:.0f} pts:")
    print(f"  {'Entry':>10} | {'N_valid':>7} | {'Med SL dist':>12} | "
          f"{'Med remaining':>14} | {'Med R/R':>8} | {'%R/R>1':>7} | {'%R/R>2':>7}")
    print(f"  {'-' * 74}")
    for N in [30, 45, 60]:
        ks = f"sl_dist_+{N}"; kr = f"rem_+{N}"; kv = f"rr_+{N}"
        valid = [r for r in rows
                 if not np.isnan(r.get(kv, np.nan)) and r.get(ks, 0) > 0]
        nv = len(valid)
        if nv == 0:
            print(f"  {f'+{N} min':>10} | {0:>7} | {'—':>12} | {'—':>14} | "
                  f"{'—':>8} | {'—':>7} | {'—':>7}")
            continue
        sl_a  = np.array([r[ks] for r in valid])
        rem_a = np.array([r[kr] for r in valid])
        rr_a  = np.array([r[kv] for r in valid])
        p1    = (rr_a > 1.0).mean() * 100
        p2    = (rr_a > 2.0).mean() * 100
        print(f"  {f'+{N} min':>10} | {nv:>7} | {np.median(sl_a):>+11.1f} pts | "
              f"{np.median(rem_a):>+13.1f} pts | {np.median(rr_a):>8.2f}x | "
              f"{p1:>6.1f}% | {p2:>6.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 22 — ENTRY TIMING ANALYSIS")
    print("  Bearish spike->drop (n=148)  |  Bullish dip->rally (n=34)")
    print("=" * W)

    cfg    = Config()
    df_raw = load_data(cfg)
    print(f"  Data: {df_raw.index[0]} -> {df_raw.index[-1]}")

    feat_clean = add_causal_features(extract_daily_features(df_raw))
    feat_clean = (feat_clean[feat_clean["date"].astype(str) <= TRAIN_END]
                  .dropna(subset=["volume_ratio", "daily_atr"])
                  .reset_index(drop=True))

    X_tr = feat_clean[["daily_return", "volume_ratio"]].values
    model, states_tr = train_hmm(X_tr, 3, seed=42)
    lmap = label_states(states_tr, feat_clean["daily_return"].values, 3)
    inv  = {v: k for k, v in lmap.items()}

    print(f"  Training sessions: {len(feat_clean)}  |  State map: {lmap}")

    date_idx_map = _build_date_index(df_raw)

    # ── bearish spike->drop ───────────────────────────────────────────────────
    bear_dates = feat_clean["date"].values[states_tr == inv["bearish"]]
    bear_all   = build_records(bear_dates, feat_clean, df_raw, date_idx_map)
    bear_spike = [r for r in bear_all if r["order"] == "high_first"]

    print(f"\n{'=' * W}")
    print(f"  BEARISH SPIKE->DROP  (n={len(bear_spike)} of {len(bear_all)} bearish days)")
    print(f"  mean_ret=-0.415%  vol_ratio=1.273  spike at median 09:45  low at median 14:37")

    bear_rows = compute_entry_rows(bear_spike, direction="short")
    print_decay_table(bear_rows, "short")

    print(f"\n  Specific thresholds:")
    print_additional_stats(bear_rows, "short")

    print_entry_table(bear_rows, "short")

    # ── bullish dip->rally ────────────────────────────────────────────────────
    bull_dates = feat_clean["date"].values[states_tr == inv["bullish"]]
    bull_all   = build_records(bull_dates, feat_clean, df_raw, date_idx_map)
    bull_dip   = [r for r in bull_all if r["order"] == "low_first"]

    print(f"\n{'=' * W}")
    print(f"  BULLISH DIP->RALLY  (n={len(bull_dip)} of {len(bull_all)} bullish days)")
    print(f"  mean_ret=+0.222%  vol_ratio=0.489  dip at median 09:35  high at median 15:02")

    bull_rows = compute_entry_rows(bull_dip, direction="long")
    print_decay_table(bull_rows, "long")

    print(f"\n  Specific thresholds:")
    print_additional_stats(bull_rows, "long")

    print_entry_table(bull_rows, "long")

    # ── save CSV ──────────────────────────────────────────────────────────────
    all_rows = bear_rows + bull_rows
    # Drop the private arrays from output
    save_cols = [
        "date", "direction", "open_930", "sess_high", "sess_low",
        "extreme_t", "extreme_p", "target_p",
    ] + [f"price_+{N}" for N in INTERVALS] \
      + [f"vs_extreme_+{N}" for N in INTERVALS] \
      + [f"vs_open_+{N}" for N in INTERVALS] \
      + [f"sl_dist_+{N}" for N in [30, 45, 60]] \
      + [f"rem_+{N}"     for N in [30, 45, 60]] \
      + [f"rr_+{N}"      for N in [30, 45, 60]]

    clean = [{k: r.get(k, np.nan) for k in save_cols} for r in all_rows]
    _save_csv(clean, "p22_entry_timing.csv")

    print(f"\n  Saved: results/p22_entry_timing.csv  ({len(clean)} rows)")
    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    main()
