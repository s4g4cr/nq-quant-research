#!/usr/bin/env python3
"""
Phase 22 — Spike/Dip Magnitude vs Volatility Analysis.

Bearish spike->drop (n=148): spike_magnitude = session_high - open_930
Bullish dip->rally  (n=34):  dip_magnitude   = open_930 - session_low

Sections per subset:
  1. Pearson correlation with prev_range, prev_atr, realized_vol
  2. Normalized spike distribution (spike_vs_atr, spike_vs_range)
  3. OLS regression: spike_magnitude ~ prev_atr  and  ~ prev_range
  4. Quintile analysis: prev_atr quintiles
  5. Year-by-year breakdown

Final: three adaptive SL formula candidates with coverage stats.
"""
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
from orb_system.strategy.hmm_transition import (
    add_causal_features, extract_daily_features, label_states, train_hmm,
)

RESULTS    = os.path.join(ROOT, "results")
TRAIN_END  = "2024-12-31"
W          = 72
SESS_START = dt_time(9, 30)
SESS_END   = dt_time(15, 45)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_csv(rows, fname):
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, fname), index=False)


def _build_date_index(df):
    da = np.array(df.index.date)
    m = {}
    for i, d in enumerate(da):
        m.setdefault(d, []).append(i)
    return {d: np.array(v) for d, v in m.items()}


def _corr(x, y, label_x, label_y):
    mask = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p, mask.sum()


def _reg(x, y, label):
    mask = np.isfinite(x) & np.isfinite(y)
    sl, ic, r, p, se = stats.linregress(x[mask], y[mask])
    return sl, ic, r**2, p, mask.sum()


# ── order map: determine high_first / low_first from 1-min bars ───────────────

def build_order_map(df_raw, date_idx_map):
    high_v = df_raw["high"].values
    low_v  = df_raw["low"].values
    tv     = np.array(df_raw.index.time)
    omap   = {}
    for d, abs_idx in date_idx_map.items():
        dt = tv[abs_idx]
        sel = (dt >= SESS_START) & (dt <= SESS_END)
        if not sel.any():
            continue
        sa = abs_idx[sel]; st = dt[sel]
        h = high_v[sa]; l = low_v[sa]
        sh = h.max(); sl_ = l.min()
        hi = int(np.argmax(h == sh))
        li = int(np.argmax(l == sl_))
        hm = st[hi].hour * 60 + st[hi].minute
        lm = st[li].hour * 60 + st[li].minute
        omap[d] = "high_first" if hm < lm else ("low_first" if lm < hm else "same")
    return omap


# ── build analysis DataFrame for one subset ───────────────────────────────────

def build_df(dates, feat_by_date, order_map, direction, target_order):
    rows = []
    sorted_all = sorted(feat_by_date.index.tolist())
    pos_map = {d: i for i, d in enumerate(sorted_all)}

    for d in dates:
        if d not in feat_by_date.index:
            continue
        if order_map.get(d) != target_order:
            continue
        pos = pos_map.get(d, 0)
        if pos == 0:
            continue

        fr = feat_by_date.loc[d]
        prev_atr   = float(fr["daily_atr"])          # 20-day causal ATR
        prev_range = float(fr["prev_range"])          # immediate prior session range (shifted)
        open_930   = float(fr["open_930"])
        sess_high  = float(fr["session_high"])
        sess_low   = float(fr["session_low"])
        realized_vol = sess_high - sess_low

        if direction == "short":
            spike_mag = sess_high - open_930   # above open
        else:
            spike_mag = open_930 - sess_low    # below open

        if prev_atr <= 0 or prev_range <= 0:
            continue

        rows.append({
            "date":         d,
            "year":         d.year,
            "direction":    direction,
            "open_930":     open_930,
            "sess_high":    sess_high,
            "sess_low":     sess_low,
            "spike_mag":    round(spike_mag, 2),
            "prev_atr":     round(prev_atr, 2),
            "prev_range":   round(prev_range, 2),
            "realized_vol": round(realized_vol, 2),
            "spike_vs_atr":   round(spike_mag / prev_atr, 4),
            "spike_vs_range": round(spike_mag / prev_range, 4),
        })

    df = pd.DataFrame(rows).dropna(subset=["spike_mag", "prev_atr", "prev_range"])
    return df


# ── section printers ──────────────────────────────────────────────────────────

def section1_corr(df, label):
    print(f"\n  [1] CORRELATION ANALYSIS — {label}")
    print(f"  {'Variable pair':40}  {'r':>8}  {'p-value':>10}  {'N':>6}")
    print(f"  {'-' * 68}")
    pairs = [
        ("spike_mag", "prev_range",   "spike_magnitude vs prev_session_range"),
        ("spike_mag", "prev_atr",     "spike_magnitude vs prev_session_atr"),
        ("spike_mag", "realized_vol", "spike_magnitude vs session_realized_vol"),
    ]
    for xk, yk, lbl in pairs:
        x = df[xk].values; y = df[yk].values
        r, p, n = _corr(x, y, xk, yk)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {lbl:40}  {r:>+8.3f}  {p:>10.4f}  {n:>6}  {sig}")


def section2_norm(df, label):
    sva = df["spike_vs_atr"].values
    svr = df["spike_vs_range"].values
    sm  = df["spike_mag"].values
    print(f"\n  [2] NORMALIZED SPIKE DISTRIBUTION — {label}")
    print(f"  {'Metric':30}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p90':>8}  {'std':>8}")
    print(f"  {'-' * 74}")
    for arr, name in [(sm, "spike_magnitude (raw pts)"),
                      (sva, "spike_vs_atr"),
                      (svr, "spike_vs_range")]:
        p25, p50, p75, p90 = np.percentile(arr, [25, 50, 75, 90])
        print(f"  {name:30}  {p25:>8.3f}  {p50:>8.3f}  {p75:>8.3f}  {p90:>8.3f}  {arr.std():>8.3f}")
    print(f"\n  Stability check (lower = more consistent when normalized):")
    print(f"    std(spike_magnitude):  {sm.std():>8.1f} pts")
    cv_raw = sm.std() / sm.mean() * 100
    cv_atr = sva.std() / sva.mean() * 100
    print(f"    CV(spike_magnitude):   {cv_raw:>7.1f}%   (raw)")
    print(f"    CV(spike_vs_atr):      {cv_atr:>7.1f}%   (normalized by ATR)")
    if cv_atr < cv_raw:
        print(f"    → ATR normalization REDUCES dispersion by {cv_raw - cv_atr:.1f} pp")
    else:
        print(f"    → ATR normalization does NOT reduce dispersion (+{cv_atr - cv_raw:.1f} pp)")


def section3_reg(df, label):
    sm = df["spike_mag"].values
    pa = df["prev_atr"].values
    pr = df["prev_range"].values
    print(f"\n  [3] OLS REGRESSION — {label}")
    print(f"  {'Model':35}  {'slope':>8}  {'intercept':>10}  {'R²':>7}  {'p-value':>10}")
    print(f"  {'-' * 74}")
    for x, xname in [(pa, "spike_mag ~ prev_atr"), (pr, "spike_mag ~ prev_range")]:
        sl, ic, r2, p, n = _reg(x, sm, xname)
        print(f"  {xname:35}  {sl:>8.4f}  {ic:>10.2f}  {r2:>7.3f}  {p:>10.4f}")
    sl_a, ic_a, r2_a, p_a, _ = _reg(pa, sm, "atr")
    sl_r, ic_r, r2_r, p_r, _ = _reg(pr, sm, "range")
    return sl_a, ic_a, sl_r, ic_r


def section4_quintile(df, label):
    print(f"\n  [4] QUINTILE ANALYSIS (by prev_session_atr) — {label}")
    df2 = df.copy()
    df2["quintile"] = pd.qcut(df2["prev_atr"], q=5, labels=[1, 2, 3, 4, 5])
    grp = df2.groupby("quintile", observed=True)
    print(f"  {'Q':>3}  {'N':>5}  {'Mean prev_atr':>14}  {'Mean spike_mag':>15}  "
          f"{'spike/atr ratio':>16}  {'CV spike':>10}")
    print(f"  {'-' * 68}")
    ratios = []
    for q, g in grp:
        ma = g["prev_atr"].mean()
        ms = g["spike_mag"].mean()
        ratio = ms / ma if ma > 0 else np.nan
        cv    = g["spike_mag"].std() / ms * 100 if ms > 0 else np.nan
        ratios.append(ratio)
        print(f"  {q:>3}  {len(g):>5}  {ma:>14.1f}  {ms:>15.1f}  {ratio:>16.3f}  {cv:>9.1f}%")
    ratios = np.array(ratios)
    print(f"\n  Ratio range: {ratios.min():.3f} – {ratios.max():.3f}  "
          f"| Spread: {ratios.max()-ratios.min():.3f}  "
          f"| std: {ratios.std():.3f}")
    if ratios.std() < 0.05:
        print("  → Ratio stable across quintiles: ATR is a good normalizer")
    elif ratios.std() < 0.10:
        print("  → Ratio moderately stable: ATR is a decent normalizer")
    else:
        print("  → Ratio varies across quintiles: relationship may be non-linear")


def section5_year(df, label):
    print(f"\n  [5] YEAR-BY-YEAR BREAKDOWN — {label}")
    print(f"  {'Year':>6}  {'N':>5}  {'Mean spike_mag':>15}  {'Mean prev_atr':>14}  {'ratio':>8}")
    print(f"  {'-' * 52}")
    for yr, g in df.groupby("year"):
        ms = g["spike_mag"].mean(); ma = g["prev_atr"].mean()
        ratio = ms / ma if ma > 0 else np.nan
        print(f"  {yr:>6}  {len(g):>5}  {ms:>15.1f}  {ma:>14.1f}  {ratio:>8.3f}")


def section_sl(df, label, direction, sl_a, ic_a, sl_r, ic_r):
    sm  = df["spike_mag"].values
    pa  = df["prev_atr"].values
    pr  = df["prev_range"].values
    sva = df["spike_vs_atr"].values
    n   = len(sm)

    p75_ratio = float(np.percentile(sva, 75))
    p85_ratio = float(np.percentile(sva, 85))
    p90_ratio = float(np.percentile(sva, 90))

    # Option A: expected_spike = slope_atr × prev_atr + intercept_atr; SL = expected + 1×ATR
    sl_A = (sl_a * pa + ic_a) + 1.0 * pa
    cov_A = (sm < sl_A).mean() * 100

    # Option B: expected_spike = slope_range × prev_range + intercept_range; SL = expected + 0.5×range
    sl_B = (sl_r * pr + ic_r) + 0.5 * pr
    cov_B = (sm < sl_B).mean() * 100

    # Option C: SL = p75_ratio × prev_atr
    sl_C75 = p75_ratio * pa
    sl_C85 = p85_ratio * pa
    sl_C90 = p90_ratio * pa
    cov_C75 = (sm < sl_C75).mean() * 100
    cov_C85 = (sm < sl_C85).mean() * 100
    cov_C90 = (sm < sl_C90).mean() * 100

    sign = "above open" if direction == "short" else "below open"
    print(f"\n  [SL] ADAPTIVE SL FORMULA — {label}")
    print(f"  SL = level {sign} that covers the spike; target coverage > 85%")
    print()
    med_A = np.median(sl_A); med_B = np.median(sl_B)
    med_C75 = np.median(sl_C75); med_C85 = np.median(sl_C85); med_C90 = np.median(sl_C90)

    print(f"  Option A (ATR regression + 1×ATR buffer):")
    print(f"    Formula:  SL = ({sl_a:.4f} × prev_atr + {ic_a:.2f}) + 1.0 × prev_atr")
    print(f"            = {sl_a+1:.4f} × prev_atr + {ic_a:.2f}")
    print(f"    Median SL: {med_A:.1f} pts  |  Coverage: {cov_A:.1f}%  "
          f"{'✓' if cov_A >= 85 else '✗'} (target 85%)")

    print(f"\n  Option B (range regression + 0.5×range buffer):")
    print(f"    Formula:  SL = ({sl_r:.4f} × prev_range + {ic_r:.2f}) + 0.5 × prev_range")
    print(f"            = {sl_r+0.5:.4f} × prev_range + {ic_r:.2f}")
    print(f"    Median SL: {med_B:.1f} pts  |  Coverage: {cov_B:.1f}%  "
          f"{'✓' if cov_B >= 85 else '✗'} (target 85%)")

    print(f"\n  Option C (fixed ATR multiple):")
    print(f"    X = p75  of spike/atr = {p75_ratio:.3f}  →  SL = {p75_ratio:.3f} × prev_atr  "
          f"|  Median: {med_C75:.1f} pts  |  Coverage: {cov_C75:.1f}%")
    print(f"    X = p85  of spike/atr = {p85_ratio:.3f}  →  SL = {p85_ratio:.3f} × prev_atr  "
          f"|  Median: {med_C85:.1f} pts  |  Coverage: {cov_C85:.1f}%")
    print(f"    X = p90  of spike/atr = {p90_ratio:.3f}  →  SL = {p90_ratio:.3f} × prev_atr  "
          f"|  Median: {med_C90:.1f} pts  |  Coverage: {cov_C90:.1f}%")

    best = "A" if cov_A >= 85 else ("C(p85)" if cov_C85 >= 85 else "C(p90)")
    print(f"\n  → Recommended: Option {best}")

    return {
        f"{label}_optA_formula": f"{sl_a+1:.4f}*atr + {ic_a:.2f}",
        f"{label}_optA_coverage": round(cov_A, 1),
        f"{label}_optB_formula": f"{sl_r+0.5:.4f}*range + {ic_r:.2f}",
        f"{label}_optB_coverage": round(cov_B, 1),
        f"{label}_optC_p75_X": round(p75_ratio, 3),
        f"{label}_optC_p75_cov": round(cov_C75, 1),
        f"{label}_optC_p85_X": round(p85_ratio, 3),
        f"{label}_optC_p85_cov": round(cov_C85, 1),
        f"{label}_optC_p90_X": round(p90_ratio, 3),
        f"{label}_optC_p90_cov": round(cov_C90, 1),
    }


def run_analysis(df, label, direction):
    print(f"\n{'=' * W}")
    print(f"  {label}  (n={len(df)})")
    print(f"{'=' * W}")
    section1_corr(df, label)
    section2_norm(df, label)
    sl_a, ic_a, sl_r, ic_r = section3_reg(df, label)
    section4_quintile(df, label)
    section5_year(df, label)
    sl_meta = section_sl(df, label, direction, sl_a, ic_a, sl_r, ic_r)
    return sl_meta


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 22 — SPIKE/DIP MAGNITUDE vs VOLATILITY")
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

    # Add prev_range (prior session's range) via shift — causal since it's yesterday's range
    feat_sorted = feat_clean.sort_values("date").copy()
    feat_sorted["prev_range"] = (
        feat_sorted["session_high"] - feat_sorted["session_low"]
    ).shift(1)
    feat_by_date = feat_sorted.set_index("date")

    date_idx_map = _build_date_index(df_raw)
    order_map    = build_order_map(df_raw, date_idx_map)

    bear_dates = feat_clean["date"].values[states_tr == inv["bearish"]]
    bull_dates = feat_clean["date"].values[states_tr == inv["bullish"]]

    df_bear = build_df(bear_dates, feat_by_date, order_map, "short", "high_first")
    df_bull = build_df(bull_dates, feat_by_date, order_map, "long",  "low_first")

    print(f"  Bearish spike->drop subset: {len(df_bear)} days")
    print(f"  Bullish dip->rally  subset: {len(df_bull)} days")

    sl_bear = run_analysis(df_bear, "BEARISH SPIKE->DROP", "short")
    sl_bull = run_analysis(df_bull, "BULLISH DIP->RALLY",  "long")

    # ── save CSV ──────────────────────────────────────────────────────────────
    df_bear["subset"] = "bearish_spike_drop"
    df_bull["subset"] = "bullish_dip_rally"
    all_rows = pd.concat([df_bear, df_bull], ignore_index=True)
    save_cols = [
        "subset", "date", "year", "direction",
        "open_930", "sess_high", "sess_low",
        "spike_mag", "prev_atr", "prev_range", "realized_vol",
        "spike_vs_atr", "spike_vs_range",
    ]
    all_rows[save_cols].to_csv(
        os.path.join(RESULTS, "p22_spike_volatility.csv"), index=False
    )
    print(f"\n  Saved: results/p22_spike_volatility.csv  ({len(all_rows)} rows)")
    print(f"\n{'=' * W}\n")


if __name__ == "__main__":
    main()
