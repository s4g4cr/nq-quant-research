"""
Phase 9 daily features for HMM regime detection.

Changes from features.py (Phase 5-8):
  Removed: or_range_normalized (was causing HMM mislabeling in Phase 8 --
           low realized_vol days had OR/ATR = 26 and strong trends, not ranging)
  Added:   intraday_direction = sign(close_10:30 - open_09:30)
           Captures direction of the first 60 minutes of trading.
  Added:   or_range_atr = OR_range / prev_day_ATR
           Used as a HARD routing threshold (not fed into HMM).

HMM feature columns (4):
  realized_vol, overnight_gap, trend_strength, intraday_direction

Routing column (not z-scored, not used in HMM):
  or_range_atr
"""

import numpy as np
import pandas as pd
from datetime import time as dt_time
from scipy.stats import linregress

HMM_COLS = ["realized_vol", "overnight_gap", "trend_strength", "intraday_direction"]
ROUTING_COLS = ["or_range_atr"]

_OR_START = dt_time(9, 30)
_OR_END   = dt_time(10, 30)   # exclusive
_POST_OR  = dt_time(10, 30)   # first post-OR bar
_EOD      = dt_time(15, 45)


def compute_daily_features_p9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily features indexed by date.
    All features are strictly causal (use only data available before 10:30 NY
    or from prior sessions).
    """
    date_arr = np.array(df.index.date)
    u_dates, first_idx, counts = np.unique(
        date_arr, return_index=True, return_counts=True
    )

    # ── First pass: anchor values ────────────────────────────────────────
    daily_open0930:  dict = {}
    daily_close1030: dict = {}
    daily_or_hi:     dict = {}
    daily_or_lo:     dict = {}
    daily_close1545: dict = {}
    daily_atr_eod:   dict = {}

    for ui, d in enumerate(u_dates):
        s   = int(first_idx[ui])
        e   = s + int(counts[ui])
        seg = df.iloc[s:e]
        bt  = seg.index.time

        # 09:30 open
        for j, t in enumerate(bt):
            if t == _OR_START:
                daily_open0930[d] = float(seg["open"].iloc[j])
                break

        # First post-OR bar (10:30) close
        for j, t in enumerate(bt):
            if t == _POST_OR:
                daily_close1030[d] = float(seg["close"].iloc[j])
                break

        # OR high / low  (09:30 <= time < 10:30)
        or_mask = np.array([(t >= _OR_START and t < _OR_END) for t in bt])
        or_bars = seg[or_mask]
        if len(or_bars) > 0:
            daily_or_hi[d] = float(or_bars["high"].max())
            daily_or_lo[d] = float(or_bars["low"].min())

        # EOD close and ATR
        last_eod = -1
        for j, t in enumerate(bt):
            if t <= _EOD:
                last_eod = j
        if last_eod >= 0:
            daily_close1545[d] = float(seg["close"].iloc[last_eod])
            daily_atr_eod[d]   = float(seg["atr"].iloc[last_eod])

    dates_sorted = sorted(daily_close1545.keys())

    # ── Second pass: compute features ────────────────────────────────────
    rows = []
    for i, d in enumerate(dates_sorted):
        uidx = int(np.searchsorted(u_dates, d))
        if uidx >= len(u_dates) or u_dates[uidx] != d:
            continue
        s   = int(first_idx[uidx])
        e   = s + int(counts[uidx])
        seg = df.iloc[s:e]
        bt  = seg.index.time

        or_mask = np.array([(t >= _OR_START and t < _OR_END) for t in bt])
        or_bars = seg[or_mask]
        if len(or_bars) < 2:
            continue

        # 1 — realized_vol (std of OR log-returns)
        cl_or    = or_bars["close"].values.astype(float)
        log_rets = np.diff(np.log(np.maximum(cl_or, 1e-10)))
        rv       = float(np.std(log_rets)) if len(log_rets) >= 1 else np.nan

        # Prev-day ATR (causal)
        prev_atr = np.nan
        if i > 0:
            prev_atr = daily_atr_eod.get(dates_sorted[i - 1], np.nan)

        # 2 — overnight_gap / ATR
        gap = np.nan
        if i > 0:
            prev_cl  = daily_close1545.get(dates_sorted[i - 1], np.nan)
            today_op = daily_open0930.get(d, np.nan)
            if not np.isnan(prev_cl) and not np.isnan(today_op) \
                    and not np.isnan(prev_atr) and prev_atr > 0:
                gap = (today_op - prev_cl) / prev_atr

        # 3 — trend_strength (5-session slope / ATR)
        trend = np.nan
        if i >= 5:
            prev_cls = [daily_close1545.get(dates_sorted[j], np.nan)
                        for j in range(i - 5, i)]
            if not any(np.isnan(c) for c in prev_cls) \
                    and not np.isnan(prev_atr) and prev_atr > 0:
                slope, *_ = linregress(np.arange(5, dtype=float), prev_cls)
                trend = float(slope) / prev_atr

        # 4 — intraday_direction = sign(close_10:30 - open_09:30)
        op  = daily_open0930.get(d, np.nan)
        cl0 = daily_close1030.get(d, np.nan)
        intraday_dir = float(np.sign(cl0 - op)) if not (np.isnan(op) or np.isnan(cl0)) else np.nan

        # Routing — or_range_atr (not used in HMM)
        or_hi = daily_or_hi.get(d, np.nan)
        or_lo = daily_or_lo.get(d, np.nan)
        or_rng_atr = np.nan
        if not (np.isnan(or_hi) or np.isnan(or_lo) or np.isnan(prev_atr)) and prev_atr > 0:
            or_rng_atr = (or_hi - or_lo) / prev_atr

        rows.append({
            "date":              d,
            "realized_vol":      rv,
            "overnight_gap":     gap,
            "trend_strength":    trend,
            "intraday_direction":intraday_dir,
            "or_range_atr":      or_rng_atr,
        })

    return pd.DataFrame(rows).set_index("date")


def zscore_normalize_p9(features: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling z-score on HMM_COLS only (strictly causal via shift(1)).
    Returns DataFrame with HMM_COLS; days with < window prior obs become NaN.
    """
    result = pd.DataFrame(index=features.index, columns=HMM_COLS, dtype=float)
    for col in HMM_COLS:
        if col not in features.columns:
            continue
        s         = features[col]
        roll_mean = s.rolling(window=window, min_periods=window).mean().shift(1)
        roll_std  = s.rolling(window=window, min_periods=window).std().shift(1)
        result[col] = (s - roll_mean) / roll_std.replace(0.0, np.nan)
    return result
