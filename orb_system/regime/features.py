"""
Daily feature computation for HMM regime detection.
All features use only data available before 10:30 NY (causal, no lookahead).

Features:
  realized_vol          std of OR log-returns (09:30-10:29)
  or_range_normalized   (or_high - or_low) / prev_day_ATR20
  overnight_gap         (open_0930_today - close_1545_yday) / prev_day_ATR20
  trend_strength        slope of last-5-session-closes linear regression / prev_day_ATR20
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress

_OR_START_STR = "09:30"
_OR_END_STR   = "10:29"   # inclusive upper bound (< 10:30)
_EOD_STR      = "15:45"


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date with columns:
      realized_vol, or_range_normalized, overnight_gap, trend_strength
    All values are raw (not z-scored).
    """
    _or_start = pd.Timestamp(f"2000-01-01 {_OR_START_STR}").time()
    _or_end   = pd.Timestamp(f"2000-01-01 10:30").time()   # exclusive
    _eod      = pd.Timestamp(f"2000-01-01 {_EOD_STR}").time()

    # vectorised date grouping
    date_arr   = np.array(df.index.date)
    u_dates, first_idx, counts = np.unique(date_arr, return_index=True, return_counts=True)

    # --- single-pass: collect per-session anchor values ---------------
    daily_open0930: dict = {}
    daily_close1545: dict = {}
    daily_atr_eod: dict = {}

    for ui, d in enumerate(u_dates):
        s = first_idx[ui]; e = s + counts[ui]
        sess   = df.iloc[s:e]
        bt     = sess.index.time

        # open at 09:30
        for j, t in enumerate(bt):
            if t == _or_start:
                daily_open0930[d] = float(sess["open"].iloc[j])
                break

        # last bar at/before 15:45
        last_eod_idx = -1
        for j, t in enumerate(bt):
            if t <= _eod:
                last_eod_idx = j
        if last_eod_idx >= 0:
            daily_close1545[d] = float(sess["close"].iloc[last_eod_idx])
            daily_atr_eod[d]   = float(sess["atr"].iloc[last_eod_idx])

    dates_sorted = sorted(daily_close1545.keys())

    # --- second-pass: compute features --------------------------------
    rows = []
    for i, d in enumerate(dates_sorted):
        uidx = int(np.searchsorted(u_dates, d))
        if uidx >= len(u_dates) or u_dates[uidx] != d:
            continue
        s = first_idx[uidx]; e = s + counts[uidx]
        sess = df.iloc[s:e]
        bt   = sess.index.time

        # OR bars: 09:30 <= time < 10:30
        or_mask = np.array([(t >= _or_start and t < _or_end) for t in bt])
        or_bars = sess[or_mask]
        if len(or_bars) < 2:
            continue

        # feature 1 — realized_vol
        cl_or    = or_bars["close"].values.astype(float)
        log_rets = np.diff(np.log(np.maximum(cl_or, 1e-10)))
        rv = float(np.std(log_rets)) if len(log_rets) >= 1 else np.nan

        # prev-day ATR (causal)
        prev_atr = np.nan
        if i > 0:
            prev_d   = dates_sorted[i - 1]
            prev_atr = daily_atr_eod.get(prev_d, np.nan)

        # feature 2 — or_range_normalized
        or_hi  = float(or_bars["high"].max())
        or_lo  = float(or_bars["low"].min())
        or_rng = or_hi - or_lo
        or_rng_norm = (or_rng / prev_atr) if (not np.isnan(prev_atr) and prev_atr > 0) else np.nan

        # feature 3 — overnight_gap
        gap = np.nan
        if i > 0:
            prev_d  = dates_sorted[i - 1]
            prev_cl = daily_close1545.get(prev_d, np.nan)
            today_op = daily_open0930.get(d, np.nan)
            if not any(np.isnan(v) for v in [prev_cl, today_op, prev_atr]) and prev_atr > 0:
                gap = (today_op - prev_cl) / prev_atr

        # feature 4 — trend_strength (slope of last-5 closes / prev_atr)
        trend = np.nan
        if i >= 5:
            prev_closes = [daily_close1545.get(dates_sorted[j], np.nan) for j in range(i - 5, i)]
            if not any(np.isnan(c) for c in prev_closes) and not np.isnan(prev_atr) and prev_atr > 0:
                x = np.arange(5, dtype=float)
                slope, *_ = linregress(x, prev_closes)
                trend = float(slope) / prev_atr

        rows.append({
            "date":               d,
            "realized_vol":       rv,
            "or_range_normalized": or_rng_norm,
            "overnight_gap":      gap,
            "trend_strength":     trend,
        })

    feat = pd.DataFrame(rows).set_index("date")
    return feat


def zscore_normalize(features: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling z-score using the previous `window` days (strictly causal — current
    day excluded from the rolling mean/std via shift(1)).
    Days with fewer than `window` prior observations → NaN (exclude from HMM).
    """
    result = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)
    for col in features.columns:
        s         = features[col]
        roll_mean = s.rolling(window=window, min_periods=window).mean().shift(1)
        roll_std  = s.rolling(window=window, min_periods=window).std().shift(1)
        result[col] = (s - roll_mean) / roll_std.replace(0.0, np.nan)
    return result
