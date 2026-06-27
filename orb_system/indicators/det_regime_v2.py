"""
Phase 15 corrected deterministic regime features.

Fix: F1 and F3 now normalized by daily_atr (20-day rolling mean of RTH
session ranges) instead of the 5-minute bar ATR(20). This makes F1 and
F3 dimensionally consistent — session-level numerators divided by a
session-level denominator.

  daily_atr        : mean of last 20 completed RTH session ranges (causal)
  prev_range_ratio : prev_session_range / daily_atr  [Filter 1]
  trend_5d         : (close_yday - close_5d_ago) / daily_atr  [Filter 3]
"""

from datetime import time as dt_time

import numpy as np
import pandas as pd

_RTH_S           = dt_time(9, 30)
_RTH_E           = dt_time(15, 45)
_DAILY_ATR_WINDOW = 20


def compute_det_regime_features_v2(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (prev_range_ratio, trend_5d, daily_atr) as pd.Series aligned
    to df.index. NaN where insufficient history exists.

    daily_atr[today] = mean of the last 20 completed RTH session ranges,
    computed before today's session opens — strictly causal.
    """
    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    u_dates  = np.unique(date_arr)

    hi_v = df["high"].values
    lo_v = df["low"].values
    cl_v = df["close"].values

    # ── per-session summaries ─────────────────────────────────────────────────
    sess: dict = {}
    for d in u_dates:
        mask     = date_arr == d
        idxs     = all_pos[mask]
        times    = time_arr[mask]
        rth_mask = np.array([_RTH_S <= t <= _RTH_E for t in times])
        if rth_mask.sum() == 0:
            continue
        rth_idxs = idxs[rth_mask]
        rth_hi   = float(hi_v[rth_idxs].max())
        rth_lo   = float(lo_v[rth_idxs].min())
        rth_cl   = float(cl_v[rth_idxs[-1]])
        sess_rng = rth_hi - rth_lo
        sess[d]  = (rth_hi, rth_lo, rth_cl, sess_rng)

    u          = sorted(sess.keys())
    n          = len(u)
    sess_ranges = np.array([sess[d][3] for d in u])

    # ── daily_atr: 20-session rolling mean, strictly causal ───────────────────
    daily_atr_arr = np.full(n, np.nan)
    for i in range(_DAILY_ATR_WINDOW, n):
        daily_atr_arr[i] = float(np.mean(sess_ranges[i - _DAILY_ATR_WINDOW: i]))

    # ── F1 and F3 using daily_atr ─────────────────────────────────────────────
    prr_map: dict = {}
    t5d_map: dict = {}
    da_map:  dict = {}

    for i, d in enumerate(u):
        da        = daily_atr_arr[i]
        da_map[d] = da

        # Filter 1: prev session range / daily_atr
        if i >= 1 and not np.isnan(da) and da > 0:
            prr_map[d] = sess[u[i - 1]][3] / da
        else:
            prr_map[d] = np.nan

        # Filter 3: (close_yday - close_5d_ago) / daily_atr
        if i >= 5 and not np.isnan(da) and da > 0:
            t5d_map[d] = (sess[u[i - 1]][2] - sess[u[i - 5]][2]) / da
        else:
            t5d_map[d] = np.nan

    prr_arr = np.array([prr_map.get(d, np.nan) for d in date_arr])
    t5d_arr = np.array([t5d_map.get(d, np.nan) for d in date_arr])
    da_arr  = np.array([da_map.get(d,  np.nan) for d in date_arr])

    return (
        pd.Series(prr_arr, index=df.index, name="prev_range_ratio"),
        pd.Series(t5d_arr, index=df.index, name="trend_5d"),
        pd.Series(da_arr,  index=df.index, name="daily_atr"),
    )
