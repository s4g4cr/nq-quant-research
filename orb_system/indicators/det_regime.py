"""
Deterministic regime filter features for Phase 14.

  prev_range_ratio : prev RTH session range / ATR(20) today  [Filter 1]
  trend_5d         : (close_yday - close_5d_ago) / ATR(20) yday  [Filter 3]

Both strictly causal — only completed prior session data is used.
"""

from datetime import time as dt_time

import numpy as np
import pandas as pd

_RTH_S = dt_time(9, 30)
_RTH_E = dt_time(15, 45)


def compute_det_regime_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Returns (prev_range_ratio, trend_5d) as pd.Series aligned to df.index.
    NaN for sessions with insufficient history.
    """
    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    u_dates  = np.unique(date_arr)

    hi_v  = df["high"].values
    lo_v  = df["low"].values
    cl_v  = df["close"].values
    atr_v = df["atr"].values

    # Collect per-session summaries
    sess: dict = {}
    for d in u_dates:
        mask     = date_arr == d
        idxs     = all_pos[mask]
        times    = time_arr[mask]
        rth_mask = np.array([_RTH_S <= t <= _RTH_E for t in times])
        if rth_mask.sum() == 0:
            continue
        rth_idxs = idxs[rth_mask]
        sess[d]  = (
            float(hi_v[rth_idxs].max()),   # rth_hi
            float(lo_v[rth_idxs].min()),   # rth_lo
            float(cl_v[rth_idxs[-1]]),     # rth_close (last RTH bar)
            float(atr_v[rth_idxs[0]]),     # atr at first RTH bar (today's open ATR)
        )

    u = sorted(sess.keys())
    prr_map: dict = {}
    t5d_map: dict = {}

    for i, d in enumerate(u):
        _, _, _, atr_today = sess[d]

        # Filter 1: prev day range / today's ATR
        if i >= 1:
            pd_hi, pd_lo, _, _ = sess[u[i - 1]]
            prr_map[d] = (pd_hi - pd_lo) / atr_today if atr_today > 0 else np.nan
        else:
            prr_map[d] = np.nan

        # Filter 3: 5-day trend using yesterday's ATR as denominator
        if i >= 5:
            cl_yday  = sess[u[i - 1]][2]
            cl_5ago  = sess[u[i - 5]][2]
            atr_yday = sess[u[i - 1]][3]
            t5d_map[d] = (cl_yday - cl_5ago) / atr_yday if atr_yday > 0 else np.nan
        else:
            t5d_map[d] = np.nan

    prr_arr = np.array([prr_map.get(d, np.nan) for d in date_arr])
    t5d_arr = np.array([t5d_map.get(d, np.nan) for d in date_arr])

    return (
        pd.Series(prr_arr, index=df.index, name="prev_range_ratio"),
        pd.Series(t5d_arr, index=df.index, name="trend_5d"),
    )
