"""
Volume profile and Point of Control (POC) calculations.
Resolution: 0.25pt ticks (NQ minimum tick size).

prev_poc    — full RTH session POC of prior day (09:30-15:45), fixed all day
session_poc — rolling intraday POC, updates each bar from 09:30
Both are strictly causal: no future data used.
"""

from datetime import time as dt_time

import numpy as np
import pandas as pd

TICK      = 0.25
_RTH_S    = dt_time(9, 30)
_RTH_E    = dt_time(15, 45)


def _q(x: float) -> float:
    return round(round(x / TICK) * TICK, 4)


def _full_session_poc(lo_v, hi_v, vol_v) -> float:
    lo_b  = _q(float(lo_v.min()))
    hi_b  = _q(float(hi_v.max()))
    n     = max(1, round((hi_b - lo_b) / TICK) + 1)
    cum   = np.zeros(n, dtype=np.float64)
    for j in range(len(lo_v)):
        lo_i = max(0, round((_q(float(lo_v[j])) - lo_b) / TICK))
        hi_i = min(n - 1, round((_q(float(hi_v[j])) - lo_b) / TICK))
        w    = hi_i - lo_i + 1
        if w > 0:
            cum[lo_i:hi_i + 1] += float(vol_v[j]) / w
    return round(lo_b + int(np.argmax(cum)) * TICK, 4)


def compute_poc_features(
    df: pd.DataFrame,
    confluence_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Return DataFrame aligned to df's index with columns:
      prev_poc, session_poc, poc_confluence, target_poc

    prev_poc is NaN for the first session (no prior day available).
    session_poc is NaN before the RTH open (09:30).
    """
    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    u_dates  = np.unique(date_arr)

    # Pass 1: full RTH POC per day (used as prev_poc for the next day)
    full_poc: dict = {}
    for d in u_dates:
        mask = (date_arr == d) & np.array([_RTH_S <= t <= _RTH_E for t in time_arr])
        if mask.sum() < 2:
            continue
        seg = df[mask]
        full_poc[d] = _full_session_poc(
            seg["low"].values, seg["high"].values, seg["volume"].values
        )

    # Pass 2: rolling session_poc + prev_poc per bar
    n_bars          = len(df)
    prev_poc_col    = np.full(n_bars, np.nan)
    session_poc_col = np.full(n_bars, np.nan)
    all_pos         = np.arange(n_bars)

    # Pre-extract numpy arrays to avoid repeated DataFrame attribute lookups
    _lo  = df["low"].values.astype(float)
    _hi  = df["high"].values.astype(float)
    _vol = df["volume"].values.astype(float)

    for i, d in enumerate(u_dates):
        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]

        # prev_poc fixed for this day
        pp = full_poc.get(u_dates[i - 1], np.nan) if i > 0 else np.nan
        prev_poc_col[idxs] = pp

        # determine bin range from RTH bars only
        rth_mask = np.array([t >= _RTH_S for t in times])
        rth_idxs = idxs[rth_mask]
        if rth_idxs.size == 0:
            continue
        lo_base = _q(_lo[rth_idxs].min())
        hi_base = _q(_hi[rth_idxs].max())
        n_bins  = max(1, round((hi_base - lo_base) / TICK) + 1)
        cum     = np.zeros(n_bins, dtype=np.float64)

        for j, pos in enumerate(idxs):
            if times[j] >= _RTH_S:
                lo_b = _q(_lo[pos])
                hi_b = _q(_hi[pos])
                lo_i = max(0, round((lo_b - lo_base) / TICK))
                hi_i = min(n_bins - 1, round((hi_b - lo_base) / TICK))
                w    = hi_i - lo_i + 1
                if w > 0:
                    cum[lo_i:hi_i + 1] += _vol[pos] / w

            if cum.sum() > 0:
                session_poc_col[pos] = round(lo_base + int(np.argmax(cum)) * TICK, 4)

    # Assemble
    out = pd.DataFrame(index=df.index)
    out["prev_poc"]    = prev_poc_col
    out["session_poc"] = session_poc_col

    pp   = out["prev_poc"]
    sp   = out["session_poc"]
    conf = (~pp.isna()) & (~sp.isna()) & ((pp - sp).abs() <= confluence_threshold)
    out["poc_confluence"] = conf
    out["target_poc"] = np.where(conf, (pp + sp) / 2.0, pp)
    out.loc[pp.isna(), "target_poc"]     = np.nan
    out.loc[pp.isna(), "poc_confluence"] = False

    return out
