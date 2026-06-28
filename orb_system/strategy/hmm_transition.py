"""
Phase 22 — Predictive HMM via Transition Matrix.
Hamilton (1989) regime-switching model.

Extracts daily (return, volume) features, trains GaussianHMM, computes
empirical transition matrix, and labels states by mean daily return.
"""

from datetime import time as dt_time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chisquare

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("hmmlearn required: pip install hmmlearn")

SESS_START = dt_time(9, 30)
SESS_END   = dt_time(15, 45)


def extract_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-session summary features from 1-min bars.
    Session window: [09:30, 15:45] inclusive.

    Returned columns:
      date, open_930, close_1545, session_high, session_low,
      volume, daily_return
    """
    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    open_v   = df["open"].values
    high_v   = df["high"].values
    low_v    = df["low"].values
    close_v  = df["close"].values
    vol_v    = df["volume"].values

    rows = []
    for d in np.unique(date_arr):
        mask    = date_arr == d
        d_idx   = np.where(mask)[0]
        d_times = time_arr[d_idx]

        loc_930  = np.where(d_times == SESS_START)[0]
        loc_1545 = np.where(d_times == SESS_END)[0]
        if len(loc_930) == 0 or len(loc_1545) == 0:
            continue

        abs_930  = d_idx[loc_930[0]]
        abs_1545 = d_idx[loc_1545[0]]
        sess_abs = d_idx[(d_times >= SESS_START) & (d_times <= SESS_END)]

        open_930   = float(open_v[abs_930])
        close_1545 = float(close_v[abs_1545])
        if open_930 <= 0:
            continue

        rows.append({
            "date":         d,
            "open_930":     open_930,
            "close_1545":   close_1545,
            "session_high": float(high_v[sess_abs].max()),
            "session_low":  float(low_v[sess_abs].min()),
            "volume":       float(vol_v[sess_abs].sum()),
            "daily_return": (close_1545 - open_930) / open_930,
        })

    return pd.DataFrame(rows)


def add_causal_features(feat: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add two causal features (computed from prior sessions only):
      volume_ratio  = volume / mean(prior_20_session_volumes)
      daily_atr     = mean(prior_20_session_ranges)
    Rows with < 5 prior sessions get NaN (excluded from HMM training).
    """
    df   = feat.copy()
    vols  = df["volume"].values
    rngs  = (df["session_high"] - df["session_low"]).values
    n     = len(df)
    vr    = np.full(n, np.nan)
    atr   = np.full(n, np.nan)

    for i in range(1, n):
        pv = vols[max(0, i - window):i]
        pr = rngs[max(0, i - window):i]
        if len(pv) >= 5:
            vr[i]  = vols[i] / pv.mean()
        if len(pr) >= 5:
            atr[i] = pr.mean()

    df["volume_ratio"] = vr
    df["daily_atr"]    = atr
    return df


def train_hmm(
    X: np.ndarray,
    n_states: int,
    seed: int = 42,
) -> Tuple[GaussianHMM, np.ndarray]:
    """
    Train GaussianHMM on standardised observation matrix X (n_samples × 2).
    Returns (model, state_sequence).
    """
    mu  = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig == 0] = 1.0
    Xz  = (X - mu) / sig

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=seed,
    )
    model.fit(Xz)
    states = model.predict(Xz)
    # Attach normalisation stats to model for reuse on test set
    model._norm_mu  = mu
    model._norm_sig = sig
    return model, states


def predict_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Apply Viterbi to X using normalisation from training."""
    mu  = model._norm_mu
    sig = model._norm_sig
    Xz  = (X - mu) / sig
    return model.predict(Xz)


def label_states(
    states: np.ndarray,
    daily_returns: np.ndarray,
    n_states: int,
) -> Dict[int, str]:
    """
    Map integer state IDs → semantic labels by mean daily_return rank.
    Lowest mean → 'bearish', highest → 'bullish'.
    """
    means  = {s: float(daily_returns[states == s].mean()) for s in range(n_states)}
    ranked = sorted(means.keys(), key=lambda k: means[k])

    if n_states == 2:
        label_list = ["bearish", "bullish"]
    elif n_states == 3:
        label_list = ["bearish", "ranging", "bullish"]
    elif n_states == 4:
        label_list = ["bear_strong", "bear_weak", "bull_weak", "bull_strong"]
    else:
        label_list = [f"s{i}" for i in range(n_states)]

    return {ranked[i]: label_list[i] for i in range(n_states)}


def compute_transition_matrix(
    states: np.ndarray,
    n_states: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical T[i][j] = P(state_j tomorrow | state_i today).
    Returns (probability matrix, raw count matrix).
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(states) - 1):
        counts[states[t], states[t + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    T = np.where(row_sums > 0, counts / row_sums, 1.0 / n_states)
    return T, counts


def chi_sq_row(count_row: np.ndarray) -> Tuple[float, float]:
    """
    Chi-square test: is the transition distribution significantly non-uniform?
    H0: all next-state transitions equally likely (1/N each).
    """
    n = count_row.sum()
    if n < 10:
        return np.nan, np.nan
    expected = np.full(len(count_row), n / len(count_row))
    stat, p  = chisquare(count_row, expected)
    return float(stat), float(p)
