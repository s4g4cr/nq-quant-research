"""
Phase 21 — Intraday Return Seasonality strategy engine.
Heston, Korajczyk, Sadka (2010): return continuation at exact half-hour
multiples of a trading day.

13 half-hour intervals. Signal = sign(mean return over prior N days).
Entry: close of first bar of interval + slippage.
Exit A: close of last bar of interval (hard, no SL/TP).
Exit B/C: SL=1.0×ATR, TP=tp_mult×ATR, hard exit at interval end if not hit.
"""

from dataclasses import dataclass
from datetime import time as dt_time, date as dt_date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SLIP    = 0.25
COMM_RT = 4.0   # $4 round-trip per contract
PV      = 20.0  # $20 per point

INTERVALS: List[Tuple[str, dt_time, dt_time]] = [
    ("I1",  dt_time(9, 30),  dt_time(10,  0)),
    ("I2",  dt_time(10,  0), dt_time(10, 30)),
    ("I3",  dt_time(10, 30), dt_time(11,  0)),
    ("I4",  dt_time(11,  0), dt_time(11, 30)),
    ("I5",  dt_time(11, 30), dt_time(12,  0)),
    ("I6",  dt_time(12,  0), dt_time(12, 30)),
    ("I7",  dt_time(12, 30), dt_time(13,  0)),
    ("I8",  dt_time(13,  0), dt_time(13, 30)),
    ("I9",  dt_time(13, 30), dt_time(14,  0)),
    ("I10", dt_time(14,  0), dt_time(14, 30)),
    ("I11", dt_time(14, 30), dt_time(15,  0)),
    ("I12", dt_time(15,  0), dt_time(15, 30)),
    ("I13", dt_time(15, 30), dt_time(16,  0)),
]
IV_NAMES = [iv[0] for iv in INTERVALS]
_IV_MAP  = {iv[0]: (iv[1], iv[2]) for iv in INTERVALS}


@dataclass
class SeasonalityTrade:
    date:             dt_date
    interval:         str
    lookback:         int
    direction:        str
    entry_ts:         object
    entry_px:         float
    exit_px:          float
    exit_reason:      str        # "time" | "sl" | "tp"
    atr_at_entry:     float
    n_contracts:      int
    capital_at_entry: float
    pnl_pts:          float
    pnl_net:          float
    bars_held:        int


def compute_interval_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute half-hour interval returns for all 13 intervals across all dates.
    return_Ik = (close_end - close_start) / close_start
    close_start = close of last bar BEFORE interval start
                  (uses prior day's last close for I1).
    close_end   = close of last bar in [istart, iend).
    """
    date_arr  = np.array(df.index.date)
    time_arr  = np.array(df.index.time)
    close_v   = df["close"].values
    u_dates   = np.unique(date_arr)

    rows: list = []
    prev_last_close: Optional[float] = None

    for d in u_dates:
        mask    = date_arr == d
        d_idx   = np.where(mask)[0]
        d_times = time_arr[d_idx]
        row     = {"date": d}

        for iname, istart, iend in INTERVALS:
            in_iv  = (d_times >= istart) & (d_times < iend)
            iv_abs = d_idx[in_iv]
            if len(iv_abs) == 0:
                row[iname] = np.nan
                continue

            before_abs = d_idx[d_times < istart]
            if len(before_abs) > 0:
                cs = float(close_v[before_abs[-1]])
            elif prev_last_close is not None:
                cs = prev_last_close
            else:
                row[iname] = np.nan
                continue

            ce = float(close_v[iv_abs[-1]])
            row[iname] = (ce - cs) / cs

        prev_last_close = float(close_v[d_idx[-1]])
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def _autocorr_lag(series: pd.Series, lag: int):
    s = series.dropna()
    if len(s) <= lag + 5:
        return np.nan, np.nan
    x = s.values[:-lag]; y = s.values[lag:]
    v = ~(np.isnan(x) | np.isnan(y))
    x, y = x[v], y[v]
    if len(x) < 10:
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _dir_accuracy(series: pd.Series) -> float:
    """Directional accuracy: sign(ret_yesterday) predicts sign(ret_today)."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    sig = np.sign(s.values[:-1]); act = np.sign(s.values[1:])
    v   = (sig != 0) & (act != 0)
    return float((sig[v] == act[v]).mean() * 100) if v.sum() > 0 else np.nan


def autocorrelation_table(rets_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iname in IV_NAMES:
        s = rets_df[iname]
        ac1, p1 = _autocorr_lag(s, 1)
        ac2, p2 = _autocorr_lag(s, 2)
        ac5, p5 = _autocorr_lag(s, 5)
        da = _dir_accuracy(s)
        rows.append({
            "interval": iname,
            "ac1": round(ac1, 4), "p1": round(p1, 4),
            "ac2": round(ac2, 4), "p2": round(p2, 4),
            "ac5": round(ac5, 4), "p5": round(p5, 4),
            "dacc_pct": round(da, 1),
        })
    return pd.DataFrame(rows)


def lookback_directional_accuracy(
    rets_df: pd.DataFrame, iname: str, ns
) -> pd.DataFrame:
    """Directional accuracy on rets_df for each lookback N."""
    s = rets_df[iname].dropna()
    rows = []
    for N in ns:
        correct = total = 0
        for i in range(N, len(s)):
            prior = s.iloc[i - N:i]
            if prior.isna().any():
                continue
            sig = np.sign(float(prior.mean()))
            act = np.sign(float(s.iloc[i]))
            if sig != 0 and act != 0:
                total += 1
                if sig == act:
                    correct += 1
        acc = correct / total * 100 if total > 0 else np.nan
        rows.append({"N": N, "dacc_pct": round(acc, 1), "n_obs": total})
    return pd.DataFrame(rows)


def run_interval_backtest(
    df: pd.DataFrame,
    rets_df: pd.DataFrame,
    interval_name: str,
    lookback: int = 1,
    exit_method: str = "A",
    tp_mult: float = 1.5,
    initial_capital: float = 100_000.0,
    risk_pct: float = 1.0,
) -> List[SeasonalityTrade]:
    """
    Backtest one interval on the dates in df, using rets_df for causal signal.

    df        - 1-min bars with columns: close, high, low, atr.
    rets_df   - full-history interval returns (signal lookback reaches before df).
    exit_method "A": hard at interval end.
                "B"/"C": SL=1.0xATR, TP=tp_mult×ATR, hard exit at end if not hit.
    """
    if interval_name not in _IV_MAP:
        raise ValueError(f"Unknown interval: {interval_name}")
    istart, iend = _IV_MAP[interval_name]

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    close_v  = df["close"].values
    high_v   = df["high"].values
    low_v    = df["low"].values
    atr_v    = df["atr"].values

    ret_series = rets_df[interval_name]
    u_dates    = np.unique(date_arr)
    trades:    List[SeasonalityTrade] = []
    capital    = float(initial_capital)

    for d in u_dates:
        d_ts      = pd.Timestamp(d)
        prior_idx = ret_series.index[ret_series.index < d_ts]
        if len(prior_idx) < lookback:
            continue
        last_N = ret_series.loc[prior_idx[-lookback:]]
        if last_N.isna().any() or len(last_N) < lookback:
            continue
        sig = np.sign(float(last_N.mean()))
        if sig == 0:
            continue
        direction = "long" if sig > 0 else "short"

        mask    = date_arr == d
        d_idx   = np.where(mask)[0]
        d_times = time_arr[d_idx]
        in_iv   = (d_times >= istart) & (d_times < iend)
        iv_abs  = d_idx[in_iv]
        if len(iv_abs) == 0:
            continue

        entry_abs = int(iv_abs[0])
        ec        = float(close_v[entry_abs])
        cur_atr   = float(atr_v[entry_abs])
        if np.isnan(cur_atr) or cur_atr <= 0:
            continue

        ep  = ec + SLIP if direction == "long" else ec - SLIP
        n_c = max(1, int(capital * risk_pct / 100.0 / (cur_atr * PV)))

        if exit_method == "A":
            xi          = int(iv_abs[-1])
            xc          = float(close_v[xi])
            xp          = xc - SLIP if direction == "long" else xc + SLIP
            exit_reason = "time"
            bars_held   = int(xi - entry_abs)
        else:
            sl = ep - cur_atr             if direction == "long" else ep + cur_atr
            tp = ep + tp_mult * cur_atr   if direction == "long" else ep - tp_mult * cur_atr
            xp = None; exit_reason = "time"; bars_held = 0
            for pos in iv_abs[1:]:
                bars_held += 1
                h = float(high_v[pos]); l = float(low_v[pos]); c = float(close_v[pos])
                sl_hit = (l <= sl) if direction == "long" else (h >= sl)
                tp_hit = (h >= tp) if direction == "long" else (l <= tp)
                if sl_hit:
                    xp = sl; exit_reason = "sl"; break
                if tp_hit:
                    xp = tp; exit_reason = "tp"; break
            if xp is None:
                xc = float(close_v[int(iv_abs[-1])])
                xp = xc - SLIP if direction == "long" else xc + SLIP

        pnl_pts = (xp - ep) if direction == "long" else (ep - xp)
        pnl_net = pnl_pts * n_c * PV - COMM_RT * n_c

        trades.append(SeasonalityTrade(
            date=d, interval=interval_name, lookback=lookback, direction=direction,
            entry_ts=df.index[entry_abs], entry_px=ep, exit_px=xp,
            exit_reason=exit_reason, atr_at_entry=cur_atr, n_contracts=n_c,
            capital_at_entry=capital, pnl_pts=pnl_pts, pnl_net=pnl_net,
            bars_held=bars_held,
        ))
        capital += pnl_net

    return trades
