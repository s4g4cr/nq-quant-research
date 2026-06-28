"""
Phase 19 strategy engine: Intraday Momentum (Gao et al. 2018).

HYPOTHESIS: The first half-hour return of NQ futures (09:30–10:00 NY,
measured from previous session close) positively predicts the last
half-hour return (15:30–16:00 NY). Enter at 15:30 in the direction
of the first half-hour return and hold to session close.

Reference: Gao, Han, Li, Zhou (2018) "Market Intraday Momentum"
           Journal of Financial Economics 129(2), 394-414.
           Original finding: S&P 500 ETF 1993-2013. Replicated across
           60+ futures and international ETFs.

r1  = (close_09:59 - close_prev_15:59) / close_prev_15:59
r16 = (close_14:59 - close_14:29) / close_14:29   (second-to-last h-hr)
r17 = (close_15:59 - close_15:29) / close_15:29   (last half-hour)

Entry:  close of 15:29 bar (default) + slippage
Exit:   close of 15:59 bar always — no SL, no TP
SL ref: ATR(20) at entry bar — used for position sizing only
Slippage:   0.25 pts per side (1 tick)
Commission: $4.00 round-trip per contract ($2/side)

──────────────────────────────────────────────────────────────────────────────
HYPOTHESIS FALSIFIED — Phase 19

The Gao et al. (2018) intraday momentum effect does not hold in NQ
E-mini futures over the 2021–2026 period. Diagnostic halted experiments.

Full dataset (1,203 sessions with data):
  r1 > 0 → mean r17 = −0.0117%  (NEGATIVE — reversal, not momentum)
  r1 < 0 → mean r17 = +0.0227%  (POSITIVE — reversal, not momentum)
  Directional accuracy: 48.6% (< 50% → diagnostic gate failed)

Two-sample t-test (r17 | r1>0 vs r17 | r1<0):
  t = −1.939  p = 0.974 (momentum direction)
  t = −1.939  p = 0.026 (reversal direction) ← statistically significant

Key insight: NQ shows INTRADAY REVERSAL, not momentum. When the first
half-hour is strongly positive, the last half-hour tends to be slightly
negative, and vice versa. This is the opposite of the paper's finding in
S&P 500 ETF (1993-2013). Possible explanations:

  1. NQ is a momentum-heavy instrument — early movers tend to overshoot
     intraday and revert by the close, rather than continuing.
  2. The paper's effect may be concentrated in lower-beta, mean-reverting
     instruments, or may have decayed post-2013 as it became well-known.
  3. The 2021-2026 sample includes several high-volatility regimes
     (COVID recovery, rate hike cycle) where intraday reversal is
     more pronounced.

The reversal hypothesis (enter OPPOSITE to r1 direction) was not tested
per the Phase 19 spec — experiments are gated on the stated hypothesis.
It is a viable candidate for a future phase.

Parameters tested: diagnostic only — no experiments ran.
──────────────────────────────────────────────────────────────────────────────
"""
from dataclasses import dataclass
from datetime import time as dt_time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SLIP = 0.25   # pts per side (1 NQ tick)
COMM = 4.0    # $ round-trip per contract ($2/side)
PV   = 20.0   # NQ point value $/pt

_R1_END_T    = dt_time(9, 59)   # close of 09:59 = end of first half-hour
_R16_REF_T   = dt_time(14, 29)  # close of 14:29 = r16 start reference
_R16_END_T   = dt_time(14, 59)  # close of 14:59 = end of r16 period
_ENTRY_T     = dt_time(15, 29)  # default entry bar (→ enter at 15:30)
_EXIT_T      = dt_time(15, 59)  # always exit here (→ session close)


@dataclass
class SessionInfo:
    date: object
    close_prev: float           # prior session's 15:59 close
    close_r1_end: float         # current 09:59 close
    r1: float                   # first half-hour return (from prev close)
    range_first30: float        # high-low range of 09:30–09:59 bars
    vol_first30: float          # total volume 09:30–09:59 bars
    r16: float                  # 14:30–15:00 half-hour return (NaN if missing)
    close_entry: float          # close of default entry bar (15:29)
    close_exit: float           # close of 15:59 bar
    r17: float                  # last half-hour return (15:29→15:59)


@dataclass
class MomentumTrade:
    date: object
    direction: str              # "long" or "short"
    r1: float
    r17: float
    range_first30: float
    r16: float
    vol_first30: float
    entry_ts: object
    entry_price: float
    exit_ts: object
    exit_price: float
    n_contracts: int
    capital_at_entry: float
    atr_at_entry: float
    pnl_pts: float = 0.0
    pnl_net: float = 0.0


def detect_sessions(
    df: pd.DataFrame,
    atr_1min_series: pd.Series,
) -> Dict[object, Optional[SessionInfo]]:
    """
    Compute SessionInfo for every session date in df.
    Returns dict[date -> SessionInfo | None].
    None when critical bars are missing.
    """
    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    u_dates  = np.unique(date_arr)

    hi_v  = df["high"].values
    lo_v  = df["low"].values
    cl_v  = df["close"].values
    vol_v = df["volume"].values
    a1_v  = atr_1min_series.values

    # Build prev-session close lookup (15:59 bar close of prior session)
    prev_close: dict = {}
    for i, d in enumerate(u_dates):
        mask = date_arr == d
        idxs = all_pos[mask]
        t15  = [times for times, pos in zip(time_arr[mask], idxs)
                if times == _EXIT_T]
        pos15 = [pos for times, pos in zip(time_arr[mask], idxs)
                 if times == _EXIT_T]
        if pos15 and i + 1 < len(u_dates):
            prev_close[u_dates[i + 1]] = float(cl_v[pos15[0]])

    result: dict = {}
    for d in u_dates:
        if d not in prev_close:
            result[d] = None
            continue

        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]

        # Build time→pos map for this session
        t2p = {t: p for t, p in zip(times, idxs)}

        # Require critical bars
        if _R1_END_T not in t2p or _EXIT_T not in t2p or _ENTRY_T not in t2p:
            result[d] = None
            continue

        c_prev     = prev_close[d]
        c_r1_end   = float(cl_v[t2p[_R1_END_T]])
        c_entry    = float(cl_v[t2p[_ENTRY_T]])
        c_exit     = float(cl_v[t2p[_EXIT_T]])

        r1  = (c_r1_end - c_prev) / c_prev if c_prev != 0 else float("nan")
        r17 = (c_exit - c_entry) / c_entry if c_entry != 0 else float("nan")

        # First 30-min range and volume
        sp_mask = np.array([dt_time(9, 30) <= t <= dt_time(9, 59) for t in times])
        sp_idxs = idxs[sp_mask]
        if len(sp_idxs) == 0:
            result[d] = None
            continue
        rng30  = float(hi_v[sp_idxs].max() - lo_v[sp_idxs].min())
        vol30  = float(vol_v[sp_idxs].sum())

        # r16: 14:30–15:00 half-hour return
        r16 = float("nan")
        if _R16_REF_T in t2p and _R16_END_T in t2p:
            c16s = float(cl_v[t2p[_R16_REF_T]])
            c16e = float(cl_v[t2p[_R16_END_T]])
            r16  = (c16e - c16s) / c16s if c16s != 0 else float("nan")

        result[d] = SessionInfo(
            date=d, close_prev=c_prev, close_r1_end=c_r1_end,
            r1=r1, range_first30=rng30, vol_first30=vol30,
            r16=r16, close_entry=c_entry, close_exit=c_exit, r17=r17,
        )
    return result


def run(
    df: pd.DataFrame,
    session_infos: dict,
    atr_1min_series: pd.Series,
    entry_bar_time: dt_time = _ENTRY_T,
    r1_threshold: float = 0.0,
    high_vol_only: bool = False,
    rv_median_dict: Optional[dict] = None,
    vol_filter: bool = False,
    vol_median_dict: Optional[dict] = None,
    r16_agreement: bool = False,
    initial_capital: float = 100_000.0,
    risk_pct: float = 1.0,
) -> List[MomentumTrade]:
    """
    Simulate Phase 19 momentum strategy.
    Entry at close of entry_bar_time bar. Exit always at 15:59 close.
    No SL, no TP — position held to session close.
    """
    trades: List[MomentumTrade] = []
    capital = float(initial_capital)

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    cl_v     = df["close"].values
    a1_v     = atr_1min_series.values

    for d in sorted(k for k, v in session_infos.items() if v is not None):
        si = session_infos[d]
        if np.isnan(si.r1) or si.r1 == 0.0:
            continue

        # Magnitude filter
        if abs(si.r1) <= r1_threshold:
            continue

        # Volatility filter (high-vol days only)
        if high_vol_only and rv_median_dict is not None:
            rv_med = rv_median_dict.get(d)
            if rv_med is None or np.isnan(rv_med) or si.range_first30 <= rv_med:
                continue

        # Volume filter
        if vol_filter and vol_median_dict is not None:
            vol_med = vol_median_dict.get(d)
            if vol_med is None or np.isnan(vol_med) or si.vol_first30 <= vol_med:
                continue

        # r16 agreement filter
        if r16_agreement:
            if np.isnan(si.r16):
                continue
            if np.sign(si.r1) != np.sign(si.r16):
                continue

        # Find entry bar and exit bar for this date
        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]
        t2p   = {t: p for t, p in zip(times, idxs)}

        if entry_bar_time not in t2p or _EXIT_T not in t2p:
            continue

        entry_pos = t2p[entry_bar_time]
        exit_pos  = t2p[_EXIT_T]

        b_entry_cl = float(cl_v[entry_pos])
        b_exit_cl  = float(cl_v[exit_pos])
        b_atr      = float(a1_v[entry_pos])

        if np.isnan(b_atr) or b_atr <= 0:
            continue

        direction = "long" if si.r1 > 0 else "short"

        if direction == "long":
            ep = b_entry_cl + SLIP
            xp = b_exit_cl  - SLIP
        else:
            ep = b_entry_cl - SLIP
            xp = b_exit_cl  + SLIP

        risk_usd = capital * risk_pct / 100.0
        n_c      = max(1, int(risk_usd / (b_atr * PV)))

        pnl_pts = (xp - ep) if direction == "long" else (ep - xp)
        pnl_net = pnl_pts * n_c * PV - COMM * n_c

        trade = MomentumTrade(
            date=d, direction=direction,
            r1=si.r1, r17=si.r17,
            range_first30=si.range_first30, r16=si.r16,
            vol_first30=si.vol_first30,
            entry_ts=df.index[entry_pos], entry_price=ep,
            exit_ts=df.index[exit_pos], exit_price=xp,
            n_contracts=n_c, capital_at_entry=capital,
            atr_at_entry=b_atr, pnl_pts=pnl_pts, pnl_net=pnl_net,
        )
        capital += pnl_net
        trades.append(trade)

    return trades
