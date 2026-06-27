"""
Failed Opening Spike Reversion strategy engine — Phase 17.

HYPOTHESIS: The first 5 minutes of NQ trading frequently produce a spike that
sweeps stops before the real move begins. When that spike fails — price cannot
hold the extreme and begins reversing — trapped participants fuel a fast,
clean reversion.

Spike detection: bars 09:30–09:34 inclusive (5 bars).
Entry window:   bars 09:35–09:54 (first valid signal taken, no re-entry).
SL:             spike extreme ± 0.5 pts (fixed — does not move).
TP-A:           session_open (conservative).
TP-B:           opposite spike extreme (full reversion).
Sizing:         percentage of current capital, normalised by SL distance.
Slippage:       0.5 pts per side (2 ticks — wider due to open spreads).
Commission:     $4.00 round-trip per contract.

──────────────────────────────────────────────────────────────────────────────
HYPOTHESIS FALSIFIED — Phase 17

The reversion behavior is real: 71.3% of sessions with a qualifying
spike (>1.5×ATR) return to session_open within 60 bars (median 4 bars).

The tradeable edge is not: the entry sits between the spike extreme
(SL) and session_open (TP-A), creating structural negative R/R
regardless of parameter choices.

  TP-A (session_open):   WR 63.5%, avg R/R 0.39, expectancy -0.115
  TP-B (spike opposite): WR 52.2%, avg R/R 0.73, expectancy -0.099

Key insight: to trade this pattern profitably, the entry would need
to be BEFORE the spike forms (not after) — a prediction problem,
not a confirmation problem. Or the SL would need to be inside the
spike range, which creates a different and untested hypothesis.
──────────────────────────────────────────────────────────────────────────────
"""
import math
from dataclasses import dataclass
from datetime import time as dt_time
from typing import List, Optional

import numpy as np
import pandas as pd

SLIP   = 0.5    # pts per side
COMM   = 4.0    # $ round-trip per contract
PV     = 20.0   # NQ point value $/pt
SL_BUF = 0.5    # buffer beyond spike extreme

_SPIKE_S = dt_time(9, 30)
_SPIKE_E = dt_time(9, 34)
_ENTRY_S = dt_time(9, 35)
_ENTRY_E = dt_time(9, 54)
_EOD_T   = dt_time(15, 45)


@dataclass
class SpikeInfo:
    date: object
    session_open: float
    spike_high: float
    spike_low: float
    spike_magnitude: float      # max(up_move, down_move) in pts
    spike_direction: str        # "up" or "down"
    spike_extreme: float        # spike_high if up, spike_low if down
    opposite_extreme: float
    daily_atr: float            # 20-session rolling mean of RTH ranges
    atr_1min: float             # Wilder ATR(20) on 1-min bars at spike open
    n_bars_to_extreme: int      # 1–5
    spike_speed: float          # magnitude / n_bars_to_extreme  (pts/bar)
    prev_poc: float             # NaN if unavailable


@dataclass
class SpikeTrade:
    date: object
    direction: str              # "long" or "short"
    entry_ts: object
    entry_price: float
    sl_price: float
    tp_a: float                 # session_open
    tp_b: float                 # opposite spike extreme
    n_contracts: int
    capital_at_entry: float
    spike_magnitude: float
    daily_atr: float
    rr_a: float                 # abs(tp_a - entry) / abs(entry - sl)
    rr_b: float
    exit_ts: object = None
    exit_price: float = 0.0
    exit_reason: str = ""       # SL / TP / TIME / EOD
    pnl_pts: float = 0.0
    pnl_net: float = 0.0


def detect_spikes(
    df: pd.DataFrame,
    daily_atr_series: pd.Series,
    atr_1min_series: pd.Series,
    prev_poc_series: pd.Series,
) -> dict:
    """
    Compute SpikeInfo for every session date in df.
    Returns dict[date -> SpikeInfo | None].
    None means ambiguous (up_move == down_move) or no spike-window bars.
    """
    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    u_dates  = np.unique(date_arr)

    hi_v   = df["high"].values
    lo_v   = df["low"].values
    op_v   = df["open"].values
    da_v   = daily_atr_series.values
    a1_v   = atr_1min_series.values
    poc_v  = prev_poc_series.values

    result: dict = {}
    for d in u_dates:
        mask   = date_arr == d
        idxs   = all_pos[mask]
        times  = time_arr[mask]

        sp_mask = np.array([_SPIKE_S <= t <= _SPIKE_E for t in times])
        sp_idxs = idxs[sp_mask]
        if len(sp_idxs) == 0:
            result[d] = None
            continue

        s_open = float(op_v[sp_idxs[0]])
        s_hi   = float(hi_v[sp_idxs].max())
        s_lo   = float(lo_v[sp_idxs].min())
        up_mv  = s_hi - s_open
        dn_mv  = s_open - s_lo

        if up_mv == dn_mv:
            result[d] = None
            continue

        mag  = max(up_mv, dn_mv)
        dirn = "up" if up_mv > dn_mv else "down"

        if dirn == "up":
            extreme = s_hi
            opp     = s_lo
            n_bar   = int(np.argmax(hi_v[sp_idxs])) + 1
        else:
            extreme = s_lo
            opp     = s_hi
            n_bar   = int(np.argmin(lo_v[sp_idxs])) + 1

        da_val  = float(da_v[sp_idxs[0]])
        a1_val  = float(a1_v[sp_idxs[0]])
        poc_val = float(poc_v[sp_idxs[0]])

        result[d] = SpikeInfo(
            date=d,
            session_open=s_open,
            spike_high=s_hi,
            spike_low=s_lo,
            spike_magnitude=mag,
            spike_direction=dirn,
            spike_extreme=extreme,
            opposite_extreme=opp,
            daily_atr=da_val,
            atr_1min=a1_val,
            n_bars_to_extreme=n_bar,
            spike_speed=mag / n_bar,
            prev_poc=poc_val,
        )
    return result


def run(
    df: pd.DataFrame,
    spike_infos: dict,
    avg_vol_series: pd.Series,
    tp_variant: str = "A",
    spike_mult: float = 1.5,
    vol_mult: float = 1.3,
    poc_filter_pts: Optional[float] = None,
    speed_filter: Optional[str] = None,    # "fast" or "slow"
    speed_thresh_fast: float = 1.5,        # spike_speed > N × atr_1min
    speed_thresh_slow: float = 0.8,        # spike_speed < N × atr_1min
    max_bars: int = 60,
    initial_capital: float = 100_000.0,
    risk_pct: float = 1.0,
) -> List[SpikeTrade]:
    """
    Simulate the strategy on df.
    Capital accumulates across trades (percentage sizing updates each trade).
    One trade per session — first valid signal in 09:35–09:54 window.
    """
    trades: List[SpikeTrade] = []
    capital = float(initial_capital)

    date_arr  = np.array(df.index.date)
    time_arr  = np.array(df.index.time)
    all_pos   = np.arange(len(df))
    hi_v      = df["high"].values
    lo_v      = df["low"].values
    cl_v      = df["close"].values
    vol_v     = df["volume"].values
    avg_v     = avg_vol_series.values

    for d in np.unique(date_arr):
        si = spike_infos.get(d)
        if si is None:
            continue

        if np.isnan(si.daily_atr) or si.daily_atr <= 0:
            continue
        if np.isnan(si.atr_1min) or si.atr_1min <= 0:
            continue
        if si.spike_magnitude <= spike_mult * si.atr_1min:
            continue

        # Speed filter (uses 1-min ATR)
        if speed_filter == "fast":
            if np.isnan(si.atr_1min) or si.spike_speed <= speed_thresh_fast * si.atr_1min:
                continue
        elif speed_filter == "slow":
            if np.isnan(si.atr_1min) or si.spike_speed >= speed_thresh_slow * si.atr_1min:
                continue

        # POC confluence filter
        if poc_filter_pts is not None:
            if np.isnan(si.prev_poc):
                continue
            if abs(si.spike_extreme - si.prev_poc) >= poc_filter_pts:
                continue

        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]

        en_mask = np.array([_ENTRY_S <= t <= _ENTRY_E for t in times])
        en_idxs = idxs[en_mask]
        if len(en_idxs) == 0:
            continue

        trade: Optional[SpikeTrade] = None
        entry_sess_k = -1

        for pos in en_idxs:
            b_hi  = float(hi_v[pos])
            b_lo  = float(lo_v[pos])
            b_cl  = float(cl_v[pos])
            b_vol = float(vol_v[pos])
            b_avg = float(avg_v[pos])
            p_cl  = float(cl_v[pos - 1])

            if si.spike_direction == "down":
                # LONG: failed bearish spike
                if not (b_cl > p_cl):                               continue
                if not (b_cl > b_lo + 0.6 * (b_hi - b_lo)):        continue
                if not (b_vol > vol_mult * b_avg):                  continue
                if not (b_lo > si.spike_low):                       continue
                if not (b_cl < si.session_open):                    continue
                ep   = b_cl + SLIP
                sl   = si.spike_low - SL_BUF
                tp_a = si.session_open
                tp_b = si.spike_high
                dirn = "long"
            else:
                # SHORT: failed bullish spike
                if not (b_cl < p_cl):                               continue
                if not (b_cl < b_hi - 0.6 * (b_hi - b_lo)):        continue
                if not (b_vol > vol_mult * b_avg):                  continue
                if not (b_hi < si.spike_high):                      continue
                if not (b_cl > si.session_open):                    continue
                ep   = b_cl - SLIP
                sl   = si.spike_high + SL_BUF
                tp_a = si.session_open
                tp_b = si.spike_low
                dirn = "short"

            if dirn == "long" and (tp_a <= ep or tp_b <= ep):      continue
            if dirn == "short" and (tp_a >= ep or tp_b >= ep):     continue

            sl_pts   = abs(ep - sl)
            if sl_pts <= 0:                                         continue
            risk_usd = capital * risk_pct / 100.0
            n_c      = max(1, int(risk_usd / (sl_pts * PV)))
            rr_a     = abs(tp_a - ep) / sl_pts
            rr_b     = abs(tp_b - ep) / sl_pts

            sess_k   = int(np.where(idxs == pos)[0][0])
            trade    = SpikeTrade(
                date=d, direction=dirn, entry_ts=df.index[pos],
                entry_price=ep, sl_price=sl, tp_a=tp_a, tp_b=tp_b,
                n_contracts=n_c, capital_at_entry=capital,
                spike_magnitude=si.spike_magnitude, daily_atr=si.daily_atr,
                rr_a=rr_a, rr_b=rr_b,
            )
            entry_sess_k = sess_k
            break

        if trade is None:
            continue

        tp_price = trade.tp_a if tp_variant == "A" else trade.tp_b
        is_long  = (trade.direction == "long")
        post_idxs = idxs[entry_sess_k + 1:]

        for j, pos in enumerate(post_idxs):
            t_bar  = times[entry_sess_k + 1 + j]
            b_hi   = float(hi_v[pos])
            b_lo   = float(lo_v[pos])
            b_cl   = float(cl_v[pos])
            eod    = (t_bar >= _EOD_T)
            timout = (j + 1 >= max_bars)

            if is_long:
                if b_lo <= trade.sl_price:
                    ep_x, er = trade.sl_price - SLIP, "SL"
                elif b_hi >= tp_price:
                    ep_x, er = tp_price - SLIP, "TP"
                elif eod:
                    ep_x, er = b_cl - SLIP, "EOD"
                elif timout:
                    ep_x, er = b_cl - SLIP, "TIME"
                else:
                    continue
                pnl_pts = ep_x - trade.entry_price
            else:
                if b_hi >= trade.sl_price:
                    ep_x, er = trade.sl_price + SLIP, "SL"
                elif b_lo <= tp_price:
                    ep_x, er = tp_price + SLIP, "TP"
                elif eod:
                    ep_x, er = b_cl + SLIP, "EOD"
                elif timout:
                    ep_x, er = b_cl + SLIP, "TIME"
                else:
                    continue
                pnl_pts = trade.entry_price - ep_x

            pnl_net           = pnl_pts * trade.n_contracts * PV - COMM * trade.n_contracts
            trade.exit_ts     = df.index[pos]
            trade.exit_price  = ep_x
            trade.exit_reason = er
            trade.pnl_pts     = pnl_pts
            trade.pnl_net     = pnl_net
            capital          += pnl_net
            trades.append(trade)
            break

    return trades
