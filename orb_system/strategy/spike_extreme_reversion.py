"""
Opening Spike Extreme as Support/Resistance — Phase 18.

HYPOTHESIS: The opening spike extreme (spike_low for bearish spikes, spike_high
for bullish spikes) acts as a structural S/R level for the first 2 hours.
Each time price returns to within 0.5×ATR of the spike extreme and shows
rejection, there is a reversion opportunity toward session_open and beyond.

Key fix from Phase 17: SL is ATR-based, not spike-extreme-based.
  Phase 17 failed because entry sat between SL (spike extreme) and TP,
  making SL always farther than TP. Here, SL = 1.0×ATR_1min at entry.
  As long as entry is > 1 ATR below session_open, TP-A has R/R > 1.0.

Spike detection:  09:30–09:34 (first 5 bars, ties go to "up").
Entry window:     09:35–11:30 · max 2 entries per session (sequential).
SL:               entry ± 1.0×ATR_1min at entry bar (fixed, never moves).
TP-A:             session_open.
TP-B:             session_open ± 0.5×spike_magnitude (extended).
TP-C:             opposite spike extreme (full reversion).
Hard R/R gate:    skip if (tp − entry) / sl_pts < 1.0.
Trailing stop:    activates at 1.0×ATR profit, trails at 0.75×ATR from peak.
Slippage:         0.5 pts per side (2 ticks).
Commission:       $4.00 round-trip per contract ($2/side).

──────────────────────────────────────────────────────────────────────────────
HYPOTHESIS FALSIFIED — Phase 18

The ATR-based SL fix resolved Phase 17's geometry problem. The signal has
valid R/R (median 1.78–3.09 depending on TP variant) and marginally
positive theoretical expectancy (TP-A: +0.022, TP-B: +0.018).

The tradeable edge is not sustained OOS: win rate collapsed 12pp from
train to test period, producing negative PF across all configurations.

  Diagnostic (full dataset):  93.3% of qualifying sessions retest the
                               spike extreme. 81.7% show bar rejection.
                               Theoretical WR: 32.4% (TP-A, costs excluded).

  Train (2021–2024): PF=1.044  WR=35.7%  SR=0.344  (Exp 1 baseline)
  Test  (2025–2026): PF=0.733  WR=23.4%  SR=-2.370 (Exp 1 baseline)
  Best combined:     PF=0.915  WR=21.1%  SR=-0.673 (Exp 9, spike>3×ATR, R/R>2)

Key insight: The spike extreme acts as a visible S/R magnet — which is
precisely why the market tests and rejects it frequently. But the rejection
bar entry fires mostly on bars that subsequently fail to follow through,
suggesting the signal captures real structure but the entry timing (at the
rejection candle close) is still one step too late. In 2025, with higher
volatility and wider ATRs, entries require a larger SL which drags WR down
further relative to the fixed TP at session_open.

Parameters tested: 9 experiments across TP variants (A/B/C), spike_mult
[1.5, 2.0, 2.5, 3.0], trailing stop, quality filter, POC confluence,
and min_rr [1.0, 1.5, 2.0]. No configuration cleared PF > 1.0 OOS.
──────────────────────────────────────────────────────────────────────────────
"""
from dataclasses import dataclass
from datetime import time as dt_time
from typing import List, Optional

import numpy as np
import pandas as pd

SLIP  = 0.5    # pts per side
COMM  = 4.0    # $ round-trip per contract
PV    = 20.0   # NQ point value $/pt

_SPIKE_S = dt_time(9, 30)
_SPIKE_E = dt_time(9, 34)
_ENTRY_S = dt_time(9, 35)
_ENTRY_E = dt_time(11, 30)
_EOD_T   = dt_time(15, 45)


@dataclass
class SpikeInfo:
    date: object
    session_open: float
    spike_high: float
    spike_low: float
    spike_magnitude: float
    spike_direction: str      # "up" (up_move >= down_move) or "down"
    spike_extreme: float      # spike_high if up, spike_low if down
    opposite_extreme: float
    atr_spike: float          # ATR_1min at 09:30 bar


@dataclass
class SpikeTrade:
    date: object
    direction: str            # "long" or "short"
    entry_num: int            # 1 or 2 (which entry in the session)
    entry_ts: object
    entry_price: float
    sl_price: float
    sl_pts: float             # 1.0 × ATR_1min at entry bar
    tp_a: float
    tp_b: float
    tp_c: float
    tp_used: float
    n_contracts: int
    capital_at_entry: float
    spike_magnitude: float
    atr_at_entry: float
    rr_a: float
    rr_b: float
    rr_c: float
    rr_used: float
    exit_ts: object = None
    exit_price: float = 0.0
    exit_reason: str = ""     # SL / TP / TRAIL / TIME / EOD
    pnl_pts: float = 0.0
    pnl_net: float = 0.0


def detect_spikes(
    df: pd.DataFrame,
    atr_1min_series: pd.Series,
) -> dict:
    """
    Compute SpikeInfo for every session date in df.
    Returns dict[date -> SpikeInfo | None].
    spike_direction = "up" if up_move >= down_move (ties to "up").
    No magnitude filter applied here — applied in run().
    """
    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    u_dates  = np.unique(date_arr)

    hi_v = df["high"].values
    lo_v = df["low"].values
    op_v = df["open"].values
    a1_v = atr_1min_series.values

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
        mag    = max(up_mv, dn_mv)
        dirn   = "up" if up_mv >= dn_mv else "down"
        extreme = s_hi if dirn == "up" else s_lo
        opp     = s_lo if dirn == "up" else s_hi

        result[d] = SpikeInfo(
            date=d,
            session_open=s_open,
            spike_high=s_hi,
            spike_low=s_lo,
            spike_magnitude=mag,
            spike_direction=dirn,
            spike_extreme=extreme,
            opposite_extreme=opp,
            atr_spike=float(a1_v[sp_idxs[0]]),
        )
    return result


def run(
    df: pd.DataFrame,
    spike_infos: dict,
    avg_vol_series: pd.Series,
    atr_1min_series: pd.Series,
    prev_poc_series: Optional[pd.Series] = None,
    tp_variant: str = "A",
    spike_mult: float = 1.5,
    vol_mult: float = 1.2,
    retest_zone: float = 0.5,
    min_rr: float = 1.0,
    use_trailing: bool = False,
    trail_activate: float = 1.0,
    trail_dist: float = 0.75,
    poc_filter_pts: Optional[float] = None,
    retest_quality: bool = False,
    retest_range_mult: float = 0.8,
    retest_vol_mult: float = 1.5,
    max_bars: int = 90,
    max_entries: int = 2,
    initial_capital: float = 100_000.0,
    risk_pct: float = 1.0,
) -> List[SpikeTrade]:
    """
    Simulate Phase 18 strategy on df.
    Capital accumulates across trades. Up to max_entries per session,
    sequential (second entry only after first closes).
    """
    trades: List[SpikeTrade] = []
    capital = float(initial_capital)

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    hi_v     = df["high"].values
    lo_v     = df["low"].values
    cl_v     = df["close"].values
    op_v     = df["open"].values
    vol_v    = df["volume"].values
    avg_v    = avg_vol_series.values
    a1_v     = atr_1min_series.values
    poc_v    = prev_poc_series.values if prev_poc_series is not None else None

    for d in np.unique(date_arr):
        si = spike_infos.get(d)
        if si is None:
            continue
        if np.isnan(si.atr_spike) or si.atr_spike <= 0:
            continue
        if si.spike_magnitude <= spike_mult * si.atr_spike:
            continue

        # POC confluence on spike_extreme
        if poc_filter_pts is not None:
            m0    = date_arr == d
            i0    = all_pos[m0]
            if poc_v is None or np.isnan(poc_v[i0[0]]):
                continue
            if abs(si.spike_extreme - poc_v[i0[0]]) >= poc_filter_pts:
                continue

        mask  = date_arr == d
        idxs  = all_pos[mask]
        times = time_arr[mask]

        is_long      = (si.spike_direction == "down")
        entry_count  = 0
        open_trade: Optional[SpikeTrade] = None
        peak_fav     = 0.0
        trail_active = False
        bars_open    = 0

        for k, pos in enumerate(idxs):
            t_bar = times[k]
            b_hi  = float(hi_v[pos])
            b_lo  = float(lo_v[pos])
            b_cl  = float(cl_v[pos])
            b_op  = float(op_v[pos])
            b_vol = float(vol_v[pos])
            b_atr = float(a1_v[pos])
            b_avg = float(avg_v[pos])
            eod   = (t_bar >= _EOD_T)

            # ── Exit ───────────────────────────────────────────────────────────
            if open_trade is not None:
                bars_open += 1
                timout     = (bars_open >= max_bars)

                # Update trailing peak
                if use_trailing:
                    if is_long:
                        peak_fav = max(peak_fav, b_hi)
                        if not trail_active:
                            if (peak_fav - open_trade.entry_price) >= trail_activate * open_trade.sl_pts:
                                trail_active = True
                        t_stop = (peak_fav - trail_dist * open_trade.sl_pts) if trail_active else None
                    else:
                        peak_fav = min(peak_fav, b_lo)
                        if not trail_active:
                            if (open_trade.entry_price - peak_fav) >= trail_activate * open_trade.sl_pts:
                                trail_active = True
                        t_stop = (peak_fav + trail_dist * open_trade.sl_pts) if trail_active else None

                # Determine exit — SL checked first (wins on same-bar tie)
                ep_x = er = None
                if is_long:
                    if b_lo <= open_trade.sl_price:
                        ep_x = open_trade.sl_price - SLIP; er = "SL"
                    elif use_trailing and trail_active and t_stop is not None and b_lo <= t_stop:
                        ep_x = t_stop - SLIP; er = "TRAIL"
                    elif not use_trailing and b_hi >= open_trade.tp_used:
                        ep_x = open_trade.tp_used - SLIP; er = "TP"
                    elif eod:
                        ep_x = b_cl - SLIP; er = "EOD"
                    elif timout:
                        ep_x = b_cl - SLIP; er = "TIME"
                else:
                    if b_hi >= open_trade.sl_price:
                        ep_x = open_trade.sl_price + SLIP; er = "SL"
                    elif use_trailing and trail_active and t_stop is not None and b_hi >= t_stop:
                        ep_x = t_stop + SLIP; er = "TRAIL"
                    elif not use_trailing and b_lo <= open_trade.tp_used:
                        ep_x = open_trade.tp_used + SLIP; er = "TP"
                    elif eod:
                        ep_x = b_cl + SLIP; er = "EOD"
                    elif timout:
                        ep_x = b_cl + SLIP; er = "TIME"

                if ep_x is not None:
                    pnl_pts = (ep_x - open_trade.entry_price) if is_long \
                              else (open_trade.entry_price - ep_x)
                    pnl_net = pnl_pts * open_trade.n_contracts * PV \
                              - COMM * open_trade.n_contracts
                    open_trade.exit_ts     = df.index[pos]
                    open_trade.exit_price  = ep_x
                    open_trade.exit_reason = er
                    open_trade.pnl_pts     = pnl_pts
                    open_trade.pnl_net     = pnl_net
                    capital               += pnl_net
                    trades.append(open_trade)
                    open_trade   = None
                    peak_fav     = 0.0
                    trail_active = False
                    bars_open    = 0

            # ── Entry ──────────────────────────────────────────────────────────
            if (open_trade is None
                    and entry_count < max_entries
                    and _ENTRY_S <= t_bar <= _ENTRY_E
                    and not eod):

                if np.isnan(b_atr) or b_atr <= 0 or np.isnan(b_avg) or b_avg <= 0:
                    continue
                if (b_hi - b_lo) == 0:
                    continue  # zero-range bar — skip

                if is_long:
                    if not (b_lo <= si.spike_low + retest_zone * b_atr):  continue
                    if not (b_cl > si.spike_low):                          continue
                    if not (b_cl > b_op):                                  continue
                    if not (b_cl > b_lo + 0.5 * (b_hi - b_lo)):           continue
                    if not (b_vol > vol_mult * b_avg):                     continue
                    if not (b_cl < si.session_open):                       continue
                    if retest_quality:
                        if (b_hi - b_lo) < retest_range_mult * b_atr:     continue
                        if b_vol <= retest_vol_mult * b_avg:               continue
                    ep   = b_cl + SLIP
                    sl   = ep - b_atr
                    tp_a = si.session_open
                    tp_b = si.session_open + 0.5 * (si.session_open - si.spike_low)
                    tp_c = si.spike_high
                else:
                    if not (b_hi >= si.spike_high - retest_zone * b_atr):  continue
                    if not (b_cl < si.spike_high):                          continue
                    if not (b_cl < b_op):                                   continue
                    if not (b_cl < b_hi - 0.5 * (b_hi - b_lo)):            continue
                    if not (b_vol > vol_mult * b_avg):                      continue
                    if not (b_cl > si.session_open):                        continue
                    if retest_quality:
                        if (b_hi - b_lo) < retest_range_mult * b_atr:      continue
                        if b_vol <= retest_vol_mult * b_avg:                continue
                    ep   = b_cl - SLIP
                    sl   = ep + b_atr
                    tp_a = si.session_open
                    tp_b = si.session_open - 0.5 * (si.spike_high - si.session_open)
                    tp_c = si.spike_low

                sl_pts = b_atr

                # Select and validate TP
                tp_used = {"A": tp_a, "B": tp_b, "C": tp_c}.get(tp_variant, tp_a)
                if is_long:
                    if tp_used <= ep:  continue
                    rr_used = (tp_used - ep) / sl_pts
                    rr_a    = (tp_a - ep) / sl_pts
                    rr_b    = (tp_b - ep) / sl_pts
                    rr_c    = (tp_c - ep) / sl_pts
                else:
                    if tp_used >= ep:  continue
                    rr_used = (ep - tp_used) / sl_pts
                    rr_a    = (ep - tp_a) / sl_pts
                    rr_b    = (ep - tp_b) / sl_pts
                    rr_c    = (ep - tp_c) / sl_pts

                if rr_used < min_rr:
                    continue

                risk_usd = capital * risk_pct / 100.0
                n_c      = max(1, int(risk_usd / (sl_pts * PV)))

                open_trade = SpikeTrade(
                    date=d,
                    direction="long" if is_long else "short",
                    entry_num=entry_count + 1,
                    entry_ts=df.index[pos],
                    entry_price=ep, sl_price=sl, sl_pts=sl_pts,
                    tp_a=tp_a, tp_b=tp_b, tp_c=tp_c, tp_used=tp_used,
                    n_contracts=n_c, capital_at_entry=capital,
                    spike_magnitude=si.spike_magnitude,
                    atr_at_entry=b_atr,
                    rr_a=rr_a, rr_b=rr_b, rr_c=rr_c, rr_used=rr_used,
                )
                peak_fav   = open_trade.entry_price  # initialise; updated from next bar
                entry_count += 1
                bars_open    = 0

    return trades
