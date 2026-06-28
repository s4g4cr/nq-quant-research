"""
Phase 22B — Bearish Spike Fade strategy engine.

Entry: SHORT when price spikes to (1/3 × expected_spike) above open then rejects.
SL:    open + 0.662×prev_atr + 5 pts.
TP:    Four variants (A–D).

Signal: HMM transition matrix P(bearish | state_yesterday) > 0.60.
"""
from datetime import time as dt_time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from orb_system.strategy.hmm_transition import (
    compute_transition_matrix,
    predict_states,
)

# ── constants ────────────────────────────────────────────────────────────────

SLIP          = 0.25
COMM_RT       = 4.0     # $2/side × 2
PV            = 20.0
SPIKE_RATIO   = 0.662   # expected_spike = SPIKE_RATIO × prev_atr
SL_BUFFER     = 5.0     # pts above expected spike
VOL_MULT      = 1.2     # bar_vol > VOL_MULT × avg_bar_vol
P_BEAR_THRESH = 0.60
TRAIL_MULT    = 0.75    # trail at 0.75 × sl_dist from peak
TRAIL_ACT     = 1.0     # activate trail when profit >= 1.0 × sl_dist
BARS_PER_SESS = 375     # approx 1-min bars in 09:30-15:45 RTH window
SESS_OPEN     = dt_time(9,  30)
ENTRY_END     = dt_time(10, 30)
HARD_EXIT     = dt_time(15, 45)
MAX_BARS      = 180     # 3 hours from entry


# ── session params builder ────────────────────────────────────────────────────

def _add_params(out, row, p_bear, poc_per_date, avg_vol_per_date):
    d = row["date"]
    prev_atr = float(row["daily_atr"])
    if np.isnan(prev_atr) or prev_atr <= 0:
        return
    open_930 = float(row["open_930"])
    exp_spike = SPIKE_RATIO * prev_atr
    out[d] = {
        "signal_active": p_bear > P_BEAR_THRESH,
        "p_bearish":     round(float(p_bear), 4),
        "prev_atr":      round(prev_atr, 2),
        "open_930":      round(open_930, 2),
        "expected_spike": round(exp_spike, 2),
        "sl_price":      round(open_930 + exp_spike + SL_BUFFER, 2),
        "prev_poc":      poc_per_date.get(d, np.nan),
        "avg_vol":       avg_vol_per_date.get(d, np.nan),
    }


def compute_session_params(
    feat_all: pd.DataFrame,
    poc_per_date: dict,
    avg_vol_per_date: dict,
    model,
    states_tr: np.ndarray,
    lmap: Dict[int, str],
    n_states: int,
    train_end_str: str,
) -> dict:
    """
    Build per-session signal & price params for ALL dates in feat_all.

    Training dates: use states_tr for the Viterbi sequence.
    Test dates:     run predict_states(model, X_te) for Viterbi, then
                    compute P(bearish | yesterday's state) via transition T.
    The first test day's 'yesterday' = last training day's state.
    """
    inv = {v: k for k, v in lmap.items()}
    T, _ = compute_transition_matrix(states_tr, n_states)
    bear_s = inv["bearish"]

    fa = feat_all.sort_values("date").reset_index(drop=True)
    is_tr = fa["date"].astype(str) <= train_end_str
    fa_tr = fa[is_tr].reset_index(drop=True)
    fa_te = fa[~is_tr].reset_index(drop=True)

    X_te      = fa_te[["daily_return", "volume_ratio"]].values
    states_te = predict_states(model, X_te) if len(X_te) > 0 else np.array([], int)

    params: dict = {}

    for i in range(1, len(fa_tr)):
        p = float(T[states_tr[i - 1], bear_s])
        _add_params(params, fa_tr.iloc[i], p, poc_per_date, avg_vol_per_date)

    for i in range(len(fa_te)):
        state_yest = states_tr[-1] if i == 0 else states_te[i - 1]
        p = float(T[state_yest, bear_s])
        _add_params(params, fa_te.iloc[i], p, poc_per_date, avg_vol_per_date)

    return params


# ── backtest engine ───────────────────────────────────────────────────────────

def run_backtest(
    df_raw,
    time_arr: np.ndarray,
    date_idx_map: dict,
    session_params: dict,
    trade_dates,
    tp_variant: str,       # "A" | "B" | "C" | "D"
    tp_x: Optional[float], # X for TP-B (1.0, 1.5, 2.0); None otherwise
    entry_frac: float,     # fraction of expected_spike for entry trigger (default 1/3)
    initial_capital: float,
    risk_pct: float,
    theoretical: bool = False,  # True → no costs, 1 contract
) -> List[dict]:
    """
    Simulate SHORT entries on each date in trade_dates.
    Returns list of trade dicts.
    """
    slip = 0.0 if theoretical else SLIP
    comm = 0.0 if theoretical else COMM_RT

    high_v  = df_raw["high"].values
    low_v   = df_raw["low"].values
    close_v = df_raw["close"].values
    open_v  = df_raw["open"].values
    vol_v   = df_raw["volume"].values

    capital = float(initial_capital)
    trades: List[dict] = []

    for d in sorted(trade_dates):
        p = session_params.get(d)
        if p is None or not p["signal_active"]:
            continue
        if d not in date_idx_map:
            continue

        abs_idx  = date_idx_map[d]
        d_times  = time_arr[abs_idx]
        rth_sel  = (d_times >= SESS_OPEN) & (d_times <= HARD_EXIT)
        rth_abs  = abs_idx[rth_sel]
        rth_t    = d_times[rth_sel]
        if len(rth_abs) == 0:
            continue

        open_930    = p["open_930"]
        exp_spike   = p["expected_spike"]
        sl_price    = p["sl_price"]
        prev_atr    = p["prev_atr"]
        prev_poc    = p["prev_poc"]
        avg_vol     = p["avg_vol"]

        entry_trigger = open_930 + entry_frac * exp_spike
        sl_dist_sz    = sl_price - entry_trigger   # for sizing (pre-computed, ≥ 0)

        # TP-D: skip session if prev_poc not below trigger (causal check)
        if tp_variant == "D":
            if np.isnan(prev_poc) or prev_poc >= entry_trigger:
                continue
            tp_price_sess = float(prev_poc)
        elif tp_variant == "B":
            tp_price_sess = open_930 - tp_x * prev_atr
        else:
            tp_price_sess = None   # computed after entry (A) or trailing (C)

        bar_vol_thresh = (avg_vol / BARS_PER_SESS * VOL_MULT
                         if not np.isnan(avg_vol) and avg_vol > 0 else 0.0)

        # ── entry scan 09:30 – 10:30 ─────────────────────────────────────
        entry_price  = None
        entry_j      = None
        entry_time   = None

        for j, aj in enumerate(rth_abs):
            bt = rth_t[j]
            if bt > ENTRY_END:
                break
            bh = high_v[aj]; bl = low_v[aj]
            bc = close_v[aj]; bo = open_v[aj]
            bv = vol_v[aj]

            if (bh >= entry_trigger        # touched trigger
                    and bc < bo            # bearish bar
                    and bc < entry_trigger # closed below trigger
                    and bv > bar_vol_thresh):
                entry_price = bc - slip
                entry_j     = j
                entry_time  = bt
                break

        if entry_price is None:
            continue

        # ── position sizing ───────────────────────────────────────────────
        if theoretical:
            n_contracts = 1
        else:
            risk_usd    = capital * risk_pct / 100.0
            n_contracts = max(1, int(risk_usd / (sl_dist_sz * PV)))

        # Finalise TP for A (needs actual entry price)
        if tp_variant == "A":
            tp_price_sess = entry_price - 2.0 * sl_dist_sz

        # ── trade management ──────────────────────────────────────────────
        exit_price   = None
        exit_reason  = None
        exit_time    = None
        peak_fav     = entry_price   # running min low for SHORT (favorable = down)
        trail_active = False
        bars_in      = 0

        for j2 in range(entry_j + 1, len(rth_abs)):
            aj2 = rth_abs[j2]
            bt2 = rth_t[j2]
            bh2 = high_v[aj2]
            bl2 = low_v[aj2]
            bc2 = close_v[aj2]
            bars_in += 1

            # 1. SL (checked first; SL wins if same bar as TP)
            if bh2 >= sl_price:
                exit_price  = sl_price + slip
                exit_reason = "SL"
                exit_time   = bt2
                break

            # 2. Update peak favorable (track lowest low)
            if bl2 < peak_fav:
                peak_fav = bl2

            # 3. TP-C trailing stop
            if tp_variant == "C":
                profit_pts = entry_price - peak_fav
                if not trail_active and profit_pts >= TRAIL_ACT * sl_dist_sz:
                    trail_active = True
                if trail_active:
                    trail_stop = peak_fav + TRAIL_MULT * sl_dist_sz
                    if bh2 >= trail_stop:
                        exit_price  = trail_stop + slip
                        exit_reason = "TRAIL"
                        exit_time   = bt2
                        break
            else:
                # 4. Fixed TP (A, B, D)
                if tp_price_sess is not None and bl2 <= tp_price_sess:
                    exit_price  = tp_price_sess + slip
                    exit_reason = "TP"
                    exit_time   = bt2
                    break

            # 5. Time / EOD limit
            if bars_in >= MAX_BARS or bt2 >= HARD_EXIT:
                exit_price  = bc2 + slip
                exit_reason = "EOD" if bt2 >= HARD_EXIT else "TIME"
                exit_time   = bt2
                break

        # If session ended without exit
        if exit_price is None and len(rth_abs) > entry_j + 1:
            last = len(rth_abs) - 1
            exit_price  = close_v[rth_abs[last]] + slip
            exit_reason = "EOD"
            exit_time   = rth_t[last]
        elif exit_price is None:
            continue

        # ── P&L ──────────────────────────────────────────────────────────
        pnl_pts = entry_price - exit_price          # positive = gain (SHORT)
        pnl_usd = pnl_pts * PV * n_contracts - comm * n_contracts
        capital += pnl_usd

        trades.append({
            "date":          str(d),
            "year":          d.year,
            "entry_time":    str(entry_time),
            "exit_time":     str(exit_time),
            "entry_price":   round(entry_price, 2),
            "sl_price":      round(sl_price, 2),
            "tp_price":      (round(tp_price_sess, 2)
                              if tp_price_sess is not None else None),
            "exit_price":    round(exit_price, 2),
            "exit_reason":   exit_reason,
            "n_contracts":   n_contracts,
            "pnl_pts":       round(pnl_pts, 2),
            "pnl_usd":       round(pnl_usd, 2),
            "capital_after": round(capital, 2),
            "prev_atr":      round(prev_atr, 2),
            "sl_dist":       round(sl_dist_sz, 2),
            "rr_realized":   (round(pnl_pts / sl_dist_sz, 3)
                              if sl_dist_sz > 0 else np.nan),
            "p_bearish":     p["p_bearish"],
            "tp_variant":    f"{tp_variant}" + (f"_{tp_x}" if tp_x else ""),
            "entry_frac":    round(entry_frac, 4),
        })

    return trades
