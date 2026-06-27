#!/usr/bin/env python3
"""
Phase 17 — Failed Opening Spike Reversion.

Mandatory pre-PnL diagnostic runs first. Experiments only execute if
theoretical expectancy is positive.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import atr as compute_atr, avg_volume
from orb_system.indicators.volume_profile import compute_poc_features
from orb_system.strategy.failed_spike import (
    SpikeInfo, SpikeTrade, detect_spikes, run,
    SLIP, COMM, PV, SL_BUF, _SPIKE_S, _SPIKE_E, _ENTRY_S, _ENTRY_E, _EOD_T,
)

# ── constants ─────────────────────────────────────────────────────────────────
INITIAL_CAP  = 100_000.0
TRAIN_END    = "2024-12-31"
TEST_START   = "2025-01-01"
W = 68
RESULTS_DIR  = os.path.join(ROOT, "results")

# WFO windows
WFO_WINDOWS = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _pf(trades):
    v = np.array([t.pnl_net for t in trades])
    w = v[v > 0]; l = v[v <= 0]
    gw = float(w.sum()) if w.size else 0.0
    gl = float(abs(l.sum())) if l.size else 0.0
    return gw / gl if gl > 0 else float("inf")

def _sr(trades):
    if not trades:
        return 0.0
    by_date: dict = {}
    for t in trades:
        d = str(t.date)
        by_date[d] = by_date.get(d, 0.0) + t.pnl_net
    v = np.array(list(by_date.values()))
    s = float(v.std())
    return float(v.mean() / s * math.sqrt(252)) if s > 0 else 0.0

def _wr(trades):
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl_net > 0) / len(trades)

def _max_dd(trades):
    if not trades:
        return 0.0
    cap = INITIAL_CAP
    peak = cap
    worst = 0.0
    for t in trades:
        cap += t.pnl_net
        peak = max(peak, cap)
        dd = (peak - cap) / peak
        worst = max(worst, dd)
    return worst

def _tpd(trades, df):
    if not trades:
        return 0.0
    n_days = len(np.unique(np.array(df.index.date)))
    return len(trades) / n_days if n_days > 0 else 0.0

def _ret(trades):
    return sum(t.pnl_net for t in trades) / INITIAL_CAP * 100.0

def _exits(trades):
    n = len(trades)
    if n == 0:
        return 0, 0, 0, 0
    sl  = sum(1 for t in trades if t.exit_reason == "SL")
    tp  = sum(1 for t in trades if t.exit_reason == "TP")
    tm  = sum(1 for t in trades if t.exit_reason == "TIME")
    eod = sum(1 for t in trades if t.exit_reason == "EOD")
    return sl * 100 / n, tp * 100 / n, tm * 100 / n, eod * 100 / n

def _avg_w_l(trades):
    wins  = [t.pnl_net for t in trades if t.pnl_net > 0]
    loses = [t.pnl_net for t in trades if t.pnl_net <= 0]
    aw = float(np.mean(wins))  if wins  else 0.0
    al = float(np.mean(loses)) if loses else 0.0
    wp = float(np.mean([t.pnl_pts for t in trades if t.pnl_net > 0])) if wins else 0.0
    lp = float(np.mean([t.pnl_pts for t in trades if t.pnl_net <= 0])) if loses else 0.0
    rr = abs(aw / al) if al != 0 else float("inf")
    return aw, wp, al, lp, rr

def _nc(trades):
    return float(np.mean([t.n_contracts for t in trades])) if trades else 0.0

def _annual(trades, df):
    by_year: dict = {}
    for t in trades:
        y = pd.Timestamp(str(t.date)).year
        by_year.setdefault(y, []).append(t)
    years = sorted(by_year.keys())
    rows = []
    for y in years:
        tt = by_year[y]
        n = len(tt)
        wr = _wr(tt) * 100
        pf = _pf(tt)
        ret = sum(t.pnl_net for t in tt) / INITIAL_CAP * 100
        rows.append((y, n, wr, pf, ret))
    return rows

def _fmt(v, d=3):
    if v != v or v == float("inf"):
        return "  inf"
    return f"{v:.{d}f}"

def _pct(v):
    return f"{v:.1f}%"


def compute_daily_atr(df: pd.DataFrame) -> pd.Series:
    """
    20-session rolling mean of RTH session ranges, strictly causal.
    Replicates det_regime_v2.py approach. Returns a Series aligned to df.index.
    """
    from datetime import time as dt_time
    _RTH_S = dt_time(9, 30)
    _RTH_E = dt_time(15, 45)
    WINDOW = 20

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    u_dates  = np.unique(date_arr)
    all_pos  = np.arange(len(df))
    hi_v     = df["high"].values
    lo_v     = df["low"].values

    sess: dict = {}
    for d in u_dates:
        mask     = date_arr == d
        idxs     = all_pos[mask]
        times    = time_arr[mask]
        rth_mask = np.array([_RTH_S <= t <= _RTH_E for t in times])
        if rth_mask.sum() == 0:
            continue
        ri = idxs[rth_mask]
        sess[d] = float(hi_v[ri].max()) - float(lo_v[ri].min())

    u = sorted(sess.keys())
    n = len(u)
    ranges = np.array([sess[d] for d in u])

    da_by_date: dict = {}
    for i in range(n):
        if i >= WINDOW:
            da_by_date[u[i]] = float(np.mean(ranges[i - WINDOW: i]))
        else:
            da_by_date[u[i]] = np.nan

    da_arr = np.array([da_by_date.get(d, np.nan) for d in date_arr])
    return pd.Series(da_arr, index=df.index, name="daily_atr")


# ── diagnostic ────────────────────────────────────────────────────────────────

def run_diagnostic(
    df: pd.DataFrame,
    spike_infos: dict,
    avg_vol_series: pd.Series,
    baseline_mult: float = 1.5,
    vol_mult: float = 1.3,
) -> bool:
    """
    Print mandatory pre-PnL diagnostics.
    Returns True if theoretical expectancy > 0.
    """
    from datetime import time as dt_time

    date_arr = np.array(df.index.date)
    time_arr = np.array(df.index.time)
    all_pos  = np.arange(len(df))
    hi_v     = df["high"].values
    lo_v     = df["low"].values
    cl_v     = df["close"].values
    vol_v    = df["volume"].values
    avg_v    = avg_vol_series.values

    valid_infos = [si for si in spike_infos.values() if si is not None]
    total_sessions = len([d for d, si in spike_infos.items() if si is not None or si is None])
    has_spike = len(valid_infos)  # sessions with non-ambiguous spike info

    # ── 1. Spike frequency & size ─────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  DIAGNOSTIC — PRE-PnL STATISTICS (full dataset, train + test)")
    print(f"{'─'*W}")

    ratio_vals = [si.spike_magnitude / si.atr_1min
                  for si in valid_infos
                  if not np.isnan(si.atr_1min) and si.atr_1min > 0]

    q15 = [r for r in ratio_vals if r > 1.5]
    q20 = [r for r in ratio_vals if r > 2.0]

    print(f"\n  1. SPIKE FREQUENCY & SIZE  (sessions with valid spike info: {has_spike})")
    print(f"     Sessions with spike > 1.5 × atr_1min : {len(q15):4d} ({100*len(q15)/max(1,len(ratio_vals)):.1f}%)")
    print(f"     Sessions with spike > 2.0 × atr_1min : {len(q20):4d} ({100*len(q20)/max(1,len(ratio_vals)):.1f}%)")

    pcts = [10, 25, 50, 75, 90]
    if ratio_vals:
        percs = np.percentile(ratio_vals, pcts)
        print(f"     Spike magnitude / atr_1min percentiles:")
        print(f"       p10={percs[0]:.3f}  p25={percs[1]:.3f}  p50={percs[2]:.3f}"
              f"  p75={percs[3]:.3f}  p90={percs[4]:.3f}")

    up_ct  = sum(1 for si in valid_infos if si.spike_direction == "up")
    dn_ct  = sum(1 for si in valid_infos if si.spike_direction == "down")
    tot    = up_ct + dn_ct
    print(f"     Spike direction: up={up_ct} ({100*up_ct/max(1,tot):.1f}%)  "
          f"down={dn_ct} ({100*dn_ct/max(1,tot):.1f}%)")

    # ── 2. Reversal rates (no SL applied) ─────────────────────────────────────
    print(f"\n  2. REVERSAL RATE (no SL, spike > {baseline_mult} × atr_1min)")
    qualifying = [si for si in valid_infos
                  if not np.isnan(si.atr_1min) and si.atr_1min > 0
                  and si.spike_magnitude > baseline_mult * si.atr_1min]
    print(f"     Qualifying sessions: {len(qualifying)}")

    rev30 = rev60 = opp60 = bars_list = 0
    bars_to_rev = []
    if qualifying:
        rev30_ct = rev60_ct = opp60_ct = 0
        for si in qualifying:
            mask  = date_arr == si.date
            idxs  = all_pos[mask]
            times = time_arr[mask]

            post_mask = np.array([t > _SPIKE_E for t in times])
            post_idxs = idxs[post_mask]

            is_long = (si.spike_direction == "down")
            rev_bar = None
            opp_bar = None
            for j, pos in enumerate(post_idxs):
                if is_long:
                    if j < 30 and hi_v[pos] >= si.session_open:
                        if rev_bar is None:
                            rev_bar = j + 1
                    if j < 60 and hi_v[pos] >= si.session_open:
                        pass  # handled below
                    if j < 60 and hi_v[pos] >= si.spike_high:
                        if opp_bar is None:
                            opp_bar = j + 1
                else:
                    if j < 30 and lo_v[pos] <= si.session_open:
                        if rev_bar is None:
                            rev_bar = j + 1
                    if j < 60 and lo_v[pos] <= si.spike_low:
                        if opp_bar is None:
                            opp_bar = j + 1

            # recheck 60-bar window for session_open
            rev60_bar = None
            for j, pos in enumerate(post_idxs[:60]):
                if is_long:
                    if hi_v[pos] >= si.session_open:
                        rev60_bar = j + 1
                        break
                else:
                    if lo_v[pos] <= si.session_open:
                        rev60_bar = j + 1
                        break

            if rev_bar is not None:
                rev30_ct += 1
            if rev60_bar is not None:
                rev60_ct += 1
                bars_to_rev.append(rev60_bar)
            if opp_bar is not None:
                opp60_ct += 1

        n_q = max(1, len(qualifying))
        print(f"     Return to session_open within 30 bars: "
              f"{rev30_ct} ({100*rev30_ct/n_q:.1f}%)")
        print(f"     Return to session_open within 60 bars: "
              f"{rev60_ct} ({100*rev60_ct/n_q:.1f}%)")
        print(f"     Reach opposite extreme within 60 bars: "
              f"{opp60_ct} ({100*opp60_ct/n_q:.1f}%)")
        if bars_to_rev:
            print(f"     Median bars to session_open (of those reverting): "
                  f"{np.median(bars_to_rev):.0f}")

    # ── 3. Signal frequency after all entry filters ───────────────────────────
    print(f"\n  3. SIGNAL FREQUENCY (all entry filters, spike_mult={baseline_mult} × atr_1min)")
    sig_sessions = sig_count = 0
    signal_data = []  # (si, entry_bar_close, sl, tp_a, tp_b, pos_for_path)

    for si in qualifying:
        mask  = date_arr == si.date
        idxs  = all_pos[mask]
        times = time_arr[mask]

        en_mask = np.array([_ENTRY_S <= t <= _ENTRY_E for t in times])
        en_idxs = idxs[en_mask]
        found = False
        for pos in en_idxs:
            b_hi  = float(hi_v[pos])
            b_lo  = float(lo_v[pos])
            b_cl  = float(cl_v[pos])
            b_vol = float(vol_v[pos])
            b_avg = float(avg_v[pos])
            p_cl  = float(cl_v[pos - 1])

            if si.spike_direction == "down":
                if not (b_cl > p_cl):                               continue
                if not (b_cl > b_lo + 0.6 * (b_hi - b_lo)):        continue
                if not (b_vol > vol_mult * b_avg):                  continue
                if not (b_lo > si.spike_low):                       continue
                if not (b_cl < si.session_open):                    continue
                ep   = b_cl
                sl   = si.spike_low - SL_BUF
                tp_a = si.session_open
                tp_b = si.spike_high
            else:
                if not (b_cl < p_cl):                               continue
                if not (b_cl < b_hi - 0.6 * (b_hi - b_lo)):        continue
                if not (b_vol > vol_mult * b_avg):                  continue
                if not (b_hi < si.spike_high):                      continue
                if not (b_cl > si.session_open):                    continue
                ep   = b_cl
                sl   = si.spike_high + SL_BUF
                tp_a = si.session_open
                tp_b = si.spike_low

            is_long = (si.spike_direction == "down")
            if is_long and (tp_a <= ep or tp_b <= ep):             continue
            if not is_long and (tp_a >= ep or tp_b >= ep):         continue

            sl_pts   = abs(ep - sl)
            if sl_pts <= 0:                                         continue

            if not found:
                sig_sessions += 1
                found = True
            sig_count += 1
            signal_data.append((si, ep, sl, tp_a, tp_b, pos, idxs, times,
                                 int(np.where(idxs == pos)[0][0])))
            break  # only first signal per session

    n_q = max(1, len(qualifying))
    print(f"     Sessions with ≥1 valid signal: {sig_sessions} ({100*sig_sessions/n_q:.1f}%"
          f" of qualifying)")
    print(f"     Total signals (1 per session): {sig_count}")

    # ── 4. Geometric R/R (no costs) ───────────────────────────────────────────
    print(f"\n  4. GEOMETRIC R/R  (no slippage or costs)")
    if signal_data:
        rr_a_list = []; rr_b_list = []
        tp_a_d = []; tp_b_d = []; sl_d_list = []
        for si, ep, sl, tp_a, tp_b, pos, idxs, times, sess_k in signal_data:
            sl_pts   = abs(ep - sl)
            ta_pts   = abs(tp_a - ep)
            tb_pts   = abs(tp_b - ep)
            rr_a_list.append(ta_pts / sl_pts if sl_pts > 0 else 0.0)
            rr_b_list.append(tb_pts / sl_pts if sl_pts > 0 else 0.0)
            tp_a_d.append(ta_pts); tp_b_d.append(tb_pts); sl_d_list.append(sl_pts)

        print(f"     Entry→TP_A (session_open) distance: "
              f"p25={np.percentile(tp_a_d,25):.1f}  "
              f"p50={np.percentile(tp_a_d,50):.1f}  "
              f"p75={np.percentile(tp_a_d,75):.1f}  pts")
        print(f"     Entry→TP_B (opp. extreme)  distance: "
              f"p25={np.percentile(tp_b_d,25):.1f}  "
              f"p50={np.percentile(tp_b_d,50):.1f}  "
              f"p75={np.percentile(tp_b_d,75):.1f}  pts")
        print(f"     Entry→SL distance:                   "
              f"p25={np.percentile(sl_d_list,25):.1f}  "
              f"p50={np.percentile(sl_d_list,50):.1f}  "
              f"p75={np.percentile(sl_d_list,75):.1f}  pts")
        print(f"     R/R TP-A/SL: p25={np.percentile(rr_a_list,25):.2f}  "
              f"p50={np.percentile(rr_a_list,50):.2f}  "
              f"p75={np.percentile(rr_a_list,75):.2f}")
        print(f"     R/R TP-B/SL: p25={np.percentile(rr_b_list,25):.2f}  "
              f"p50={np.percentile(rr_b_list,50):.2f}  "
              f"p75={np.percentile(rr_b_list,75):.2f}"
              f"  >2.0: {sum(1 for r in rr_b_list if r>2.0)} "
              f"({100*sum(1 for r in rr_b_list if r>2.0)/max(1,len(rr_b_list)):.1f}%)")

    # ── 5. Theoretical expectancy (pure price, no costs) ─────────────────────
    print(f"\n  5. THEORETICAL EXPECTANCY (pure price, both TP variants)")
    if not signal_data:
        print("     No signals — cannot compute. STOPPING.")
        return False

    def _sim_expectancy(use_b: bool) -> tuple:
        tp_h = sl_h = 0
        rr_w = []
        for si, ep, sl, tp_a, tp_b, pos, idxs, times, sess_k in signal_data:
            tp_price = tp_b if use_b else tp_a
            is_long  = (si.spike_direction == "down")
            post_idx = idxs[sess_k + 1:]
            sl_pts   = abs(ep - sl)
            tp_pts   = abs(tp_price - ep)
            rr       = tp_pts / sl_pts if sl_pts > 0 else 0.0
            hit = None
            for p in post_idx:
                b_hi = float(hi_v[p])
                b_lo = float(lo_v[p])
                if is_long:
                    if b_lo <= sl:
                        hit = "SL"; break
                    if b_hi >= tp_price:
                        hit = "TP"; break
                else:
                    if b_hi >= sl:
                        hit = "SL"; break
                    if b_lo <= tp_price:
                        hit = "TP"; break
            if hit == "TP":
                tp_h += 1
                rr_w.append(rr)
            elif hit == "SL":
                sl_h += 1
        n   = len(signal_data)
        wr  = tp_h / n if n > 0 else 0.0
        avg = float(np.mean(rr_w)) if rr_w else 0.0
        exp = wr * avg - (1.0 - wr)
        return tp_h, sl_h, n - tp_h - sl_h, wr, avg, exp

    n_sig = len(signal_data)
    print(f"     Signals simulated: {n_sig}")

    tp_h_a, sl_h_a, no_a, wr_a, rr_a, exp_a = _sim_expectancy(False)
    tp_h_b, sl_h_b, no_b, wr_b, rr_b, exp_b = _sim_expectancy(True)

    print(f"\n     TP-A (session_open):")
    print(f"       Hit TP: {tp_h_a} ({100*wr_a:.1f}%)  Hit SL: {sl_h_a} ({100*sl_h_a/max(1,n_sig):.1f}%)"
          f"  No exit: {no_a}")
    print(f"       Avg R/R on winners: {rr_a:.2f}")
    print(f"       Theoretical expectancy: {exp_a:.3f}")

    print(f"\n     TP-B (opposite spike extreme):")
    print(f"       Hit TP: {tp_h_b} ({100*wr_b:.1f}%)  Hit SL: {sl_h_b} ({100*sl_h_b/max(1,n_sig):.1f}%)"
          f"  No exit: {no_b}")
    print(f"       Avg R/R on winners: {rr_b:.2f}")
    print(f"       Theoretical expectancy: {exp_b:.3f}")

    best_exp = max(exp_a, exp_b)
    if best_exp <= 0:
        print(f"\n  *** THEORETICAL EXPECTANCY NEGATIVE (A={exp_a:.3f}, B={exp_b:.3f}) ***")
        print("  *** Hypothesis has no structural edge. STOPPING. ***")
        return False

    best_tp = "A" if exp_a >= exp_b else "B"
    print(f"\n  Best TP variant: {best_tp}  (expectancy={best_exp:.3f})")
    print(f"  Theoretical expectancy POSITIVE — proceeding to experiments.")
    return True


# ── metrics printer ───────────────────────────────────────────────────────────

def print_block(label: str, trades, df, show_year_breakdown=True, show_2025_2026=False):
    if not trades:
        print(f"  {label}: No trades")
        return
    n   = len(trades)
    tpd = _tpd(trades, df)
    wr  = _wr(trades) * 100
    pf  = _pf(trades)
    sr  = _sr(trades)
    ret = _ret(trades)
    mdd = _max_dd(trades) * 100
    sl_p, tp_p, tm_p, eod_p = _exits(trades)
    aw, wp, al, lp, rr = _avg_w_l(trades)
    nc  = _nc(trades)

    print(f"  {label}: N={n}  TPD={tpd:.3f}  WR={wr:.1f}%  PF={_fmt(pf,3)}  "
          f"SR={sr:.3f}  Ret={ret:.1f}%  MaxDD={mdd:.1f}%")
    print(f"         Exits: SL={sl_p:.1f}%  TP={tp_p:.1f}%  "
          f"TIME={tm_p:.1f}%  EOD={eod_p:.1f}%")
    print(f"         AvgW=${aw:.0f}({wp:.1f}pt)  AvgL=${al:.0f}({lp:.1f}pt)"
          f"  R/R={_fmt(rr,2)}  AvgContr={nc:.1f}")

    if show_year_breakdown:
        ann = _annual(trades, df)
        for (y, n_y, wr_y, pf_y, ret_y) in ann:
            print(f"           {y}: N={n_y:3d}  WR={wr_y:.1f}%  "
                  f"PF={_fmt(pf_y,3)}  Ret={ret_y:.1f}%")

    if show_2025_2026:
        for yr in [2025, 2026]:
            tt = [t for t in trades if pd.Timestamp(str(t.date)).year == yr]
            if tt:
                print(f"         {yr}: N={len(tt):3d}  WR={_wr(tt)*100:.1f}%  "
                      f"PF={_fmt(_pf(tt),3)}  SR={_sr(tt):.3f}  "
                      f"Ret={_ret(tt):.1f}%")
            else:
                print(f"         {yr}: No trades")


def print_exp_header(num, name, params):
    print(f"\n{'═'*W}")
    print(f"  EXPERIMENT {num} — {name}")
    print(f"  {params}")
    print(f"{'═'*W}")


# ── experiment runner ─────────────────────────────────────────────────────────

def run_exp(df_tr, df_te, si_tr, si_te, avg_tr, avg_te,
            tp_variant="A", spike_mult=1.5, vol_mult=1.3,
            poc_filter=None, speed_filter=None,
            speed_fast=1.5, speed_slow=0.8,
            max_bars=60, risk_pct=1.0,
            exp_num=0, label="", save_name=None):

    tr = run(df_tr, si_tr, avg_tr, tp_variant, spike_mult, vol_mult,
             poc_filter, speed_filter, speed_fast, speed_slow,
             max_bars, INITIAL_CAP, risk_pct)
    te = run(df_te, si_te, avg_te, tp_variant, spike_mult, vol_mult,
             poc_filter, speed_filter, speed_fast, speed_slow,
             max_bars, INITIAL_CAP, risk_pct)

    params = (f"TP={tp_variant}  spike_mult={spike_mult}  vol_mult={vol_mult}"
              f"  speed={speed_filter or 'none'}  poc_filter={poc_filter or 'none'}"
              f"  risk_pct={risk_pct}%")
    print_exp_header(exp_num, label, params)
    print_block("TRAIN", tr, df_tr, show_year_breakdown=True)
    print_block("TEST ", te, df_te, show_year_breakdown=True, show_2025_2026=True)

    if save_name:
        _save_csv(tr + te, save_name)

    return tr, te


def _save_csv(trades, fname):
    if not trades:
        return
    rows = []
    for t in trades:
        rows.append({
            "date": t.date, "direction": t.direction,
            "entry_ts": t.entry_ts, "entry_price": t.entry_price,
            "sl_price": t.sl_price, "tp_a": t.tp_a, "tp_b": t.tp_b,
            "n_contracts": t.n_contracts, "exit_ts": t.exit_ts,
            "exit_price": t.exit_price, "exit_reason": t.exit_reason,
            "pnl_pts": t.pnl_pts, "pnl_net": t.pnl_net,
            "spike_magnitude": t.spike_magnitude, "daily_atr": t.daily_atr,
            "rr_a": t.rr_a, "rr_b": t.rr_b,
        })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, fname), index=False
    )


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_wfo(df, daily_atr, atr1m, avg_vol, poc_df, best_cfg: dict):
    print(f"\n{'═'*W}")
    print("  WALK-FORWARD VALIDATION (5 anchored windows)")
    print(f"  Config: {best_cfg}")
    print(f"{'═'*W}")

    prev_poc = poc_df["prev_poc"]
    all_oos  = []
    pf_wins  = 0

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(WFO_WINDOWS, 1):
        df_tr = df.loc[tr_s:tr_e]
        df_te = df.loc[te_s:te_e]

        si_tr = detect_spikes(df_tr, daily_atr[df_tr.index],
                              atr1m[df_tr.index], prev_poc[df_tr.index])
        si_te = detect_spikes(df_te, daily_atr[df_te.index],
                              atr1m[df_te.index], prev_poc[df_te.index])

        te_trades = run(
            df_te, si_te, avg_vol[df_te.index],
            best_cfg.get("tp_variant", "A"),
            best_cfg.get("spike_mult", 1.5),
            best_cfg.get("vol_mult", 1.3),
            best_cfg.get("poc_filter", None),
            best_cfg.get("speed_filter", None),
            best_cfg.get("speed_fast", 1.5),
            best_cfg.get("speed_slow", 0.8),
            best_cfg.get("max_bars", 60),
            INITIAL_CAP,
            best_cfg.get("risk_pct", 1.0),
        )
        all_oos.extend(te_trades)
        pf_v = _pf(te_trades)
        if pf_v > 1.0:
            pf_wins += 1
        print(f"  V{i}  Train {tr_s}–{tr_e} | Test {te_s}–{te_e}")
        print(f"       N={len(te_trades)}  WR={_wr(te_trades)*100:.1f}%  "
              f"PF={_fmt(pf_v,3)}  SR={_sr(te_trades):.3f}  "
              f"Ret={_ret(te_trades):.1f}%  MaxDD={_max_dd(te_trades)*100:.1f}%")

    print(f"\n  WFO Summary: Windows PF>1.0: {pf_wins}/5")

    if not all_oos:
        print("  No OOS trades — cannot compute statistical tests.")
        return

    # Statistical tests on pooled OOS
    pnl_arr = np.array([t.pnl_net for t in all_oos])
    t_stat, p_val = stats.ttest_1samp(pnl_arr, 0.0, alternative="greater")
    pf_pool  = _pf(all_oos)

    N_BOOT = 1000
    rng = np.random.default_rng(42)
    boot_pf = []
    for _ in range(N_BOOT):
        sample = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
        w = sample[sample > 0].sum()
        l = abs(sample[sample <= 0].sum())
        boot_pf.append(w / l if l > 0 else float("inf"))
    bp5  = float(np.percentile([x for x in boot_pf if x != float("inf")], 5))
    bmn  = float(np.mean([x for x in boot_pf if x != float("inf")]))

    print(f"\n  POOLED OOS  ({len(all_oos)} trades)")
    print_block("OOS", all_oos, df, show_year_breakdown=False)
    print(f"  T-test:    t={t_stat:.3f}  p={p_val:.3f}")
    print(f"  Bootstrap: mean_PF={bmn:.3f}  p5={bp5:.3f}")
    print(f"  Pooled PF: {_fmt(pf_pool,3)}")

    edge = (p_val < 0.10 and bp5 > 0.95 and pf_wins >= 3)
    verdict = "EDGE CONFIRMED" if edge else "Edge NOT confirmed"
    print(f"\n  VERDICT: {verdict}")
    if edge:
        print("  Proceed to Monte Carlo FTMO sizing.")
    else:
        failures = []
        if p_val >= 0.10:      failures.append(f"p={p_val:.3f} >= 0.10")
        if bp5 <= 0.95:        failures.append(f"bootstrap p5={bp5:.3f} <= 0.95")
        if pf_wins < 3:        failures.append(f"only {pf_wins}/5 windows PF>1.0")
        print(f"  Failures: {', '.join(failures)}")

    _save_csv(all_oos, "p17_wfo_oos_pooled.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 17 — FAILED OPENING SPIKE REVERSION")
    print("=" * W)

    cfg    = Config()
    df     = load_data(cfg)

    print(f"  Data: {df.index[0]}  →  {df.index[-1]}  ({len(df):,} bars)")

    # ── compute indicators ────────────────────────────────────────────────────
    print("  Computing daily ATR (20-session rolling mean of RTH ranges)...")
    daily_atr = compute_daily_atr(df)

    print("  Computing 1-min ATR(20) and avg_volume(20)...")
    atr1m   = compute_atr(df, 20)
    avg_vol = avg_volume(df, 20)

    print("  Computing volume profile (prev_poc)...")
    poc_df      = compute_poc_features(df)
    prev_poc    = poc_df["prev_poc"]

    # ── detect spikes on full dataset ────────────────────────────────────────
    print("  Detecting opening spikes...")
    spike_infos_full = detect_spikes(df, daily_atr, atr1m, prev_poc)

    # ── run mandatory diagnostic ──────────────────────────────────────────────
    ok = run_diagnostic(df, spike_infos_full, avg_vol)
    if not ok:
        return

    # ── train / test split ───────────────────────────────────────────────────
    df_tr  = df.loc[:TRAIN_END]
    df_te  = df.loc[TEST_START:]
    avg_tr = avg_vol.loc[df_tr.index]
    avg_te = avg_vol.loc[df_te.index]

    si_tr  = detect_spikes(df_tr, daily_atr[df_tr.index],
                           atr1m[df_tr.index], prev_poc[df_tr.index])
    si_te  = detect_spikes(df_te, daily_atr[df_te.index],
                           atr1m[df_te.index], prev_poc[df_te.index])

    print(f"\n  Train: {df_tr.index[0].date()} → {df_tr.index[-1].date()}  "
          f"({len(df_tr):,} bars)")
    print(f"  Test:  {df_te.index[0].date()} → {df_te.index[-1].date()}  "
          f"({len(df_te):,} bars)")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 1 — Baseline · TP Variant A (session_open)
    # ─────────────────────────────────────────────────────────────────────────
    tr1, te1 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te,
        tp_variant="A", spike_mult=1.5, risk_pct=1.0,
        exp_num=1, label="Baseline · TP-A (session_open)",
        save_name="p17_exp1_baseline_tpA.csv",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 2 — TP Variant B (full reversion to opposite extreme)
    # ─────────────────────────────────────────────────────────────────────────
    tr2, te2 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te,
        tp_variant="B", spike_mult=1.5, risk_pct=1.0,
        exp_num=2, label="TP-B (opposite spike extreme)",
        save_name="p17_exp2_tpB.csv",
    )

    # TP comparison
    print(f"\n  TP comparison (baseline params):")
    print(f"    TP-A  TRAIN PF={_fmt(_pf(tr1))}  TEST PF={_fmt(_pf(te1))}"
          f"  WR={_wr(te1)*100:.1f}%  AvgW=${_avg_w_l(te1)[0]:.0f}")
    print(f"    TP-B  TRAIN PF={_fmt(_pf(tr2))}  TEST PF={_fmt(_pf(te2))}"
          f"  WR={_wr(te2)*100:.1f}%  AvgW=${_avg_w_l(te2)[0]:.0f}")
    best_tp = "A" if _pf(te1) >= _pf(te2) else "B"
    print(f"    → Best TP variant: {best_tp}")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 3 — Spike size filter
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  EXPERIMENT 3 — Spike size filter")
    print(f"  TP={best_tp}  spike_mult: [1.5, 2.0, 2.5, 3.0]")
    print(f"{'═'*W}")
    print(f"  {'mult':>5} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6}")
    print(f"  {'-'*55}")
    best3_mult = 1.5; best3_pf = -1.0
    exp3_results = {}
    for mult in [1.5, 2.0, 2.5, 3.0]:
        tr_x = run(df_tr, si_tr, avg_tr, best_tp, mult, 1.3,
                   None, None, 1.5, 0.8, 60, INITIAL_CAP, 1.0)
        te_x = run(df_te, si_te, avg_te, best_tp, mult, 1.3,
                   None, None, 1.5, 0.8, 60, INITIAL_CAP, 1.0)
        pf_te = _pf(te_x)
        print(f"  {mult:>5.1f} | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x),3):>6} | "
              f"{_fmt(pf_te,3):>6} | {_sr(te_x):>6.3f}")
        exp3_results[mult] = (tr_x, te_x)
        if pf_te > best3_pf and len(te_x) >= 10:
            best3_pf = pf_te; best3_mult = mult
    print(f"  → Best spike_mult: {best3_mult}")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 4 — Spike speed filter
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  EXPERIMENT 4 — Spike speed filter")
    print(f"  spike_mult={best3_mult}  TP={best_tp}")
    print(f"{'═'*W}")
    speed_configs = [
        ("none",  None,    1.5, 0.8),
        ("fast",  "fast",  1.5, 0.8),
        ("slow",  "slow",  1.5, 0.8),
    ]
    print(f"  {'filter':>6} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6}")
    print(f"  {'-'*60}")
    best4_speed = None; best4_pf = -1.0
    exp4_results = {}
    for name, sf, sff, sfs in speed_configs:
        tr_x = run(df_tr, si_tr, avg_tr, best_tp, best3_mult, 1.3,
                   None, sf, sff, sfs, 60, INITIAL_CAP, 1.0)
        te_x = run(df_te, si_te, avg_te, best_tp, best3_mult, 1.3,
                   None, sf, sff, sfs, 60, INITIAL_CAP, 1.0)
        pf_te = _pf(te_x)
        print(f"  {name:>6} | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x),3):>6} | "
              f"{_fmt(pf_te,3):>6} | {_sr(te_x):>6.3f}")
        exp4_results[name] = (tr_x, te_x)
        if pf_te > best4_pf and len(te_x) >= 10:
            best4_pf = pf_te; best4_speed = sf
    print(f"  → Best speed filter: {best4_speed or 'none'}")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 5 — POC confluence filter
    # ─────────────────────────────────────────────────────────────────────────
    print_exp_header(5, "POC confluence filter",
                     f"spike_extreme within 5.0 pts of prev_poc  "
                     f"spike_mult={best3_mult}  TP={best_tp}")
    tr5 = run(df_tr, si_tr, avg_tr, best_tp, best3_mult, 1.3,
              5.0, best4_speed, 1.5, 0.8, 60, INITIAL_CAP, 1.0)
    te5 = run(df_te, si_te, avg_te, best_tp, best3_mult, 1.3,
              5.0, best4_speed, 1.5, 0.8, 60, INITIAL_CAP, 1.0)
    print_block("TRAIN", tr5, df_tr)
    print_block("TEST ", te5, df_te, show_2025_2026=True)
    use_poc = (_pf(te5) > _pf(exp4_results.get(
        "none" if best4_speed is None else best4_speed, ([], te1))[1])
               and len(te5) >= 10)
    poc_filter_final = 5.0 if use_poc else None
    print(f"  → POC filter: {'use (improved PF)' if use_poc else 'skip (did not improve)'}")
    _save_csv(tr5 + te5, "p17_exp5_poc.csv")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 6 — Risk percentage sensitivity
    # ─────────────────────────────────────────────────────────────────────────
    print_exp_header(6, "Risk % sensitivity",
                     f"best config  spike_mult={best3_mult}  TP={best_tp}  "
                     f"speed={best4_speed or 'none'}  poc={poc_filter_final or 'none'}")
    print(f"  {'risk%':>6} | {'N_te':>5} | {'AvgContr':>8} | {'PF_te':>6} | "
          f"{'SR_te':>6} | {'MaxDD':>6}")
    print(f"  {'-'*50}")
    for rp in [0.5, 1.0, 1.5]:
        te_x = run(df_te, si_te, avg_te, best_tp, best3_mult, 1.3,
                   poc_filter_final, best4_speed, 1.5, 0.8, 60, INITIAL_CAP, rp)
        print(f"  {rp:>5.1f}% | {len(te_x):>5} | {_nc(te_x):>8.1f} | "
              f"{_fmt(_pf(te_x),3):>6} | {_sr(te_x):>6.3f} | "
              f"{_max_dd(te_x)*100:>5.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 7 — Combined best configuration
    # ─────────────────────────────────────────────────────────────────────────
    print_exp_header(7, "Combined best configuration",
                     f"spike_mult={best3_mult}  TP={best_tp}  "
                     f"speed={best4_speed or 'none'}  poc={poc_filter_final or 'none'}")
    tr7, te7 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te,
        tp_variant=best_tp, spike_mult=best3_mult, risk_pct=1.0,
        poc_filter=poc_filter_final,
        speed_filter=best4_speed,
        exp_num=7, label="Combined best",
        save_name="p17_exp7_best.csv",
    )

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  SUMMARY TABLE")
    print(f"  {'Exp':>3} | {'TP':>3} | {'mult':>5} | {'speed':>5} | {'poc':>4} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6} | {'TPD':>5} | {'WR':>5}")
    print(f"  {'-'*75}")
    summary_rows = [
        (1,  "A",    1.5, "none", "no",  tr1,  te1),
        (2,  "B",    1.5, "none", "no",  tr2,  te2),
        (3,  best_tp, best3_mult, "none", "no",
         exp3_results[best3_mult][0], exp3_results[best3_mult][1]),
        (7,  best_tp, best3_mult, best4_speed or "none",
         "yes" if poc_filter_final else "no", tr7, te7),
    ]
    for (en, tp, mult, spd, poc, trr, ter) in summary_rows:
        print(f"  {en:>3} | {tp:>3} | {mult:>5.1f} | {spd:>5} | {poc:>4} | "
              f"{_fmt(_pf(trr),3):>6} | {_fmt(_pf(ter),3):>6} | "
              f"{_sr(ter):>6.3f} | {_tpd(ter, df_te):>5.3f} | "
              f"{_wr(ter)*100:>4.1f}%")

    # ── WFO decision ──────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    pf7_te   = _pf(te7)
    n7_te    = len(te7)
    go_wfo   = (pf7_te > 1.0 and n7_te >= 80)
    print(f"  Exp 7 test: PF={_fmt(pf7_te,3)}  N={n7_te}")
    if go_wfo:
        print("  → Thresholds met (PF>1.0 and N>=80). Running walk-forward.")
        best_cfg = dict(
            tp_variant=best_tp, spike_mult=best3_mult, vol_mult=1.3,
            poc_filter=poc_filter_final, speed_filter=best4_speed,
            speed_fast=1.5, speed_slow=0.8, max_bars=60, risk_pct=1.0,
        )
        run_wfo(df, daily_atr, atr1m, avg_vol, poc_df, best_cfg)
    else:
        reasons = []
        if pf7_te <= 1.0: reasons.append(f"PF_test={_fmt(pf7_te,3)} ≤ 1.0")
        if n7_te < 80:    reasons.append(f"N_test={n7_te} < 80")
        print(f"  → WFO skipped: {'; '.join(reasons)}")
        print("  Hypothesis not supported. Do not proceed to FTMO sizing.")

    print(f"\n{'='*W}")
    print("  PHASE 17 COMPLETE")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
