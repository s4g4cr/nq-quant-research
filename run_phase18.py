#!/usr/bin/env python3
"""
Phase 18 — Opening Spike Extreme as Support/Resistance.

Key fix from Phase 17: SL is ATR-based, not spike-extreme-based.
Mandatory 6-section pre-PnL diagnostic gates all experiments.
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
from orb_system.strategy.spike_extreme_reversion import (
    SpikeInfo, SpikeTrade, detect_spikes, run,
    SLIP, COMM, PV, _SPIKE_S, _SPIKE_E, _ENTRY_S, _ENTRY_E, _EOD_T,
)

INITIAL_CAP = 100_000.0
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
W           = 72
RESULTS_DIR = os.path.join(ROOT, "results")

WFO_WINDOWS = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-30"),
]


# ── stat helpers ──────────────────────────────────────────────────────────────

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
    return sum(1 for t in trades if t.pnl_net > 0) / len(trades) if trades else 0.0

def _max_dd(trades):
    cap = peak = INITIAL_CAP; worst = 0.0
    for t in trades:
        cap += t.pnl_net; peak = max(peak, cap)
        worst = max(worst, (peak - cap) / peak)
    return worst

def _tpd(trades, df):
    n_days = len(np.unique(np.array(df.index.date)))
    return len(trades) / n_days if n_days > 0 else 0.0

def _ret(trades):
    return sum(t.pnl_net for t in trades) / INITIAL_CAP * 100.0

def _exits(trades):
    n = max(1, len(trades))
    sl    = sum(1 for t in trades if t.exit_reason == "SL")
    tp    = sum(1 for t in trades if t.exit_reason == "TP")
    trail = sum(1 for t in trades if t.exit_reason == "TRAIL")
    tm    = sum(1 for t in trades if t.exit_reason == "TIME")
    eod   = sum(1 for t in trades if t.exit_reason == "EOD")
    return sl*100/n, tp*100/n, trail*100/n, tm*100/n, eod*100/n

def _avg_w_l(trades):
    wins  = [t.pnl_net for t in trades if t.pnl_net > 0]
    loses = [t.pnl_net for t in trades if t.pnl_net <= 0]
    wp    = [t.pnl_pts for t in trades if t.pnl_net > 0]
    lp    = [t.pnl_pts for t in trades if t.pnl_net <= 0]
    aw = float(np.mean(wins))  if wins  else 0.0
    al = float(np.mean(loses)) if loses else 0.0
    awp = float(np.mean(wp)) if wp else 0.0
    alp = float(np.mean(lp)) if lp else 0.0
    rr  = abs(aw / al) if al != 0 else float("inf")
    return aw, awp, al, alp, rr

def _nc(trades):
    return float(np.mean([t.n_contracts for t in trades])) if trades else 0.0

def _annual(trades):
    by_year: dict = {}
    for t in trades:
        y = pd.Timestamp(str(t.date)).year
        by_year.setdefault(y, []).append(t)
    return [(y, len(tt), _wr(tt)*100, _pf(tt), _ret(tt))
            for y, tt in sorted(by_year.items())]

def _fmt(v, d=3):
    if v != v or v == float("inf"): return "  inf"
    return f"{v:.{d}f}"


# ── diagnostic ────────────────────────────────────────────────────────────────

def run_diagnostic(
    df: pd.DataFrame,
    spike_infos: dict,
    avg_vol_series: pd.Series,
    atr_1min_series: pd.Series,
    baseline_mult: float = 1.5,
    retest_zone: float = 0.5,
    vol_mult: float = 1.2,
    min_rr: float = 1.0,
) -> bool:

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

    valid_infos = [si for si in spike_infos.values() if si is not None]
    n_sessions  = len(valid_infos)

    print(f"\n{'─'*W}")
    print("  DIAGNOSTIC — PRE-PnL STATISTICS (full dataset)")
    print(f"{'─'*W}")

    # ── 1. Spike statistics ───────────────────────────────────────────────────
    ratio_vals = [si.spike_magnitude / si.atr_spike
                  for si in valid_infos
                  if not np.isnan(si.atr_spike) and si.atr_spike > 0]
    q15 = sum(1 for r in ratio_vals if r > 1.5)
    q20 = sum(1 for r in ratio_vals if r > 2.0)
    q25 = sum(1 for r in ratio_vals if r > 2.5)
    n_rv = max(1, len(ratio_vals))

    print(f"\n  1. SPIKE STATISTICS  (sessions: {n_sessions})")
    print(f"     Spike > 1.5×ATR : {q15:4d} ({100*q15/n_rv:.1f}%)")
    print(f"     Spike > 2.0×ATR : {q20:4d} ({100*q20/n_rv:.1f}%)")
    print(f"     Spike > 2.5×ATR : {q25:4d} ({100*q25/n_rv:.1f}%)")
    if ratio_vals:
        p = np.percentile(ratio_vals, [10, 25, 50, 75, 90])
        print(f"     spike/ATR: p10={p[0]:.2f}  p25={p[1]:.2f}  p50={p[2]:.2f}"
              f"  p75={p[3]:.2f}  p90={p[4]:.2f}")
    up = sum(1 for si in valid_infos if si.spike_direction == "up")
    dn = sum(1 for si in valid_infos if si.spike_direction == "down")
    print(f"     Direction: up={up} ({100*up/max(1,n_sessions):.1f}%)  "
          f"down={dn} ({100*dn/max(1,n_sessions):.1f}%)")

    # Qualifying sessions for further analysis
    qualifying = [si for si in valid_infos
                  if not np.isnan(si.atr_spike) and si.atr_spike > 0
                  and si.spike_magnitude > baseline_mult * si.atr_spike]
    nq = max(1, len(qualifying))

    # ── 2. Retest frequency ───────────────────────────────────────────────────
    print(f"\n  2. RETEST FREQUENCY  (spike > {baseline_mult}×ATR, zone = ±{retest_zone}×ATR)")
    sessions_with_retest = 0
    total_retests        = 0
    retest_counts        = []

    for si in qualifying:
        mask  = date_arr == si.date
        idxs  = all_pos[mask]
        times = time_arr[mask]
        is_long = (si.spike_direction == "down")

        in_zone = False
        sess_retests = 0
        for k, pos in enumerate(idxs):
            t_bar = times[k]
            if not (_ENTRY_S <= t_bar <= _ENTRY_E):
                continue
            b_hi  = float(hi_v[pos])
            b_lo  = float(lo_v[pos])
            b_atr = float(a1_v[pos])
            if np.isnan(b_atr) or b_atr <= 0:
                continue

            in_z = (b_lo <= si.spike_low + retest_zone * b_atr) if is_long \
                   else (b_hi >= si.spike_high - retest_zone * b_atr)
            if in_z and not in_zone:
                sess_retests += 1
            in_zone = in_z

        retest_counts.append(sess_retests)
        if sess_retests > 0:
            sessions_with_retest += 1
        total_retests += sess_retests

    print(f"     Qualifying sessions: {len(qualifying)}")
    print(f"     Sessions with ≥1 retest: {sessions_with_retest} "
          f"({100*sessions_with_retest/nq:.1f}%)")
    print(f"     Total retest events: {total_retests}")
    avg_rt = total_retests / max(1, len(qualifying))
    print(f"     Avg retests per qualifying session: {avg_rt:.2f}")
    if retest_counts:
        p = np.percentile(retest_counts, [50, 75, 90])
        print(f"     Retests/session: p50={p[0]:.0f}  p75={p[1]:.0f}  p90={p[2]:.0f}")

    # ── 3. Rejection rate ─────────────────────────────────────────────────────
    print(f"\n  3. REJECTION RATE  (of retest-zone entries)")
    zone_entries     = 0
    zone_rejections  = 0

    for si in qualifying:
        mask  = date_arr == si.date
        idxs  = all_pos[mask]
        times = time_arr[mask]
        is_long = (si.spike_direction == "down")

        in_zone = False
        for k, pos in enumerate(idxs):
            t_bar = times[k]
            if not (_ENTRY_S <= t_bar <= _ENTRY_E):
                continue
            b_hi  = float(hi_v[pos])
            b_lo  = float(lo_v[pos])
            b_cl  = float(cl_v[pos])
            b_atr = float(a1_v[pos])
            if np.isnan(b_atr) or b_atr <= 0:
                continue

            in_z = (b_lo <= si.spike_low + retest_zone * b_atr) if is_long \
                   else (b_hi >= si.spike_high - retest_zone * b_atr)
            if in_z and not in_zone:
                zone_entries += 1
                rejected = (b_cl > si.spike_low) if is_long else (b_cl < si.spike_high)
                if rejected:
                    zone_rejections += 1
            in_zone = in_z

    rej_rate = zone_rejections / max(1, zone_entries) * 100
    print(f"     Zone-entry bars:     {zone_entries}")
    print(f"     Rejection confirmed: {zone_rejections} ({rej_rate:.1f}%)")

    # ── 4. R/R distribution with ATR-based SL ────────────────────────────────
    print(f"\n  4. R/R DISTRIBUTION  (ATR-based SL, hard R/R≥{min_rr})")
    signal_data = []  # (si, ep, sl_pts, tp_a, tp_b, tp_c, pos, k, idxs)

    for si in qualifying:
        mask  = date_arr == si.date
        idxs  = all_pos[mask]
        times = time_arr[mask]
        is_long = (si.spike_direction == "down")
        found = 0

        for k, pos in enumerate(idxs):
            t_bar = times[k]
            if not (_ENTRY_S <= t_bar <= _ENTRY_E):
                continue
            b_hi  = float(hi_v[pos])
            b_lo  = float(lo_v[pos])
            b_cl  = float(cl_v[pos])
            b_op  = float(op_v[pos])
            b_vol = float(vol_v[pos])
            b_atr = float(a1_v[pos])
            b_avg = float(avg_v[pos])
            if np.isnan(b_atr) or b_atr <= 0 or np.isnan(b_avg) or b_avg <= 0:
                continue
            if (b_hi - b_lo) == 0:
                continue

            if is_long:
                if not (b_lo <= si.spike_low + retest_zone * b_atr):  continue
                if not (b_cl > si.spike_low):                          continue
                if not (b_cl > b_op):                                  continue
                if not (b_cl > b_lo + 0.5 * (b_hi - b_lo)):           continue
                if not (b_vol > vol_mult * b_avg):                     continue
                if not (b_cl < si.session_open):                       continue
                ep   = b_cl + SLIP
                tp_a = si.session_open
                tp_b = si.session_open + 0.5 * (si.session_open - si.spike_low)
                tp_c = si.spike_high
                sl_pts = b_atr
                rr_a = (tp_a - ep) / sl_pts
                rr_b = (tp_b - ep) / sl_pts
                rr_c = (tp_c - ep) / sl_pts
            else:
                if not (b_hi >= si.spike_high - retest_zone * b_atr):  continue
                if not (b_cl < si.spike_high):                          continue
                if not (b_cl < b_op):                                   continue
                if not (b_cl < b_hi - 0.5 * (b_hi - b_lo)):            continue
                if not (b_vol > vol_mult * b_avg):                      continue
                if not (b_cl > si.session_open):                        continue
                ep   = b_cl - SLIP
                tp_a = si.session_open
                tp_b = si.session_open - 0.5 * (si.spike_high - si.session_open)
                tp_c = si.spike_low
                sl_pts = b_atr
                rr_a = (ep - tp_a) / sl_pts
                rr_b = (ep - tp_b) / sl_pts
                rr_c = (ep - tp_c) / sl_pts

            signal_data.append((si, ep, sl_pts, tp_a, tp_b, tp_c, pos, k, idxs))
            found += 1
            if found >= 2:
                break  # max 2 per session in diagnostic too

    n_sig = len(signal_data)
    print(f"     Raw signals (before R/R gate): {n_sig}")

    if n_sig > 0:
        rr_a_l = []; rr_b_l = []; rr_c_l = []
        sl_l   = []; tpa_l  = []; tpb_l  = []; tpc_l = []
        for si, ep, sl_pts, tp_a, tp_b, tp_c, pos, k, idxs in signal_data:
            is_long = (si.spike_direction == "down")
            if is_long:
                rra = (tp_a - ep) / sl_pts
                rrb = (tp_b - ep) / sl_pts
                rrc = (tp_c - ep) / sl_pts
            else:
                rra = (ep - tp_a) / sl_pts
                rrb = (ep - tp_b) / sl_pts
                rrc = (ep - tp_c) / sl_pts
            rr_a_l.append(rra); rr_b_l.append(rrb); rr_c_l.append(rrc)
            sl_l.append(sl_pts)
            tpa_l.append(abs(tp_a - ep))
            tpb_l.append(abs(tp_b - ep))
            tpc_l.append(abs(tp_c - ep))

        def _rr_stats(arr, label):
            a = np.array(arr)
            p = np.percentile(a, [25, 50, 75])
            gt = [(t, sum(1 for x in a if x > t)) for t in [1.0, 1.5, 2.0, 3.0]]
            s = f"     R/R {label}: p25={p[0]:.2f}  p50={p[1]:.2f}  p75={p[2]:.2f}"
            s += "  " + "  ".join(f">{t:.0f}:{n}({100*n/max(1,len(a)):.0f}%)" for t,n in gt)
            print(s)

        _rr_stats(rr_a_l, "TP-A")
        _rr_stats(rr_b_l, "TP-B")
        _rr_stats(rr_c_l, "TP-C")
        print(f"     SL dist (ATR): p25={np.percentile(sl_l,25):.1f}  "
              f"p50={np.percentile(sl_l,50):.1f}  p75={np.percentile(sl_l,75):.1f}  pts")
        print(f"     TP-A dist:     p25={np.percentile(tpa_l,25):.1f}  "
              f"p50={np.percentile(tpa_l,50):.1f}  p75={np.percentile(tpa_l,75):.1f}  pts")

        # Gate check
        med_a = np.median(rr_a_l)
        med_b = np.median(rr_b_l)
        med_c = np.median(rr_c_l)
        print(f"\n     Median R/R: TP-A={med_a:.2f}  TP-B={med_b:.2f}  TP-C={med_c:.2f}")
        best_med = max(med_a, med_b, med_c)
        if best_med <= 1.0:
            print(f"\n  *** ALL MEDIAN R/R < 1.0 ({best_med:.2f}) — geometry still broken ***")
            print("  *** STOPPING — do not run experiments. ***")
            return False

    # ── 5. Theoretical expectancy ─────────────────────────────────────────────
    print(f"\n  5. THEORETICAL EXPECTANCY  (pure price, no costs)")
    if n_sig == 0:
        print("     No signals — STOPPING.")
        return False

    def _sim_tp(tp_key: str):
        tp_h = sl_h = 0; rr_w = []
        for si, ep, sl_pts, tp_a, tp_b, tp_c, pos, k, idxs in signal_data:
            is_long = (si.spike_direction == "down")
            tp_map  = {"A": tp_a, "B": tp_b, "C": tp_c}
            tp_price = tp_map[tp_key]
            sl_price = (ep - sl_pts) if is_long else (ep + sl_pts)

            if is_long:
                if tp_price <= ep:  continue
                rr = (tp_price - ep) / sl_pts
            else:
                if tp_price >= ep:  continue
                rr = (ep - tp_price) / sl_pts
            if rr < min_rr:  continue

            hit = None
            for p2 in idxs[k + 1:]:
                bh = float(hi_v[p2]); bl = float(lo_v[p2])
                if is_long:
                    if bl <= sl_price: hit = "SL"; break
                    if bh >= tp_price: hit = "TP"; break
                else:
                    if bh >= sl_price: hit = "SL"; break
                    if bl <= tp_price: hit = "TP"; break
            if hit == "TP":   tp_h += 1; rr_w.append(rr)
            elif hit == "SL": sl_h += 1

        n = tp_h + sl_h
        if n == 0: return 0.0, 0.0, float("nan"), 0, 0
        wr  = tp_h / n
        avg = float(np.mean(rr_w)) if rr_w else 0.0
        exp = wr * avg - (1.0 - wr)
        return wr, avg, exp, tp_h, sl_h

    any_positive = False
    for tpk in ["A", "B", "C"]:
        wr, avg_rr, exp, tp_h, sl_h = _sim_tp(tpk)
        n_sim = tp_h + sl_h
        tag   = " ← POSITIVE" if exp > 0 else ""
        print(f"     TP-{tpk}: sims={n_sim}  WR={100*wr:.1f}%  avgR/R={avg_rr:.2f}"
              f"  expectancy={exp:.3f}{tag}")
        if exp > 0:
            any_positive = True

    if not any_positive:
        print(f"\n  *** ALL THEORETICAL EXPECTANCY NEGATIVE — STOPPING. ***")
        return False

    print(f"\n  At least one TP variant positive — proceeding to experiments.")

    # ── 6. Trades per day estimate ────────────────────────────────────────────
    print(f"\n  6. SIGNAL FREQUENCY")
    n_days = len(np.unique(date_arr))
    n_tdays = len(qualifying)   # approximate trading days with activity
    print(f"     Total signals (≤2/session): {n_sig}")
    print(f"     Sessions with qualifying spike: {len(qualifying)}")
    print(f"     Avg signals per qualifying session: {n_sig/max(1,len(qualifying)):.2f}")
    print(f"     Signals per calendar trading day: {n_sig/max(1,n_days):.3f}")
    if n_sig / max(1, n_days) < 0.8:
        print("     WARNING: < 0.8 signals/day — may be too sparse for FTMO monthly target.")

    return True


# ── metrics printer ───────────────────────────────────────────────────────────

def print_block(label: str, trades, df, show_annual=True, show_2025_2026=False):
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
    sl_p, tp_p, tr_p, tm_p, eod_p = _exits(trades)
    aw, awp, al, alp, rr = _avg_w_l(trades)
    nc  = _nc(trades)
    e2  = sum(1 for t in trades if t.entry_num == 2)

    print(f"  {label}: N={n}  TPD={tpd:.3f}  WR={wr:.1f}%  PF={_fmt(pf)}  "
          f"SR={sr:.3f}  Ret={ret:.1f}%  MaxDD={mdd:.1f}%")
    print(f"         Exits: SL={sl_p:.1f}%  TP={tp_p:.1f}%  Trail={tr_p:.1f}%  "
          f"TIME={tm_p:.1f}%  EOD={eod_p:.1f}%")
    print(f"         AvgW=${aw:.0f}({awp:.1f}pt)  AvgL=${al:.0f}({alp:.1f}pt)"
          f"  R/R={_fmt(rr,2)}  AvgContr={nc:.1f}  2nd-entry={e2}({100*e2/n:.0f}%)")

    if show_annual:
        for y, n_y, wr_y, pf_y, ret_y in _annual(trades):
            print(f"           {y}: N={n_y:3d}  WR={wr_y:.1f}%  "
                  f"PF={_fmt(pf_y)}  Ret={ret_y:.1f}%")

    if show_2025_2026:
        for yr in [2025, 2026]:
            tt = [t for t in trades if pd.Timestamp(str(t.date)).year == yr]
            if tt:
                print(f"         {yr}: N={len(tt):3d}  WR={_wr(tt)*100:.1f}%  "
                      f"PF={_fmt(_pf(tt))}  SR={_sr(tt):.3f}  Ret={_ret(tt):.1f}%")
            else:
                print(f"         {yr}: No trades")


def print_exp_header(num, name, params):
    print(f"\n{'═'*W}")
    print(f"  EXPERIMENT {num} — {name}")
    print(f"  {params}")
    print(f"{'═'*W}")


def _save_csv(trades, fname):
    if not trades:
        return
    rows = []
    for t in trades:
        rows.append({
            "date": t.date, "direction": t.direction, "entry_num": t.entry_num,
            "entry_ts": t.entry_ts, "entry_price": t.entry_price,
            "sl_price": t.sl_price, "sl_pts": t.sl_pts,
            "tp_a": t.tp_a, "tp_b": t.tp_b, "tp_c": t.tp_c,
            "tp_used": t.tp_used, "n_contracts": t.n_contracts,
            "exit_ts": t.exit_ts, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "pnl_pts": t.pnl_pts,
            "pnl_net": t.pnl_net, "spike_magnitude": t.spike_magnitude,
            "atr_at_entry": t.atr_at_entry, "rr_used": t.rr_used,
        })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, fname), index=False)


def run_exp(
    df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
    poc_tr=None, poc_te=None,
    tp_variant="A", spike_mult=1.5, vol_mult=1.2, retest_zone=0.5,
    min_rr=1.0, use_trailing=False, trail_activate=1.0, trail_dist=0.75,
    poc_filter=None, retest_quality=False, retest_range_mult=0.8,
    retest_vol_mult=1.5, max_bars=90, max_entries=2, risk_pct=1.0,
    exp_num=0, label="", save_name=None,
):
    kw = dict(
        tp_variant=tp_variant, spike_mult=spike_mult, vol_mult=vol_mult,
        retest_zone=retest_zone, min_rr=min_rr,
        use_trailing=use_trailing, trail_activate=trail_activate,
        trail_dist=trail_dist, poc_filter_pts=poc_filter,
        retest_quality=retest_quality, retest_range_mult=retest_range_mult,
        retest_vol_mult=retest_vol_mult, max_bars=max_bars,
        max_entries=max_entries, initial_capital=INITIAL_CAP, risk_pct=risk_pct,
    )
    tr = run(df_tr, si_tr, avg_tr, a1_tr, poc_tr, **kw)
    te = run(df_te, si_te, avg_te, a1_te, poc_te, **kw)

    trailing_tag = "trailing" if use_trailing else f"TP-{tp_variant}"
    params = (f"{trailing_tag}  spike_mult={spike_mult}  vol_mult={vol_mult}"
              f"  min_rr={min_rr}  quality={retest_quality}"
              f"  poc={poc_filter or 'none'}  risk={risk_pct}%")
    print_exp_header(exp_num, label, params)
    print_block("TRAIN", tr, df_tr)
    print_block("TEST ", te, df_te, show_2025_2026=True)

    if save_name:
        _save_csv(tr + te, save_name)

    return tr, te


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_wfo(df, atr1m, avg_vol, poc_df, best_cfg: dict):
    print(f"\n{'═'*W}")
    print("  WALK-FORWARD VALIDATION (5 anchored windows)")
    print(f"  Config: {best_cfg}")
    print(f"{'═'*W}")

    prev_poc = poc_df["prev_poc"] if poc_df is not None else None
    all_oos  = []
    pf_wins  = 0

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(WFO_WINDOWS, 1):
        df_tr = df.loc[tr_s:tr_e]
        df_te = df.loc[te_s:te_e]

        si_tr = detect_spikes(df_tr, atr1m[df_tr.index])
        si_te = detect_spikes(df_te, atr1m[df_te.index])
        poc_tr = prev_poc[df_tr.index] if prev_poc is not None else None
        poc_te = prev_poc[df_te.index] if prev_poc is not None else None

        kw = dict(
            tp_variant=best_cfg.get("tp_variant", "A"),
            spike_mult=best_cfg.get("spike_mult", 1.5),
            vol_mult=best_cfg.get("vol_mult", 1.2),
            retest_zone=best_cfg.get("retest_zone", 0.5),
            min_rr=best_cfg.get("min_rr", 1.0),
            use_trailing=best_cfg.get("use_trailing", False),
            trail_activate=best_cfg.get("trail_activate", 1.0),
            trail_dist=best_cfg.get("trail_dist", 0.75),
            poc_filter_pts=best_cfg.get("poc_filter", None),
            retest_quality=best_cfg.get("retest_quality", False),
            retest_range_mult=best_cfg.get("retest_range_mult", 0.8),
            retest_vol_mult=best_cfg.get("retest_vol_mult", 1.5),
            max_bars=best_cfg.get("max_bars", 90),
            max_entries=best_cfg.get("max_entries", 2),
            initial_capital=INITIAL_CAP,
            risk_pct=best_cfg.get("risk_pct", 1.0),
        )
        te_trades = run(df_te, si_te, avg_vol[df_te.index], atr1m[df_te.index],
                        poc_te, **kw)
        all_oos.extend(te_trades)
        pf_v = _pf(te_trades)
        if pf_v > 1.0:
            pf_wins += 1
        print(f"  V{i}  Train {tr_s}–{tr_e} | Test {te_s}–{te_e}")
        print(f"       N={len(te_trades):3d}  WR={_wr(te_trades)*100:.1f}%  "
              f"PF={_fmt(pf_v)}  SR={_sr(te_trades):.3f}  "
              f"Ret={_ret(te_trades):.1f}%  MaxDD={_max_dd(te_trades)*100:.1f}%")

    print(f"\n  WFO Summary: Windows PF>1.0: {pf_wins}/5")

    if not all_oos:
        print("  No OOS trades — cannot compute statistical tests.")
        return

    pnl_arr      = np.array([t.pnl_net for t in all_oos])
    t_stat, p_val = stats.ttest_1samp(pnl_arr, 0.0, alternative="greater")
    pf_pool       = _pf(all_oos)

    rng = np.random.default_rng(42)
    boot_pf = []
    for _ in range(1000):
        s = rng.choice(pnl_arr, size=len(pnl_arr), replace=True)
        w = s[s > 0].sum(); l = abs(s[s <= 0].sum())
        boot_pf.append(w / l if l > 0 else float("inf"))
    finite = [x for x in boot_pf if x != float("inf")]
    bp5  = float(np.percentile(finite, 5))
    bmn  = float(np.mean(finite))

    print(f"\n  POOLED OOS  ({len(all_oos)} trades)")
    print_block("OOS", all_oos, df, show_annual=False)
    print(f"  T-test:    t={t_stat:.3f}  p={p_val:.3f}")
    print(f"  Bootstrap: mean_PF={bmn:.3f}  p5={bp5:.3f}")
    print(f"  Pooled PF: {_fmt(pf_pool)}")

    edge = (p_val < 0.10 and bp5 > 0.95 and pf_wins >= 3)
    verdict = "EDGE CONFIRMED" if edge else "Edge NOT confirmed"
    print(f"\n  VERDICT: {verdict}")
    if not edge:
        fail = []
        if p_val >= 0.10:  fail.append(f"p={p_val:.3f}≥0.10")
        if bp5 <= 0.95:    fail.append(f"bootstrap p5={bp5:.3f}≤0.95")
        if pf_wins < 3:    fail.append(f"{pf_wins}/5 windows PF>1.0")
        print(f"  Failures: {', '.join(fail)}")
    else:
        print("  Proceed to Monte Carlo FTMO sizing (Phase 19).")

    _save_csv(all_oos, "p18_wfo_oos_pooled.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 18 — OPENING SPIKE EXTREME AS SUPPORT / RESISTANCE")
    print("=" * W)

    cfg = Config()
    df  = load_data(cfg)
    print(f"  Data: {df.index[0]}  →  {df.index[-1]}  ({len(df):,} bars)")

    print("  Computing ATR(20) and avg_volume(20)...")
    atr1m   = compute_atr(df, 20)
    avg_vol = avg_volume(df, 20)

    print("  Computing volume profile (prev_poc)...")
    poc_df   = compute_poc_features(df)
    prev_poc = poc_df["prev_poc"]

    print("  Detecting opening spikes...")
    si_full = detect_spikes(df, atr1m)

    ok = run_diagnostic(df, si_full, avg_vol, atr1m)
    if not ok:
        return

    # ── split ─────────────────────────────────────────────────────────────────
    df_tr   = df.loc[:TRAIN_END];    df_te   = df.loc[TEST_START:]
    avg_tr  = avg_vol[df_tr.index];  avg_te  = avg_vol[df_te.index]
    a1_tr   = atr1m[df_tr.index];    a1_te   = atr1m[df_te.index]
    poc_tr  = prev_poc[df_tr.index]; poc_te  = prev_poc[df_te.index]
    si_tr   = detect_spikes(df_tr, a1_tr)
    si_te   = detect_spikes(df_te, a1_te)

    print(f"\n  Train: {df_tr.index[0].date()} → {df_tr.index[-1].date()}  "
          f"({len(df_tr):,} bars)")
    print(f"  Test:  {df_te.index[0].date()} → {df_te.index[-1].date()}  "
          f"({len(df_te):,} bars)")

    # ── Experiments 1–3: TP variant comparison ────────────────────────────────
    tr1, te1 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
        tp_variant="A", exp_num=1, label="Baseline TP-A (session_open)",
        save_name="p18_exp1_tpA.csv",
    )
    tr2, te2 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
        tp_variant="B", exp_num=2, label="TP-B (session_open + 0.5×spike)",
        save_name="p18_exp2_tpB.csv",
    )
    tr3, te3 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
        tp_variant="C", exp_num=3, label="TP-C (opposite spike extreme)",
        save_name="p18_exp3_tpC.csv",
    )

    # Pick best TP on test PF
    tp_results = {"A": (_pf(te1), tr1, te1), "B": (_pf(te2), tr2, te2),
                  "C": (_pf(te3), tr3, te3)}
    best_tp = max(tp_results, key=lambda k: tp_results[k][0])
    best_tp_pf = tp_results[best_tp][0]
    print(f"\n  TP comparison (test PF):  A={_fmt(_pf(te1))}  "
          f"B={_fmt(_pf(te2))}  C={_fmt(_pf(te3))}  → best: {best_tp}")

    # ── Experiment 4: Trailing stop ───────────────────────────────────────────
    tr4, te4 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
        use_trailing=True, trail_activate=1.0, trail_dist=0.75,
        exp_num=4, label="Trailing stop (activates at 1.0×ATR, trails 0.75×ATR)",
        save_name="p18_exp4_trail.csv",
    )
    use_trail_best = _pf(te4) > best_tp_pf and len(te4) >= 10
    print(f"\n  Trail vs fixed TP-{best_tp}: trail PF={_fmt(_pf(te4))}  "
          f"fixed PF={_fmt(best_tp_pf)}")
    print(f"  → {'Use trailing stop' if use_trail_best else f'Keep TP-{best_tp}'}")

    # For remaining experiments, use best exit method
    use_trailing_final = use_trail_best
    tp_final = best_tp if not use_trail_best else best_tp

    # ── Experiment 5: Spike size filter ──────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  EXPERIMENT 5 — Spike size filter")
    print(f"  TP-{tp_final}  trailing={use_trailing_final}  spike_mult: [1.5, 2.0, 2.5, 3.0]")
    print(f"{'═'*W}")
    print(f"  {'mult':>5} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6}")
    print(f"  {'-'*60}")

    best5_mult = 1.5; best5_pf = -1.0
    exp5_res = {}
    for mult in [1.5, 2.0, 2.5, 3.0]:
        tr_x = run(df_tr, si_tr, avg_tr, a1_tr, tp_variant=tp_final,
                   spike_mult=mult, use_trailing=use_trailing_final,
                   initial_capital=INITIAL_CAP)
        te_x = run(df_te, si_te, avg_te, a1_te, tp_variant=tp_final,
                   spike_mult=mult, use_trailing=use_trailing_final,
                   initial_capital=INITIAL_CAP)
        pf_te = _pf(te_x)
        print(f"  {mult:>5.1f} | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x)):>6} | "
              f"{_fmt(pf_te):>6} | {_sr(te_x):>6.3f}")
        exp5_res[mult] = (tr_x, te_x)
        if pf_te > best5_pf and len(te_x) >= 10:
            best5_pf = pf_te; best5_mult = mult
    print(f"  → Best spike_mult: {best5_mult}")

    # ── Experiment 6: Retest quality filter ───────────────────────────────────
    tr6, te6 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
        tp_variant=tp_final, spike_mult=best5_mult,
        use_trailing=use_trailing_final,
        retest_quality=True, retest_range_mult=0.8, retest_vol_mult=1.5,
        exp_num=6, label="Retest quality filter (range>0.8ATR, vol>1.5avg)",
        save_name="p18_exp6_quality.csv",
    )
    use_quality = _pf(te6) > _pf(exp5_res[best5_mult][1]) and len(te6) >= 10
    print(f"\n  Quality filter: PF={_fmt(_pf(te6))}  vs  no-filter={_fmt(_pf(exp5_res[best5_mult][1]))}")
    print(f"  → {'Use quality filter' if use_quality else 'Skip quality filter'}")

    # ── Experiment 7: POC confluence ──────────────────────────────────────────
    tr7, te7 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
        poc_tr=poc_tr, poc_te=poc_te,
        tp_variant=tp_final, spike_mult=best5_mult,
        use_trailing=use_trailing_final,
        retest_quality=use_quality,
        poc_filter=5.0,
        exp_num=7, label="POC confluence (spike_extreme within 5pts of prev_poc)",
        save_name="p18_exp7_poc.csv",
    )
    base_pf_for_poc = _pf(te6) if use_quality else _pf(exp5_res[best5_mult][1])
    use_poc = _pf(te7) > base_pf_for_poc and len(te7) >= 10
    poc_filter_final = 5.0 if use_poc else None
    print(f"\n  POC filter: PF={_fmt(_pf(te7))}  vs  no-poc={_fmt(base_pf_for_poc)}")
    print(f"  → {'Use POC filter' if use_poc else 'Skip POC filter'}")

    # ── Experiment 8: Hard R/R sensitivity ────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  EXPERIMENT 8 — Hard R/R filter sensitivity")
    print(f"  Best config so far  spike_mult={best5_mult}  TP-{tp_final}  "
          f"trailing={use_trailing_final}")
    print(f"{'═'*W}")
    print(f"  {'min_rr':>6} | {'N_tr':>5} | {'N_te':>5} | {'WR_te':>6} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6}")
    print(f"  {'-'*60}")

    best8_rr = 1.0; best8_pf = -1.0
    exp8_res = {}
    for mrr in [1.0, 1.5, 2.0]:
        tr_x = run(df_tr, si_tr, avg_tr, a1_tr, poc_tr,
                   tp_variant=tp_final, spike_mult=best5_mult,
                   use_trailing=use_trailing_final, min_rr=mrr,
                   retest_quality=use_quality, poc_filter_pts=poc_filter_final,
                   initial_capital=INITIAL_CAP)
        te_x = run(df_te, si_te, avg_te, a1_te, poc_te,
                   tp_variant=tp_final, spike_mult=best5_mult,
                   use_trailing=use_trailing_final, min_rr=mrr,
                   retest_quality=use_quality, poc_filter_pts=poc_filter_final,
                   initial_capital=INITIAL_CAP)
        pf_te = _pf(te_x)
        print(f"  {mrr:>6.1f} | {len(tr_x):>5} | {len(te_x):>5} | "
              f"{_wr(te_x)*100:>5.1f}% | {_fmt(_pf(tr_x)):>6} | "
              f"{_fmt(pf_te):>6} | {_sr(te_x):>6.3f}")
        exp8_res[mrr] = (tr_x, te_x)
        if pf_te > best8_pf and len(te_x) >= 10:
            best8_pf = pf_te; best8_rr = mrr
    print(f"  → Best min_rr: {best8_rr}")

    # ── Experiment 9: Combined best ───────────────────────────────────────────
    tr9, te9 = run_exp(
        df_tr, df_te, si_tr, si_te, avg_tr, avg_te, a1_tr, a1_te,
        poc_tr=poc_tr, poc_te=poc_te,
        tp_variant=tp_final, spike_mult=best5_mult,
        use_trailing=use_trailing_final,
        retest_quality=use_quality,
        poc_filter=poc_filter_final,
        min_rr=best8_rr,
        exp_num=9, label="Combined best configuration",
        save_name="p18_exp9_best.csv",
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print("  SUMMARY TABLE")
    print(f"  {'Exp':>3} | {'Exit':>8} | {'mult':>5} | {'qual':>4} | {'poc':>4} | "
          f"{'PF_tr':>6} | {'PF_te':>6} | {'SR_te':>6} | {'TPD':>5} | {'WR':>5}")
    print(f"  {'-'*80}")

    rows = [
        (1,  "TP-A",    1.5, "no",  "no",  tr1, te1),
        (2,  "TP-B",    1.5, "no",  "no",  tr2, te2),
        (3,  "TP-C",    1.5, "no",  "no",  tr3, te3),
        (4,  "trail",   1.5, "no",  "no",  tr4, te4),
        (5,  f"TP-{tp_final}", best5_mult, "no", "no",
             exp5_res[best5_mult][0], exp5_res[best5_mult][1]),
        (6,  f"TP-{tp_final}", best5_mult, "yes", "no",  tr6, te6),
        (7,  f"TP-{tp_final}", best5_mult,
             "yes" if use_quality else "no", "yes", tr7, te7),
        (8,  f"TP-{tp_final}", best5_mult,
             "yes" if use_quality else "no",
             "yes" if poc_filter_final else "no",
             exp8_res[best8_rr][0], exp8_res[best8_rr][1]),
        (9,  "trail" if use_trailing_final else f"TP-{tp_final}",
             best5_mult,
             "yes" if use_quality else "no",
             "yes" if poc_filter_final else "no", tr9, te9),
    ]
    for (en, ex, mult, qual, poc, trr, ter) in rows:
        print(f"  {en:>3} | {ex:>8} | {mult:>5.1f} | {qual:>4} | {poc:>4} | "
              f"{_fmt(_pf(trr)):>6} | {_fmt(_pf(ter)):>6} | "
              f"{_sr(ter):>6.3f} | {_tpd(ter, df_te):>5.3f} | "
              f"{_wr(ter)*100:>4.1f}%")

    # ── WFO decision ──────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    pf9_te = _pf(te9); n9_te = len(te9)
    print(f"  Exp 9 test: PF={_fmt(pf9_te)}  N={n9_te}")
    if pf9_te > 1.0 and n9_te >= 80:
        print("  → Thresholds met (PF>1.0 and N>=80). Running walk-forward.")
        best_cfg = dict(
            tp_variant=tp_final, spike_mult=best5_mult, vol_mult=1.2,
            retest_zone=0.5, min_rr=best8_rr,
            use_trailing=use_trailing_final, trail_activate=1.0, trail_dist=0.75,
            poc_filter=poc_filter_final, retest_quality=use_quality,
            retest_range_mult=0.8, retest_vol_mult=1.5,
            max_bars=90, max_entries=2, risk_pct=1.0,
        )
        run_wfo(df, atr1m, avg_vol, poc_df, best_cfg)
    else:
        reasons = []
        if pf9_te <= 1.0: reasons.append(f"PF={_fmt(pf9_te)}≤1.0")
        if n9_te < 80:    reasons.append(f"N={n9_te}<80")
        print(f"  → WFO skipped: {', '.join(reasons)}")
        print("  Do not proceed to FTMO sizing.")

    print(f"\n{'='*W}")
    print("  PHASE 18 COMPLETE")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
