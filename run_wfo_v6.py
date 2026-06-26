"""
Phase 6 — Walk-Forward Optimization of the HMM-guided combined system.

Architecture (3-phase search per window):
  Phase 1: ORB SHORT grid (432 combos), MR fixed at Phase 5 defaults.
  Phase 2: MR grid (81 combos), ORB fixed at Phase 1 winner.
  Phase 3: Top-5 ORB × Top-5 MR combined grid (25 combos).

5 anchored WFO windows.  HMM re-trained per window.
Metric: annualized trade-level Sharpe on training set.
"""

import itertools
import logging
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from orb_system.config import Config as _Cfg
from orb_system.data.loader import load_data, split_train_test
from orb_system.indicators.technical import add_indicators
from orb_system.regime.features import compute_daily_features, zscore_normalize
from orb_system.regime.hmm import RegimeHMM, FEATURE_COLS
from orb_system.strategy.mean_reversion import detect_rejection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
SLIP      = 0.25
COMM      = 4.0
PV        = 20.0
_EOD_MIN  = 15 * 60 + 45   # 945 — 15:45
_NE_MIN   = 14 * 60        # 840 — 14:00  (MR no-entry cutoff)
_OR_START = time_mod = None   # set in main after importing `time`

# post_bar numpy columns
_H = 0; _L = 1; _C = 2; _V = 3; _ATR = 4; _AVGV = 5; _CRNG = 6; _TMIN = 7

N_MIN_TRADES = 30  # minimum trades in training for a combo to be valid

# ── walk-forward windows ──────────────────────────────────────────────────────
WINDOWS = [
    ("2021-06-25", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2021-06-25", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2021-06-25", "2023-12-31", "2024-01-01", "2024-06-30"),
    ("2021-06-25", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2021-06-25", "2024-12-31", "2025-01-01", "2026-06-17"),
]

# ── parameter grids ───────────────────────────────────────────────────────────
ORB_SL    = [0.50, 0.75, 1.00, 1.25]
ORB_TP    = [1.50, 2.00, 2.50, 3.00]
ORB_VOL   = [1.00, 1.20, 1.50]
ORB_CRNG  = [1.00, 1.25, 1.50]
ORB_BARS  = [60, 90, 120]

MR_ENTRY  = [0.30, 0.40, 0.50]
# tp_type: 0=or_mid  1=or_extreme(far)  2=1.5x_or_from_entry
MR_TPTYPE = [0, 1, 2]
MR_SL     = [0.50, 0.75, 1.00]
MR_BARS   = [60, 90, 120]

# Phase 5 MR defaults (used while optimising ORB)
MR_BASE = dict(entry_pct=0.50, tp_type=1, sl_mult=0.50, max_bars=120)

# Phase 5 ORB defaults (used while optimising MR, overwritten per window)
ORB_BASE = dict(sl_mult=0.75, tp_mult=2.00, vol_mult=1.20, crng_mult=1.50, max_bars=120)


# ══════════════════════════════════════════════════════════════════════════════
# Session pre-computation
# ══════════════════════════════════════════════════════════════════════════════

_OR_START_T = None
_OR_END_T   = None


def _ensure_times():
    import datetime as _dt
    global _OR_START_T, _OR_END_T
    if _OR_START_T is None:
        _OR_START_T = _dt.time(9, 30)
        _OR_END_T   = _dt.time(10, 30)


def precompute_sessions(df: pd.DataFrame, regime_map: dict) -> list:
    """
    One dict per trading session:
      date, regime, or_hi, or_lo, or_mid, or_rng, atr_or_end,
      rej_long (bool), rej_short (bool),
      post_bars (np.ndarray shape [N, 8]: H,L,C,V,ATR,AVGV,CRNG,TMIN),
      year (int)
    """
    _ensure_times()

    date_arr = np.array(df.index.date)
    u_dates, first_idx, counts = np.unique(date_arr, return_index=True, return_counts=True)

    sessions = []
    for ui in range(len(u_dates)):
        d   = u_dates[ui]
        s   = int(first_idx[ui])
        e   = s + int(counts[ui])
        seg = df.iloc[s:e]
        bt  = seg.index.time

        or_mask = np.array([(t >= _OR_START_T and t < _OR_END_T) for t in bt])
        or_bars = seg[or_mask]
        if len(or_bars) < 2:
            continue

        or_hi  = float(or_bars["high"].max())
        or_lo  = float(or_bars["low"].min())
        or_rng = or_hi - or_lo
        if or_rng <= 0.0:
            continue
        or_mid     = (or_hi + or_lo) * 0.5
        atr_end    = float(or_bars["atr"].iloc[-1])
        if atr_end <= 0.0:
            continue

        # MR rejection detection (causal, OR bars only)
        rej = detect_rejection(or_bars)
        rej_l = bool(rej["long"]["valid"])
        rej_s = bool(rej["short"]["valid"])

        # Post-OR bars → numpy [H, L, C, V, ATR, AVGV, CRNG, TMIN]
        post_mask = np.array([t >= _OR_END_T for t in bt])
        post_seg  = seg[post_mask]
        if post_seg.empty:
            continue

        n_p = len(post_seg)
        pb  = np.empty((n_p, 8), dtype=np.float64)
        pb[:, _H]    = post_seg["high"].values
        pb[:, _L]    = post_seg["low"].values
        pb[:, _C]    = post_seg["close"].values
        pb[:, _V]    = post_seg["volume"].values
        pb[:, _ATR]  = post_seg["atr"].values
        pb[:, _AVGV] = post_seg["avg_vol"].values
        pb[:, _CRNG] = post_seg["candle_rng"].values
        pb[:, _TMIN] = np.array([t.hour * 60 + t.minute for t in post_seg.index.time],
                                 dtype=np.float64)

        sessions.append({
            "date":    d,
            "regime":  regime_map.get(d, "volatile"),
            "or_hi":   or_hi,
            "or_lo":   or_lo,
            "or_mid":  or_mid,
            "or_rng":  or_rng,
            "atr_end": atr_end,
            "rej_l":   rej_l,
            "rej_s":   rej_s,
            "pb":      pb,
            "year":    post_seg.index[0].year,
        })

    return sessions


# ══════════════════════════════════════════════════════════════════════════════
# Fast simulation functions
# ══════════════════════════════════════════════════════════════════════════════

def _exit_short(pb_rem, sl_px: float, tp_px: float, max_bars: int):
    """Return (exit_px, reason) for a SHORT trade. pb_rem starts AFTER entry bar."""
    n = len(pb_rem)
    lim = min(n, max_bars)
    for j in range(lim):
        h = pb_rem[j, _H]; l = pb_rem[j, _L]
        c = pb_rem[j, _C]; t = pb_rem[j, _TMIN]
        if h >= sl_px:
            return sl_px + SLIP, "sl"
        if l <= tp_px:
            return tp_px + SLIP, "tp"
        if t >= _EOD_MIN:
            return c + SLIP, "eod"
        if j == lim - 1 and j < max_bars - 1:  # ran out of bars before timeout
            return c + SLIP, "eod"
    # Exact timeout or end-of-array
    j = lim - 1
    reason = "timeout" if lim == max_bars and lim <= n else "eod"
    return pb_rem[j, _C] + SLIP, reason


def _exit_long(pb_rem, sl_px: float, tp_px: float, max_bars: int):
    """Return (exit_px, reason) for a LONG trade. pb_rem starts AFTER entry bar.
    MR LONG: no slip on SL/TP exits (matches mean_reversion.py _exit logic).
    EOD/timeout: close - SLIP.
    """
    n = len(pb_rem)
    lim = min(n, max_bars)
    for j in range(lim):
        h = pb_rem[j, _H]; l = pb_rem[j, _L]
        c = pb_rem[j, _C]; t = pb_rem[j, _TMIN]
        if l <= sl_px:
            return sl_px, "sl"       # MR: no slip on SL exit
        if h >= tp_px:
            return tp_px, "tp"       # MR: no slip on TP exit
        if t >= _EOD_MIN:
            return c - SLIP, "eod"
        if j == lim - 1 and j < max_bars - 1:
            return c - SLIP, "eod"
    j = lim - 1
    reason = "timeout" if lim == max_bars and lim <= n else "eod"
    return pb_rem[j, _C] - SLIP, reason


def _exit_mr_short(pb_rem, sl_px: float, tp_px: float, max_bars: int):
    """MR SHORT exit: no slip on SL/TP (mirrors _exit_long)."""
    n = len(pb_rem)
    lim = min(n, max_bars)
    for j in range(lim):
        h = pb_rem[j, _H]; l = pb_rem[j, _L]
        c = pb_rem[j, _C]; t = pb_rem[j, _TMIN]
        if h >= sl_px:
            return sl_px, "sl"
        if l <= tp_px:
            return tp_px, "tp"
        if t >= _EOD_MIN:
            return c + SLIP, "eod"
        if j == lim - 1 and j < max_bars - 1:
            return c + SLIP, "eod"
    j = lim - 1
    reason = "timeout" if lim == max_bars and lim <= n else "eod"
    return pb_rem[j, _C] + SLIP, reason


def sim_orb_short(sess: dict, sl_mult: float, tp_mult: float,
                  vol_mult: float, crng_mult: float, max_bars: int):
    """
    Simulate ORB SHORT for one 'trending' session.
    Returns pnl_net (float) or None.
    """
    if sess["regime"] != "trending":
        return None

    pb    = sess["pb"]
    or_lo = sess["or_lo"]
    n     = len(pb)
    if n == 0:
        return None

    # Vectorised entry scan
    valid = (pb[:, _TMIN] < _EOD_MIN) & (pb[:, _ATR] > 0) & (pb[:, _AVGV] > 0)
    sig   = valid & (pb[:, _C] < or_lo) & \
            (pb[:, _CRNG] > crng_mult * pb[:, _ATR]) & \
            (pb[:, _V] > vol_mult * pb[:, _AVGV])
    idx   = np.where(sig)[0]
    if idx.size == 0:
        return None

    ei     = int(idx[0])
    e_atr  = float(pb[ei, _ATR])
    e_px   = float(pb[ei, _C]) - SLIP
    sl_px  = e_px + sl_mult * e_atr
    tp_px  = e_px - tp_mult * e_atr

    pb_rem = pb[ei + 1:]
    if len(pb_rem) == 0:
        return (e_px - (float(pb[ei, _C]) + SLIP)) * PV - COMM

    ep, _ = _exit_short(pb_rem, sl_px, tp_px, max_bars)
    return (e_px - ep) * PV - COMM


def sim_mr(sess: dict, entry_pct: float, tp_type: int,
           sl_mult: float, max_bars: int):
    """
    Simulate MR (LONG or SHORT) for one 'ranging' session.
    One trade per session; LONG priority over SHORT on the same bar.
    Returns pnl_net (float) or None.
    """
    if sess["regime"] != "ranging":
        return None

    pb    = sess["pb"]
    or_lo = sess["or_lo"]
    or_hi = sess["or_hi"]
    or_mid = sess["or_mid"]
    or_rng = sess["or_rng"]
    rej_l  = sess["rej_l"]
    rej_s  = sess["rej_s"]
    if not rej_l and not rej_s:
        return None

    n = len(pb)
    if n == 0:
        return None

    trig_l = or_lo + entry_pct * or_rng if rej_l else np.inf
    trig_s = or_hi - entry_pct * or_rng if rej_s else -np.inf

    # guard and TP for LONG
    if tp_type == 0:      # or_mid
        guard_l = or_mid; tp_l_fixed = or_mid; tp_l_dyn = False
    elif tp_type == 1:    # or_extreme
        guard_l = or_hi;  tp_l_fixed = or_hi;  tp_l_dyn = False
    else:                 # 1.5x_or_from_entry
        guard_l = or_hi;  tp_l_fixed = 0.0;    tp_l_dyn = True

    # guard and TP for SHORT
    if tp_type == 0:
        guard_s = or_mid; tp_s_fixed = or_mid; tp_s_dyn = False
    elif tp_type == 1:
        guard_s = or_lo;  tp_s_fixed = or_lo;  tp_s_dyn = False
    else:
        guard_s = or_lo;  tp_s_fixed = 0.0;    tp_s_dyn = True

    c    = pb[:, _C]
    atr  = pb[:, _ATR]
    tmin = pb[:, _TMIN]

    # entry scan: LONG
    ei_l = n  # infinity
    if rej_l:
        long_sig = (tmin < _NE_MIN) & (atr > 0) & (c >= trig_l) & (c < guard_l)
        arr = np.where(long_sig)[0]
        if arr.size:
            ei_l = int(arr[0])

    # entry scan: SHORT
    ei_s = n
    if rej_s:
        short_sig = (tmin < _NE_MIN) & (atr > 0) & (c <= trig_s) & (c > guard_s)
        arr = np.where(short_sig)[0]
        if arr.size:
            ei_s = int(arr[0])

    if ei_l > ei_s:
        # SHORT fires first (or only SHORT)
        ei     = ei_s
        e_atr  = float(atr[ei])
        e_px   = float(c[ei]) - SLIP
        tp_px  = (e_px - 1.5 * or_rng) if tp_s_dyn else tp_s_fixed
        sl_px  = e_px + sl_mult * e_atr
        if e_px <= tp_px:
            return None
        pb_rem = pb[ei + 1:]
        if len(pb_rem) == 0:
            return (e_px - (float(c[ei]) + SLIP)) * PV - COMM
        ep, _ = _exit_mr_short(pb_rem, sl_px, tp_px, max_bars)
        return (e_px - ep) * PV - COMM

    elif ei_l < n:
        # LONG fires first (or ties: LONG wins)
        ei     = ei_l
        e_atr  = float(atr[ei])
        e_px   = float(c[ei]) + SLIP
        tp_px  = (e_px + 1.5 * or_rng) if tp_l_dyn else tp_l_fixed
        sl_px  = e_px - sl_mult * e_atr
        if e_px >= tp_px:
            return None
        pb_rem = pb[ei + 1:]
        if len(pb_rem) == 0:
            return (float(c[ei]) - SLIP - e_px) * PV - COMM
        ep, _ = _exit_long(pb_rem, sl_px, tp_px, max_bars)
        return (ep - e_px) * PV - COMM

    return None  # no entry


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

_NAN = float("nan")


def _metrics(pnl_list: list) -> dict:
    n = len(pnl_list)
    if n < 1:
        return {"n": 0, "sharpe": -9.0, "pf": 0.0, "wr": 0.0, "ret_pct": 0.0}
    arr = np.asarray(pnl_list, dtype=float)
    w   = arr[arr > 0]; l = arr[arr <= 0]
    gw  = float(w.sum()) if w.size else 0.0
    gl  = float(abs(l.sum())) if l.size else 0.0
    pf  = gw / gl if gl > 0 else (_NAN if gw == 0 else float("inf"))
    wr  = float(w.size) / n
    std = float(arr.std())
    sr  = float(arr.mean() / std * math.sqrt(252)) if std > 0 else 0.0
    ret = float(arr.sum()) / 100_000.0 * 100.0
    return {"n": n, "sharpe": sr, "pf": pf, "wr": wr, "ret_pct": ret}


def _select_top(rows: list, n: int = 5) -> list:
    """Select top-n by (sharpe_tr desc, n_tr desc), filtering < N_MIN_TRADES."""
    valid = [r for r in rows if r["n_tr"] >= N_MIN_TRADES]
    valid.sort(key=lambda r: (r["sharpe_tr"], r["n_tr"]), reverse=True)
    return valid[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Grid runners
# ══════════════════════════════════════════════════════════════════════════════

def _run_orb_grid(sess_tr, sess_te, mr_p, win_idx, phase_label, t0):
    """Phase 1: iterate ORB combos with MR fixed."""
    orb_combos = [
        (sl, tp, vol, crng, bars)
        for sl, tp, vol, crng, bars in itertools.product(
            ORB_SL, ORB_TP, ORB_VOL, ORB_CRNG, ORB_BARS
        )
        if tp > sl   # R/R restriction
    ]
    total = len(orb_combos)
    results = []
    for i, (sl, tp, vol, crng, bars) in enumerate(orb_combos):
        pnl_tr = [sim_orb_short(s, sl, tp, vol, crng, bars) for s in sess_tr]
        pnl_tr = [p for p in pnl_tr if p is not None]
        pnl_mr_tr = [sim_mr(s, mr_p["entry_pct"], mr_p["tp_type"],
                             mr_p["sl_mult"], mr_p["max_bars"]) for s in sess_tr]
        pnl_mr_tr = [p for p in pnl_mr_tr if p is not None]
        m_tr = _metrics(pnl_tr)

        pnl_te = [sim_orb_short(s, sl, tp, vol, crng, bars) for s in sess_te]
        pnl_te = [p for p in pnl_te if p is not None]
        m_te = _metrics(pnl_te)

        results.append({
            "sl": sl, "tp": tp, "vol": vol, "crng": crng, "bars": bars,
            "n_tr": m_tr["n"], "sharpe_tr": m_tr["sharpe"], "pf_tr": m_tr["pf"],
            "n_te": m_te["n"], "sharpe_te": m_te["sharpe"], "pf_te": m_te["pf"],
        })
        if (i + 1) % 72 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1) / 60.0
            print(f"\r  V{win_idx}/5 | {phase_label} | {i+1}/{total} | ETA {eta:.1f}min  ",
                  end="", flush=True)
    print()
    return results


def _run_mr_grid(sess_tr, sess_te, orb_p, win_idx, t0):
    """Phase 2: iterate MR combos with ORB fixed."""
    mr_combos = list(itertools.product(MR_ENTRY, MR_TPTYPE, MR_SL, MR_BARS))
    total = len(mr_combos)
    results = []
    for i, (ep, tpt, slm, bars) in enumerate(mr_combos):
        pnl_mr_tr = [sim_mr(s, ep, tpt, slm, bars) for s in sess_tr]
        pnl_mr_tr = [p for p in pnl_mr_tr if p is not None]
        m_tr = _metrics(pnl_mr_tr)

        pnl_mr_te = [sim_mr(s, ep, tpt, slm, bars) for s in sess_te]
        pnl_mr_te = [p for p in pnl_mr_te if p is not None]
        m_te = _metrics(pnl_mr_te)

        results.append({
            "entry_pct": ep, "tp_type": tpt, "sl_mult": slm, "max_bars": bars,
            "n_tr": m_tr["n"], "sharpe_tr": m_tr["sharpe"], "pf_tr": m_tr["pf"],
            "n_te": m_te["n"], "sharpe_te": m_te["sharpe"], "pf_te": m_te["pf"],
        })
        if (i + 1) % 27 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1) / 60.0
            print(f"\r  V{win_idx}/5 | Phase2-MR | {i+1}/{total} | ETA {eta:.1f}min  ",
                  end="", flush=True)
    print()
    return results


def _run_combined_grid(sess_tr, sess_te, top5_orb, top5_mr, win_idx, t0):
    """Phase 3: top-5 ORB × top-5 MR combined."""
    combos = list(itertools.product(range(len(top5_orb)), range(len(top5_mr))))
    total  = len(combos)
    results = []
    for i, (oi, mi) in enumerate(combos):
        op = top5_orb[oi]; mp = top5_mr[mi]
        pnl_tr = []
        for s in sess_tr:
            p_orb = sim_orb_short(s, op["sl"], op["tp"], op["vol"], op["crng"], op["bars"])
            p_mr  = sim_mr(s, mp["entry_pct"], mp["tp_type"], mp["sl_mult"], mp["max_bars"])
            if p_orb is not None: pnl_tr.append(p_orb)
            if p_mr  is not None: pnl_tr.append(p_mr)
        m_tr = _metrics(pnl_tr)

        pnl_te = []
        for s in sess_te:
            p_orb = sim_orb_short(s, op["sl"], op["tp"], op["vol"], op["crng"], op["bars"])
            p_mr  = sim_mr(s, mp["entry_pct"], mp["tp_type"], mp["sl_mult"], mp["max_bars"])
            if p_orb is not None: pnl_te.append(p_orb)
            if p_mr  is not None: pnl_te.append(p_mr)
        m_te = _metrics(pnl_te)

        results.append({
            "orb_rank": oi, "mr_rank": mi,
            "sl": op["sl"], "tp": op["tp"], "vol": op["vol"],
            "crng": op["crng"], "orb_bars": op["bars"],
            "entry_pct": mp["entry_pct"], "tp_type": mp["tp_type"],
            "sl_mr": mp["sl_mult"], "mr_bars": mp["max_bars"],
            "n_tr": m_tr["n"], "sharpe_tr": m_tr["sharpe"], "pf_tr": m_tr["pf"],
            "n_te": m_te["n"], "sharpe_te": m_te["sharpe"], "pf_te": m_te["pf"],
            "ret_te": m_te["ret_pct"],
        })
        if (i + 1) % 5 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1) / 60.0
            print(f"\r  V{win_idx}/5 | Phase3-Combo | {i+1}/{total} | ETA {eta:.1f}min  ",
                  end="", flush=True)
    print()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Per-window runner
# ══════════════════════════════════════════════════════════════════════════════

_TP_TYPE_NAMES = {0: "or_mid", 1: "or_extreme", 2: "1.5x_or_from_entry"}


def run_window(win_idx, tr_start, tr_end, te_start, te_end, df_ind, feat_norm):
    """Run one WFO window. Returns dict of best parameters."""
    W = 66
    print("\n" + "=" * W)
    print(f"  VENTANA {win_idx}/5")
    print(f"  Train: {tr_start} -> {tr_end}")
    print(f"  Test : {te_start} -> {te_end}")
    print("=" * W)

    tr_s = pd.Timestamp(tr_start).date()
    tr_e = pd.Timestamp(tr_end).date()
    te_s = pd.Timestamp(te_start).date()
    te_e = pd.Timestamp(te_end).date()

    # Filter df to window dates
    date_arr = np.array(df_ind.index.date)
    df_tr = df_ind[(date_arr >= tr_s) & (date_arr <= tr_e)]
    df_te = df_ind[(date_arr >= te_s) & (date_arr <= te_e)]

    # Re-train HMM on window training features only
    feat_idx = np.array(feat_norm.index)
    fn_tr_mask = (feat_idx >= tr_s) & (feat_idx <= tr_e)
    fn_te_mask = (feat_idx >= te_s) & (feat_idx <= te_e)
    fn_win_mask = (feat_idx >= tr_s) & (feat_idx <= te_e)

    feat_norm_tr  = feat_norm[fn_tr_mask]
    feat_norm_win = feat_norm[fn_win_mask]

    log.info("V%d — Training HMM (%d days train)...", win_idx, fn_tr_mask.sum())
    hmm_win = RegimeHMM(n_states=3, random_state=42)
    hmm_win.fit(feat_norm_tr)

    regime_ser = hmm_win.predict_regimes(feat_norm_win)
    regime_map = {d: lbl for d, lbl in zip(regime_ser.index, regime_ser.values)}

    # Pre-compute sessions
    log.info("V%d — Precomputing sessions...", win_idx)
    sess_tr = precompute_sessions(df_tr, regime_map)
    sess_te = precompute_sessions(df_te, regime_map)

    n_tr_all   = len(sess_tr)
    n_te_all   = len(sess_te)
    n_tr_trend = sum(1 for s in sess_tr if s["regime"] == "trending")
    n_tr_range = sum(1 for s in sess_tr if s["regime"] == "ranging")
    n_te_trend = sum(1 for s in sess_te if s["regime"] == "trending")
    n_te_range = sum(1 for s in sess_te if s["regime"] == "ranging")
    print(f"  Sessions — Train: {n_tr_all} (trending={n_tr_trend} ranging={n_tr_range})"
          f"  | Test: {n_te_all} (trending={n_te_trend} ranging={n_te_range})")

    t0 = time.time()

    # ── Phase 1: ORB grid ─────────────────────────────────────────────────────
    print(f"\n  FASE 1 — ORB SHORT optimization  ({4*4*3*3*3} combos)")
    orb_rows = _run_orb_grid(sess_tr, sess_te, MR_BASE, win_idx, "Phase1-ORB", t0)
    top5_orb = _select_top(orb_rows, 5)

    print(f"  Top 5 ORB (train Sharpe):")
    print(f"  {'Rk':>2} | {'SL':>4} | {'TP':>4} | {'Vol':>4} | {'Crng':>4} | {'Bars':>4}"
          f" | {'ShTr':>5} | {'PFTr':>5} | {'NTr':>4} | {'ShTe':>5} | {'PFTe':>5}")
    print("  " + "-" * 68)
    for rk, r in enumerate(top5_orb, 1):
        pf_s = f"{r['pf_tr']:.2f}" if not math.isnan(r['pf_tr']) else "  inf"
        print(f"  {rk:>2} | {r['sl']:>4.2f} | {r['tp']:>4.2f} | {r['vol']:>4.2f}"
              f" | {r['crng']:>4.2f} | {int(r['bars']):>4}"
              f" | {r['sharpe_tr']:>5.2f} | {pf_s:>5}"
              f" | {r['n_tr']:>4} | {r['sharpe_te']:>5.2f} | {r['pf_te']:>5.2f}")

    best_orb = top5_orb[0] if top5_orb else ORB_BASE
    print(f"\n  Seleccionado: SL={best_orb['sl']} TP={best_orb['tp']}"
          f" Vol={best_orb['vol']} Crng={best_orb['crng']} Bars={int(best_orb['bars'])}")

    # ── Phase 2: MR grid ──────────────────────────────────────────────────────
    print(f"\n  FASE 2 — MR LONG optimization  ({3*3*3*3} combos)")
    mr_rows = _run_mr_grid(sess_tr, sess_te, best_orb, win_idx, t0)
    top5_mr = _select_top(mr_rows, 5)

    print(f"  Top 5 MR (train Sharpe):")
    print(f"  {'Rk':>2} | {'Ent%':>5} | {'TP type':>16} | {'SL':>4} | {'Bars':>4}"
          f" | {'ShTr':>5} | {'PFTr':>5} | {'NTr':>4} | {'ShTe':>5} | {'PFTe':>5}")
    print("  " + "-" * 78)
    for rk, r in enumerate(top5_mr, 1):
        pf_s = f"{r['pf_tr']:.2f}" if not math.isnan(r['pf_tr']) else "  inf"
        tpt_s = _TP_TYPE_NAMES.get(int(r['tp_type']), "?")
        print(f"  {rk:>2} | {r['entry_pct']:>5.2f} | {tpt_s:>16}"
              f" | {r['sl_mult']:>4.2f} | {int(r['max_bars']):>4}"
              f" | {r['sharpe_tr']:>5.2f} | {pf_s:>5}"
              f" | {r['n_tr']:>4} | {r['sharpe_te']:>5.2f} | {r['pf_te']:>5.2f}")

    best_mr = top5_mr[0] if top5_mr else MR_BASE
    tpt_name = _TP_TYPE_NAMES.get(int(best_mr['tp_type']), "?")
    print(f"\n  Seleccionado: entry_pct={best_mr['entry_pct']}"
          f" tp_type={tpt_name} sl={best_mr['sl_mult']} bars={int(best_mr['max_bars'])}")

    # ── Phase 3: combined top5×top5 ───────────────────────────────────────────
    print(f"\n  FASE 3 — Combined grid  ({len(top5_orb)}×{len(top5_mr)} = {len(top5_orb)*len(top5_mr)} combos)")
    combo_rows = _run_combined_grid(sess_tr, sess_te, top5_orb, top5_mr, win_idx, t0)
    combo_rows_sorted = sorted(combo_rows, key=lambda r: (r["sharpe_tr"], r["n_tr"]), reverse=True)
    best_combo = combo_rows_sorted[0] if combo_rows_sorted else None

    if best_combo:
        tpt_name = _TP_TYPE_NAMES.get(int(best_combo['tp_type']), "?")
        print(f"  Mejor combinacion (train Sharpe={best_combo['sharpe_tr']:.3f}):")
        print(f"    ORB: SL={best_combo['sl']} TP={best_combo['tp']} Vol={best_combo['vol']}"
              f" Crng={best_combo['crng']} Bars={int(best_combo['orb_bars'])}")
        print(f"    MR : entry={best_combo['entry_pct']} tp={tpt_name}"
              f" sl={best_combo['sl_mr']} bars={int(best_combo['mr_bars'])}")
        print(f"  Train: Sharpe={best_combo['sharpe_tr']:.3f} PF={best_combo['pf_tr']:.3f}"
              f" N={best_combo['n_tr']}")
        print(f"  Test : Sharpe={best_combo['sharpe_te']:.3f} PF={best_combo['pf_te']:.3f}"
              f" N={best_combo['n_te']} Ret={best_combo['ret_te']:+.1f}%")

    # Save window CSV (all combo results)
    all_rows = (
        [{"phase": 1, **r} for r in orb_rows] +
        [{"phase": 2, **r} for r in mr_rows] +
        [{"phase": 3, **r} for r in combo_rows]
    )
    pd.DataFrame(all_rows).to_csv(
        RESULTS_DIR / f"wfo_v6_ventana_{win_idx}.csv", index=False
    )
    log.info("V%d — Saved results/wfo_v6_ventana_%d.csv", win_idx, win_idx)

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\n  RESULTADO FINAL VENTANA {win_idx}:")
    if best_combo:
        print(f"  Sharpe test: {best_combo['sharpe_te']:.3f}"
              f" | PF test: {best_combo['pf_te']:.3f}"
              f" | Ret test: {best_combo['ret_te']:+.1f}%"
              f" | Trades: {best_combo['n_te']}")
    print(f"  Tiempo: {elapsed_min:.1f} min")
    print("=" * W)

    if best_combo:
        return {
            "win": win_idx,
            "tr_start": tr_start, "tr_end": tr_end,
            "te_start": te_start, "te_end": te_end,
            # ORB params
            "sl_orb":  best_combo["sl"],
            "tp_orb":  best_combo["tp"],
            "vol":     best_combo["vol"],
            "crng":    best_combo["crng"],
            "bars_orb": int(best_combo["orb_bars"]),
            # MR params
            "entry_pct": best_combo["entry_pct"],
            "tp_type":   int(best_combo["tp_type"]),
            "sl_mr":     best_combo["sl_mr"],
            "bars_mr":   int(best_combo["mr_bars"]),
            # metrics
            "sharpe_tr": best_combo["sharpe_tr"],
            "sharpe_te": best_combo["sharpe_te"],
            "pf_tr":     best_combo["pf_tr"],
            "pf_te":     best_combo["pf_te"],
            "n_te":      best_combo["n_te"],
        }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Stability analysis + final backtest
# ══════════════════════════════════════════════════════════════════════════════

def _print_stability(window_results: list):
    W = 80
    print("\n" + "=" * W)
    print("  ANALISIS DE ESTABILIDAD")
    print("=" * W)

    params = ["sl_orb", "tp_orb", "vol", "crng", "bars_orb",
              "entry_pct", "tp_type", "sl_mr", "bars_mr"]
    param_labels = [
        "sl_atr (ORB)", "tp_atr (ORB)", "vol_mult",
        "crng_mult", "max_bars (ORB)",
        "entry_pct (MR)", "tp_type (MR)", "sl_mult (MR)", "max_bars (MR)",
    ]

    print(f"  {'Parametro':<20}", end="")
    for r in window_results:
        print(f" | V{r['win']:>1}", end="")
    print(f" | Moda  | Estable?")
    print("  " + "-" * 20 + ("-----" * len(window_results)) + "-+---------+---------")

    for pk, pl in zip(params, param_labels):
        vals = [r[pk] for r in window_results if r is not None]
        print(f"  {pl:<20}", end="")
        for r in window_results:
            v = r[pk] if r is not None else "?"
            if pk == "tp_type":
                print(f" |  {_TP_TYPE_NAMES.get(int(v), '?')[:3]}", end="")
            elif isinstance(v, float):
                print(f" |{v:>5.2f}", end="")
            else:
                print(f" |{int(v):>5}", end="")

        if vals:
            from collections import Counter
            cnt = Counter([str(v) for v in vals])
            mode_val, mode_cnt = cnt.most_common(1)[0]
            stable = "Si" if mode_cnt >= 3 else "No"
            print(f" |  {mode_val:>5}  |  {stable}")
        else:
            print(" |  N/A   |  No")

    # WFO summary table
    print(f"\n  {'V':>1} | {'Sh Tr':>7} | {'Sh Te':>7} | {'PF Tr':>6} | {'PF Te':>6} | {'N Te':>6}")
    print("  " + "-" * 46)
    sr_te_vals = []
    pf_te_vals = []
    n_te_vals  = []
    for r in window_results:
        if r is None:
            print(f"  ? | {'N/A':>7} | {'N/A':>7} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6}")
            continue
        print(f"  {r['win']:>1} | {r['sharpe_tr']:>7.3f} | {r['sharpe_te']:>7.3f}"
              f" | {r['pf_tr']:>6.3f} | {r['pf_te']:>6.3f} | {r['n_te']:>6}")
        sr_te_vals.append(r["sharpe_te"])
        pf_te_vals.append(r["pf_te"])
        n_te_vals.append(r["n_te"])

    print("  " + "-" * 46)
    if sr_te_vals:
        print(f"  {'mu':>1} | {'':>7} | {np.mean(sr_te_vals):>7.3f}"
              f" | {'':>6} | {np.mean(pf_te_vals):>6.3f} | {int(np.mean(n_te_vals)):>6}")
    print("=" * W)

    return np.mean(sr_te_vals) if sr_te_vals else 0.0, np.mean(pf_te_vals) if pf_te_vals else 0.0


def _run_final_backtest(df_ind, feat_norm, best_params, split_date_str="2024-12-16"):
    """
    Full backtest using best_params (from V5) on complete train and test periods.
    """
    W = 66
    print("\n" + "=" * W)
    print("  BACKTEST FINAL — parametros optimos de V5")
    print("=" * W)

    split_date = pd.Timestamp(split_date_str).date()
    date_arr   = np.array(df_ind.index.date)
    df_tr = df_ind[date_arr <= split_date]
    df_te = df_ind[date_arr > split_date]

    # Re-train HMM on full training set (same as Phase 5)
    feat_idx = np.array(feat_norm.index)
    feat_norm_tr = feat_norm[feat_idx <= split_date]

    hmm_final = RegimeHMM(n_states=3, random_state=42)
    hmm_final.fit(feat_norm_tr)

    regime_ser = hmm_final.predict_regimes(feat_norm)
    regime_map = {d: lbl for d, lbl in zip(regime_ser.index, regime_ser.values)}

    sess_tr = precompute_sessions(df_tr, regime_map)
    sess_te = precompute_sessions(df_te, regime_map)

    sl_o = best_params["sl_orb"]; tp_o = best_params["tp_orb"]
    vol  = best_params["vol"];    crng = best_params["crng"]
    bo   = best_params["bars_orb"]
    ep   = best_params["entry_pct"]; tpt = best_params["tp_type"]
    slm  = best_params["sl_mr"];     bm  = best_params["bars_mr"]

    tpt_name = _TP_TYPE_NAMES.get(tpt, "?")
    print(f"\n  ORB: SL={sl_o} TP={tp_o} Vol={vol} Crng={crng} Bars={bo}")
    print(f"  MR : entry={ep} tp={tpt_name} sl={slm} bars={bm}")

    def _run_period(sessions, label):
        pnl_orb = []; pnl_mr = []
        years = {}
        for s in sessions:
            p_orb = sim_orb_short(s, sl_o, tp_o, vol, crng, bo)
            p_mr  = sim_mr(s, ep, tpt, slm, bm)
            if p_orb is not None:
                pnl_orb.append(p_orb)
                years.setdefault(s["year"], []).append(("orb", p_orb))
            if p_mr is not None:
                pnl_mr.append(p_mr)
                years.setdefault(s["year"], []).append(("mr", p_mr))
        pnl_all = pnl_orb + pnl_mr
        m = _metrics(pnl_all)
        m_o = _metrics(pnl_orb)
        m_m = _metrics(pnl_mr)

        print(f"\n  {label}:")
        print(f"    Trades: {m['n']}  (ORB: {m_o['n']} | MR: {m_m['n']})")
        print(f"    Sharpe: {m['sharpe']:+.3f}  | PF: {m['pf']:.3f}"
              f"  | Win%: {m['wr']:.1%}  | Ret: {m['ret_pct']:+.1f}%")
        print(f"    Desglose anual:")
        print(f"    {'Ano':>4} | {'N':>5} | {'Sharpe':>7} | {'PF':>5} | {'Ret%':>7}")
        print("    " + "-" * 35)
        for yr in sorted(years):
            ypnl = [p for _, p in years[yr]]
            ym = _metrics(ypnl)
            pf_s = f"{ym['pf']:.2f}" if not math.isnan(ym["pf"]) else " inf"
            print(f"    {yr:>4} | {ym['n']:>5} | {ym['sharpe']:>7.3f}"
                  f" | {pf_s:>5} | {ym['ret_pct']:>+7.1f}%")
        return m, pnl_all, years

    m_tr, pnl_tr, _ = _run_period(sess_tr, "TRAINING")
    m_te, pnl_te, y_te = _run_period(sess_te, "TEST")

    # 2025 vs 2026 breakdown
    pnl_2025 = [p for _, p in y_te.get(2025, [])] if 2025 in y_te else []
    pnl_2026 = [p for _, p in y_te.get(2026, [])] if 2026 in y_te else []
    m25 = _metrics(pnl_2025); m26 = _metrics(pnl_2026)
    print(f"\n  Test por subperiodo:")
    print(f"    2025: N={m25['n']} Sharpe={m25['sharpe']:+.3f} PF={m25['pf']:.3f} Ret={m25['ret_pct']:+.1f}%")
    print(f"    2026: N={m26['n']} Sharpe={m26['sharpe']:+.3f} PF={m26['pf']:.3f} Ret={m26['ret_pct']:+.1f}%")

    print("=" * W)
    return m_tr, m_te, m25, m26


# ══════════════════════════════════════════════════════════════════════════════
# Verdict
# ══════════════════════════════════════════════════════════════════════════════

def _print_verdict(mean_sr_te, mean_pf_te, stable_count, window_results):
    W = 70
    print("\n" + "=" * W)
    print("  VEREDICTO FINAL — PHASE 6 WFO")
    print("=" * W)
    print(f"  Sharpe medio WF test     : {mean_sr_te:+.3f}  (umbral: > 0.50)")
    print(f"  PF medio WF test         : {mean_pf_te:.3f}  (umbral: > 1.10)")
    print(f"  Parametros estables      : {stable_count}/9  (umbral: >= 6/9)")

    sr_ok = mean_sr_te > 0.50
    pf_ok = mean_pf_te > 1.10
    st_ok = stable_count >= 6

    vs_phase5 = f"Phase 5 baseline Sharpe test=+0.04  WFO mean={mean_sr_te:+.3f}"
    print(f"\n  vs Phase 5               : {vs_phase5}")

    ready = sr_ok and pf_ok and st_ok
    verdict = "LISTO para adaptacion FTMO" if ready else "NO listo — revisar parametros"
    met = sum([sr_ok, pf_ok, st_ok])
    print(f"\n  Criterios cumplidos      : {met}/3")
    print(f"  Sharpe > 0.5             : {'SI' if sr_ok else 'NO'}")
    print(f"  PF > 1.1                 : {'SI' if pf_ok else 'NO'}")
    print(f"  Parametros estables >= 6 : {'SI' if st_ok else 'NO'}")
    print(f"\n  VEREDICTO: {verdict}")
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global time
    import time as time_mod_import
    time = time_mod_import

    # ── load data (once) ──────────────────────────────────────────────────────
    cfg    = _Cfg()
    df_raw = load_data(cfg)
    df_ind = add_indicators(df_raw, cfg)
    log.info("Dataset: %d bars", len(df_ind))

    # ── compute features (once, causal) ──────────────────────────────────────
    log.info("Computing daily features...")
    feat_raw  = compute_daily_features(df_ind)
    feat_norm = zscore_normalize(feat_raw, window=20)
    log.info("Features computed: %d days", len(feat_raw))

    # ── 5 WFO windows ────────────────────────────────────────────────────────
    all_params = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(WINDOWS, start=1):
        result = run_window(wi, tr_s, tr_e, te_s, te_e, df_ind, feat_norm)
        all_params.append(result)

    # ── stability analysis ────────────────────────────────────────────────────
    valid_results = [r for r in all_params if r is not None]
    mean_sr_te, mean_pf_te = _print_stability(valid_results)

    # count stable parameters (mode in >= 3/5 windows)
    from collections import Counter
    params_keys = ["sl_orb", "tp_orb", "vol", "crng", "bars_orb",
                   "entry_pct", "tp_type", "sl_mr", "bars_mr"]
    stable_count = 0
    for pk in params_keys:
        vals = [str(r[pk]) for r in valid_results]
        if vals:
            cnt = Counter(vals)
            if cnt.most_common(1)[0][1] >= 3:
                stable_count += 1

    # save summary CSV
    pd.DataFrame(valid_results).to_csv(RESULTS_DIR / "wfo_v6_summary.csv", index=False)

    # ── final backtest with V5 params ─────────────────────────────────────────
    best_v5 = next((r for r in all_params if r is not None and r["win"] == 5), None)
    if best_v5 is None and valid_results:
        best_v5 = valid_results[-1]

    if best_v5:
        pd.DataFrame([best_v5]).to_csv(RESULTS_DIR / "wfo_v6_best_params.csv", index=False)
        m_tr, m_te, m25, m26 = _run_final_backtest(df_ind, feat_norm, best_v5)
    else:
        log.warning("No valid V5 params — skipping final backtest")
        m_te = {"sharpe": 0.0, "pf": 0.0}
        m25 = m26 = {"sharpe": 0.0, "pf": 0.0}

    _print_verdict(mean_sr_te, mean_pf_te, stable_count, valid_results)
    log.info("Phase 6 complete. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
