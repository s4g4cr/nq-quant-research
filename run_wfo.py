"""
NQ ORB — Walk-Forward Optimization (Phase 4).

Anchored expanding-window WFO across 5 windows.
Grid: 1,080 valid combos (TP > SL restriction applied).
Fast engine: precomputed numpy session arrays, vectorized exits.

Run with:
    python run_wfo.py
"""

import logging
import sys
import time
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import add_indicators

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_SIZE    = 0.25
POINT_VALUE  = 20.0
COMMISSION   = 4.0        # round-trip
SLIP_TICKS   = 1
SLIP         = SLIP_TICKS * TICK_SIZE
EOD_MIN      = 15 * 60 + 45   # 945  (15:45)
OR_END_MIN   = 10 * 60 + 30   # 630  (10:30)
GAP_THRESH   = 5.0
MIN_TRAIN_TRADES = 30

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

SL_VALS   = [0.75, 1.0, 1.5, 2.0]
TP_VALS   = [1.0,  1.5, 2.0, 3.0]
VOL_VALS  = [1.2,  1.5, 2.0]
RNG_VALS  = [0.5,  1.0, 1.5]
BAR_VALS  = [60, 120, 240]
CTX_VALS  = ["none", "prev_session", "gap_5pts", "both"]


def build_grid():
    combos = []
    for sl, tp, vol, rng, mb, ctx in product(
        SL_VALS, TP_VALS, VOL_VALS, RNG_VALS, BAR_VALS, CTX_VALS
    ):
        if tp > sl:
            combos.append((sl, tp, vol, rng, mb, ctx))
    return combos


GRID = build_grid()   # 1,080 combos

# ---------------------------------------------------------------------------
# Walk-forward windows
# ---------------------------------------------------------------------------

WINDOWS = [
    {
        "num": 1,
        "train_start": date(2021, 6, 25),
        "train_end":   date(2022, 12, 31),
        "test_start":  date(2023, 1,  1),
        "test_end":    date(2023, 6, 30),
    },
    {
        "num": 2,
        "train_start": date(2021, 6, 25),
        "train_end":   date(2023, 6, 30),
        "test_start":  date(2023, 7,  1),
        "test_end":    date(2023, 12, 31),
    },
    {
        "num": 3,
        "train_start": date(2021, 6, 25),
        "train_end":   date(2023, 12, 31),
        "test_start":  date(2024, 1,  1),
        "test_end":    date(2024, 6, 30),
    },
    {
        "num": 4,
        "train_start": date(2021, 6, 25),
        "train_end":   date(2024, 6, 30),
        "test_start":  date(2024, 7,  1),
        "test_end":    date(2024, 12, 31),
    },
    {
        "num": 5,
        "train_start": date(2021, 6, 25),
        "train_end":   date(2024, 12, 31),
        "test_start":  date(2025, 1,  1),
        "test_end":    date(2026, 6, 17),
    },
]

# ---------------------------------------------------------------------------
# Precompute sessions
# ---------------------------------------------------------------------------

def precompute_sessions(df: pd.DataFrame) -> list:
    """
    Extract per-session numpy arrays from the full indicator-enriched DataFrame.
    Context signals (prev_return, gap) computed causally.
    Uses vectorized groupby — avoids per-day df filtering loop.
    """
    # Add integer minute-of-day column once (vectorized)
    idx_times   = df.index
    time_min_all = (idx_times.hour * 60 + idx_times.minute).values.astype(np.int32)
    date_all     = np.array(idx_times.date)   # Python date objects, shape (n,)

    # Columns as numpy arrays for direct access
    opens_all   = df["open"].values.astype(np.float64)
    highs_all   = df["high"].values.astype(np.float64)
    lows_all    = df["low"].values.astype(np.float64)
    closes_all  = df["close"].values.astype(np.float64)
    volumes_all = df["volume"].values.astype(np.float64)
    atrs_all    = df["atr"].values.astype(np.float64)
    avgvol_all  = df["avg_vol"].values.astype(np.float64)

    # Build date -> (start_idx, end_idx) index using np.unique
    unique_dates, first_occ, counts = np.unique(
        date_all, return_index=True, return_counts=True
    )

    # Pass 1: collect daily_open_0930 and daily_close_1545 for context
    daily_open_0930  = {}
    daily_close_1545 = {}

    for i, d in enumerate(unique_dates):
        s = first_occ[i]
        e = s + counts[i]
        tm  = time_min_all[s:e]
        cls = closes_all[s:e]
        opn = opens_all[s:e]

        # open at 09:30 (= 570 min)
        mask_930 = tm == 570
        if mask_930.any():
            daily_open_0930[d] = float(opn[mask_930][0])

        # last close at or before 15:45 (= 945 min)
        mask_1545 = tm <= 945
        if mask_1545.any():
            daily_close_1545[d] = float(cls[mask_1545][-1])

    dates_ordered = sorted(daily_close_1545.keys())

    # Pass 2: build session dicts
    # Map date -> index in unique_dates for fast lookup
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    sessions = []

    for seq_idx, d in enumerate(dates_ordered):
        uidx = date_to_idx.get(d)
        if uidx is None:
            continue
        s = first_occ[uidx]
        e = s + counts[uidx]

        tm = time_min_all[s:e]

        # OR bars: 570 <= time < 630
        or_mask  = (tm >= 570) & (tm < 630)
        if not or_mask.any():
            continue
        or_high = float(highs_all[s:e][or_mask].max())
        or_low  = float(lows_all[s:e][or_mask].min())
        if or_high <= or_low:
            continue

        # Post-OR bars: time >= 630
        post_mask = tm >= 630
        if not post_mask.any():
            continue

        post_sl = slice(s, e)
        pm      = post_mask

        times_min = tm[pm]
        highs     = highs_all[post_sl][pm]
        lows      = lows_all[post_sl][pm]
        closes    = closes_all[post_sl][pm]
        volumes   = volumes_all[post_sl][pm]
        atrs      = atrs_all[post_sl][pm]
        avg_vols  = avgvol_all[post_sl][pm]

        # Context signals
        prev_return = np.nan
        gap         = np.nan
        if seq_idx > 0:
            prev_d = dates_ordered[seq_idx - 1]
            pc = daily_close_1545.get(prev_d, np.nan)
            po = daily_open_0930.get(prev_d, np.nan)
            to = daily_open_0930.get(d, np.nan)
            if not (np.isnan(pc) or np.isnan(po)):
                prev_return = pc - po
            if not (np.isnan(pc) or np.isnan(to)):
                gap = to - pc

        sessions.append({
            "date":      d,
            "or_high":   or_high,
            "or_low":    or_low,
            "prev_ret":  prev_return,
            "gap":       gap,
            "times":     times_min,
            "highs":     highs,
            "lows":      lows,
            "closes":    closes,
            "volumes":   volumes,
            "atrs":      atrs,
            "avg_vols":  avg_vols,
            "n":         int(pm.sum()),
        })

    return sessions


def filter_sessions(sessions: list, start: date, end: date) -> list:
    return [s for s in sessions if start <= s["date"] <= end]

# ---------------------------------------------------------------------------
# Fast single-combo backtest
# ---------------------------------------------------------------------------

def run_combo(sessions, sl_m, tp_m, vol_m, rng_m, max_bars, ctx,
              collect_pnls=False):
    """
    Returns (pf, n_trades, n_wins, total_pnl_usd, [pnl_list if collect_pnls]).
    pf = 999.0 if no losers and some winners; 0.0 if no trades.
    """
    gross_win  = 0.0
    gross_loss = 0.0
    n_trades   = 0
    n_wins     = 0
    pnl_list   = [] if collect_pnls else None

    for sess in sessions:
        or_high  = sess["or_high"]
        or_low   = sess["or_low"]
        prev_ret = sess["prev_ret"]
        gap_val  = sess["gap"]

        allow_l = True
        allow_s = True

        if ctx in ("prev_session", "both"):
            if np.isnan(prev_ret) or prev_ret == 0.0:
                continue
            if prev_ret > 0.0:
                allow_s = False
            else:
                allow_l = False

        if ctx in ("gap_5pts", "both"):
            if np.isnan(gap_val):
                continue
            if gap_val > GAP_THRESH:
                allow_s = False
            elif gap_val < -GAP_THRESH:
                allow_l = False
            else:
                continue   # gap too small

        if not allow_l and not allow_s:
            continue

        highs    = sess["highs"]
        lows     = sess["lows"]
        closes   = sess["closes"]
        volumes  = sess["volumes"]
        atrs     = sess["atrs"]
        avg_vols = sess["avg_vols"]
        times    = sess["times"]
        n        = sess["n"]

        candle_rng = highs - lows

        # Guard against zero ATR/avgvol
        with np.errstate(invalid="ignore", divide="ignore"):
            rng_ok  = candle_rng > (rng_m * atrs)
            vol_ok  = volumes    > (vol_m * avg_vols)

        pre_eod = times < EOD_MIN
        sig_ok  = rng_ok & vol_ok & pre_eod

        long_sig  = sig_ok & (closes > or_high) if allow_l else np.zeros(n, dtype=bool)
        short_sig = sig_ok & (closes < or_low)  if allow_s else np.zeros(n, dtype=bool)

        has_l = bool(long_sig.any())
        has_s = bool(short_sig.any())

        if not has_l and not has_s:
            continue

        fi_l = int(np.argmax(long_sig))  if has_l else n
        fi_s = int(np.argmax(short_sig)) if has_s else n

        if fi_l <= fi_s:
            entry_idx = fi_l
            direction = 1
        else:
            entry_idx = fi_s
            direction = -1

        atr_e = atrs[entry_idx]
        if atr_e <= 0.0:
            continue

        entry_close = closes[entry_idx]
        if direction == 1:
            entry_px = entry_close + SLIP
            sl_px    = entry_px - sl_m * atr_e
            tp_px    = entry_px + tp_m * atr_e
        else:
            entry_px = entry_close - SLIP
            sl_px    = entry_px + sl_m * atr_e
            tp_px    = entry_px - tp_m * atr_e

        # Exit management (vectorized)
        start = entry_idx + 1
        end   = min(start + max_bars, n)
        exit_px = None

        if start < end:
            h = highs[start:end]
            l = lows[start:end]
            c = closes[start:end]
            t = times[start:end]

            if direction == 1:
                sl_hit = l <= sl_px
                tp_hit = h >= tp_px
            else:
                sl_hit = h >= sl_px
                tp_hit = l <= tp_px

            eod_hit  = t >= EOD_MIN
            combined = sl_hit | tp_hit | eod_hit

            if combined.any():
                j = int(np.argmax(combined))
                if sl_hit[j]:
                    exit_px = sl_px            # SL wins (even if TP also hit)
                elif tp_hit[j]:
                    exit_px = tp_px
                else:
                    # EOD exit
                    exit_px = c[j] - SLIP if direction == 1 else c[j] + SLIP

        if exit_px is None:
            # Timeout or ran out of bars
            last = end - 1
            exit_px = closes[last] - SLIP if direction == 1 else closes[last] + SLIP

        pnl_pts = (exit_px - entry_px) * direction
        pnl_usd = pnl_pts * POINT_VALUE - COMMISSION

        n_trades += 1
        if pnl_usd > 0.0:
            n_wins     += 1
            gross_win  += pnl_usd
        else:
            gross_loss += abs(pnl_usd)

        if collect_pnls:
            pnl_list.append(pnl_usd)

    if n_trades == 0:
        pf = 0.0
    elif gross_loss == 0.0:
        pf = 999.0
    else:
        pf = gross_win / gross_loss

    total_pnl = gross_win - gross_loss
    return (pf, n_trades, n_wins, total_pnl, pnl_list)

# ---------------------------------------------------------------------------
# Run one WFO window
# ---------------------------------------------------------------------------

def run_window(sessions_all, win, total_combos):
    num        = win["num"]
    tr_start   = win["train_start"]
    tr_end     = win["train_end"]
    te_start   = win["test_start"]
    te_end     = win["test_end"]

    sess_train = filter_sessions(sessions_all, tr_start, tr_end)
    sess_test  = filter_sessions(sessions_all, te_start, te_end)

    print(f"\n{'=' * 62}")
    print(f"  VENTANA {num}  —  Train: {tr_start} -> {tr_end}  |  Test: {te_start} -> {te_end}")
    print(f"{'=' * 62}")
    print(f"  Sesiones train: {len(sess_train)}   |   Sesiones test: {len(sess_test)}")
    print(f"  Evaluando {total_combos} combinaciones...")

    rows = []
    t0   = time.time()

    for i, (sl, tp, vol, rng, mb, ctx) in enumerate(GRID):
        if i > 0 and i % 100 == 0:
            elapsed  = time.time() - t0
            eta_sec  = elapsed / i * (total_combos - i)
            print(f"    [{i:>5}/{total_combos}]  ETA: {eta_sec/60:.1f} min", end="\r", flush=True)

        pf_tr, n_tr, nw_tr, pnl_tr, _ = run_combo(sess_train, sl, tp, vol, rng, mb, ctx)
        rows.append({
            "sl": sl, "tp": tp, "vol": vol, "rng": rng, "bars": mb, "ctx": ctx,
            "pf_train": pf_tr, "n_train": n_tr,
        })

    print(f"    [{total_combos}/{total_combos}]  Completado en {time.time()-t0:.1f}s  " + " " * 10)

    df_res = pd.DataFrame(rows)

    # Filter: >= MIN_TRAIN_TRADES
    n_discarded = (df_res["n_train"] < MIN_TRAIN_TRADES).sum()
    valid       = df_res[df_res["n_train"] >= MIN_TRAIN_TRADES].copy()

    if valid.empty:
        print("  ERROR: ninguna combinacion alcanza el minimo de trades en train.")
        return None

    valid_sorted = valid.sort_values("pf_train", ascending=False).reset_index(drop=True)

    # Top 5 — also compute pf_test for display
    top5_rows = valid_sorted.head(5)
    top5_test = []
    for _, row in top5_rows.iterrows():
        pf_te, n_te, nw_te, pnl_te, _ = run_combo(
            sess_test, row.sl, row.tp, row.vol, row.rng, row.bars, row.ctx
        )
        top5_test.append((pf_te, n_te))

    print(f"\n  Combinaciones evaluadas : {total_combos}")
    print(f"  Descartadas (< {MIN_TRAIN_TRADES} trades train): {n_discarded}")
    print(f"\n  TOP 5 EN TRAIN:")
    print(f"  {'Rank':>4} | {'SL':>4} | {'TP':>4} | {'Vol':>4} | {'Rng':>4} | "
          f"{'Bars':>4} | {'Context':<12} | {'PF Tr':>7} | {'N Tr':>5} | {'PF Te':>7}")
    print(f"  {'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+"
          f"-{'-'*4}-+-{'-'*12}-+-{'-'*7}-+-{'-'*5}-+-{'-'*7}")

    for rank, (_, row) in enumerate(top5_rows.iterrows(), 1):
        pf_te, n_te = top5_test[rank - 1]
        print(
            f"  {rank:>4} | {row.sl:>4.2f} | {row.tp:>4.2f} | {row.vol:>4.1f} | "
            f"{row.rng:>4.1f} | {int(row.bars):>4d} | {row.ctx:<12} | "
            f"{row.pf_train:>7.3f} | {int(row.n_train):>5d} | {pf_te:>7.3f}"
        )

    # Best combo (rank 1)
    best = valid_sorted.iloc[0]
    sl_b, tp_b, vol_b, rng_b, mb_b, ctx_b = (
        best.sl, best.tp, best.vol, best.rng, best.bars, best.ctx
    )

    print(f"\n  PARAMETROS SELECCIONADOS:")
    print(f"    sl_atr_multiplier       : {sl_b}")
    print(f"    tp_atr_multiplier       : {tp_b}")
    print(f"    volume_multiplier       : {vol_b}")
    print(f"    candle_range_multiplier : {rng_b}")
    print(f"    max_bars_in_trade       : {int(mb_b)}")
    print(f"    context_filter          : {ctx_b}")

    # Full metrics on test
    pf_te_b, n_te_b, nw_te_b, pnl_te_b, pnls_te = run_combo(
        sess_test, sl_b, tp_b, vol_b, rng_b, mb_b, ctx_b, collect_pnls=True
    )
    wr_te  = nw_te_b / n_te_b * 100 if n_te_b > 0 else 0.0
    ret_te = pnl_te_b / 100_000.0 * 100 if n_te_b > 0 else 0.0
    sr_te  = float(np.mean(pnls_te) / np.std(pnls_te)) if len(pnls_te) > 1 else 0.0

    print(f"\n  RESULTADO EN TEST:")
    print(f"    PF Test    : {pf_te_b:.3f}")
    print(f"    Trades     : {n_te_b}")
    print(f"    Win Rate   : {wr_te:.1f}%")
    print(f"    Return     : {ret_te:+.1f}%")
    print(f"    Sharpe     : {sr_te:.2f}")
    print(f"{'=' * 62}")

    # Save per-window CSV
    df_res["window"] = num
    df_res.to_csv(RESULTS_DIR / f"wfo_ventana_{num}.csv", index=False)

    # Full metrics on train (for summary table)
    pf_tr_b, n_tr_b, nw_tr_b, pnl_tr_b, _ = run_combo(
        sess_train, sl_b, tp_b, vol_b, rng_b, mb_b, ctx_b
    )

    return {
        "num":       num,
        "tr_start":  tr_start,
        "tr_end":    tr_end,
        "te_start":  te_start,
        "te_end":    te_end,
        "sl":        sl_b,
        "tp":        tp_b,
        "vol":       vol_b,
        "rng":       rng_b,
        "bars":      int(mb_b),
        "ctx":       ctx_b,
        "pf_train":  pf_tr_b,
        "n_train":   n_tr_b,
        "pf_test":   pf_te_b,
        "n_test":    n_te_b,
        "wr_test":   wr_te,
        "ret_test":  ret_te,
        "sr_test":   sr_te,
    }

# ---------------------------------------------------------------------------
# Final analysis
# ---------------------------------------------------------------------------

def print_final_analysis(results):
    print("\n\n" + "=" * 70)
    print("  RESUMEN WALK-FORWARD — 5 VENTANAS")
    print("=" * 70)

    header = (f"  {'V':>1} | {'Train':^20} | {'Test':^20} | "
              f"{'PF Tr':>6} | {'PF Te':>6} | {'N Te':>5}")
    print(header)
    print("  " + "-" * 66)

    pf_train_all = []
    pf_test_all  = []
    n_test_all   = []

    for r in results:
        tr = f"{r['tr_start'].strftime('%Y-%m')} -> {r['tr_end'].strftime('%Y-%m')}"
        te = f"{r['te_start'].strftime('%Y-%m')} -> {r['te_end'].strftime('%Y-%m')}"
        print(f"  {r['num']:>1} | {tr:^20} | {te:^20} | "
              f"{r['pf_train']:>6.3f} | {r['pf_test']:>6.3f} | {r['n_test']:>5d}")
        pf_train_all.append(r["pf_train"])
        pf_test_all.append(r["pf_test"])
        n_test_all.append(r["n_test"])

    print("  " + "-" * 66)
    print(f"  {'MEDIA':>1}   {'':^20}   {'':^20}   "
          f"{np.mean(pf_train_all):>6.3f} | {np.mean(pf_test_all):>6.3f} | "
          f"{int(np.mean(n_test_all)):>5d}")

    # Parameter stability
    params = ["sl", "tp", "vol", "rng", "bars", "ctx"]
    labels = {
        "sl":   "sl_atr_multiplier      ",
        "tp":   "tp_atr_multiplier      ",
        "vol":  "volume_multiplier      ",
        "rng":  "candle_range_mult      ",
        "bars": "max_bars_in_trade      ",
        "ctx":  "context_filter         ",
    }

    print("\n\n  ESTABILIDAD DE PARAMETROS:")
    print(f"  {'Parametro':<24} | {'V1':>6} | {'V2':>6} | {'V3':>6} | {'V4':>6} | {'V5':>6} | {'Moda':>8} | {'Estable?':>8}")
    print("  " + "-" * 80)

    for p in params:
        vals = [str(r[p]) for r in results]
        from collections import Counter
        counts = Counter(vals)
        mode_val, mode_cnt = counts.most_common(1)[0]
        stable = "Si" if mode_cnt >= 3 else "No"
        row_vals = "  |  ".join(f"{v:>6}" for v in vals)
        print(f"  {labels[p]:<24} | {vals[0]:>6} | {vals[1]:>6} | {vals[2]:>6} | "
              f"{vals[3]:>6} | {vals[4]:>6} | {mode_val:>8} | {stable:>8}")

    # Verdict
    mean_pf_test  = np.mean(pf_test_all)
    n_above_1     = sum(1 for p in pf_test_all if p > 1.0)
    stables       = []
    for p in params:
        vals = [str(r[p]) for r in results]
        from collections import Counter
        c = Counter(vals)
        if c.most_common(1)[0][1] >= 3:
            stables.append(f"{p}={c.most_common(1)[0][0]}")

    print("\n\n  VEREDICTO FINAL:")
    print(f"  PF medio walk-forward (test): {mean_pf_test:.3f}")
    print(f"  Ventanas con PF test > 1.0  : {n_above_1}/5")
    if stables:
        print(f"  Parametros estables         : {', '.join(stables)}")
    else:
        print(f"  Parametros estables         : ninguno")

    if mean_pf_test > 1.0 and n_above_1 >= 3:
        conclusion = "EVIDENCIA POSITIVA: el PF medio supera 1.0 y >= 3 ventanas son rentables."
        recomendacion = "Proceder con parametros estables en paper trading con seguimiento estricto."
    elif mean_pf_test > 1.0 and n_above_1 >= 2:
        conclusion = "EVIDENCIA DEBIL: PF medio > 1.0 pero solo en pocas ventanas."
        recomendacion = "Continuar investigacion. Considerar filtros de regimen o hipotesis alternativas."
    else:
        conclusion = "SIN EVIDENCIA DE EDGE: el walk-forward no valida robustez OOS."
        recomendacion = "Cambiar de hipotesis. El ORB 1-min con estos parametros no generaliza."

    print(f"  Conclusion     : {conclusion}")
    print(f"  Recomendacion  : {recomendacion}")

    # Save summaries
    summary_rows = [{
        "ventana":    r["num"],
        "tr_start":   r["tr_start"],
        "tr_end":     r["tr_end"],
        "te_start":   r["te_start"],
        "te_end":     r["te_end"],
        "pf_train":   r["pf_train"],
        "pf_test":    r["pf_test"],
        "n_test":     r["n_test"],
        "wr_test":    r["wr_test"],
        "ret_test":   r["ret_test"],
        "sl":         r["sl"],
        "tp":         r["tp"],
        "vol":        r["vol"],
        "rng":        r["rng"],
        "bars":       r["bars"],
        "ctx":        r["ctx"],
    } for r in results]
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "wfo_summary.csv", index=False)

    # Best params = window 5 (most recent)
    best = results[-1]
    best_row = {k: v for k, v in best.items()}
    pd.DataFrame([best_row]).to_csv(RESULTS_DIR / "wfo_best_params.csv", index=False)

    print(f"\n  CSVs guardados en {RESULTS_DIR}/")
    print("=" * 70)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  NQ ORB — WALK-FORWARD OPTIMIZATION")
    print(f"  Grid: {len(GRID)} combinaciones validas (TP > SL)")
    print(f"  Ventanas: {len(WINDOWS)}  |  Min trades train: {MIN_TRAIN_TRADES}")
    print("=" * 70)

    # Load and prepare data
    cfg = Config()
    print("\nCargando datos...")
    df_raw = load_data(cfg, use_cache=True)
    print(f"  {len(df_raw):,} barras de 1 minuto | {df_raw.index[0]} -> {df_raw.index[-1]}")

    print("Calculando indicadores...")
    df = add_indicators(df_raw, cfg)

    print("Precomputando sesiones...")
    t0 = time.time()
    sessions_all = precompute_sessions(df)
    print(f"  {len(sessions_all)} sesiones precomputadas en {time.time()-t0:.1f}s")

    results = []
    for win in WINDOWS:
        res = run_window(sessions_all, win, len(GRID))
        if res is not None:
            results.append(res)

    if results:
        print_final_analysis(results)
    else:
        print("ERROR: ninguna ventana produjo resultados validos.")


if __name__ == "__main__":
    main()
