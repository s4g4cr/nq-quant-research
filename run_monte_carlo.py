"""
Phase 7 — Monte Carlo + FTMO Risk Sizing Optimization.

Uses OOS trades from Phase 6 final backtest (2024-12-16 -> 2026-06-17).
Finds the risk sizing that maximizes P(reach +10% before -10%)
respecting the daily -5% loss limit.
"""

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from orb_system.config import Config
from orb_system.data.loader import load_data
from orb_system.indicators.technical import add_indicators
from orb_system.backtester.engine import BacktestEngine
from orb_system.strategy.mean_reversion import MeanReversionEngine
from orb_system.strategy.orb import make_orb_short_config
from orb_system.regime.features import compute_daily_features, zscore_normalize
from orb_system.regime.hmm import RegimeHMM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── FTMO challenge parameters ──────────────────────────────────────────────────
CAPITAL_INIT   = 100_000.0
CAPITAL_TARGET = 110_000.0
CAPITAL_FLOOR  =  90_000.0
DAILY_LOSS_LIM =  -5_000.0
PV   = 20.0    # NQ point value $/pt
COMM = 4.0     # commission per contract round-trip

# ── Simulation parameters ──────────────────────────────────────────────────────
N_SIMS = 10_000
MAX_T  = 2_000   # max trades per simulation path

# ── Phase 6 V5 optimal parameters ─────────────────────────────────────────────
SPLIT_DATE = "2024-12-16"

V5_ORB = dict(sl_atr_multiplier=1.0, tp_atr_multiplier=1.5,
              volume_multiplier=1.5, candle_range_multiplier=1.5,
              max_bars_in_trade=60)

V5_MR = dict(sl_atr_mult=1.0, entry_pct=0.40,
             max_bars=60, tp_far_extreme=False)   # or_mid TP

# ── Grid parameters ────────────────────────────────────────────────────────────
RISK_LEVELS_FIXED = [0.0025, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150, 0.0200]

_RISK_BASE   = [0.0050, 0.0075, 0.0100, 0.0125]
_U1          = [0.03, 0.04, 0.05]
_U2          = [0.06, 0.07, 0.08]
_F_RED       = [0.50, 0.75]
_F_MIN       = [0.25, 0.50]


# ══════════════════════════════════════════════════════════════════════════════
# Trade generation
# ══════════════════════════════════════════════════════════════════════════════

def _load_regime_map() -> dict:
    p = RESULTS_DIR / "hmm_regime_labels.csv"
    df = pd.read_csv(p)
    dates = pd.to_datetime(df["date"]).dt.date
    return dict(zip(dates, df["regime"].values))


def generate_oos_trades(df_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Run BacktestEngine + MeanReversionEngine with V5 params on the test split.
    Filter ORB trades to 'trending' days, MR trades to 'ranging' days.
    Returns DataFrame with one row per trade.
    """
    log.info("Generating OOS trades with V5 params...")

    split    = pd.Timestamp(SPLIT_DATE).date()
    date_arr = np.array(df_ind.index.date)
    df_te    = df_ind[date_arr > split]
    log.info("Test period: %s -> %s  (%d bars)",
             df_te.index[0].date(), df_te.index[-1].date(), len(df_te))

    regime_map = _load_regime_map()

    # ── ORB SHORT ─────────────────────────────────────────────────────────────
    orb_cfg = make_orb_short_config(**V5_ORB)
    orb_eng = BacktestEngine(orb_cfg)
    orb_res = orb_eng.run(df_te, label="test_oos")
    log.info("ORB total trades: %d", len(orb_res.trades))

    orb_rows = []
    for t in orb_res.trades:
        d = t.entry_ts.date()
        if regime_map.get(d, "volatile") != "trending":
            continue
        # sl_pts = SL distance from entry in NQ points
        sl_pts = abs(t.sl_price - t.entry_price)
        if sl_pts <= 0:
            sl_pts = t.atr_at_entry * V5_ORB["sl_atr_multiplier"]
        orb_rows.append({
            "entry_ts":   t.entry_ts,
            "exit_ts":    t.exit_ts,
            "date":       d,
            "direction":  t.direction,
            "exit_reason": t.exit_reason,
            "pnl_pts":    t.pnl_points,
            "pnl_net":    t.pnl_net,
            "sl_pts":     sl_pts,
            "strategy":   "orb_short",
        })

    # ── MR ────────────────────────────────────────────────────────────────────
    mr_res = MeanReversionEngine.run(df_te, **V5_MR)
    log.info("MR total trades: %d", len(mr_res.trades))

    mr_rows = []
    for t in mr_res.trades:
        d = t.entry_ts.date()
        if regime_map.get(d, "volatile") != "ranging":
            continue
        sl_pts = abs(t.sl_px - t.entry_px)
        if sl_pts <= 0:
            sl_pts = t.atr_entry * V5_MR["sl_atr_mult"]
        mr_rows.append({
            "entry_ts":   t.entry_ts,
            "exit_ts":    t.exit_ts,
            "date":       d,
            "direction":  t.direction,
            "exit_reason": t.exit_reason,
            "pnl_pts":    t.pnl_pts,
            "pnl_net":    t.pnl_net,
            "sl_pts":     sl_pts,
            "strategy":   "mr",
        })

    df = pd.DataFrame(orb_rows + mr_rows).sort_values("entry_ts").reset_index(drop=True)
    log.info("OOS trades after regime filter: ORB=%d  MR=%d  Total=%d",
             len(orb_rows), len(mr_rows), len(df))

    csv_path = RESULTS_DIR / "wfo_v6_final_trades.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s", csv_path)
    return df


def load_or_generate_trades(df_ind: pd.DataFrame) -> pd.DataFrame:
    p = RESULTS_DIR / "wfo_v6_final_trades.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["entry_ts", "exit_ts"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        log.info("Loaded %d OOS trades from %s", len(df), p)
        return df
    return generate_oos_trades(df_ind)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Base statistics
# ══════════════════════════════════════════════════════════════════════════════

def _streaks(mask: np.ndarray) -> int:
    """Longest consecutive True streak in mask."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def print_base_stats(df: pd.DataFrame) -> None:
    W = 70
    print("\n" + "=" * W)
    print("  PASO 1 — ESTADISTICAS BASE: 193 TRADES OOS")
    print("=" * W)

    pts    = df["pnl_pts"].values
    net    = df["pnl_net"].values
    sl_pts = df["sl_pts"].values
    wins   = pts > 0
    losses = pts <= 0
    n      = len(df)

    print(f"\n  Total trades      : {n}")
    print(f"  Win rate global   : {wins.mean():.1%}  ({wins.sum()} W / {losses.sum()} L)")

    if "direction" in df.columns:
        for d in ["long", "short"]:
            m = df["direction"] == d
            if m.sum():
                wr = (pts[m.values] > 0).mean()
                print(f"  Win rate {d:<5}   : {wr:.1%}  (n={m.sum()})")

    avg_w = float(pts[wins].mean()) if wins.any() else 0.0
    avg_l = float(abs(pts[losses]).mean()) if losses.any() else 0.0
    rr    = avg_w / avg_l if avg_l > 0 else float("inf")
    print(f"\n  Avg winner (pts)  : {avg_w:+.2f}")
    print(f"  Avg loser  (pts)  : {-avg_l:+.2f}")
    print(f"  R/R medio realiz. : {rr:.2f}:1")
    print(f"  SL medio   (pts)  : {sl_pts.mean():.2f}")

    pctiles = np.percentile(pts, [5, 25, 50, 75, 95])
    print(f"\n  Distribucion pnl_pts:")
    print(f"    p5={pctiles[0]:+.1f}  p25={pctiles[1]:+.1f}  p50={pctiles[2]:+.1f}"
          f"  p75={pctiles[3]:+.1f}  p95={pctiles[4]:+.1f}")

    print(f"\n  Racha perdedora max: {_streaks(losses):d} trades consecutivos")
    print(f"  Racha ganadora max : {_streaks(wins):d} trades consecutivos")

    # Trades per month
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["entry_ts"], utc=True).dt.to_period("M")
    by_month = df2.groupby("month").size()
    print(f"\n  Trades/mes — media={by_month.mean():.1f}"
          f"  min={by_month.min()}  max={by_month.max()}")

    # Exit reasons
    if "exit_reason" in df.columns:
        er = df["exit_reason"].value_counts(normalize=True)
        print(f"\n  Salidas por razon:")
        for reason, pct in er.items():
            print(f"    {reason:<10}: {pct:.1%}")

    print(f"\n  Expectancy net (1 contrato)  : ${net.mean():+.2f}/trade")
    print(f"  PnL total 1 contrato         : ${net.sum():+,.0f}")
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo core
# ══════════════════════════════════════════════════════════════════════════════

def _prep_arrays(df: pd.DataFrame):
    """Pre-process trade arrays for fast simulation access."""
    df = df.sort_values("entry_ts").reset_index(drop=True)
    pts   = df["pnl_pts"].values.astype(np.float64)
    sl    = df["sl_pts"].values.astype(np.float64)
    sl    = np.where(sl > 0, sl, np.median(sl[sl > 0]))  # guard zeros

    # Encode dates as int indices (for fast daily-loss tracking)
    dates_raw = df["date"].values
    unique_d, date_enc = np.unique(dates_raw, return_inverse=True)
    date_enc = date_enc.astype(np.int32)
    n_dates = len(unique_d)

    return pts, sl, date_enc, n_dates


def simulate_batch(pts, sl, date_enc, n_dates, risk_fn,
                   n_sims, rng, max_t=MAX_T, track_n=0):
    """
    Core MC simulation.
    risk_fn(capital, sl_pts) -> n_contracts (int)
    Returns dict of result arrays.
    """
    n_trades = len(pts)
    idx      = rng.integers(0, n_trades, size=(n_sims, max_t), dtype=np.int32)

    p_succ   = np.zeros(n_sims, dtype=bool)
    p_ftot   = np.zeros(n_sims, dtype=bool)
    p_fdaily = np.zeros(n_sims, dtype=bool)
    n_t_res  = np.zeros(n_sims, dtype=np.int32)
    max_dd   = np.zeros(n_sims, dtype=np.float64)
    fin_cap  = np.full(n_sims, CAPITAL_INIT, dtype=np.float64)
    daily_buf = np.zeros(n_dates, dtype=np.float64)
    curves   = [] if track_n > 0 else None

    for si in range(n_sims):
        capital = CAPITAL_INIT
        daily_buf[:] = 0.0
        outcome = "timeout"
        curve = [capital] if (si < track_n) else None

        for ti in range(max_t):
            i    = int(idx[si, ti])
            p    = float(pts[i])
            s    = float(sl[i])
            d_i  = int(date_enc[i])

            n_c  = risk_fn(capital, s)
            pnl  = p * PV * n_c - COMM * n_c

            capital       += pnl
            daily_buf[d_i] += pnl

            if capital > CAPITAL_INIT:
                pass  # drawdown measured from initial
            dd = max(0.0, (CAPITAL_INIT - capital) / CAPITAL_INIT)
            if dd > max_dd[si]:
                max_dd[si] = dd

            if curve is not None:
                curve.append(capital)

            if capital >= CAPITAL_TARGET:
                outcome = "success"; break
            if capital <= CAPITAL_FLOOR:
                outcome = "fail_total"; break
            if daily_buf[d_i] <= DAILY_LOSS_LIM:
                outcome = "fail_daily"; break

        n_t_res[si]  = ti + 1
        fin_cap[si]  = capital
        p_succ[si]   = outcome == "success"
        p_ftot[si]   = outcome == "fail_total"
        p_fdaily[si] = outcome == "fail_daily"
        if si < track_n:
            curves.append(curve)

    r = {
        "success":     p_succ,
        "fail_total":  p_ftot,
        "fail_daily":  p_fdaily,
        "n_trades":    n_t_res,
        "max_dd":      max_dd,
        "final_cap":   fin_cap,
        "p_success":   float(p_succ.mean()),
        "p_fail_total": float(p_ftot.mean()),
        "p_fail_daily": float(p_fdaily.mean()),
    }
    if track_n > 0:
        r["curves"] = curves
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Fixed sizing
# ══════════════════════════════════════════════════════════════════════════════

def run_fixed(df: pd.DataFrame, rng) -> dict:
    """10,000 sims per risk level."""
    pts, sl, date_enc, n_dates = _prep_arrays(df)

    W = 70
    print("\n" + "=" * W)
    print("  PASO 2 — MONTE CARLO SIZING FIJO")
    print("=" * W)
    print(f"  {'Risk%':>6} | {'P(exit)':>7} | {'P(fail)':>7} | {'P(daily)':>8}"
          f" | {'DD med%':>7} | {'Trades':>6} | p50 cap")
    print("  " + "-" * 68)

    results = {}
    t0 = time.time()
    for rp in RISK_LEVELS_FIXED:
        def rfn(capital, sl_pts, rp=rp):
            sl_usd = capital * rp
            return max(1, int(sl_usd / (sl_pts * PV)))

        res = simulate_batch(pts, sl, date_enc, n_dates, rfn, N_SIMS, rng)

        succ_dd = res["max_dd"][res["success"]]
        dd_med = float(succ_dd.mean()) if succ_dd.size else 0.0
        n_t_med = float(res["n_trades"].mean())
        p50 = np.percentile(res["final_cap"], 50)

        print(f"  {rp:.2%}  | {res['p_success']:.1%}   | "
              f"{res['p_fail_total']:.1%}   | {res['p_fail_daily']:.1%}    "
              f"| {dd_med:.1%}   | {n_t_med:>6.0f} | ${p50:,.0f}")

        results[rp] = res
        results[rp]["dd_med_succ"] = dd_med
        results[rp]["n_trades_med"] = n_t_med

    print(f"\n  Tiempo: {(time.time()-t0):.1f}s")
    print("=" * W)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Dynamic sizing
# ══════════════════════════════════════════════════════════════════════════════

def run_dynamic(df: pd.DataFrame, rng) -> list:
    """144 param combos × 10,000 sims = 1,440,000 simulations."""
    import itertools
    pts, sl, date_enc, n_dates = _prep_arrays(df)

    combos = list(itertools.product(_RISK_BASE, _U1, _U2, _F_RED, _F_MIN))
    combos = [(rb, u1, u2, fr, fm) for rb, u1, u2, fr, fm in combos
              if u1 < u2]  # u1 must be < u2
    total = len(combos)

    print(f"\n  PASO 3 — MONTE CARLO SIZING DINAMICO  ({total} combos)")

    all_rows = []
    t0 = time.time()
    for ci, (rb, u1, u2, fr, fm) in enumerate(combos):
        def rfn(capital, sl_pts, rb=rb, u1=u1, u2=u2, fr=fr, fm=fm):
            dd = max(0.0, (CAPITAL_INIT - capital) / CAPITAL_INIT)
            if dd < u1:
                rp = rb
            elif dd < u2:
                rp = rb * fr
            else:
                rp = rb * fm
            sl_usd = capital * rp
            return max(1, int(sl_usd / (sl_pts * PV)))

        res = simulate_batch(pts, sl, date_enc, n_dates, rfn, N_SIMS, rng)
        succ_dd = res["max_dd"][res["success"]]
        dd_med = float(succ_dd.mean()) if succ_dd.size else 0.0

        all_rows.append({
            "risk_base": rb, "u1": u1, "u2": u2, "f_red": fr, "f_min": fm,
            "p_success": res["p_success"],
            "p_fail":    res["p_fail_total"] + res["p_fail_daily"],
            "p_fail_daily": res["p_fail_daily"],
            "p_fail_total": res["p_fail_total"],
            "dd_med":    dd_med,
            "n_trades_med": float(res["n_trades"].mean()),
        })

        if (ci + 1) % 18 == 0 or (ci + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (ci + 1) * (total - ci - 1) / 60.0
            print(f"\r  Combo {ci+1}/{total} | ETA {eta:.1f}min  ", end="", flush=True)

    print()
    df_dyn = pd.DataFrame(all_rows).sort_values("p_success", ascending=False)
    df_dyn.to_csv(RESULTS_DIR / "monte_carlo_dynamic_results.csv", index=False)

    W = 70
    print(f"\n  Top 10 combinaciones por P(exito):")
    print(f"  {'rb':>5} | {'u1':>4} | {'u2':>4} | {'f_r':>4} | {'f_m':>4}"
          f" | {'P(exit)':>7} | {'P(fail)':>7} | {'DD':>6} | {'Trades':>6}")
    print("  " + "-" * 65)
    for _, r in df_dyn.head(10).iterrows():
        print(f"  {r['risk_base']:.3f} | {r['u1']:.3f} | {r['u2']:.3f}"
              f" | {r['f_red']:.2f} | {r['f_min']:.2f}"
              f" | {r['p_success']:.1%}   | {r['p_fail']:.1%}   "
              f"| {r['dd_med']:.1%}  | {r['n_trades_med']:>6.0f}")
    print("=" * W)

    return df_dyn.to_dict("records")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(fixed_results: dict, dyn_rows: list, df_trades: pd.DataFrame,
                          best_dyn: dict, rng_seed: int = 99) -> None:
    W = 70
    print("\n" + "=" * W)
    print("  PASO 4 — ANALISIS DE SENSIBILIDAD")
    print("=" * W)

    # Q1: optimal fixed risk
    best_fixed_rp = max(fixed_results, key=lambda r: fixed_results[r]["p_success"])
    print(f"\n  1. Riesgo optimo (fijo): {best_fixed_rp:.2%}"
          f"  P(exito)={fixed_results[best_fixed_rp]['p_success']:.1%}")

    # Q2: dynamic vs fixed
    best_dyn_ps = best_dyn["p_success"]
    best_fix_ps = fixed_results[best_fixed_rp]["p_success"]
    delta = best_dyn_ps - best_fix_ps
    print(f"\n  2. Sizing dinamico vs fijo:")
    print(f"     Fijo optimo: {best_fix_ps:.1%}  |  Dinamico optimo: {best_dyn_ps:.1%}")
    print(f"     Diferencia: {delta:+.1%}  {'(mejor)' if delta > 0 else '(peor)'}")

    # Q3: where does P(daily fail) > 5%
    threshold_rp = None
    for rp in RISK_LEVELS_FIXED:
        if fixed_results[rp]["p_fail_daily"] > 0.05:
            threshold_rp = rp; break
    if threshold_rp:
        print(f"\n  3. P(fallo diario) > 5% a partir de: {threshold_rp:.2%}")
    else:
        print(f"\n  3. P(fallo diario) nunca supera 5% en el rango testado")

    # Q4: avg days (proxy: trades / 10 per month * 21 days)
    best_res = fixed_results[best_fixed_rp]
    avg_trades = best_res["n_trades_med"]
    avg_days   = avg_trades / 10.0 * 21  # ~10 trades/month, 21 trading days
    print(f"\n  4. Trades medios hasta resultado (sizing optimo): {avg_trades:.0f}")
    print(f"     Dias de trading estimados: ~{avg_days:.0f}")

    # Q5: max drawdown p95 during successful paths
    succ_dd = best_res["max_dd"][best_res["success"]]
    if succ_dd.size:
        dd_p95 = np.percentile(succ_dd, 95)
        dd_p50 = np.percentile(succ_dd, 50)
        print(f"\n  5. DD maximo en rutas exitosas (sizing optimo):")
        print(f"     Mediana: {dd_p50:.1%}  |  p95: {dd_p95:.1%}  (estres psicologico)")

    # Q6: stress test — 20% worse win rate
    print(f"\n  6. STRESS TEST: win rate degradado 20%")
    pts, sl, date_enc, n_dates = _prep_arrays(df_trades)
    wins_idx  = np.where(pts > 0)[0]
    loses_pts = pts[pts < 0]
    loses_sl  = sl[pts < 0]
    n_flip    = int(len(wins_idx) * 0.20)

    rng2 = np.random.default_rng(rng_seed)
    flip_sel = rng2.choice(len(wins_idx), size=n_flip, replace=False)
    replace_sel = rng2.integers(0, len(loses_pts), size=n_flip)

    pts_stress = pts.copy()
    sl_stress  = sl.copy()
    for fi, ri in zip(flip_sel, replace_sel):
        orig_i = wins_idx[fi]
        pts_stress[orig_i] = loses_pts[ri]
        sl_stress[orig_i]  = loses_sl[ri]

    orig_wr = float((pts > 0).mean())
    stress_wr = float((pts_stress > 0).mean())
    print(f"     WR original: {orig_wr:.1%}  WR stress: {stress_wr:.1%}")

    rng3 = np.random.default_rng(42)

    def rfn_best(capital, sl_pts, rp=best_fixed_rp):
        return max(1, int(capital * rp / (sl_pts * PV)))

    res_stress = simulate_batch(pts_stress, sl_stress, date_enc, n_dates,
                                rfn_best, N_SIMS, rng3)
    print(f"     P(exito) stress : {res_stress['p_success']:.1%}"
          f"  (original: {best_fix_ps:.1%})")
    print(f"     P(fallo)  stress: {res_stress['p_fail_total']+res_stress['p_fail_daily']:.1%}")

    print("=" * W)
    return best_fixed_rp, best_dyn_ps, best_fix_ps, avg_days


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Recommendation
# ══════════════════════════════════════════════════════════════════════════════

def print_recommendation(best_dyn: dict, best_fixed_rp: float,
                          fixed_results: dict, avg_days: float) -> None:
    bd = best_dyn
    rb = bd["risk_base"]
    u1 = bd["u1"]
    u2 = bd["u2"]
    fr = bd["f_red"]
    fm = bd["f_min"]

    W = 70
    print("\n" + "=" * W)
    print("  PASO 5 — RECOMENDACION FINAL PARA FTMO")
    print("=" * W)

    # Translate to approximate contract counts at $100k
    # Using median sl_pts from the trade data (passed indirectly via best_dyn)
    # Rule: n_c = capital * risk_pct / (sl_pts * 20)
    # Assume sl_pts ~ 15 pts (typical NQ ATR × 1.0)
    SL_APPROX = 15.0
    def n_c(capital, risk_pct):
        return max(1, int(capital * risk_pct / (SL_APPROX * PV)))

    n_green   = n_c(CAPITAL_INIT, rb)
    n_yellow  = n_c(CAPITAL_INIT, rb * fr)
    n_red     = n_c(CAPITAL_INIT, rb * fm)

    print(f"\n  SIZING DINAMICO OPTIMO:")
    print(f"  {'':=<60}")
    print(f"  Zona VERDE  (DD < {u1:.0%}) : risk={rb:.2%} ~ {n_green} contrato(s)")
    print(f"  Zona AMARILLA ({u1:.0%}-{u2:.0%}): risk={rb*fr:.2%} ~ {n_yellow} contrato(s)")
    print(f"  Zona ROJA   (DD > {u2:.0%}) : risk={rb*fm:.2%} ~ {n_red} contrato(s)")
    print(f"  {'':=<60}")
    print(f"  P(pasar fase)            : {bd['p_success']:.1%}")
    print(f"  Dias estimados           : ~{avg_days:.0f} dias de trading")
    print(f"  DD medio (rutas exitosas): {bd['dd_med']:.1%}")
    print(f"  Comparacion sizing fijo {best_fixed_rp:.2%}: "
          f"{bd['p_success'] - fixed_results[best_fixed_rp]['p_success']:+.1%} prob. exito")

    print(f"\n  REGLA OPERACIONAL SIMPLE:")
    print(f"  " + "-" * 57)
    print(f"  Calcula: DD = max(0, 1 - capital/$100k)")
    print(f"  Si DD < {u1:.0%}:  opera con {n_green} contrato(s)  (riesgo normal)")
    print(f"  Si DD {u1:.0%}-{u2:.0%}: opera con {n_yellow} contrato(s)  (riesgo reducido)")
    print(f"  Si DD > {u2:.0%}:  opera con {n_red} contrato(s)  (riesgo minimo)")
    print(f"  Revisa la zona al inicio de cada sesion de trading")
    print(f"  " + "-" * 57)

    print(f"\n  NOTA FTMO:")
    print(f"  Limite diario $5k = {abs(DAILY_LOSS_LIM / (n_green * SL_APPROX * PV)):.1f}x SL por dia (zona verde)")
    print(f"  Con {n_green} contrato(s): 1 trade SL = ${n_green * SL_APPROX * PV:.0f}"
          f" ({n_green * SL_APPROX * PV / abs(DAILY_LOSS_LIM):.1%} del limite diario)")
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_pct(x, pos):
    return f"{x/100_000-1:.0%}"


def plot_fixed_distributions(fixed_results: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(RISK_LEVELS_FIXED)))

    for col, rp in zip(colors, RISK_LEVELS_FIXED):
        fc = fixed_results[rp]["final_cap"]
        ax.hist(fc, bins=60, alpha=0.55, label=f"{rp:.2%}", color=col,
                density=True, range=(50_000, 150_000))

    ax.axvline(CAPITAL_FLOOR,  color="red",   lw=2, ls="--", label="Fallo $90k")
    ax.axvline(CAPITAL_TARGET, color="green", lw=2, ls="--", label="Exito $110k")
    ax.axvline(CAPITAL_INIT,   color="gray",  lw=1, ls=":",  label="Capital inicial $100k")
    ax.set_xlabel("Capital final ($)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion de Capital Final — Sizing Fijo (10,000 sims)")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    fig.savefig(RESULTS_DIR / "monte_carlo_fixed_sizing.png", dpi=120)
    plt.close()
    log.info("Saved monte_carlo_fixed_sizing.png")


def plot_success_rates(fixed_results: dict) -> None:
    rps  = [rp for rp in RISK_LEVELS_FIXED]
    ps   = [fixed_results[rp]["p_success"] for rp in rps]
    pfd  = [fixed_results[rp]["p_fail_daily"] for rp in rps]
    pft  = [fixed_results[rp]["p_fail_total"] for rp in rps]

    best_rp = max(rps, key=lambda r: fixed_results[r]["p_success"])
    x_labels = [f"{rp:.2%}" for rp in rps]

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(rps))
    bars = ax.bar(xs, ps, color=["#2ecc71" if rp == best_rp else "#3498db" for rp in rps],
                  alpha=0.85, label="P(exito)")
    ax.bar(xs, pfd, bottom=ps, color="#e74c3c", alpha=0.6, label="P(fallo diario)")
    ax.bar(xs, pft, bottom=[p + d for p, d in zip(ps, pfd)], color="#c0392b",
           alpha=0.6, label="P(fallo total)")

    ax.axhline(0.5, color="orange", lw=1.5, ls="--", label="50% threshold")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Probabilidad")
    ax.set_title("P(Exito) FTMO por Nivel de Riesgo Fijo")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig.savefig(RESULTS_DIR / "monte_carlo_success_rate.png", dpi=120)
    plt.close()
    log.info("Saved monte_carlo_success_rate.png")


def plot_equity_curves(pts, sl, date_enc, n_dates, best_rp: float, rng) -> dict:
    n_track = 200
    rp = best_rp

    def rfn(capital, sl_pts, rp=rp):
        return max(1, int(capital * rp / (sl_pts * PV)))

    res = simulate_batch(pts, sl, date_enc, n_dates, rfn, n_track, rng,
                         max_t=MAX_T, track_n=n_track)
    curves = res["curves"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for ci, (curve, succ) in enumerate(zip(curves, res["success"])):
        col   = "#27ae60" if succ else "#e74c3c"
        alpha = 0.25 if succ else 0.18
        ax.plot(curve, color=col, alpha=alpha, lw=0.8)

    ax.axhline(CAPITAL_TARGET, color="green", lw=2, ls="--", label="Objetivo $110k")
    ax.axhline(CAPITAL_INIT,   color="gray",  lw=1, ls=":",  label="Capital inicial $100k")
    ax.axhline(CAPITAL_FLOOR,  color="red",   lw=2, ls="--", label="Fallo $90k")

    ax.set_xlabel("Numero de Trades")
    ax.set_ylabel("Capital ($)")
    ax.set_title(f"Curvas de Equity — Sizing Fijo {best_rp:.2%}  (200 simulaciones)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.set_ylim(75_000, 125_000)
    fig.savefig(RESULTS_DIR / "monte_carlo_equity_curves.png", dpi=120)
    plt.close()
    log.info("Saved monte_carlo_equity_curves.png")
    return res


def plot_drawdown_distribution(fixed_results: dict, best_rp: float) -> None:
    succ_dd = fixed_results[best_rp]["max_dd"][fixed_results[best_rp]["success"]]
    if succ_dd.size == 0:
        return

    pctiles = np.percentile(succ_dd, [25, 50, 75, 95])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(succ_dd * 100, bins=50, color="#3498db", alpha=0.75, edgecolor="white")

    for pct_v, pct_lbl, col in zip(pctiles, ["p25", "p50", "p75", "p95"],
                                    ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]):
        ax.axvline(pct_v * 100, color=col, lw=2,
                   label=f"{pct_lbl}: {pct_v:.1%}")

    ax.set_xlabel("Drawdown maximo (%)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"DD Maximo en Rutas Exitosas — Sizing {best_rp:.2%}")
    ax.legend()
    fig.savefig(RESULTS_DIR / "drawdown_distribution.png", dpi=120)
    plt.close()
    log.info("Saved drawdown_distribution.png")


def plot_dynamic_vs_fixed(fixed_results: dict, dyn_rows: list, best_rp: float) -> None:
    fix_ps = {rp: fixed_results[rp]["p_success"] for rp in RISK_LEVELS_FIXED}
    best_fix_ps = fix_ps[best_rp]
    dyn_top10_ps = [r["p_success"] for r in dyn_rows[:10]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fixed sizing
    xs = np.arange(len(RISK_LEVELS_FIXED))
    colors = ["#2ecc71" if rp == best_rp else "#3498db" for rp in RISK_LEVELS_FIXED]
    ax1.bar(xs, [fix_ps[rp] for rp in RISK_LEVELS_FIXED], color=colors, alpha=0.85)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{rp:.2%}" for rp in RISK_LEVELS_FIXED], fontsize=9)
    ax1.set_title("Sizing Fijo — P(Exito)")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_ylim(0, 1)

    # Right: top-10 dynamic vs best fixed
    y_vals = dyn_top10_ps + [best_fix_ps]
    x_vals = np.arange(len(y_vals))
    dyn_cols = ["#9b59b6"] * 10 + ["#2ecc71"]
    ax2.bar(x_vals, y_vals, color=dyn_cols, alpha=0.85)
    ax2.axhline(best_fix_ps, color="green", lw=1.5, ls="--",
                label=f"Mejor fijo {best_rp:.2%}")
    ax2.set_xticks(x_vals)
    labels = [f"D{i+1}" for i in range(10)] + ["Fix*"]
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Top 10 Dinamico vs Mejor Fijo")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend()
    ax2.set_ylim(0, 1)

    plt.suptitle("Sizing Dinamico vs Fijo — Probabilidad de Exito FTMO", fontsize=13)
    fig.savefig(RESULTS_DIR / "dynamic_vs_fixed.png", dpi=120)
    plt.close()
    log.info("Saved dynamic_vs_fixed.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(42)

    # ── Load data ──────────────────────────────────────────────────────────────
    cfg    = Config()
    df_raw = load_data(cfg)
    df_ind = add_indicators(df_raw, cfg)

    # ── Load / generate OOS trades ─────────────────────────────────────────────
    df_trades = load_or_generate_trades(df_ind)

    # ── Step 1: base statistics ────────────────────────────────────────────────
    print_base_stats(df_trades)

    # ── Prepare arrays (shared across all MC runs) ─────────────────────────────
    pts, sl, date_enc, n_dates = _prep_arrays(df_trades)

    # ── Step 2: fixed sizing ───────────────────────────────────────────────────
    fixed_results = run_fixed(df_trades, rng)
    best_fixed_rp = max(RISK_LEVELS_FIXED,
                         key=lambda r: fixed_results[r]["p_success"])

    # ── Step 3: dynamic sizing ─────────────────────────────────────────────────
    dyn_rows = run_dynamic(df_trades, rng)
    best_dyn = dyn_rows[0]  # sorted by p_success descending

    # ── Step 4: sensitivity analysis ──────────────────────────────────────────
    best_fixed_rp2, best_dyn_ps, best_fix_ps, avg_days = sensitivity_analysis(
        fixed_results, dyn_rows, df_trades, best_dyn
    )

    # ── Step 5: recommendation ─────────────────────────────────────────────────
    print_recommendation(best_dyn, best_fixed_rp, fixed_results, avg_days)

    # ── Save summary CSV ───────────────────────────────────────────────────────
    rows = []
    for rp in RISK_LEVELS_FIXED:
        r = fixed_results[rp]
        rows.append({
            "type": "fixed", "risk_base": rp,
            "p_success": r["p_success"],
            "p_fail_total": r["p_fail_total"],
            "p_fail_daily": r["p_fail_daily"],
            "dd_med_succ": r["dd_med_succ"],
            "n_trades_med": r["n_trades_med"],
        })
    for r in dyn_rows[:20]:
        rows.append({"type": "dynamic", **r})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "monte_carlo_results.csv", index=False)
    log.info("Saved monte_carlo_results.csv")

    # ── Plots ──────────────────────────────────────────────────────────────────
    log.info("Generating plots...")
    try:
        plot_fixed_distributions(fixed_results)
        plot_success_rates(fixed_results)
        plot_equity_curves(pts, sl, date_enc, n_dates, best_fixed_rp,
                           np.random.default_rng(77))
        plot_drawdown_distribution(fixed_results, best_fixed_rp)
        plot_dynamic_vs_fixed(fixed_results, dyn_rows, best_fixed_rp)
        log.info("All plots saved.")
    except Exception as exc:
        log.warning("Plot generation failed: %s", exc)
        log.warning("Numeric results are complete — plots skipped.")

    log.info("Phase 7 complete. Outputs in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
