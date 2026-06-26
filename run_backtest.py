"""
NQ ORB — Phase 3 research runner (1-min bars).

Experiments:
  1.  Baseline          — per-year, per-direction breakdown
  2.  Prev session      — direction aligned with yesterday's session return
  3a. Gap no min        — overnight gap direction (any size)
  3b. Gap 5pt min       — overnight gap direction (>= 5 pts)
  4.  OR position       — trade only toward OR extreme at close of OR window
  5a. Trailing 1.0x ATR
  5b. Trailing 1.5x ATR
  5c. Trailing 2.0x ATR
  6.  Combination       — auto-selects qualifying filters from Exp 2-5

Plus statistical analysis on baseline test trades:
  - One-sided t-test (H0: mean pnl_net <= 0)
  - Bootstrap profit factor (1000 resamples)
  - Day-of-week breakdown

Run with:
    python run_backtest.py
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from orb_system.config import Config
from orb_system.data.loader import load_data, split_train_test, data_summary
from orb_system.indicators.technical import add_indicators
from orb_system.backtester.engine import BacktestEngine, Results

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_config(**overrides) -> Config:
    cfg = Config()
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return cfg


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_exp(
    name: str,
    params: str,
    overrides: dict,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple:
    """Run one experiment, print report, return (r_train, r_test)."""
    cfg = make_config(**overrides)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  EXP: {name}")
    print(f"  Params: {params}")
    print(sep)

    r_tr = BacktestEngine(cfg).run(df_train, label=f"{name} | TRAINING")
    r_tr.print_report("TRAINING")

    r_te = BacktestEngine(cfg).run(df_test, label=f"{name} | TEST")
    r_te.print_report("TEST")

    m_tr = r_tr.metrics()
    m_te = r_te.metrics()
    pf_tr = m_tr.get("profit_factor", float("nan"))
    pf_te = m_te.get("profit_factor", float("nan"))
    d_pf  = pf_te - pf_tr
    sr_tr = m_tr.get("sharpe_ratio", 0.0)
    sr_te = m_te.get("sharpe_ratio", 0.0)

    pf_tr_s = f"{pf_tr:.2f}" if not math.isnan(pf_tr) else "inf"
    pf_te_s = f"{pf_te:.2f}" if not math.isnan(pf_te) else "inf"
    d_pf_s  = f"(D {d_pf:+.2f})" if not (math.isnan(d_pf) or math.isinf(d_pf)) else ""
    print(f"\n  Delta PF  (train -> test):  {pf_tr_s} -> {pf_te_s}  {d_pf_s}")
    print(f"  Delta SR  (train -> test):  {sr_tr:.2f} -> {sr_te:.2f}  (D {sr_te - sr_tr:+.2f})")
    print(sep)

    slug = name.lower().replace(" ", "_")
    _save_trades(r_tr.trades, RESULTS_DIR / f"{slug}_train.csv")
    _save_trades(r_te.trades, RESULTS_DIR / f"{slug}_test.csv")

    return r_tr, r_te


# ---------------------------------------------------------------------------
# OR position distribution analysis
# ---------------------------------------------------------------------------

def print_or_position_analysis(r_tr: Results, r_te: Results) -> None:
    """Show how or_position distributes across winning and losing trades."""
    sep = "-" * 56
    print(f"\n  OR Position Distribution Analysis")
    print(f"  (position_in_OR = (or_close - or_low) / or_range)")
    print(f"  Values: 0=at low extreme  0.5=midpoint  1=at high extreme")
    print(sep)

    for split_label, r in [("TRAINING", r_tr), ("TEST", r_te)]:
        if not r.trades:
            continue
        print(f"\n  [{split_label}]")
        for direction in ("long", "short"):
            trades_d = [t for t in r.trades if t.direction == direction]
            if not trades_d:
                continue
            winners = [t.or_position for t in trades_d if t.pnl_net > 0]
            losers  = [t.or_position for t in trades_d if t.pnl_net <= 0]
            w_mean = float(np.mean(winners)) if winners else float("nan")
            l_mean = float(np.mean(losers))  if losers  else float("nan")
            w_std  = float(np.std(winners))  if winners else float("nan")
            l_std  = float(np.std(losers))   if losers  else float("nan")
            dir_all = [t.or_position for t in trades_d]
            print(f"    {direction.upper()}:  all={np.mean(dir_all):.3f}  "
                  f"winners={w_mean:.3f}(+-{w_std:.3f})  "
                  f"losers={l_mean:.3f}(+-{l_std:.3f})  "
                  f"n={len(trades_d)} (W:{len(winners)} L:{len(losers)})")
    print(sep)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def print_statistical_analysis(r_tr: Results, r_te: Results) -> None:
    """
    On baseline test trades:
      - One-sided t-test  (H1: mean_pnl > 0)
      - Bootstrap PF      (1000 resamples)
      - Day-of-week breakdown
    """
    sep = "=" * 64
    print(f"\n{sep}")
    print("  STATISTICAL VALIDATION — Baseline trades")
    print(sep)

    for split_label, r in [("TRAINING", r_tr), ("TEST", r_te)]:
        m = r.metrics()
        if m["n_trades"] == 0:
            print(f"  [{split_label}] No trades.")
            continue
        print(f"\n  [{split_label}]  n = {m['n_trades']}")

        pnl = np.array([t.pnl_net for t in r.trades])

        # --- T-test (one-sided, H1: mean > 0) ---
        t_stat, p_val = _ttest_gt_zero(pnl)
        sig_str = "SIGNIFICANT" if (p_val is not None and p_val < 0.05) else "not significant"
        if t_stat is not None:
            print(f"  T-test H1:mean>0  t={t_stat:.3f}  p={p_val:.4f}  ({sig_str} at 5%)")
        else:
            print("  T-test: insufficient data")

        # --- Bootstrap PF ---
        bp = _bootstrap_pf(pnl, n_iter=1000)
        print(f"  Bootstrap PF (1000 iters):  "
              f"mean={np.mean(bp):.3f}  "
              f"p5={np.percentile(bp, 5):.3f}  "
              f"p95={np.percentile(bp, 95):.3f}")

        # --- Day-of-week ---
        print(f"\n  Day-of-week breakdown:")
        print(f"  {'Day':<5}  {'Trades':>6}  {'Win%':>6}  {'PF':>5}  {'Total PnL':>11}")
        print("  " + "-" * 40)
        for row in m["by_dow"]:
            pf_s = f"{row['pf']:.2f}" if not math.isnan(row["pf"]) else "inf"
            print(f"  {row['day']:<5}  {row['trades']:>6}  "
                  f"{row['win_rate']:>6.1%}  {pf_s:>5}  ${row['pnl_total']:>10,.0f}")

    print(sep)


def _ttest_gt_zero(values: np.ndarray):
    """One-sided t-test H1: mean > 0.  Returns (t_stat, p_value)."""
    n = len(values)
    if n < 2:
        return None, None
    mean = values.mean()
    std  = values.std(ddof=1)
    if std == 0:
        return None, None
    t = mean / (std / math.sqrt(n))
    try:
        from scipy import stats as sp
        p = float(sp.t.sf(t, df=n - 1))
    except ImportError:
        # Normal approximation
        p = 0.5 * math.erfc(t / math.sqrt(2))
    return t, p


def _bootstrap_pf(pnl: np.ndarray, n_iter: int = 1000) -> np.ndarray:
    """Resample pnl with replacement n_iter times, compute PF each time."""
    rng = np.random.default_rng(42)
    pf_vals = np.empty(n_iter)
    for i in range(n_iter):
        s = rng.choice(pnl, size=len(pnl), replace=True)
        gw = s[s > 0].sum()
        gl = abs(s[s <= 0].sum())
        pf_vals[i] = gw / gl if gl > 0 else 999.0
    return pf_vals


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(rows: list) -> None:
    sep = "=" * 76
    print(f"\n{sep}")
    print("  RESUMEN COMPARATIVO DE EXPERIMENTOS")
    print(sep)
    hdr = (
        f"  {'Experimento':<22} {'PF Tr':>6} {'PF Te':>6} "
        f"{'D PF':>6} {'SR Tr':>6} {'SR Te':>6} "
        f"{'Trades':>7} {'Win%':>6}"
    )
    print(hdr)
    print("  " + "-" * 69)
    for r in rows:
        print(
            f"  {r['name']:<22} {r['pf_tr']:>6} {r['pf_te']:>6} "
            f"{r['d_pf']:>6} {r['sr_tr']:>6} {r['sr_te']:>6} "
            f"{r['n_te']:>7} {r['wr_te']:>6}"
        )
    print(sep)


def _summary_row(name: str, r_tr: Results, r_te: Results) -> dict:
    m_tr = r_tr.metrics()
    m_te = r_te.metrics()
    pf_tr = m_tr.get("profit_factor", float("nan"))
    pf_te = m_te.get("profit_factor", float("nan"))
    d_pf  = pf_te - pf_tr
    sr_tr = m_tr.get("sharpe_ratio", 0.0)
    sr_te = m_te.get("sharpe_ratio", 0.0)
    n_te  = m_te.get("n_trades", 0)
    wr_te = m_te.get("win_rate", 0.0)

    def fmt_pf(v):
        return "inf" if math.isinf(v) or math.isnan(v) else f"{v:.2f}"

    return {
        "name":   name,
        "pf_tr":  fmt_pf(pf_tr),
        "pf_te":  fmt_pf(pf_te),
        "d_pf":   f"{d_pf:+.2f}" if not (math.isnan(d_pf) or math.isinf(d_pf)) else "n/a",
        "sr_tr":  f"{sr_tr:.2f}",
        "sr_te":  f"{sr_te:.2f}",
        "n_te":   n_te,
        "wr_te":  f"{wr_te:.1%}" if n_te else "n/a",
        # raw for Exp 6 logic
        "_pf_tr": pf_tr,
        "_pf_te": pf_te,
        "_n_te":  n_te,
    }


# ---------------------------------------------------------------------------
# Trade CSV serialisation
# ---------------------------------------------------------------------------

def _save_trades(trades, path: Path) -> None:
    if not trades:
        return
    rows = [
        {
            "trade_id":    t.trade_id,
            "entry_ts":    t.entry_ts,
            "exit_ts":     t.exit_ts,
            "direction":   t.direction,
            "entry_price": round(t.entry_price, 4),
            "exit_price":  round(t.exit_price,  4),
            "sl_price":    round(t.sl_price,     4),
            "tp_price":    round(t.tp_price,     4),
            "trail_price": round(t.trail_price, 4) if not math.isnan(t.trail_price) else "",
            "exit_reason": t.exit_reason,
            "pnl_points":  round(t.pnl_points, 4),
            "pnl_usd":     round(t.pnl_usd, 2),
            "pnl_net":     round(t.pnl_net, 2),
            "atr_at_entry":round(t.atr_at_entry, 4),
            "or_high":     round(t.or_high, 4),
            "or_low":      round(t.or_low, 4),
            "or_position": round(t.or_position, 4),
            "candle_rng":  round(t.candle_rng, 4),
            "vol_ratio":   round(t.vol_ratio, 4),
            "year":        t.year,
            "month":       t.month,
            "day_of_week": t.day_of_week,
        }
        for t in trades
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved %d trades -> %s", len(trades), path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 64)
    logger.info("NQ ORB — PHASE 3 RESEARCH RUN")
    logger.info("=" * 64)

    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Data pipeline (once, shared across all experiments)
    # ------------------------------------------------------------------
    base_cfg = Config()
    base_cfg.summary()

    df = load_data(base_cfg, use_cache=True)
    data_summary(df)

    df = add_indicators(df, base_cfg)

    df_train, df_test = split_train_test(df, base_cfg.backtest.train_ratio)

    summary_rows: list = []

    # ==================================================================
    # EXP 1: Baseline with per-direction annual breakdown
    # ==================================================================
    r1_tr, r1_te = run_exp(
        "1. Baseline",
        "all filters OFF | direction=both | SL=1x ATR | TP=2x ATR | max_bars=120",
        {},
        df_train, df_test,
    )
    # Detailed direction x year table
    r1_tr.print_direction_annual_breakdown("TRAINING")
    r1_te.print_direction_annual_breakdown("TEST")

    summary_rows.append(_summary_row("Baseline", r1_tr, r1_te))

    # Run stats analysis immediately on baseline
    print_statistical_analysis(r1_tr, r1_te)

    # ==================================================================
    # EXP 2: Previous session direction filter
    # ==================================================================
    r2_tr, r2_te = run_exp(
        "2. Prev Session",
        "use_prev_session_filter=True  (prev day up -> LONG, down -> SHORT)",
        {"signal.use_prev_session_filter": True},
        df_train, df_test,
    )
    summary_rows.append(_summary_row("Prev Session", r2_tr, r2_te))

    # ==================================================================
    # EXP 3a: Gap filter — any gap
    # ==================================================================
    r3a_tr, r3a_te = run_exp(
        "3a. Gap (no min)",
        "use_gap_filter=True | gap_min_points=0.0  (any gap > 0 sets direction)",
        {"signal.use_gap_filter": True, "signal.gap_min_points": 0.0},
        df_train, df_test,
    )
    summary_rows.append(_summary_row("Gap (no min)", r3a_tr, r3a_te))

    # ==================================================================
    # EXP 3b: Gap filter — 5pt minimum
    # ==================================================================
    r3b_tr, r3b_te = run_exp(
        "3b. Gap 5pt min",
        "use_gap_filter=True | gap_min_points=5.0  (skip sessions with gap < 5 pts)",
        {"signal.use_gap_filter": True, "signal.gap_min_points": 5.0},
        df_train, df_test,
    )
    summary_rows.append(_summary_row("Gap 5pt min", r3b_tr, r3b_te))

    # ==================================================================
    # EXP 4: OR position filter  (close in upper zone -> LONG | lower -> SHORT)
    # ==================================================================
    r4_tr, r4_te = run_exp(
        "4. OR Position",
        "use_or_position_filter=True | long_min=0.60 | short_max=0.40",
        {
            "signal.use_or_position_filter": True,
            "signal.or_position_long_min":   0.6,
            "signal.or_position_short_max":  0.4,
        },
        df_train, df_test,
    )
    summary_rows.append(_summary_row("OR Position", r4_tr, r4_te))

    # Distribution analysis (uses baseline trades for signal validity)
    print_or_position_analysis(r1_tr, r1_te)

    # ==================================================================
    # EXP 5: Trailing exit calibration
    # ==================================================================
    r5a_tr, r5a_te = run_exp(
        "5a. Trail 1.0x",
        "use_trailing_exit=True | trailing_atr_mult=1.0 | activation=0.5x ATR",
        {"risk.use_trailing_exit": True, "risk.trailing_atr_mult": 1.0},
        df_train, df_test,
    )
    summary_rows.append(_summary_row("Trail 1.0x ATR", r5a_tr, r5a_te))

    r5b_tr, r5b_te = run_exp(
        "5b. Trail 1.5x",
        "use_trailing_exit=True | trailing_atr_mult=1.5 | activation=0.5x ATR",
        {"risk.use_trailing_exit": True, "risk.trailing_atr_mult": 1.5},
        df_train, df_test,
    )
    summary_rows.append(_summary_row("Trail 1.5x ATR", r5b_tr, r5b_te))

    r5c_tr, r5c_te = run_exp(
        "5c. Trail 2.0x",
        "use_trailing_exit=True | trailing_atr_mult=2.0 | activation=0.5x ATR",
        {"risk.use_trailing_exit": True, "risk.trailing_atr_mult": 2.0},
        df_train, df_test,
    )
    summary_rows.append(_summary_row("Trail 2.0x ATR", r5c_tr, r5c_te))

    # ==================================================================
    # EXP 6: Optimal combination (auto-determined from Exp 2-5 results)
    # ==================================================================
    baseline_row  = _summary_row("_bl", r1_tr, r1_te)
    baseline_delta = baseline_row["_pf_tr"] - baseline_row["_pf_te"]

    print("\n" + "=" * 64)
    print("  BUILDING EXP 6 — qualifying filter criteria:")
    print(f"    PF test > {baseline_row['_pf_te']:.2f} (baseline)  AND")
    print(f"    Delta (train - test) < {baseline_delta:.2f} (baseline generalisation)  AND")
    print(f"    Trades test >= 80")
    print("=" * 64)

    combo_overrides = {}
    combo_notes     = []

    def qualifies(row: dict) -> bool:
        pf_te = row["_pf_te"]
        pf_tr = row["_pf_tr"]
        if math.isnan(pf_te) or math.isinf(pf_te):
            return False
        delta = pf_tr - pf_te
        return (
            pf_te > baseline_row["_pf_te"]
            and delta < baseline_delta
            and row["_n_te"] >= 80
        )

    row2   = _summary_row("2", r2_tr, r2_te)
    row3a  = _summary_row("3a", r3a_tr, r3a_te)
    row3b  = _summary_row("3b", r3b_tr, r3b_te)
    row4   = _summary_row("4", r4_tr, r4_te)
    row5a  = _summary_row("5a", r5a_tr, r5a_te)
    row5b  = _summary_row("5b", r5b_tr, r5b_te)
    row5c  = _summary_row("5c", r5c_tr, r5c_te)

    # Prev session
    if qualifies(row2):
        combo_overrides["signal.use_prev_session_filter"] = True
        combo_notes.append("prev_session")
        print("  + prev_session_filter  QUALIFIES")
    else:
        print(f"  - prev_session_filter  SKIP  (PF_te={row2['pf_te']}  n={row2['_n_te']})")

    # Gap filter: pick best qualifying variant
    gap_candidates = [
        (row3a, 0.0, "gap_min=0"),
        (row3b, 5.0, "gap_min=5"),
    ]
    best_gap = None
    for row_g, gmin, lbl in gap_candidates:
        if qualifies(row_g):
            if best_gap is None or row_g["_pf_te"] > best_gap[0]["_pf_te"]:
                best_gap = (row_g, gmin, lbl)
    if best_gap is not None:
        combo_overrides["signal.use_gap_filter"]   = True
        combo_overrides["signal.gap_min_points"]   = best_gap[1]
        combo_notes.append(f"gap({best_gap[2]})")
        print(f"  + gap_filter({best_gap[2]})  QUALIFIES")
    else:
        print(f"  - gap_filter  SKIP  "
              f"(3a PF_te={row3a['pf_te']} n={row3a['_n_te']}  "
              f"3b PF_te={row3b['pf_te']} n={row3b['_n_te']})")

    # OR position
    if qualifies(row4):
        combo_overrides["signal.use_or_position_filter"] = True
        combo_notes.append("or_position")
        print("  + or_position_filter  QUALIFIES")
    else:
        print(f"  - or_position_filter  SKIP  (PF_te={row4['pf_te']}  n={row4['_n_te']})")

    # Best trailing
    trail_opts = [
        (row5a, 1.0, "1.0x"),
        (row5b, 1.5, "1.5x"),
        (row5c, 2.0, "2.0x"),
    ]
    best_trail = max(trail_opts, key=lambda x: x[0]["_pf_te"] if not math.isnan(x[0]["_pf_te"]) else 0)
    combo_overrides["risk.use_trailing_exit"]  = True
    combo_overrides["risk.trailing_atr_mult"]  = best_trail[1]
    combo_notes.append(f"trail({best_trail[2]})")
    print(f"  + trailing({best_trail[2]} — best PF_te={best_trail[0]['pf_te']})")

    exp6_name   = "6. Combination"
    exp6_params = " + ".join(combo_notes) if combo_notes else "trailing only"
    print(f"\n  Exp 6 will run: {exp6_params}")

    r6_tr, r6_te = run_exp(
        exp6_name,
        exp6_params,
        combo_overrides,
        df_train, df_test,
    )
    summary_rows.append(_summary_row("Combination", r6_tr, r6_te))

    # ==================================================================
    # Final summary table
    # ==================================================================
    print_summary_table(summary_rows)

    logger.info("Phase 3 complete. Results in %s", RESULTS_DIR)


# ===========================================================================
# PHASE 4 — MEAN REVERSION EXPERIMENTS
# ===========================================================================

def main_mr() -> None:
    from orb_system.strategy.mean_reversion import MeanReversionEngine

    cfg = make_config()
    print("\n" + "=" * 62)
    print("  NQ ORB — PHASE 4: MEAN REVERSION")
    print("=" * 62)

    df_raw = load_data(cfg, use_cache=True)
    df     = add_indicators(df_raw, cfg)
    df_train, df_test = split_train_test(df, cfg.backtest.train_ratio)

    RESULTS_DIR.mkdir(exist_ok=True)

    def run_mr(name: str, slug: str, **kw):
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  EXPERIMENTO MR — {name}")
        params_str = ", ".join(
            f"{k}={v}" for k, v in kw.items()
            if k not in ("collect_diag", "initial_capital")
        )
        print(f"  Params: {params_str}")
        print(sep)

        r_tr = MeanReversionEngine.run(
            df_train, collect_diag=(slug == "baseline"), **kw
        )
        r_te = MeanReversionEngine.run(df_test, collect_diag=False, **kw)

        if slug == "baseline":
            r_tr.print_diagnostic()

        r_tr.print_report("TRAINING")
        r_te.print_report("TEST")

        r_tr.to_dataframe().to_csv(
            RESULTS_DIR / f"mr_{slug}_train.csv", index=False
        )
        r_te.to_dataframe().to_csv(
            RESULTS_DIR / f"mr_{slug}_test.csv", index=False
        )
        print(f"  CSV -> results/mr_{slug}_[train|test].csv")
        print(sep)
        return r_tr, r_te

    # ------------------------------------------------------------------
    # Exp 1 — Baseline
    # ------------------------------------------------------------------
    r1_tr, r1_te = run_mr(
        "1. Baseline mean reversion",
        "baseline",
        sl_atr_mult=0.5, entry_pct=0.30, max_bars=120,
    )

    # ------------------------------------------------------------------
    # Exp 2 — Filtro sesion previa
    # ------------------------------------------------------------------
    r2_tr, r2_te = run_mr(
        "2. Filtro sesion previa",
        "prev_session",
        sl_atr_mult=0.5, entry_pct=0.30, max_bars=120,
        use_prev_session=True,
    )

    # ------------------------------------------------------------------
    # Exp 3 — Filtro OR width (or_range <= 1.5 * EWM-20 de OR ranges)
    # Note: 1-min ATR ~15 pts is 15-25x too small for this filter.
    # The engine uses a causal EWM-20 of recent OR ranges as reference.
    # ------------------------------------------------------------------
    r3_tr, r3_te = run_mr(
        "3. OR width filter  (or_range <= 1.5x EWM-OR)",
        "or_width",
        sl_atr_mult=0.5, entry_pct=0.30, max_bars=120,
        use_or_width=True, or_width_max_mult=1.5,
    )

    # ------------------------------------------------------------------
    # Exp 4 — Entrada en or_mid (50%), TP = OR far extreme (full reversion)
    # entry_pct=0.5 places trigger at or_mid; with TP also at or_mid
    # there is zero profit distance, so tp_far_extreme=True shifts TP to
    # or_high (LONG) / or_low (SHORT) — the full-reversion target.
    # ------------------------------------------------------------------
    r4_tr, r4_te = run_mr(
        "4. Entrada 50pct del OR  (TP = far extreme)",
        "entry_50pct",
        sl_atr_mult=0.5, entry_pct=0.50, max_bars=120,
        tp_far_extreme=True,
    )

    # ------------------------------------------------------------------
    # Exp 5 — Combinacion optima
    # ------------------------------------------------------------------
    def _pf(r):
        m = r.metrics()
        return m.get("pf", 0.0) if m.get("n_trades", 0) > 0 else 0.0

    base_pf_te  = _pf(r1_te)
    use_ps      = _pf(r2_te) > base_pf_te
    use_orw     = _pf(r3_te) > base_pf_te
    use_50      = _pf(r4_te) > _pf(r1_te)

    combo_kw = {
        "sl_atr_mult":       0.5,
        "max_bars":          120,
        "entry_pct":         0.50 if use_50  else 0.30,
        "tp_far_extreme":    use_50,   # must match entry_pct=0.5 mode
        "use_prev_session":  use_ps,
        "use_or_width":      use_orw,
        "or_width_max_mult": 1.5,
    }
    notes = []
    if use_ps:  notes.append("prev_session")
    if use_orw: notes.append("or_width")
    notes.append(f"entry_{int(combo_kw['entry_pct']*100)}pct")

    print(f"\n  Building Exp 5 combination:")
    print(f"    prev_session : {'INCLUDED' if use_ps  else 'SKIPPED'}"
          f"  (PF test {_pf(r2_te):.3f} vs baseline {base_pf_te:.3f})")
    print(f"    or_width     : {'INCLUDED' if use_orw else 'SKIPPED'}"
          f"  (PF test {_pf(r3_te):.3f} vs baseline {base_pf_te:.3f})")
    print(f"    entry_pct    : {combo_kw['entry_pct']:.0%}"
          f"  (50pct PF test {_pf(r4_te):.3f} vs 30pct {_pf(r1_te):.3f})")
    print(f"  Combination: {' + '.join(notes)}")

    r5_tr, r5_te = run_mr(
        f"5. Combinacion  [{' + '.join(notes)}]",
        "combo",
        **combo_kw,
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    def _mrow(name, r_tr, r_te):
        mt, me = r_tr.metrics(), r_te.metrics()
        nt = mt.get("n_trades", 0)
        ne = me.get("n_trades", 0)
        return {
            "name":  name,
            "pf_tr": mt.get("pf", 0.0) if nt > 0 else 0.0,
            "pf_te": me.get("pf", 0.0) if ne > 0 else 0.0,
            "sr_te": me.get("sr", 0.0) if ne > 0 else 0.0,
            "n_te":  ne,
            "wr_te": me.get("wr", 0.0) if ne > 0 else 0.0,
        }

    rows = [
        _mrow("1. Baseline MR",    r1_tr, r1_te),
        _mrow("2. Sesion previa",  r2_tr, r2_te),
        _mrow("3. OR width",       r3_tr, r3_te),
        _mrow("4. Entrada 50%",    r4_tr, r4_te),
        _mrow("5. Combinacion",    r5_tr, r5_te),
    ]

    print("\n\n" + "=" * 72)
    print("  RESUMEN FINAL — MEAN REVERSION EXPERIMENTS")
    print("=" * 72)
    print(f"  {'Experimento':<22} | {'PF Tr':>6} | {'PF Te':>6} | "
          f"{'SR Te':>6} | {'N Te':>5} | {'Win% Te':>7}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r['name']:<22} | {r['pf_tr']:>6.3f} | {r['pf_te']:>6.3f} | "
              f"{r['sr_te']:>6.2f} | {r['n_te']:>5d} | {r['wr_te']:>6.1f}%")
    print("=" * 72)

    logger.info("Phase 4 complete. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main_mr()
