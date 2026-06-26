"""
Phase 5 — HMM Regime Filter + Strategy Combination.

Experiments:
  1. ORB SHORT solo        (reference, no HMM)
  2. MR LONG solo          (Exp4 params, no HMM)
  3. HMM combined          (HMM assigns strategy per session)
  4. HMM filter-only       (both strategies, skip volatile days)
"""
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
from orb_system.data.loader import load_data, split_train_test
from orb_system.indicators.technical import add_indicators
from orb_system.config import Config as _Cfg
from orb_system.backtester.engine import Results as ORBResults
from orb_system.strategy import mean_reversion as _mr
from orb_system.strategy import orb as _orb
from orb_system.regime.features import compute_daily_features, zscore_normalize
from orb_system.regime.hmm import RegimeHMM, FEATURE_COLS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_NAN = float("nan")
POINT_VALUE = 20.0

# ── MR LONG parameters (Phase 4 Exp4 — best variant) ─────────────────────────
MR_PARAMS = dict(
    sl_atr_mult   = 0.5,
    entry_pct     = 0.50,
    max_bars      = 120,
    tp_far_extreme = True,
)

# ── ORB SHORT parameters (Phase 5 spec) ──────────────────────────────────────
ORB_PARAMS: dict = {}   # use defaults from orb.py


# ══════════════════════════════════════════════════════════════════════════════
# Combined results container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlatTrade:
    entry_ts:    pd.Timestamp
    exit_ts:     pd.Timestamp
    pnl_net:     float
    direction:   str
    exit_reason: str
    strategy:    str    # "orb_short" | "mr_long"
    year:        int


class CombinedResults:
    def __init__(
        self,
        orb_trades:  list,
        mr_trades:   list,
        initial_capital: float = 100_000.0,
        label: str = "",
    ):
        self.initial_capital = initial_capital
        self.label           = label

        combined = []
        for t in orb_trades:
            combined.append(FlatTrade(
                entry_ts    = t.entry_ts,
                exit_ts     = t.exit_ts,
                pnl_net     = t.pnl_net,
                direction   = t.direction,
                exit_reason = t.exit_reason,
                strategy    = "orb_short",
                year        = t.year,
            ))
        for t in mr_trades:
            combined.append(FlatTrade(
                entry_ts    = t.entry_ts,
                exit_ts     = t.exit_ts,
                pnl_net     = t.pnl_net,
                direction   = t.direction,
                exit_reason = t.exit_reason,
                strategy    = "mr_long",
                year        = t.year,
            ))
        self.trades = sorted(combined, key=lambda x: x.entry_ts)

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        if not self.trades:
            return {"n_trades": 0}

        pnl     = [t.pnl_net for t in self.trades]
        winners = [p for p in pnl if p > 0]
        losers  = [p for p in pnl if p <= 0]

        equity = [self.initial_capital]
        for p in pnl:
            equity.append(equity[-1] + p)
        peak, max_dd = equity[0], 0.0
        for e in equity:
            peak   = max(peak, e)
            max_dd = max(max_dd, peak - e)

        arr    = np.array(pnl)
        sharpe = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0
        gw     = sum(winners)
        gl     = abs(sum(losers))
        pf     = gw / gl if gl > 0 else _NAN

        exits: dict = {}
        for t in self.trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
        n = len(self.trades)
        exit_pct = {k: v / n * 100 for k, v in exits.items()}

        yr_data: dict = {}
        for t in self.trades:
            yr_data.setdefault(t.year, []).append(t.pnl_net)
        yearly = []
        for yr in sorted(yr_data):
            yp   = yr_data[yr]
            yw   = [p for p in yp if p > 0]
            yl   = [p for p in yp if p <= 0]
            ygw  = sum(yw); ygl = abs(sum(yl))
            yearly.append({
                "year":    yr, "trades":  len(yp),
                "win_rate": len(yw) / len(yp) if yp else 0,
                "pf":      ygw / ygl if ygl > 0 else _NAN,
                "pnl_total": sum(yp),
            })

        # strategy breakdown
        n_orb = sum(1 for t in self.trades if t.strategy == "orb_short")
        n_mr  = sum(1 for t in self.trades if t.strategy == "mr_long")

        return {
            "n_trades":         n,
            "n_orb":            n_orb,
            "n_mr":             n_mr,
            "win_rate":         len(winners) / n,
            "pf":               pf,
            "sharpe":           sharpe,
            "return_pct":       (equity[-1] - self.initial_capital) / self.initial_capital * 100,
            "initial_capital":  self.initial_capital,
            "final_capital":    equity[-1],
            "max_drawdown_usd": max_dd,
            "max_drawdown_pct": max_dd / self.initial_capital * 100,
            "exit_pct":         exit_pct,
            "yearly":           yearly,
        }

    # ------------------------------------------------------------------
    def print_report(self, label: str = "") -> None:
        lbl = label or self.label
        m   = self.metrics()
        W   = 62

        print("=" * W)
        print(f"  EXPERIMENTO — {lbl}")
        print("=" * W)

        if m["n_trades"] == 0:
            print("  Sin trades.")
            print("=" * W)
            return

        n   = m["n_trades"]
        ep  = m["exit_pct"]
        sl_ = ep.get("sl",      0)
        tp_ = ep.get("tp",      0)
        eod_ = ep.get("eod",    0)
        tim_ = ep.get("timeout", 0)

        print(f"  Trades : {n}  (ORB SHORT: {m['n_orb']} | MR LONG: {m['n_mr']})")
        print(f"  Win%   : {m['win_rate']:.1%}  |  PF: {m['pf']:.3f}  |  Sharpe: {m['sharpe']:.2f}"
              f"  |  Return: {m['return_pct']:+.1f}%")
        print(f"  Exits  -> SL:{sl_:.0f}%  TP:{tp_:.0f}%  EOD:{eod_:.0f}%  Time:{tim_:.0f}%")
        print(f"  MaxDD  : ${m['max_drawdown_usd']:,.0f} ({m['max_drawdown_pct']:.1f}%)")

        print(f"\n  Desglose anual:")
        print(f"  {'Ano':>4} | {'N':>5} | {'Win%':>5} | {'PF':>5} | {'Ret%':>7}")
        print("  " + "-" * 38)
        for r in m["yearly"]:
            pf_s  = f"{r['pf']:.2f}" if not math.isnan(r['pf']) else "  inf"
            ret_s = f"{r['pnl_total'] / self.initial_capital * 100:+.1f}"
            print(f"  {r['year']:>4} | {r['trades']:>5} | {r['win_rate']:>5.1%} | "
                  f"{pf_s:>5} | {ret_s:>7}%")
        print("=" * W)

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "entry_ts":    t.entry_ts,
            "exit_ts":     t.exit_ts,
            "pnl_net":     t.pnl_net,
            "direction":   t.direction,
            "exit_reason": t.exit_reason,
            "strategy":    t.strategy,
            "year":        t.year,
        } for t in self.trades])


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pf_for_year(trades: list, year: int) -> float:
    yp = [t.pnl_net for t in trades if t.year == year]
    if not yp:
        return _NAN
    gw = sum(p for p in yp if p > 0)
    gl = abs(sum(p for p in yp if p <= 0))
    return gw / gl if gl > 0 else _NAN


def _filter_by_dates(trades: list, dates_set: set) -> list:
    return [t for t in trades if t.entry_ts.date() in dates_set]


def _orb_to_flat(orb_results) -> list:
    trades = orb_results.trades if hasattr(orb_results, "trades") else []
    return [FlatTrade(
        entry_ts    = t.entry_ts,
        exit_ts     = t.exit_ts,
        pnl_net     = t.pnl_net,
        direction   = t.direction,
        exit_reason = t.exit_reason,
        strategy    = "orb_short",
        year        = t.year,
    ) for t in trades]


def _mr_to_flat(mr_results) -> list:
    trades = mr_results.trades if hasattr(mr_results, "trades") else []
    return [FlatTrade(
        entry_ts    = t.entry_ts,
        exit_ts     = t.exit_ts,
        pnl_net     = t.pnl_net,
        direction   = t.direction,
        exit_reason = t.exit_reason,
        strategy    = "mr_long",
        year        = t.year,
    ) for t in trades]


def _build_combined(orb_flat, mr_flat, regime_map, regime_filter, initial_capital, label):
    """
    regime_filter: callable(regime_label, strategy) -> bool
      returns True if the trade should be included.
    """
    kept_orb = [t for t in orb_flat if regime_filter(regime_map.get(t.entry_ts.date(), ""), "orb_short")]
    kept_mr  = [t for t in mr_flat  if regime_filter(regime_map.get(t.entry_ts.date(), ""), "mr_long")]

    all_trades = sorted(kept_orb + kept_mr, key=lambda t: t.entry_ts)
    res = CombinedResults.__new__(CombinedResults)
    res.initial_capital = initial_capital
    res.label           = label
    res.trades          = all_trades
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Summary table helper
# ══════════════════════════════════════════════════════════════════════════════

def _summary_row(label: str, combined_tr: CombinedResults, combined_te: CombinedResults) -> dict:
    mtr = combined_tr.metrics()
    mte = combined_te.metrics()

    pf25 = _pf_for_year(combined_te.trades, 2025)
    pf26 = _pf_for_year(combined_te.trades, 2026)
    return {
        "label":   label,
        "pf_tr":   mtr.get("pf",      _NAN),
        "pf_te":   mte.get("pf",      _NAN),
        "sr_te":   mte.get("sharpe",  0.0),
        "n_te":    mte.get("n_trades", 0),
        "pf_2025": pf25,
        "pf_2026": pf26,
    }


def _print_summary(rows: list[dict]) -> None:
    W = 76
    print("\n" + "=" * W)
    print("  RESUMEN FINAL — PHASE 5 COMBINED SYSTEM")
    print("=" * W)
    hdr = (f"  {'Experimento':<28} | {'PF Tr':>6} | {'PF Te':>6} | "
           f"{'SR Te':>6} | {'N Te':>5} | {'2025 PF':>7} | {'2026 PF':>7}")
    print(hdr)
    print("  " + "-" * 74)

    def _fs(v): return f"{v:.3f}" if not math.isnan(v) else "  N/A"

    for r in rows:
        print(
            f"  {r['label']:<28} | {_fs(r['pf_tr']):>6} | {_fs(r['pf_te']):>6} | "
            f"{r['sr_te']:>6.2f} | {r['n_te']:>5d} | {_fs(r['pf_2025']):>7} | {_fs(r['pf_2026']):>7}"
        )
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    cfg      = _Cfg()
    df_raw   = load_data(cfg)
    df_ind   = add_indicators(df_raw, cfg)
    df_tr, df_te = split_train_test(df_ind, cfg.backtest.train_ratio)

    split_ts   = df_tr.index[-1]
    split_date = split_ts.date()

    log.info("Train: %d bars  (%s → %s)", len(df_tr), df_tr.index[0].date(), df_tr.index[-1].date())
    log.info("Test : %d bars  (%s → %s)", len(df_te), df_te.index[0].date(), df_te.index[-1].date())

    # ── 2. Compute features (causal) ─────────────────────────────────────────
    log.info("Computing daily features...")
    feat_raw  = compute_daily_features(df_ind)          # full dataset
    feat_norm = zscore_normalize(feat_raw, window=20)   # rolling z-score, causal

    # ── 3. Train HMM on training features only ────────────────────────────────
    log.info("Training HMM on training features...")
    feat_norm_tr = feat_norm[np.array(feat_norm.index) <= split_date]
    hmm_model = RegimeHMM(n_states=3, random_state=42)
    hmm_model.fit(feat_norm_tr)

    # ── 4. Predict regimes for full dataset (train + test) ────────────────────
    log.info("Predicting regimes...")
    regime_series = hmm_model.predict_regimes(feat_norm)      # pd.Series date -> label

    # ── 5. Print HMM diagnostics (BEFORE any PnL) ────────────────────────────
    hmm_model.print_diagnostics(feat_raw, regime_series, split_date)

    # Save hmm model and regime labels
    hmm_model.save(str(RESULTS_DIR / "hmm_model.pkl"))

    regime_df = pd.DataFrame({
        "date":    regime_series.index,
        "regime":  regime_series.values,
    })
    for col in FEATURE_COLS:
        regime_df[col] = feat_raw.reindex(regime_series.index)[col].values
    regime_df.to_csv(RESULTS_DIR / "hmm_regime_labels.csv", index=False)
    log.info("HMM model and regime labels saved.")

    # build date -> label mapping for filtering trades
    regime_map: dict = {d: lbl for d, lbl in zip(regime_series.index, regime_series.values)}

    # ── 6. Run base engines on full train/test ────────────────────────────────
    log.info("Running ORB SHORT on full data...")
    orb_res_tr  = _orb.run(df_tr, label="ORB SHORT train")
    orb_res_te  = _orb.run(df_te, label="ORB SHORT test")

    log.info("Running MR LONG on full data...")
    mr_res_tr   = _mr.MeanReversionEngine.run(df_tr, collect_diag=False, **MR_PARAMS)
    mr_res_te   = _mr.MeanReversionEngine.run(df_te, collect_diag=False, **MR_PARAMS)

    orb_flat_tr = _orb_to_flat(orb_res_tr)
    orb_flat_te = _orb_to_flat(orb_res_te)
    mr_flat_tr  = _mr_to_flat(mr_res_tr)
    mr_flat_te  = _mr_to_flat(mr_res_te)

    log.info("Base engines done. ORB: %d/%d trades (tr/te)  MR: %d/%d trades",
             len(orb_flat_tr), len(orb_flat_te), len(mr_flat_tr), len(mr_flat_te))

    CAPITAL = 100_000.0
    summary_rows = []

    # ── EXPERIMENTO 1 — ORB SHORT solo ───────────────────────────────────────
    def _exp1_filter(regime, strategy): return strategy == "orb_short"

    e1_tr = _build_combined(orb_flat_tr, [], regime_map, _exp1_filter, CAPITAL, "ORB SHORT solo")
    e1_te = _build_combined(orb_flat_te, [], regime_map, _exp1_filter, CAPITAL, "ORB SHORT solo")

    print("\n" + "=" * 62)
    print("  EXPERIMENTO 1 — ORB SHORT SOLO  (sin HMM, referencia)")
    print("=" * 62)
    print("\n  TRAINING:")
    e1_tr.print_report()
    print("\n  TEST:")
    e1_te.print_report()
    e1_tr.to_dataframe().to_csv(RESULTS_DIR / "combined_exp1_orb_short_train.csv", index=False)
    e1_te.to_dataframe().to_csv(RESULTS_DIR / "combined_exp1_orb_short_test.csv",  index=False)
    print(f"  CSV -> results/combined_exp1_orb_short_[train|test].csv")
    summary_rows.append(_summary_row("1. ORB SHORT solo", e1_tr, e1_te))

    # ── EXPERIMENTO 2 — MR LONG solo ─────────────────────────────────────────
    def _exp2_filter(regime, strategy): return strategy == "mr_long"

    e2_tr = _build_combined([], mr_flat_tr, regime_map, _exp2_filter, CAPITAL, "MR LONG solo")
    e2_te = _build_combined([], mr_flat_te, regime_map, _exp2_filter, CAPITAL, "MR LONG solo")

    print("\n" + "=" * 62)
    print("  EXPERIMENTO 2 — MR LONG SOLO  (sin HMM, referencia)")
    print("=" * 62)
    print("\n  TRAINING:")
    e2_tr.print_report()
    print("\n  TEST:")
    e2_te.print_report()
    e2_tr.to_dataframe().to_csv(RESULTS_DIR / "combined_exp2_mr_long_train.csv", index=False)
    e2_te.to_dataframe().to_csv(RESULTS_DIR / "combined_exp2_mr_long_test.csv",  index=False)
    print(f"  CSV -> results/combined_exp2_mr_long_[train|test].csv")
    summary_rows.append(_summary_row("2. MR LONG solo", e2_tr, e2_te))

    # ── EXPERIMENTO 3 — HMM combinado ────────────────────────────────────────
    def _exp3_filter(regime, strategy):
        if regime == "ranging"  and strategy == "mr_long":    return True
        if regime == "trending" and strategy == "orb_short":  return True
        return False

    e3_tr = _build_combined(orb_flat_tr, mr_flat_tr, regime_map, _exp3_filter, CAPITAL, "HMM combinado")
    e3_te = _build_combined(orb_flat_te, mr_flat_te, regime_map, _exp3_filter, CAPITAL, "HMM combinado")

    print("\n" + "=" * 62)
    print("  EXPERIMENTO 3 — SISTEMA COMBINADO CON HMM")
    print("  ranging=MR LONG | trending=ORB SHORT | volatile=no operar")
    print("=" * 62)
    print("\n  TRAINING:")
    e3_tr.print_report()
    print("\n  TEST:")
    e3_te.print_report()
    e3_tr.to_dataframe().to_csv(RESULTS_DIR / "combined_exp3_hmm_combined_train.csv", index=False)
    e3_te.to_dataframe().to_csv(RESULTS_DIR / "combined_exp3_hmm_combined_test.csv",  index=False)
    print(f"  CSV -> results/combined_exp3_hmm_combined_[train|test].csv")
    summary_rows.append(_summary_row("3. HMM combinado", e3_tr, e3_te))

    # ── EXPERIMENTO 4 — HMM como filtro no-operar únicamente ─────────────────
    def _exp4_filter(regime, strategy):
        return regime != "volatile"   # skip volatile only; both strategies run in ranging+trending

    e4_tr = _build_combined(orb_flat_tr, mr_flat_tr, regime_map, _exp4_filter, CAPITAL, "HMM filtro no-operar")
    e4_te = _build_combined(orb_flat_te, mr_flat_te, regime_map, _exp4_filter, CAPITAL, "HMM filtro no-operar")

    print("\n" + "=" * 62)
    print("  EXPERIMENTO 4 — HMM FILTRO NO-OPERAR (volatile=skip, ambas en resto)")
    print("=" * 62)
    print("\n  TRAINING:")
    e4_tr.print_report()
    print("\n  TEST:")
    e4_te.print_report()
    e4_tr.to_dataframe().to_csv(RESULTS_DIR / "combined_exp4_hmm_filter_train.csv", index=False)
    e4_te.to_dataframe().to_csv(RESULTS_DIR / "combined_exp4_hmm_filter_test.csv",  index=False)
    print(f"  CSV -> results/combined_exp4_hmm_filter_[train|test].csv")
    summary_rows.append(_summary_row("4. HMM filtro no-operar", e4_tr, e4_te))

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_summary(summary_rows)
    log.info("Phase 5 complete. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
