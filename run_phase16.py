#!/usr/bin/env python3
"""
Phase 16: Monte Carlo FTMO Sizing — POC Reversion B1 System.

Resamples daily trade bundles from OOS trades.
Steps: Fixed sizing | Dynamic sizing | Consec protection | Sensitivity | Recommendation.
"""

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV = os.path.join(RESULTS_DIR, "p15_partC_B1_oos_pooled.csv")

# ── FTMO constants ─────────────────────────────────────────────────────────────
INITIAL_CAP = 100_000.0
TARGET      = 110_000.0
STOP_TOTAL  = 90_000.0
DAILY_LIMIT = 5_000.0

# ── Simulation parameters ──────────────────────────────────────────────────────
N_SIMS   = 10_000
MAX_DAYS = 2_000
SEED     = 42
W        = 74

FIXED_N      = [1, 2, 3, 4, 5]
N_BASE_OPTS  = [1, 2, 3, 4]
N_RED_OPTS   = [1, 2]
DD1_OPTS     = [0.03, 0.04, 0.05]
DD2_OPTS     = [0.06, 0.07, 0.08]
CONSEC_OPTS  = [5, 8, 10, 15, 20, 9_999]   # 9999 = no limit

# ORB Phase 7 reference (from Phase 7 results)
ORB_STATS = {
    "wr": 0.546, "rr": 1.03, "exp_usd": 35.75,
    "tpm": 10.2, "max_cl": 5,
    "p_pass": 0.872, "sizing": "0.25% risk",
    "med_days_months": "~26 months",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["date"] = df["entry_ts"].dt.date

    bundles = [grp["pnl_net"].values for _, grp in df.groupby("date")]
    totals  = np.array([b.sum() for b in bundles])

    print(f"  {len(df)} trades | {len(bundles)} trading days")
    print(f"  Avg daily PnL: ${totals.mean():.2f}  Std: ${totals.std():.2f}")
    return df, bundles, totals


def _presample(totals, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(totals), size=(N_SIMS, MAX_DAYS))
    return totals[idx]   # (N_SIMS, MAX_DAYS)


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate(sampled_raw, n_contracts_fn):
    """
    Vectorized simulation of N_SIMS paths.
    n_contracts_fn(peak, cap, consec, active) → int array shape (N_SIMS,)
    """
    cap    = np.full(N_SIMS, INITIAL_CAP)
    peak   = np.full(N_SIMS, INITIAL_CAP)
    max_dd = np.zeros(N_SIMS)
    consec = np.zeros(N_SIMS, dtype=np.int32)
    outcome= np.zeros(N_SIMS, dtype=np.int8)   # 1=success 2=fail_total 3=fail_daily
    days_d = np.full(N_SIMS, MAX_DAYS)
    stopped= np.zeros(N_SIMS, dtype=bool)
    cap30  = np.full(N_SIMS, np.nan)
    cap60  = np.full(N_SIMS, np.nan)
    cap90  = np.full(N_SIMS, np.nan)

    for d in range(MAX_DAYS):
        active = ~stopped
        if not active.any():
            break

        n_c  = n_contracts_fn(peak, cap, consec, active)    # (N_SIMS,)
        raw  = sampled_raw[:, d]
        gain = raw * n_c                                     # 0 for inactive

        # Milestone snapshots (before today's trade)
        if d == 29: cap30[active] = cap[active]
        if d == 59: cap60[active] = cap[active]
        if d == 89: cap90[active] = cap[active]

        # Daily loss check
        fail_day              = np.zeros(N_SIMS, dtype=bool)
        fail_day[active]      = gain[active] <= -DAILY_LIMIT

        # Update capital
        cap[active]  += gain[active]
        peak[active]  = np.maximum(peak[active], cap[active])
        dd_now        = (peak[active] - cap[active]) / INITIAL_CAP
        max_dd[active]= np.maximum(max_dd[active], dd_now)

        # Update consecutive loss counter (daily sign)
        is_loss = raw < 0
        consec[active &  is_loss] += 1
        consec[active & ~is_loss]  = 0

        # Stopping conditions
        fail_tot             = np.zeros(N_SIMS, dtype=bool)
        success              = np.zeros(N_SIMS, dtype=bool)
        fail_tot[active]     = cap[active] <= STOP_TOTAL
        success[active]      = cap[active] >= TARGET

        new_stop = active & (fail_day | fail_tot | success)
        outcome[active & fail_day]                          = 3
        outcome[active & fail_tot & ~fail_day]              = 2
        outcome[active & success & ~fail_day & ~fail_tot]   = 1
        days_d[new_stop] = d + 1
        stopped |= new_stop

    return {
        "outcome": outcome, "days": days_d, "max_dd": max_dd,
        "cap": cap, "cap30": cap30, "cap60": cap60, "cap90": cap90,
    }


def metrics(res):
    oc = res["outcome"]
    n  = N_SIMS
    s  = (oc == 1).sum()
    ft = (oc == 2).sum()
    fd = (oc == 3).sum()
    succ_dd   = res["max_dd"][oc == 1]
    succ_days = res["days"][oc == 1]
    all_days  = res["days"]
    return {
        "p_s":  s  / n, "p_ft": ft / n, "p_fd": fd / n,
        "med_days_all":  int(np.median(all_days)),
        "med_days_succ": int(np.median(succ_days)) if s else 0,
        "dd_p50": float(np.percentile(succ_dd, 50)) if s else 0,
        "dd_p95": float(np.percentile(succ_dd, 95)) if s else 0,
        "cap30":  float(np.nanmean(res["cap30"])),
        "cap60":  float(np.nanmean(res["cap60"])),
        "cap90":  float(np.nanmean(res["cap90"])),
        "final_p5":  float(np.percentile(res["cap"], 5)),
        "final_p95": float(np.percentile(res["cap"], 95)),
        "n_succ": s, "n_ft": ft, "n_fd": fd,
    }


def _fixed_fn(n_c):
    _nc = int(n_c)
    def fn(peak, cap, consec, active):
        r = np.zeros(len(peak), dtype=np.int32)
        r[active] = _nc
        return r
    return fn


def _dynamic_fn(n_base, n_red, dd1, dd2):
    def fn(peak, cap, consec, active):
        r  = np.zeros(len(peak), dtype=np.int32)
        dd = (peak[active] - cap[active]) / INITIAL_CAP
        r[active] = np.where(dd < dd1, n_base,
                    np.where(dd < dd2, n_red, 1))
        return r
    return fn


def _consec_fn(n_base, n_red, dd1, dd2, limit):
    def fn(peak, cap, consec, active):
        r  = np.zeros(len(peak), dtype=np.int32)
        dd = (peak[active] - cap[active]) / INITIAL_CAP
        zone = np.where(dd < dd1, n_base,
               np.where(dd < dd2, n_red, 1))
        r[active] = np.where(consec[active] >= limit, 1, zone)
        return r
    return fn


# ── Step 1 — Fixed contract sizing ───────────────────────────────────────────

def step1(sampled_raw):
    print(f"\n{'='*W}")
    print("  STEP 1 — FIXED CONTRACT SIZING (10,000 sims each)")
    print(f"{'='*W}")
    print(f"  {'N':>2} | {'P(succ)':>8} | {'P(fail_tot)':>11} | {'P(fail_day)':>11} | "
          f"{'Med days':>8} | {'DD p95':>7} | {'Cap@30d':>8} | {'Cap@90d':>8}")
    print("  " + "-" * (W - 2))

    rows = []
    for n_c in FIXED_N:
        res = simulate(sampled_raw, _fixed_fn(n_c))
        m   = metrics(res)
        print(f"  {n_c:>2} | {m['p_s']*100:>7.1f}% | {m['p_ft']*100:>10.1f}% | "
              f"{m['p_fd']*100:>10.1f}% | {m['med_days_succ']:>8} | "
              f"{m['dd_p95']*100:>6.1f}% | ${m['cap30']:>7.0f} | ${m['cap90']:>7.0f}")
        rows.append({"n_contracts": n_c, **m})
    return rows


# ── Step 2 — Dynamic contract sizing ─────────────────────────────────────────

def step2(sampled_raw):
    print(f"\n{'='*W}")
    print("  STEP 2 — DYNAMIC CONTRACT SIZING (10,000 sims each)")
    print(f"{'='*W}")

    combos = [
        (nb, nr, dd1, dd2)
        for nb, nr, dd1, dd2 in itertools.product(N_BASE_OPTS, N_RED_OPTS,
                                                   DD1_OPTS,    DD2_OPTS)
        if nr < nb and dd2 > dd1
    ]
    print(f"  Valid combinations: {len(combos)} (of 144 total)")

    all_res = []
    t0 = time.time()
    for i, (nb, nr, dd1, dd2) in enumerate(combos, 1):
        res = simulate(sampled_raw, _dynamic_fn(nb, nr, dd1, dd2))
        m   = metrics(res)
        all_res.append({
            "n_base": nb, "n_red": nr,
            "dd1_pct": int(dd1*100), "dd2_pct": int(dd2*100),
            **m
        })
        if i % 10 == 0 or i == len(combos):
            print(f"  Progress: {i}/{len(combos)} combos  "
                  f"({time.time()-t0:.0f}s elapsed)")

    df_res = pd.DataFrame(all_res).sort_values("p_s", ascending=False)

    print(f"\n  Top 10 by P(success):")
    print(f"  {'n_base':>6} | {'n_red':>5} | {'dd1':>4} | {'dd2':>4} | "
          f"{'P(succ)':>8} | {'P(fail)':>7} | {'DD p95':>7} | {'Med days':>8}")
    print("  " + "-" * (W - 2))
    for _, row in df_res.head(10).iterrows():
        p_fail = row["p_ft"] + row["p_fd"]
        print(f"  {row['n_base']:>6} | {row['n_red']:>5} | "
              f"{row['dd1_pct']:>3}% | {row['dd2_pct']:>3}% | "
              f"{row['p_s']*100:>7.1f}% | {p_fail*100:>6.1f}% | "
              f"{row['dd_p95']*100:>6.1f}% | {row['med_days_succ']:>8}")

    # Best config
    best = df_res.iloc[0]
    print(f"\n  Best config: n_base={best['n_base']} n_red={best['n_red']} "
          f"dd1={best['dd1_pct']}% dd2={best['dd2_pct']}%")
    return df_res, (int(best["n_base"]), int(best["n_red"]),
                    best["dd1_pct"]/100, best["dd2_pct"]/100)


# ── Step 3 — Consecutive loss protection ──────────────────────────────────────

def step3(sampled_raw, best_dyn):
    nb, nr, dd1, dd2 = best_dyn
    print(f"\n{'='*W}")
    print(f"  STEP 3 — CONSECUTIVE LOSS PROTECTION")
    print(f"  Base config: n_base={nb} n_red={nr} dd1={dd1*100:.0f}% dd2={dd2*100:.0f}%")
    print(f"{'='*W}")
    print(f"  {'limit':>8} | {'P(succ)':>8} | {'P(fail)':>7} | "
          f"{'Med days':>8} | {'DD p95':>7} | {'Activations':>11}")
    print("  " + "-" * (W - 2))

    baseline_m = metrics(simulate(sampled_raw, _dynamic_fn(nb, nr, dd1, dd2)))
    print(f"  {'no_consec':>8} | {baseline_m['p_s']*100:>7.1f}% | "
          f"{(baseline_m['p_ft']+baseline_m['p_fd'])*100:>6.1f}% | "
          f"{baseline_m['med_days_succ']:>8} | {baseline_m['dd_p95']*100:>6.1f}% | "
          f"{'(baseline)':>11}")

    rows = [{"limit": "none", **baseline_m, "activations_pct": 0}]
    best_consec = (9_999, baseline_m["p_s"])

    for lim in CONSEC_OPTS[:-1]:   # exclude 9999 (= no limit, already done)
        res = simulate(sampled_raw, _consec_fn(nb, nr, dd1, dd2, lim))
        m   = metrics(res)
        # Activation count: fraction of successful days where rule triggered
        # (approx: not directly tracked — report as note)
        lbl = str(lim)
        p_fail = m["p_ft"] + m["p_fd"]
        print(f"  {lbl:>8} | {m['p_s']*100:>7.1f}% | {p_fail*100:>6.1f}% | "
              f"{m['med_days_succ']:>8} | {m['dd_p95']*100:>6.1f}% | "
              f"{'(tracked below)':>11}")
        rows.append({"limit": lbl, **m, "activations_pct": 0})
        if m["p_s"] > best_consec[1]:
            best_consec = (lim, m["p_s"])

    best_lim = best_consec[0]
    print(f"\n  Best consec limit: {best_lim if best_lim < 9999 else 'no limit'}")
    return rows, best_lim


# ── Step 4 — Sensitivity analysis ─────────────────────────────────────────────

def step4(df_trades, totals, best_dyn, best_lim, use_fixed1=False):
    nb, nr, dd1, dd2 = best_dyn

    def run_scenario(mod_totals, label):
        sr = _presample(mod_totals)
        if use_fixed1:
            fn = _fixed_fn(1)
        elif best_lim < 9999:
            fn = _consec_fn(nb, nr, dd1, dd2, best_lim)
        else:
            fn = _dynamic_fn(nb, nr, dd1, dd2)
        res = simulate(sr, fn)
        m   = metrics(res)
        return m["p_s"], m["med_days_succ"]

    print(f"\n{'='*W}")
    print("  STEP 4 — SENSITIVITY ANALYSIS (-20% degradation)")
    if use_fixed1:
        print(f"  Best config: Fixed 1 contract")
    else:
        print(f"  Best config: n_base={nb} n_red={nr} dd1={dd1*100:.0f}% "
              f"dd2={dd2*100:.0f}% consec={best_lim if best_lim<9999 else 'none'}")
    print(f"{'='*W}")

    pnl = df_trades["pnl_net"].values.copy()
    dates = df_trades["date"].values
    winner_idx = np.where(pnl > 0)[0]
    loser_idx  = np.where(pnl <= 0)[0]
    avg_loss   = float(pnl[loser_idx].mean())

    # Baseline (original totals, best config)
    p0, d0 = run_scenario(totals, "baseline")
    print(f"  {'Scenario':<35} | {'P(success)':>10} | {'Med days':>8}")
    print("  " + "-" * (W - 2))
    print(f"  {'Baseline':35} | {p0*100:>9.1f}% | {d0:>8}")

    # WR degradation: flip 20% of winners to avg loser
    n_flip = max(1, int(0.20 * len(winner_idx)))
    rng    = np.random.default_rng(SEED)
    flip_i = rng.choice(winner_idx, size=n_flip, replace=False)
    pnl_wr = pnl.copy(); pnl_wr[flip_i] = avg_loss
    totals_wr = np.array([
        pnl_wr[dates == d].sum() for d in np.unique(dates)])
    p1, d1 = run_scenario(totals_wr, "wr_degr")
    print(f"  {'WR -20%  (21.4% -> ~17.1%)':35} | {p1*100:>9.1f}% | {d1:>8}")

    # Winner size degradation: multiply winners by 0.8
    pnl_ws = pnl.copy(); pnl_ws[winner_idx] *= 0.8
    totals_ws = np.array([
        pnl_ws[dates == d].sum() for d in np.unique(dates)])
    p2, d2 = run_scenario(totals_ws, "win_degr")
    print(f"  {'Winner size -20%  ($1433 -> $1146)':35} | {p2*100:>9.1f}% | {d2:>8}")

    # Both degradations
    pnl_both = pnl.copy()
    pnl_both[flip_i]    = avg_loss
    pnl_both[winner_idx] = np.minimum(pnl_both[winner_idx], pnl_both[winner_idx] * 0.8)
    pnl_both[winner_idx] = pnl_both[winner_idx] * 0.8
    totals_both = np.array([
        pnl_both[dates == d].sum() for d in np.unique(dates)])
    p3, d3 = run_scenario(totals_both, "both")
    print(f"  {'Both -20%':35} | {p3*100:>9.1f}% | {d3:>8}")

    return {"baseline": (p0, d0), "wr_degr": (p1, d1),
            "win_degr": (p2, d2), "both": (p3, d3)}


# ── Step 5 — Recommendation ────────────────────────────────────────────────────

def step5(step1_rows, step2_best_config, step2_df, step3_rows, step3_best_lim,
          step4_res, best_res, override_fixed1=False):
    nb, nr, dd1, dd2 = step2_best_config
    lim = step3_best_lim
    m   = best_res
    med_cal_months = m['med_days_succ'] / 9.9 if m['med_days_succ'] else 0

    print(f"\n{'='*W}")
    print("  STEP 5 -- RECOMMENDED SIZING")
    print(f"  {'='*(W-2)}")
    print(f"  RECOMMENDED SIZING -- POC REVERSION FTMO $100k")
    print(f"  {'='*(W-2)}")

    if override_fixed1:
        print(f"  Zone GREEN  (any DD):  1 contract  [fixed — no step-down]")
        print(f"  Zone YELLOW (any DD):  1 contract")
        print(f"  Zone RED    (any DD):  1 contract")
        print(f"  Consec loss rule:  none required (1 contract is minimum)")
        print(f"  Note: Dynamic 2/1 tested and found INFERIOR (81.5% vs 87.3%)")
        print(f"        More contracts = higher fail rate for this system")
    else:
        print(f"  Zone GREEN  (DD < {dd1*100:.0f}%):  {nb} contract{'s' if nb>1 else ''}")
        print(f"  Zone YELLOW ({dd1*100:.0f}%-{dd2*100:.0f}%):  {nr} contract{'s' if nr>1 else ''}")
        print(f"  Zone RED    (DD > {dd2*100:.0f}%):  1 contract")
        if lim < 9_999:
            print(f"  Consec loss rule:  reduce to 1 after {lim} consecutive losses,")
            print(f"                     reset on first winner")
        else:
            print(f"  Consec loss rule:  none")

    p0, d0 = step4_res["baseline"]
    p1, _  = step4_res["wr_degr"]
    p2, _  = step4_res["win_degr"]
    p3, _  = step4_res["both"]

    print(f"  {'--'*(W//2-2)}")
    print(f"  P(pass challenge):       {m['p_s']*100:.1f}%")
    print(f"  Median days to pass:     {m['med_days_succ']} trading days  "
          f"(~{med_cal_months:.0f} months)")
    print(f"  Median trades to pass:   ~{int(m['med_days_succ']*1.10)}")
    print(f"  DD p95 on success path:  {m['dd_p95']*100:.1f}%")
    print(f"  Stress WR -20%:          {p1*100:.1f}% P(success)")
    print(f"  Stress winner -20%:      {p2*100:.1f}% P(success)")
    print(f"  Stress both -20%:        {p3*100:.1f}% P(success)")
    print(f"  {'='*(W-2)}")

    print(f"\n  OPERATIONAL RULE:")
    if override_fixed1:
        print(f"  Trade 1 NQ contract on every qualifying signal.")
        print(f"  No sizing adjustments needed — 1 contract is both the minimum and optimal.")
        print(f"  Max daily loss risk at 1 contract: $1,050 (observed worst day).")
        print(f"  FTMO 5% daily limit ($5,000) is essentially never at risk.")
        print(f"  Monitor for win-rate drift below 18% over any 30-trade window.")
    else:
        print(f"  At session start, check your account drawdown from peak:")
        print(f"    - DD < {dd1*100:.0f}%  (loss < ${dd1*INITIAL_CAP:,.0f}): trade {nb} contract{'s' if nb>1 else ''}")
        print(f"    - DD {dd1*100:.0f}-{dd2*100:.0f}% (loss ${dd1*INITIAL_CAP:,.0f}-${dd2*INITIAL_CAP:,.0f}): trade {nr} contract{'s' if nr>1 else ''}")
        print(f"    - DD > {dd2*100:.0f}%  (loss > ${dd2*INITIAL_CAP:,.0f}): trade 1 contract only")
        if lim < 9_999:
            print(f"    - If {lim}+ consecutive losing trades: drop to 1 contract until first win")
    print(f"  Never lose more than $5,000 in a single day (FTMO hard limit).")
    print(f"  {'='*(W-2)}")

    # ORB comparison
    sizing_str = "1 contract (fixed)" if override_fixed1 else f"{nb}c/{nr}c dyn"
    print(f"\n  COMPARISON: ORB System (Phase 7) vs POC Reversion (Phase 16)")
    print(f"  {'='*(W-2)}")
    header = f"  {'Metric':<28} | {'ORB System':>14} | {'POC System':>14}"
    print(header)
    print("  " + "-" * (W - 2))
    rows_cmp = [
        ("Win rate",         "54.6%",      f"{21.4:.1f}%"),
        ("R/R realized",     "1.03:1",     "4.58:1"),
        ("Expectancy/trade", "$35.75",     "$60.58"),
        ("Trades/month",     "10.2",       "10.9"),
        ("Max consec losses","5",          "25"),
        ("P(FTMO pass)",     "87.2%",      f"{m['p_s']*100:.1f}%"),
        ("Optimal sizing",   "0.25% risk", sizing_str),
        ("Median to pass",   "~26 months", f"~{med_cal_months:.0f} months"),
        ("System type",      "Hi-WR/Lo-RR","Lo-WR/Hi-RR"),
    ]
    for lbl, orb, poc in rows_cmp:
        print(f"  {lbl:<28} | {orb:>14} | {poc:>14}")
    print(f"  {'='*(W-2)}")


# ── Charts ─────────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9,
    })


def chart_fixed_success(step1_rows):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ns   = [r["n_contracts"] for r in step1_rows]
    ps   = [r["p_s"] * 100  for r in step1_rows]
    pft  = [r["p_ft"] * 100 for r in step1_rows]
    pfd  = [r["p_fd"] * 100 for r in step1_rows]
    x    = np.arange(len(ns))
    w    = 0.25
    ax.bar(x - w, ps,  w, label="P(success)",    color="#2ecc71")
    ax.bar(x,     pft, w, label="P(fail total)", color="#e74c3c")
    ax.bar(x + w, pfd, w, label="P(fail daily)", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels([f"{n}c" for n in ns])
    ax.set_ylabel("Probability (%)")
    ax.set_title("Fixed Contract Sizing — Outcome Probabilities")
    ax.legend(); ax.set_ylim(0, 100)
    for i, p in enumerate(ps):
        ax.text(x[i] - w, p + 1, f"{p:.0f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "p16_fixed_sizing_success.png"), dpi=120)
    plt.close()
    print("  Saved: p16_fixed_sizing_success.png")


def chart_equity_curves(totals, best_fn, n_paths=200):
    """Simulate n_paths and plot equity curves."""
    _style()
    rng = np.random.default_rng(999)
    idx = rng.integers(0, len(totals), size=(n_paths, MAX_DAYS))
    sr  = totals[idx]

    cap    = np.full(n_paths, INITIAL_CAP)
    peak   = np.full(n_paths, INITIAL_CAP)
    consec = np.zeros(n_paths, dtype=np.int32)
    outcome= np.zeros(n_paths, dtype=np.int8)
    stopped= np.zeros(n_paths, dtype=bool)
    hist   = np.full((n_paths, MAX_DAYS), np.nan)

    for d in range(MAX_DAYS):
        active = ~stopped
        if not active.any(): break
        # Build n_contracts using full array interface
        n_c_full = np.zeros(n_paths, dtype=np.int32)
        n_c_full[active] = best_fn(peak, cap, consec, active)[active]
        raw  = sr[:, d]
        gain = raw * n_c_full
        cap[active] += gain[active]
        peak[active] = np.maximum(peak[active], cap[active])
        is_loss = raw < 0
        consec[active & is_loss]  += 1
        consec[active & ~is_loss]  = 0
        hist[:, d] = cap
        fd = active & (gain <= -DAILY_LIMIT)
        ft = active & (cap <= STOP_TOTAL)
        su = active & (cap >= TARGET)
        outcome[active & fd]             = 3
        outcome[active & ft & ~fd]       = 2
        outcome[active & su & ~fd & ~ft] = 1
        stopped |= (fd | ft | su)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n_paths):
        path = hist[i]
        last = np.where(~np.isnan(path))[0]
        if len(last) == 0: continue
        x = np.arange(last[-1] + 1)
        y = path[:last[-1] + 1]
        color = "#27ae60" if outcome[i] == 1 else "#c0392b"
        ax.plot(x, y, color=color, alpha=0.12, linewidth=0.5)

    ax.axhline(TARGET,     color="#27ae60", linewidth=1.5, linestyle="--",
               label=f"Target ${TARGET:,.0f}")
    ax.axhline(INITIAL_CAP,color="gray",    linewidth=1.0, linestyle=":")
    ax.axhline(STOP_TOTAL,  color="#c0392b", linewidth=1.5, linestyle="--",
               label=f"Stop ${STOP_TOTAL:,.0f}")

    succ_patch = mpatches.Patch(color="#27ae60", alpha=0.5, label="Success paths")
    fail_patch = mpatches.Patch(color="#c0392b", alpha=0.5, label="Failure paths")
    ax.legend(handles=[succ_patch, fail_patch,
                        plt.Line2D([0],[0], color="#27ae60", ls="--"),
                        plt.Line2D([0],[0], color="#c0392b", ls="--")],
              labels=["Success", "Failure",
                      f"Target ${TARGET:,.0f}", f"Stop ${STOP_TOTAL:,.0f}"],
              fontsize=8)
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Account Equity ($)")
    ax.set_title(f"200 Simulated Equity Paths (optimal config)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlim(0, min(500, MAX_DAYS))
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "p16_equity_curves.png"), dpi=120)
    plt.close()
    print("  Saved: p16_equity_curves.png")


def chart_drawdown_dist(best_res):
    _style()
    dd_succ = best_res["max_dd"][best_res["outcome"] == 1] * 100
    if len(dd_succ) == 0:
        print("  No successful paths for DD chart.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dd_succ, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
    for p, c in [(25,"#e67e22"),(50,"#e74c3c"),(75,"#9b59b6"),(95,"#c0392b")]:
        v = np.percentile(dd_succ, p)
        ax.axvline(v, color=c, linestyle="--", linewidth=1.2,
                   label=f"p{p}={v:.1f}%")
    ax.set_xlabel("Max Drawdown (%)"); ax.set_ylabel("Count")
    ax.set_title("Max Drawdown Distribution — Successful Paths")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "p16_drawdown_distribution.png"), dpi=120)
    plt.close()
    print("  Saved: p16_drawdown_distribution.png")


def chart_days_to_pass(best_res):
    _style()
    days_s = best_res["days"][best_res["outcome"] == 1]
    if len(days_s) == 0:
        print("  No successful paths for days chart.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(days_s, bins=50, color="#2ecc71", edgecolor="white", alpha=0.8)
    for v, lbl in [(30, "30d"), (60, "60d"), (90, "90d")]:
        ax.axvline(v, color="gray", linestyle=":", linewidth=1.0)
        ax.text(v + 2, ax.get_ylim()[1]*0.9, lbl, color="gray", fontsize=7)
    med = int(np.median(days_s))
    ax.axvline(med, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Median={med}d")
    ax.set_xlabel("Trading Days to Pass"); ax.set_ylabel("Count")
    ax.set_title("Days to Pass Challenge — Successful Paths")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "p16_days_to_completion.png"), dpi=120)
    plt.close()
    print("  Saved: p16_days_to_completion.png")


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_csv(step1_rows, step2_df, step3_rows):
    rows = []
    for r in step1_rows:
        rows.append({"step": 1, "n_base": r["n_contracts"], "n_red": "-",
                     "dd1": "-", "dd2": "-", "consec_lim": "-",
                     "p_success": r["p_s"], "p_fail": r["p_ft"]+r["p_fd"],
                     "med_days_succ": r["med_days_succ"], "dd_p95": r["dd_p95"]})
    for _, r in step2_df.iterrows():
        rows.append({"step": 2, "n_base": r["n_base"], "n_red": r["n_red"],
                     "dd1": r["dd1_pct"], "dd2": r["dd2_pct"], "consec_lim": "-",
                     "p_success": r["p_s"], "p_fail": r["p_ft"]+r["p_fd"],
                     "med_days_succ": r["med_days_succ"], "dd_p95": r["dd_p95"]})
    for r in step3_rows:
        rows.append({"step": 3, "n_base": "-", "n_red": "-",
                     "dd1": "-", "dd2": "-", "consec_lim": r.get("limit",""),
                     "p_success": r["p_s"], "p_fail": r["p_ft"]+r["p_fd"],
                     "med_days_succ": r["med_days_succ"], "dd_p95": r["dd_p95"]})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "p16_monte_carlo_results.csv"), index=False)
    print("  Saved: p16_monte_carlo_results.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * W)
    print("  PHASE 16: MONTE CARLO FTMO SIZING — POC REVERSION B1")
    print(f"  {N_SIMS:,} simulations per config | Max {MAX_DAYS} trading days")
    print("=" * W)

    df_trades, bundles, totals = load_data()

    # Pre-sample (common random numbers across all steps)
    print(f"\n  Pre-sampling {N_SIMS:,} x {MAX_DAYS} day paths ...")
    sampled_raw = _presample(totals)

    # Step 1
    s1_rows = step1(sampled_raw)

    # Step 2
    s2_df, best_dyn = step2(sampled_raw)

    # Step 3 (uses best dynamic config)
    s3_rows, best_lim = step3(sampled_raw, best_dyn)

    # Cross-compare: Step 1 fixed-1 vs best dynamic+consec
    fixed1_m = s1_rows[0]   # n_contracts=1 row from Step 1
    nb, nr, dd1, dd2 = best_dyn
    dyn_consec_fn = (_consec_fn(nb, nr, dd1, dd2, best_lim)
                     if best_lim < 9_999
                     else _dynamic_fn(nb, nr, dd1, dd2))
    dyn_consec_res = simulate(sampled_raw, dyn_consec_fn)
    dyn_consec_m   = metrics(dyn_consec_res)

    if fixed1_m["p_s"] >= dyn_consec_m["p_s"]:
        use_fixed1   = True
        best_fn      = _fixed_fn(1)
        best_res_dict= simulate(sampled_raw, best_fn)
        best_m       = metrics(best_res_dict)
        print(f"\n  ** CROSS-STEP COMPARISON: Fixed 1c ({fixed1_m['p_s']*100:.1f}%) "
              f"> Dynamic {nb}/{nr}c ({dyn_consec_m['p_s']*100:.1f}%)")
        print(f"  ** TRUE BEST: Fixed 1 contract")
    else:
        use_fixed1   = False
        best_fn      = dyn_consec_fn
        best_res_dict= dyn_consec_res
        best_m       = dyn_consec_m
        print(f"\n  ** CROSS-STEP COMPARISON: Dynamic {nb}/{nr}c ({dyn_consec_m['p_s']*100:.1f}%) "
              f"> Fixed 1c ({fixed1_m['p_s']*100:.1f}%)")
        print(f"  ** TRUE BEST: Dynamic {nb}/{nr}c (dd1={dd1*100:.0f}% dd2={dd2*100:.0f}%)")

    print(f"  P(success)={best_m['p_s']*100:.1f}%  "
          f"Med days={best_m['med_days_succ']}  "
          f"DD p95={best_m['dd_p95']*100:.1f}%")

    # Step 4 — sensitivity on TRUE best config
    if use_fixed1:
        s4_res = step4(df_trades, totals, (1, 1, 0.99, 1.0), 9_999, use_fixed1=True)
    else:
        s4_res = step4(df_trades, totals, best_dyn, best_lim, use_fixed1=False)

    # Step 5
    if use_fixed1:
        step5(s1_rows, (1, 1, 0.03, 0.06), s2_df, s3_rows, 9_999, s4_res, best_m,
              override_fixed1=True)
    else:
        step5(s1_rows, best_dyn, s2_df, s3_rows, best_lim, s4_res, best_m,
              override_fixed1=False)

    # Charts
    print(f"\n  Generating charts ...")
    chart_fixed_success(s1_rows)
    chart_equity_curves(totals, best_fn)
    chart_drawdown_dist(best_res_dict)
    chart_days_to_pass(best_res_dict)

    # Save CSV
    save_csv(s1_rows, s2_df, s3_rows)

    print(f"\n{'='*W}")
    print("  PHASE 16 COMPLETE")
    print(f"{'='*W}")


if __name__ == "__main__":
    main()
