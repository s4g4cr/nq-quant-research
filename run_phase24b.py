#!/usr/bin/env python3
"""
Phase 24B -- Monte Carlo FTMO: Percentage-Based Position Sizing
POC B1 + HMM Ranging Filter (A1)

Uses actual per-trade sl_dist from p23_wfo_oos_pooled.csv.

n_contracts = max(1, floor(capital * risk_pct / (sl_dist * 20)))

FTMO $100k: target=$110k | floor=$90k | daily_limit=-$5k
"""

import itertools
import math
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')

INITIAL  = 100_000.0
TARGET   = 110_000.0
FLOOR    =  90_000.0
DAILY_L  =  -5_000.0
N_SIMS   =  10_000
PV       =  20.0
MIN_RP   =  0.005          # 0.5% minimum
W        =  78


# ── Data ──────────────────────────────────────────────────────────────────────
def load_buckets(path):
    df = pd.read_csv(path)
    df['entry_ts']   = pd.to_datetime(df['entry_ts'], utc=True)
    df['entry_date'] = df['entry_ts'].dt.date
    return [list(zip(g['pnl_net'].values, g['sl_dist'].values))
            for _, g in df.groupby('entry_date')]


# ── get_rp factories ──────────────────────────────────────────────────────────
def fixed_pct(rp):
    return lambda cap, peak, cl: rp


def dynamic_pct(rp_base, rp_red, dd1, dd2):
    def fn(cap, peak, cl):
        dd = (peak - cap) / INITIAL
        if   dd >= dd2: return MIN_RP
        elif dd >= dd1: return rp_red
        else:           return rp_base
    return fn


def dyn_consec_pct(rp_base, rp_red, dd1, dd2, cl_limit):
    base = dynamic_pct(rp_base, rp_red, dd1, dd2)
    def fn(cap, peak, cl):
        if cl_limit is not None and cl >= cl_limit:
            return MIN_RP
        return base(cap, peak, cl)
    return fn


# ── Core simulation ───────────────────────────────────────────────────────────
def simulate(buckets, n_sims, get_rp, seed=42, track_paths=0):
    rng    = np.random.default_rng(seed)
    n_pool = len(buckets)
    outcomes, tdays, maxdd, capend = [], [], [], []
    cap_ms = {30: [], 60: [], 90: []}
    paths  = []
    n_sum  = 0; n_days = 0   # for avg_nc across all sims

    for si in range(n_sims):
        cap    = INITIAL; peak = INITIAL
        consec = 0; td = 0; mdd = 0.0
        ms_cap = {30: INITIAL, 60: INITIAL, 90: INITIAL}
        path   = [INITIAL] if si < track_paths else None
        out    = None

        while out is None:
            rp  = get_rp(cap, peak, consec)
            idx = int(rng.integers(0, n_pool))
            day = buckets[idx]
            dpnl = 0.0
            trade_results = []
            for pnl, sl in day:
                n     = max(1, math.floor(cap * rp / (sl * PV)))
                dpnl += pnl * n
                trade_results.append((pnl, n))
                n_sum  += n
                n_days += 1

            if dpnl < DAILY_L:
                out = 'fail_daily'; cap += dpnl; break

            cap  += dpnl
            peak  = max(peak, cap)
            td   += 1

            for pnl, n in trade_results:
                if pnl <= 0: consec += 1
                else:        consec  = 0

            dd = (peak - cap) / INITIAL
            if dd > mdd: mdd = dd

            for m in (30, 60, 90):
                if td == m: ms_cap[m] = cap

            if path is not None: path.append(cap)

            if   cap <= FLOOR:  out = 'fail_total'
            elif cap >= TARGET: out = 'success'

        outcomes.append(out); tdays.append(td)
        maxdd.append(mdd * 100); capend.append(cap)
        for m in (30, 60, 90): cap_ms[m].append(ms_cap[m])
        if si < track_paths: paths.append(path)

    td_a = np.array(tdays); dd_a = np.array(maxdd); ce_a = np.array(capend)
    sm   = np.array([o == 'success' for o in outcomes])
    return {
        'outcomes': outcomes, 'tdays': td_a, 'maxdd': dd_a, 'capend': ce_a,
        'succ_mask': sm,
        'cap_ms': {m: np.array(v) for m, v in cap_ms.items()},
        'paths': paths,
        'avg_nc': n_sum / max(n_days, 1),
    }


def S(r, label=''):
    o  = r['outcomes']; td = r['tdays']; dd = r['maxdd']
    ce = r['capend'];   sm = r['succ_mask']; n = len(o)
    ns  = int(sm.sum())
    nft = sum(1 for x in o if x == 'fail_total')
    nfd = sum(1 for x in o if x == 'fail_daily')
    td_s = td[sm]; dd_s = dd[sm]
    med_td = float(np.median(td_s)) if td_s.size else float('nan')
    med_cd = med_td * 365 / 252 if not math.isnan(med_td) else float('nan')
    pct = lambda v, p: float(np.percentile(v, p)) if len(v) else float('nan')
    return {
        'label': label,
        'p_succ': ns / n * 100, 'p_ftot': nft / n * 100, 'p_fdly': nfd / n * 100,
        'med_td': med_td, 'med_cd': med_cd,
        'dd_p50': pct(dd_s, 50), 'dd_p75': pct(dd_s, 75), 'dd_p95': pct(dd_s, 95),
        'ce_p5':  pct(ce, 5),  'ce_p25': pct(ce, 25),
        'ce_p50': pct(ce, 50), 'ce_p75': pct(ce, 75), 'ce_p95': pct(ce, 95),
        'cap_ms': {m: float(np.mean(r['cap_ms'][m])) for m in (30, 60, 90)},
        'avg_nc': r['avg_nc'],
        'td_s': td_s, 'dd_s': dd_s, 'sm': sm, 'ce': ce,
    }


# ── Stress helpers ─────────────────────────────────────────────────────────────
def stress_wr(buckets, factor=0.80, seed=1234):
    """Convert 20% of winning trades to avg_loser. sl_dist unchanged."""
    rng_s = np.random.default_rng(seed)
    all_l = [pnl for b in buckets for pnl, sl in b if pnl <= 0]
    avg_l = float(np.mean(all_l)) if all_l else 0.0
    return [[(pnl if pnl <= 0 or rng_s.random() > (1 - factor) else avg_l, sl)
             for pnl, sl in b] for b in buckets]

def stress_win(buckets, factor=0.80):
    """Scale all winning trades by factor. sl_dist unchanged."""
    return [[(pnl * factor if pnl > 0 else pnl, sl) for pnl, sl in b] for b in buckets]


# ── Charts ─────────────────────────────────────────────────────────────────────
def chart_fixed_success(step1_rows, fpath):
    labels = [f'{r["rp"]*100:.2f}%' for r in step1_rows]
    ps     = [r['s']['p_succ'] for r in step1_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, ps, color='#2563eb', edgecolor='white', width=0.6)
    for b, v in zip(bars, ps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xlabel('Risk % per trade', fontsize=12)
    ax.set_ylabel('P(FTMO Pass) %', fontsize=12)
    ax.set_title('Fixed Risk % -- P(FTMO Pass)\nPOC B1 + HMM Ranging (A1)', fontsize=13)
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()


def chart_equity_curves(buckets, get_rp, fpath, n_paths=200, seed=99):
    r = simulate(buckets, n_paths, get_rp, seed=seed, track_paths=n_paths)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, path in enumerate(r['paths']):
        c = '#16a34a' if r['outcomes'][i] == 'success' else '#dc2626'
        ax.plot(path, color=c, alpha=0.14, linewidth=0.6)
    ax.axhline(TARGET,  color='#16a34a', linewidth=1.5, label=f'Target ${TARGET:,.0f}')
    ax.axhline(INITIAL, color='#888',    linewidth=1.0, linestyle='--',
               label=f'Start ${INITIAL:,.0f}')
    ax.axhline(FLOOR,   color='#dc2626', linewidth=1.5, label=f'Floor ${FLOOR:,.0f}')
    ax.set_xlabel('Trading Days', fontsize=12); ax.set_ylabel('Account ($)', fontsize=12)
    ax.set_title('200 Simulated FTMO Paths (Green=Pass, Red=Fail)\nOptimal Config', fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()


def chart_days_to_pass(td_s, fpath):
    cd_s = td_s * 365 / 252
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(cd_s, bins=40, color='#16a34a', edgecolor='white', alpha=0.8)
    for d, c in [(30,'#bbb'),(60,'#999'),(90,'#666'),(180,'#333'),(365,'#dc2626')]:
        ax.axvline(d, color=c, linewidth=1.2, linestyle='--', label=f'{d}d')
    ax.set_xlabel('Calendar Days to Pass', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Days to Pass FTMO -- Successful Paths\nOptimal Config', fontsize=13)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * W)
    print('  PHASE 24B -- MONTE CARLO FTMO: PERCENTAGE-BASED SIZING (A1)')
    print('=' * W)
    print(f'  FTMO ${INITIAL:,.0f}: target=${TARGET:,.0f} | '
          f'floor=${FLOOR:,.0f} | daily_limit=${DAILY_L:,.0f}')
    print(f'  {N_SIMS:,} sims per config | resample by calendar day')
    print(f'  sl_dist: actual per-trade value from p23_wfo_oos_pooled.csv')
    print(f'  n = max(1, floor(capital x risk_pct / (sl_dist x {PV:.0f})))')

    # ── Load data ─────────────────────────────────────────────────────────────
    a1_path = os.path.join(RESULTS, 'p23_wfo_oos_pooled.csv')
    bk_a1   = load_buckets(a1_path)
    n_t     = sum(len(b) for b in bk_a1)
    print(f'\n  A1 pool: {n_t} trades  {len(bk_a1)} active trading days')

    bk_swr  = stress_wr(bk_a1)
    bk_swin = stress_win(bk_a1)
    bk_both = stress_win(stress_wr(bk_a1))

    # =====================================================================
    # STEP 1 — Fixed risk_pct
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 1 -- FIXED RISK PCT SIZING')
    print(f'  {"="*W}')
    print(f'\n  {"risk%":>6} | {"P(succ)":>8} | {"P(fail)":>8} | {"P(daily)":>9} | '
          f'{"MedDays":>8} | {"DD p95":>7} | {"Avg_nc":>6}')
    print('  ' + '-' * 72)

    step1_rows = []
    for rp in [0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020]:
        r = simulate(bk_a1, N_SIMS, fixed_pct(rp), seed=42)
        s = S(r)
        step1_rows.append({'rp': rp, 's': s, 'r': r})
        print(f'  {rp*100:>5.2f}% | {s["p_succ"]:>7.1f}% | {s["p_ftot"]:>7.1f}% | '
              f'{s["p_fdly"]:>8.1f}% | {s["med_cd"]:>7.0f}d | '
              f'{s["dd_p95"]:>6.1f}% | {s["avg_nc"]:>6.2f}')

    # =====================================================================
    # STEP 2 — Dynamic risk_pct grid
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 2 -- DYNAMIC RISK PCT GRID')
    print(f'  {"="*W}')
    print('  GREEN(DD<dd1)=rp_base | YELLOW(dd1-dd2)=rp_red | RED(DD>=dd2)=0.50%\n')

    valid_cfgs = [
        (rpb, rpr, d1, d2)
        for rpb, rpr, d1, d2 in itertools.product(
            [0.010, 0.015, 0.020],
            [0.005, 0.0075, 0.010],
            [0.02, 0.03, 0.04],
            [0.05, 0.06, 0.07],
        )
        if rpr < rpb
    ]
    total_cfgs = len(valid_cfgs)
    grid = []

    for ci, (rpb, rpr, d1, d2) in enumerate(valid_cfgs, 1):
        r = simulate(bk_a1, N_SIMS, dynamic_pct(rpb, rpr, d1, d2), seed=42)
        s = S(r)
        grid.append({'rpb': rpb, 'rpr': rpr, 'd1': d1, 'd2': d2, 's': s})
        print(f'  [{ci:>2}/{total_cfgs}] rp_base={rpb*100:.2f}% rp_red={rpr*100:.2f}% '
              f'dd1={d1*100:.0f}% dd2={d2*100:.0f}%  '
              f'P(succ)={s["p_succ"]:.1f}%  '
              f'DD_p95={s["dd_p95"]:.1f}%')

    grid.sort(key=lambda x: -x['s']['p_succ'])
    print(f'\n  TOP 10 BY P(SUCCESS):')
    print(f'  {"rpb":>7} | {"rpr":>7} | {"dd1":>5} | {"dd2":>5} | '
          f'{"P(succ)":>8} | {"P(fail)":>8} | {"DDp95":>6} | {"MedDays":>8} | {"Avg_nc":>6}')
    print('  ' + '-' * 80)
    for row in grid[:10]:
        s = row['s']
        print(f'  {row["rpb"]*100:>6.2f}% | {row["rpr"]*100:>6.2f}% | '
              f'{row["d1"]*100:>4.0f}% | {row["d2"]*100:>4.0f}% | '
              f'{s["p_succ"]:>7.1f}% | '
              f'{s["p_ftot"]+s["p_fdly"]:>7.1f}% | '
              f'{s["dd_p95"]:>5.1f}% | {s["med_cd"]:>7.0f}d | {s["avg_nc"]:>6.2f}')

    best = grid[0]
    opt_rpb = best['rpb']; opt_rpr = best['rpr']
    opt_d1  = best['d1'];  opt_d2  = best['d2']
    print(f'\n  Best: rp_base={opt_rpb*100:.2f}% rp_red={opt_rpr*100:.2f}% '
          f'dd1={opt_d1*100:.0f}% dd2={opt_d2*100:.0f}%  '
          f'P(succ)={best["s"]["p_succ"]:.1f}%')

    # =====================================================================
    # STEP 3 — Consecutive loss protection
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 3 -- CONSECUTIVE LOSS PROTECTION')
    print(f'  {"="*W}')
    print(f'  Base: rp_base={opt_rpb*100:.2f}% rp_red={opt_rpr*100:.2f}% '
          f'dd1={opt_d1*100:.0f}% dd2={opt_d2*100:.0f}%\n')
    print(f'  {"Limit":>8} | {"P(succ)":>8} | {"P(fail)":>8} | '
          f'{"DDp95":>6} | {"MedDays":>8} | {"ActRate%":>9}')
    print('  ' + '-' * 62)

    step3 = []
    best_cl_ps = -1.0; opt_cl = None

    def make_fn(rpb, rpr, d1, d2, cl_lim, act_c, tot_c):
        base_fn = dynamic_pct(rpb, rpr, d1, d2)
        def fn(cap, peak, clv):
            tot_c[0] += 1
            if cl_lim is not None and clv >= cl_lim:
                act_c[0] += 1
                return MIN_RP
            return base_fn(cap, peak, clv)
        return fn

    for cl in [3, 5, 8, 10, None]:
        act_c = [0]; tot_c = [0]
        gn = make_fn(opt_rpb, opt_rpr, opt_d1, opt_d2, cl, act_c, tot_c)
        r  = simulate(bk_a1, N_SIMS, gn, seed=42)
        s  = S(r)
        act_rate = act_c[0] / max(tot_c[0], 1) * 100
        p_fail   = s['p_ftot'] + s['p_fdly']
        cl_str   = str(cl) if cl is not None else 'none'
        print(f'  {cl_str:>8} | {s["p_succ"]:>7.1f}% | {p_fail:>7.1f}% | '
              f'{s["dd_p95"]:>5.1f}% | {s["med_cd"]:>7.0f}d | {act_rate:>8.1f}%')
        step3.append({'cl': cl, 's': s, 'act_rate': act_rate})
        if s['p_succ'] > best_cl_ps:
            best_cl_ps = s['p_succ']
            opt_cl = cl

    print(f'\n  Best consec limit: {opt_cl}  P(succ)={best_cl_ps:.1f}%')

    get_n_opt = dyn_consec_pct(opt_rpb, opt_rpr, opt_d1, opt_d2, opt_cl)
    r_opt = simulate(bk_a1, N_SIMS, get_n_opt, seed=42, track_paths=200)
    s_opt = S(r_opt, 'optimal')
    cl_str = str(opt_cl) if opt_cl is not None else 'none'
    print(f'  Optimal: rp_base={opt_rpb*100:.2f}% rp_red={opt_rpr*100:.2f}% '
          f'dd1={opt_d1*100:.0f}% dd2={opt_d2*100:.0f}% cl={cl_str}  '
          f'P(succ)={s_opt["p_succ"]:.1f}%  '
          f'DD p95={s_opt["dd_p95"]:.1f}%  Med={s_opt["med_cd"]:.0f}d  '
          f'Avg_nc={s_opt["avg_nc"]:.2f}')

    # =====================================================================
    # STEP 4 — Sensitivity analysis
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 4 -- SENSITIVITY ANALYSIS')
    print(f'  {"="*W}')
    print(f'\n  {"Config":<22} | {"P(succ)":>8} | {"vs base":>8} | '
          f'{"DDp95":>6} | {"MedDays":>8}')
    print('  ' + '-' * 62)

    s_base_ps   = s_opt['p_succ']
    stress_runs = {}
    for name, bk in [
        ('Baseline A1',   bk_a1),
        ('WR -20% (19%)', bk_swr),
        ('Winner -20%',   bk_swin),
        ('Both -20%',     bk_both),
    ]:
        r = simulate(bk, N_SIMS, get_n_opt, seed=42)
        s = S(r)
        stress_runs[name] = s
        delta = s['p_succ'] - s_base_ps
        print(f'  {name:<22} | {s["p_succ"]:>7.1f}% | '
              f'{delta:>+7.1f}pp | {s["dd_p95"]:>5.1f}% | {s["med_cd"]:>7.0f}d')

    # =====================================================================
    # STEP 5 — Days to completion
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 5 -- DAYS TO COMPLETION (successful paths)')
    print(f'  {"="*W}')
    td_s = s_opt['td_s']
    cd_s = td_s * 365 / 252
    print(f'\n  Successful paths: {len(td_s)} / {N_SIMS} '
          f'({len(td_s)/N_SIMS*100:.1f}%)')
    print(f'\n  Calendar days:')
    for p in [10, 25, 50, 75, 90, 95]:
        print(f'    p{p:>2}: {np.percentile(cd_s, p):>6.0f}d')
    print(f'\n  % completing within:')
    for d in [30, 60, 90, 120, 180, 365]:
        pct = (cd_s <= d).mean() * 100
        print(f'    {d:>4} days: {pct:>5.1f}%  ({int((cd_s <= d).sum())} paths)')

    # =====================================================================
    # STEP 6 — Recommendation
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 6 -- RECOMMENDED SIZING')
    print(f'  {"="*W}')
    green_fl    = INITIAL * (1 - opt_d1)
    yellow_fl   = INITIAL * (1 - opt_d2)
    s_swr       = stress_runs['WR -20% (19%)']
    s_swin      = stress_runs['Winner -20%']
    s_both      = stress_runs['Both -20%']
    all_sl      = [sl for b in bk_a1 for _, sl in b]
    avg_sl_cost = float(np.mean(all_sl)) * PV
    ex_100k     = max(1, math.floor(100_000 * opt_rpb / avg_sl_cost))
    ex_105k     = max(1, math.floor(105_000 * opt_rpb / avg_sl_cost))
    ex_98k      = max(1, math.floor(98_000 * opt_rpr / avg_sl_cost))
    ex_94k      = max(1, math.floor(94_000 * MIN_RP / avg_sl_cost))

    print(f"""
  =====================================================================
  RECOMMENDED SIZING -- POC B1 + HMM RANGING -- FTMO $100k
  PERCENTAGE-BASED SIZING
  =====================================================================
  Zone GREEN  (DD < {opt_d1*100:.0f}%, acct > ${green_fl:,.0f}):
    risk_pct = {opt_rpb*100:.2f}%
    e.g. at $100k: n={ex_100k}c  |  at $105k: n={ex_105k}c

  Zone YELLOW ({opt_d1*100:.0f}% <= DD < {opt_d2*100:.0f}%, ${yellow_fl:,.0f}-${green_fl:,.0f}):
    risk_pct = {opt_rpr*100:.2f}%
    e.g. at $96k: n={ex_98k}c

  Zone RED    (DD >= {opt_d2*100:.0f}%, acct < ${yellow_fl:,.0f}):
    risk_pct = 0.50% (minimum)
    e.g. at $94k: n={ex_94k}c

  Consec loss rule: drop to 0.50% after {cl_str} consecutive losses
                    reset to zone sizing on first winning trade
  ---------------------------------------------------------------------
  P(pass challenge):        {s_opt["p_succ"]:>5.1f}%
  Median days to pass:      {s_opt["med_cd"]:>4.0f} calendar days
  P(pass within 90 days):   {(cd_s <= 90).mean()*100:>5.1f}%
  P(pass within 180 days):  {(cd_s <= 180).mean()*100:>5.1f}%
  DD p50 on success paths:  {s_opt["dd_p50"]:>5.1f}%
  DD p95 on success paths:  {s_opt["dd_p95"]:>5.1f}%
  Avg contracts per day:    {s_opt["avg_nc"]:>5.2f}
  Stress WR -20%:           {s_swr["p_succ"]:>5.1f}%
  Stress winner -20%:       {s_swin["p_succ"]:>5.1f}%
  Stress both -20%:         {s_both["p_succ"]:>5.1f}%
  =====================================================================

  OPERATIONAL RULE (plain language):
  "Risk {opt_rpb*100:.2f}% of current account per trade. Compute
   n = floor(account x {opt_rpb*100:.2f}% / (sl_dist_pts x $20)).
   Minimum 1 contract. If account drops below ${green_fl:,.0f}
   (DD={opt_d1*100:.0f}%), switch to {opt_rpr*100:.2f}% risk. If
   account drops below ${yellow_fl:,.0f} (DD={opt_d2*100:.0f}%),
   use minimum 0.50% risk. After {cl_str} consecutive losing trades,
   drop to 0.50% risk until first winning trade."

  NOTE: sl_dist is the actual per-trade SL distance in points from
  the entry signal. Avg sl_dist in this dataset = {float(np.mean(all_sl)):.1f} pts
  (avg risk per contract = ${avg_sl_cost:.0f}).""")

    # =====================================================================
    # CHARTS
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  SAVING CHARTS')
    print(f'  {"="*W}')

    chart_fixed_success(
        step1_rows, os.path.join(RESULTS, 'p24b_fixed_risk_success.png'))
    print('  Saved: p24b_fixed_risk_success.png')

    chart_equity_curves(
        bk_a1, get_n_opt, os.path.join(RESULTS, 'p24b_equity_curves.png'))
    print('  Saved: p24b_equity_curves.png')

    chart_days_to_pass(
        s_opt['td_s'], os.path.join(RESULTS, 'p24b_days_to_completion.png'))
    print('  Saved: p24b_days_to_completion.png')

    # =====================================================================
    # SAVE CSV
    # =====================================================================
    csv_rows = []
    for row in step1_rows:
        s = row['s']
        csv_rows.append({
            'step': 1, 'rp_base': row['rp'], 'rp_red': row['rp'],
            'dd1': 0, 'dd2': 0, 'consec_limit': None,
            'p_succ': s['p_succ'], 'p_ftot': s['p_ftot'], 'p_fdly': s['p_fdly'],
            'med_cd': s['med_cd'], 'dd_p50': s['dd_p50'], 'dd_p95': s['dd_p95'],
            'avg_nc': s['avg_nc'],
        })
    for row in grid:
        s = row['s']
        csv_rows.append({
            'step': 2, 'rp_base': row['rpb'], 'rp_red': row['rpr'],
            'dd1': row['d1'], 'dd2': row['d2'], 'consec_limit': None,
            'p_succ': s['p_succ'], 'p_ftot': s['p_ftot'], 'p_fdly': s['p_fdly'],
            'med_cd': s['med_cd'], 'dd_p50': s['dd_p50'], 'dd_p95': s['dd_p95'],
            'avg_nc': s['avg_nc'],
        })
    for row in step3:
        s = row['s']
        csv_rows.append({
            'step': 3, 'rp_base': opt_rpb, 'rp_red': opt_rpr,
            'dd1': opt_d1, 'dd2': opt_d2, 'consec_limit': row['cl'],
            'p_succ': s['p_succ'], 'p_ftot': s['p_ftot'], 'p_fdly': s['p_fdly'],
            'med_cd': s['med_cd'], 'dd_p50': s['dd_p50'], 'dd_p95': s['dd_p95'],
            'avg_nc': s['avg_nc'],
        })
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(RESULTS, 'p24b_monte_carlo_pct_actual.csv'), index=False)
    print('\n  Saved: p24b_monte_carlo_pct_actual.csv')
    print('=' * W)


if __name__ == '__main__':
    main()
