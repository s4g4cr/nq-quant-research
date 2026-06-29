#!/usr/bin/env python3
"""
Phase 24 -- Monte Carlo FTMO Sizing: POC B1 + HMM Ranging Filter (A1).

FTMO $100k: Target $110k (+10%) | Floor $90k (-10%) | Daily limit -$5k
Bootstrap: resample by calendar day (days with 2 trades stay together).
10,000 simulations per configuration.
"""

import itertools
import math
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')

INITIAL = 100_000.0
TARGET  = 110_000.0
FLOOR   =  90_000.0
DAILY_L =  -5_000.0
N_SIMS  =  10_000
W       = 78


# ── Data ──────────────────────────────────────────────────────────────────────
def load_buckets(path):
    df = pd.read_csv(path)
    df['entry_ts']   = pd.to_datetime(df['entry_ts'], utc=True)
    df['entry_date'] = df['entry_ts'].dt.date
    return [list(g['pnl_net'].values) for _, g in df.groupby('entry_date')]


# ── get_n factories ───────────────────────────────────────────────────────────
def fixed(n):
    return lambda cap, peak, cl: n

def dynamic(n_base, n_red, dd1, dd2):
    def fn(cap, peak, cl):
        dd = (peak - cap) / INITIAL
        if   dd >= dd2: return 1
        elif dd >= dd1: return n_red
        else:           return n_base
    return fn

def dyn_consec(n_base, n_red, dd1, dd2, cl_limit):
    base = dynamic(n_base, n_red, dd1, dd2)
    def fn(cap, peak, cl):
        if cl_limit is not None and cl >= cl_limit:
            return 1
        return base(cap, peak, cl)
    return fn


# ── Core simulation ───────────────────────────────────────────────────────────
def simulate(buckets, n_sims, get_n, seed=42, track_paths=0):
    rng    = np.random.default_rng(seed)
    n_pool = len(buckets)
    outcomes, tdays, maxdd, capend = [], [], [], []
    cap_ms = {30: [], 60: [], 90: []}
    paths  = []

    for si in range(n_sims):
        cap    = INITIAL; peak = INITIAL
        consec = 0; td = 0; mdd = 0.0
        ms_cap = {30: INITIAL, 60: INITIAL, 90: INITIAL}
        path   = [INITIAL] if si < track_paths else None
        out    = None

        while out is None:
            n   = get_n(cap, peak, consec)
            idx = int(rng.integers(0, n_pool))
            day = buckets[idx]
            dpnl = sum(p * n for p in day)

            if dpnl < DAILY_L:
                out = 'fail_daily'; cap += dpnl; break

            cap  += dpnl
            peak  = max(peak, cap)
            td   += 1

            for p in day:
                if p * n <= 0: consec += 1
                else:          consec  = 0

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

    td = np.array(tdays); dd = np.array(maxdd); ce = np.array(capend)
    sm = np.array([o == 'success' for o in outcomes])
    return {'outcomes': outcomes, 'tdays': td, 'maxdd': dd, 'capend': ce,
            'succ_mask': sm, 'cap_ms': {m: np.array(v) for m, v in cap_ms.items()},
            'paths': paths}


def S(r, label=''):
    o = r['outcomes']; td = r['tdays']; dd = r['maxdd']
    ce = r['capend'];  sm = r['succ_mask']; n = len(o)
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
        'ce_p5':  pct(ce, 5), 'ce_p25': pct(ce, 25), 'ce_p50': pct(ce, 50),
        'ce_p75': pct(ce, 75), 'ce_p95': pct(ce, 95),
        'cap_ms': {m: float(np.mean(r['cap_ms'][m])) for m in (30, 60, 90)},
        'td_s': td_s, 'dd_s': dd_s, 'sm': sm, 'ce': ce,
    }


def _f(v, d=1):
    return '  nan' if v != v else f'{v:.{d}f}'


# ── Stress buckets ────────────────────────────────────────────────────────────
def stress_wr(buckets, factor=0.80, seed=1234):
    """Convert 20% of winning trades to avg_loser."""
    rng_s = np.random.default_rng(seed)
    all_l = [p for b in buckets for p in b if p <= 0]
    avg_l = float(np.mean(all_l)) if all_l else 0.0
    return [[p if p <= 0 or rng_s.random() > (1 - factor) else avg_l
             for p in b] for b in buckets]

def stress_win(buckets, factor=0.80):
    return [[p * factor if p > 0 else p for p in b] for b in buckets]


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_fixed_success(step1_rows, fpath):
    nc  = [r['nc'] for r in step1_rows]
    ps  = [r['s']['p_succ'] for r in step1_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(nc, ps, color='#2563eb', edgecolor='white', width=0.6)
    for b, v in zip(bars, ps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(87.3, color='#dc2626', linestyle='--', linewidth=1.2,
               label='Phase 16 B1 1c (87.3%)')
    ax.set_xlabel('Contracts', fontsize=12)
    ax.set_ylabel('P(FTMO Pass) %', fontsize=12)
    ax.set_title('Fixed Contract Sizing — P(FTMO Pass)\nPOC B1 + HMM Ranging (A1)', fontsize=13)
    ax.set_ylim(0, 110); ax.set_xticks(nc); ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()

def chart_equity_curves(buckets, get_n, fpath, n_paths=200, seed=99):
    r = simulate(buckets, n_paths, get_n, seed=seed, track_paths=n_paths)
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

def chart_dd_dist(dd_s, fpath):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dd_s, bins=40, color='#2563eb', edgecolor='white', alpha=0.8)
    for p, lbl, c in [(25,'p25','#aaa'),(50,'p50','#666'),(75,'p75','#333'),(95,'p95','#dc2626')]:
        v = np.percentile(dd_s, p)
        ax.axvline(v, color=c, linewidth=1.5, linestyle='--', label=f'{lbl}={v:.1f}%')
    ax.set_xlabel('Max Drawdown % (Successful Paths)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Max Drawdown Distribution — Successful Paths\nOptimal Config', fontsize=13)
    ax.legend(fontsize=10)
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
    ax.set_title('Days to Pass FTMO — Successful Paths\nOptimal Config', fontsize=13)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * W)
    print('  PHASE 24 -- MONTE CARLO FTMO SIZING (POC B1 + HMM RANGING A1)')
    print('=' * W)
    print(f'  FTMO ${INITIAL:,.0f}: target=${TARGET:,.0f} | '
          f'floor=${FLOOR:,.0f} | daily_limit=${DAILY_L:,.0f}')
    print(f'  {N_SIMS:,} sims per config | resample by calendar day')

    # ── Load data ─────────────────────────────────────────────────────────────
    a1_path = os.path.join(RESULTS, 'p23_wfo_oos_pooled.csv')
    bk_a1   = load_buckets(a1_path)
    n_t     = sum(len(b) for b in bk_a1)
    print(f'\n  A1 pool: {n_t} trades  {len(bk_a1)} active trading days')

    b1_path = os.path.join(RESULTS, 'p15_partC_B1_oos_pooled.csv')
    bk_b1   = load_buckets(b1_path) if os.path.exists(b1_path) else None
    if bk_b1:
        print(f'  B1 pool: {sum(len(b) for b in bk_b1)} trades  '
              f'{len(bk_b1)} active trading days')
    else:
        print('  B1 pool not found -- Phase 16 comparison will use hardcoded values')

    # Pre-compute stressed buckets (used in Steps 4, 6, 7)
    bk_swr  = stress_wr(bk_a1)
    bk_swin = stress_win(bk_a1)
    bk_both = stress_win(stress_wr(bk_a1))

    # =====================================================================
    # STEP 1 — Fixed contracts
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 1 -- FIXED CONTRACT SIZING')
    print(f'  {"="*W}')
    print(f'\n  {"Nc":>3} | {"P(succ)":>8} | {"P(ftot)":>8} | {"P(fdly)":>8} | '
          f'{"MedTrd":>7} | {"MedCal":>7} | {"DD p50":>7} | {"DD p95":>7}')
    print('  ' + '-' * 72)

    step1_rows = []
    for nc in [1, 2, 3, 4, 5]:
        r = simulate(bk_a1, N_SIMS, fixed(nc), seed=42)
        s = S(r, f'fixed_{nc}c')
        step1_rows.append({'nc': nc, 's': s, 'r': r})
        print(f'  {nc:>3} | {s["p_succ"]:>7.1f}% | {s["p_ftot"]:>7.1f}% | '
              f'{s["p_fdly"]:>7.1f}% | {s["med_td"]:>6.0f}td | {s["med_cd"]:>6.0f}d | '
              f'{s["dd_p50"]:>6.1f}% | {s["dd_p95"]:>6.1f}%')

    # Capital milestones + distribution for 1c and 2c
    for nc in [1, 2]:
        s = [r for r in step1_rows if r['nc'] == nc][0]['s']
        print(f'\n  Fixed {nc}c -- avg capital at trading day milestone:')
        for m in [30, 60, 90]:
            print(f'    Day {m:>3}: ${s["cap_ms"][m]:>9,.0f}')
        print(f'  Fixed {nc}c -- final capital distribution:')
        for lbl, key in [('p5','ce_p5'),('p25','ce_p25'),('p50','ce_p50'),
                          ('p75','ce_p75'),('p95','ce_p95')]:
            print(f'    {lbl}: ${s[key]:>9,.0f}')

    # =====================================================================
    # STEP 2 — Dynamic sizing grid
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 2 -- DYNAMIC SIZING GRID')
    print(f'  {"="*W}')
    print('  GREEN(DD<dd1)=n_base | YELLOW(dd1<=DD<dd2)=n_red | RED(DD>=dd2)=1\n')

    grid = []
    valid_cfgs = [
        (nb, nr, d1, d2)
        for nb, nr, d1, d2 in itertools.product(
            [1, 2, 3], [1, 2],
            [0.02, 0.03, 0.04, 0.05],
            [0.05, 0.06, 0.07, 0.08]
        )
        if nr < nb and d2 > d1
    ]
    total_cfgs = len(valid_cfgs)

    for ci, (nb, nr, d1, d2) in enumerate(valid_cfgs, 1):
        r = simulate(bk_a1, N_SIMS, dynamic(nb, nr, d1, d2), seed=42)
        s = S(r, f'dyn_{nb}_{nr}_{int(d1*100)}_{int(d2*100)}')
        grid.append({'nb': nb, 'nr': nr, 'd1': d1, 'd2': d2, 's': s})
        print(f'  [{ci:>2}/{total_cfgs}] nb={nb} nr={nr} '
              f'dd1={d1*100:.0f}% dd2={d2*100:.0f}%  '
              f'P(succ)={s["p_succ"]:.1f}%  '
              f'DD_p95={s["dd_p95"]:.1f}%')

    grid.sort(key=lambda x: -x['s']['p_succ'])
    print(f'\n  TOP 10 BY P(SUCCESS):')
    print(f'  {"nb":>5} | {"nr":>4} | {"dd1":>5} | {"dd2":>5} | '
          f'{"P(succ)":>8} | {"P(fail)":>8} | {"DDp95":>6} | {"MedDays":>8}')
    print('  ' + '-' * 66)
    for row in grid[:10]:
        s = row['s']
        print(f'  {row["nb"]:>5} | {row["nr"]:>4} | {row["d1"]*100:>4.0f}% | '
              f'{row["d2"]*100:>4.0f}% | {s["p_succ"]:>7.1f}% | '
              f'{s["p_ftot"]+s["p_fdly"]:>7.1f}% | '
              f'{s["dd_p95"]:>5.1f}% | {s["med_cd"]:>7.0f}d')

    best = grid[0]
    opt_nb, opt_nr, opt_d1, opt_d2 = best['nb'], best['nr'], best['d1'], best['d2']
    print(f'\n  Best: nb={opt_nb} nr={opt_nr} '
          f'dd1={opt_d1*100:.0f}% dd2={opt_d2*100:.0f}%  '
          f'P(succ)={best["s"]["p_succ"]:.1f}%')

    # =====================================================================
    # STEP 3 — Consecutive loss protection
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 3 -- CONSECUTIVE LOSS PROTECTION')
    print(f'  {"="*W}')
    print(f'  Base: nb={opt_nb} nr={opt_nr} '
          f'dd1={opt_d1*100:.0f}% dd2={opt_d2*100:.0f}%\n')
    print(f'  {"Limit":>8} | {"P(succ)":>8} | {"P(fail)":>8} | '
          f'{"DDp95":>6} | {"MedDays":>8} | {"ActRate%":>9}')
    print('  ' + '-' * 60)

    step3 = []
    best_cl_ps = -1; opt_cl = None

    for cl in [3, 5, 8, 10, 12, None]:
        # Track activation rate via mutable counter
        act_count = [0]; total_td = [0]

        def make_fn(nb, nr, d1, d2, cl_lim, ac, ttd):
            base_fn = dynamic(nb, nr, d1, d2)
            def fn(cap, peak, clv):
                ttd[0] += 1  # count every day call
                if cl_lim is not None and clv >= cl_lim:
                    ac[0] += 1
                    return 1
                return base_fn(cap, peak, clv)
            return fn

        gn = make_fn(opt_nb, opt_nr, opt_d1, opt_d2, cl, act_count, total_td)
        r  = simulate(bk_a1, N_SIMS, gn, seed=42)
        s  = S(r, f'consec_{cl}')
        act_rate = act_count[0] / max(total_td[0], 1) * 100
        p_fail = s['p_ftot'] + s['p_fdly']
        cl_str = str(cl) if cl is not None else 'none'
        print(f'  {cl_str:>8} | {s["p_succ"]:>7.1f}% | {p_fail:>7.1f}% | '
              f'{s["dd_p95"]:>5.1f}% | {s["med_cd"]:>7.0f}d | {act_rate:>8.1f}%')
        step3.append({'cl': cl, 's': s, 'act_rate': act_rate})
        if s['p_succ'] > best_cl_ps:
            best_cl_ps = s['p_succ']
            opt_cl = cl

    print(f'\n  Best consec limit: {opt_cl}  '
          f'P(succ)={best_cl_ps:.1f}%')

    # Build optimal config + run once for all subsequent steps
    get_n_opt = dyn_consec(opt_nb, opt_nr, opt_d1, opt_d2, opt_cl)
    r_opt = simulate(bk_a1, N_SIMS, get_n_opt, seed=42, track_paths=200)
    s_opt = S(r_opt, 'optimal')
    print(f'  Optimal: nb={opt_nb} nr={opt_nr} '
          f'dd1={opt_d1*100:.0f}% dd2={opt_d2*100:.0f}% cl={opt_cl}  '
          f'P(succ)={s_opt["p_succ"]:.1f}%  '
          f'DD p95={s_opt["dd_p95"]:.1f}%  Med={s_opt["med_cd"]:.0f}d')

    # =====================================================================
    # STEP 4 — Sensitivity analysis
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 4 -- SENSITIVITY ANALYSIS')
    print(f'  {"="*W}')
    print(f'\n  {"Config":<22} | {"P(succ)":>8} | {"vs base":>8} | '
          f'{"DDp95":>6} | {"MedDays":>8}')
    print('  ' + '-' * 62)

    s_base_ps  = s_opt['p_succ']
    stress_runs = {}
    for name, bk, gn in [
        ('Baseline A1',     bk_a1,   get_n_opt),
        ('WR -20% (19%)',   bk_swr,  get_n_opt),
        ('Winner -20%',     bk_swin, get_n_opt),
        ('Both -20%',       bk_both, get_n_opt),
    ]:
        r = simulate(bk, N_SIMS, gn, seed=42)
        s = S(r)
        stress_runs[name] = s
        delta = s['p_succ'] - s_base_ps
        sign  = '+' if delta >= 0 else ''
        print(f'  {name:<22} | {s["p_succ"]:>7.1f}% | '
              f'{sign}{delta:>+6.1f}pp | {s["dd_p95"]:>5.1f}% | {s["med_cd"]:>7.0f}d')

    # Phase 16 B1 1c vs Phase 24 A1 optimal
    print(f'\n  --- vs Phase 16 comparison ---')
    if bk_b1:
        r_b1 = simulate(bk_b1, N_SIMS, fixed(1), seed=42)
        s_b1 = S(r_b1)
        print(f'  {"Phase16 B1 1c":<22} | {s_b1["p_succ"]:>7.1f}%  '
              f'DD p95={s_b1["dd_p95"]:.1f}%  Med={s_b1["med_cd"]:.0f}d')
    else:
        print('  Phase16 B1 1c: P(succ)=87.3%  (Phase 16 hardcoded result)')
    print(f'  {"Phase24 A1 opt":<22} | {s_opt["p_succ"]:>7.1f}%  '
          f'DD p95={s_opt["dd_p95"]:.1f}%  Med={s_opt["med_cd"]:.0f}d')

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
    green_fl  = INITIAL * (1 - opt_d1)
    yellow_fl = INITIAL * (1 - opt_d2)
    cl_str    = str(opt_cl) if opt_cl is not None else 'no limit'
    s_swr  = stress_runs['WR -20% (19%)']
    s_swin = stress_runs['Winner -20%']
    s_both = stress_runs['Both -20%']
    print(f"""
  ==================================================================
  RECOMMENDED SIZING -- POC B1 + HMM RANGING FILTER -- FTMO $100k
  ==================================================================
  Zone GREEN  (DD < {opt_d1*100:.0f}%, acct > ${green_fl:,.0f}):  {opt_nb} contract(s)
  Zone YELLOW ({opt_d1*100:.0f}% <= DD < {opt_d2*100:.0f}%, ${yellow_fl:,.0f}-${green_fl:,.0f}):  {opt_nr} contract(s)
  Zone RED    (DD >= {opt_d2*100:.0f}%, acct < ${yellow_fl:,.0f}):  1 contract
  Consec loss rule: reduce to 1c after {cl_str} consecutive losses
                    reset to zone sizing on first winning trade
  ------------------------------------------------------------------
  P(pass challenge):       {s_opt["p_succ"]:>5.1f}%
  Median days to pass:     {s_opt["med_cd"]:>4.0f} calendar days
  P(pass within 90 days):  {(cd_s <= 90).mean()*100:>5.1f}%
  P(pass within 180 days): {(cd_s <= 180).mean()*100:>5.1f}%
  DD p50 on success paths: {s_opt["dd_p50"]:>5.1f}%
  DD p95 on success paths: {s_opt["dd_p95"]:>5.1f}%
  Stress WR -20%:          {s_swr["p_succ"]:>5.1f}%
  Stress winner -20%:      {s_swin["p_succ"]:>5.1f}%
  Stress both -20%:        {s_both["p_succ"]:>5.1f}%
  ==================================================================

  OPERATIONAL RULES:
  1. Start each session with {opt_nb} contract(s).
  2. If account drops below ${green_fl:,.0f} (DD={opt_d1*100:.0f}%),
     reduce to {opt_nr} contract(s) for the next entry.
  3. If account drops below ${yellow_fl:,.0f} (DD={opt_d2*100:.0f}%),
     reduce to 1 contract.
  4. After {cl_str} consecutive losing trades, trade 1 contract.
     Resume zone sizing after the FIRST winning trade.
  5. Zone is evaluated at session open, not intra-trade.
  6. A "day" = any session with at least one B1 + HMM signal.""")

    # =====================================================================
    # STEP 7 — Comparison table
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  STEP 7 -- PHASE 16 B1 vs PHASE 24 A1 COMPARISON')
    print(f'  {"="*W}')
    opt_sizing = (f'nb={opt_nb}/nr={opt_nr}/'
                  f'dd1={opt_d1*100:.0f}%/dd2={opt_d2*100:.0f}%/cl={cl_str}')
    p16_pass  = s_b1['p_succ'] if bk_b1 else 87.3
    p16_med   = s_b1['med_cd'] if bk_b1 else 270.0
    p16_ddp95 = s_b1['dd_p95'] if bk_b1 else float('nan')

    rows_cmp = [
        ('Win rate',           '21.4%',     '23.7%'),
        ('R/R realized',       '4.58:1',    '4.52:1'),
        ('Expectancy/trade',   '$60.58',    '$93.00'),
        ('Trades/month',       '10.9',      '7.2'),
        ('Max consec losses',  '25',        '16'),
        ('Historical MaxDD',   '10.05%',    '6.38%'),
        ('Optimal sizing',     '1c fixed',  opt_sizing),
        ('P(FTMO pass)',        f'{p16_pass:.1f}%',
                                f'{s_opt["p_succ"]:.1f}%'),
        ('Median days to pass',f'{p16_med:.0f}d',
                                f'{s_opt["med_cd"]:.0f}d'),
        ('DD p95 success',     _f(p16_ddp95)+'%',
                                f'{s_opt["dd_p95"]:.1f}%'),
        ('Stress WR -20%',     '34.8%',     f'{s_swr["p_succ"]:.1f}%'),
        ('Stress winner -20%', '47.9%',     f'{s_swin["p_succ"]:.1f}%'),
        ('Stress both -20%',   'N/A',       f'{s_both["p_succ"]:.1f}%'),
    ]

    print(f'\n  {"Metric":<24} | {"Phase 16 B1":<20} | {"Phase 24 A1":<20}')
    print('  ' + '-' * 70)
    for m, v16, v24 in rows_cmp:
        print(f'  {m:<24} | {v16:<20} | {v24:<20}')

    # =====================================================================
    # CHARTS
    # =====================================================================
    print(f'\n  {"="*W}')
    print('  SAVING CHARTS')
    print(f'  {"="*W}')

    chart_fixed_success(step1_rows,
        os.path.join(RESULTS, 'p24_fixed_sizing_success.png'))
    print('  Saved: p24_fixed_sizing_success.png')

    chart_equity_curves(bk_a1, get_n_opt,
        os.path.join(RESULTS, 'p24_equity_curves.png'))
    print('  Saved: p24_equity_curves.png')

    chart_dd_dist(s_opt['dd_s'],
        os.path.join(RESULTS, 'p24_drawdown_distribution.png'))
    print('  Saved: p24_drawdown_distribution.png')

    chart_days_to_pass(s_opt['td_s'],
        os.path.join(RESULTS, 'p24_days_to_completion.png'))
    print('  Saved: p24_days_to_completion.png')

    # =====================================================================
    # SAVE CSV
    # =====================================================================
    csv_rows = []
    for row in step1_rows:
        s = row['s']
        csv_rows.append({'step': 1, 'n_base': row['nc'], 'n_red': row['nc'],
                         'dd1': 0, 'dd2': 0, 'consec_limit': None,
                         'p_succ': s['p_succ'], 'p_ftot': s['p_ftot'],
                         'p_fdly': s['p_fdly'], 'med_cd': s['med_cd'],
                         'dd_p50': s['dd_p50'], 'dd_p95': s['dd_p95']})
    for row in grid:
        s = row['s']
        csv_rows.append({'step': 2, 'n_base': row['nb'], 'n_red': row['nr'],
                         'dd1': row['d1'], 'dd2': row['d2'], 'consec_limit': None,
                         'p_succ': s['p_succ'], 'p_ftot': s['p_ftot'],
                         'p_fdly': s['p_fdly'], 'med_cd': s['med_cd'],
                         'dd_p50': s['dd_p50'], 'dd_p95': s['dd_p95']})
    for row in step3:
        s = row['s']
        csv_rows.append({'step': 3, 'n_base': opt_nb, 'n_red': opt_nr,
                         'dd1': opt_d1, 'dd2': opt_d2, 'consec_limit': row['cl'],
                         'p_succ': s['p_succ'], 'p_ftot': s['p_ftot'],
                         'p_fdly': s['p_fdly'], 'med_cd': s['med_cd'],
                         'dd_p50': s['dd_p50'], 'dd_p95': s['dd_p95']})
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(RESULTS, 'p24_monte_carlo_results.csv'), index=False)
    print('\n  Saved: p24_monte_carlo_results.csv')
    print('=' * W)


if __name__ == '__main__':
    main()
