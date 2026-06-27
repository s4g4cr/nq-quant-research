#!/usr/bin/env python3
"""OOS pooled trade statistics for Phase 15 B1 configuration."""

import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(ROOT, "results", "p15_partC_B1_oos_pooled.csv")

PV   = 20.0
CAP  = 100_000.0
W    = 68


def _pf(v):
    a = np.array(v); w = a[a > 0]; l = a[a <= 0]
    return float(w.sum()) / float(abs(l.sum())) if l.size and abs(l.sum()) > 0 else float("inf")


def main():
    df = pd.read_csv(CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"],  utc=True)
    df = df.sort_values("entry_ts").reset_index(drop=True)

    pnl_net = df["pnl_net"].values          # USD
    pnl_pts = df["pnl_pts"].values          # points
    n       = len(df)

    wins   = pnl_net[pnl_net > 0]
    losses = pnl_net[pnl_net <= 0]
    n_wins = len(wins); n_loss = len(losses)

    print("=" * W)
    print("  PHASE 15 B1 — OOS POOLED TRADE STATISTICS")
    print(f"  Source: p15_partC_B1_oos_pooled.csv  ({n} trades)")
    print("=" * W)

    # ── 1. Basic stats ─────────────────────────────────────────────────────────
    print("\n  1. BASIC STATS")
    print(f"  {'='*(W-2)}")

    wr   = n_wins / n
    aw_p = float(pnl_pts[pnl_net > 0].mean())  if n_wins else 0.0
    al_p = float(pnl_pts[pnl_net <= 0].mean()) if n_loss else 0.0
    aw_d = float(wins.mean())                   if n_wins else 0.0
    al_d = float(losses.mean())                 if n_loss else 0.0
    rr   = aw_d / abs(al_d)                     if al_d != 0 else float("inf")
    exp_p = wr * aw_p + (1 - wr) * al_p
    exp_d = wr * aw_d + (1 - wr) * al_d

    print(f"  Total trades   : {n}")
    print(f"  Win rate       : {wr*100:.1f}%  ({n_wins}W / {n_loss}L)")
    print(f"  Avg winner     : {aw_p:.2f} pts  (${aw_d:.0f})")
    print(f"  Avg loser      : {al_p:.2f} pts  (${al_d:.0f})")
    print(f"  R/R realized   : {rr:.2f}  ({aw_p:.2f} / {abs(al_p):.2f} pts)")
    print(f"  Expectancy     : {exp_p:.3f} pts/trade  (${exp_d:.2f}/trade)")
    print(f"  Profit factor  : {_pf(pnl_net):.3f}")

    # Consecutive runs
    max_cl = max_cw = cur_l = cur_w = 0
    for x in pnl_net:
        if x > 0:
            cur_w += 1; cur_l = 0; max_cw = max(max_cw, cur_w)
        else:
            cur_l += 1; cur_w = 0; max_cl = max(max_cl, cur_l)
    print(f"  Max consec losses : {max_cl}")
    print(f"  Max consec wins   : {max_cw}")

    # Trades per month
    df["ym"] = df["entry_ts"].dt.to_period("M")
    tpm = df.groupby("ym").size()
    print(f"  Trades/month   : min={tpm.min()}  avg={tpm.mean():.1f}  max={tpm.max()}")

    # Exit breakdown
    ex  = df["exit_reason"].value_counts()
    tot = max(len(df), 1)
    print(f"  SL={ex.get('sl',0)/tot*100:.0f}%  "
          f"TP={ex.get('tp',0)/tot*100:.0f}%  "
          f"Time={ex.get('timeout',0)/tot*100:.0f}%  "
          f"EOD={ex.get('eod',0)/tot*100:.0f}%")

    # ── 2. PnL distribution ────────────────────────────────────────────────────
    print(f"\n  2. PnL DISTRIBUTION (USD per trade)")
    print(f"  {'='*(W-2)}")
    ps = np.percentile(pnl_net, [5, 25, 50, 75, 95])
    print(f"  p5={ps[0]:.0f}  p25={ps[1]:.0f}  p50={ps[2]:.0f}  "
          f"p75={ps[3]:.0f}  p95={ps[4]:.0f}")
    worst = float(pnl_net.min()); best = float(pnl_net.max())
    print(f"  Worst trade : ${worst:.0f}")
    print(f"  Best trade  : ${best:.0f}")

    # ── 3. Daily P&L ──────────────────────────────────────────────────────────
    print(f"\n  3. DAILY P&L")
    print(f"  {'='*(W-2)}")
    df["date"] = df["entry_ts"].dt.date
    daily = df.groupby("date")["pnl_net"].sum()
    n_days = len(daily)
    avg_t_per_day = n / n_days
    dps = np.percentile(daily.values, [5, 25, 50, 75, 95])
    print(f"  Trading days (with trades): {n_days}")
    print(f"  Avg trades per trading day : {avg_t_per_day:.2f}")
    print(f"  Daily PnL:  p5=${dps[0]:.0f}  p25=${dps[1]:.0f}  p50=${dps[2]:.0f}  "
          f"p75=${dps[3]:.0f}  p95=${dps[4]:.0f}")
    print(f"  Worst day   : ${daily.min():.0f}  ({daily.idxmin()})")
    print(f"  Best day    : ${daily.max():.0f}  ({daily.idxmax()})")
    dv = daily.values
    for thresh in [1000, 2000, 3000, 5000]:
        pct = (dv < -thresh).mean() * 100
        print(f"  Days > ${thresh:,} loss: {pct:.1f}%  ({int((dv < -thresh).sum())}/{n_days})")

    # ── 4. Drawdown ────────────────────────────────────────────────────────────
    print(f"\n  4. DRAWDOWN (equity curve, trade-level)")
    print(f"  {'='*(W-2)}")
    curve  = np.concatenate([[0.0], np.cumsum(pnl_net)])
    peak   = np.maximum.accumulate(curve)
    dd     = peak - curve            # drawdown in USD at each trade
    max_dd = float(dd.max())
    max_dd_pct = max_dd / CAP * 100

    # Drawdown periods: sequences of dd > 0
    in_dd     = dd > 0
    dd_lens   = []
    cur_len   = 0
    for x in in_dd:
        if x:
            cur_len += 1
        elif cur_len > 0:
            dd_lens.append(cur_len); cur_len = 0
    if cur_len > 0:
        dd_lens.append(cur_len)

    avg_dd_dur = float(np.mean(dd_lens)) if dd_lens else 0.0
    max_dd_dur = max(dd_lens) if dd_lens else 0
    print(f"  Max drawdown  : ${max_dd:.0f}  ({max_dd_pct:.2f}% of ${CAP:,.0f})")
    print(f"  Avg DD duration : {avg_dd_dur:.1f} trades")
    print(f"  Max DD duration : {max_dd_dur} trades")
    print(f"  # DD periods    : {len(dd_lens)}")

    print(f"\n  {'='*W}")


if __name__ == "__main__":
    main()
