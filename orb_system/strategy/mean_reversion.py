"""
Mean Reversion ORB strategy for NQ futures (1-min bars).

Hypothesis: OR extremes act as rejection levels. Price that touches but
does not close beyond an OR extreme during the OR period tends to revert
toward the OR midpoint once the OR is complete.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TICK      = 0.25
PV        = 20.0
COMM      = 4.0      # round-trip
SLIP      = 0.25     # 1-tick adverse

_OR_START = pd.Timestamp("2000-01-01 09:30").time()
_OR_END   = pd.Timestamp("2000-01-01 10:30").time()
_EOD      = pd.Timestamp("2000-01-01 15:45").time()
_NO_ENTRY = pd.Timestamp("2000-01-01 14:00").time()


# ---------------------------------------------------------------------------
# OR analytics
# ---------------------------------------------------------------------------

def compute_poc(or_bars: pd.DataFrame) -> float:
    """Volume-weighted POC via tick-level profile."""
    lo_f = math.floor(float(or_bars["low"].min()) / TICK) * TICK
    hi_c = math.ceil(float(or_bars["high"].max()) / TICK) * TICK
    n    = round((hi_c - lo_f) / TICK) + 1
    levels     = np.linspace(lo_f, hi_c, n)
    vol_profile = np.zeros(n)

    for _, bar in or_bars.iterrows():
        lo_i = max(0, round((float(bar["low"])  - lo_f) / TICK))
        hi_i = min(n - 1, round((float(bar["high"]) - lo_f) / TICK))
        nb   = max(1, hi_i - lo_i + 1)
        vol_profile[lo_i : hi_i + 1] += float(bar["volume"]) / nb

    return float(levels[int(np.argmax(vol_profile))])


def detect_rejection(or_bars: pd.DataFrame) -> dict:
    """
    Causal bar-by-bar rejection detection over the OR period.

    LONG rejection: a bar after bar-0 whose low reaches/extends the
    running OR low but closes above it; invalidated if a later bar
    closes at or below the rejection level.

    Returns {
        'long':  {'valid': bool, 'price': float, 'pen_ticks': int},
        'short': {'valid': bool, 'price': float, 'pen_ticks': int},
    }
    """
    if len(or_bars) < 2:
        return {
            "long":  {"valid": False, "price": np.nan, "pen_ticks": 0},
            "short": {"valid": False, "price": np.nan, "pen_ticks": 0},
        }

    or_lo = float(or_bars.iloc[0]["low"])
    or_hi = float(or_bars.iloc[0]["high"])

    lv, lp, lpen = False, np.nan, 0
    sv, sp, spen = False, np.nan, 0

    for i in range(1, len(or_bars)):
        bar    = or_bars.iloc[i]
        b_lo   = float(bar["low"])
        b_hi   = float(bar["high"])
        b_cl   = float(bar["close"])
        prev_lo = or_lo
        prev_hi = or_hi
        or_lo  = min(or_lo, b_lo)
        or_hi  = max(or_hi, b_hi)

        # LONG — invalidate then detect
        if lv and b_cl <= lp:
            lv, lp, lpen = False, np.nan, 0
        if not lv and b_lo <= prev_lo:
            if b_cl > or_lo:
                lv  = True
                lp  = or_lo
                lpen = max(0, round((prev_lo - b_lo) / TICK))

        # SHORT — invalidate then detect
        if sv and b_cl >= sp:
            sv, sp, spen = False, np.nan, 0
        if not sv and b_hi >= prev_hi:
            if b_cl < or_hi:
                sv  = True
                sp  = or_hi
                spen = max(0, round((b_hi - prev_hi) / TICK))

    return {
        "long":  {"valid": lv, "price": lp, "pen_ticks": lpen},
        "short": {"valid": sv, "price": sp, "pen_ticks": spen},
    }


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class MRSetupInfo:
    date: object
    direction: str
    or_high: float
    or_low: float
    or_mid: float
    or_poc: float
    or_range: float
    rejection_price: float
    pen_ticks: int
    entry_trigger: float
    atr_or_end: float


@dataclass
class MRTrade:
    trade_id: int
    date: object
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: str
    entry_px: float
    exit_px: float
    sl_px: float
    tp_px: float
    or_mid: float
    exit_reason: str
    pnl_pts: float
    pnl_usd: float
    pnl_net: float
    rr: float
    year: int
    month: int
    dow: int
    or_range: float
    atr_entry: float


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

class MRResults:
    def __init__(self, trades: list, setups: list,
                 total_or_sessions: int, initial_capital: float = 100_000.0):
        self.trades = trades
        self.setups = setups
        self.total_or_sessions = total_or_sessions
        self.initial_capital   = initial_capital

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        if not self.trades:
            return {"n_trades": 0}
        pnls    = [t.pnl_net for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p <= 0]
        gw = sum(winners)
        gl = sum(abs(p) for p in losers)
        pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 0.0)

        eq = [self.initial_capital]
        for p in pnls:
            eq.append(eq[-1] + p)
        peak, mdd = eq[0], 0.0
        for e in eq:
            peak = max(peak, e)
            mdd  = max(mdd, peak - e)

        sr   = np.mean(pnls) / np.std(pnls) if len(pnls) > 1 else 0.0
        ret  = sum(pnls) / self.initial_capital * 100
        rrs  = [t.rr for t in self.trades]
        n    = len(self.trades)

        exits = {"sl": 0, "tp": 0, "eod": 0, "time": 0}
        for t in self.trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

        yearly: dict = {}
        for t in self.trades:
            yearly.setdefault(t.year, []).append(t)
        yearly_rows = []
        for y in sorted(yearly):
            yt = yearly[y]
            yp = [t.pnl_net for t in yt]
            yw = [p for p in yp if p > 0]
            yl = [p for p in yp if p <= 0]
            ygw, ygl = sum(yw), sum(abs(p) for p in yl)
            yearly_rows.append({
                "year":   y,
                "n":      len(yt),
                "win_pct": len(yw) / len(yt) * 100,
                "pf":     ygw / ygl if ygl > 0 else (999.0 if ygw > 0 else 0.0),
                "ret":    sum(yp) / self.initial_capital * 100,
                "n_long":  sum(1 for t in yt if t.direction == "long"),
                "n_short": sum(1 for t in yt if t.direction == "short"),
            })

        return {
            "n_trades": n,
            "n_long":  sum(1 for t in self.trades if t.direction == "long"),
            "n_short": sum(1 for t in self.trades if t.direction == "short"),
            "wr":      len(winners) / n * 100,
            "pf":      pf,
            "sr":      sr,
            "ret":     ret,
            "avg_win": np.mean(winners) if winners else 0.0,
            "avg_los": np.mean(losers)  if losers  else 0.0,
            "mdd_usd": mdd,
            "mdd_pct": mdd / self.initial_capital * 100,
            "exits":   exits,
            "rr_mean":   np.mean(rrs),
            "rr_med":    np.median(rrs),
            "rr_min":    min(rrs),
            "rr_max":    max(rrs),
            "rr_gt1":    sum(1 for r in rrs if r > 1.0) / n * 100,
            "yearly":    yearly_rows,
        }

    # ------------------------------------------------------------------
    def print_diagnostic(self) -> None:
        sep = "=" * 62
        print(f"\n{sep}")
        print("  DIAGNOSTICO DE SETUPS (periodo OR)")
        print(sep)

        tot = self.total_or_sessions
        long_s  = [s for s in self.setups if s.direction == "long"]
        short_s = [s for s in self.setups if s.direction == "short"]
        long_d  = {s.date for s in long_s}
        short_d = {s.date for s in short_s}
        both_d  = long_d & short_d
        any_d   = long_d | short_d
        none_n  = tot - len(any_d)

        print(f"\n  1. FRECUENCIA DE SETUPS  (sesiones OR analizadas: {tot})")
        for lbl, cnt in [("LONG", len(long_d)), ("SHORT", len(short_d)),
                          ("Ambos", len(both_d)), ("Ninguno", none_n)]:
            pct = cnt / tot * 100 if tot > 0 else 0.0
            print(f"     {lbl:<8}: {cnt:>4}  ({pct:.1f}%)")

        print(f"\n  2. CALIDAD DEL RECHAZO (penetracion en ticks desde extreme previo)")
        for lbl, slist in [("LONG", long_s), ("SHORT", short_s)]:
            if not slist:
                continue
            pens = [s.pen_ticks for s in slist]
            dist: dict = {}
            for p in pens:
                k = p if p < 5 else "5+"
                dist[k] = dist.get(k, 0) + 1
            atrs = [s.atr_or_end for s in slist if s.atr_or_end > 0]
            avg_atr = np.mean(atrs) if atrs else 0.0
            print(f"     {lbl}  (ATR medio al cierre del OR: {avg_atr:.1f} pts, "
                  f"SL=0.5xATR={avg_atr*0.5:.1f} pts)")
            for k in [0, 1, 2, 3, 4, "5+"]:
                cnt = dist.get(k, 0)
                pct = cnt / len(pens) * 100 if pens else 0.0
                bar = "#" * int(pct / 2)
                print(f"       {str(k):>3} ticks: {cnt:>4} ({pct:>5.1f}%)  {bar}")

        if self.trades:
            print(f"\n  3. DISTANCIA ENTRADA -> TP (trades ejecutados: {len(self.trades)})")
            dists = [abs(t.tp_px - t.entry_px) for t in self.trades]
            atr_e = [t.atr_entry for t in self.trades]
            print(f"     Media              : {np.mean(dists):.2f} pts")
            print(f"     p25 / p50 / p75    : {np.percentile(dists,25):.2f} / "
                  f"{np.percentile(dists,50):.2f} / {np.percentile(dists,75):.2f}")
            print(f"     0.5 x ATR medio    : {np.mean(atr_e)*0.5:.2f} pts  (= riesgo)")

            print(f"\n  4. R/R EFECTIVO (trades ejecutados)")
            rrs = [t.rr for t in self.trades]
            print(f"     Media   : {np.mean(rrs):.2f}")
            print(f"     Mediana : {np.median(rrs):.2f}")
            print(f"     Min/Max : {min(rrs):.2f} / {max(rrs):.2f}")
            print(f"     R/R > 1 : {sum(1 for r in rrs if r > 1.0) / len(rrs)*100:.1f}%")
        print(sep)

    # ------------------------------------------------------------------
    def print_report(self, label: str = "") -> None:
        m = self.metrics()
        if m["n_trades"] == 0:
            print(f"  {label}: sin trades.")
            return
        tag = label.upper() if label else "RESULTADOS"
        n   = m["n_trades"]
        print(f"\n  {tag}:")
        print(f"    Trades : {n}  (LONG: {m['n_long']} | SHORT: {m['n_short']})")
        print(f"    Win%   : {m['wr']:.1f}%  |  PF: {m['pf']:.3f}  "
              f"|  Sharpe: {m['sr']:.2f}  |  Return: {m['ret']:+.1f}%")
        e = m["exits"]
        print(f"    Exits  -> SL: {e.get('sl',0)/n*100:.0f}%  TP: {e.get('tp',0)/n*100:.0f}%  "
              f"EOD: {e.get('eod',0)/n*100:.0f}%  Time: {e.get('time',0)/n*100:.0f}%")
        print(f"    Win med: ${m['avg_win']:>8.2f}  |  Los med: ${m['avg_los']:>8.2f}  "
              f"|  R/R med: {m['rr_mean']:.2f}  |  MaxDD: ${m['mdd_usd']:,.0f} ({m['mdd_pct']:.1f}%)")
        print(f"    Desglose anual:")
        print(f"      {'Ano':>4} | {'N':>4} | {'Win%':>5} | {'PF':>5} | {'Ret%':>6} | L | S")
        print(f"      {'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+---+---")
        for r in m["yearly"]:
            print(f"      {r['year']:>4} | {r['n']:>4} | {r['win_pct']:>5.1f} | "
                  f"{r['pf']:>5.2f} | {r['ret']:>+6.1f} | {r['n_long']:>1} | {r['n_short']:>1}")

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {"trade_id": t.trade_id, "date": t.date, "entry_ts": t.entry_ts,
             "exit_ts": t.exit_ts, "direction": t.direction,
             "entry_px": t.entry_px, "exit_px": t.exit_px,
             "sl_px": t.sl_px, "tp_px": t.tp_px, "or_mid": t.or_mid,
             "exit_reason": t.exit_reason, "pnl_pts": t.pnl_pts,
             "pnl_usd": t.pnl_usd, "pnl_net": t.pnl_net, "rr": t.rr,
             "year": t.year, "month": t.month, "dow": t.dow,
             "or_range": t.or_range, "atr_entry": t.atr_entry}
            for t in self.trades
        ])


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MeanReversionEngine:

    # ------------------------------------------------------------------
    @staticmethod
    def _prev_returns(df: pd.DataFrame) -> dict:
        """date -> prev_session_return  (close_1545 - open_0930 of prev day)."""
        daily_open:  dict = {}
        daily_close: dict = {}

        df2 = df.copy()
        df2["_d"]   = df2.index.date
        df2["_min"] = df2.index.hour * 60 + df2.index.minute

        for d, grp in df2.groupby("_d"):
            ob = grp[grp["_min"] == 570]
            if not ob.empty:
                daily_open[d] = float(ob["open"].iloc[0])
            cb = grp[grp["_min"] <= 945]
            if not cb.empty:
                daily_close[d] = float(cb["close"].iloc[-1])

        dates = sorted(daily_close)
        result: dict = {}
        for i, d in enumerate(dates):
            if i == 0:
                result[d] = np.nan; continue
            pd_ = dates[i - 1]
            pc  = daily_close.get(pd_, np.nan)
            po  = daily_open.get(pd_, np.nan)
            result[d] = pc - po if not (np.isnan(pc) or np.isnan(po)) else np.nan
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _exit(post_bars: pd.DataFrame, entry_ts: pd.Timestamp,
              direction: int, entry_px: float, sl_px: float,
              tp_px: float, max_bars: int):
        """
        Iterate bars after entry_ts, return (exit_px, reason, exit_ts).
        Priority: SL > TP > EOD (15:45) > timeout.
        """
        bars_in = 0
        for ts, bar in post_bars[post_bars.index > entry_ts].iterrows():
            bars_in += 1
            h, l, c, t = float(bar["high"]), float(bar["low"]), float(bar["close"]), ts.time()

            if direction == 1:
                sl_hit = l <= sl_px
                tp_hit = h >= tp_px
            else:
                sl_hit = h >= sl_px
                tp_hit = l <= tp_px

            if sl_hit:
                return sl_px, "sl", ts
            if tp_hit:
                return tp_px, "tp", ts
            if t >= _EOD:
                ep = c - SLIP if direction == 1 else c + SLIP
                return ep, "eod", ts
            if bars_in >= max_bars:
                ep = c - SLIP if direction == 1 else c + SLIP
                return ep, "time", ts

        # Ran out of session bars
        last_ts  = post_bars[post_bars.index > entry_ts].index[-1] if len(post_bars[post_bars.index > entry_ts]) > 0 else entry_ts
        last_cl  = float(post_bars.loc[last_ts, "close"]) if last_ts != entry_ts else entry_px
        ep = last_cl - SLIP if direction == 1 else last_cl + SLIP
        return ep, "eod", last_ts

    # ------------------------------------------------------------------
    @classmethod
    def run(
        cls,
        df: pd.DataFrame,
        sl_atr_mult: float       = 0.5,
        entry_pct: float         = 0.30,
        max_bars: int            = 120,
        use_prev_session: bool   = False,
        use_or_width: bool       = False,
        or_width_max_mult: float = 1.5,
        tp_far_extreme: bool     = False,  # TP = OR far extreme instead of or_mid
        initial_capital: float   = 100_000.0,
        collect_diag: bool       = True,
    ) -> "MRResults":
        """
        tp_far_extreme=False (default): TP = or_mid (half-reversion, 30% entry)
        tp_far_extreme=True:  TP = or_high (LONG) / or_low (SHORT) — full reversion;
                              intended for entry_pct=0.50 where entry IS at or_mid
        use_or_width: filter to narrow ORs; reference is causal EWM-20 of recent
                      OR ranges (NOT 1-min ATR, which would be ~15-25x too small)
        """
        prev_ret = cls._prev_returns(df) if use_prev_session else {}

        # --- Precompute causal EWM of OR ranges (for OR-width filter) ----------
        date_arr = np.array(df.index.date)
        unique_dates, first_idx, counts = np.unique(
            date_arr, return_index=True, return_counts=True
        )
        or_ewm_ref: dict = {}   # date -> EWM OR-range reference (causal)
        if use_or_width:
            ewm_val   = np.nan
            alpha     = 1.0 / 20.0
            for ui, d in enumerate(unique_dates):
                s2 = first_idx[ui]; e2 = s2 + counts[ui]
                bt2 = df.index.time[s2:e2]
                or_m2 = np.array(
                    [(t >= _OR_START and t < _OR_END) for t in bt2]
                )
                sess_or = df.iloc[s2:e2][or_m2]
                if len(sess_or) < 2:
                    continue
                or_rng2 = float(sess_or["high"].max()) - float(sess_or["low"].min())
                if or_rng2 <= 0:
                    continue
                # Store reference BEFORE updating EWM (causal)
                or_ewm_ref[d] = ewm_val
                ewm_val = or_rng2 if np.isnan(ewm_val) else (
                    alpha * or_rng2 + (1 - alpha) * ewm_val
                )

        # --- Main loop ----------------------------------------------------------
        trades:   list = []
        setups:   list = []
        tid            = 0
        total_or_sess  = 0

        for ui, d in enumerate(unique_dates):
            s = first_idx[ui]
            e = s + counts[ui]
            session = df.iloc[s:e]

            bar_times = session.index.time

            # OR bars
            or_mask  = np.array([(t >= _OR_START and t < _OR_END) for t in bar_times])
            or_bars  = session[or_mask]
            if len(or_bars) < 2:
                continue

            total_or_sess += 1

            or_hi  = float(or_bars["high"].max())
            or_lo  = float(or_bars["low"].min())
            or_rng = or_hi - or_lo
            if or_rng <= 0.0:
                continue
            or_mid  = (or_hi + or_lo) / 2.0
            atr_end = float(or_bars["atr"].iloc[-1])
            if atr_end <= 0.0:
                continue

            # OR width filter — uses causal EWM of OR ranges
            if use_or_width:
                ewm_ref = or_ewm_ref.get(d, np.nan)
                if np.isnan(ewm_ref):
                    continue   # skip until EWM is warm (first session)
                if or_rng > or_width_max_mult * ewm_ref:
                    continue

            # POC (diagnostic only — TP is always or_mid)
            try:
                poc = compute_poc(or_bars)
            except Exception:
                poc = or_mid

            # Causal rejection detection
            rej  = detect_rejection(or_bars)
            l_ok = rej["long"]["valid"]
            s_ok = rej["short"]["valid"]
            if not l_ok and not s_ok:
                continue

            # Prev-session direction filter
            if use_prev_session:
                pr = prev_ret.get(d, np.nan)
                if np.isnan(pr) or pr == 0.0:
                    l_ok = s_ok = False
                elif pr > 0.0:
                    s_ok = False   # bullish prev → only LONG
                else:
                    l_ok = False   # bearish prev → only SHORT
            if not l_ok and not s_ok:
                continue

            # Entry triggers
            trig_l = or_lo + entry_pct * or_rng if l_ok else np.nan
            trig_s = or_hi - entry_pct * or_rng if s_ok else np.nan

            # Store setups for diagnostic
            if collect_diag:
                if l_ok:
                    setups.append(MRSetupInfo(
                        date=d, direction="long",
                        or_high=or_hi, or_low=or_lo, or_mid=or_mid,
                        or_poc=poc, or_range=or_rng,
                        rejection_price=rej["long"]["price"],
                        pen_ticks=rej["long"]["pen_ticks"],
                        entry_trigger=trig_l, atr_or_end=atr_end,
                    ))
                if s_ok:
                    setups.append(MRSetupInfo(
                        date=d, direction="short",
                        or_high=or_hi, or_low=or_lo, or_mid=or_mid,
                        or_poc=poc, or_range=or_rng,
                        rejection_price=rej["short"]["price"],
                        pen_ticks=rej["short"]["pen_ticks"],
                        entry_trigger=trig_s, atr_or_end=atr_end,
                    ))

            # Post-OR bars
            post_mask = np.array([t >= _OR_END for t in bar_times])
            post = session[post_mask]
            if post.empty:
                continue

            # TP targets depend on mode
            tp_l = or_hi if tp_far_extreme else or_mid   # LONG TP
            tp_s = or_lo if tp_far_extreme else or_mid   # SHORT TP
            # Guard: "don't enter if price already past TP"
            # Normal mode: close < or_mid | Far-extreme mode: close < or_hi
            guard_l = tp_l   # close must be strictly below
            guard_s = tp_s   # close must be strictly above

            # Scan for first entry trigger
            for ts, bar in post.iterrows():
                bt    = ts.time()
                bcl   = float(bar["close"])
                batr  = float(bar["atr"])

                if bt >= _NO_ENTRY:
                    break
                if batr <= 0.0:
                    continue

                traded = False

                # LONG
                if l_ok and bcl >= trig_l and bcl < guard_l:
                    epx = bcl + SLIP
                    sl  = epx - sl_atr_mult * batr
                    tp  = tp_l
                    if epx >= tp:
                        continue   # no room to TP — skip bar
                    rr  = (tp - epx) / (epx - sl)
                    xpx, xrsn, xts = cls._exit(post, ts, 1, epx, sl, tp, max_bars)
                    pnl_pts = xpx - epx
                    pnl_usd = pnl_pts * PV
                    trades.append(MRTrade(
                        trade_id=tid, date=d, entry_ts=ts, exit_ts=xts,
                        direction="long", entry_px=epx, exit_px=xpx,
                        sl_px=sl, tp_px=tp, or_mid=or_mid, exit_reason=xrsn,
                        pnl_pts=pnl_pts, pnl_usd=pnl_usd, pnl_net=pnl_usd - COMM,
                        rr=rr, year=ts.year, month=ts.month, dow=ts.dayofweek,
                        or_range=or_rng, atr_entry=batr,
                    ))
                    tid += 1
                    traded = True

                # SHORT
                elif s_ok and bcl <= trig_s and bcl > guard_s:
                    epx = bcl - SLIP
                    sl  = epx + sl_atr_mult * batr
                    tp  = tp_s
                    if epx <= tp:
                        continue
                    rr  = (epx - tp) / (sl - epx)
                    xpx, xrsn, xts = cls._exit(post, ts, -1, epx, sl, tp, max_bars)
                    pnl_pts = epx - xpx
                    pnl_usd = pnl_pts * PV
                    trades.append(MRTrade(
                        trade_id=tid, date=d, entry_ts=ts, exit_ts=xts,
                        direction="short", entry_px=epx, exit_px=xpx,
                        sl_px=sl, tp_px=tp, or_mid=or_mid, exit_reason=xrsn,
                        pnl_pts=pnl_pts, pnl_usd=pnl_usd, pnl_net=pnl_usd - COMM,
                        rr=rr, year=ts.year, month=ts.month, dow=ts.dayofweek,
                        or_range=or_rng, atr_entry=batr,
                    ))
                    tid += 1
                    traded = True

                if traded:
                    break   # one trade per session

        return MRResults(
            trades=trades,
            setups=setups,
            total_or_sessions=total_or_sess,
            initial_capital=initial_capital,
        )
