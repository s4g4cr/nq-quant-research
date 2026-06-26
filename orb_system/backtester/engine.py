"""
ORB Backtest Engine — 1-min bar edition, v2.

Filters applied in this priority order (session-level first, then bar-level):
  1. OR width filter     (session: skip if OR too wide vs prev ATR)
  2. Prev-session filter (session: align direction with yesterday's session return)
  3. Gap-of-open filter  (session: align direction with overnight gap)
  4. OR-position filter  (session: require close-in-OR to be near relevant extreme)
  5. Trend filter        (bar: SMA alignment at signal time)
  6. Candle+volume entry (bar: core ORB signal)

Exit priority within each bar:
  SL fixed  >  TP fixed  >  Trailing stop  >  EOD (15:45)  >  Timeout (120 bars)

Conservative assumption: if SL and TP both touched in the same bar, SL wins.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    trade_id: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: str              # "long" | "short"
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    trail_price: float          # NaN if trailing never armed
    exit_reason: str            # "sl" | "tp" | "trailing" | "eod" | "timeout"
    pnl_points: float
    pnl_usd: float
    pnl_net: float
    atr_at_entry: float
    or_high: float
    or_low: float
    or_position: float          # (or_close - or_low) / (or_high - or_low)
    candle_rng: float
    vol_ratio: float
    year: int
    month: int
    day_of_week: int            # 0=Mon … 4=Fri


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Results:
    trades: list[Trade] = field(default_factory=list)
    initial_capital: float = 100_000.0
    label: str = ""

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict:
        if not self.trades:
            return {"n_trades": 0}

        pnl = [t.pnl_net for t in self.trades]
        winners = [p for p in pnl if p > 0]
        losers  = [p for p in pnl if p <= 0]

        # Equity / drawdown
        equity = [self.initial_capital]
        for p in pnl:
            equity.append(equity[-1] + p)
        peak, max_dd = equity[0], 0.0
        for e in equity:
            peak = max(peak, e)
            max_dd = max(max_dd, peak - e)
        max_dd_pct = max_dd / self.initial_capital * 100

        # Sharpe (per-trade, annualised)
        arr = np.array(pnl)
        sharpe = (arr.mean() / arr.std() * math.sqrt(252)) if arr.std() > 0 else 0.0

        # Profit factor
        gw = sum(winners)
        gl = abs(sum(losers))
        pf = gw / gl if gl > 0 else _NAN

        # Exit reasons
        exits: dict[str, int] = {}
        for t in self.trades:
            exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1

        # Per-year
        yr_data: dict[int, list] = {}
        for t in self.trades:
            yr_data.setdefault(t.year, []).append(t.pnl_net)

        yearly = []
        for yr in sorted(yr_data):
            yp = yr_data[yr]
            yw = [p for p in yp if p > 0]
            yl = [p for p in yp if p <= 0]
            ygw, ygl = sum(yw), abs(sum(yl))
            yearly.append({
                "year":     yr,
                "trades":   len(yp),
                "win_rate": len(yw) / len(yp),
                "pf":       ygw / ygl if ygl > 0 else _NAN,
                "pnl_total":sum(yp),
                "pnl_avg":  sum(yp) / len(yp),
            })

        # Per-year per-direction
        yd_data: dict[tuple, list] = {}
        for t in self.trades:
            yd_data.setdefault((t.year, t.direction), []).append(t.pnl_net)

        yearly_by_dir = []
        for (yr, d), yp in sorted(yd_data.items()):
            yw = [p for p in yp if p > 0]
            yl = [p for p in yp if p <= 0]
            ygw, ygl = sum(yw), abs(sum(yl))
            yearly_by_dir.append({
                "year":      yr,
                "direction": d,
                "trades":    len(yp),
                "win_rate":  len(yw) / len(yp),
                "pf":        ygw / ygl if ygl > 0 else _NAN,
                "pnl_total": sum(yp),
            })

        # Day-of-week
        dow_data: dict[int, list] = {}
        for t in self.trades:
            dow_data.setdefault(t.day_of_week, []).append(t.pnl_net)

        dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        by_dow = []
        for d in sorted(dow_data):
            dp = dow_data[d]
            dw = [p for p in dp if p > 0]
            dl = [p for p in dp if p <= 0]
            dgw, dgl = sum(dw), abs(sum(dl))
            by_dow.append({
                "day":      dow_names.get(d, str(d)),
                "trades":   len(dp),
                "win_rate": len(dw) / len(dp),
                "pf":       dgw / dgl if dgl > 0 else _NAN,
                "pnl_total":sum(dp),
            })

        n_long = sum(1 for t in self.trades if t.direction == "long")
        final = equity[-1]

        return {
            "n_trades":        len(self.trades),
            "n_long":          n_long,
            "n_short":         len(self.trades) - n_long,
            "win_rate":        len(winners) / len(pnl),
            "profit_factor":   pf,
            "sharpe_ratio":    sharpe,
            "total_return_pct":(final - self.initial_capital) / self.initial_capital * 100,
            "initial_capital": self.initial_capital,
            "final_capital":   final,
            "avg_per_trade":   float(np.mean(pnl)),
            "avg_winner":      float(np.mean(winners)) if winners else 0.0,
            "avg_loser":       float(np.mean(losers))  if losers  else 0.0,
            "max_drawdown_usd":max_dd,
            "max_drawdown_pct":max_dd_pct,
            "exit_reasons":    exits,
            "yearly":          yearly,
            "yearly_by_dir":   yearly_by_dir,
            "by_dow":          by_dow,
        }

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_report(self, split_label: str = "") -> None:
        m = self.metrics()
        lbl = split_label or self.label
        sep = "=" * 62

        print(f"\n{sep}")
        print(f"  {lbl}")
        print(sep)

        if m["n_trades"] == 0:
            print("  No trades generated.")
            print(sep)
            return

        exits = m["exit_reasons"]
        n = m["n_trades"]
        pf_str = f"{m['profit_factor']:.2f}" if not math.isnan(m["profit_factor"]) else "inf"

        def epct(k): return exits.get(k, 0) / n * 100

        print(f"  Trades: {n}  (LONG: {m['n_long']} | SHORT: {m['n_short']})")
        print(f"  Win Rate: {m['win_rate']:.1%} | PF: {pf_str} | "
              f"Sharpe: {m['sharpe_ratio']:.2f} | Return: {m['total_return_pct']:+.2f}%")
        print(f"  Exits  SL:{epct('sl'):.0f}%  TP:{epct('tp'):.0f}%  "
              f"Trail:{epct('trailing'):.0f}%  EOD:{epct('eod'):.0f}%  "
              f"Time:{epct('timeout'):.0f}%")
        print(f"  Avg winner: ${m['avg_winner']:,.2f}  |  Avg loser: ${m['avg_loser']:,.2f}")
        print(f"  Capital: ${m['initial_capital']:,.0f} -> ${m['final_capital']:,.0f}  "
              f"(MaxDD: ${m['max_drawdown_usd']:,.0f}  {m['max_drawdown_pct']:.1f}%)")

        print(f"\n  {'Year':<5} {'Trades':>6} {'Win%':>6} {'PF':>5}  {'Total PnL':>11}  {'Avg/Tr':>9}")
        print("  " + "-" * 47)
        for r in m["yearly"]:
            pf_r = f"{r['pf']:.2f}" if not math.isnan(r["pf"]) else "inf"
            print(f"  {r['year']:<5} {r['trades']:>6} {r['win_rate']:>6.1%} "
                  f"{pf_r:>5}  ${r['pnl_total']:>10,.0f}  ${r['pnl_avg']:>8,.0f}")
        print(sep)

    def print_direction_annual_breakdown(self, split_label: str = "") -> None:
        """Detailed per-direction, per-year breakdown for Experiment 1."""
        m = self.metrics()
        lbl = split_label or self.label

        print(f"\n  === Direction x Year breakdown: {lbl} ===")
        print(f"  {'Year':<5} {'Dir':<6} {'Trades':>6} {'Win%':>6} {'PF':>5}  {'Total PnL':>11}")
        print("  " + "-" * 45)
        for r in m["yearly_by_dir"]:
            pf_r = f"{r['pf']:.2f}" if not math.isnan(r["pf"]) else "inf"
            print(f"  {r['year']:<5} {r['direction']:<6} {r['trades']:>6} "
                  f"{r['win_rate']:>6.1%} {pf_r:>5}  ${r['pnl_total']:>10,.0f}")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Bar-by-bar ORB backtest engine (1-min, multi-filter).

    All context filters are computed from the df slice passed to run()
    so no future data leaks across train/test boundaries.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, label: str = "") -> Results:
        logger.info("Backtest [%s]: %d bars  %s -> %s",
                    label, len(df), df.index[0], df.index[-1])

        cfg   = self.cfg
        risk  = cfg.risk
        sig   = cfg.signal
        orb   = cfg.orb

        slip       = risk.slippage_ticks * risk.tick_size
        commission = risk.commission_per_side * 2.0 * risk.contracts
        eod_t      = pd.Timestamp(f"2000-01-01 {risk.eod_exit_time}").time()
        rs_t       = pd.Timestamp(f"2000-01-01 {orb.range_start}").time()
        re_t       = pd.Timestamp(f"2000-01-01 {orb.range_end}").time()

        results = Results(initial_capital=cfg.backtest.initial_capital, label=label)
        capital = cfg.backtest.initial_capital
        tid     = 0

        # Precompute context used by session-level filters
        prev_day_atr     = self._build_prev_day_atr(df)
        session_context  = self._build_session_context(df)

        date_arr = df.index.normalize()
        for date in sorted(date_arr.unique()):
            session_df = df[date_arr == date]
            if len(session_df) < 2:
                continue

            # ---- OR computation ----
            bar_t = session_df.index.time
            or_bars = session_df[(bar_t >= rs_t) & (bar_t < re_t)]
            if or_bars.empty:
                continue
            or_high   = float(or_bars["high"].max())
            or_low    = float(or_bars["low"].min())
            or_close  = float(or_bars["close"].iloc[-1])
            or_range  = or_high - or_low
            if or_range <= 0:
                continue
            pos_in_or = (or_close - or_low) / or_range

            # ---- Session-level filters ----
            allow_L = sig.direction_filter in ("long",  "both")
            allow_S = sig.direction_filter in ("short", "both")

            # OR width filter
            if sig.or_width_filter:
                prev_atr = prev_day_atr.get(date)
                if prev_atr and (or_range > sig.or_width_max_atr_mult * prev_atr):
                    continue

            # Previous-session direction
            if sig.use_prev_session_filter:
                ctx = session_context.get(date, {})
                pr  = ctx.get("prev_return")
                if pr is None or pr == 0:
                    continue
                if pr > 0:
                    allow_S = False
                else:
                    allow_L = False

            # Gap-of-open direction
            if sig.use_gap_filter:
                ctx = session_context.get(date, {})
                gap = ctx.get("gap")
                if gap is None:
                    continue
                if gap > sig.gap_min_points:
                    allow_S = False
                elif gap < -sig.gap_min_points:
                    allow_L = False
                else:
                    continue   # gap too small or zero

            # OR-position filter
            if sig.use_or_position_filter:
                if pos_in_or < sig.or_position_long_min:
                    allow_L = False
                if pos_in_or > sig.or_position_short_max:
                    allow_S = False

            if not allow_L and not allow_S:
                continue

            # ---- Bar-by-bar loop ----
            sig_df = session_df[session_df.index.time >= re_t]
            if sig_df.empty:
                continue

            in_pos  = False
            dirn: Optional[str] = None
            entry_px = sl_px = tp_px = 0.0
            entry_ts: Optional[pd.Timestamp] = None
            atr_e = crng_e = vratio_e = 0.0
            bars_held = 0
            peak_fav  = 0.0
            trail_live = _NAN
            fired = False

            for bar in sig_df.itertuples():
                ts       = bar.Index
                btime    = ts.time()
                h, l, c  = bar.high, bar.low, bar.close
                vol      = bar.volume
                atr_v    = bar.atr
                avgv_v   = bar.avg_vol
                crng_v   = bar.candle_rng
                vratio_v = bar.vol_ratio

                # ---- Manage open position ----
                if in_pos:
                    bars_held += 1

                    # Update peak
                    if dirn == "long":
                        peak_fav = max(peak_fav, h)
                    else:
                        peak_fav = min(peak_fav, l)

                    # Recompute trail
                    if risk.use_trailing_exit:
                        trail_live = self._trail(
                            dirn, peak_fav, entry_px, atr_e,
                            risk.trailing_atr_mult,
                            risk.trailing_activation_atr_mult,
                        )

                    exit_px: Optional[float] = None
                    exit_rs: Optional[str]   = None

                    # 1. SL
                    if dirn == "long" and l <= sl_px:
                        exit_px, exit_rs = sl_px - slip, "sl"
                    elif dirn == "short" and h >= sl_px:
                        exit_px, exit_rs = sl_px + slip, "sl"

                    # 2. TP
                    if exit_rs is None:
                        if dirn == "long" and h >= tp_px:
                            exit_px, exit_rs = tp_px - slip, "tp"
                        elif dirn == "short" and l <= tp_px:
                            exit_px, exit_rs = tp_px + slip, "tp"

                    # 3. Trailing
                    if exit_rs is None and not math.isnan(trail_live):
                        if dirn == "long" and l <= trail_live:
                            exit_px, exit_rs = trail_live - slip, "trailing"
                        elif dirn == "short" and h >= trail_live:
                            exit_px, exit_rs = trail_live + slip, "trailing"

                    # 4. EOD
                    if exit_rs is None and btime >= eod_t:
                        exit_px = c - slip if dirn == "long" else c + slip
                        exit_rs = "eod"

                    # 5. Timeout
                    if exit_rs is None and bars_held >= risk.max_bars_in_trade:
                        exit_px = c - slip if dirn == "long" else c + slip
                        exit_rs = "timeout"

                    if exit_px is not None:
                        ppts = (exit_px - entry_px) if dirn == "long" else (entry_px - exit_px)
                        pusd = ppts * risk.point_value * risk.contracts
                        pnet = pusd - commission
                        capital += pnet
                        results.trades.append(Trade(
                            trade_id=tid, entry_ts=entry_ts, exit_ts=ts,
                            direction=dirn,
                            entry_price=entry_px, exit_price=exit_px,
                            sl_price=sl_px, tp_price=tp_px, trail_price=trail_live,
                            exit_reason=exit_rs,
                            pnl_points=ppts, pnl_usd=pusd, pnl_net=pnet,
                            atr_at_entry=atr_e,
                            or_high=or_high, or_low=or_low, or_position=pos_in_or,
                            candle_rng=crng_e, vol_ratio=vratio_e,
                            year=ts.year, month=ts.month,
                            day_of_week=ts.dayofweek,
                        ))
                        tid += 1
                        in_pos = False
                        dirn   = None
                    continue   # no new entry on same bar

                # ---- Entry detection ----
                if fired or btime >= eod_t:
                    continue
                if math.isnan(atr_v) or math.isnan(avgv_v) or atr_v == 0:
                    continue

                # Trend filter (bar-level)
                if sig.use_trend_filter:
                    lt_ok = c > bar.sma_trend
                    st_ok = c < bar.sma_trend
                else:
                    lt_ok = st_ok = True

                l_sig = (
                    allow_L and lt_ok
                    and c > or_high
                    and crng_v > sig.candle_range_multiplier * atr_v
                    and vol  > sig.volume_multiplier * avgv_v
                )
                s_sig = (
                    allow_S and st_ok
                    and c < or_low
                    and crng_v > sig.candle_range_multiplier * atr_v
                    and vol  > sig.volume_multiplier * avgv_v
                )

                if not (l_sig or s_sig):
                    continue

                dirn = "long" if l_sig else "short"
                if dirn == "long":
                    entry_px = c + slip
                    sl_px    = entry_px - risk.sl_atr_multiplier * atr_v
                    tp_px    = entry_px + risk.tp_atr_multiplier * atr_v
                    peak_fav = entry_px
                else:
                    entry_px = c - slip
                    sl_px    = entry_px + risk.sl_atr_multiplier * atr_v
                    tp_px    = entry_px - risk.tp_atr_multiplier * atr_v
                    peak_fav = entry_px

                trail_live = _NAN
                in_pos   = True
                entry_ts = ts
                atr_e    = atr_v
                crng_e   = crng_v
                vratio_e = vratio_v
                bars_held = 0
                fired     = True

            # Safety: still open at session end
            if in_pos:
                last = list(sig_df.itertuples())[-1]
                ep   = last.close - slip if dirn == "long" else last.close + slip
                ppts = (ep - entry_px) if dirn == "long" else (entry_px - ep)
                pusd = ppts * risk.point_value * risk.contracts
                pnet = pusd - commission
                capital += pnet
                results.trades.append(Trade(
                    trade_id=tid, entry_ts=entry_ts, exit_ts=last.Index,
                    direction=dirn,
                    entry_price=entry_px, exit_price=ep,
                    sl_price=sl_px, tp_price=tp_px, trail_price=trail_live,
                    exit_reason="eod",
                    pnl_points=ppts, pnl_usd=pusd, pnl_net=pnet,
                    atr_at_entry=atr_e,
                    or_high=or_high, or_low=or_low, or_position=pos_in_or,
                    candle_rng=crng_e, vol_ratio=vratio_e,
                    year=last.Index.year, month=last.Index.month,
                    day_of_week=last.Index.dayofweek,
                ))
                tid += 1

        logger.info("Backtest [%s] done: %d trades, capital $%.2f",
                    label, len(results.trades), capital)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prev_day_atr(df: pd.DataFrame) -> dict:
        date_arr = df.index.normalize()
        last_atr: dict = {}
        for date, sess in df.groupby(date_arr):
            v = sess["atr"].dropna()
            if not v.empty:
                last_atr[date] = float(v.iloc[-1])
        dates = sorted(last_atr)
        return {dates[i]: last_atr[dates[i - 1]] for i in range(1, len(dates))}

    @staticmethod
    def _build_session_context(df: pd.DataFrame) -> dict:
        """
        Compute per-session:
          prev_return = close_yesterday_15:45 - open_yesterday_09:30
          gap         = open_today_09:30 - close_yesterday_15:45
        All causal: uses only data from previous completed sessions.
        """
        open_t = pd.Timestamp("2000-01-01 09:30").time()
        eod_t  = pd.Timestamp("2000-01-01 15:45").time()

        sess_data: dict = {}
        date_arr = df.index.normalize()
        for date, sess in df.groupby(date_arr):
            ob = sess[sess.index.time == open_t]
            eb = sess[sess.index.time <= eod_t]
            if ob.empty or eb.empty:
                continue
            sess_data[date] = {
                "open_0930":  float(ob.iloc[0]["open"]),
                "close_1545": float(eb.iloc[-1]["close"]),
            }

        dates = sorted(sess_data)
        ctx: dict = {}
        for i in range(1, len(dates)):
            d, pd_ = dates[i], dates[i - 1]
            prev  = sess_data[pd_]
            today = sess_data[d]
            ctx[d] = {
                "prev_return": prev["close_1545"]  - prev["open_0930"],
                "gap":         today["open_0930"]  - prev["close_1545"],
            }
        return ctx

    @staticmethod
    def _trail(direction: str, peak: float, entry: float,
               atr: float, trail_mult: float, arm_mult: float) -> float:
        """Return trailing-stop price, or NaN if not yet armed."""
        if direction == "long":
            if peak > entry + arm_mult * atr:
                return peak - trail_mult * atr
        else:
            if peak < entry - arm_mult * atr:
                return peak + trail_mult * atr
        return _NAN
