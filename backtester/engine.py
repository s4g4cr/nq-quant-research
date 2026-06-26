"""
backtester/engine.py — Motor de backtest independiente de la estrategia.

Diseño:
  - Itera vela a vela (event-driven, no vectorizado) para evitar lookahead.
  - Recibe señales a través de una interfaz estándar.
  - Gestiona posiciones, SL/TP y tiempo máximo.
  - Calcula métricas completas al final.
  - Produce un DataFrame de trades y curva de equity.

Reglas de simulación:
  1. Una sola posición abierta en cualquier momento.
  2. SL/TP evaluados usando High y Low de cada vela posterior.
  3. Si en la misma vela se toca SL y TP → gana el SL (caso conservador).
  4. Máximo N velas en posición → salida al cierre de la N-ésima vela.
  5. Cierre de sesión (16:00) → cierre forzado al close.
  6. Comisión + slippage aplicados en entrada y salida.

El engine NO sabe nada de la estrategia. Solo sabe:
  - Cuándo tiene una señal (entrada, SL, TP, dirección)
  - Cómo gestionar esa posición
  - Cómo calcular el PnL
"""

import logging
from typing import List, Optional, Dict, Any, Callable
from datetime import time as dt_time

import pandas as pd
import numpy as np

from config import RiskConfig, BacktestConfig
from strategy.orb import Signal, Trade, ORBSignalGenerator
from indicators.technical import add_all_indicators

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Motor de backtest
# ─────────────────────────────────────────────

class BacktestEngine:
    """
    Motor de backtest event-driven vela a vela.

    Args:
        signal_generator: Instancia del generador de señales (cualquier estrategia).
        risk_cfg: Configuración de gestión de riesgo.
        bt_cfg: Configuración del backtest.
    """

    def __init__(
        self,
        signal_generator: ORBSignalGenerator,
        risk_cfg: RiskConfig,
        bt_cfg: BacktestConfig,
    ):
        self.signal_gen = signal_generator
        self.risk_cfg = risk_cfg
        self.bt_cfg = bt_cfg

        self._trades: List[Trade] = []
        self._trade_counter: int = 0
        self._equity_curve: List[Dict] = []

        # Estado de posición abierta
        self._open_trade: Optional[Trade] = None
        self._bars_in_trade: int = 0

        # Capital actual
        self._capital: float = bt_cfg.initial_capital

        # Hora de cierre de sesión
        self._session_close = pd.Timestamp("2000-01-01 16:00").time()

    # ─────────────────────────────────────────────
    # Ejecución principal
    # ─────────────────────────────────────────────

    def run(self, df: pd.DataFrame, label: str = "backtest") -> "BacktestResults":
        """
        Ejecuta el backtest sobre un DataFrame OHLCV con indicadores.

        Args:
            df: DataFrame con OHLCV + indicadores (output de add_all_indicators).
            label: Etiqueta para los logs ("train" | "test").

        Returns:
            BacktestResults con trades, equity curve y métricas.
        """
        logger.info(f"Iniciando backtest [{label}]: {len(df)} velas")

        # Agrupar por sesión de trading
        trading_days = df.index.normalize().unique().sort_values()
        n_days = len(trading_days)

        for day_idx, day in enumerate(trading_days):
            session_mask = df.index.normalize() == day
            session_df = df[session_mask]

            self._process_session(session_df, day_idx, n_days)

        # Si queda posición abierta al final de los datos
        if self._open_trade is not None:
            logger.warning("Posición abierta al final del dataset, cerrando a precio de mercado")
            last_bar = df.iloc[-1]
            self._close_trade(last_bar, "end_of_data")

        results = BacktestResults(
            trades=self._trades,
            equity_curve=pd.DataFrame(self._equity_curve),
            initial_capital=self.bt_cfg.initial_capital,
            final_capital=self._capital,
            label=label,
        )

        logger.info(
            f"Backtest [{label}] completado: "
            f"{len(self._trades)} trades | "
            f"Capital final: ${self._capital:,.2f}"
        )
        return results

    # ─────────────────────────────────────────────
    # Procesamiento por sesión
    # ─────────────────────────────────────────────

    def _process_session(self, session_df: pd.DataFrame, day_idx: int, total_days: int) -> None:
        """Procesa una sesión completa vela a vela."""
        self.signal_gen.reset()

        # Calcular Opening Range antes de iterar las velas
        or_ok = self.signal_gen.compute_opening_range_for_session(session_df)
        if not or_ok:
            logger.debug(f"Sesión {session_df.index[0].date()}: OR no calculable, saltando")
            return

        # Iterar vela a vela
        for bar_pos, (timestamp, bar) in enumerate(session_df.iterrows()):

            # ── 1. GESTIÓN DE POSICIÓN ABIERTA ──
            if self._open_trade is not None:
                self._bars_in_trade += 1
                closed = self._evaluate_exit(bar, bar_pos)

                if not closed:
                    # Forzar salida por tiempo
                    if self._bars_in_trade >= self.risk_cfg.max_bars_in_trade:
                        self._close_trade(bar, "time_limit")
                        closed = True

                    # Forzar salida por cierre de sesión
                    elif bar.name.time() >= self._session_close:
                        self._close_trade(bar, "end_of_day")
                        closed = True

                # Registrar equity tras gestión de posición
                self._record_equity(bar)
                continue  # No buscar nueva señal mientras hay posición abierta

            # ── 2. BÚSQUEDA DE NUEVA SEÑAL ──
            signal = self.signal_gen.evaluate_bar(bar)

            if signal is not None:
                self._open_position(signal, bar_pos)

            # Registrar equity
            self._record_equity(bar)

    # ─────────────────────────────────────────────
    # Apertura y cierre de posiciones
    # ─────────────────────────────────────────────

    def _open_position(self, signal: Signal, bar_idx: int) -> None:
        """Abre una posición. Aplica slippage en entrada."""
        slippage = self._calc_slippage(signal.direction)
        actual_entry = signal.entry_price + slippage

        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            signal=signal,
            entry_bar=bar_idx,
        )
        # Ajustar entry con slippage
        trade.signal = Signal(
            timestamp=signal.timestamp,
            direction=signal.direction,
            entry_price=actual_entry,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            atr_at_entry=signal.atr_at_entry,
            or_high=signal.or_high,
            or_low=signal.or_low,
            candle_range=signal.candle_range,
            volume_ratio=signal.volume_ratio,
        )

        self._open_trade = trade
        self._bars_in_trade = 0

        logger.debug(
            f"[{signal.timestamp}] ABRIR {signal.direction.upper()} @ {actual_entry:.2f} "
            f"(SL={signal.sl_price:.2f}, TP={signal.tp_price:.2f})"
        )

    def _evaluate_exit(self, bar: pd.Series, bar_pos: int) -> bool:
        """
        Evalúa si la posición debe cerrarse por SL o TP.

        Usa High y Low de la vela para simular el rango intravela.
        Caso conservador: si toca SL y TP en la misma vela → SL gana.

        Returns:
            True si se cerró la posición.
        """
        trade = self._open_trade
        direction = trade.signal.direction
        sl = trade.signal.sl_price
        tp = trade.signal.tp_price
        high = bar["high"]
        low = bar["low"]

        sl_hit = False
        tp_hit = False

        if direction == "long":
            sl_hit = low <= sl
            tp_hit = high >= tp
        else:  # short
            sl_hit = high >= sl
            tp_hit = low <= tp

        if sl_hit:
            # SL primero (caso conservador)
            self._close_trade(bar, "sl", exit_price=sl)
            return True
        elif tp_hit:
            self._close_trade(bar, "tp", exit_price=tp)
            return True

        return False

    def _close_trade(
        self,
        bar: pd.Series,
        reason: str,
        exit_price: Optional[float] = None,
    ) -> None:
        """Cierra la posición abierta y calcula PnL."""
        trade = self._open_trade
        if trade is None:
            return

        direction = trade.signal.direction
        entry_price = trade.signal.entry_price

        # Precio de salida
        if exit_price is None:
            exit_price = bar["close"]  # Salida al cierre

        # Slippage en salida (adverso a la posición)
        slippage = self._calc_slippage(direction, is_exit=True)
        actual_exit = exit_price + slippage

        # PnL en puntos
        if direction == "long":
            pnl_points = actual_exit - entry_price
        else:
            pnl_points = entry_price - actual_exit

        # PnL en USD
        pnl_usd = pnl_points * self.risk_cfg.point_value * self.risk_cfg.contracts

        # Comisión (ambos lados)
        commission = self.risk_cfg.commission_per_side * 2 * self.risk_cfg.contracts

        # PnL neto
        pnl_net = pnl_usd - commission

        # Actualizar trade
        trade.exit_bar = bar.name
        trade.exit_price = actual_exit
        trade.exit_reason = reason
        trade.exit_timestamp = bar.name
        trade.pnl_points = pnl_points
        trade.pnl_usd = pnl_usd
        trade.pnl_net = pnl_net
        trade.is_winner = pnl_net > 0

        # Actualizar capital
        self._capital += pnl_net

        self._trades.append(trade)
        self._open_trade = None
        self._bars_in_trade = 0

        logger.debug(
            f"[{bar.name}] CERRAR {direction.upper()} @ {actual_exit:.2f} "
            f"| razón={reason} | PnL={pnl_net:+.2f}$ | capital={self._capital:,.2f}"
        )

    # ─────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────

    def _calc_slippage(self, direction: str, is_exit: bool = False) -> float:
        """
        Calcula el slippage en puntos.

        Entrada LONG: precio real es más alto (adverso).
        Salida LONG: precio real es más bajo (adverso).
        Entrada SHORT: precio real es más bajo (adverso).
        Salida SHORT: precio real es más alto (adverso).
        """
        tick_size = 0.25  # NQ: 1 tick = 0.25 puntos
        slip_points = self.risk_cfg.slippage_ticks * tick_size

        if not is_exit:
            # Entrada: slippage adverso
            return slip_points if direction == "long" else -slip_points
        else:
            # Salida: slippage adverso
            return -slip_points if direction == "long" else slip_points

    def _record_equity(self, bar: pd.Series) -> None:
        """Registra punto en la curva de equity."""
        self._equity_curve.append({
            "timestamp": bar.name,
            "capital": self._capital,
            "open_position": self._open_trade is not None,
        })


# ─────────────────────────────────────────────
# Resultados y métricas
# ─────────────────────────────────────────────

class BacktestResults:
    """
    Contenedor de resultados con cálculo de métricas completo.
    """

    def __init__(
        self,
        trades: List[Trade],
        equity_curve: pd.DataFrame,
        initial_capital: float,
        final_capital: float,
        label: str,
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.label = label
        self.trades_df = self._trades_to_df()

    def _trades_to_df(self) -> pd.DataFrame:
        """Convierte lista de Trade a DataFrame para análisis."""
        if not self.trades:
            return pd.DataFrame()

        rows = []
        for t in self.trades:
            rows.append({
                "trade_id": t.trade_id,
                "entry_timestamp": t.signal.timestamp,
                "exit_timestamp": t.exit_timestamp,
                "direction": t.signal.direction,
                "entry_price": t.signal.entry_price,
                "exit_price": t.exit_price,
                "sl_price": t.signal.sl_price,
                "tp_price": t.signal.tp_price,
                "exit_reason": t.exit_reason,
                "pnl_points": t.pnl_points,
                "pnl_usd": t.pnl_usd,
                "pnl_net": t.pnl_net,
                "is_winner": t.is_winner,
                "atr_at_entry": t.signal.atr_at_entry,
                "or_high": t.signal.or_high,
                "or_low": t.signal.or_low,
                "candle_range": t.signal.candle_range,
                "volume_ratio": t.signal.volume_ratio,
                "year": t.signal.timestamp.year,
                "month": t.signal.timestamp.month,
            })
        return pd.DataFrame(rows)

    def compute_metrics(self) -> Dict[str, Any]:
        """Calcula todas las métricas de rendimiento."""
        df = self.trades_df

        if df.empty or len(df) == 0:
            return {"error": "Sin trades para calcular métricas"}

        n_trades = len(df)
        winners = df[df["is_winner"] == True]
        losers = df[df["is_winner"] == False]

        win_rate = len(winners) / n_trades if n_trades > 0 else 0

        # Retorno acumulado
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital

        # Retorno medio por trade
        avg_return_per_trade = df["pnl_net"].mean()

        # Profit Factor
        gross_profit = winners["pnl_net"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["pnl_net"].sum()) if len(losers) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe Ratio (anualizado, asumiendo 252 días trading)
        returns = df["pnl_net"].values
        sharpe = self._calc_sharpe(returns)

        # Máximo Drawdown
        max_dd, max_dd_pct = self._calc_max_drawdown()

        # Exit reasons
        exit_counts = df["exit_reason"].value_counts().to_dict()

        # Por año
        yearly = self._yearly_breakdown(df)

        return {
            "label": self.label,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "total_return_pct": total_return * 100,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "avg_return_per_trade": avg_return_per_trade,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown_usd": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_winner": winners["pnl_net"].mean() if len(winners) > 0 else 0,
            "avg_loser": losers["pnl_net"].mean() if len(losers) > 0 else 0,
            "exit_reasons": exit_counts,
            "yearly": yearly,
            "n_long": (df["direction"] == "long").sum(),
            "n_short": (df["direction"] == "short").sum(),
        }

    def _calc_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Sharpe ratio anualizado sobre retornos por trade (en USD)."""
        if len(returns) < 2:
            return 0.0
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r == 0:
            return 0.0
        # Convertir a ratio anualizado asumiendo 1 trade promedio por día
        daily_sharpe = mean_r / std_r
        return daily_sharpe * np.sqrt(periods_per_year)

    def _calc_max_drawdown(self):
        """Calcula máximo drawdown sobre la curva de equity."""
        if self.equity_curve.empty:
            return 0.0, 0.0

        equity = self.equity_curve["capital"].values
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_dd_usd = abs(drawdown.min())
        max_dd_pct = (max_dd_usd / peak[np.argmin(drawdown)]) * 100 if peak.max() > 0 else 0
        return max_dd_usd, max_dd_pct

    def _yearly_breakdown(self, df: pd.DataFrame) -> List[Dict]:
        """Métricas anuales."""
        yearly = []
        for year, group in df.groupby("year"):
            n = len(group)
            wr = (group["is_winner"] == True).sum() / n if n > 0 else 0
            yearly.append({
                "year": year,
                "n_trades": n,
                "win_rate": wr,
                "total_pnl": group["pnl_net"].sum(),
                "avg_pnl": group["pnl_net"].mean(),
            })
        return yearly

    def print_report(self) -> None:
        """Imprime reporte formateado en consola."""
        m = self.compute_metrics()
        if "error" in m:
            print(f"\n[{self.label.upper()}] {m['error']}")
            return

        sep = "═" * 62
        print(f"\n{sep}")
        print(f"  RESULTADOS — {self.label.upper()}")
        print(sep)
        print(f"  Trades totales     : {m['n_trades']}")
        print(f"  Trades LONG        : {m['n_long']}")
        print(f"  Trades SHORT       : {m['n_short']}")
        print(f"  Win Rate           : {m['win_rate']*100:.1f}%")
        print(f"  Profit Factor      : {m['profit_factor']:.2f}")
        print(f"  Sharpe Ratio       : {m['sharpe_ratio']:.2f}")
        print(sep)
        print(f"  Capital Inicial    : ${m['initial_capital']:>12,.2f}")
        print(f"  Capital Final      : ${m['final_capital']:>12,.2f}")
        print(f"  Retorno Total      : {m['total_return_pct']:>+.2f}%")
        print(f"  PnL Medio/Trade    : ${m['avg_return_per_trade']:>+.2f}")
        print(f"  Max Drawdown       : ${m['max_drawdown_usd']:>,.2f} ({m['max_drawdown_pct']:.1f}%)")
        print(sep)
        print(f"  Ganancia media     : ${m['avg_winner']:>+.2f}")
        print(f"  Pérdida media      : ${m['avg_loser']:>+.2f}")
        print(sep)
        print(f"  Razones de salida  :")
        for reason, count in m["exit_reasons"].items():
            pct = count / m["n_trades"] * 100
            print(f"    {reason:<16}: {count:>4} ({pct:.1f}%)")
        print(sep)
        print(f"  RESULTADOS POR AÑO:")
        print(f"  {'Año':<6} {'Trades':>8} {'Win%':>8} {'PnL Total':>12} {'PnL Medio':>12}")
        print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*12} {'─'*12}")
        for y in m["yearly"]:
            print(
                f"  {y['year']:<6} {y['n_trades']:>8} "
                f"{y['win_rate']*100:>7.1f}% "
                f"${y['total_pnl']:>11,.2f} "
                f"${y['avg_pnl']:>11,.2f}"
            )
        print(sep)
