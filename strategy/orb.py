"""
strategy/orb.py — Opening Range Breakout Strategy.

Hipótesis a testear:
  El rango formado durante 09:30–10:30 contiene información direccional.
  Una ruptura con momentum (rango vela > 1 ATR) y volumen superior
  (> 1.5x media) genera continuación.

Reglas exactas:
  LONG:
    - Timestamp > 10:30
    - close > OR High
    - (high - low) > 1 * ATR(20)
    - volume > 1.5 * avg_volume(20)
    - Entrada al cierre de la vela de ruptura

  SHORT:
    - Timestamp > 10:30
    - close < OR Low
    - (high - low) > 1 * ATR(20)
    - volume > 1.5 * avg_volume(20)
    - Entrada al cierre de la vela de ruptura

  Gestión:
    - SL = entrada ∓ 1 ATR
    - TP = entrada ± 2 ATR
    - Max 5 velas en posición
    - Una sola posición abierta por sesión

Garantías anti-lookahead:
  - El Opening Range se calcula con velas ANTERIORES a 10:30.
  - Las señales se generan usando solo datos de la vela actual.
  - No se accede a ningún dato futuro dentro del loop.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import time

import pandas as pd
import numpy as np

from config import ORBConfig, SignalConfig, RiskConfig
from indicators.technical import compute_opening_range

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────

@dataclass
class Signal:
    """Señal de entrada generada por la estrategia."""
    timestamp: pd.Timestamp
    direction: str          # "long" | "short"
    entry_price: float      # Precio de entrada (cierre de vela de ruptura)
    sl_price: float         # Stop Loss
    tp_price: float         # Take Profit
    atr_at_entry: float     # ATR en el momento de la señal
    or_high: float          # OR High de la sesión
    or_low: float           # OR Low de la sesión
    candle_range: float     # Rango de la vela de ruptura
    volume_ratio: float     # Ratio volumen / vol_media


@dataclass
class Trade:
    """Registro completo de una operación."""
    trade_id: int
    signal: Signal
    entry_bar: int          # Índice de barra de entrada
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None   # "sl" | "tp" | "time" | "eod"
    exit_timestamp: Optional[pd.Timestamp] = None
    pnl_points: float = 0.0            # PnL en puntos
    pnl_usd: float = 0.0               # PnL en USD (antes de costes)
    pnl_net: float = 0.0               # PnL neto (después de costes)
    is_winner: Optional[bool] = None


# ─────────────────────────────────────────────
# Generador de señales
# ─────────────────────────────────────────────

class ORBSignalGenerator:
    """
    Genera señales ORB para una sesión de trading.

    Una instancia por sesión. Se reinicia cada día.
    """

    def __init__(self, orb_cfg: ORBConfig, signal_cfg: SignalConfig, risk_cfg: RiskConfig):
        self.orb_cfg = orb_cfg
        self.signal_cfg = signal_cfg
        self.risk_cfg = risk_cfg

        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_computed: bool = False
        self._signal_fired: bool = False  # Una sola señal por sesión

        # Hora a partir de la cual se permiten señales
        self._signal_start = pd.Timestamp(f"2000-01-01 {orb_cfg.range_end}").time()

    def reset(self) -> None:
        """Reiniciar para nueva sesión."""
        self._or_high = None
        self._or_low = None
        self._or_computed = False
        self._signal_fired = False

    def compute_opening_range_for_session(self, session_df: pd.DataFrame) -> bool:
        """
        Calcula el Opening Range de la sesión.

        Args:
            session_df: DataFrame de toda la sesión (se filtra internamente).

        Returns:
            True si el OR se calculó correctamente.
        """
        or_data = compute_opening_range(
            session_df,
            range_start=self.orb_cfg.range_start,
            range_end=self.orb_cfg.range_end,
        )
        if or_data is None:
            return False

        self._or_high = or_data["or_high"]
        self._or_low = or_data["or_low"]
        self._or_computed = True
        return True

    def evaluate_bar(self, bar: pd.Series) -> Optional[Signal]:
        """
        Evalúa una vela para generar señal.

        Se llama vela a vela en el loop del backtest.
        Usa solo información disponible en la vela actual.

        Args:
            bar: Fila del DataFrame (una vela de 15min).

        Returns:
            Signal si se cumplen todas las condiciones, None en caso contrario.
        """
        if not self._or_computed:
            return None

        if self._signal_fired:
            return None  # Solo una señal por sesión

        # Solo después del cierre del Opening Range
        if bar.name.time() < self._signal_start:
            return None

        # Verificar que tenemos indicadores válidos
        atr_val = bar.get("atr", np.nan)
        avg_vol = bar.get("avg_vol", np.nan)
        if pd.isna(atr_val) or pd.isna(avg_vol) or atr_val <= 0 or avg_vol <= 0:
            return None

        close = bar["close"]
        high = bar["high"]
        low = bar["low"]
        volume = bar["volume"]
        candle_rng = high - low
        vol_ratio = volume / avg_vol

        # ─── CONDICIONES DE RUPTURA ───
        long_break = close > self._or_high
        short_break = close < self._or_low

        # Condición de momentum (rango vela)
        momentum_ok = candle_rng > (self.signal_cfg.candle_range_multiplier * atr_val)

        # Condición de volumen
        volume_ok = vol_ratio > self.signal_cfg.volume_multiplier

        direction = None
        if long_break and momentum_ok and volume_ok:
            direction = "long"
        elif short_break and momentum_ok and volume_ok:
            direction = "short"

        if direction is None:
            return None

        # ─── CONSTRUCCIÓN DE LA SEÑAL ───
        entry_price = close  # Entrada al cierre de la vela de ruptura

        if direction == "long":
            sl_price = entry_price - (self.risk_cfg.sl_atr_multiplier * atr_val)
            tp_price = entry_price + (self.risk_cfg.tp_atr_multiplier * atr_val)
        else:
            sl_price = entry_price + (self.risk_cfg.sl_atr_multiplier * atr_val)
            tp_price = entry_price - (self.risk_cfg.tp_atr_multiplier * atr_val)

        self._signal_fired = True

        signal = Signal(
            timestamp=bar.name,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            atr_at_entry=atr_val,
            or_high=self._or_high,
            or_low=self._or_low,
            candle_range=candle_rng,
            volume_ratio=vol_ratio,
        )

        logger.debug(
            f"[{bar.name}] SEÑAL {direction.upper()} | "
            f"entry={entry_price:.2f} | sl={sl_price:.2f} | tp={tp_price:.2f} | "
            f"ATR={atr_val:.2f} | vol_ratio={vol_ratio:.2f}"
        )
        return signal

    @property
    def or_levels(self) -> Dict[str, Optional[float]]:
        return {"or_high": self._or_high, "or_low": self._or_low}
