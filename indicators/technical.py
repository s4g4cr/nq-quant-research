"""
indicators/technical.py — Indicadores técnicos causales.

Principio fundamental:
  Todos los indicadores son CAUSALES: el valor en el índice i
  solo usa información disponible en i o antes.
  Ningún cálculo mira hacia adelante (no lookahead bias).

Indicadores implementados:
  - ATR (Average True Range)
  - Average Volume
  - Candle Range (High - Low)
  - Indicadores derivados para señales ORB
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from config import IndicatorConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Indicadores base
# ─────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Average True Range (Wilder's smoothing).

    True Range = max(H-L, |H-Cprev|, |L-Cprev|)

    Args:
        df: DataFrame con columnas high, low, close.
        period: Período de suavizado.

    Returns:
        Serie con ATR. Primeros `period` valores son NaN (comportamiento correcto).

    Note:
        Usamos EWM con alpha=1/period para replicar el cálculo de Wilder.
        Es equivalente a RMA (Running Moving Average) en TradingView.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing = EMA con alpha = 1/period
    atr_values = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    logger.debug(f"ATR({period}) calculado: {atr_values.notna().sum()} valores válidos")
    return atr_values.rename(f"atr_{period}")


def avg_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Media móvil simple del volumen.

    Args:
        df: DataFrame con columna volume.
        period: Período de la media.

    Returns:
        Serie con volumen medio. Primeros `period-1` valores son NaN.
    """
    avg_vol = df["volume"].rolling(window=period, min_periods=period).mean()
    return avg_vol.rename(f"avg_vol_{period}")


def candle_range(df: pd.DataFrame) -> pd.Series:
    """
    Rango de cada vela: High - Low.

    Returns:
        Serie con el rango punto a punto de cada vela.
    """
    return (df["high"] - df["low"]).rename("candle_range")


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average estándar.

    Args:
        series: Serie de precios.
        period: Período.
    """
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


# ─────────────────────────────────────────────
# Cálculo batch para el backtest
# ─────────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """
    Añade todos los indicadores necesarios al DataFrame.

    Devuelve un DataFrame nuevo (no modifica el original).
    Los indicadores se añaden como columnas adicionales.

    IMPORTANTE: Todos los cálculos son sobre el DataFrame completo
    usando solo operaciones causales (rolling, ewm con closed='left' implícito
    a través de shift). No hay riesgo de lookahead aquí porque los indicadores
    se calculan antes del loop del backtest.

    Args:
        df: DataFrame OHLCV limpio.
        cfg: Configuración de indicadores.

    Returns:
        DataFrame con columnas adicionales de indicadores.
    """
    result = df.copy()

    # ATR
    result["atr"] = atr(result, cfg.atr_period)

    # Volumen medio
    result["avg_vol"] = avg_volume(result, cfg.volume_period)

    # Rango de vela
    result["candle_range"] = candle_range(result)

    # Ratios para debugging / análisis
    result["vol_ratio"] = result["volume"] / result["avg_vol"]
    result["range_atr_ratio"] = result["candle_range"] / result["atr"]

    # Cuántas velas tienen todos los indicadores válidos
    valid = result[["atr", "avg_vol"]].notna().all(axis=1).sum()
    logger.info(f"Indicadores añadidos: {valid} velas con datos completos de {len(result)}")

    return result


# ─────────────────────────────────────────────
# Cálculo del Opening Range (por sesión)
# ─────────────────────────────────────────────

def compute_opening_range(session_df: pd.DataFrame,
                          range_start: str = "09:30",
                          range_end: str = "10:30") -> Optional[dict]:
    """
    Calcula el Opening Range de una sesión de trading.

    Solo usa las velas cuyo timestamp está dentro de [range_start, range_end).
    La vela de las 10:30 NO forma parte del rango (es la primera candidata
    a señal, pero su información es posterior al rango).

    Args:
        session_df: DataFrame de un solo día de trading.
        range_start: Hora de inicio del rango (inclusive).
        range_end: Hora de fin del rango (exclusive).

    Returns:
        Dict con {or_high, or_low, or_range, n_bars} o None si no hay datos.
    """
    # Velas dentro del Opening Range
    # Usamos between_time con inclusive='left' para excluir la vela de las 10:30
    try:
        or_bars = session_df.between_time(range_start, range_end, inclusive="left")
    except Exception:
        # Fallback para versiones antiguas de pandas
        or_bars = session_df.between_time(range_start, range_end)
        # Excluir manualmente la última vela si cae exactamente a range_end
        end_time = pd.Timestamp(f"2000-01-01 {range_end}").time()
        or_bars = or_bars[or_bars.index.time < end_time]

    if len(or_bars) < 2:
        logger.debug(f"Opening Range insuficiente: {len(or_bars)} velas")
        return None

    or_high = or_bars["high"].max()
    or_low = or_bars["low"].min()
    or_range = or_high - or_low

    if or_range <= 0:
        logger.debug("Opening Range con rango cero, ignorando sesión")
        return None

    return {
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_range,
        "n_bars": len(or_bars),
        "or_end_time": or_bars.index[-1],
    }


# ─────────────────────────────────────────────
# Diagnóstico de indicadores
# ─────────────────────────────────────────────

def indicators_summary(df: pd.DataFrame) -> str:
    """Resumen estadístico de los indicadores para diagnóstico."""
    cols = ["atr", "avg_vol", "candle_range", "vol_ratio", "range_atr_ratio"]
    available = [c for c in cols if c in df.columns]

    if not available:
        return "No hay indicadores calculados."

    stats = df[available].describe()
    lines = [
        "\n" + "─" * 50,
        "  ESTADÍSTICAS DE INDICADORES",
        "─" * 50,
    ]
    for col in available:
        if col in stats.columns:
            lines.append(
                f"  {col:<20}: media={stats[col]['mean']:.2f} | "
                f"std={stats[col]['std']:.2f} | "
                f"min={stats[col]['min']:.2f} | "
                f"max={stats[col]['max']:.2f}"
            )
    lines.append("─" * 50)
    return "\n".join(lines)
