"""
Causal technical indicators for the NQ ORB backtest.

All indicators use only past and current data (no lookahead).
ATR uses Wilder's EWM smoothing; avg_volume uses simple rolling mean.
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder's Average True Range (causal, EWM with alpha=1/period).

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    First bar has TR = high - low (no previous close available).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing: EWM with alpha = 1/period, adjust=False
    atr_series = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr_series


def avg_volume(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Simple rolling mean of volume over `period` bars (causal).
    Uses min_periods=1 so early bars get a partial average instead of NaN.
    """
    return df["volume"].rolling(window=period, min_periods=1).mean()


def session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session VWAP with expanding standard-deviation bands, causal and reset daily.

    Adds columns:
      vwap        – cumulative (typical_price × volume) / cumulative volume
      vwap_std    – expanding std of (typical_price – vwap) from session open
      vwap_upper1 – vwap + 1 × vwap_std
      vwap_lower1 – vwap - 1 × vwap_std
      vwap_upper2 – vwap + 2 × vwap_std
      vwap_lower2 – vwap - 2 × vwap_std
    """
    tp   = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv  = tp * df["volume"]

    # Use date as groupby key (timezone-aware safe)
    date_key = df.index.to_series().dt.normalize()

    cum_tpv = tpv.groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum().replace(0, np.nan)
    vwap    = cum_tpv / cum_vol

    # Expanding std of deviation within session — fully vectorised
    dev      = tp - vwap
    n        = df.groupby(date_key).cumcount() + 1       # 1-indexed bar count
    sum_d    = dev.groupby(date_key).cumsum()
    sum_d2   = (dev ** 2).groupby(date_key).cumsum()
    denom    = (n - 1).clip(lower=1).astype(float)
    var      = ((sum_d2 - sum_d ** 2 / n) / denom).clip(lower=0)
    vwap_std = np.sqrt(var).fillna(0.0)

    out = df.copy()
    out["vwap"]        = vwap
    out["vwap_std"]    = vwap_std
    out["vwap_upper1"] = vwap + vwap_std
    out["vwap_lower1"] = vwap - vwap_std
    out["vwap_upper2"] = vwap + 2.0 * vwap_std
    out["vwap_lower2"] = vwap - 2.0 * vwap_std
    return out


def add_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Compute and attach all indicator columns to a copy of df.

    Adds:
      atr           – Wilder ATR(atr_period)
      avg_vol       – rolling avg volume(volume_period)
      candle_rng    – high - low of each bar
      vol_ratio     – volume / avg_vol
      rng_atr_ratio – candle_rng / atr
      sma_trend     – SMA(trend_period) of close, used by the trend filter
      vwap          – session VWAP (reset daily at 09:30)
      vwap_std      – expanding intra-session std of (tp – vwap)
      vwap_upper1/2 – VWAP + 1σ / 2σ
      vwap_lower1/2 – VWAP - 1σ / 2σ
    """
    logger.info("Computing indicators (ATR=%d, VolPeriod=%d, SMA_trend=%d)",
                cfg.indicators.atr_period, cfg.indicators.volume_period,
                cfg.signal.trend_period)

    out = df.copy()
    out["atr"] = atr(out, cfg.indicators.atr_period)
    out["avg_vol"] = avg_volume(out, cfg.indicators.volume_period)
    out["candle_rng"] = out["high"] - out["low"]
    out["vol_ratio"] = out["volume"] / out["avg_vol"].replace(0, np.nan)
    out["rng_atr_ratio"] = out["candle_rng"] / out["atr"].replace(0, np.nan)
    out["sma_trend"] = out["close"].rolling(
        window=cfg.signal.trend_period, min_periods=1
    ).mean()
    out = session_vwap(out)

    logger.info("Indicators added: %d rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Opening Range
# ---------------------------------------------------------------------------

def compute_opening_range(
    session_df: pd.DataFrame,
    range_start: str,
    range_end: str,
) -> tuple[float, float]:
    """
    Calculate the Opening Range (OR) high and low for a single session.

    Includes bars whose time >= range_start and time < range_end.
    The bar AT range_end is NOT part of the OR — it is the first candidate
    for a breakout signal.

    Returns (or_high, or_low). Returns (nan, nan) if no bars fall in range.
    """
    start_time = pd.Timestamp(f"2000-01-01 {range_start}").time()
    end_time = pd.Timestamp(f"2000-01-01 {range_end}").time()

    bar_times = session_df.index.time
    or_bars = session_df[(bar_times >= start_time) & (bar_times < end_time)]

    if or_bars.empty:
        return float("nan"), float("nan")

    or_high = or_bars["high"].max()
    or_low = or_bars["low"].min()
    return or_high, or_low
