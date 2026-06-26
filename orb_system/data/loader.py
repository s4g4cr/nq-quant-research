"""
Data loading, preprocessing, and splitting for the NQ ORB backtest.

Pipeline:
  1. Read CSV (selected columns only)
  2. Build continuous 1-min series using roll schedule
  3. Filter to regular session (09:30-16:00 ET), Mon-Fri
  4. Cache result as pickle
"""

import logging
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(cfg, use_cache: bool = True) -> pd.DataFrame:
    """
    Load and preprocess NQ continuous futures data.

    Returns a DataFrame indexed by NY-local timestamp (tz-aware),
    columns: open, high, low, close, volume.
    """
    cache_path = Path(cfg.data.cache_path)

    if use_cache and cache_path.exists():
        logger.info("Cache found at %s — loading from disk", cache_path)
        df = pd.read_pickle(cache_path)
        logger.info("Cache loaded: %d rows, %s to %s", len(df), df.index[0], df.index[-1])
        return df

    logger.info("No cache found — processing raw CSV")
    df = _build_continuous(cfg)
    df = _filter_session(df, cfg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    logger.info("Data cached to %s", cache_path)

    return df


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict temporal split — first train_ratio of rows go to train, remainder to test.
    Never shuffles; ensures no future data leaks into training set.
    """
    n = len(df)
    cutoff = int(n * train_ratio)
    df_train = df.iloc[:cutoff].copy()
    df_test = df.iloc[cutoff:].copy()

    logger.info(
        "Train/test split at %.0f%%: train=%d rows (%s to %s), test=%d rows (%s to %s)",
        train_ratio * 100,
        len(df_train), df_train.index[0], df_train.index[-1],
        len(df_test), df_test.index[0], df_test.index[-1],
    )
    return df_train, df_test


def data_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics about the processed dataset."""
    sep = "-" * 50
    print(sep)
    print("DATA SUMMARY")
    print(sep)
    print(f"  Rows:          {len(df):,}")
    print(f"  Start:         {df.index[0]}")
    print(f"  End:           {df.index[-1]}")
    days = df.index.normalize().nunique()
    print(f"  Trading days:  {days:,}")
    print(f"  Avg bars/day:  {len(df) / days:.1f}")
    print(f"\n  Price range:   {df['low'].min():.2f} – {df['high'].max():.2f}")
    print(f"  Avg close:     {df['close'].mean():.2f}")
    print(f"  Avg volume:    {df['volume'].mean():,.0f} (per 1-min bar)")
    print(sep)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_continuous(cfg) -> pd.DataFrame:
    """
    Read the raw CSV and stitch together a continuous series using
    the roll schedule defined in cfg.data.roll_schedule.
    """
    logger.info("Reading CSV: %s", cfg.data.csv_path)

    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    raw = pd.read_csv(
        cfg.data.csv_path,
        usecols=usecols,
        dtype={
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "symbol": str,
        },
        low_memory=False,
    )
    logger.info("Raw CSV rows: %d", len(raw))

    # Parse timestamps (nanosecond precision, UTC)
    raw["ts_event"] = pd.to_datetime(raw["ts_event"], utc=True)
    raw = raw.sort_values("ts_event").reset_index(drop=True)

    # Build the continuous series bar-by-bar from the roll schedule
    pieces = []
    roll = cfg.data.roll_schedule

    for symbol, (start_str, end_str) in roll.items():
        start_utc = pd.Timestamp(start_str, tz="UTC")
        end_utc = pd.Timestamp(end_str, tz="UTC")

        mask = (
            (raw["symbol"] == symbol)
            & (raw["ts_event"] >= start_utc)
            & (raw["ts_event"] < end_utc)
        )
        chunk = raw.loc[mask].copy()

        if chunk.empty:
            logger.warning("No data found for %s in [%s, %s)", symbol, start_str, end_str)
        else:
            logger.info("  %s: %d 1-min bars (%s to %s)", symbol, len(chunk),
                        chunk["ts_event"].iloc[0], chunk["ts_event"].iloc[-1])
            pieces.append(chunk)

    if not pieces:
        raise ValueError("No data matched the roll schedule — check CSV and symbol names")

    df = pd.concat(pieces).sort_values("ts_event").reset_index(drop=True)
    df = df.set_index("ts_event")[["open", "high", "low", "close", "volume"]]

    logger.info("Continuous series: %d 1-min bars", len(df))
    return df


def _resample_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-min OHLCV to 15-min OHLCV.
    label='left', closed='left': the bar labeled 09:30 covers [09:30, 09:45).
    """
    logger.info("Resampling 1-min -> 15-min")

    df_15 = df.resample("15min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )

    # Drop bars where no trades occurred (e.g., overnight gaps)
    df_15 = df_15.dropna(subset=["close"])
    df_15 = df_15[df_15["volume"] > 0]

    logger.info("After resample: %d 15-min bars", len(df_15))
    return df_15


def _filter_session(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Keep only regular session bars (09:30–16:00 NY time), Monday–Friday.
    The index is converted to America/New_York (tz-aware).
    """
    logger.info("Filtering to regular session (%s–%s ET), Mon–Fri",
                cfg.data.session_open, cfg.data.session_close)

    # Convert UTC index to NY time
    df = df.copy()
    df.index = df.index.tz_convert(cfg.data.timezone)

    # Keep Mon–Fri
    df = df[df.index.dayofweek < 5]

    # Keep session bars: 09:30 <= bar_time < 16:00
    bar_time = df.index.time
    session_open = pd.Timestamp(f"2000-01-01 {cfg.data.session_open}").time()
    session_close = pd.Timestamp(f"2000-01-01 {cfg.data.session_close}").time()

    df = df[(bar_time >= session_open) & (bar_time < session_close)]

    logger.info("After session filter: %d 1-min bars", len(df))
    return df
