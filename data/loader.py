"""
data/loader.py — Carga de datos desde Databento CSV
Reagrupa velas de 1min en 15min y filtra sesión regular NY.
"""

import os
import pickle
import logging
from typing import Tuple

import pandas as pd

from config import DataConfig

logger = logging.getLogger(__name__)

DATABENTO_CSV = "glbx-mdp3-20210625-20260624.ohlcv-1m.csv"  # <-- pon aquí el nombre exacto de tu CSV


def load_data(cfg: DataConfig, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and os.path.exists(cfg.cache_path):
        logger.info(f"Cargando datos desde caché: {cfg.cache_path}")
        df = _load_cache(cfg.cache_path)
    else:
        logger.info(f"Cargando CSV de Databento: {DATABENTO_CSV}")
        df = _load_databento_csv(cfg)
        if use_cache:
            _save_cache(df, cfg.cache_path)

    df = _filter_session(df, cfg)
    df = _clean(df)

    logger.info(
        f"Datos cargados: {len(df)} velas | "
        f"{df.index[0].date()} → {df.index[-1].date()} | "
        f"{df.index.normalize().nunique()} días de trading"
    )
    return df


def _load_databento_csv(cfg: DataConfig) -> pd.DataFrame:
    if not os.path.exists(DATABENTO_CSV):
        raise FileNotFoundError(
            f"No se encuentra '{DATABENTO_CSV}'. "
            f"Asegúrate de que está en la carpeta del proyecto: {os.getcwd()}"
        )

    logger.info("Leyendo CSV (puede tardar unos segundos con 5 años de datos)...")
    df = pd.read_csv(
        DATABENTO_CSV,
        usecols=["ts_event", "open", "high", "low", "close", "volume"],
        parse_dates=["ts_event"],
    )

    # Limpiar timestamp: quitar nanosegundos extra y convertir a tz NY
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.set_index("ts_event")
    df.index = df.index.tz_convert(cfg.timezone)

    # Renombrar columnas a minúsculas
    df.columns = [c.lower() for c in df.columns]

    logger.info(f"CSV leído: {len(df)} velas de 1min")

    # Reagrupar de 1min a 15min
    logger.info("Reagrupando a 15 minutos...")
    df_15 = df.resample("15min", label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    logger.info(f"Velas de 15min generadas: {len(df_15)}")
    return df_15


def _filter_session(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    df_s = df.between_time(cfg.session_open, cfg.session_close)
    df_s = df_s[df_s.index.dayofweek < 5]
    return df_s


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[df["volume"] > 0]
    df = df[df["high"] >= df["low"]]
    df = df[df["close"] > 0]
    return df.sort_index()


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.70) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trading_days = df.index.normalize().unique().sort_values()
    n_train = int(len(trading_days) * train_ratio)
    split_date = trading_days[n_train]
    df_train = df[df.index.normalize() < split_date].copy()
    df_test  = df[df.index.normalize() >= split_date].copy()
    logger.info(
        f"Split → TRAIN: {df_train.index[0].date()} a {df_train.index[-1].date()} | "
        f"TEST: {df_test.index[0].date()} a {df_test.index[-1].date()}"
    )
    return df_train, df_test


def get_daily_sessions(df: pd.DataFrame) -> dict:
    sessions = {}
    for date, group in df.groupby(df.index.normalize()):
        if len(group) >= 4:
            sessions[date.date()] = group
    return sessions


def data_summary(df: pd.DataFrame) -> str:
    trading_days = df.index.normalize().nunique()
    years = sorted(df.index.year.unique().tolist())
    avg_bars = len(df) / trading_days if trading_days > 0 else 0
    lines = [
        "\n" + "─" * 50,
        "  RESUMEN DE DATOS",
        "─" * 50,
        f"  Velas totales   : {len(df):,}",
        f"  Días trading    : {trading_days:,}",
        f"  Años            : {years}",
        f"  Velas/día (avg) : {avg_bars:.1f}",
        f"  Precio medio    : {df['close'].mean():.2f}",
        f"  Precio rango    : {df['close'].min():.2f} – {df['close'].max():.2f}",
        f"  Volumen medio   : {df['volume'].mean():,.0f}",
        "─" * 50,
    ]
    return "\n".join(lines)


def _save_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(df, f)
    logger.info(f"Cache guardado en {path}")


def _load_cache(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return pickle.load(f)