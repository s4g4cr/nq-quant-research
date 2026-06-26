from dataclasses import dataclass, field

@dataclass
class DataConfig:
    ticker: str = "^NDX"
    interval: str = "1h"
    start_date: str = "2021-01-01"
    end_date: str = "2026-06-25"
    timezone: str = "America/New_York"
    session_open: str = "09:30"
    session_close: str = "16:00"
    cache_path: str = "results/data_cache.pkl"

@dataclass
class IndicatorConfig:
    atr_period: int = 20
    volume_period: int = 20

@dataclass
class ORBConfig:
    range_start: str = "09:30"
    range_end: str = "10:30"

@dataclass
class SignalConfig:
    candle_range_multiplier: float = 1.0
    volume_multiplier: float = 1.5

@dataclass
class RiskConfig:
    sl_atr_multiplier: float = 1.0
    tp_atr_multiplier: float = 2.0
    max_bars_in_trade: int = 5
    commission_per_side: float = 2.0
    slippage_ticks: float = 1.0
    tick_value: float = 5.0
    point_value: float = 1.0   # NDX: tratamos cada punto como $1 para normalizar
    contracts: int = 1

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    train_ratio: float = 0.70
    results_dir: str = "results"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    orb: ORBConfig = field(default_factory=ORBConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  NQ ORB BACKTEST — CONFIGURACIÓN",
            "=" * 60,
            f"  Ticker          : {self.data.ticker}",
            f"  Intervalo       : {self.data.interval}",
            f"  Período         : {self.data.start_date} → {self.data.end_date}",
            f"  Sesión          : {self.data.session_open} – {self.data.session_close} NY",
            f"  Opening Range   : {self.orb.range_start} – {self.orb.range_end}",
            f"  ATR período     : {self.indicators.atr_period}",
            f"  Vol. período    : {self.indicators.volume_period}",
            f"  Rango vela min  : {self.signal.candle_range_multiplier}x ATR",
            f"  Volumen mín     : {self.signal.volume_multiplier}x vol. media",
            f"  Stop Loss       : {self.risk.sl_atr_multiplier}x ATR",
            f"  Take Profit     : {self.risk.tp_atr_multiplier}x ATR",
            f"  Max velas       : {self.risk.max_bars_in_trade}",
            f"  Comisión/lado   : ${self.risk.commission_per_side}",
            f"  Capital inicial : ${self.backtest.initial_capital:,.0f}",
            f"  Train/Test      : {int(self.backtest.train_ratio*100)}/{int((1-self.backtest.train_ratio)*100)}%",
            "=" * 60,
        ]
        return "\n".join(lines)

DEFAULT_CONFIG = Config()