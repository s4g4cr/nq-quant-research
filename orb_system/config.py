"""
Centralized configuration for the NQ ORB Backtest System.
All parameters are defined here to avoid magic numbers scattered in the codebase.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading and caching configuration."""
    csv_path: str = r"c:\Users\samue\nq_orb_backtest\glbx-mdp3-20210625-20260624.ohlcv-1m.csv"
    timezone: str = "America/New_York"
    session_open: str = "09:30"
    session_close: str = "16:00"
    cache_path: str = r"c:\Users\samue\nq_orb_backtest\orb_system\data\nq_continuous_1m.pkl"

    # NQ roll schedule: symbol -> (start_date, end_date) inclusive/exclusive
    roll_schedule: dict = field(default_factory=lambda: {
        "NQU1": ("2021-06-01", "2021-09-10"),
        "NQZ1": ("2021-09-10", "2021-12-10"),
        "NQH2": ("2021-12-10", "2022-03-11"),
        "NQM2": ("2022-03-11", "2022-06-10"),
        "NQU2": ("2022-06-10", "2022-09-09"),
        "NQZ2": ("2022-09-09", "2022-12-09"),
        "NQH3": ("2022-12-09", "2023-03-10"),
        "NQM3": ("2023-03-10", "2023-06-09"),
        "NQU3": ("2023-06-09", "2023-09-15"),
        "NQZ3": ("2023-09-15", "2023-12-15"),
        "NQH4": ("2023-12-15", "2024-03-15"),
        "NQM4": ("2024-03-15", "2024-06-14"),
        "NQU4": ("2024-06-14", "2024-09-13"),
        "NQZ4": ("2024-09-13", "2024-12-13"),
        "NQH5": ("2024-12-13", "2025-03-14"),
        "NQM5": ("2025-03-14", "2025-06-13"),
        "NQU5": ("2025-06-13", "2025-09-12"),
        "NQZ5": ("2025-09-12", "2025-12-12"),
        "NQH6": ("2025-12-12", "2026-03-13"),
        "NQM6": ("2026-03-13", "2026-12-31"),
    })


@dataclass
class IndicatorConfig:
    """Technical indicator parameters."""
    atr_period: int = 20
    volume_period: int = 20


@dataclass
class ORBConfig:
    """Opening Range Breakout range definition."""
    range_start: str = "09:30"
    range_end: str = "10:30"


@dataclass
class SignalConfig:
    """Entry signal filters."""
    candle_range_multiplier: float = 1.0
    volume_multiplier: float = 1.5
    direction_filter: str = "both"              # "long", "short", "both"
    # OR width filter (scale note: OR/ATR_1min ~ 15-25x; use ~20 to drop widest 20% of sessions)
    or_width_filter: bool = False
    or_width_max_atr_mult: float = 20.0
    # Trend filter
    use_trend_filter: bool = False
    trend_period: int = 50
    # Previous-session direction filter
    use_prev_session_filter: bool = False
    # Gap-of-open filter
    use_gap_filter: bool = False
    gap_min_points: float = 0.0             # minimum gap size to act; 0 = any non-zero gap
    # OR-position filter
    use_or_position_filter: bool = False
    or_position_long_min: float = 0.6       # LONG valid only if position_in_OR >= this
    or_position_short_max: float = 0.4      # SHORT valid only if position_in_OR <= this


@dataclass
class RiskConfig:
    """Position sizing, stops, and cost model."""
    sl_atr_multiplier: float = 1.0
    tp_atr_multiplier: float = 2.0
    max_bars_in_trade: int = 120            # 2 hours in 1-min bars
    use_trailing_exit: bool = False
    trailing_atr_mult: float = 1.0          # trail = peak ∓ mult * ATR_at_entry
    trailing_activation_atr_mult: float = 0.5   # min unrealised profit (x ATR) to arm trail
    eod_exit_time: str = "15:45"
    point_value: float = 20.0
    tick_size: float = 0.25
    commission_per_side: float = 2.0
    slippage_ticks: int = 1
    contracts: int = 1


@dataclass
class BacktestConfig:
    """Backtest execution parameters."""
    initial_capital: float = 100_000.0
    train_ratio: float = 0.70


@dataclass
class MRConfig:
    """Mean Reversion strategy parameters (Phase 4)."""
    entry_pct: float          = 0.30   # trigger = extreme + pct * or_range
    sl_atr_mult: float        = 0.50   # SL = entry +/- sl_atr_mult * ATR
    max_bars: int             = 120
    use_prev_session: bool    = False
    use_or_width: bool        = False
    or_width_max_mult: float  = 1.50   # skip if or_range > mult * ATR


@dataclass
class HMMConfig:
    """HMM regime filter parameters (Phase 5)."""
    n_states: int             = 3
    feature_window: int       = 20    # rolling z-score window (days)
    random_state: int         = 42


@dataclass
class Config:
    """Master configuration object."""
    data: DataConfig = field(default_factory=DataConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    orb: ORBConfig = field(default_factory=ORBConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    mr: MRConfig = field(default_factory=MRConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)

    def summary(self) -> None:
        """Print full configuration summary."""
        sep = "=" * 60
        print(sep)
        print("NQ ORB BACKTEST SYSTEM — CONFIGURATION SUMMARY")
        print(sep)

        print("\n[DATA]")
        print(f"  CSV path:       {self.data.csv_path}")
        print(f"  Timezone:       {self.data.timezone}")
        print(f"  Session:        {self.data.session_open} - {self.data.session_close}")
        print(f"  Cache path:     {self.data.cache_path}")

        print("\n[INDICATORS]")
        print(f"  ATR period:     {self.indicators.atr_period}")
        print(f"  Volume period:  {self.indicators.volume_period}")

        print("\n[OPENING RANGE]")
        print(f"  Range start:    {self.orb.range_start}")
        print(f"  Range end:      {self.orb.range_end} (exclusive)")

        print("\n[SIGNAL FILTERS]")
        print(f"  Candle range:   > {self.signal.candle_range_multiplier}x ATR")
        print(f"  Volume:         > {self.signal.volume_multiplier}x avg volume")
        print(f"  Direction:      {self.signal.direction_filter}")
        print(f"  OR width filter:{self.signal.or_width_filter}  (max {self.signal.or_width_max_atr_mult}x prev ATR)")
        print(f"  Prev session:   {self.signal.use_prev_session_filter}")
        print(f"  Gap filter:     {self.signal.use_gap_filter}  (min {self.signal.gap_min_points} pts)")
        print(f"  OR pos filter:  {self.signal.use_or_position_filter}  "
              f"(L>={self.signal.or_position_long_min} / S<={self.signal.or_position_short_max})")
        print(f"  Trend filter:   {self.signal.use_trend_filter}  (SMA{self.signal.trend_period})")

        print("\n[RISK]")
        print(f"  Stop Loss:      {self.risk.sl_atr_multiplier}x ATR")
        print(f"  Take Profit:    {self.risk.tp_atr_multiplier}x ATR")
        print(f"  Max bars:       {self.risk.max_bars_in_trade} (1-min bars)")
        print(f"  Trailing exit:  {self.risk.use_trailing_exit}  "
              f"(trail={self.risk.trailing_atr_mult}x / arm@{self.risk.trailing_activation_atr_mult}x ATR)")
        print(f"  EOD exit:       {self.risk.eod_exit_time} NY")
        print(f"  Point value:    ${self.risk.point_value}/pt")
        print(f"  Tick size:      {self.risk.tick_size}")
        print(f"  Commission:     ${self.risk.commission_per_side}/side")
        print(f"  Slippage:       {self.risk.slippage_ticks} tick(s)")
        print(f"  Contracts:      {self.risk.contracts}")

        print("\n[BACKTEST]")
        print(f"  Initial capital: ${self.backtest.initial_capital:,.0f}")
        print(f"  Train ratio:     {self.backtest.train_ratio:.0%}")
        print(f"  Test ratio:      {1 - self.backtest.train_ratio:.0%}")
        print(sep)
