"""
Thin wrapper for ORB SHORT strategy with Phase 5 optimal parameters.
Delegates to BacktestEngine — do not modify engine.py.
"""
from orb_system.backtester.engine import BacktestEngine, Results
from orb_system.config import Config


# Phase 5 optimal parameters (from Phase 3 walk-forward)
_ORB_SHORT_PARAMS = dict(
    direction_filter        = "short",
    sl_atr_multiplier       = 0.75,
    tp_atr_multiplier       = 2.0,
    candle_range_multiplier = 1.5,
    volume_multiplier       = 1.2,
    max_bars_in_trade       = 120,
    use_trailing_exit       = False,
    # All context filters off: HMM handles regime selection
    or_width_filter         = False,
    use_prev_session_filter = False,
    use_gap_filter          = False,
    use_or_position_filter  = False,
    use_trend_filter        = False,
)


def make_orb_short_config(**overrides) -> Config:
    cfg = Config()
    params = {**_ORB_SHORT_PARAMS, **overrides}

    # risk params
    for k in ("sl_atr_multiplier", "tp_atr_multiplier", "max_bars_in_trade", "use_trailing_exit"):
        if k in params:
            setattr(cfg.risk, k, params[k])

    # signal params
    for k in (
        "direction_filter", "candle_range_multiplier", "volume_multiplier",
        "or_width_filter", "use_prev_session_filter", "use_gap_filter",
        "use_or_position_filter", "use_trend_filter",
    ):
        if k in params:
            setattr(cfg.signal, k, params[k])

    return cfg


def run(df, label: str = "", **overrides) -> Results:
    cfg    = make_orb_short_config(**overrides)
    engine = BacktestEngine(cfg)
    return engine.run(df, label=label)
