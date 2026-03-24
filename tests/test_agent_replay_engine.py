import numpy as np
import pandas as pd

from core.agent_replay import AgentReplayEngine
from core.backtester import BacktestConfig


def _mock_daily_frame(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    base = 100.0 + np.linspace(0.0, 6.0, n)
    wave = np.sin(np.arange(n) / 3.0) * 0.4
    close = base + wave
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000 + np.arange(n) * 5_000,
        }
    )


def test_agent_replay_uses_trade_tape_for_simulated_prices():
    cfg = BacktestConfig(
        strategy_name="portfolio_system",
        lookback=12,
        rebalance_frequency=4,
        period_days=0,
        random_seed=7,
        feature_flags={"agent_replay": True},
    )
    engine = AgentReplayEngine(cfg)
    frame = _mock_daily_frame(70)
    engine.historical_data = frame
    engine.benchmark_data = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})

    result = engine.run_backtest()

    assert result.total_days == len(frame)
    assert result.simulated_bars
    assert len(result.simulated_bars) == result.total_days
    assert result.simulated_prices == [bar["close"] for bar in result.simulated_bars]
    assert result.metadata["replay_mode"] == "agent"
    assert result.metadata["simulated_price_source"] == "trade_tape_close"
    assert result.metadata["data_snapshot"]["rows"] == len(frame)
    assert result.metadata["config_hash"]
    assert result.trade_log


def test_agent_replay_is_reproducible_with_fixed_seed():
    cfg = BacktestConfig(
        strategy_name="portfolio_system",
        lookback=12,
        rebalance_frequency=4,
        period_days=0,
        random_seed=11,
        feature_flags={"agent_replay": True},
    )
    frame = _mock_daily_frame(50)

    engine_a = AgentReplayEngine(cfg)
    engine_a.historical_data = frame
    engine_a.benchmark_data = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    result_a = engine_a.run_backtest()

    engine_b = AgentReplayEngine(cfg)
    engine_b.historical_data = frame
    engine_b.benchmark_data = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    result_b = engine_b.run_backtest()

    assert result_a.simulated_prices == result_b.simulated_prices
    assert result_a.metadata["config_hash"] == result_b.metadata["config_hash"]
