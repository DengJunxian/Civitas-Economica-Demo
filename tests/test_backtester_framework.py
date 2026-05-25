import pandas as pd

from core.backtester import BacktestConfig, HistoricalBacktester


def _mock_daily_frame(n: int = 160) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    price = pd.Series(range(n), dtype=float) * 0.8 + 3000.0
    close = price + (pd.Series(range(n), dtype=float) % 7 - 3) * 0.6
    open_ = close - 2.0
    high = close + 4.0
    low = close - 4.0
    volume = 1_000_000 + (pd.Series(range(n), dtype=float) % 11) * 50_000
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_backtester_runs_with_synthetic_data():
    cfg = BacktestConfig(
        strategy_name="momentum",
        lookback=15,
        rebalance_frequency=5,
        period_days=0,
    )
    bt = HistoricalBacktester(cfg)
    bt.historical_data = _mock_daily_frame(220)
    bt.benchmark_data = bt.historical_data[["date", "close"]].rename(
        columns={"close": "benchmark_close"}
    )

    result = bt.run_backtest()

    assert result.total_days > 100
    assert len(result.equity_curve) == result.total_days
    assert len(result.real_prices) == result.total_days
    assert result.strategy_name == "momentum"
    assert isinstance(result.price_correlation, float)


def test_backtester_export_qlib_bundle(tmp_path):
    cfg = BacktestConfig(
        strategy_name="news_driven",
        lookback=10,
        rebalance_frequency=3,
        period_days=0,
    )
    bt = HistoricalBacktester(cfg)
    bt.historical_data = _mock_daily_frame(180)
    bt.benchmark_data = bt.historical_data[["date", "close"]].rename(
        columns={"close": "benchmark_close"}
    )

    bt.run_backtest()
    output = bt.export_qlib_bundle(str(tmp_path))

    assert (tmp_path / "features.csv").exists()
    assert (tmp_path / "labels.csv").exists()
    assert (tmp_path / "meta.json").exists()
    assert output
