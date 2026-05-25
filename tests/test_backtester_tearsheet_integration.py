import pandas as pd

from core.backtester import BacktestConfig, BacktestReportGenerator, HistoricalBacktester


def _mock_daily_frame(n: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    base = 3000.0 + pd.Series(range(n), dtype=float) * 0.6
    close = base + (pd.Series(range(n), dtype=float) % 9 - 4) * 0.7
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 2.0,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": 900_000 + (pd.Series(range(n), dtype=float) % 11) * 40_000,
        }
    )


def test_backtester_extended_metrics_and_tearsheet_export(tmp_path):
    cfg = BacktestConfig(
        strategy_name="portfolio_system",
        lookback=12,
        rebalance_frequency=4,
        period_days=0,
    )
    bt = HistoricalBacktester(cfg)
    bt.historical_data = _mock_daily_frame(220)
    bt.benchmark_data = bt.historical_data[["date", "close"]].rename(columns={"close": "benchmark_close"})

    result = bt.run_backtest()
    assert result.total_days > 100
    assert isinstance(result.var_95, float)
    assert isinstance(result.cvar_95, float)
    assert isinstance(result.credibility_score, float)
    assert 0.0 <= result.credibility_score <= 1.0

    payload = BacktestReportGenerator.build_tear_sheet_payload(result, scenario_name="policy_A")
    assert payload["scenario_name"] == "policy_A"
    assert "metrics" in payload and "sharpe_ratio" in payload["metrics"]

    exported = BacktestReportGenerator.export_tear_sheet_files(result, str(tmp_path), scenario_name="policy_A")
    assert (tmp_path / "policy_A_tearsheet.json").exists()
    assert (tmp_path / "policy_A_tearsheet.html").exists()
    assert set(exported.keys()) == {"json", "html"}

