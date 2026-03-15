import pandas as pd

from core.backtest_robustness import RobustnessAnalyzer
from core.backtester import BacktestConfig


def _mock_daily_frame(n: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    base = 3000.0 + pd.Series(range(n), dtype=float) * 0.45
    close = base + (pd.Series(range(n), dtype=float) % 9 - 4) * 0.7
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 2.0,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": 1_200_000 + (pd.Series(range(n), dtype=float) % 13) * 65_000,
        }
    )


def test_grid_search_returns_all_combinations():
    frame = _mock_daily_frame(280)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(period_days=0, strategy_name="momentum")

    analyzer = RobustnessAnalyzer(cfg, historical_data=frame, benchmark_data=benchmark)
    grid = analyzer.run_grid_search(
        {
            "lookback": [10, 20],
            "rebalance_frequency": [3, 5],
            "signal_threshold": [0.03, 0.06],
        },
        optimize_metric="score",
    )

    assert len(grid.records) == 8
    assert set(grid.best_params.keys()) == {"lookback", "rebalance_frequency", "signal_threshold"}
    assert isinstance(grid.best_score, float)


def test_walk_forward_generates_folds():
    frame = _mock_daily_frame(360)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(period_days=0, strategy_name="risk_parity")

    analyzer = RobustnessAnalyzer(cfg, historical_data=frame, benchmark_data=benchmark)
    wf = analyzer.run_walk_forward(
        param_grid={
            "lookback": [12, 18],
            "rebalance_frequency": [3, 5],
            "signal_threshold": [0.03],
        },
        train_days=126,
        test_days=63,
        step_days=63,
        optimize_metric="sharpe_ratio",
    )

    assert len(wf.folds) >= 2
    assert isinstance(wf.avg_test_return, float)
    assert isinstance(wf.avg_test_sharpe, float)
    assert 0.0 <= wf.pass_rate <= 1.0
