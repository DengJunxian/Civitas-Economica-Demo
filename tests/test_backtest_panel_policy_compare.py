import pandas as pd

from core.backtester import BacktestConfig, HistoricalBacktester
from ui.backtest_panel import _run_policy_ab_comparison


def _mock_daily_frame(n: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    trend = 3100.0 + pd.Series(range(n), dtype=float) * 0.5
    close = trend + (pd.Series(range(n), dtype=float) % 13 - 6) * 0.8
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 2.2,
            "high": close + 3.5,
            "low": close - 3.5,
            "close": close,
            "volume": 1_000_000 + (pd.Series(range(n), dtype=float) % 17) * 60_000,
        }
    )


def test_policy_ab_compare_generates_outputs(tmp_path):
    cfg = BacktestConfig(strategy_name="portfolio_system", period_days=0, policy_shock=0.0)
    bt = HistoricalBacktester(cfg)
    bt.historical_data = _mock_daily_frame()
    bt.benchmark_data = bt.historical_data[["date", "close"]].rename(columns={"close": "benchmark_close"})
    bt.run_backtest()

    bundle = _run_policy_ab_comparison(
        base_backtester=bt,
        policy_a=-0.30,
        policy_b=0.30,
        output_root=str(tmp_path),
    )
    compare_df = bundle["compare_df"]
    assert not compare_df.empty
    assert set(compare_df["scenario"].tolist()) == {"policy_A_-0.30", "policy_B_+0.30"}

    files = bundle["files"]
    assert tmp_path.joinpath(bundle["run_id"], "policy_A_tearsheet.json").exists()
    assert tmp_path.joinpath(bundle["run_id"], "policy_B_tearsheet.html").exists()
    assert files["compare_csv"].endswith(".csv")

