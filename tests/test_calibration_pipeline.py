from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.backtester import BacktestConfig
from core.calibration_pipeline import CalibrationPipeline, CalibrationSpec


def _mock_price_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    returns = rng.normal(0.0006, 0.01, size=n)
    close = 3000.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 - rng.normal(0.0, 0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.002, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.002, size=n)))
    volume = np.clip(rng.normal(1_000_000, 120_000, size=n), 100_000, None)
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


def test_calibration_pipeline_minimal_run(tmp_path):
    frame = _mock_price_frame(120)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(
        symbol="sh000001",
        benchmark_symbol="sh000001",
        period_days=0,
        random_seed=11,
        strategy_name="portfolio_system",
    )
    spec = CalibrationSpec(
        parameter_space={
            "policy_shock": (-0.2, 0.2),
            "rebalance_frequency": [3, 5, 10],
            "max_position": [0.8, 1.0],
        },
        method="bayesian",
        bayes_init=2,
        bayes_iterations=3,
        bayes_patience=3,
        random_seed=11,
        dataset_snapshot_id="snap-calib-test",
        output_dir=str(Path(tmp_path) / "calibration"),
    )
    pipeline = CalibrationPipeline(
        base_config=cfg,
        spec=spec,
        historical_data=frame,
        benchmark_data=benchmark,
    )
    result = pipeline.run()
    assert isinstance(result.best_config, dict)
    assert "calibration_manifest" in result.artifacts
    assert Path(result.artifacts["calibration_manifest"]).exists()
    assert Path(result.artifacts["artifact_root"]).name == result.manifest["run_id"]
    assert Path(result.artifacts["objective_breakdown"]).exists()
    assert Path(result.artifacts["radar_chart_json"]).exists()
    assert Path(result.artifacts["bar_chart_csv"]).exists()
    assert "microstructure_fit" in result.holdout_report
    assert result.manifest["model_version"] == cfg.strategy_name


def test_calibration_pipeline_bayesian_uses_gp_backend_by_default(tmp_path):
    frame = _mock_price_frame(90)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(
        symbol="sh000001",
        benchmark_symbol="sh000001",
        period_days=0,
        random_seed=7,
        strategy_name="portfolio_system",
    )
    spec = CalibrationSpec(
        parameter_space={
            "policy_shock": (-0.2, 0.2),
            "rebalance_frequency": [3, 5, 10],
            "max_position": [0.8, 1.0],
        },
        method="bayesian",
        bayes_init=3,
        bayes_iterations=4,
        bayes_patience=4,
        random_seed=7,
        feature_flags={"calibration_gp_bo_v1": True},
        dataset_snapshot_id="snap-calib-gp",
        output_dir=str(Path(tmp_path) / "calibration_gp"),
    )
    pipeline = CalibrationPipeline(
        base_config=cfg,
        spec=spec,
        historical_data=frame,
        benchmark_data=benchmark,
    )
    result = pipeline.run()
    assert result.method == "bayesian_gp"
    assert result.records


def test_calibration_pipeline_can_fallback_to_bayesian_like_backend(tmp_path):
    frame = _mock_price_frame(90)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(
        symbol="sh000001",
        benchmark_symbol="sh000001",
        period_days=0,
        random_seed=8,
        strategy_name="portfolio_system",
    )
    spec = CalibrationSpec(
        parameter_space={
            "policy_shock": (-0.2, 0.2),
            "rebalance_frequency": [3, 5, 10],
            "max_position": [0.8, 1.0],
        },
        method="bayesian",
        bayes_init=2,
        bayes_iterations=3,
        bayes_patience=3,
        random_seed=8,
        feature_flags={"calibration_gp_bo_v1": False},
        dataset_snapshot_id="snap-calib-like",
        output_dir=str(Path(tmp_path) / "calibration_like"),
    )
    pipeline = CalibrationPipeline(
        base_config=cfg,
        spec=spec,
        historical_data=frame,
        benchmark_data=benchmark,
    )
    result = pipeline.run()
    assert result.method == "bayesian_like"
    assert result.records


def test_calibration_pipeline_rolling_window_and_5_objective_weights(tmp_path):
    frame = _mock_price_frame(150)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(
        symbol="sh000001",
        benchmark_symbol="sh000001",
        period_days=0,
        random_seed=13,
        strategy_name="portfolio_system",
    )
    spec = CalibrationSpec(
        parameter_space={
            "policy_shock": (-0.1, 0.1),
            "rebalance_frequency": [3, 5],
            "max_position": [0.8, 1.0],
        },
        objective_weights={
            "path_fit": 0.20,
            "vol_fit": 0.20,
            "turnover_fit": 0.10,
            "micro_fit": 0.25,
            "stylized_fit": 0.25,
        },
        method="bayesian_like",
        bayes_init=2,
        bayes_iterations=2,
        bayes_patience=2,
        random_seed=13,
        rolling_window=True,
        dataset_snapshot_id="snap-calib-rolling",
        output_dir=str(Path(tmp_path) / "calibration_rolling"),
    )
    weights = spec.normalized_weights()
    assert set(weights.keys()) == {
        "path_fit",
        "volatility_fit",
        "turnover_fit",
        "microstructure_fit",
        "stylized_facts_fit",
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-9

    pipeline = CalibrationPipeline(
        base_config=cfg,
        spec=spec,
        historical_data=frame,
        benchmark_data=benchmark,
    )
    result = pipeline.run()

    assert result.records
    assert result.records[0]["rolling_windows_evaluated"] >= 2
    assert "microstructure_fit" in result.records[0]
    assert result.holdout_report["rolling_windows_evaluated"] >= 2


def test_calibration_pipeline_legacy_artifact_layout_flag(tmp_path):
    frame = _mock_price_frame(100)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(
        symbol="sh000001",
        benchmark_symbol="sh000001",
        period_days=0,
        random_seed=5,
        strategy_name="portfolio_system",
    )
    output_dir = Path(tmp_path) / "legacy_layout"
    spec = CalibrationSpec(
        parameter_space={
            "policy_shock": (-0.1, 0.1),
            "rebalance_frequency": [3, 5],
            "max_position": [0.8, 1.0],
        },
        method="bayesian_like",
        bayes_init=2,
        bayes_iterations=2,
        bayes_patience=2,
        random_seed=5,
        dataset_snapshot_id="snap-calib-legacy",
        output_dir=str(output_dir),
        feature_flags={"calibration_legacy_artifact_layout": True},
    )
    pipeline = CalibrationPipeline(
        base_config=cfg,
        spec=spec,
        historical_data=frame,
        benchmark_data=benchmark,
    )
    result = pipeline.run()
    assert Path(result.artifacts["artifact_root"]) == output_dir
