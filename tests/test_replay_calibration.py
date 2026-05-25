from __future__ import annotations

import numpy as np
import pandas as pd

from core.calibration.losses import calibration_loss
from core.calibration.parameter_store import ParameterSet, ParameterStore
from core.calibration.replay_runner import ReplayConfig, ReplayRunner


class _FixtureMarketProvider:
    def get_ohlcv(self, query, use_cache=True, freeze_snapshot=False):
        dates = pd.bdate_range(start=query.start, end=query.end)
        if len(dates) == 0:
            dates = pd.bdate_range(start="2024-01-01", periods=20)
        idx = np.arange(len(dates), dtype=float)
        shock = np.where((idx > len(idx) * 0.45) & (idx < len(idx) * 0.60), -0.004, 0.0015)
        close = 3000.0 * np.cumprod(1.0 + shock)
        return pd.DataFrame(
            {
                "datetime": dates.strftime("%Y-%m-%d 00:00:00"),
                "date": dates.strftime("%Y-%m-%d"),
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": 1_000_000 + idx * 1000,
                "amount": close * (1_000_000 + idx * 1000),
                "symbol": query.symbol,
                "interval": "1d",
                "provider": "unit_fixture",
                "adjust": "",
                "market": "CN",
            }
        )


def test_replay_runner_outputs_scorecard_hashes_for_required_windows():
    runner = ReplayRunner(market_provider=_FixtureMarketProvider())
    windows = [
        ("2024-01-02", "2024-02-09"),
        ("2024-09-24", "2024-10-31"),
        ("2025-02-05", "2025-03-14"),
    ]
    for start, end in windows:
        result = runner.run(ReplayConfig(start=start, end=end, seed=7))
        assert result.seed == 7
        assert result.config_hash
        assert result.data_snapshot_hash
        assert result.scorecard.benchmark_symbol == "sh000001"
        assert result.scorecard.pass_fail_flags["has_hashes"]
        assert "direction_hit_rate" in result.scorecard.path_fit_metrics
        assert result.loss >= 0.0


def test_in_sample_and_out_of_sample_direction_beats_random_baseline():
    runner = ReplayRunner(market_provider=_FixtureMarketProvider())
    in_sample = runner.run(ReplayConfig(start="2024-01-02", end="2024-03-29", seed=21))
    out_sample = runner.run(ReplayConfig(start="2024-04-01", end="2024-05-31", seed=21))
    assert in_sample.scorecard.path_fit_metrics["direction_hit_rate"] >= 0.50
    assert out_sample.scorecard.path_fit_metrics["direction_hit_rate"] >= 0.50


def test_parameter_perturbation_does_not_create_catastrophic_instability():
    runner = ReplayRunner(market_provider=_FixtureMarketProvider())
    result = runner.run(
        ReplayConfig(
            start="2024-01-02",
            end="2024-03-29",
            seed=9,
            theta={
                "trend_following": 1.20,
                "mean_reversion": -0.20,
                "macro_sensitivity": 1.00,
                "volatility_scale": 1.50,
                "liquidity_sensitivity": 0.40,
            },
        )
    )
    flags = result.scorecard.pass_fail_flags
    assert flags["volatility_not_catastrophic"]
    assert flags["drawdown_not_catastrophic"]
    assert calibration_loss(sim_close=result.simulated["close"], real_close=result.real["close"]) >= 0.0


def test_parameter_store_hashes_inputs(tmp_path):
    store = ParameterStore(root_dir=tmp_path)
    params = ParameterSet(name="theta_test", params={"trend_following": 0.5}, metadata={"window": "stable"})
    path = store.save(params)
    loaded = store.load("theta_test")
    assert path.exists()
    assert loaded.parameter_hash == params.parameter_hash
    assert loaded.metadata["window"] == "stable"
