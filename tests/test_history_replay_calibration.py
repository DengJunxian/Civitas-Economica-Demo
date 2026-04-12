from __future__ import annotations

import numpy as np

from core.backtester import BacktestResult
from ui.history_replay import _apply_moderate_calibration


def test_moderate_calibration_uses_mixed_profile_and_large_bias_points() -> None:
    real_prices = [3000.0 + i * 6.0 for i in range(31)]
    simulated_prices = [3000.0 + i * 7.8 + ((-1) ** i) * 8.0 for i in range(31)]
    result = BacktestResult(
        strategy_name="history_replay",
        real_prices=real_prices,
        simulated_prices=simulated_prices,
        simulated_bars=[{"open": p, "high": p, "low": p, "close": p, "volume": 1_000.0} for p in simulated_prices],
        metadata={},
    )

    _apply_moderate_calibration(result)

    profile = dict(result.metadata.get("calibration_mix_profile", {}) or {})
    assert profile["total_adjusted_points"] == 30
    assert 0.60 <= float(profile["achieved_direction_match"]) <= 0.72
    assert 0.60 <= float(profile["target_direction_match"]) <= 0.72
    assert int(profile["anchor_points"]) >= 1
    assert int(profile["local_heterogeneous_points"]) >= 3

    calibrated = np.asarray(result.simulated_prices, dtype=float)
    real = np.asarray(real_prices, dtype=float)
    real_ret = np.diff(real) / np.maximum(real[:-1], 1e-9)
    sim_ret = np.diff(calibrated) / np.maximum(calibrated[:-1], 1e-9)
    sign_match = float(np.mean(np.sign(real_ret) == np.sign(sim_ret)))
    assert 0.60 <= sign_match <= 0.72
    assert abs(float(profile["achieved_direction_match"]) - sign_match) <= 0.05

    rel_dev = np.abs(calibrated[1:] - real[1:]) / np.maximum(real[1:], 1e-9)
    assert float(np.max(rel_dev)) >= 0.02
    assert int(np.sum(rel_dev >= 0.006)) >= 10
    assert float(np.median(rel_dev)) <= 0.10

    bars = result.simulated_bars
    assert bars
    intraday_span = np.asarray(
        [
            (float(bar["high"]) - float(bar["low"])) / max(float(bar["close"]), 1e-9)
            for bar in bars
        ],
        dtype=float,
    )
    assert float(np.median(intraday_span)) >= 0.002

    shape_meta = dict(result.metadata.get("presentation_shape_optimization", {}) or {})
    assert shape_meta.get("enabled") is True
    assert shape_meta.get("mode") == "interval_morphology_on_agent_simulation_base"
    assert int(shape_meta.get("interval_count", 0)) >= 1
