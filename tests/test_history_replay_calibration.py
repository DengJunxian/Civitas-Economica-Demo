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
    assert profile["anchor_points"] == 10
    assert profile["large_bias_points"] == 6

    calibrated = np.asarray(result.simulated_prices, dtype=float)
    real = np.asarray(real_prices, dtype=float)
    rel_dev = np.abs(calibrated[1:] - real[1:]) / np.maximum(real[1:], 1e-9)
    assert float(np.max(rel_dev)) >= 0.02
    assert int(np.sum(rel_dev >= 0.015)) >= 6
