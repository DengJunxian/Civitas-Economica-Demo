from __future__ import annotations

import pandas as pd

from ui.demo_wind_tunnel import _counterfactual_world


def test_counterfactual_world_accepts_integer_close_without_dtype_errors() -> None:
    metrics = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "close": [100, 98, 95, 93],
            "panic_level": [0.10, 0.30, 0.55, 0.80],
        }
    )

    world_b = _counterfactual_world(metrics)

    assert len(world_b) == len(metrics)
    assert world_b["close"].dtype.kind == "f"
    assert (world_b["close"] >= 82.0).all()
    assert metrics["close"].dtype.kind in {"i", "u"}
    assert metrics["close"].tolist() == [100, 98, 95, 93]
