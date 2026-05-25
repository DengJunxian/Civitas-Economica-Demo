from __future__ import annotations

import pandas as pd

from ui.policy_lab import _build_regulation_counterfactual_worlds


def _sample_frame() -> pd.DataFrame:
    rows = []
    close = 3000.0
    for step in range(1, 13):
        open_price = close
        close = close * (0.998 if step < 8 else 0.999)
        panic = min(0.9, 0.22 + step * 0.03)
        rows.append(
            {
                "step": step,
                "time": f"2026-01-{step:02d}",
                "open": round(open_price, 2),
                "high": round(max(open_price, close) * 1.002, 2),
                "low": round(min(open_price, close) * 0.998, 2),
                "close": round(close, 2),
                "volume": float(1_000_000 + step * 20_000),
                "csad": float(0.06 + step * 0.004),
                "panic_level": float(panic),
            }
        )
    return pd.DataFrame(rows)


def test_regulation_counterfactual_worlds_have_scorecard_and_recommendation() -> None:
    frame = _sample_frame()
    payload = _build_regulation_counterfactual_worlds(frame, intensity=1.2)

    assert payload["recommended_timing"] in {
        "no_intervention",
        "early_intervention",
        "late_intervention",
    }
    scorecards = payload["scorecards"]
    assert "early_intervention" in scorecards
    assert "late_intervention" in scorecards
    assert "no_intervention" in scorecards

    early_panic = float(scorecards["early_intervention"]["max_panic"])
    base_panic = float(scorecards["no_intervention"]["max_panic"])
    assert early_panic <= base_panic


def test_regulation_counterfactual_worlds_support_integer_price_columns() -> None:
    frame = _sample_frame()
    frame["open"] = frame["open"].round().astype(int)
    frame["high"] = frame["high"].round().astype(int)
    frame["low"] = frame["low"].round().astype(int)
    frame["close"] = frame["close"].round().astype(int)

    payload = _build_regulation_counterfactual_worlds(frame, intensity=1.2)

    worlds = payload["worlds"]
    early_df = pd.DataFrame(worlds["early_intervention"])
    late_df = pd.DataFrame(worlds["late_intervention"])

    assert not early_df.empty
    assert not late_df.empty
    assert early_df["close"].dtype.kind == "f"
    assert late_df["close"].dtype.kind == "f"
