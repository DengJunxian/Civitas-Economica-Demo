from __future__ import annotations

import pandas as pd

from core.objective_discovery import ObjectiveDiscoveryEngine, discover_objectives


def _path_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6],
            "close": [3000.0, 3012.0, 2998.0, 3026.0, 3038.0, 3042.0],
            "open": [2995.0, 3000.0, 3012.0, 2998.0, 3026.0, 3038.0],
            "high": [3010.0, 3020.0, 3018.0, 3034.0, 3048.0, 3050.0],
            "low": [2988.0, 2998.0, 2990.0, 2994.0, 3022.0, 3030.0],
            "volume": [1_000_000, 1_100_000, 1_220_000, 1_160_000, 1_300_000, 1_280_000],
            "panic_level": [0.18, 0.22, 0.35, 0.24, 0.20, 0.19],
            "csad": [0.05, 0.07, 0.12, 0.08, 0.06, 0.05],
            "买入量": [600, 640, 560, 700, 720, 710],
            "卖出量": [520, 620, 740, 560, 530, 520],
        }
    )


def _reports():
    return [
        {
            "macro_state": {"liquidity_index": 0.72, "credit_spread": 0.012, "unemployment": 0.05, "wage_growth": 0.035, "inflation": 0.021},
            "microstructure_metrics": {
                "spread_pct": 0.0008,
                "depth_bid_total": 1_000_000.0,
                "depth_ask_total": 900_000.0,
                "cancel_to_trade_ratio": 0.42,
                "slippage_bps": 4.0,
            },
            "realism_diagnostics": {"microstructure_score": 0.78, "liquidity_thinness": 0.18},
            "policy_transmission_chain": {"agent_beliefs": {"dispersion": 0.21}},
        }
    ]


def test_objective_discovery_outputs_ranked_metrics():
    result = ObjectiveDiscoveryEngine().discover(_path_frame(), reports=_reports()).to_dict()

    assert result["schema_version"] == "objective_discovery_v1"
    assert result["ranked_metrics"]
    assert result["top_metrics"]
    assert result["composite_score"] >= 0


def test_objective_discovery_includes_shanghai_index_but_not_only_it():
    result = discover_objectives(_path_frame(), reports=_reports(), top_k=8)
    names = [item["name"] for item in result["ranked_metrics"]]
    top_names = [item["name"] for item in result["top_metrics"]]

    assert "shanghai_index_return" in names
    assert result["shanghai_index_metric"]["name"] == "shanghai_index_return"
    assert any(name != "shanghai_index_return" for name in top_names)


def test_objective_discovery_produces_pareto_frontier():
    result = discover_objectives([_path_frame(), _path_frame().assign(close=lambda frame: frame["close"] * 1.002)], reports=_reports())

    assert result["pareto_frontier"]
    assert all("policy_sensitivity" in row and "robustness" in row for row in result["pareto_frontier"])
    assert result["weight_decomposition"]

