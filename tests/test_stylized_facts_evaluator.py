import numpy as np

from core.behavioral_finance import StylizedFactsEvaluator


def _sample_inputs():
    real_prices = [100.0, 101.2, 102.1, 101.5, 103.0, 103.8, 104.5, 103.9, 105.1]
    simulated_prices = [100.0, 100.9, 101.8, 101.2, 102.6, 103.4, 104.2, 103.6, 104.8]
    real_volumes = [1000, 1020, 980, 1110, 1090, 1130, 1160, 1145, 1185]
    simulated_volumes = [990, 1010, 995, 1095, 1085, 1125, 1150, 1135, 1175]
    order_signs = [1, 1, 1, -1, 1, 1, 1, -1, 1]
    trade_sizes = [10, 12, 11, 14, 13, 12, 15, 10, 16]
    market_returns = np.diff(real_prices) / np.array(real_prices[:-1], dtype=float)
    cross_sectional_returns = [[float(r * 0.9), float(r * 1.1), float(r * 1.05)] for r in market_returns]
    timestamps = [f"2024-01-0{i+1}" for i in range(len(real_prices))]
    return {
        "real_prices": real_prices,
        "simulated_prices": simulated_prices,
        "real_volumes": real_volumes,
        "simulated_volumes": simulated_volumes,
        "order_signs": order_signs,
        "trade_sizes": trade_sizes,
        "market_returns": market_returns,
        "cross_sectional_returns": cross_sectional_returns,
        "timestamps": timestamps,
    }


def test_stylized_facts_evaluator_builds_three_layer_report_and_reproducible_hash() -> None:
    inputs = _sample_inputs()
    evaluator = StylizedFactsEvaluator(feature_flag=True, seed=7, config={"window": 8, "mode": "agent_replay"})

    report = evaluator.evaluate(**inputs)
    payload = report.to_dict()

    assert report.feature_flag is True
    assert report.path_fit["enabled"] is True
    assert report.microstructure_fit["enabled"] is True
    assert report.behavioral_fit["enabled"] is True
    assert 0.0 <= report.credibility_score <= 1.0
    assert len(report.config_hash) == 64
    assert payload["reproducibility"]["seed"] == 7
    assert payload["snapshot_info"]["real_points"] == len(inputs["real_prices"])
    assert "price_correlation" in payload["metrics"]
    assert "turning_point_f1" in payload["metrics"]
    assert len(payload["charts"]) >= 3
    assert 0.0 <= payload["microstructure_fit"]["score"] <= 1.0
    assert 0.0 <= payload["behavioral_fit"]["score"] <= 1.0


def test_stylized_facts_evaluator_feature_flag_disables_extended_layers() -> None:
    inputs = _sample_inputs()
    evaluator = StylizedFactsEvaluator(feature_flag=False, seed=11, config={"window": 4})

    report = evaluator.evaluate(**inputs)

    assert report.path_fit["enabled"] is True
    assert report.microstructure_fit["enabled"] is False
    assert report.behavioral_fit["enabled"] is False
    assert len(report.charts) == 1
    assert 0.0 <= report.credibility_score <= 1.0
