from __future__ import annotations

import json

from core.eval import (
    build_replay_scorecard,
    compute_shanghai_tracking_metrics,
    event_study,
    scorecard_to_json,
)


def test_eval_suite_handles_empty_single_and_invalid_data():
    empty = build_replay_scorecard(
        sim_close=[],
        real_close=[],
        benchmark_symbol="sh000001",
        replay_window={"start": "2024-01-01", "end": "2024-01-02"},
        seed=1,
        config_hash="cfg",
        data_snapshot_hash="snap",
    )
    assert empty.path_fit_metrics["tracking_rmse"] == 0.0
    assert empty.risk_metrics["max_drawdown"] == 0.0
    assert empty.pass_fail_flags["has_hashes"] is True

    single = build_replay_scorecard(
        sim_close=[3000.0, float("nan")],
        real_close=[3000.0],
        benchmark_symbol="sh000001",
        replay_window={"start": "2024-01-01", "end": "2024-01-01"},
        seed=1,
        config_hash="cfg",
        data_snapshot_hash="snap",
    )
    assert single.path_fit_metrics["direction_hit_rate"] == 0.0
    assert single.risk_metrics["volatility"] == 0.0


def test_scorecard_is_deterministic_and_json_serializable():
    kwargs = dict(
        sim_close=[3000.0, 3010.0, 3005.0, 3020.0],
        real_close=[3000.0, 3008.0, 3000.0, 3018.0],
        benchmark_symbol="sh000001",
        replay_window={"start": "2024-09-24", "end": "2024-09-27"},
        seed=7,
        config_hash="cfg",
        data_snapshot_hash="snap",
        event_points=[2],
    )
    left = build_replay_scorecard(**kwargs)
    right = build_replay_scorecard(**kwargs)
    assert left.to_dict() == right.to_dict()
    payload = json.loads(scorecard_to_json(left))
    assert payload["benchmark_symbol"] == "sh000001"
    assert "path_fit_metrics" in payload
    assert "statistical_tests" in payload


def test_shanghai_tracking_metrics_are_independently_testable():
    metrics = compute_shanghai_tracking_metrics(
        sim_close=[100.0, 101.0, 102.0, 101.5],
        real_close=[100.0, 100.8, 101.6, 101.2],
        benchmark_symbol="sh000001",
        event_points=[2],
    )
    assert metrics["tracking_rmse"] >= 0.0
    assert metrics["shanghai_index_path_error"] == metrics["tracking_rmse"]
    assert 0.0 <= metrics["direction_hit_rate"] <= 1.0
    assert "abnormal_return_error" in metrics


def test_event_study_supports_pre_post_comparison():
    result = event_study(
        sim_close=[100.0, 99.5, 99.0, 100.5, 101.0, 101.5],
        real_close=[100.0, 99.6, 99.1, 100.4, 101.2, 101.4],
        event_points=[3],
        pre_window=2,
        post_window=2,
        seed=3,
    )
    assert result["event_count"] == 1
    assert 0.0 <= result["event_window_direction_hit_rate"] <= 1.0
    assert result["abnormal_return_error"] >= 0.0
    assert result["bootstrap_ci"]["valid"] is True


def test_frontend_and_report_layers_can_consume_scorecard_payload():
    scorecard = build_replay_scorecard(
        sim_close=[100.0, 100.5, 101.0],
        real_close=[100.0, 100.4, 100.9],
        benchmark_symbol="sh000001",
        replay_window={"start": "2024-01-01", "end": "2024-01-03"},
        seed=9,
        config_hash="cfg",
        data_snapshot_hash="snap",
        microstructure_metrics={"spread": 0.01, "depth": 1000.0},
        regulatory_metrics={"fairness_compliance": 0.8, "financing_function": 0.7},
    ).to_dict()
    required = {
        "experiment_id",
        "config_hash",
        "data_snapshot_hash",
        "seed",
        "benchmark_symbol",
        "replay_window",
        "path_fit_metrics",
        "risk_metrics",
        "microstructure_metrics",
        "behavioral_metrics",
        "contagion_metrics",
        "regulatory_metrics",
        "pass_fail_flags",
        "narrative_summary",
    }
    assert required <= set(scorecard)
    assert scorecard["path_fit_metrics"]["tracking_rmse"] >= 0.0
