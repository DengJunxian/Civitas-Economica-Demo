from __future__ import annotations

import pandas as pd

from core.ui_text import localize_dataframe_columns
from core.runtime_mode import resolve_runtime_mode_profile
from ui.dashboard import normalize_discovered_metrics_payload
from ui.policy_lab import (
    _build_regulation_counterfactual_worlds,
    _policy_session_advance,
    _policy_session_enqueue_runtime_event,
    _policy_session_new,
    _policy_session_report_payload,
)
from ui.regulator_optimization import _build_regulator_result_frames


def _reference_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    close = pd.Series([3000.0, 3005.0, 3011.0, 3008.0, 3020.0, 3024.0])
    return pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "open": close - 3.0,
            "high": close + 5.0,
            "low": close - 6.0,
            "close": close,
            "volume": [1_000_000 + i * 20_000 for i in range(len(close))],
        }
    )


def test_policy_lab_timeline_payload_contains_runtime_events():
    profile = resolve_runtime_mode_profile("SMART")
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="下调印花税并释放流动性",
        policy_type="市场稳定",
        total_days=6,
        intensity=1.0,
        effective_day=1,
        half_life_days=20,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=profile,
    )
    session["status"] = "running"
    event = _policy_session_enqueue_runtime_event(
        session,
        event_type="major_news",
        title="重大新闻",
        raw_text="盘中出现重大新闻冲击",
        effective_day=1,
        strength=1.0,
        half_life_days=5,
    )
    _policy_session_advance(session, 1)
    payload = _policy_session_report_payload(session, profile)

    assert event["event_type"] == "major_news"
    assert payload["runtime_events"]["timeline"]
    assert payload["runtime_events"]["digest"]["active_count"] >= 1
    assert payload["discovered_metrics"]["schema_version"] == "objective_discovery_v1"


def test_dashboard_supports_discovered_metrics_schema():
    payload = normalize_discovered_metrics_payload(
        {
            "ranked_metrics": [
                {"name": "shanghai_index_return", "rank_score": 0.6, "relation_to_shanghai_index": "self"},
                {"name": "microstructure_score", "rank_score": 0.7, "relation_to_shanghai_index": "complement"},
            ],
            "pareto_frontier": [{"name": "microstructure_score", "policy_sensitivity": 0.8, "robustness": 0.7}],
            "composite_score": 0.62,
            "weight_decomposition": {"microstructure_score": 0.55},
            "candidate_pool": ["shanghai_index_return", "microstructure_score"],
        }
    )

    assert payload["schema_version"] == "objective_discovery_v1"
    assert payload["top_metrics"]
    assert payload["shanghai_index_metric"]["name"] == "shanghai_index_return"
    assert payload["composite_score"] == 0.62


def test_localized_dataframe_does_not_mutate_internal_keys():
    frame = pd.DataFrame(
        [
            {
                "name": "shanghai_index_return",
                "rank_score": 0.6,
                "relation_to_shanghai_index": "self",
            }
        ]
    )
    localized = localize_dataframe_columns(frame)

    assert list(frame.columns) == ["name", "rank_score", "relation_to_shanghai_index"]
    assert "rank_score" in frame.columns
    assert "综合排序分" in localized.columns
    assert localized.iloc[0]["与上证指数关系"] == "核心基准"


def test_counterfactual_internal_world_keys_remain_stable_after_display_localization():
    source = _reference_frame()
    source["step"] = range(1, len(source) + 1)
    source["panic_level"] = [0.2, 0.25, 0.3, 0.45, 0.5, 0.55]
    source["csad"] = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    payload = _build_regulation_counterfactual_worlds(source, intensity=1.0)

    assert {"no_intervention", "early_intervention", "late_intervention"} <= set(payload["worlds"].keys())
    score_frame = pd.DataFrame(
        [{"world": key, **value} for key, value in payload["scorecards"].items()]
    )
    localized = localize_dataframe_columns(score_frame)

    assert "world" in score_frame.columns
    assert "世界线" in localized.columns
    assert set(score_frame["world"]) == {"no_intervention", "early_intervention", "late_intervention"}


def test_regulator_frame_builder_keeps_internal_schema_before_display_localization():
    result = {
        "counterfactual_ab": {
            "baseline": {"macro_stability": 0.5, "intervention_cost": 0.1},
            "candidates": [{"action_signature": "a", "macro_stability": 0.7, "intervention_cost": 0.2}],
            "deltas": [{"metric": "macro_stability", "delta": 0.2}],
        },
        "pareto_frontier": [{"macro_stability": 0.7, "intervention_cost": 0.2, "liquidity": 0.6}],
        "recommendation": {
            "scorecard": {"composite_score": 0.72},
            "evidence_chain": [{"metric": "macro_stability", "value": 0.7}],
        },
    }
    frames = _build_regulator_result_frames(result)
    localized = localize_dataframe_columns(frames["pareto"])

    assert "macro_stability" in frames["pareto"].columns
    assert "intervention_cost" in frames["pareto"].columns
    assert "宏观稳定性" in localized.columns
    assert "干预成本" in localized.columns
