from __future__ import annotations

import pandas as pd

from core.runtime_mode import resolve_runtime_mode_profile
from ui.dashboard import normalize_discovered_metrics_payload
from ui.policy_lab import (
    _policy_session_advance,
    _policy_session_enqueue_runtime_event,
    _policy_session_new,
    _policy_session_report_payload,
)


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
