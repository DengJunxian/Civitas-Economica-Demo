from __future__ import annotations

from datetime import datetime

from core.events import RuntimeEventCompiler, RuntimeEventQueue


def test_runtime_event_compiler_structures_natural_language_event():
    compiler = RuntimeEventCompiler()
    event = compiler.compile_text(
        "盘中传闻融资保证金收紧，券商板块承压",
        timestamp=datetime(2026, 3, 23, 10, 0).timestamp(),
        source="unit_test",
    )
    assert event.event_type == "rumor"
    assert event.event_id
    assert event.trading_day == "2026-03-23"
    assert "leveraged_flow" in event.affected_agent_groups
    assert event.structured_payload["funding_stress"] > 0


def test_runtime_event_queue_triggers_by_time_and_affects_summary():
    queue = RuntimeEventQueue()
    t0 = datetime(2026, 3, 23, 10, 0).timestamp()
    event = queue.append("央行宣布流动性投放支持市场稳定", timestamp=t0, source="unit_test")
    assert queue.due_events(t0 - 1) == []
    due = queue.trigger_due(t0)
    assert due == [event]
    assert queue.trigger_due(t0) == []

    summary = queue.effect_summary(due)
    assert summary["active_count"] == 1
    assert summary["macro_state_delta"]["liquidity_index"] > 0
    assert summary["agent_belief_delta"]["credibility_weighted_shock"] > 0
    assert summary["order_flow_delta"]["buy_pressure"] > 0
    assert "流动性投放" in summary["report_narrative"]


def test_runtime_event_converts_to_experiment_event():
    queue = RuntimeEventQueue()
    event = queue.append(
        {
            "raw_text": "监管部门盘中喊话打击异常交易",
            "event_type": "regulatory_action",
            "timestamp": datetime(2026, 3, 23, 10, 30).timestamp(),
            "source": "unit_test",
            "credibility": 0.9,
            "affected_agent_groups": ["quant", "leveraged_flow"],
        }
    )
    experiment_event = event.to_experiment_event(effective_day=2, created_day=1)
    assert experiment_event.event_id == event.event_id
    assert experiment_event.event_type == "regulatory_action"
    assert experiment_event.effective_day == 2
    assert experiment_event.metadata["runtime_event_schema"] == "runtime_event_v1"
