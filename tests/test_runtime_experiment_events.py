from __future__ import annotations

import pandas as pd

from core.event_store import EventStore, EventType
from core.policy_session import PolicySession, PolicySessionConfig


class _DummyEventBus:
    def __init__(self) -> None:
        self.events = []

    def publish(self, **kwargs):
        self.events.append(kwargs)


class _DummyEnvironment:
    def __init__(self) -> None:
        self.simulation_time = 0
        self.event_bus = _DummyEventBus()
        self.scheduled = []

    def schedule_policy_shock(self, policy_text, **kwargs):
        self.scheduled.append({"policy_text": policy_text, **kwargs})

    async def simulation_step(self):
        self.simulation_time += 1
        return {
            "old_price": 100.0 + self.simulation_time,
            "new_price": 101.0 + self.simulation_time,
            "buy_volume": 1200.0,
            "sell_volume": 900.0,
            "trade_count": 3,
            "macro_state": {"sentiment_index": 0.58, "liquidity_index": 0.64},
            "behavioral_diagnostics": {"csad": 0.08},
            "policy_input": {"policy_text": "", "policy_intensity": 1.0, "policy_source": "test"},
        }


def _session(tmp_path, *, persist: bool = False) -> PolicySession:
    return PolicySession(
        environment=_DummyEnvironment(),
        config=PolicySessionConfig(
            total_days=4,
            start_date="2026-01-02",
            event_dataset_version="runtime_test",
            persist_runtime_events=persist,
        ),
        event_store=EventStore(root_dir=tmp_path / "event_store"),
    )


def test_policy_session_can_append_policy_mid_run(tmp_path):
    session = _session(tmp_path)
    session.enqueue_policy("基础稳市政策", effective_day=1, strength=1.0)
    session.advance(1)

    policy_id = session.append_policy("追加流动性工具", effective_day=2, strength=0.8)
    result = session.advance(1)

    assert policy_id
    assert any(item["event_id"] == policy_id for item in result["active_events"])
    assert "追加流动性工具" in result["event_digest"]["policy_digest"]


def test_policy_session_can_append_news_mid_run(tmp_path):
    session = _session(tmp_path)
    session.advance(1)

    event_id = session.append_news_event("盘中突发重大利空新闻", effective_day=2, strength=1.2)
    result = session.advance(1)

    assert event_id
    assert any(item["event_type"] == "major_news" for item in result["active_events"])
    assert "重大利空" in result["event_digest"]["news_digest"]


def test_runtime_events_affect_active_digest(tmp_path):
    session = _session(tmp_path)
    session.append_rumor_event("未经证实的融资收紧传闻扩散", effective_day=1, strength=1.0)
    result = session.advance(1)

    digest = result["event_digest"]
    impact = digest["aggregate_impact"]
    assert digest["active_count"] == 1
    assert digest["rumor_digest"]
    assert impact["social_delta"]["rumor_pressure"] > 0
    assert impact["macro_delta"]["sentiment_index"] < 0


def test_event_store_persists_runtime_injected_events(tmp_path):
    store = EventStore(root_dir=tmp_path / "event_store")
    session = PolicySession(
        environment=_DummyEnvironment(),
        config=PolicySessionConfig(
            total_days=2,
            start_date="2026-01-02",
            event_dataset_version="runtime_persist_test",
            persist_runtime_events=True,
        ),
        event_store=store,
    )
    event_id = session.append_news_event("重大新闻写入事件仓库", effective_day=1, strength=1.0)

    frame = store.query_events("runtime_persist_test", event_types=[EventType.NEWS.value])

    assert not frame.empty
    assert event_id in set(frame["event_id"])
    payloads = frame["payload_json"].astype(str).tolist()
    assert any("experiment_event_v1" in payload for payload in payloads)

