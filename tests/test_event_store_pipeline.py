from __future__ import annotations

from core.event_store import EventRecord, EventStore, EventType


def test_event_store_visibility_and_snapshot(tmp_path):
    store = EventStore(root_dir=tmp_path / "event_store")
    dataset_version = "vtest"

    records = [
        EventRecord(
            event_type=EventType.NEWS,
            timestamp="2025-01-01T09:00:00Z",
            visibility_time="2025-01-01T09:00:00Z",
            source="news_feed",
            confidence=0.9,
            payload={"headline": "open"},
        ),
        EventRecord(
            event_type=EventType.NEWS,
            timestamp="2025-01-01T10:00:00Z",
            visibility_time="2025-01-01T11:00:00Z",
            source="news_feed",
            confidence=0.9,
            payload={"headline": "future"},
        ),
        EventRecord(
            event_type=EventType.MARKET_BAR,
            timestamp="2025-01-01T09:30:00Z",
            visibility_time="2025-01-01T09:30:00Z",
            source="market",
            confidence=1.0,
            payload={"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
        ),
    ]
    written = store.append_events(dataset_version, records, seed=7, config_hash="abc", snapshot_id="snap0")
    assert written

    visible = store.query_events(
        dataset_version,
        event_types=[EventType.NEWS.value],
        visible_at="2025-01-01T10:30:00Z",
    )
    assert len(visible) == 1
    assert "open" in visible.iloc[0]["payload_json"]

    snapshot = store.create_snapshot(
        dataset_version,
        seed=7,
        config_hash="abc",
        feature_flags={"event_store_v1": True},
    )
    assert snapshot.snapshot_id

    scenario = store.write_scenario_manifest(
        dataset_version,
        scenario_id="s1",
        snapshot_id=snapshot.snapshot_id,
        start_time="2025-01-01T09:00:00Z",
        end_time="2025-01-01T12:00:00Z",
        seed=7,
        config_hash="abc",
        event_types=[EventType.NEWS.value, EventType.MARKET_BAR.value],
    )
    assert scenario.scenario_id == "s1"

    frame = store.query_scenario_events(dataset_version, "s1", visible_at="2025-01-01T10:30:00Z")
    assert not frame.empty
    assert set(frame["event_type"].unique()).issubset({EventType.NEWS.value, EventType.MARKET_BAR.value})
