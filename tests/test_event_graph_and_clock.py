from pathlib import Path

from data_flywheel.event_graph_store import EventGraphStore
from data_flywheel.schemas import SeedEvent, ExtractedEntity
from core.time_manager import SimulationClock


def test_event_graph_store_ingest_and_query(tmp_path: Path):
    graph_path = tmp_path / "event_graph.graphml"
    store = EventGraphStore(str(graph_path))

    event = SeedEvent(
        source="test",
        title="政策冲击测试",
        summary="监管趋严",
        entities=[ExtractedEntity(name="证监会", entity_type="policy", confidence=0.9)],
        affected_sectors=["券商"],
        text_factors={
            "dominant_topic": "regulation",
            "topic_signals": [{"topic": "regulation", "score": 0.8, "source": "keyword"}],
            "financial_factors": {
                "panic_index": 0.7,
                "greed_index": 0.2,
                "policy_shock": 0.8,
                "regime_bias": "risk_off",
            },
            "impact_paths": [{"source": "regulation", "relation": "impacts_sector", "target": "券商", "weight": 0.8}],
        },
    )

    store.ingest(event)
    neighbors = store.query_neighbors("regulation")
    assert len(neighbors) > 0


def test_simulation_clock_mode_alignment():
    clock = SimulationClock(mode="FAST")
    assert clock.mode == "FAST"
    assert clock.time_step_seconds == 1.0

    clock.configure_mode("DEEP")
    assert clock.mode == "DEEP"
    assert clock.time_step_seconds == 6.0

