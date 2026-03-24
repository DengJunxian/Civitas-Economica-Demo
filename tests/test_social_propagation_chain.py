from __future__ import annotations

import json
from pathlib import Path

from core.macro.state import MacroState
from core.social import (
    SocialContagionEngine,
    SocialGraphState,
    SocialMessage,
    SocialNodeType,
    write_social_propagation_report,
)


def test_social_propagation_chain_emits_node_observations_and_report(tmp_path: Path) -> None:
    graph = SocialGraphState()
    graph.set_node_profile("official", SocialNodeType.OFFICIAL_MEDIA)
    graph.set_node_profile("regulator", SocialNodeType.REGULATOR_VOICE)
    graph.set_node_profile("kol", SocialNodeType.KOL_SOCIAL)
    graph.set_node_profile("retail", SocialNodeType.RETAIL_DAY_TRADER)
    graph.set_node_profile("institution", SocialNodeType.INSTITUTION)
    graph.set_node_profile("rumor", SocialNodeType.RUMOR_SOURCE)

    graph.add_edge("official", "retail")
    graph.add_edge("regulator", "retail")
    graph.add_edge("kol", "retail")
    graph.add_edge("rumor", "kol")
    graph.add_edge("retail", "institution")

    engine = SocialContagionEngine(feature_flag=True, seed=7, config={"scenario": "policy_rumor"})
    snapshot = engine.step(
        graph,
        MacroState(sentiment_index=0.46),
        messages=[
            SocialMessage(
                topic="policy_cut",
                source_id="regulator",
                source_type=SocialNodeType.REGULATOR_VOICE.value,
                kind="policy",
                polarity=0.7,
                strength=1.0,
                credibility=0.98,
                created_tick=1,
                scheduled_tick=1,
            ),
            SocialMessage(
                topic="rumor_selloff",
                source_id="rumor",
                source_type=SocialNodeType.RUMOR_SOURCE.value,
                kind="rumor",
                polarity=-0.8,
                strength=1.0,
                credibility=0.12,
                created_tick=1,
                scheduled_tick=1,
            ),
        ],
    )

    assert snapshot.propagation_chain
    assert snapshot.node_influence["retail"] > 0.0
    assert snapshot.narrative_heat["policy_cut"] > 0.0
    assert snapshot.metadata["feature_flag"] is True
    assert "config_hash" in snapshot.metadata
    assert snapshot.observation_packets["retail"]["memory_seed"]["rumor_pressure"] >= 0.0

    report_path = write_social_propagation_report(snapshot, graph, tmp_path / "social_propagation_report.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["node_type_distribution"]["retail_day_trader"] >= 1
    assert payload["propagation_chain"][0]["topic"] in {"policy_cut", "rumor_selloff"}
