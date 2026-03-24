from __future__ import annotations

from core.macro.state import MacroState
from core.social import SocialContagionEngine, SocialGraphState, SocialMessage, SocialNodeType


def test_rumor_refutation_reduces_remaining_rumor_heat() -> None:
    graph = SocialGraphState()
    graph.set_node_profile("official", SocialNodeType.OFFICIAL_MEDIA)
    graph.set_node_profile("rumor", SocialNodeType.RUMOR_SOURCE)
    graph.set_node_profile("retail", SocialNodeType.RETAIL_DAY_TRADER)
    graph.add_edge("official", "retail")
    graph.add_edge("rumor", "retail")

    engine = SocialContagionEngine(feature_flag=True, seed=13, config={"scenario": "refutation"})

    rumor_snapshot = engine.step(
        graph,
        MacroState(sentiment_index=0.42),
        messages=[
            SocialMessage(
                topic="panic_rumor",
                source_id="rumor",
                source_type=SocialNodeType.RUMOR_SOURCE.value,
                kind="rumor",
                polarity=-0.9,
                strength=1.0,
                credibility=0.10,
                created_tick=1,
                scheduled_tick=1,
            )
        ],
    )
    rumor_sentiment = rumor_snapshot.node_sentiment["retail"]
    assert rumor_snapshot.rumor_suppression["rumor_heat_before"] > 0.0

    refutation_snapshot = engine.step(
        graph,
        MacroState(sentiment_index=0.58),
        messages=[
            SocialMessage(
                topic="panic_rumor",
                source_id="official",
                source_type=SocialNodeType.OFFICIAL_MEDIA.value,
                kind="refutation",
                polarity=0.9,
                strength=1.0,
                credibility=0.99,
                created_tick=2,
                scheduled_tick=2,
            )
        ],
    )

    assert refutation_snapshot.rumor_suppression["rumor_heat_after"] <= refutation_snapshot.rumor_suppression["rumor_heat_before"]
    assert refutation_snapshot.node_sentiment["retail"] >= rumor_sentiment
    assert refutation_snapshot.observation_packets["retail"]["memory_seed"]["refutation_pressure"] >= 0.0
    assert refutation_snapshot.source_rankings[0]["source_id"] == "official"
