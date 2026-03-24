from __future__ import annotations

from core.macro.state import MacroState
from core.mesa.civitas_model import social_snapshot_to_orderflow_reversal
from core.social import SocialContagionEngine, SocialGraphState, SocialMessage, SocialNodeType


def test_social_to_orderflow_reversal_after_refutation() -> None:
    graph = SocialGraphState()
    graph.set_node_profile("rumor", SocialNodeType.RUMOR_SOURCE)
    graph.set_node_profile("official", SocialNodeType.OFFICIAL_MEDIA)
    graph.set_node_profile("retail", SocialNodeType.RETAIL_DAY_TRADER)
    graph.add_edge("rumor", "retail")
    graph.add_edge("official", "retail")

    engine = SocialContagionEngine(feature_flag=True, seed=21)
    rumor_snapshot = engine.step(
        graph,
        MacroState(sentiment_index=0.40),
        messages=[
            SocialMessage(
                topic="panic_rumor",
                source_id="rumor",
                source_type=SocialNodeType.RUMOR_SOURCE.value,
                kind="rumor",
                polarity=-0.8,
                credibility=0.1,
                audience_tags=[SocialNodeType.RETAIL_DAY_TRADER.value],
            )
        ],
    )
    rumor_flow = social_snapshot_to_orderflow_reversal(rumor_snapshot.to_dict())

    refutation_snapshot = engine.step(
        graph,
        MacroState(sentiment_index=0.60),
        messages=[
            SocialMessage(
                topic="panic_rumor",
                source_id="official",
                source_type=SocialNodeType.OFFICIAL_MEDIA.value,
                kind="refutation",
                polarity=0.9,
                credibility=0.99,
                audience_tags=[SocialNodeType.RETAIL_DAY_TRADER.value],
                rebuttal_of="panic_rumor",
            )
        ],
    )
    refutation_flow = social_snapshot_to_orderflow_reversal(refutation_snapshot.to_dict())

    assert refutation_flow["suppression_ratio"] >= rumor_flow["suppression_ratio"]
    assert refutation_flow["orderflow_bias"] >= rumor_flow["orderflow_bias"]
