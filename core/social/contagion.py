"""Social contagion engine for sentiment diffusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from core.macro.state import MacroState
from core.social.graph_state import SocialGraphState


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class ContagionSnapshot:
    """Outcome of one social contagion step."""

    mean_sentiment: float
    stressed_nodes: List[str]
    node_sentiment: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SocialContagionEngine:
    """Diffuses sentiment through graph topology and macro anchors."""

    contagion_strength: float = 0.55
    self_memory: float = 0.35
    macro_anchor: float = 0.25

    def step(self, graph: SocialGraphState, macro_state: MacroState, rumor_shock: float = 0.0) -> ContagionSnapshot:
        """Run one diffusion step and update graph sentiments in place."""
        if not graph.nodes:
            return ContagionSnapshot(mean_sentiment=0.0, stressed_nodes=[], node_sentiment={})

        new_values: Dict[str, float] = {}
        for node_id, node in graph.nodes.items():
            neighbors = graph.neighbors(node_id)
            neighborhood = sum(n.sentiment for n in neighbors) / len(neighbors) if neighbors else 0.0
            macro_term = (macro_state.sentiment_index - 0.5) * 2.0
            exposure_term = 0.5 * node.news_exposure * rumor_shock + 0.4 * node.social_exposure * neighborhood
            updated = (
                self.self_memory * node.sentiment
                + self.contagion_strength * neighborhood
                + self.macro_anchor * macro_term
                + exposure_term
            )
            new_values[node_id] = _clip(updated, -1.0, 1.0)

        for node_id, sentiment in new_values.items():
            graph.nodes[node_id].sentiment = sentiment

        stressed = sorted([node_id for node_id, value in new_values.items() if value < -0.35])
        mean_sentiment = sum(new_values.values()) / len(new_values)
        return ContagionSnapshot(
            mean_sentiment=mean_sentiment,
            stressed_nodes=stressed,
            node_sentiment=new_values,
        )


if __name__ == "__main__":
    from core.social.graph_state import SocialGraphState

    graph = SocialGraphState.ring(["a", "b", "c", "d"])
    graph.set_sentiment("a", -0.4)
    graph.set_sentiment("b", 0.1)
    graph.set_sentiment("c", 0.2)
    graph.set_sentiment("d", 0.0)
    macro = MacroState(sentiment_index=0.45)
    engine = SocialContagionEngine()
    snap = engine.step(graph, macro_state=macro, rumor_shock=-0.3)
    print(snap.to_dict())
