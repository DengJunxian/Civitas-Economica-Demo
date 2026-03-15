"""Social contagion engine for sentiment diffusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    edge_channel_means: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SocialContagionEngine:
    """Diffuses sentiment through graph topology and macro anchors."""

    contagion_strength: float = 0.55
    self_memory: float = 0.35
    macro_anchor: float = 0.25
    trust_weight: float = 0.35
    position_similarity_weight: float = 0.25
    news_exposure_weight: float = 0.20
    institution_affiliation_weight: float = 0.20

    def _edge_weight(self, graph: SocialGraphState, src: str, dst: str) -> Dict[str, float]:
        profile = graph.get_edge_profile(src, dst) if hasattr(graph, "get_edge_profile") else None
        trust = float(getattr(profile, "trust_edge", 0.5)) if profile is not None else 0.5
        position = float(getattr(profile, "position_similarity_edge", 0.5)) if profile is not None else 0.5
        news = float(getattr(profile, "news_exposure_edge", 0.5)) if profile is not None else 0.5
        institution = float(getattr(profile, "institution_affiliation_edge", 0.0)) if profile is not None else 0.0
        edge_score = (
            self.trust_weight * trust
            + self.position_similarity_weight * position
            + self.news_exposure_weight * news
            + self.institution_affiliation_weight * institution
        )
        return {
            "edge_score": _clip(edge_score, 0.0, 1.0),
            "trust_edge": _clip(trust, 0.0, 1.0),
            "position_similarity_edge": _clip(position, 0.0, 1.0),
            "news_exposure_edge": _clip(news, 0.0, 1.0),
            "institution_affiliation_edge": _clip(institution, 0.0, 1.0),
        }

    def step(self, graph: SocialGraphState, macro_state: MacroState, rumor_shock: float = 0.0) -> ContagionSnapshot:
        """Run one diffusion step and update graph sentiments in place."""
        if not graph.nodes:
            return ContagionSnapshot(mean_sentiment=0.0, stressed_nodes=[], node_sentiment={})

        new_values: Dict[str, float] = {}
        edge_stats = {
            "trust_edge": 0.0,
            "position_similarity_edge": 0.0,
            "news_exposure_edge": 0.0,
            "institution_affiliation_edge": 0.0,
            "edge_score": 0.0,
        }
        edge_count = 0.0

        for node_id, node in graph.nodes.items():
            neighbor_ids = graph.adjacency.get(node_id, [])
            weighted_sum = 0.0
            weight_total = 0.0
            weighted_news_edge = 0.0
            weighted_trust_edge = 0.0
            for neighbor_id in neighbor_ids:
                neighbor = graph.nodes.get(neighbor_id)
                if neighbor is None:
                    continue
                channels = self._edge_weight(graph, node_id, neighbor_id)
                w = channels["edge_score"]
                weighted_sum += w * neighbor.sentiment
                weight_total += w
                weighted_news_edge += w * channels["news_exposure_edge"]
                weighted_trust_edge += w * channels["trust_edge"]
                edge_stats["trust_edge"] += channels["trust_edge"]
                edge_stats["position_similarity_edge"] += channels["position_similarity_edge"]
                edge_stats["news_exposure_edge"] += channels["news_exposure_edge"]
                edge_stats["institution_affiliation_edge"] += channels["institution_affiliation_edge"]
                edge_stats["edge_score"] += channels["edge_score"]
                edge_count += 1.0

            neighborhood = weighted_sum / weight_total if weight_total > 1e-8 else 0.0
            avg_news_edge = weighted_news_edge / weight_total if weight_total > 1e-8 else 0.0
            avg_trust_edge = weighted_trust_edge / weight_total if weight_total > 1e-8 else 0.0
            macro_term = (macro_state.sentiment_index - 0.5) * 2.0
            exposure_term = (
                (0.35 + 0.65 * avg_news_edge) * node.news_exposure * rumor_shock
                + (0.25 + 0.75 * avg_trust_edge) * node.social_exposure * neighborhood
            )
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
        channel_means = {}
        for key, value in edge_stats.items():
            channel_means[key] = float(value / edge_count) if edge_count > 0 else 0.0
        return ContagionSnapshot(
            mean_sentiment=mean_sentiment,
            stressed_nodes=stressed,
            node_sentiment=new_values,
            edge_channel_means=channel_means,
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
