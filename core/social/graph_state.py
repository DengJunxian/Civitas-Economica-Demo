"""Social graph state for sentiment diffusion and exposure tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class GraphNodeState:
    """Node-level social state."""

    node_id: str
    sentiment: float = 0.0
    news_exposure: float = 0.4
    social_exposure: float = 0.5


@dataclass(slots=True)
class SocialGraphState:
    """Container for social graph topology and dynamic sentiment states."""

    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    nodes: Dict[str, GraphNodeState] = field(default_factory=dict)

    def ensure_node(self, node_id: str) -> GraphNodeState:
        """Create or return an existing graph node."""
        key = str(node_id)
        if key not in self.nodes:
            self.nodes[key] = GraphNodeState(node_id=key)
        self.adjacency.setdefault(key, [])
        return self.nodes[key]

    def add_edge(self, src: str, dst: str, *, bidirectional: bool = True) -> None:
        """Add an edge between two nodes."""
        self.ensure_node(src)
        self.ensure_node(dst)
        if dst not in self.adjacency[src]:
            self.adjacency[src].append(dst)
        if bidirectional and src not in self.adjacency[dst]:
            self.adjacency[dst].append(src)

    def set_sentiment(self, node_id: str, sentiment: float) -> None:
        """Set node sentiment in [-1, 1]."""
        node = self.ensure_node(node_id)
        node.sentiment = _clip(sentiment, -1.0, 1.0)

    def mean_sentiment(self) -> float:
        """Return average sentiment across nodes."""
        if not self.nodes:
            return 0.0
        return sum(node.sentiment for node in self.nodes.values()) / len(self.nodes)

    def neighbors(self, node_id: str) -> List[GraphNodeState]:
        """Return neighbor node states."""
        return [self.nodes[n] for n in self.adjacency.get(node_id, []) if n in self.nodes]

    @classmethod
    def ring(cls, node_ids: Iterable[str]) -> "SocialGraphState":
        """Build a simple ring graph as a deterministic default topology."""
        ids = [str(item) for item in node_ids]
        graph = cls()
        if not ids:
            return graph
        for item in ids:
            graph.ensure_node(item)
        if len(ids) == 1:
            return graph
        for idx, src in enumerate(ids):
            dst = ids[(idx + 1) % len(ids)]
            graph.add_edge(src, dst, bidirectional=True)
        return graph


if __name__ == "__main__":
    g = SocialGraphState.ring(["a", "b", "c"])
    g.set_sentiment("a", -0.2)
    g.set_sentiment("b", 0.4)
    g.set_sentiment("c", 0.1)
    print("mean_sentiment", g.mean_sentiment())
