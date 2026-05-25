"""Social-layer graph state and contagion engines."""

from core.social.contagion import (
    ContagionSnapshot,
    PropagationTrace,
    SocialContagionEngine,
    SocialMessage,
    build_social_propagation_report,
    write_social_propagation_report,
)
from core.social.graph_state import (
    DEFAULT_NODE_PROFILES,
    EdgeProfile,
    GraphNodeState,
    SocialGraphState,
    SocialNodeType,
)

__all__ = [
    "ContagionSnapshot",
    "DEFAULT_NODE_PROFILES",
    "EdgeProfile",
    "GraphNodeState",
    "PropagationTrace",
    "SocialContagionEngine",
    "SocialGraphState",
    "SocialMessage",
    "SocialNodeType",
    "build_social_propagation_report",
    "write_social_propagation_report",
]
