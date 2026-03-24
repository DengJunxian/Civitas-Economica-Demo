"""Social contagion engine for sentiment diffusion and propagation reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import random
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.macro.state import MacroState
from core.social.graph_state import DEFAULT_NODE_PROFILES, GraphNodeState, SocialGraphState, SocialNodeType


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _profile_for(node_type: str | SocialNodeType) -> Dict[str, float]:
    key = str(node_type.value if isinstance(node_type, SocialNodeType) else node_type)
    return dict(DEFAULT_NODE_PROFILES.get(key, DEFAULT_NODE_PROFILES[SocialNodeType.RETAIL_DAY_TRADER.value]))


@dataclass(slots=True)
class SocialMessage:
    """Structured social message used by the enriched propagation engine."""

    topic: str
    source_id: str
    source_type: str
    kind: str = "rumor"
    polarity: float = 0.0
    strength: float = 1.0
    credibility: float = 0.5
    created_tick: int = 0
    scheduled_tick: int = 0
    decay: float = 0.1
    amplification: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PropagationTrace:
    """A single source-to-target propagation record."""

    topic: str
    kind: str
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    created_tick: int
    received_tick: int
    delay: int
    source_credibility: float
    target_receptivity: float
    amplification: float
    decay: float
    signal: float
    belief_delta: float
    refuted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContagionSnapshot:
    """Outcome of one social contagion step."""

    mean_sentiment: float
    stressed_nodes: List[str]
    node_sentiment: Dict[str, float]
    edge_channel_means: Dict[str, float] = field(default_factory=dict)
    propagation_chain: List[Dict[str, Any]] = field(default_factory=list)
    node_influence: Dict[str, float] = field(default_factory=dict)
    narrative_heat: Dict[str, float] = field(default_factory=dict)
    observation_packets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_rankings: List[Dict[str, Any]] = field(default_factory=list)
    rumor_suppression: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    feature_flag: bool = False
    seed: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    current_tick: int = 0
    event_queue: List[SocialMessage] = field(default_factory=list)

    def _edge_weight(self, graph: SocialGraphState, src: str, dst: str) -> Dict[str, float]:
        profile = graph.get_edge_profile(src, dst) if hasattr(graph, "get_edge_profile") else None
        trust = float(getattr(profile, "trust_edge", 0.5)) if profile is not None else 0.5
        position = float(getattr(profile, "position_similarity_edge", 0.5)) if profile is not None else 0.5
        news = float(getattr(profile, "news_exposure_edge", 0.5)) if profile is not None else 0.5
        institution = float(getattr(profile, "institution_affiliation_edge", 0.0)) if profile is not None else 0.0
        delay = float(getattr(profile, "propagation_delay_edge", 1.0)) if profile is not None else 1.0
        decay = float(getattr(profile, "decay_edge", 1.0)) if profile is not None else 1.0
        amplification = float(getattr(profile, "amplification_edge", 1.0)) if profile is not None else 1.0
        contradiction = float(getattr(profile, "contradiction_edge", 0.5)) if profile is not None else 0.5
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
            "propagation_delay_edge": max(0.0, delay),
            "decay_edge": max(0.0, decay),
            "amplification_edge": max(0.0, amplification),
            "contradiction_edge": _clip(contradiction, 0.0, 1.0),
        }

    def enqueue_message(self, message: SocialMessage | Mapping[str, Any]) -> SocialMessage:
        if isinstance(message, SocialMessage):
            payload = message
        else:
            payload = SocialMessage(
                topic=str(message.get("topic", "market")),
                source_id=str(message.get("source_id", "synthetic_source")),
                source_type=str(message.get("source_type", SocialNodeType.RUMOR_SOURCE.value)),
                kind=str(message.get("kind", "rumor")),
                polarity=float(message.get("polarity", 0.0)),
                strength=float(message.get("strength", 1.0)),
                credibility=float(message.get("credibility", 0.5)),
                created_tick=int(message.get("created_tick", self.current_tick)),
                scheduled_tick=int(message.get("scheduled_tick", self.current_tick)),
                decay=float(message.get("decay", 0.1)),
                amplification=float(message.get("amplification", 1.0)),
                metadata=dict(message.get("metadata", {})),
            )
        self.event_queue.append(payload)
        return payload

    def _default_message_from_rumor_shock(self, rumor_shock: float, macro_state: MacroState) -> SocialMessage:
        polarity = float(_clip(rumor_shock, -1.0, 1.0))
        if abs(polarity) < 1e-9:
            polarity = float(_clip((macro_state.sentiment_index - 0.5) * 2.0, -1.0, 1.0))
        return SocialMessage(
            topic="policy_sentiment",
            source_id="synthetic_rumor_source",
            source_type=SocialNodeType.RUMOR_SOURCE.value,
            kind="rumor" if polarity <= 0 else "support",
            polarity=polarity,
            strength=max(0.1, abs(polarity)),
            credibility=0.12,
            created_tick=self.current_tick,
            scheduled_tick=self.current_tick,
            decay=0.24,
            amplification=1.55,
            metadata={"rumor_shock": float(rumor_shock)},
        )

    def _default_message_from_macro(self, macro_state: MacroState) -> SocialMessage:
        polarity = float(_clip((macro_state.sentiment_index - 0.5) * 2.0, -1.0, 1.0))
        return SocialMessage(
            topic="macro_sentiment",
            source_id="macro_anchor",
            source_type=SocialNodeType.OFFICIAL_MEDIA.value,
            kind="policy",
            polarity=polarity,
            strength=max(0.1, abs(polarity)),
            credibility=0.62,
            created_tick=self.current_tick,
            scheduled_tick=self.current_tick,
            decay=0.12,
            amplification=1.10,
            metadata={"macro_anchor": True},
        )

    def _legacy_step(self, graph: SocialGraphState, macro_state: MacroState, rumor_shock: float = 0.0) -> ContagionSnapshot:
        """Run the legacy diffusion logic unchanged for backward compatibility."""
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

    def _normalize_messages(
        self,
        messages: Optional[Iterable[SocialMessage | Mapping[str, Any]]],
    ) -> List[SocialMessage]:
        normalized: List[SocialMessage] = []
        if not messages:
            return normalized
        for message in messages:
            if isinstance(message, SocialMessage):
                normalized.append(message)
            else:
                normalized.append(
                    SocialMessage(
                        topic=str(message.get("topic", "market")),
                        source_id=str(message.get("source_id", "synthetic_source")),
                        source_type=str(message.get("source_type", SocialNodeType.RUMOR_SOURCE.value)),
                        kind=str(message.get("kind", "rumor")),
                        polarity=float(message.get("polarity", 0.0)),
                        strength=float(message.get("strength", 1.0)),
                        credibility=float(message.get("credibility", 0.5)),
                        created_tick=int(message.get("created_tick", self.current_tick)),
                        scheduled_tick=int(message.get("scheduled_tick", self.current_tick)),
                        decay=float(message.get("decay", 0.1)),
                        amplification=float(message.get("amplification", 1.0)),
                        metadata=dict(message.get("metadata", {})),
                    )
                )
        return normalized

    def _propagate_message(
        self,
        graph: SocialGraphState,
        message: SocialMessage,
        macro_state: MacroState,
    ) -> List[PropagationTrace]:
        traces: List[PropagationTrace] = []
        source_node = graph.nodes.get(message.source_id)
        source_profile = source_node if source_node is not None else GraphNodeState(node_id=message.source_id)
        source_profile.apply_profile(message.source_type)

        if message.source_id in graph.nodes:
            target_ids = [message.source_id] + graph.adjacency.get(message.source_id, [])
        else:
            target_ids = list(graph.nodes.keys())

        if message.kind == "refutation":
            polarity = abs(float(message.polarity))
        else:
            polarity = float(message.polarity)

        base_strength = max(0.0, float(message.strength))
        for target_id in target_ids:
            target = graph.ensure_node(target_id)
            edge_profile = graph.get_edge_profile(message.source_id, target_id) if message.source_id in graph.nodes else None
            edge_info = self._edge_weight(graph, message.source_id, target_id) if message.source_id in graph.nodes else {
                "trust_edge": 0.5,
                "position_similarity_edge": 0.5,
                "news_exposure_edge": 0.5,
                "institution_affiliation_edge": 0.0,
                "propagation_delay_edge": 1.0,
                "decay_edge": 1.0,
                "amplification_edge": 1.0,
                "contradiction_edge": 0.5,
                "edge_score": 0.5,
            }
            delay = int(
                max(
                    0,
                    source_profile.propagation_delay
                    + target.propagation_delay
                    + int(round(edge_info["propagation_delay_edge"])),
                )
            )
            received_tick = self.current_tick + delay
            decay_window = max(0, received_tick - message.created_tick)
            decay = math.exp(-(target.decay_rate + edge_info["decay_edge"] * 0.05) * decay_window)
            credibility = _clip(
                message.credibility * source_profile.source_credibility * edge_info["trust_edge"],
                0.0,
                1.0,
            )
            amplification = max(
                0.0,
                source_profile.amplification
                * target.amplification
                * edge_info["amplification_edge"]
                * message.amplification,
            )
            delay_factor = 1.0 / max(1.0, 1.0 + delay)
            contradiction = 1.0 - target.contradiction_sensitivity * edge_info["contradiction_edge"]
            contradiction = _clip(contradiction, 0.15, 1.15)
            macro_bias = 1.0 + 0.15 * (macro_state.sentiment_index - 0.5)
            signal = polarity * base_strength * credibility * amplification * decay * delay_factor * contradiction * macro_bias
            belief_delta = signal * (0.55 + 0.45 * target.social_exposure)
            if message.kind == "refutation":
                belief_delta = abs(belief_delta)
                signal = abs(signal)
            else:
                belief_delta = float(belief_delta)

            target.sentiment = _clip(target.sentiment + belief_delta, -1.0, 1.0)
            target.belief_strength = _clip(target.belief_strength + abs(belief_delta), 0.0, 1.0)
            target.reach_score += abs(signal) * amplification
            target.narrative_heat += abs(signal)
            if target.first_seen_tick < 0:
                target.first_seen_tick = received_tick
            target.last_seen_tick = received_tick
            target.record_event(
                {
                    "topic": message.topic,
                    "kind": message.kind,
                    "source_id": message.source_id,
                    "source_type": message.source_type,
                    "signal": signal,
                    "belief_delta": belief_delta,
                    "received_tick": received_tick,
                    "source_credibility": credibility,
                }
            )
            target.observation_state = {
                "topic": message.topic,
                "kind": message.kind,
                "dominant_signal": signal,
                "rumor_pressure": max(0.0, -signal) if message.kind != "refutation" else 0.0,
                "refutation_pressure": abs(signal) if message.kind == "refutation" else 0.0,
                "source_credibility": credibility,
                "received_tick": received_tick,
            }
            traces.append(
                PropagationTrace(
                    topic=message.topic,
                    kind=message.kind,
                    source_id=message.source_id,
                    source_type=message.source_type,
                    target_id=target_id,
                    target_type=target.node_type,
                    created_tick=message.created_tick,
                    received_tick=received_tick,
                    delay=delay,
                    source_credibility=credibility,
                    target_receptivity=target.social_exposure,
                    amplification=amplification,
                    decay=decay,
                    signal=signal,
                    belief_delta=belief_delta,
                    refuted=message.kind == "refutation",
                    metadata=dict(message.metadata),
                )
            )
            graph.record_observation(target_id, traces[-1].to_dict())

        return traces

    def _build_snapshot(
        self,
        graph: SocialGraphState,
        traces: List[PropagationTrace],
        macro_state: MacroState,
        *,
        rumor_shock: float,
    ) -> ContagionSnapshot:
        node_sentiment = {node_id: float(node.sentiment) for node_id, node in graph.nodes.items()}
        stressed = sorted([node_id for node_id, value in node_sentiment.items() if value < -0.35])
        mean_sentiment = sum(node_sentiment.values()) / len(node_sentiment) if node_sentiment else 0.0

        edge_channel_means: Dict[str, float] = {
            "trust_edge": 0.0,
            "position_similarity_edge": 0.0,
            "news_exposure_edge": 0.0,
            "institution_affiliation_edge": 0.0,
            "edge_score": 0.0,
            "propagation_delay": 0.0,
            "amplification": 0.0,
            "decay": 0.0,
        }
        if traces:
            edge_channel_means["trust_edge"] = float(sum(t.source_credibility for t in traces) / len(traces))
            edge_channel_means["position_similarity_edge"] = float(sum(t.target_receptivity for t in traces) / len(traces))
            edge_channel_means["news_exposure_edge"] = float(
                sum(graph.get_edge_profile(t.source_id, t.target_id).news_exposure_edge if t.source_id in graph.nodes else 0.5 for t in traces)
                / len(traces)
            )
            edge_channel_means["institution_affiliation_edge"] = float(
                sum(graph.get_edge_profile(t.source_id, t.target_id).institution_affiliation_edge if t.source_id in graph.nodes else 0.0 for t in traces)
                / len(traces)
            )
            edge_channel_means["edge_score"] = float(sum(abs(t.signal) for t in traces) / len(traces))
            edge_channel_means["propagation_delay"] = float(sum(t.delay for t in traces) / len(traces))
            edge_channel_means["amplification"] = float(sum(t.amplification for t in traces) / len(traces))
            edge_channel_means["decay"] = float(sum(t.decay for t in traces) / len(traces))

        influence: Dict[str, float] = {}
        narrative_heat: Dict[str, float] = {}
        source_totals: Dict[str, float] = {}
        source_counts: Dict[str, int] = {}
        rumor_heat_before = 0.0
        refutation_heat = 0.0

        for trace in traces:
            influence[trace.target_id] = influence.get(trace.target_id, 0.0) + abs(trace.signal)
            narrative_heat[trace.topic] = narrative_heat.get(trace.topic, 0.0) + abs(trace.signal)
            source_totals[trace.source_id] = source_totals.get(trace.source_id, 0.0) + abs(trace.signal)
            source_counts[trace.source_id] = source_counts.get(trace.source_id, 0) + 1
            if trace.kind == "rumor":
                rumor_heat_before += abs(trace.signal)
            if trace.kind == "refutation":
                refutation_heat += abs(trace.signal)

        source_rankings = [
            {
                "source_id": source_id,
                "total_influence": float(total),
                "mean_influence": float(total / max(1, source_counts[source_id])),
                "spread_count": int(source_counts[source_id]),
            }
            for source_id, total in sorted(source_totals.items(), key=lambda item: item[1], reverse=True)
        ]

        observation_packets = {
            node_id: graph.build_observation_payload(node_id, current_tick=self.current_tick)
            for node_id in graph.nodes.keys()
        }

        rumor_heat_after = max(0.0, rumor_heat_before - refutation_heat)
        rumor_suppression = {
            "rumor_heat_before": float(rumor_heat_before),
            "refutation_heat": float(refutation_heat),
            "rumor_heat_after": float(rumor_heat_after),
            "delta": float(rumor_heat_after - rumor_heat_before),
            "suppression_ratio": float(1.0 - rumor_heat_after / max(rumor_heat_before, 1e-12)) if rumor_heat_before > 0 else 0.0,
        }
        snapshot = ContagionSnapshot(
            mean_sentiment=mean_sentiment,
            stressed_nodes=stressed,
            node_sentiment=node_sentiment,
            edge_channel_means=edge_channel_means,
            propagation_chain=[trace.to_dict() for trace in traces],
            node_influence=influence,
            narrative_heat=narrative_heat,
            observation_packets=observation_packets,
            source_rankings=source_rankings,
            rumor_suppression=rumor_suppression,
            metadata={
                "feature_flag": bool(self.feature_flag),
                "seed": int(self.seed),
                "config_hash": _stable_hash(
                    {
                        "seed": int(self.seed),
                        "feature_flag": bool(self.feature_flag),
                        "config": dict(self.config),
                        "graph_signature": graph.graph_signature(),
                        "tick": int(self.current_tick),
                        "rumor_shock": float(rumor_shock),
                    }
                ),
                "snapshot_info": {
                    "node_count": len(graph.nodes),
                    "edge_count": sum(len(v) for v in graph.adjacency.values()),
                    "tick": int(self.current_tick),
                    "node_type_distribution": graph.node_type_counts(),
                    "macro_sentiment_index": float(macro_state.sentiment_index),
                },
            },
        )
        return snapshot

    def step(
        self,
        graph: SocialGraphState,
        macro_state: MacroState,
        rumor_shock: float = 0.0,
        *,
        messages: Optional[Iterable[SocialMessage | Mapping[str, Any]]] = None,
        tick: Optional[int] = None,
    ) -> ContagionSnapshot:
        """Run one diffusion step and update graph sentiments in place."""
        if not graph.nodes:
            return ContagionSnapshot(mean_sentiment=0.0, stressed_nodes=[], node_sentiment={})

        self.current_tick = int(self.current_tick + 1 if tick is None else tick)

        if not self.feature_flag:
            return self._legacy_step(graph, macro_state, rumor_shock=rumor_shock)

        active_messages = list(self.event_queue)
        active_messages.extend(self._normalize_messages(messages))

        if abs(rumor_shock) > 1e-12:
            active_messages.append(self._default_message_from_rumor_shock(rumor_shock, macro_state))
        if not active_messages:
            active_messages.append(self._default_message_from_macro(macro_state))

        due_messages = [message for message in active_messages if message.scheduled_tick <= self.current_tick]
        self.event_queue = [message for message in active_messages if message.scheduled_tick > self.current_tick]

        traces: List[PropagationTrace] = []
        for message in due_messages:
            traces.extend(self._propagate_message(graph, message, macro_state))

        if not traces:
            return self._legacy_step(graph, macro_state, rumor_shock=rumor_shock)

        snapshot = self._build_snapshot(graph, traces, macro_state, rumor_shock=rumor_shock)
        return snapshot

    def write_report(
        self,
        snapshot: ContagionSnapshot,
        graph: SocialGraphState,
        path: str | Path,
        *,
        title: str = "social_propagation",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        report = build_social_propagation_report(snapshot, graph, title=title, metadata=metadata or self.config)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def build_social_propagation_report(
    snapshot: ContagionSnapshot,
    graph: SocialGraphState,
    *,
    title: str = "social_propagation",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = snapshot.to_dict()
    report = {
        "report_type": title,
        "feature_flag": bool(snapshot.metadata.get("feature_flag", False)),
        "seed": int(snapshot.metadata.get("seed", 0)),
        "config_hash": str(snapshot.metadata.get("config_hash", "")),
        "snapshot_info": dict(snapshot.metadata.get("snapshot_info", {})),
        "node_type_distribution": graph.node_type_counts(),
        "mean_sentiment": float(snapshot.mean_sentiment),
        "stressed_nodes": list(snapshot.stressed_nodes),
        "edge_channel_means": dict(snapshot.edge_channel_means),
        "propagation_chain": list(snapshot.propagation_chain),
        "node_influence": dict(snapshot.node_influence),
        "narrative_heat": dict(snapshot.narrative_heat),
        "rumor_suppression": dict(snapshot.rumor_suppression),
        "observation_packets": dict(snapshot.observation_packets),
        "source_rankings": list(snapshot.source_rankings),
        "snapshot": payload,
        "metadata": dict(metadata or {}),
    }
    report["report_hash"] = _stable_hash(
        {
            "report_type": title,
            "config_hash": report["config_hash"],
            "node_count": report["snapshot_info"].get("node_count", 0),
            "chain_size": len(report["propagation_chain"]),
            "metadata": report["metadata"],
        }
    )
    return report


def write_social_propagation_report(
    snapshot: ContagionSnapshot,
    graph: SocialGraphState,
    path: str | Path,
    *,
    title: str = "social_propagation",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    report = build_social_propagation_report(snapshot, graph, title=title, metadata=metadata)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    from core.social.graph_state import SocialGraphState

    graph = SocialGraphState.ring(["a", "b", "c", "d"])
    graph.set_node_profile("a", SocialNodeType.OFFICIAL_MEDIA)
    graph.set_sentiment("a", -0.4)
    graph.set_sentiment("b", 0.1)
    graph.set_sentiment("c", 0.2)
    graph.set_sentiment("d", 0.0)
    macro = MacroState(sentiment_index=0.45)
    engine = SocialContagionEngine(feature_flag=True)
    snap = engine.step(
        graph,
        macro_state=macro,
        rumor_shock=-0.3,
        messages=[{"topic": "policy", "source_id": "a", "source_type": "official_media", "kind": "policy", "polarity": 0.4}],
    )
    print(json.dumps(snap.to_dict(), ensure_ascii=False, indent=2))
