"""Structured policy parsing and transmission models.

This module keeps the legacy macro shock path intact while adding a typed
policy representation with explicit transmission channels and layered outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    """Return a deterministic hash for nested JSON-like data."""

    raw = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _match_any(text: str, *patterns: str) -> bool:
    lower = text.lower()
    return any(pattern in text or pattern in lower for pattern in patterns)


def _first_numeric_hint(text: str) -> float:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return 0.0
    value = abs(float(match.group(1)))
    if value > 1000:
        return 1.0
    if value > 100:
        return 0.7
    if value > 10:
        return 0.35
    if value > 1:
        return 0.1
    return 0.0


@dataclass(slots=True)
class TransmissionChannel:
    """A single policy transmission channel."""

    name: str
    impact: float
    lag_days: int
    decay_half_life: float
    direction: str
    macro_delta: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyUncertainty:
    """Parse confidence and ambiguity for a policy event."""

    confidence: float
    ambiguity: float
    rumor_risk: float
    source_reliability: float
    fallback_used: bool = False
    parser_mode: str = "structured"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyEvent:
    """Structured policy event produced by the parser."""

    policy_id: str
    raw_text: str
    policy_type: str
    policy_label: str
    direction: str
    intensity: float
    tick: int
    matched_tokens: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source: str = "structured_parser"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyPackage:
    """Structured policy package with layered transmission outputs."""

    event: PolicyEvent
    channels: List[TransmissionChannel]
    uncertainty: PolicyUncertainty
    sector_effects: Dict[str, float]
    factor_effects: Dict[str, float]
    agent_class_effects: Dict[str, float]
    market_effects: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parser_version: str = "structured_policy_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_policy_shock_fields(self) -> Dict[str, float]:
        fields = {
            "policy_rate_delta": 0.0,
            "fiscal_stimulus_delta": 0.0,
            "liquidity_injection": 0.0,
            "credit_spread_delta": 0.0,
            "stamp_tax_delta": 0.0,
            "sentiment_delta": 0.0,
            "rumor_shock": 0.0,
        }
        for channel in self.channels:
            for key, value in channel.macro_delta.items():
                if key in fields:
                    fields[key] += float(value)
        return fields

    def top_layers(self, limit: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        def _top(items: Mapping[str, float]) -> List[Tuple[str, float]]:
            ranked = sorted(items.items(), key=lambda kv: abs(kv[1]), reverse=True)
            return [(str(name), float(value)) for name, value in ranked[:limit]]

        return {
            "sector": _top(self.sector_effects),
            "factor": _top(self.factor_effects),
            "agent_class": _top(self.agent_class_effects),
        }


class StructuredPolicyParser:
    """Parse natural-language policy text into a structured policy package."""

    _PATTERNS: Sequence[Tuple[str, str, Tuple[str, ...], str]] = (
        ("liquidity_easing", "Liquidity easing", ("流动性", "注入", "降准", "rrr", "rrr cut", "liquidity"), "easing"),
        ("rate_cut", "Rate cut", ("降息", "下调利率", "cut rate", "rate cut", "policy rate"), "easing"),
        ("fiscal_stimulus", "Fiscal stimulus", ("财政", "补贴", "基建", "stimulus", "fiscal"), "easing"),
        ("tax_cut", "Tax cut", ("印花税下调", "减税", "税率下调", "tax cut", "stamp duty cut"), "easing"),
        ("tax_hike", "Tax hike", ("印花税上调", "加税", "税率上调", "tax hike", "stamp duty hike"), "tightening"),
        ("stabilization", "Stabilization", ("平准", "维稳", "国家队", "backstop", "stabilize"), "stabilizing"),
        ("rumor_refutation", "Rumor refutation", ("辟谣", "澄清", "压制谣言", "refute", "rumor suppression"), "stabilizing"),
        ("tightening", "Tightening", ("收紧", "加息", "tighten", "tightening"), "tightening"),
    )

    def __init__(self, *, feature_flags: Optional[Mapping[str, bool]] = None, seed: int = 0, version: str = "structured_policy_v1") -> None:
        self.feature_flags = dict(feature_flags or {})
        self.seed = int(seed)
        self.version = version

    def parse(
        self,
        policy_text: str,
        *,
        tick: int = 0,
        policy_type_hint: Optional[str] = None,
        intensity: float = 1.0,
        market_regime: Optional[str] = None,
        snapshot_info: Optional[Mapping[str, Any]] = None,
        fallback_used: bool = False,
    ) -> PolicyPackage:
        text = str(policy_text or "").strip()
        text_lower = text.lower()
        numeric_hint = _first_numeric_hint(text)
        base_intensity = _clip(float(intensity) * (1.0 + 0.2 * numeric_hint), 0.0, 2.0)
        preferred_policy_type: Optional[str] = None
        if _match_any(text, "印花税", "stamp duty", "税", "税率", "税费", "减税"):
            if _match_any(text, "上调", "提高", "加税", "hike", "提高税", "税率上调"):
                preferred_policy_type = "tax_hike"
            elif _match_any(text, "下调", "降低", "减免", "调整", "cut", "lower", "tax cut"):
                preferred_policy_type = "tax_cut"
        if preferred_policy_type is None and _match_any(text, "负面谣言", "恐慌", "panic", "selloff", "谣言", "rumor"):
            if not _match_any(text, "辟谣", "澄清", "压制", "refut", "稳定", "维稳", "rumor suppression"):
                preferred_policy_type = "tightening"

        matches: List[Tuple[str, str, str]] = []
        for policy_type, label, tokens, direction in self._PATTERNS:
            hits = [token for token in tokens if token in text or token in text_lower]
            if hits:
                matches.append((policy_type, label, direction))

        if policy_type_hint:
            hint = str(policy_type_hint).strip().lower()
            for policy_type, label, _, direction in self._PATTERNS:
                if hint == policy_type or hint == label.lower():
                    matches.insert(0, (policy_type, label, direction))
                    break

        if preferred_policy_type:
            for idx, item in enumerate(matches):
                if item[0] == preferred_policy_type:
                    matches.insert(0, matches.pop(idx))
                    break
            else:
                for policy_type, label, _, direction in self._PATTERNS:
                    if policy_type == preferred_policy_type:
                        matches.insert(0, (policy_type, label, direction))
                        break

        if matches:
            policy_type, policy_label, direction = matches[0]
        else:
            policy_type, policy_label, direction = "unclassified", "Unclassified", "neutral"
            fallback_used = True

        if len(matches) > 1 and policy_type != "unclassified":
            # Keep the first match, but lower confidence when multiple families fire.
            ambiguous = True
        else:
            ambiguous = False

        channels = self._build_channels(policy_type, base_intensity, direction, text)
        sector_effects, factor_effects, agent_class_effects, market_effects = self._build_layers(
            policy_type=policy_type,
            direction=direction,
            channels=channels,
            text=text,
            market_regime=market_regime,
        )

        confidence = 0.40 + 0.12 * len(matches) + 0.12 * len([token for token in self._PATTERNS if _match_any(text, *token[2])])
        if policy_type_hint:
            confidence += 0.08
        if fallback_used:
            confidence -= 0.18
        if ambiguous:
            confidence -= 0.12
        confidence = _clip(confidence, 0.10, 0.96)

        event = PolicyEvent(
            policy_id=f"policy-{tick}-{uuid.uuid4().hex[:8]}",
            raw_text=text,
            policy_type=policy_type,
            policy_label=policy_label,
            direction=direction,
            intensity=base_intensity,
            tick=int(tick),
            matched_tokens=self._matched_tokens(text),
            source="structured_parser" if not fallback_used else "legacy_fallback",
        )

        uncertainty = PolicyUncertainty(
            confidence=confidence,
            ambiguity=_clip(1.0 - confidence + (0.18 if ambiguous else 0.0), 0.0, 1.0),
            rumor_risk=_clip(0.25 + 0.35 * (1.0 - confidence) + (0.15 if "rumor" in text_lower or "谣言" in text else 0.0), 0.0, 1.0),
            source_reliability=_clip(0.90 if policy_type_hint else 0.75, 0.0, 1.0),
            fallback_used=bool(fallback_used),
            parser_mode="structured" if not fallback_used else "legacy",
        )

        snapshot = dict(snapshot_info or {})
        snapshot.setdefault("policy_text_length", len(text))
        snapshot.setdefault("matched_token_count", len(event.matched_tokens))
        snapshot.setdefault("market_regime", market_regime or "neutral")
        snapshot.setdefault("parser_mode", uncertainty.parser_mode)

        metadata = {
            "seed": int(self.seed),
            "config_hash": stable_json_hash(
                {
                    "seed": int(self.seed),
                    "version": self.version,
                    "policy_type_hint": policy_type_hint or "",
                    "policy_type": policy_type,
                    "policy_text": text,
                    "intensity": float(base_intensity),
                    "market_regime": market_regime or "neutral",
                    "feature_flags": dict(self.feature_flags),
                }
            ),
            "snapshot_info": snapshot,
            "feature_flags": dict(self.feature_flags),
            "parser_version": self.version,
        }

        return PolicyPackage(
            event=event,
            channels=channels,
            uncertainty=uncertainty,
            sector_effects=sector_effects,
            factor_effects=factor_effects,
            agent_class_effects=agent_class_effects,
            market_effects=market_effects,
            metadata=metadata,
            parser_version=self.version,
        )

    def _matched_tokens(self, text: str) -> List[str]:
        tokens: List[str] = []
        for _, _, keywords, _ in self._PATTERNS:
            tokens.extend([token for token in keywords if token in text or token in text.lower()])
        # Preserve order while removing duplicates.
        seen: set[str] = set()
        unique: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique.append(token)
        return unique

    def _build_channels(
        self,
        policy_type: str,
        intensity: float,
        direction: str,
        text: str,
    ) -> List[TransmissionChannel]:
        sign = -1.0 if direction == "tightening" else 1.0
        if policy_type == "tax_hike":
            sign = -1.0
        if policy_type == "unclassified":
            sign = 0.0

        def channel(name: str, impact: float, lag: int, half_life: float, macro_delta: Dict[str, float], note: str) -> TransmissionChannel:
            scaled_delta = {k: float(v * intensity * sign) for k, v in macro_delta.items()}
            return TransmissionChannel(
                name=name,
                impact=_clip(impact * intensity * sign, -1.0, 1.0),
                lag_days=int(lag),
                decay_half_life=float(half_life),
                direction="supportive" if impact * intensity * sign >= 0 else "restrictive",
                macro_delta=scaled_delta,
                note=note,
            )

        if policy_type in {"liquidity_easing", "rate_cut", "stabilization", "rumor_refutation"}:
            return [
                channel("liquidity", 0.85, 0, 7.0, {"liquidity_injection": 0.08, "sentiment_delta": 0.05}, "Liquidity support"),
                channel("rate", 0.70 if policy_type != "stabilization" else 0.35, 1, 10.0, {"policy_rate_delta": -0.0015, "credit_spread_delta": -0.0008}, "Lower discount rate"),
                channel("credit_spread", 0.55, 2, 12.0, {"credit_spread_delta": -0.0020, "liquidity_injection": 0.02}, "Credit easing"),
                channel("volatility_expectation", 0.50, 1, 6.0, {"sentiment_delta": 0.04, "rumor_shock": 0.02}, "Lower expected volatility"),
                channel("sentiment_confidence", 0.60, 0, 5.0, {"sentiment_delta": 0.08}, "Confidence repair"),
                channel("rumor_suppression", 0.40 if policy_type != "stabilization" else 0.75, 0, 4.0, {"rumor_shock": 0.12, "sentiment_delta": 0.06}, "Refutation / backstop"),
            ]

        if policy_type == "fiscal_stimulus":
            return [
                channel("fiscal_demand", 0.90, 2, 14.0, {"fiscal_stimulus_delta": 0.06, "sentiment_delta": 0.05}, "Direct demand support"),
                channel("liquidity", 0.35, 1, 8.0, {"liquidity_injection": 0.03}, "Secondary liquidity effect"),
                channel("sentiment_confidence", 0.55, 0, 6.0, {"sentiment_delta": 0.06}, "Confidence uplift"),
                channel("credit_spread", 0.20, 2, 10.0, {"credit_spread_delta": -0.0006}, "Mild credit improvement"),
            ]

        if policy_type in {"tax_cut", "tax_hike"}:
            return [
                channel("tax_frictions", 0.90, 0, 5.0, {"stamp_tax_delta": -0.0005, "liquidity_injection": 0.03, "sentiment_delta": 0.03}, "Lower / higher trading friction"),
                channel("liquidity", 0.30, 0, 6.0, {"liquidity_injection": 0.02}, "Turnover response"),
                channel("sentiment_confidence", 0.35, 0, 5.0, {"sentiment_delta": 0.04}, "Interpretation effect"),
                channel("volatility_expectation", 0.25, 1, 5.0, {"sentiment_delta": 0.02}, "Participation change"),
            ]

        if policy_type == "tightening":
            return [
                channel("rate", 0.80, 1, 10.0, {"policy_rate_delta": -0.0018, "credit_spread_delta": -0.0010}, "Higher discount rate"),
                channel("liquidity", 0.65, 0, 8.0, {"liquidity_injection": 0.07, "sentiment_delta": 0.04}, "Liquidity withdrawal"),
                channel("credit_spread", 0.55, 1, 12.0, {"credit_spread_delta": -0.0022}, "Funding stress"),
                channel("volatility_expectation", 0.55, 1, 7.0, {"sentiment_delta": 0.05, "rumor_shock": 0.03}, "Risk repricing"),
                channel("sentiment_confidence", 0.50, 0, 6.0, {"sentiment_delta": 0.07}, "Confidence decay"),
                channel("rumor_suppression", 0.20, 1, 4.0, {"rumor_shock": 0.05}, "Less supportive communication"),
            ]

        return [
            channel("sentiment_confidence", 0.0, 0, 5.0, {"sentiment_delta": 0.0}, "Neutral fallback"),
        ]

    def _build_layers(
        self,
        *,
        policy_type: str,
        direction: str,
        channels: Sequence[TransmissionChannel],
        text: str,
        market_regime: Optional[str],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        channel_map = {channel.name: channel.impact for channel in channels}

        sector_weights = {
            "financials": {"liquidity": 0.60, "rate": -0.45, "credit_spread": -0.50, "tax_frictions": -0.10, "sentiment_confidence": 0.20},
            "growth": {"liquidity": 0.70, "rate": 0.55, "fiscal_demand": 0.30, "sentiment_confidence": 0.35, "volatility_expectation": 0.10},
            "consumer": {"fiscal_demand": 0.65, "tax_frictions": 0.30, "sentiment_confidence": 0.25, "liquidity": 0.10},
            "defensive": {"volatility_expectation": -0.35, "sentiment_confidence": 0.20, "rumor_suppression": 0.25},
            "cyclical": {"liquidity": 0.40, "fiscal_demand": 0.55, "rate": 0.25, "credit_spread": -0.20},
            "property": {"rate": 0.65, "liquidity": 0.35, "credit_spread": -0.30, "sentiment_confidence": 0.15},
            "state_owned": {"stabilization": 0.60, "liquidity": 0.35, "rumor_suppression": 0.30, "sentiment_confidence": 0.25},
            "speculative": {"tax_frictions": -0.75, "volatility_expectation": -0.55, "sentiment_confidence": 0.45, "liquidity": 0.20},
        }

        factor_weights = {
            "value": {"tax_frictions": 0.25, "rate": 0.20, "credit_spread": 0.20, "volatility_expectation": 0.10},
            "growth": {"liquidity": 0.45, "rate": 0.35, "fiscal_demand": 0.30, "sentiment_confidence": 0.20},
            "momentum": {"sentiment_confidence": 0.55, "liquidity": 0.25, "rumor_suppression": 0.20},
            "quality": {"credit_spread": 0.40, "rumor_suppression": 0.20, "volatility_expectation": 0.20},
            "low_vol": {"volatility_expectation": -0.70, "rumor_suppression": 0.20, "liquidity": 0.10},
            "liquidity": {"liquidity": 0.85, "credit_spread": 0.20},
            "size": {"liquidity": 0.20, "fiscal_demand": 0.15, "tax_frictions": -0.10},
            "sentiment": {"sentiment_confidence": 0.80, "rumor_suppression": 0.45, "volatility_expectation": -0.25},
            "turnover": {"tax_frictions": -0.65, "liquidity": 0.25, "sentiment_confidence": 0.20},
        }

        agent_weights = {
            "retail": {"sentiment_confidence": 0.65, "volatility_expectation": -0.45, "tax_frictions": 0.20, "rumor_suppression": 0.10},
            "institution": {"liquidity": 0.35, "credit_spread": 0.35, "quality": 0.30, "low_vol": 0.20},
            "market_maker": {"liquidity": 0.50, "volatility_expectation": -0.70, "tax_frictions": -0.20, "credit_spread": 0.25},
            "state_stabilization": {"stabilization": 0.90, "liquidity": 0.50, "rumor_suppression": 0.70, "sentiment_confidence": 0.45},
            "rumor_trader": {"rumor_suppression": -0.80, "volatility_expectation": 0.45, "sentiment_confidence": 0.30},
            "etf_arbitrageur": {"liquidity": 0.55, "low_vol": 0.35, "rate": 0.20, "tax_frictions": -0.15},
        }

        sector_effects = {name: _layer_score(channel_map, weights) for name, weights in sector_weights.items()}
        factor_effects = {name: _layer_score(channel_map, weights) for name, weights in factor_weights.items()}
        agent_class_effects = {name: _layer_score(channel_map, weights) for name, weights in agent_weights.items()}

        # Small regime adjustments so the UI can show a distinct story by backdrop.
        regime_bias = {
            "risk_off": -0.10,
            "risk_on": 0.10,
            "inflation": -0.08,
            "liquidity": 0.12,
            "neutral": 0.0,
        }.get(str(market_regime or "neutral").lower(), 0.0)
        market_effects = {
            "market_bias": _clip(sum(channel_map.values()) + regime_bias, -1.0, 1.0),
            "liquidity_bias": _clip(channel_map.get("liquidity", 0.0), -1.0, 1.0),
            "confidence_bias": _clip(channel_map.get("sentiment_confidence", 0.0) + channel_map.get("rumor_suppression", 0.0), -1.0, 1.0),
            "volatility_bias": _clip(channel_map.get("volatility_expectation", 0.0), -1.0, 1.0),
        }
        if policy_type == "stabilization":
            market_effects["market_bias"] = _clip(market_effects["market_bias"] + 0.12, -1.0, 1.0)
        if direction == "tightening":
            market_effects["market_bias"] = _clip(market_effects["market_bias"] - 0.08, -1.0, 1.0)
        if "谣言" in text or "rumor" in text.lower():
            market_effects["volatility_bias"] = _clip(market_effects["volatility_bias"] - 0.06, -1.0, 1.0)

        return sector_effects, factor_effects, agent_class_effects, market_effects


def _layer_score(channel_map: Mapping[str, float], weights: Mapping[str, float]) -> float:
    score = 0.0
    for channel_name, weight in weights.items():
        score += float(channel_map.get(channel_name, 0.0)) * float(weight)
    return _clip(score, -1.0, 1.0)
