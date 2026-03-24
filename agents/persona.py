"""Persona model with archetype + mutable-state layers."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from agents.brain import stable_json_hash


class RiskAppetite(str, Enum):
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    AGGRESSIVE = "Aggressive"
    GAMBLER = "Gambler"


class InvestmentHorizon(str, Enum):
    SHORT_TERM = "Short-term"
    MEDIUM_TERM = "Medium-term"
    LONG_TERM = "Long-term"


@dataclass(slots=True)
class PersonaArchetype:
    key: str
    name: str
    mandate: str
    benchmark: str
    holding_period: str
    turnover_target: float
    max_drawdown: float
    liquidity_preference: float
    leverage_limit: float
    policy_channel_sensitivity: float
    rumor_sensitivity: float
    benchmark_tracking_pressure: float
    redemption_pressure: float
    inventory_limit: float
    risk_appetite: RiskAppetite
    investment_horizon: InvestmentHorizon
    conformity: float
    influence: float
    patience: float
    loss_aversion: float
    overconfidence: float
    reference_adaptivity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "mandate": self.mandate,
            "benchmark": self.benchmark,
            "holding_period": self.holding_period,
            "turnover_target": float(self.turnover_target),
            "max_drawdown": float(self.max_drawdown),
            "liquidity_preference": float(self.liquidity_preference),
            "leverage_limit": float(self.leverage_limit),
            "policy_channel_sensitivity": float(self.policy_channel_sensitivity),
            "rumor_sensitivity": float(self.rumor_sensitivity),
            "benchmark_tracking_pressure": float(self.benchmark_tracking_pressure),
            "redemption_pressure": float(self.redemption_pressure),
            "inventory_limit": float(self.inventory_limit),
            "risk_appetite": self.risk_appetite.value,
            "investment_horizon": self.investment_horizon.value,
            "conformity": float(self.conformity),
            "influence": float(self.influence),
            "patience": float(self.patience),
            "loss_aversion": float(self.loss_aversion),
            "overconfidence": float(self.overconfidence),
            "reference_adaptivity": float(self.reference_adaptivity),
        }

    def signature(self) -> str:
        return stable_json_hash(self.to_dict())

    def clone(self, **overrides: Any) -> "PersonaArchetype":
        return replace(self, **overrides)


@dataclass(slots=True)
class MutableState:
    episodic_memory: List[Dict[str, Any]] = field(default_factory=list)
    semantic_memory: Dict[str, Any] = field(default_factory=dict)
    procedural_memory: Dict[str, Any] = field(default_factory=dict)
    recent_trauma: float = 0.0
    source_credibility: float = 0.5
    policy_reaction_delay: int = 0
    attention_fatigue: float = 0.0
    narrative_anchor: str = ""
    imitation_pressure: float = 0.5
    panic_pressure: float = 0.5
    last_policy_channel: str = ""
    last_updated_at: float = 0.0

    def record_episode(
        self,
        *,
        event: str,
        outcome: float,
        content: str = "",
        timestamp: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.episodic_memory.append(
            {
                "event": event,
                "outcome": float(outcome),
                "content": content,
                "timestamp": float(timestamp if timestamp is not None else 0.0),
                "metadata": dict(metadata or {}),
            }
        )
        self.recent_trauma = float(max(0.0, min(1.0, self.recent_trauma * 0.85 + max(0.0, -outcome))))
        self.attention_fatigue = float(max(0.0, min(1.0, self.attention_fatigue * 0.95 + 0.05)))
        self.last_updated_at = float(timestamp if timestamp is not None else self.last_updated_at)

    def update_semantic(self, key: str, value: Any) -> None:
        self.semantic_memory[key] = value

    def update_procedural(self, key: str, value: Any) -> None:
        self.procedural_memory[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _make_archetype(**kwargs: Any) -> PersonaArchetype:
    return PersonaArchetype(**kwargs)


ARCHETYPE_REGISTRY: Dict[str, PersonaArchetype] = {
    "retail_day_trader": _make_archetype(
        key="retail_day_trader",
        name="Retail Day Trader",
        mandate="Exploit intraday volatility and momentum.",
        benchmark="cash + short-term momentum benchmark",
        holding_period="intraday",
        turnover_target=4.8,
        max_drawdown=0.12,
        liquidity_preference=0.35,
        leverage_limit=1.2,
        policy_channel_sensitivity=0.45,
        rumor_sensitivity=0.85,
        benchmark_tracking_pressure=0.12,
        redemption_pressure=0.05,
        inventory_limit=0.15,
        risk_appetite=RiskAppetite.AGGRESSIVE,
        investment_horizon=InvestmentHorizon.SHORT_TERM,
        conformity=0.55,
        influence=0.18,
        patience=0.18,
        loss_aversion=1.35,
        overconfidence=0.62,
        reference_adaptivity=0.25,
    ),
    "retail_swing": _make_archetype(
        key="retail_swing",
        name="Retail Swing",
        mandate="Capture multi-day theme and policy repricing.",
        benchmark="CSI 300",
        holding_period="3-20 trading days",
        turnover_target=1.6,
        max_drawdown=0.16,
        liquidity_preference=0.48,
        leverage_limit=1.0,
        policy_channel_sensitivity=0.62,
        rumor_sensitivity=0.55,
        benchmark_tracking_pressure=0.22,
        redemption_pressure=0.08,
        inventory_limit=0.22,
        risk_appetite=RiskAppetite.BALANCED,
        investment_horizon=InvestmentHorizon.MEDIUM_TERM,
        conformity=0.58,
        influence=0.22,
        patience=0.46,
        loss_aversion=2.0,
        overconfidence=0.32,
        reference_adaptivity=0.45,
    ),
    "mutual_fund": _make_archetype(
        key="mutual_fund",
        name="Mutual Fund",
        mandate="Beat benchmark with controlled tracking error.",
        benchmark="CSI 300",
        holding_period="1-4 quarters",
        turnover_target=0.42,
        max_drawdown=0.10,
        liquidity_preference=0.74,
        leverage_limit=1.1,
        policy_channel_sensitivity=0.52,
        rumor_sensitivity=0.12,
        benchmark_tracking_pressure=0.88,
        redemption_pressure=0.35,
        inventory_limit=0.45,
        risk_appetite=RiskAppetite.BALANCED,
        investment_horizon=InvestmentHorizon.LONG_TERM,
        conformity=0.34,
        influence=0.56,
        patience=0.74,
        loss_aversion=2.15,
        overconfidence=0.12,
        reference_adaptivity=0.28,
    ),
    "pension_fund": _make_archetype(
        key="pension_fund",
        name="Pension Fund",
        mandate="Preserve capital and match long-term liabilities.",
        benchmark="liability-matched portfolio",
        holding_period="multi-year",
        turnover_target=0.12,
        max_drawdown=0.07,
        liquidity_preference=0.86,
        leverage_limit=0.6,
        policy_channel_sensitivity=0.72,
        rumor_sensitivity=0.08,
        benchmark_tracking_pressure=0.72,
        redemption_pressure=0.18,
        inventory_limit=0.30,
        risk_appetite=RiskAppetite.CONSERVATIVE,
        investment_horizon=InvestmentHorizon.LONG_TERM,
        conformity=0.28,
        influence=0.42,
        patience=0.88,
        loss_aversion=2.45,
        overconfidence=0.08,
        reference_adaptivity=0.18,
    ),
    "insurer": _make_archetype(
        key="insurer",
        name="Insurer",
        mandate="Protect solvency while earning spread carry.",
        benchmark="solvency-adjusted asset mix",
        holding_period="multi-quarter",
        turnover_target=0.18,
        max_drawdown=0.06,
        liquidity_preference=0.90,
        leverage_limit=0.5,
        policy_channel_sensitivity=0.68,
        rumor_sensitivity=0.10,
        benchmark_tracking_pressure=0.58,
        redemption_pressure=0.22,
        inventory_limit=0.25,
        risk_appetite=RiskAppetite.CONSERVATIVE,
        investment_horizon=InvestmentHorizon.LONG_TERM,
        conformity=0.26,
        influence=0.38,
        patience=0.84,
        loss_aversion=2.55,
        overconfidence=0.06,
        reference_adaptivity=0.16,
    ),
    "prop_desk": _make_archetype(
        key="prop_desk",
        name="Prop Desk",
        mandate="Harvest short-horizon alpha and microstructure edge.",
        benchmark="intraday alpha basket",
        holding_period="minutes-hours",
        turnover_target=5.6,
        max_drawdown=0.14,
        liquidity_preference=0.42,
        leverage_limit=2.6,
        policy_channel_sensitivity=0.34,
        rumor_sensitivity=0.30,
        benchmark_tracking_pressure=0.10,
        redemption_pressure=0.03,
        inventory_limit=0.40,
        risk_appetite=RiskAppetite.AGGRESSIVE,
        investment_horizon=InvestmentHorizon.SHORT_TERM,
        conformity=0.18,
        influence=0.72,
        patience=0.22,
        loss_aversion=1.55,
        overconfidence=0.58,
        reference_adaptivity=0.22,
    ),
    "market_maker": _make_archetype(
        key="market_maker",
        name="Market Maker",
        mandate="Provide liquidity and manage inventory.",
        benchmark="spread capture + inventory risk budget",
        holding_period="seconds-hours",
        turnover_target=8.5,
        max_drawdown=0.08,
        liquidity_preference=0.98,
        leverage_limit=1.8,
        policy_channel_sensitivity=0.28,
        rumor_sensitivity=0.15,
        benchmark_tracking_pressure=0.06,
        redemption_pressure=0.02,
        inventory_limit=0.85,
        risk_appetite=RiskAppetite.BALANCED,
        investment_horizon=InvestmentHorizon.SHORT_TERM,
        conformity=0.14,
        influence=0.88,
        patience=0.36,
        loss_aversion=1.85,
        overconfidence=0.10,
        reference_adaptivity=0.14,
    ),
    "state_stabilization_fund": _make_archetype(
        key="state_stabilization_fund",
        name="State Stabilization Fund",
        mandate="Stabilize market dislocation and anchor confidence.",
        benchmark="market stability objective",
        holding_period="policy window",
        turnover_target=0.24,
        max_drawdown=0.05,
        liquidity_preference=0.95,
        leverage_limit=0.8,
        policy_channel_sensitivity=0.98,
        rumor_sensitivity=0.02,
        benchmark_tracking_pressure=0.04,
        redemption_pressure=0.00,
        inventory_limit=0.50,
        risk_appetite=RiskAppetite.CONSERVATIVE,
        investment_horizon=InvestmentHorizon.LONG_TERM,
        conformity=0.10,
        influence=0.95,
        patience=0.92,
        loss_aversion=2.75,
        overconfidence=0.04,
        reference_adaptivity=0.08,
    ),
    "rumor_trader": _make_archetype(
        key="rumor_trader",
        name="Rumor Trader",
        mandate="Trade on rumor flow and narrative momentum.",
        benchmark="rumor-adjusted alpha bucket",
        holding_period="minutes-days",
        turnover_target=3.9,
        max_drawdown=0.18,
        liquidity_preference=0.24,
        leverage_limit=1.5,
        policy_channel_sensitivity=0.28,
        rumor_sensitivity=0.96,
        benchmark_tracking_pressure=0.08,
        redemption_pressure=0.10,
        inventory_limit=0.18,
        risk_appetite=RiskAppetite.GAMBLER,
        investment_horizon=InvestmentHorizon.SHORT_TERM,
        conformity=0.74,
        influence=0.25,
        patience=0.16,
        loss_aversion=1.1,
        overconfidence=0.80,
        reference_adaptivity=0.30,
    ),
    "etf_arbitrageur": _make_archetype(
        key="etf_arbitrageur",
        name="ETF Arbitrageur",
        mandate="Maintain ETF/NAV spread parity.",
        benchmark="ETF tracking basket",
        holding_period="intraday",
        turnover_target=2.8,
        max_drawdown=0.09,
        liquidity_preference=0.88,
        leverage_limit=1.4,
        policy_channel_sensitivity=0.40,
        rumor_sensitivity=0.20,
        benchmark_tracking_pressure=0.94,
        redemption_pressure=0.12,
        inventory_limit=0.32,
        risk_appetite=RiskAppetite.BALANCED,
        investment_horizon=InvestmentHorizon.SHORT_TERM,
        conformity=0.22,
        influence=0.50,
        patience=0.40,
        loss_aversion=1.9,
        overconfidence=0.18,
        reference_adaptivity=0.24,
    ),
}

_DEFAULT_ARCHETYPE_KEY = "retail_swing"


def list_archetype_keys() -> List[str]:
    return list(ARCHETYPE_REGISTRY.keys())


def get_archetype(key: str) -> PersonaArchetype:
    base = ARCHETYPE_REGISTRY.get(str(key), ARCHETYPE_REGISTRY[_DEFAULT_ARCHETYPE_KEY])
    return base.clone()


def _legacy_archetype_from_fields(
    *,
    name: str,
    risk_appetite: RiskAppetite,
    investment_horizon: InvestmentHorizon,
    conformity: float,
    influence: float,
    patience: float,
    loss_aversion: float,
    overconfidence: float,
    reference_adaptivity: float,
    purchase_anchor_weight: float,
    recent_high_anchor_weight: float,
    peer_anchor_weight: float,
    policy_anchor_weight: float,
) -> PersonaArchetype:
    return PersonaArchetype(
        key="legacy",
        name=name or "Anonymous",
        mandate="Legacy compatibility profile",
        benchmark="legacy benchmark",
        holding_period="legacy",
        turnover_target=1.0,
        max_drawdown=0.20,
        liquidity_preference=0.50,
        leverage_limit=1.0,
        policy_channel_sensitivity=_clip(policy_anchor_weight + 0.1, 0.0, 1.0),
        rumor_sensitivity=_clip(peer_anchor_weight + 0.1, 0.0, 1.0),
        benchmark_tracking_pressure=_clip(0.2 + recent_high_anchor_weight, 0.0, 1.0),
        redemption_pressure=_clip(0.2 + purchase_anchor_weight * 0.25, 0.0, 1.0),
        inventory_limit=0.25,
        risk_appetite=risk_appetite,
        investment_horizon=investment_horizon,
        conformity=_clip(conformity, 0.0, 1.0),
        influence=_clip(influence, 0.0, 1.0),
        patience=_clip(patience, 0.0, 1.0),
        loss_aversion=max(0.5, float(loss_aversion)),
        overconfidence=_clip(overconfidence, 0.0, 1.0),
        reference_adaptivity=_clip(reference_adaptivity, 0.0, 1.0),
    )


@dataclass
class Persona:
    """Compatibility wrapper around archetype + mutable state."""

    name: str = "Anonymous"
    risk_appetite: RiskAppetite = RiskAppetite.BALANCED
    investment_horizon: InvestmentHorizon = InvestmentHorizon.MEDIUM_TERM
    conformity: float = 0.5
    influence: float = 0.5
    patience: float = 0.5
    loss_aversion: float = 2.25
    overconfidence: float = 0.0
    reference_adaptivity: float = 0.5
    purchase_anchor_weight: float = 0.34
    recent_high_anchor_weight: float = 0.26
    peer_anchor_weight: float = 0.20
    policy_anchor_weight: float = 0.20
    archetype_key: Optional[str] = None
    archetype: Optional[PersonaArchetype] = None
    mutable_state: MutableState = field(default_factory=MutableState)

    def __post_init__(self) -> None:
        if self.archetype is None:
            if self.archetype_key and self.archetype_key in ARCHETYPE_REGISTRY:
                self.archetype = get_archetype(self.archetype_key).clone(
                    risk_appetite=self.risk_appetite,
                    investment_horizon=self.investment_horizon,
                    conformity=self.conformity,
                    influence=self.influence,
                    patience=self.patience,
                    loss_aversion=self.loss_aversion,
                    overconfidence=self.overconfidence,
                    reference_adaptivity=self.reference_adaptivity,
                )
            else:
                self.archetype = _legacy_archetype_from_fields(
                    name=self.name,
                    risk_appetite=self.risk_appetite,
                    investment_horizon=self.investment_horizon,
                    conformity=self.conformity,
                    influence=self.influence,
                    patience=self.patience,
                    loss_aversion=self.loss_aversion,
                    overconfidence=self.overconfidence,
                    reference_adaptivity=self.reference_adaptivity,
                    purchase_anchor_weight=self.purchase_anchor_weight,
                    recent_high_anchor_weight=self.recent_high_anchor_weight,
                    peer_anchor_weight=self.peer_anchor_weight,
                    policy_anchor_weight=self.policy_anchor_weight,
                )
        self.archetype_key = str(self.archetype.key if self.archetype else self.archetype_key or _DEFAULT_ARCHETYPE_KEY)
        if self.archetype:
            self.risk_appetite = self.archetype.risk_appetite
            self.investment_horizon = self.archetype.investment_horizon
            self.conformity = self.archetype.conformity
            self.influence = self.archetype.influence
            self.patience = self.archetype.patience
            self.loss_aversion = self.archetype.loss_aversion
            self.overconfidence = self.archetype.overconfidence
            self.reference_adaptivity = self.archetype.reference_adaptivity

        if not isinstance(self.mutable_state, MutableState):
            self.mutable_state = MutableState(**dict(self.mutable_state or {}))

    @classmethod
    def from_archetype(
        cls,
        archetype_key: str,
        *,
        name: Optional[str] = None,
        mutable_state: Optional[MutableState] = None,
        **legacy_overrides: Any,
    ) -> "Persona":
        archetype = get_archetype(archetype_key)
        legacy = {
            "risk_appetite": archetype.risk_appetite,
            "investment_horizon": archetype.investment_horizon,
            "conformity": archetype.conformity,
            "influence": archetype.influence,
            "patience": archetype.patience,
            "loss_aversion": archetype.loss_aversion,
            "overconfidence": archetype.overconfidence,
            "reference_adaptivity": archetype.reference_adaptivity,
            "purchase_anchor_weight": 0.34,
            "recent_high_anchor_weight": 0.26,
            "peer_anchor_weight": 0.20,
            "policy_anchor_weight": 0.20,
        }
        legacy.update(legacy_overrides)
        return cls(
            name=name or archetype.name,
            archetype_key=archetype.key,
            archetype=archetype,
            mutable_state=mutable_state or MutableState(),
            **legacy,
        )

    def __str__(self) -> str:
        return (
            f"Persona({self.name} | {self.archetype_key} | {self.risk_appetite.value} | "
            f"{self.investment_horizon.value} | Conformity: {self.conformity:.2f})"
        )

    def base_risk_score(self) -> float:
        mapping = {
            RiskAppetite.CONSERVATIVE: 0.25,
            RiskAppetite.BALANCED: 0.50,
            RiskAppetite.AGGRESSIVE: 0.72,
            RiskAppetite.GAMBLER: 0.86,
        }
        return float(mapping.get(self.risk_appetite, 0.50))

    def reference_weights(self) -> Dict[str, float]:
        return {
            "purchase_anchor": float(self.purchase_anchor_weight),
            "recent_high_anchor": float(self.recent_high_anchor_weight),
            "peer_anchor": float(self.peer_anchor_weight),
            "policy_anchor": float(self.policy_anchor_weight),
        }

    def reference_shift_profile(self) -> Dict[str, float]:
        adapt = max(0.05, min(0.95, float(self.reference_adaptivity)))
        return {
            "purchase_decay": 0.02 + 0.05 * adapt,
            "recent_high_decay": 0.08 + 0.10 * adapt,
            "peer_decay": 0.10 + 0.18 * adapt,
            "policy_decay": 0.10 + 0.20 * adapt,
            "policy_sensitivity": 0.08 + 0.25 * adapt,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "archetype_key": self.archetype_key,
            "archetype": self.archetype.to_dict() if self.archetype else {},
            "mutable_state": self.mutable_state.to_dict(),
            "risk_appetite": self.risk_appetite.value,
            "investment_horizon": self.investment_horizon.value,
            "conformity": float(self.conformity),
            "influence": float(self.influence),
            "patience": float(self.patience),
            "loss_aversion": float(self.loss_aversion),
            "overconfidence": float(self.overconfidence),
            "reference_adaptivity": float(self.reference_adaptivity),
            "purchase_anchor_weight": float(self.purchase_anchor_weight),
            "recent_high_anchor_weight": float(self.recent_high_anchor_weight),
            "peer_anchor_weight": float(self.peer_anchor_weight),
            "policy_anchor_weight": float(self.policy_anchor_weight),
        }

    def stable_signature(self) -> str:
        return stable_json_hash(self.to_dict())


def _holding_period_to_days(holding_period: str) -> float:
    text = str(holding_period).lower()
    if "intraday" in text or "seconds" in text or "minutes" in text:
        return 0.5
    if "day" in text:
        return 1.0
    if "week" in text:
        return 5.0
    if "quarter" in text:
        return 63.0
    if "year" in text:
        return 252.0
    if "multi-year" in text:
        return 504.0
    return 20.0


class PersonaGenerator:
    """Structured persona generation with legacy fallback."""

    FIRST_NAMES = [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eve",
        "Frank",
        "Grace",
        "Heidi",
        "Ivan",
        "Judy",
        "Kevin",
        "Lily",
        "Mike",
        "Nina",
        "Oscar",
        "Peggy",
    ]

    @staticmethod
    def _resolve_composition_weights(
        regime: str = "default",
        composition: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        if not composition:
            keys = list_archetype_keys()
            weight = 1.0 / max(len(keys), 1)
            return {key: weight for key in keys}
        regimes = dict(composition.get("regimes") or {})
        regime_spec = dict(regimes.get(regime) or regimes.get("default") or {})
        weights = dict(regime_spec.get("weights") or {})
        if not weights:
            keys = list_archetype_keys()
            weight = 1.0 / max(len(keys), 1)
            return {key: weight for key in keys}
        total = sum(float(v) for v in weights.values()) or 1.0
        normalized = {str(k): float(v) / total for k, v in weights.items()}
        for key in list_archetype_keys():
            normalized.setdefault(key, 0.0)
        return normalized

    @staticmethod
    def _sample_archetype_key(rng: random.Random, weights: Mapping[str, float]) -> str:
        items = [(key, max(0.0, float(weight))) for key, weight in weights.items() if key in ARCHETYPE_REGISTRY]
        if not items:
            return _DEFAULT_ARCHETYPE_KEY
        total = sum(weight for _, weight in items) or 1.0
        draw = rng.random() * total
        running = 0.0
        for key, weight in items:
            running += weight
            if draw <= running:
                return key
        return items[-1][0]

    @staticmethod
    def _generate_mutable_state(rng: random.Random) -> MutableState:
        return MutableState(
            recent_trauma=_clip(rng.betavariate(1.5, 5.0), 0.0, 1.0),
            source_credibility=_clip(rng.betavariate(4.0, 2.0), 0.0, 1.0),
            policy_reaction_delay=int(rng.choice([0, 1, 2, 3])),
            attention_fatigue=_clip(rng.betavariate(2.0, 3.0), 0.0, 1.0),
            narrative_anchor=rng.choice(["policy", "valuation", "liquidity", "rumor", "momentum", "mean-reversion"]),
            imitation_pressure=_clip(rng.betavariate(2.5, 2.5), 0.0, 1.0),
            panic_pressure=_clip(rng.betavariate(2.0, 3.5), 0.0, 1.0),
            last_updated_at=0.0,
        )

    @staticmethod
    def _generate_legacy_persona(rng: random.Random) -> Persona:
        name = f"{rng.choice(PersonaGenerator.FIRST_NAMES)}_{rng.randint(100, 999)}"
        risk = rng.choice(list(RiskAppetite))
        horizon = rng.choice(list(InvestmentHorizon))
        conformity = rng.betavariate(2, 2)
        influence = min(0.95, rng.paretovariate(3) / 5.0)
        if risk == RiskAppetite.CONSERVATIVE:
            loss_aversion = rng.uniform(2.5, 3.5)
            patience = rng.uniform(0.6, 0.9)
        elif risk == RiskAppetite.GAMBLER:
            loss_aversion = rng.uniform(1.0, 1.5)
            patience = rng.uniform(0.1, 0.4)
            conformity = rng.uniform(0.6, 0.9)
        else:
            loss_aversion = rng.uniform(2.0, 2.5)
            patience = rng.uniform(0.3, 0.7)
        return Persona(
            name=name,
            risk_appetite=risk,
            investment_horizon=horizon,
            conformity=conformity,
            influence=influence,
            patience=patience,
            loss_aversion=loss_aversion,
        )

    @staticmethod
    def generate_random_persona(
        *,
        regime: str = "default",
        composition_path: Optional[str | Path] = None,
        seed: Optional[int] = None,
    ) -> Persona:
        if not _env_flag("CIVITAS_POPULATION_PROTOCOL_V1", True):
            rng = random.Random(seed) if seed is not None else random
            return PersonaGenerator._generate_legacy_persona(rng)

        rng = random.Random(seed) if seed is not None else random
        composition = PersonaGenerator.load_market_composition(composition_path)
        weights = PersonaGenerator._resolve_composition_weights(regime=regime, composition=composition)
        archetype_key = PersonaGenerator._sample_archetype_key(rng, weights)
        name = f"{PersonaGenerator.FIRST_NAMES[rng.randint(0, len(PersonaGenerator.FIRST_NAMES) - 1)]}_{rng.randint(100, 999)}"
        persona = Persona.from_archetype(
            archetype_key,
            name=name,
            mutable_state=PersonaGenerator._generate_mutable_state(rng),
        )
        persona.mutable_state.semantic_memory.update({"market_regime": regime, "archetype_key": archetype_key})
        return persona

    @staticmethod
    def generate_distribution(
        n: int,
        *,
        regime: str = "default",
        composition_path: Optional[str | Path] = None,
        seed: Optional[int] = None,
    ) -> List[Persona]:
        rng = random.Random(seed) if seed is not None else random
        composition = PersonaGenerator.load_market_composition(composition_path)
        weights = PersonaGenerator._resolve_composition_weights(regime=regime, composition=composition)
        personas: List[Persona] = []
        for idx in range(n):
            if not _env_flag("CIVITAS_POPULATION_PROTOCOL_V1", True):
                personas.append(PersonaGenerator._generate_legacy_persona(rng))
                continue
            archetype_key = PersonaGenerator._sample_archetype_key(rng, weights)
            persona = Persona.from_archetype(
                archetype_key,
                name=f"{archetype_key}_{idx:03d}",
                mutable_state=PersonaGenerator._generate_mutable_state(rng),
            )
            persona.mutable_state.semantic_memory.update({"market_regime": regime, "archetype_key": archetype_key})
            personas.append(persona)
        return personas

    @staticmethod
    def load_market_composition(path: Optional[str | Path] = None) -> Dict[str, Any]:
        default_json = Path("data") / "market_composition.json"
        default_yaml = Path("data") / "market_composition.yaml"
        candidate_paths = []
        if path is not None:
            candidate_paths.append(Path(path))
        candidate_paths.extend([default_yaml, default_json])

        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            text = candidate.read_text(encoding="utf-8")
            if candidate.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml  # type: ignore
                except Exception:
                    continue
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
            if isinstance(data, dict):
                data.setdefault("source_path", str(candidate))
                return data

        return {
            "version": "1.0",
            "source_path": "builtin",
            "archetypes": {key: spec.to_dict() for key, spec in ARCHETYPE_REGISTRY.items()},
            "regimes": {
                "default": {"weights": {key: 1.0 / len(ARCHETYPE_REGISTRY) for key in ARCHETYPE_REGISTRY}},
            },
        }

    @staticmethod
    def build_market_distribution_report(
        personas: List[Persona],
        *,
        regime: str = "default",
        seed: int = 0,
        composition_path: Optional[str | Path] = None,
        snapshot: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        holding_periods: List[float] = []
        turnover_targets: List[float] = []
        max_drawdowns: List[float] = []
        risk_budgets: List[float] = []
        leverage_limits: List[float] = []
        for persona in personas:
            key = getattr(persona.archetype, "key", persona.archetype_key)
            counts[key] = counts.get(key, 0) + 1
            archetype = persona.archetype or get_archetype(key)
            holding_periods.append(float(_holding_period_to_days(archetype.holding_period)))
            turnover_targets.append(float(archetype.turnover_target))
            max_drawdowns.append(float(archetype.max_drawdown))
            risk_budgets.append(float(max(0.0, 1.0 - archetype.max_drawdown)))
            leverage_limits.append(float(archetype.leverage_limit))

        total = len(personas)
        denom = max(total, 1)
        composition = PersonaGenerator.load_market_composition(composition_path)
        regime_weights = PersonaGenerator._resolve_composition_weights(regime=regime, composition=composition)
        config_payload = {
            "regime": regime,
            "seed": int(seed),
            "composition_path": str(composition_path) if composition_path else str(composition.get("source_path", "")),
            "regime_weights": regime_weights,
        }
        report = {
            "feature_flag": _env_flag("CIVITAS_POPULATION_PROTOCOL_V1", True),
            "seed": int(seed),
            "config_hash": stable_json_hash(config_payload),
            "snapshot": dict(snapshot or {}),
            "regime": regime,
            "composition_path": str(composition_path) if composition_path else str(composition.get("source_path", "")),
            "total_agents": total,
            "counts": counts,
            "shares": {key: round(count / denom, 6) for key, count in counts.items()},
            "regime_weights": regime_weights,
            "averages": {
                "holding_period_days": float(sum(holding_periods) / len(holding_periods)) if holding_periods else 0.0,
                "turnover_target": float(sum(turnover_targets) / len(turnover_targets)) if turnover_targets else 0.0,
                "max_drawdown": float(sum(max_drawdowns) / len(max_drawdowns)) if max_drawdowns else 0.0,
                "risk_budget": float(sum(risk_budgets) / len(risk_budgets)) if risk_budgets else 0.0,
                "leverage_limit": float(sum(leverage_limits) / len(leverage_limits)) if leverage_limits else 0.0,
            },
            "archetype_profiles": {
                key: ARCHETYPE_REGISTRY[key].to_dict() for key in sorted(counts.keys()) if key in ARCHETYPE_REGISTRY
            },
            "snapshot_id": stable_json_hash(
                {
                    "regime": regime,
                    "counts": counts,
                    "seed": int(seed),
                    "total_agents": total,
                }
            ),
        }
        report["markdown"] = PersonaGenerator.render_market_distribution_report(report)
        return report

    @staticmethod
    def render_market_distribution_report(report: Mapping[str, Any]) -> str:
        counts = dict(report.get("counts") or {})
        shares = dict(report.get("shares") or {})
        averages = dict(report.get("averages") or {})
        lines = [
            "# Market Persona Distribution Report",
            f"- Regime: {report.get('regime', 'default')}",
            f"- Seed: {report.get('seed', 0)}",
            f"- Config hash: {report.get('config_hash', '')}",
            f"- Snapshot: {report.get('snapshot_id', '')}",
            f"- Total agents: {report.get('total_agents', 0)}",
            "",
            "## Archetype mix",
        ]
        for key in sorted(counts.keys()):
            lines.append(f"- {key}: {counts[key]} ({shares.get(key, 0.0):.1%})")
        lines.extend(
            [
                "",
                "## Average constraints",
                f"- Holding period (days): {averages.get('holding_period_days', 0.0):.2f}",
                f"- Turnover target: {averages.get('turnover_target', 0.0):.2f}",
                f"- Max drawdown: {averages.get('max_drawdown', 0.0):.2%}",
                f"- Risk budget: {averages.get('risk_budget', 0.0):.2%}",
                f"- Leverage limit: {averages.get('leverage_limit', 0.0):.2f}",
            ]
        )
        return "\n".join(lines)


__all__ = [
    "ARCHETYPE_REGISTRY",
    "InvestmentHorizon",
    "MutableState",
    "Persona",
    "PersonaArchetype",
    "PersonaGenerator",
    "RiskAppetite",
    "get_archetype",
    "list_archetype_keys",
]
