"""Policy interpretation layer: structured policy package -> per-agent belief."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from policy.structured import PolicyPackage


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class AgentBelief:
    """Belief object consumed by trading agents."""

    expected_return: Dict[str, float]
    expected_risk: Dict[str, float]
    liquidity_score: Dict[str, float]
    confidence: float
    latency_bars: int
    disagreement_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyInterpretationEngine:
    """Map a single policy package into heterogeneous agent beliefs."""

    def __init__(
        self,
        *,
        default_symbols: Optional[Sequence[str]] = None,
        symbol_sector_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.default_symbols = list(default_symbols or ["A_SHARE_IDX"])
        self.symbol_sector_map = dict(symbol_sector_map or {})

    def _resolve_symbols(self, market_state: Optional[Mapping[str, Any]]) -> List[str]:
        if not market_state:
            return list(self.default_symbols)
        if isinstance(market_state.get("symbols"), list) and market_state.get("symbols"):
            return [str(symbol) for symbol in market_state.get("symbols", [])]
        prices = market_state.get("prices")
        if isinstance(prices, Mapping) and prices:
            return [str(symbol) for symbol in prices.keys()]
        return list(self.default_symbols)

    def _persona_key(self, persona: Any) -> str:
        key = str(getattr(persona, "archetype_key", "") or "").strip()
        if key:
            return key
        archetype = getattr(persona, "archetype", None)
        if archetype is not None:
            key = str(getattr(archetype, "key", "") or "").strip()
            if key:
                return key
        return "retail"

    def _persona_policy_sensitivity(self, persona: Any) -> float:
        archetype = getattr(persona, "archetype", None)
        if archetype is None:
            return 0.5
        value = getattr(archetype, "policy_channel_sensitivity", 0.5)
        return float(_clip(value, 0.0, 1.0))

    def _persona_latency(self, persona: Any) -> int:
        archetype = getattr(persona, "archetype", None)
        mutable_state = getattr(persona, "mutable_state", None)
        delay = int(getattr(mutable_state, "policy_reaction_delay", 0) or 0)
        if archetype is None:
            return max(0, delay)
        bars = int(getattr(archetype, "info_latency_bars", 1) or 1)
        return max(0, bars + delay)

    def _persona_liquidity_pref(self, persona: Any) -> float:
        archetype = getattr(persona, "archetype", None)
        if archetype is None:
            return 0.5
        return float(_clip(getattr(archetype, "liquidity_preference", 0.5), 0.0, 1.0))

    def _resolve_sector(self, symbol: str) -> str:
        if symbol in self.symbol_sector_map:
            return str(self.symbol_sector_map[symbol])
        token = symbol.lower()
        if "bank" in token or "fin" in token:
            return "financials"
        if "chip" in token or "ai" in token or "tech" in token:
            return "growth"
        if "real" in token or "prop" in token:
            return "property"
        if "energy" in token or "new" in token:
            return "cyclical"
        if "cons" in token:
            return "consumer"
        return "state_owned" if token in {"a_share_idx", "index", "idx"} else "defensive"

    def _agent_effect_for_persona(
        self,
        package: PolicyPackage,
        persona_key: str,
    ) -> float:
        effects = dict(package.agent_class_effects or {})
        if persona_key in effects:
            return float(effects[persona_key])
        alias_map = {
            "retail_general": "retail",
            "retail_value": "retail",
            "retail_emotional": "retail",
            "retail_day_trader": "retail",
            "retail_swing": "retail",
            "retail_momentum_chaser": "retail",
            "retail_mean_reverter": "retail",
            "private_fund_discretionary": "institution",
            "mutual_fund": "institution",
            "pension_fund": "institution",
            "insurer": "institution",
            "long_term_institution": "institution",
            "quant_stock_selector": "quant",
            "quant_timing": "quant",
            "quant_arbitrage": "quant",
            "trend_trader": "quant",
            "event_driven_capital": "institution",
            "state_stabilization_fund": "state_stabilization",
            "policy_capital": "policy_capital",
            "market_maker": "market_maker",
            "media_propagator": "rumor_trader",
        }
        alias = alias_map.get(persona_key, "")
        if alias and alias in effects:
            return float(effects[alias])
        if "retail" in persona_key and "retail" in effects:
            return float(effects["retail"])
        if "quant" in persona_key and "quant" in effects:
            return float(effects["quant"])
        if "maker" in persona_key and "market_maker" in effects:
            return float(effects["market_maker"])
        if "stabilization" in persona_key and "state_stabilization" in effects:
            return float(effects["state_stabilization"])
        return 0.0

    def interpret(
        self,
        policy_pkg: PolicyPackage,
        persona: Any,
        market_state: Optional[Mapping[str, Any]] = None,
        portfolio: Optional[Mapping[str, Any]] = None,
    ) -> AgentBelief:
        symbols = self._resolve_symbols(market_state)
        market_effects = dict(policy_pkg.market_effects or {})
        persona_key = self._persona_key(persona)
        agent_bias = self._agent_effect_for_persona(policy_pkg, persona_key)
        policy_confidence = float(getattr(policy_pkg.uncertainty, "confidence", 0.5))
        sensitivity = self._persona_policy_sensitivity(persona)
        confidence = _clip(policy_confidence * (0.7 + 0.6 * sensitivity), 0.05, 0.99)
        liquidity_pref = self._persona_liquidity_pref(persona)
        latency_bars = self._persona_latency(persona)
        market_bias = float(market_effects.get("market_bias", 0.0))
        liquidity_bias = float(market_effects.get("liquidity_bias", 0.0))
        volatility_bias = float(market_effects.get("volatility_bias", 0.0))

        expected_return: Dict[str, float] = {}
        expected_risk: Dict[str, float] = {}
        liquidity_score: Dict[str, float] = {}
        disagreement_tags: List[str] = []

        for symbol in symbols:
            sector = self._resolve_sector(symbol)
            sector_bias = float((policy_pkg.sector_effects or {}).get(sector, 0.0))
            value = 0.45 * market_bias + 0.35 * agent_bias + 0.20 * sector_bias
            expected_return[symbol] = _clip(value * confidence, -0.30, 0.30)

            risk_value = 0.18 + 0.42 * abs(volatility_bias) + 0.22 * (1.0 - confidence)
            expected_risk[symbol] = _clip(risk_value, 0.01, 1.0)

            liq = 0.50 + 0.40 * liquidity_bias - 0.25 * liquidity_pref
            liquidity_score[symbol] = _clip(liq, 0.0, 1.0)

        if confidence < 0.35:
            disagreement_tags.append("low_policy_confidence")
        if abs(agent_bias) < 0.05:
            disagreement_tags.append("weak_agent_specific_signal")
        if volatility_bias < -0.15:
            disagreement_tags.append("tail_risk_elevated")
        if latency_bars > 2:
            disagreement_tags.append("slow_information_processing")
        if isinstance(portfolio, Mapping) and portfolio:
            disagreement_tags.append("position_constraint_active")

        metadata = {
            "persona_key": persona_key,
            "agent_bias": float(agent_bias),
            "market_bias": float(market_bias),
            "volatility_bias": float(volatility_bias),
            "liquidity_bias": float(liquidity_bias),
            "policy_id": getattr(policy_pkg.event, "policy_id", ""),
        }
        return AgentBelief(
            expected_return=expected_return,
            expected_risk=expected_risk,
            liquidity_score=liquidity_score,
            confidence=float(confidence),
            latency_bars=int(latency_bars),
            disagreement_tags=disagreement_tags,
            metadata=metadata,
        )

    def batch_interpret(
        self,
        policy_pkg: Optional[PolicyPackage],
        agents: Iterable[Any],
        market_state: Optional[Mapping[str, Any]] = None,
    ) -> List[AgentBelief]:
        if policy_pkg is None:
            return []
        beliefs: List[AgentBelief] = []
        for agent in agents:
            persona = getattr(agent, "persona", None)
            portfolio = getattr(agent, "portfolio", None)
            beliefs.append(self.interpret(policy_pkg, persona, market_state, portfolio))
        return beliefs

