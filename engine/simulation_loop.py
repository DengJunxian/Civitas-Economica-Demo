"""Core simulation loop with macro-social-micro coupling."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.execution_adapter import ExecutionAdapter
from agents.trading_agent_core import GLOBAL_CONFIG, Persona, TradingAgent
from core.market_metrics import MarketMetrics
from core.market_state import MarketState as UnifiedMarketState
from core.types import ExecutionPlan, Order, OrderSide
from core.behavioral_finance import (
    StylizedFactsTracker,
    behavioral_update_step,
    calculate_csad,
    initialize_reference_points,
)
from core.exchange.evolution import (
    EcologyMetricsRow,
    EcologyMetricsTracker,
    EvolutionOperators,
    StrategyGenome,
    approx_modularity,
    build_sentiment_coalitions,
    coalition_persistence,
    entropy_from_labels,
    hhi_from_shares,
    phase_change_score,
)
from core.macro.bank import BankAgent
from core.macro.firm import FirmAgent, FirmSignal
from core.macro.government import GovernmentAgent, PolicyShock
from core.macro.household import HouseholdAgent, HouseholdSignal
from core.macro.state import MacroContextDTO, MacroState
from core.market_engine import blend_price_with_backdrop
from core.regulatory_sandbox import MarketAbuseSandbox
from core.runtime_mode import RuntimeModeProfile, resolve_runtime_mode_profile
from core.social.contagion import ContagionSnapshot, SocialContagionEngine
from core.social.graph_state import SocialGraphState
from core.world.event_bus import EventBus
from engine.market_match import calculate_new_price
from policy.policy_engine import PolicyEngine
from policy.interpretation_engine import AgentBelief, PolicyInterpretationEngine
from simulation_runner import BufferedIntent, ExogenousBackdropPoint, SimulationRunner


logger = logging.getLogger("civitas.engine.simulation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(slots=True)
class LegacyActionView:
    action: str
    amount: float
    target_price: Optional[float]


class RumorManipulatorAgent:
    """Adversarial actor that injects sentiment bursts and directional rumors."""

    agent_id = "rumor_manipulator_agent"

    def __init__(self, intensity: float = 0.75) -> None:
        self.intensity = float(_clip(intensity, 0.1, 2.0))

    def generate(self, tick: int, current_price: float) -> Tuple[float, Optional[BufferedIntent]]:
        direction = -1.0 if tick % 2 == 0 else 1.0
        shock = direction * 0.28 * self.intensity
        qty = int(max(100, 800 * self.intensity))
        side = "sell" if shock < 0 else "buy"
        intent = BufferedIntent(
            intent_id=f"{self.agent_id}_{tick}",
            agent_id=self.agent_id,
            side=side,
            quantity=qty,
            price=max(0.01, float(current_price)),
            intent_type="order",
            activate_step=tick,
            metadata={"abuse_tag": "rumor", "sentiment_delta": shock},
        )
        return shock, intent


class SpoofingAgent:
    """Adversarial actor that layers and rapidly cancels large spoof orders."""

    agent_id = "spoofing_agent"

    def __init__(self, intensity: float = 1.0) -> None:
        self.intensity = float(_clip(intensity, 0.2, 2.5))
        self._last_spoof_order_id: Optional[str] = None

    def generate(self, tick: int, current_price: float) -> List[BufferedIntent]:
        intents: List[BufferedIntent] = []
        if self._last_spoof_order_id:
            intents.append(
                BufferedIntent(
                    intent_id=f"{self.agent_id}_cancel_{tick}",
                    agent_id=self.agent_id,
                    side="cancel",
                    quantity=0,
                    price=0.0,
                    intent_type="cancel",
                    activate_step=tick,
                    metadata={
                        "target_order_id": self._last_spoof_order_id,
                        "abuse_tag": "spoof_cancel",
                    },
                )
            )

        side = "buy" if tick % 2 == 0 else "sell"
        skew = -0.012 if side == "buy" else 0.012
        price = max(0.01, float(current_price) * (1.0 + skew))
        qty = int(max(1500, 5500 * self.intensity))
        order_id = f"{self.agent_id}_layer_{tick}"
        intents.append(
            BufferedIntent(
                intent_id=order_id,
                agent_id=self.agent_id,
                side=side,
                quantity=qty,
                price=price,
                intent_type="order",
                activate_step=tick,
                expire_step=tick + 1,
                metadata={"abuse_tag": "spoof_layer"},
            )
        )
        self._last_spoof_order_id = order_id
        return intents


class MarketEnvironment:
    """
    Macro-social-micro market environment.

    Stage order per simulation step:
    1) policy
    2) macro update
    3) social contagion
    4) agent cognition
    5) trading intent
    6) IPC matching
    7) metrics update
    """

    STAGE_ORDER = [
        "policy",
        "macro update",
        "social contagion",
        "agent cognition",
        "trading intent",
        "IPC matching",
        "metrics update",
    ]

    def __init__(
        self,
        agents: List[TradingAgent],
        *,
        use_isolated_matching: bool = True,
        market_pipeline_v2: bool = True,
        legacy_agent_fallback: bool = True,
        simulation_mode: str = "SMART",
        llm_primary: Optional[bool] = None,
        deep_reasoning_pause_s: float = 0.0,
        model_priority: Optional[Sequence[str]] = None,
        enable_policy_committee: Optional[bool] = None,
        simulation_runner: Optional[SimulationRunner] = None,
        runner_symbol: str = "A_SHARE_IDX",
        steps_per_day: int = 10,
        evolution_interval_days: int = 5,
        bankruptcy_threshold: float = 0.1,
        low_return_threshold: float = -0.2,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.15,
        hybrid_replay: bool = False,
        exogenous_backdrop: Optional[Sequence[Dict[str, Any]] | str | Path] = None,
        hybrid_backdrop_weight: float = 0.35,
        enable_abuse_agents: bool = False,
        abuse_agent_scale: float = 1.0,
        macro_state: Optional[MacroState] = None,
        event_bus: Optional[EventBus] = None,
        households: Optional[Sequence[HouseholdAgent]] = None,
        firms: Optional[Sequence[FirmAgent]] = None,
        social_graph: Optional[SocialGraphState] = None,
        government: Optional[GovernmentAgent] = None,
        bank: Optional[BankAgent] = None,
        contagion_engine: Optional[SocialContagionEngine] = None,
    ):
        self.agents = agents
        self.simulation_time = 0
        self.current_price = float(GLOBAL_CONFIG.get("initial_market_price", 100.0))
        self.price_history: List[float] = [self.current_price]
        self.policy_engine = PolicyEngine()
        self.use_isolated_matching = use_isolated_matching
        self.market_pipeline_v2 = bool(market_pipeline_v2)
        self.legacy_agent_fallback = bool(legacy_agent_fallback)
        self.simulation_mode = str(simulation_mode or "SMART").strip().upper()
        self.runtime_profile: RuntimeModeProfile = resolve_runtime_mode_profile(self.simulation_mode)
        self.llm_primary = bool(self.runtime_profile.llm_primary if llm_primary is None else llm_primary)
        self.deep_reasoning_pause_s = max(0.0, float(deep_reasoning_pause_s or 0.0))
        self.model_priority = list(model_priority or self.runtime_profile.model_priority)
        self.enable_policy_committee = bool(
            self.runtime_profile.enable_policy_committee
            if enable_policy_committee is None
            else enable_policy_committee
        )
        self.runner_symbol = runner_symbol
        self.last_step_report: Dict[str, Any] = {}
        self.last_stage_order: List[str] = []
        self.steps_per_day = max(1, int(steps_per_day))
        self.evolution_interval_days = max(1, int(evolution_interval_days))
        self.bankruptcy_threshold = float(bankruptcy_threshold)
        self.low_return_threshold = float(low_return_threshold)
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)
        self._day_count = 0
        self._initial_cash: Dict[str, float] = {}
        self._wealth_history: Dict[str, List[float]] = {}
        self._policy_shocks: List[str] = []
        self.policy_transmission_history: List[Dict[str, Any]] = []
        self._last_social_mean: float = 0.0
        self.stylized_facts_tracker = StylizedFactsTracker(prices=[self.current_price])
        self.stylized_facts_report_path = Path("outputs") / "stylized_facts_report.json"
        self._agent_reference_points: Dict[str, Any] = {}
        self._agent_risk_appetite: Dict[str, float] = {}
        self._agent_behavioral_state: Dict[str, Dict[str, float]] = {}
        self._agent_position_book: Dict[str, Dict[str, float]] = {}
        self._agent_genomes: Dict[str, StrategyGenome] = {}
        self._evolution_ops = EvolutionOperators(
            mutation_rate=self.mutation_rate,
            mutation_scale=self.mutation_scale,
            seed=42,
        )
        self.ecology_tracker = EcologyMetricsTracker()
        self.ecology_metrics_path = Path("outputs") / "ecology_metrics.csv"
        self.market_abuse_report_path = Path("outputs") / "market_abuse_report.json"
        self.intervention_effect_report_path = Path("outputs") / "intervention_effect_report.json"
        self.abuse_sandbox = MarketAbuseSandbox()
        self._abuse_event_count_series: List[int] = []
        self._intervention_tick: Optional[int] = None
        self._intervention_active: bool = False
        self.hybrid_replay = bool(hybrid_replay)
        self.hybrid_backdrop_weight = float(_clip(hybrid_backdrop_weight, 0.0, 1.0))
        self.enable_abuse_agents = bool(enable_abuse_agents)
        self._abuse_agent_scale = float(max(0.1, abuse_agent_scale))
        self.rumor_manipulator_agent = RumorManipulatorAgent(intensity=self._abuse_agent_scale)
        self.spoofing_agent = SpoofingAgent(intensity=self._abuse_agent_scale)
        self._exogenous_backdrop: List[ExogenousBackdropPoint] = []
        self._latest_hybrid_point: Optional[Dict[str, float]] = None
        self.policy_interpreter = PolicyInterpretationEngine(default_symbols=[self.runner_symbol])
        self.execution_adapter = ExecutionAdapter(default_symbol=self.runner_symbol)
        self._last_policy_package: Optional[Any] = None
        self._last_policy_committee_review: Dict[str, Any] = {}
        self._last_agent_beliefs: List[AgentBelief] = []
        self.unified_market_state = UnifiedMarketState.from_symbol_price(self.runner_symbol, self.current_price)

        self.macro_state = macro_state or MacroState()
        self.event_bus = event_bus or EventBus()
        self.government = government or GovernmentAgent()
        self.bank = bank or BankAgent()
        self.contagion_engine = contagion_engine or SocialContagionEngine()
        self.households = list(households) if households is not None else self._build_default_households(24)
        self.firms = list(firms) if firms is not None else self._build_default_firms()
        self.social_graph = social_graph or self._build_default_social_graph()

        self._owns_runner = simulation_runner is None
        self.simulation_runner = simulation_runner or SimulationRunner(
            symbol=self.runner_symbol,
            prev_close=self.current_price,
        )
        if self.use_isolated_matching:
            self.simulation_runner.start()
        self._configure_hybrid_backdrop(exogenous_backdrop)
        self._initialize_strategy_genomes()
        self._apply_runtime_mode_to_agents()

        logger.info(
            "Initialized MarketEnvironment: agents=%s, price=%.2f, isolated=%s, pipeline_v2=%s, mode=%s",
            len(self.agents),
            self.current_price,
            self.use_isolated_matching,
            self.market_pipeline_v2,
            self.simulation_mode,
        )

    def _apply_runtime_mode_to_agents(self) -> None:
        for agent in self.agents:
            try:
                if hasattr(agent, "use_llm"):
                    if self.llm_primary:
                        setattr(agent, "use_llm", True)
                    elif not self.runtime_profile.use_live_api:
                        setattr(agent, "use_llm", False)
                brain = getattr(agent, "brain", None)
                if brain is not None and hasattr(brain, "model_priority"):
                    setattr(brain, "model_priority", list(self.model_priority))
            except Exception:
                continue

    def _build_default_households(self, n: int) -> List[HouseholdAgent]:
        households: List[HouseholdAgent] = []
        for idx in range(max(1, n)):
            households.append(
                HouseholdAgent(
                    household_id=f"hh_{idx:03d}",
                    income=10_000.0 + (idx % 5) * 1_300.0,
                    propensity_to_consume=0.56 + (idx % 7) * 0.03,
                    savings=20_000.0 + (idx % 6) * 4_000.0,
                    risk_preference=0.35 + (idx % 8) * 0.06,
                    news_exposure=0.30 + (idx % 4) * 0.15,
                    social_exposure=0.35 + (idx % 3) * 0.20,
                )
            )
        return households

    def _build_default_firms(self) -> List[FirmAgent]:
        sectors = ("金融", "科技", "消费", "制造", "能源")
        firms: List[FirmAgent] = []
        for idx, sector in enumerate(sectors):
            firms.append(
                FirmAgent(
                    firm_id=f"firm_{idx:02d}",
                    sector=sector,
                    earnings_expectation=0.05 if sector in {"科技", "消费"} else 0.01,
                    hiring_plan=0.0,
                    inventory=100.0 + idx * 25.0,
                    financing_cost=0.03 + idx * 0.003,
                    sector_outlook=0.0,
                )
            )
        return firms

    def _build_default_social_graph(self) -> SocialGraphState:
        node_ids = [getattr(agent, "agent_id", f"agent_{idx}") for idx, agent in enumerate(self.agents)]
        if not node_ids:
            node_ids = [household.household_id for household in self.households[:12]]
        graph = SocialGraphState.ring(node_ids)
        for node in graph.nodes.values():
            node.sentiment = random.uniform(-0.05, 0.05)
        for src, neighbors in graph.adjacency.items():
            src_group = str(src).split("_")[0]
            for dst in neighbors:
                dst_group = str(dst).split("_")[0]
                graph.set_edge_profile(
                    src,
                    dst,
                    trust_edge=random.uniform(0.35, 0.90),
                    position_similarity_edge=random.uniform(0.20, 0.95),
                    news_exposure_edge=random.uniform(0.25, 0.90),
                    institution_affiliation_edge=1.0 if src_group == dst_group else random.uniform(0.0, 0.4),
                    bidirectional=False,
                )
        return graph

    def _configure_hybrid_backdrop(self, source: Optional[Sequence[Dict[str, Any]] | str | Path]) -> None:
        if source is None:
            self._exogenous_backdrop = []
            return

        loaded: List[ExogenousBackdropPoint] = []
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                self.simulation_runner.load_exogenous_backdrop_csv(path)
                for row in self.simulation_runner._exogenous_backdrop:
                    loaded.append(row)
        else:
            self.simulation_runner.set_exogenous_backdrop(list(source))
            for row in self.simulation_runner._exogenous_backdrop:
                loaded.append(row)
        self._exogenous_backdrop = loaded

    def _exogenous_point_for_tick(self, tick: int) -> Optional[Dict[str, float]]:
        point = self.simulation_runner.get_exogenous_backdrop_point(tick)
        if point is None:
            self._latest_hybrid_point = None
            return None
        payload = {
            "step": float(point.get("step", tick)),
            "price": float(point.get("price", self.current_price)),
            "volume": float(point.get("volume", 0.0)),
        }
        self._latest_hybrid_point = payload
        return payload

    def _initialize_strategy_genomes(self) -> None:
        self._agent_genomes = {}
        rng = random.Random(7)
        for agent in self.agents:
            agent_id = str(getattr(agent, "agent_id", f"agent_{len(self._agent_genomes)}"))
            genome = StrategyGenome.random(rng)
            persona = getattr(agent, "persona", None)
            if persona is not None:
                rt = getattr(persona, "risk_tolerance", None)
                if rt is not None:
                    genome.risk_aversion = float(_clip(1.0 - float(rt), 0.01, 1.0))
            self._agent_genomes[agent_id] = genome
            try:
                setattr(agent, "strategy_genome", genome)
            except Exception:
                pass

    def _genome_for_agent(self, agent: TradingAgent) -> StrategyGenome:
        agent_id = str(getattr(agent, "agent_id", "unknown"))
        genome = self._agent_genomes.get(agent_id)
        if genome is None:
            genome = StrategyGenome.random(random.Random(hash(agent_id) % (2**16)))
            self._agent_genomes[agent_id] = genome
        return genome

    def _apply_genome_behavior_overlay(self, state: Dict[str, float], genome: StrategyGenome) -> Dict[str, float]:
        out = dict(state)
        risk = float(out.get("risk_appetite", 0.5))
        intent = float(out.get("trading_intent", 0.0))
        social = float(out.get("sentiment", 0.0))
        risk = risk * (1.0 - 0.35 * genome.risk_aversion) + 0.20 * genome.order_aggressiveness
        intent = (
            intent * (0.8 + 0.2 * genome.order_aggressiveness)
            + social * 0.25 * genome.social_susceptibility
            - 0.12 * genome.risk_aversion
        )
        out["risk_appetite"] = float(_clip(risk, 0.0, 1.0))
        out["trading_intent"] = float(_clip(intent, -1.0, 1.0))
        out["memory_span"] = float(genome.memory_span)
        out["stop_loss_threshold"] = float(genome.stop_loss_threshold)
        out["order_aggressiveness"] = float(genome.order_aggressiveness)
        out["social_susceptibility"] = float(genome.social_susceptibility)
        out["risk_aversion"] = float(genome.risk_aversion)
        out["debate_participation"] = float(genome.debate_participation)
        return out

    def close(self) -> None:
        """Release subprocess resources if this environment owns the runner."""
        if self.stylized_facts_tracker.market_returns:
            try:
                path = self.stylized_facts_tracker.save_json(self.stylized_facts_report_path)
                self.last_step_report["stylized_facts_report_path"] = str(path)
            except Exception as exc:
                logger.warning("Failed to save stylized facts report: %s", exc)
        try:
            eco = self.ecology_tracker.save_csv(self.ecology_metrics_path)
            self.last_step_report["ecology_metrics_path"] = str(eco)
        except Exception as exc:
            logger.warning("Failed to save ecology metrics: %s", exc)
        try:
            abuse = self.abuse_sandbox.save_report(self.market_abuse_report_path)
            self.last_step_report["market_abuse_report_path"] = str(abuse)
        except Exception as exc:
            logger.warning("Failed to save market abuse report: %s", exc)
        try:
            self._save_intervention_effect_report()
        except Exception as exc:
            logger.warning("Failed to save intervention report: %s", exc)

        if self.use_isolated_matching and self._owns_runner:
            try:
                self.simulation_runner.stop()
            except Exception as exc:
                logger.warning("Failed to stop simulation runner cleanly: %s", exc)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def schedule_policy_shock(self, policy_text: str) -> None:
        """Inject an external policy text that will be compiled next step."""
        if policy_text:
            self._policy_shocks.append(str(policy_text))

    def _consume_policy_shock(self) -> Optional[str]:
        if not self._policy_shocks:
            return None
        return self._policy_shocks.pop(0)

    def _base_risk_for_agent(self, agent: TradingAgent) -> float:
        persona = getattr(agent, "persona", None)
        risk = getattr(persona, "risk_tolerance", None)
        if risk is not None:
            return float(np.clip(float(risk), 0.0, 1.0))
        appetite = getattr(persona, "risk_appetite", None)
        appetite_map = {
            "CONSERVATIVE": 0.25,
            "BALANCED": 0.50,
            "AGGRESSIVE": 0.72,
            "GAMBLER": 0.86,
        }
        if appetite is not None:
            key = getattr(appetite, "name", str(appetite)).upper()
            return float(appetite_map.get(key, 0.50))
        return 0.50

    def _loss_aversion_for_agent(self, agent: TradingAgent) -> float:
        persona = getattr(agent, "persona", None)
        loss = getattr(persona, "loss_aversion", None)
        if loss is not None:
            return float(np.clip(float(loss), 0.5, 6.0))
        bias = str(getattr(persona, "cognitive_bias", ""))
        if "Loss_Aversion" in bias:
            return 2.8
        return 2.0

    def _behavioral_state_for_agent(self, agent: TradingAgent) -> Dict[str, float]:
        agent_id = str(getattr(agent, "agent_id", "unknown"))
        reference_points = self._agent_reference_points.get(agent_id)
        if reference_points is None:
            reference_points = initialize_reference_points(self.current_price)
            self._agent_reference_points[agent_id] = reference_points

        node = self.social_graph.nodes.get(agent_id)
        sentiment = float(getattr(node, "sentiment", self._last_social_mean))
        peer_anchor = self.current_price * (1.0 + 0.04 * self._last_social_mean)
        policy_shock = float(np.clip((self.macro_state.sentiment_index - 0.5) * 1.6, -1.0, 1.0))
        policy_anchor = self.current_price * (1.0 + 0.08 * policy_shock)
        base_risk = self._agent_risk_appetite.get(agent_id, self._base_risk_for_agent(agent))
        step = behavioral_update_step(
            sentiment=sentiment,
            current_price=self.current_price,
            reference_points=reference_points,
            base_risk_appetite=base_risk,
            peer_anchor=peer_anchor,
            policy_anchor=policy_anchor,
            policy_shock=policy_shock,
            loss_aversion=self._loss_aversion_for_agent(agent),
        )
        self._agent_reference_points[agent_id] = step.reference_points
        self._agent_risk_appetite[agent_id] = float(step.risk_appetite)
        state = {
            "sentiment": float(step.sentiment),
            "risk_appetite": float(step.risk_appetite),
            "trading_intent": float(step.trading_intent),
            "prospect_direction": float(step.prospect_direction),
            "loss_aversion_intensity": float(step.loss_aversion_intensity),
            "purchase_anchor": float(step.reference_points.purchase_anchor),
            "recent_high_anchor": float(step.reference_points.recent_high_anchor),
            "peer_anchor": float(step.reference_points.peer_anchor),
            "policy_anchor": float(step.reference_points.policy_anchor),
        }
        self._agent_behavioral_state[agent_id] = state
        return state

    def _cross_sectional_action_returns(self, agent_actions: Sequence[Any], old_price: float) -> List[float]:
        returns: List[float] = []
        base_price = float(max(old_price, 1e-6))
        for action in agent_actions:
            action_view = self._coerce_action_view(action, default_price=base_price)
            direction = str(action_view.action).upper()
            amount = float(action_view.amount)
            target = action_view.target_price
            target_price = float(target) if target is not None else base_price
            price_signal = (target_price - base_price) / base_price
            size_signal = float(np.tanh(amount / 5000.0) * 0.04)
            if direction == "BUY":
                signal = 0.6 * price_signal + 0.4 * abs(size_signal)
            elif direction == "SELL":
                signal = -0.6 * price_signal - 0.4 * abs(size_signal)
            else:
                signal = 0.0
            returns.append(float(np.clip(signal, -0.20, 0.20)))
        if not returns:
            returns = [0.0]
        return returns

    def _coerce_action_view(self, action: Any, *, default_price: float) -> LegacyActionView:
        if isinstance(action, ExecutionPlan):
            qty = float(action.resolved_qty(default_price))
            px = action.resolved_reference_price(default_price)
            return LegacyActionView(action=str(action.action).upper(), amount=qty, target_price=float(px))
        if isinstance(action, Order):
            direction = "BUY" if action.side == OrderSide.BUY else "SELL"
            return LegacyActionView(action=direction, amount=float(action.quantity), target_price=float(action.price))
        direction = str(getattr(action, "action", "HOLD") or "HOLD").upper()
        amount = float(getattr(action, "amount", 0.0) or 0.0)
        target = getattr(action, "target_price", None)
        target_price = float(target) if target is not None else default_price
        return LegacyActionView(action=direction, amount=amount, target_price=target_price)

    def _resolve_execution_plan(
        self,
        *,
        agent: Any,
        action: Any,
        belief: Optional[AgentBelief],
        market_state_payload: Mapping[str, Any],
    ) -> Optional[ExecutionPlan]:
        if isinstance(action, ExecutionPlan):
            return action
        if isinstance(action, Order):
            return self.execution_adapter.plan_from_order(action)
        if belief is not None and hasattr(agent, "decide_from_belief"):
            try:
                plan = agent.decide_from_belief(belief, dict(market_state_payload))
                if isinstance(plan, ExecutionPlan):
                    return plan
            except Exception:
                logger.exception("decide_from_belief failed for %s", getattr(agent, "agent_id", "unknown"))
        action_view = self._coerce_action_view(action, default_price=float(market_state_payload.get("last_price", 0.0)))
        return self.execution_adapter.plan_from_legacy_action(
            agent_id=str(getattr(agent, "agent_id", "unknown")),
            symbol=str(market_state_payload.get("symbol", self.runner_symbol)),
            action=action_view.action,
            amount=action_view.amount,
            target_price=action_view.target_price,
            step=self.simulation_time,
            market_state=market_state_payload,
        )

    def _update_disposition_book(self, agent_actions: Sequence[Any], old_price: float) -> None:
        for idx, action in enumerate(agent_actions):
            if idx >= len(self.agents):
                break
            agent = self.agents[idx]
            agent_id = str(getattr(agent, "agent_id", f"agent_{idx}"))
            side = str(getattr(action, "action", "HOLD")).upper()
            qty = float(getattr(action, "amount", 0.0) or 0.0)
            if qty <= 0 or side not in {"BUY", "SELL"}:
                continue
            px = float(getattr(action, "target_price", old_price) or old_price)
            book = self._agent_position_book.setdefault(agent_id, {"qty": 0.0, "avg_cost": float(old_price)})
            current_qty = float(book.get("qty", 0.0))
            avg_cost = float(book.get("avg_cost", old_price))
            if side == "BUY":
                new_qty = current_qty + qty
                if new_qty > 0:
                    avg_cost = (current_qty * avg_cost + qty * px) / new_qty
                book["qty"] = new_qty
                book["avg_cost"] = avg_cost
            else:
                sell_qty = min(current_qty, qty)
                if sell_qty <= 0:
                    continue
                realized_pnl = (px - avg_cost) * sell_qty
                self.stylized_facts_tracker.disposition.record_realized(realized_pnl)
                book["qty"] = current_qty - sell_qty
                if book["qty"] <= 0:
                    book["avg_cost"] = float(old_price)

        for agent_id, pos in self._agent_position_book.items():
            qty = float(pos.get("qty", 0.0))
            if qty <= 0:
                continue
            avg_cost = float(pos.get("avg_cost", old_price))
            paper_pnl = (self.current_price - avg_cost) * qty
            self.stylized_facts_tracker.disposition.record_paper(paper_pnl)

    def _generate_malicious_intents(self, tick: int) -> Tuple[List[BufferedIntent], float]:
        if not self.enable_abuse_agents:
            return [], 0.0
        intents: List[BufferedIntent] = []
        rumor_shock, rumor_intent = self.rumor_manipulator_agent.generate(tick, self.current_price)
        intents.append(rumor_intent)
        intents.extend(self.spoofing_agent.generate(tick, self.current_price))
        return intents, float(rumor_shock)

    def _inject_rumor_sentiment(self, shock: float) -> None:
        if abs(shock) <= 1e-6 or not self.social_graph.nodes:
            return
        for node_id, node in self.social_graph.nodes.items():
            node.sentiment = _clip(node.sentiment + shock * 0.35, -1.0, 1.0)
            self.abuse_sandbox.register_sentiment(
                agent_id="rumor_manipulator_agent",
                sentiment_delta=shock * 0.35,
                tick=self.simulation_time,
                source="rumor",
            )

    def _register_order_submission(self, intent: BufferedIntent, *, tick: int) -> None:
        intent_kind = str(getattr(intent, "intent_type", "order") or "order").lower()
        if intent_kind == "cancel" or str(intent.side).lower() == "cancel":
            self.abuse_sandbox.register_cancellation(
                agent_id=str(intent.agent_id),
                target_order_id=str(intent.metadata.get("target_order_id", "")),
                tick=tick,
                successful=False,
            )
            return
        self.abuse_sandbox.register_submission(
            agent_id=str(intent.agent_id),
            order_id=str(intent.intent_id),
            side=str(intent.side),
            qty=int(intent.quantity),
            price=float(intent.price),
            tick=tick,
            tag=str(intent.metadata.get("abuse_tag", "organic")),
        )

    def _consume_matching_snapshot_for_abuse(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        for trade in snapshot.get("trades", []) or []:
            maker_id = str(trade.get("maker_id", ""))
            taker_id = str(trade.get("taker_id", ""))
            if maker_id:
                self.abuse_sandbox.register_trade(maker_id, self.simulation_time)
            if taker_id:
                self.abuse_sandbox.register_trade(taker_id, self.simulation_time)

        for item in snapshot.get("canceled_orders", []) or []:
            self.abuse_sandbox.register_cancellation(
                agent_id=str(item.get("agent_id", "")),
                target_order_id=str(item.get("target_order_id", "")),
                tick=self.simulation_time,
                successful=True,
            )

        return self.abuse_sandbox.detect(self.simulation_time)

    def _save_intervention_effect_report(self) -> Path:
        returns = list(self.stylized_facts_tracker.market_returns)
        abuse = list(self._abuse_event_count_series)

        def _metrics(arr_ret: Sequence[float], arr_abuse: Sequence[int]) -> Dict[str, float]:
            if not arr_ret:
                return {"volatility": 0.0, "mean_abs_return": 0.0, "abuse_event_rate": 0.0}
            vol = float(np.std(arr_ret))
            mar = float(np.mean(np.abs(arr_ret)))
            abr = float(np.mean(arr_abuse)) if arr_abuse else 0.0
            return {"volatility": vol, "mean_abs_return": mar, "abuse_event_rate": abr}

        split = None if self._intervention_tick is None else max(0, int(self._intervention_tick - 1))
        before_ret = returns[:split] if split is not None else returns
        after_ret = returns[split:] if split is not None else []
        before_abuse = abuse[:split] if split is not None else abuse
        after_abuse = abuse[split:] if split is not None else []

        before = _metrics(before_ret, before_abuse)
        after = _metrics(after_ret, after_abuse)
        payload = {
            "intervention_active": bool(self._intervention_active),
            "intervention_tick": self._intervention_tick,
            "before": before,
            "after": after,
            "delta": {
                "volatility": float(after["volatility"] - before["volatility"]),
                "mean_abs_return": float(after["mean_abs_return"] - before["mean_abs_return"]),
                "abuse_event_rate": float(after["abuse_event_rate"] - before["abuse_event_rate"]),
            },
        }
        self.intervention_effect_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.intervention_effect_report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.intervention_effect_report_path

    def _record_ecology_metrics(
        self,
        *,
        actions_by_agent: Dict[str, Any],
        market_return: float,
    ) -> Dict[str, float]:
        labels: List[str] = []
        shares: List[float] = []
        sentiments: Dict[str, float] = {}
        for agent_id, action in actions_by_agent.items():
            genome = self._agent_genomes.get(agent_id)
            if genome is not None:
                labels.append(genome.signature_key())
            qty = float(getattr(action, "amount", 0.0) or 0.0)
            if qty > 0:
                shares.append(qty)
        for node_id, node in self.social_graph.nodes.items():
            sentiments[node_id] = float(node.sentiment)
        coalition = build_sentiment_coalitions(sentiments)
        modularity = approx_modularity(self.social_graph.adjacency, coalition)

        prev_row = self.ecology_tracker.rows[-1] if self.ecology_tracker.rows else None
        entropy_val = entropy_from_labels(labels)
        hhi_val = hhi_from_shares(shares)
        persistence = (
            coalition_persistence(
                {k: int(v) for k, v in getattr(self, "_prev_coalition", {}).items()},
                coalition,
            )
            if hasattr(self, "_prev_coalition")
            else 0.0
        )
        phase_score = (
            phase_change_score(prev_row.entropy, entropy_val, getattr(self, "_prev_market_return", 0.0), market_return)
            if prev_row is not None
            else 0.0
        )
        row = EcologyMetricsRow(
            tick=self.simulation_time,
            entropy=float(entropy_val),
            hhi=float(hhi_val),
            modularity=float(modularity),
            phase_changes=float(phase_score),
            coalition_persistence=float(persistence),
        )
        self.ecology_tracker.record(row)
        self.ecology_tracker.save_csv(self.ecology_metrics_path)
        self._prev_coalition = coalition
        self._prev_market_return = float(market_return)
        return self.ecology_tracker.latest()

    async def simulation_step(self) -> Dict[str, Any]:
        """Execute one full simulation step with macro-social-micro coupling."""
        self.simulation_time += 1
        self.last_stage_order = []
        logger.info("--- Start simulation tick %s ---", self.simulation_time)

        # 1) policy
        self.last_stage_order.append("policy")
        policy_text = self._consume_policy_shock() or self.policy_engine.emit_policy(self.simulation_time)
        policy_shock = self._compile_policy(policy_text)

        # 2) macro update
        self.last_stage_order.append("macro update")
        household_signals, firm_signals, bank_signal = self._update_macro(policy_shock, policy_text)

        # 3) social contagion
        self.last_stage_order.append("social contagion")
        contagion = self._run_social_contagion(policy_shock)

        # 4) agent cognition
        self.last_stage_order.append("agent cognition")
        macro_context = self._build_macro_context(policy_text, household_signals, firm_signals, contagion)
        market_data = {
            "tick": self.simulation_time,
            "current_price": self.current_price,
            "latest_broadcast": policy_text if policy_text else "market_stable",
            "price_history": list(self.price_history),
            "symbol": self.runner_symbol,
            "macro_context": macro_context.to_payload(),
        }

        async def process_agent(agent: Any):
            behavioral_state = self._behavioral_state_for_agent(agent)
            genome = self._genome_for_agent(agent)
            behavioral_state = self._apply_genome_behavior_overlay(behavioral_state, genome)
            query_text = (
                f"price={self.current_price:.2f}; tick={self.simulation_time}; "
                f"{macro_context.to_prompt_context()}"
            )
            retrieved_context = ""
            if hasattr(agent, "memory_bank"):
                try:
                    retrieved_context = agent.memory_bank.retrieve_context(
                        current_simulation_time=self.simulation_time,
                        query_embedding=None,
                        query_text=query_text,
                        top_k=3,
                    )
                except Exception:
                    retrieved_context = ""
            payload = dict(market_data)
            payload["behavioral_context"] = behavioral_state
            payload["risk_appetite"] = behavioral_state.get("risk_appetite", 0.5)
            payload["trading_intent"] = behavioral_state.get("trading_intent", 0.0)
            payload["loss_aversion_intensity"] = behavioral_state.get("loss_aversion_intensity", 2.25)
            payload["strategy_genome"] = {
                "analyst_weight_vector": list(genome.normalized_weights()),
                "memory_span": int(genome.memory_span),
                "stop_loss_threshold": float(genome.stop_loss_threshold),
                "order_aggressiveness": float(genome.order_aggressiveness),
                "social_susceptibility": float(genome.social_susceptibility),
                "risk_aversion": float(genome.risk_aversion),
                "debate_participation": float(genome.debate_participation),
            }
            if hasattr(agent, "generate_trading_decision"):
                return await agent.generate_trading_decision(payload, retrieved_context)
            if hasattr(agent, "act"):
                try:
                    from agents.base_agent import MarketSnapshot

                    snapshot = MarketSnapshot(
                        symbol=self.runner_symbol,
                        last_price=float(self.current_price),
                        best_bid=float(self.current_price * 0.999),
                        best_ask=float(self.current_price * 1.001),
                        bid_ask_spread=float(max(self.current_price * 0.002, 0.01)),
                        mid_price=float(self.current_price),
                        total_volume=0,
                        market_trend=float(self.macro_state.sentiment_index - 0.5),
                        panic_level=float(max(0.0, 1.0 - self.macro_state.sentiment_index)),
                        timestamp=float(self.simulation_time),
                        policy_description=str(policy_text or ""),
                    )
                    return await agent.act(snapshot, [str(policy_text or "market_stable")])
                except Exception:
                    logger.exception("Agent act() fallback failed for %s", getattr(agent, "agent_id", "unknown"))
            return LegacyActionView(action="HOLD", amount=0.0, target_price=float(self.current_price))

        if self.llm_primary and self.deep_reasoning_pause_s > 0:
            await asyncio.sleep(self.deep_reasoning_pause_s)

        logger.info("Collecting agent decisions: %s agents", len(self.agents))
        agent_actions = await asyncio.gather(*(process_agent(agent) for agent in self.agents))
        actions_by_agent: Dict[str, Any] = {
            str(getattr(agent, "agent_id", f"agent_{idx}")): action
            for idx, (agent, action) in enumerate(zip(self.agents, agent_actions))
        }
        self._last_agent_beliefs = self.policy_interpreter.batch_interpret(
            self._last_policy_package,
            self.agents,
            {
                "symbols": [self.runner_symbol],
                "symbol": self.runner_symbol,
                "last_price": float(self.current_price),
                "prices": {self.runner_symbol: float(self.current_price)},
            },
        )

        # 5) trading intent
        self.last_stage_order.append("trading intent")
        malicious_intents, rumor_shock = self._generate_malicious_intents(self.simulation_time)
        self._inject_rumor_sentiment(rumor_shock)
        buy_volume = 0.0
        sell_volume = 0.0
        for action in agent_actions:
            action_view = self._coerce_action_view(action, default_price=self.current_price)
            action_type = str(action_view.action).upper()
            amount = float(action_view.amount)
            if action_type == "BUY":
                buy_volume += amount
            elif action_type == "SELL":
                sell_volume += amount
        for intent in malicious_intents:
            side = str(intent.side).upper()
            if str(intent.intent_type).lower() != "order":
                continue
            if side == "BUY":
                buy_volume += float(intent.quantity)
            elif side == "SELL":
                sell_volume += float(intent.quantity)

        # 6) IPC matching
        self.last_stage_order.append("IPC matching")
        old_price = self.current_price
        cross_returns = self._cross_sectional_action_returns(agent_actions, old_price)
        trade_count = 0
        buffered_intents = 0
        matching_mode = "impact_model"
        matching_snapshot: Dict[str, Any] = {}

        if self.use_isolated_matching:
            try:
                matching_mode = "isolated_ipc"
                if self.market_pipeline_v2:
                    plans: List[ExecutionPlan] = []
                    market_state_payload = {
                        "symbol": self.runner_symbol,
                        "symbols": [self.runner_symbol],
                        "last_price": float(self.current_price),
                        "prices": {self.runner_symbol: float(self.current_price)},
                    }
                    for idx, action in enumerate(agent_actions):
                        belief = self._last_agent_beliefs[idx] if idx < len(self._last_agent_beliefs) else None
                        plan = self._resolve_execution_plan(
                            agent=self.agents[idx],
                            action=action,
                            belief=belief,
                            market_state_payload=market_state_payload,
                        )
                        if plan is not None:
                            plans.append(plan)
                    intents = self.execution_adapter.compile_batch(
                        plans,
                        market_state=market_state_payload,
                        step=self.simulation_time,
                    )
                    for intent in malicious_intents:
                        if intent.symbol != self.runner_symbol:
                            intent.symbol = self.runner_symbol
                        if intent.activate_step is None:
                            intent.activate_step = self.simulation_time
                        intents.append(intent)
                    for intent in intents:
                        self._register_order_submission(intent, tick=self.simulation_time)
                    snapshot = self.simulation_runner.submit_batch(intents)
                    buffered_intents = int(snapshot.get("buffer_size", 0))
                else:
                    for idx, action in enumerate(agent_actions):
                        action_view = self._coerce_action_view(action, default_price=self.current_price)
                        action_type = str(action_view.action).upper()
                        amount = float(action_view.amount)
                        if action_type not in {"BUY", "SELL"} or amount <= 0:
                            continue

                        intent_price = float(action_view.target_price or self.current_price)
                        intent = BufferedIntent(
                            intent_id=str(uuid.uuid4()),
                            agent_id=getattr(self.agents[idx], "agent_id", f"agent_{idx}"),
                            side=action_type.lower(),
                            quantity=max(1, int(amount)),
                            price=max(0.01, intent_price),
                            symbol=self.runner_symbol,
                            activate_step=self.simulation_time,
                            intent_type="order",
                            metadata={"abuse_tag": "organic"},
                        )
                        self._register_order_submission(intent, tick=self.simulation_time)
                        ack = self.simulation_runner.submit_intent(intent)
                        buffered_intents = max(buffered_intents, int(ack.get("buffer_size", 0)))

                    for intent in malicious_intents:
                        if intent.symbol != self.runner_symbol:
                            intent.symbol = self.runner_symbol
                        if intent.activate_step is None:
                            intent.activate_step = self.simulation_time
                        self._register_order_submission(intent, tick=self.simulation_time)
                        ack = self.simulation_runner.submit_intent(intent)
                        buffered_intents = max(buffered_intents, int(ack.get("buffer_size", 0)))

                    snapshot = self.simulation_runner.advance_time(1)
                matching_snapshot = dict(snapshot)
                trade_count = int(snapshot.get("trade_count", 0))
                if snapshot.get("last_price") is not None:
                    self.current_price = float(snapshot["last_price"])
                buffered_intents = int(snapshot.get("buffer_size", buffered_intents))
                abuse_detection = self._consume_matching_snapshot_for_abuse(snapshot)
            except Exception as exc:
                logger.exception("isolated matching failed; fallback to impact model: %s", exc)
                matching_mode = "fallback_impact_model"
                self.current_price = calculate_new_price(buy_volume, sell_volume, self.current_price)
                matching_snapshot = {
                    "last_price": float(self.current_price),
                    "trade_count": 0,
                    "best_bid": None,
                    "best_ask": None,
                    "depth": {"bids": [], "asks": []},
                    "activity_stats": {},
                    "buffer_size": 0,
                }
                abuse_detection = self.abuse_sandbox.detect(self.simulation_time)
        else:
            self.current_price = calculate_new_price(buy_volume, sell_volume, self.current_price)
            matching_snapshot = {
                "last_price": float(self.current_price),
                "trade_count": 0,
                "best_bid": None,
                "best_ask": None,
                "depth": {"bids": [], "asks": []},
                "activity_stats": {},
                "buffer_size": 0,
            }
            for intent in malicious_intents:
                self._register_order_submission(intent, tick=self.simulation_time)
                if str(intent.intent_type).lower() == "cancel" or str(intent.side).lower() == "cancel":
                    self.abuse_sandbox.register_cancellation(
                        agent_id=str(intent.agent_id),
                        target_order_id=str(intent.metadata.get("target_order_id", "")),
                        tick=self.simulation_time,
                        successful=True,
                    )
            abuse_detection = self.abuse_sandbox.detect(self.simulation_time)

        exo_point = self._exogenous_point_for_tick(self.simulation_time) if self.hybrid_replay else None
        if exo_point is not None:
            self.current_price = blend_price_with_backdrop(
                old_price=old_price,
                endogenous_price=self.current_price,
                exogenous_price=exo_point.get("price"),
                exogenous_volume=exo_point.get("volume", 0.0),
                backdrop_weight=self.hybrid_backdrop_weight,
            )
        self._abuse_event_count_series.append(int(abuse_detection.get("events_detected", 0)))
        if abuse_detection.get("events_detected", 0) and not self._intervention_active:
            self._intervention_active = True
            self._intervention_tick = self.simulation_time
            self._abuse_agent_scale = max(0.25, self._abuse_agent_scale * 0.5)
            self.rumor_manipulator_agent.intensity = self._abuse_agent_scale
            self.spoofing_agent.intensity = self._abuse_agent_scale

        role_order_flows: Dict[str, float] = {}
        for agent, action in zip(self.agents, agent_actions):
            archetype_key = str(getattr(getattr(agent, "persona", None), "archetype_key", "unknown") or "unknown")
            action_view = self._coerce_action_view(action, default_price=self.current_price)
            signed_qty = float(action_view.amount if action_view.action == "BUY" else -action_view.amount if action_view.action == "SELL" else 0.0)
            role_order_flows[archetype_key] = role_order_flows.get(archetype_key, 0.0) + signed_qty
        thinking_stats = {
            "fast_count": int(sum(int(getattr(agent, "fast_think_count", 0) or 0) for agent in self.agents)),
            "slow_count": int(sum(int(getattr(agent, "slow_think_count", 0) or 0) for agent in self.agents)),
            "llm_enabled_agents": int(sum(1 for agent in self.agents if bool(getattr(agent, "use_llm", False)))),
            "llm_primary": bool(self.llm_primary),
        }

        self.unified_market_state = self.unified_market_state.apply_snapshot(
            matching_snapshot,
            symbol=self.runner_symbol,
            policy_pkg=self._last_policy_package.to_dict() if self._last_policy_package is not None else None,
        )
        micro_metrics = MarketMetrics.compute(
            snapshot=matching_snapshot,
            trade_tape=list(matching_snapshot.get("trades", []) or []),
            role_flows=role_order_flows,
        )

        # 7) metrics update
        self.last_stage_order.append("metrics update")
        self._update_disposition_book(agent_actions, old_price)

        market_return = (self.current_price - old_price) / old_price if old_price > 0 else 0.0
        csad_value = calculate_csad(np.asarray(cross_returns, dtype=float), float(market_return))
        prev_max = max(self.price_history) if self.price_history else self.current_price
        is_all_time_high = bool(self.current_price >= prev_max)
        loss_intensity = (
            float(np.mean([s.get("loss_aversion_intensity", 2.25) for s in self._agent_behavioral_state.values()]))
            if self._agent_behavioral_state
            else 2.25
        )
        self.stylized_facts_tracker.record_step(
            price=self.current_price,
            market_return=float(market_return),
            csad=float(csad_value),
            cross_returns=cross_returns,
            is_all_time_high=is_all_time_high,
            loss_aversion_intensity=loss_intensity,
        )
        ecology_metrics = self._record_ecology_metrics(actions_by_agent=actions_by_agent, market_return=float(market_return))
        intervention_report_path = self._save_intervention_effect_report()

        self.price_history.append(self.current_price)
        if len(self.price_history) > 500:
            self.price_history = self.price_history[-500:]

        price_change = ((self.current_price - old_price) / old_price) * 100 if old_price > 0 else 0.0
        policy_chain = self._build_policy_transmission_chain(
            policy_text=policy_text,
            household_signals=household_signals,
            firm_signals=firm_signals,
            contagion=contagion,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            trade_count=trade_count,
            matching_mode=matching_mode,
        )
        self.policy_transmission_history.append(policy_chain)
        if len(self.policy_transmission_history) > 200:
            self.policy_transmission_history = self.policy_transmission_history[-200:]

        logger.info(
            "[Tick %s] BuyVol=%.2f | SellVol=%.2f | NewPrice=%.2f (%+.2f%%)",
            self.simulation_time,
            buy_volume,
            sell_volume,
            self.current_price,
            price_change,
        )

        self.last_step_report = {
            "tick": self.simulation_time,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "old_price": old_price,
            "new_price": self.current_price,
            "price_change_pct": price_change,
            "matching_mode": matching_mode,
            "pipeline_version": "v2" if self.market_pipeline_v2 else "v1",
            "trade_count": trade_count,
            "buffered_intents": buffered_intents,
            "simulation_mode": self.simulation_mode,
            "mode_runtime": self.runtime_profile.to_dict(),
            "llm_pause_seconds": float(self.deep_reasoning_pause_s),
            "stage_order": list(self.last_stage_order),
            "macro_state": self.macro_state.to_dict(),
            "market_state": self.unified_market_state.to_dict(),
            "microstructure_metrics": micro_metrics,
            "role_order_flows": role_order_flows,
            "thinking_stats": thinking_stats,
            "social_mean_sentiment": contagion.mean_sentiment,
            "policy_transmission_chain": policy_chain,
            "macro_context": macro_context.to_payload(),
            "behavioral_diagnostics": {
                "csad": float(csad_value),
                "loss_aversion_intensity": float(loss_intensity),
                "all_time_high": bool(is_all_time_high),
            },
            "ecology_metrics": ecology_metrics,
            "abuse_detection": abuse_detection,
            "hybrid_replay": {
                "enabled": bool(self.hybrid_replay),
                "backdrop_weight": float(self.hybrid_backdrop_weight),
                "point": self._latest_hybrid_point,
            },
            "intervention": {
                "active": bool(self._intervention_active),
                "tick": self._intervention_tick,
            },
        }
        if self._last_policy_package is not None:
            self.last_step_report["policy_package"] = self._last_policy_package.to_dict()
        if self._last_policy_committee_review:
            self.last_step_report["policy_committee_review"] = dict(self._last_policy_committee_review)
        if self._last_agent_beliefs:
            self.last_step_report["agent_beliefs"] = [
                {
                    "confidence": float(belief.confidence),
                    "latency_bars": int(belief.latency_bars),
                    "disagreement_tags": list(belief.disagreement_tags),
                    "expected_return": dict(belief.expected_return),
                }
                for belief in self._last_agent_beliefs
            ]
        self.last_step_report["stylized_facts_snapshot"] = self.stylized_facts_tracker.report()
        try:
            saved = self.stylized_facts_tracker.save_json(self.stylized_facts_report_path)
            self.last_step_report["stylized_facts_report_path"] = str(saved)
        except Exception as exc:
            logger.warning("Failed to persist stylized facts report at tick %s: %s", self.simulation_time, exc)
        try:
            self.last_step_report["ecology_metrics_path"] = str(self.ecology_tracker.save_csv(self.ecology_metrics_path))
        except Exception as exc:
            logger.warning("Failed to persist ecology metrics at tick %s: %s", self.simulation_time, exc)
        try:
            self.last_step_report["market_abuse_report_path"] = str(
                self.abuse_sandbox.save_report(self.market_abuse_report_path)
            )
        except Exception as exc:
            logger.warning("Failed to persist abuse report at tick %s: %s", self.simulation_time, exc)
        self.last_step_report["intervention_effect_report_path"] = str(intervention_report_path)
        self._update_wealth_history()

        if self.simulation_time % self.steps_per_day == 0:
            self._day_count += 1
            if self._day_count % self.evolution_interval_days == 0:
                evo_report = self._evolution_step()
                self.last_step_report["evolution"] = evo_report

        self.event_bus.publish(
            event_type="metrics",
            stage="metrics_update",
            tick=self.simulation_time,
            payload=self.last_step_report,
        )
        return dict(self.last_step_report)

    def _compile_policy(self, policy_text: Optional[str]) -> Optional[PolicyShock]:
        if not policy_text:
            self._last_policy_package = None
            self._last_policy_committee_review = {}
            return None
        policy_package = self.government.compile_policy_package(policy_text, tick=self.simulation_time)
        self._last_policy_package = policy_package
        self._last_policy_committee_review = {}
        if self.llm_primary and self.enable_policy_committee and self.runtime_profile.use_live_api:
            try:
                from config import GLOBAL_CONFIG as APP_CONFIG
                from core.policy_committee import PolicyCommittee as AdversarialPolicyCommittee

                committee = AdversarialPolicyCommittee(api_key=APP_CONFIG.DEEPSEEK_API_KEY)
                committee_result = committee.interpret(str(policy_text))
                self._last_policy_committee_review = {
                    "parameters": dict(committee_result.parameters or {}),
                    "compliance_passed": bool(committee_result.compliance.passed),
                    "violations": list(committee_result.compliance.violations or []),
                    "warnings": list(committee_result.compliance.warnings or []),
                    "reasoning_chain": list(committee_result.reasoning_chain or []),
                    "final_state": dict(committee_result.final_state or {}),
                }
            except Exception as exc:
                self._last_policy_committee_review = {"error": str(exc)}
        shock = PolicyShock(
            policy_id=policy_package.event.policy_id,
            policy_text=str(policy_text),
            **policy_package.to_policy_shock_fields(),
        )
        shock.metadata = {
            "policy_event": policy_package.event.to_dict(),
            "policy_package": policy_package.to_dict(),
            "parser_version": policy_package.parser_version,
            "feature_flags": dict(policy_package.metadata.get("feature_flags", {})),
            "config_hash": str(policy_package.metadata.get("config_hash", "")),
        }
        if self._last_policy_committee_review:
            shock.metadata["policy_committee_review"] = dict(self._last_policy_committee_review)
        self.event_bus.publish(
            event_type="policy_compiled",
            stage="policy",
            tick=self.simulation_time,
            payload={
                **shock.to_dict(),
                "policy_package": policy_package.to_dict(),
                "policy_committee_review": dict(self._last_policy_committee_review),
            },
        )
        # Optional memory broadcast for agents with memory banks.
        for agent in self.agents:
            if not hasattr(agent, "memory_bank"):
                continue
            try:
                memory_strength = agent.memory_bank.calculate_memory_strength(policy_text, agent.persona)
                agent.memory_bank.add_memory(
                    timestamp=self.simulation_time,
                    content=policy_text,
                    content_embedding=None,
                    memory_strength=memory_strength,
                )
            except Exception:
                continue
        return shock

    def _update_macro(
        self, policy_shock: Optional[PolicyShock], policy_text: Optional[str]
    ) -> tuple[List[HouseholdSignal], List[FirmSignal], Dict[str, float]]:
        if policy_shock is not None:
            self.government.apply_policy_shock(self.macro_state, policy_shock)

        liquidity_shock = policy_shock.liquidity_injection if policy_shock else 0.0
        sentiment_shock = (
            (policy_shock.sentiment_delta + policy_shock.rumor_shock) if policy_shock else 0.0
        )
        bank_signal = self.bank.step(
            self.macro_state,
            liquidity_shock=liquidity_shock,
            sentiment_shock=sentiment_shock,
        ).to_dict()

        household_signals = [
            household.step(self.macro_state, self._last_social_mean, policy_text or "")
            for household in self.households
        ]
        mean_consumption = (
            sum(item.consumption for item in household_signals) / len(household_signals)
            if household_signals
            else 0.0
        )
        mean_risk_pref = (
            sum(item.risk_preference for item in household_signals) / len(household_signals)
            if household_signals
            else 0.5
        )

        firm_signals = [
            firm.step(self.macro_state, self._last_social_mean, mean_consumption)
            for firm in self.firms
        ]
        mean_hiring = (
            sum(item.hiring_plan for item in firm_signals) / len(firm_signals)
            if firm_signals
            else 0.0
        )
        mean_sector = (
            sum(item.sector_outlook for item in firm_signals) / len(firm_signals)
            if firm_signals
            else 0.0
        )

        self.macro_state.apply_delta(
            unemployment=-0.0030 * mean_hiring + 0.0010 * (0.5 - mean_risk_pref),
            wage_growth=0.0016 * mean_hiring - 0.0005 * self.macro_state.unemployment,
            inflation=0.0005 * (mean_consumption / 10_000.0 - 1.0),
            sentiment_index=0.030 * (mean_risk_pref - 0.5) + 0.025 * mean_sector,
        )
        self.event_bus.publish(
            event_type="macro_state",
            stage="macro_update",
            tick=self.simulation_time,
            payload={
                "macro_state": self.macro_state.to_dict(),
                "bank_signal": bank_signal,
                "household_count": len(household_signals),
                "firm_count": len(firm_signals),
            },
        )
        return household_signals, firm_signals, bank_signal

    def _run_social_contagion(self, policy_shock: Optional[PolicyShock]) -> ContagionSnapshot:
        rumor = policy_shock.rumor_shock if policy_shock else 0.0
        contagion = self.contagion_engine.step(self.social_graph, self.macro_state, rumor_shock=rumor)
        self._last_social_mean = contagion.mean_sentiment
        self.macro_state.sentiment_index = _clip(
            0.75 * self.macro_state.sentiment_index + 0.25 * ((contagion.mean_sentiment + 1.0) * 0.5),
            0.0,
            1.0,
        )
        self.event_bus.publish(
            event_type="social_contagion",
            stage="social_contagion",
            tick=self.simulation_time,
            payload=contagion.to_dict(),
        )
        return contagion

    def _build_macro_context(
        self,
        policy_text: Optional[str],
        household_signals: Sequence[HouseholdSignal],
        firm_signals: Sequence[FirmSignal],
        contagion: ContagionSnapshot,
    ) -> MacroContextDTO:
        sector_groups: Dict[str, List[float]] = defaultdict(list)
        for signal in firm_signals:
            sector_groups[signal.sector].append(signal.sector_outlook)
        sector_outlook = {
            sector: sum(values) / len(values)
            for sector, values in sector_groups.items()
            if values
        }
        household_risk_shift = (
            sum(item.risk_preference for item in household_signals) / len(household_signals) - 0.5
            if household_signals
            else 0.0
        )
        firm_hiring_signal = (
            sum(item.hiring_plan for item in firm_signals) / len(firm_signals)
            if firm_signals
            else 0.0
        )
        return MacroContextDTO(
            tick=self.simulation_time,
            macro_state=MacroState.from_mapping(self.macro_state.to_dict()),
            policy_summary=policy_text or "no_policy",
            social_sentiment=contagion.mean_sentiment,
            sector_outlook=sector_outlook,
            household_risk_shift=household_risk_shift,
            firm_hiring_signal=firm_hiring_signal,
        )

    def _build_policy_transmission_chain(
        self,
        *,
        policy_text: Optional[str],
        household_signals: Sequence[HouseholdSignal],
        firm_signals: Sequence[FirmSignal],
        contagion: ContagionSnapshot,
        buy_volume: float,
        sell_volume: float,
        trade_count: int,
        matching_mode: str,
    ) -> Dict[str, Any]:
        avg_news = (
            sum(item.news_exposure for item in household_signals) / len(household_signals)
            if household_signals
            else 0.0
        )
        avg_social_exposure = (
            sum(item.social_exposure for item in household_signals) / len(household_signals)
            if household_signals
            else 0.0
        )
        avg_household_risk = (
            sum(item.risk_preference for item in household_signals) / len(household_signals)
            if household_signals
            else 0.0
        )
        avg_hiring = (
            sum(item.hiring_plan for item in firm_signals) / len(firm_signals)
            if firm_signals
            else 0.0
        )
        sector_groups: Dict[str, List[float]] = defaultdict(list)
        for signal in firm_signals:
            sector_groups[signal.sector].append(signal.sector_outlook)

        return {
            "policy": policy_text or "no_policy",
            "macro_variables": {
                "inflation": self.macro_state.inflation,
                "unemployment": self.macro_state.unemployment,
                "wage_growth": self.macro_state.wage_growth,
                "credit_spread": self.macro_state.credit_spread,
                "liquidity_index": self.macro_state.liquidity_index,
                "policy_rate": self.macro_state.policy_rate,
                "fiscal_stimulus": self.macro_state.fiscal_stimulus,
                "sentiment_index": self.macro_state.sentiment_index,
            },
            "social_sentiment": {
                "mean": contagion.mean_sentiment,
                "stressed_nodes": list(contagion.stressed_nodes),
                "avg_news_exposure": avg_news,
                "avg_social_exposure": avg_social_exposure,
            },
            "industry_agent": {
                "avg_household_risk": avg_household_risk,
                "avg_firm_hiring": avg_hiring,
                "sector_outlook": {
                    sector: sum(values) / len(values) for sector, values in sector_groups.items() if values
                },
            },
            "market_microstructure": {
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "trade_count": int(trade_count),
                "matching_mode": matching_mode,
                "price": self.current_price,
            },
        }

    def _update_wealth_history(self) -> None:
        """Track each agent wealth path: cash + holdings market value."""
        for agent in self.agents:
            if agent.agent_id not in self._initial_cash:
                self._initial_cash[agent.agent_id] = float(getattr(agent, "cash", 0.0))
            cash = float(getattr(agent, "cash", 0.0))
            portfolio = getattr(agent, "portfolio", {}) or {}
            holding = float(sum(qty for qty in portfolio.values()))
            wealth = cash + holding * self.current_price
            history = self._wealth_history.setdefault(agent.agent_id, [])
            history.append(wealth)
            max_len = self.steps_per_day * self.evolution_interval_days * 2
            if len(history) > max_len:
                self._wealth_history[agent.agent_id] = history[-max_len:]

    def _evolution_step(self) -> Dict[str, Any]:
        """Apply genome-level evolutionary operators."""
        if not self.agents:
            return {"status": "skipped", "reason": "no_agents"}

        to_remove: List[TradingAgent] = []
        returns: Dict[str, float] = {}
        freed_capital = 0.0

        for agent in self.agents:
            agent_id = str(agent.agent_id)
            history = self._wealth_history.get(agent_id, [])
            initial_cash = self._initial_cash.get(agent_id, float(getattr(agent, "cash", 0.0)))
            if not history or initial_cash <= 0:
                returns[agent_id] = 0.0
                continue

            window = min(len(history), self.steps_per_day * self.evolution_interval_days)
            start_val = history[-window]
            end_val = history[-1]
            ret = (end_val / start_val - 1.0) if start_val > 0 else 0.0
            returns[agent_id] = ret
            if end_val <= initial_cash * self.bankruptcy_threshold or ret < self.low_return_threshold:
                to_remove.append(agent)
                freed_capital += max(0.0, end_val)

        self.agents = [a for a in self.agents if a not in to_remove]
        for removed in to_remove:
            self._agent_genomes.pop(str(removed.agent_id), None)
        survivor_ids = [str(a.agent_id) for a in self.agents]
        selected_ids = self._evolution_ops.selection({aid: returns.get(aid, 0.0) for aid in survivor_ids}, survival_rate=0.5)
        if not selected_ids and survivor_ids:
            selected_ids = survivor_ids[:1]

        # Local diffusion over existing topology
        adjacency = {node_id: list(nei) for node_id, nei in self.social_graph.adjacency.items()}
        diffused = self._evolution_ops.local_diffusion(self._agent_genomes, adjacency, strength=0.06)

        # Mutate selected survivors
        mutated_ids: List[str] = []
        for aid in selected_ids:
            genome = self._agent_genomes.get(aid)
            if genome is None:
                continue
            self._agent_genomes[aid] = self._evolution_ops.mutation(genome)
            mutated_ids.append(aid)

        # Crossover + mutation to spawn replacements
        offspring: List[TradingAgent] = []
        if to_remove and selected_ids:
            n_offspring = max(1, len(to_remove))
            budget = freed_capital * 0.25 if freed_capital > 0 else 0.0
            per_capital = budget / n_offspring if n_offspring > 0 else 0.0
            id_to_agent = {str(a.agent_id): a for a in self.agents}
            for _ in range(n_offspring):
                pa_id = random.choice(selected_ids)
                pb_id = random.choice(selected_ids)
                pa = self._agent_genomes.get(pa_id, StrategyGenome.random())
                pb = self._agent_genomes.get(pb_id, StrategyGenome.random())
                child_genome = self._evolution_ops.mutation(self._evolution_ops.crossover(pa, pb))
                parent = id_to_agent.get(pa_id) or (self.agents[0] if self.agents else None)
                if parent is None:
                    continue
                parent_persona = getattr(parent, "persona", None)
                try:
                    new_persona = Persona(
                        risk_tolerance=float(_clip(1.0 - child_genome.risk_aversion, 0.0, 1.0)),
                        cognitive_bias=getattr(parent_persona, "cognitive_bias"),
                        investment_horizon=getattr(parent_persona, "investment_horizon"),
                        policy_sensitivity=float(_clip(getattr(parent_persona, "policy_sensitivity", 0.5), 0.0, 1.0)),
                    )
                except Exception:
                    new_persona = parent_persona
                child_agent = TradingAgent(agent_id=f"Genome_{uuid.uuid4().hex[:8]}", persona=new_persona)
                child_agent.cash = max(0.0, per_capital)
                setattr(child_agent, "strategy_genome", child_genome)
                self._agent_genomes[str(child_agent.agent_id)] = child_genome
                offspring.append(child_agent)

        if offspring:
            self.agents.extend(offspring)
            for child in offspring:
                self.social_graph.ensure_node(child.agent_id)
                existing = [node_id for node_id in self.social_graph.nodes.keys() if node_id != child.agent_id]
                if existing:
                    peer = random.choice(existing)
                    self.social_graph.add_edge(child.agent_id, peer, bidirectional=True)

        # Selection pressure capital redistribution
        top_agents = sorted(self.agents, key=lambda a: returns.get(str(a.agent_id), 0.0), reverse=True)
        if freed_capital > 0 and top_agents:
            scores = [max(returns.get(str(a.agent_id), 0.0), -0.5) for a in top_agents]
            exps = [math.exp(s) for s in scores]
            total = sum(exps) if exps else 1.0
            for agent, weight in zip(top_agents, exps):
                agent.cash = float(getattr(agent, "cash", 0.0)) + freed_capital * (weight / total)

        return {
            "status": "ok",
            "removed": [a.agent_id for a in to_remove],
            "offspring": [a.agent_id for a in offspring],
            "selection": selected_ids,
            "mutation": mutated_ids,
            "local_diffusion": list(diffused.keys()),
            "freed_capital": float(freed_capital),
            "survivors": len(self.agents),
        }

    def run_parameter_sensitivity_scan(
        self,
        *,
        loss_aversion_grid: Sequence[float] = (1.8, 2.25, 3.0),
        reference_adaptivity_grid: Sequence[float] = (0.3, 0.6, 0.9),
        edge_weight_grid: Sequence[float] = (0.6, 1.0, 1.4),
        output_csv: str | Path = Path("outputs") / "parameter_sensitivity.csv",
    ) -> Path:
        rows: List[Dict[str, float]] = []
        base_price = max(1e-6, float(self.current_price))
        for loss_aversion in loss_aversion_grid:
            for adaptivity in reference_adaptivity_grid:
                for edge_weight in edge_weight_grid:
                    refs = initialize_reference_points(base_price)
                    risk_hist: List[float] = []
                    intent_hist: List[float] = []
                    loss_hist: List[float] = []
                    for step in range(1, 21):
                        sentiment = _clip((math.sin(step / 3.0) * 0.6) * edge_weight, -1.0, 1.0)
                        price = base_price * (1.0 + 0.001 * step * (1.0 - adaptivity))
                        update = behavioral_update_step(
                            sentiment=sentiment,
                            current_price=price,
                            reference_points=refs,
                            base_risk_appetite=0.5,
                            peer_anchor=price * (1.0 + 0.03 * edge_weight),
                            policy_anchor=price * (1.0 + 0.02 * sentiment),
                            policy_shock=sentiment * 0.5,
                            loss_aversion=float(loss_aversion),
                        )
                        refs = update.reference_points
                        risk_hist.append(float(update.risk_appetite))
                        intent_hist.append(float(update.trading_intent))
                        loss_hist.append(float(update.loss_aversion_intensity))
                    rows.append(
                        {
                            "loss_aversion": float(loss_aversion),
                            "reference_adaptivity": float(adaptivity),
                            "edge_weight": float(edge_weight),
                            "avg_risk_appetite": float(np.mean(risk_hist)),
                            "avg_trading_intent": float(np.mean(intent_hist)),
                            "avg_loss_aversion_intensity": float(np.mean(loss_hist)),
                            "intent_volatility": float(np.std(intent_hist)),
                        }
                    )

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        summary_path = output_path.with_name("parameter_sensitivity_summary.json")
        summary_path.write_text(json.dumps({"rows": rows, "n": len(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_step_report["parameter_sensitivity_path"] = str(output_path)
        return output_path


if __name__ == "__main__":
    from agents.trading_agent_core import CognitiveBias, InvestmentHorizon

    async def _demo() -> None:
        agents = [
            TradingAgent(
                agent_id="demo_1",
                persona=Persona(
                    risk_tolerance=0.6,
                    cognitive_bias=CognitiveBias.Herding,
                    investment_horizon=InvestmentHorizon.Short_term,
                    policy_sensitivity=0.7,
                ),
            )
        ]
        env = MarketEnvironment(agents, use_isolated_matching=False)
        env.schedule_policy_shock("印花税下调并流动性注入")
        report = await env.simulation_step()
        print(report["stage_order"])
        print(report["policy_transmission_chain"])

    asyncio.run(_demo())
