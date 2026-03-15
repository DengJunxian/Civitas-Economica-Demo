"""Core simulation loop with macro-social-micro coupling."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agents.trading_agent_core import GLOBAL_CONFIG, Persona, TradingAgent
from core.macro.bank import BankAgent
from core.macro.firm import FirmAgent, FirmSignal
from core.macro.government import GovernmentAgent, PolicyShock
from core.macro.household import HouseholdAgent, HouseholdSignal
from core.macro.state import MacroContextDTO, MacroState
from core.social.contagion import ContagionSnapshot, SocialContagionEngine
from core.social.graph_state import SocialGraphState
from core.world.event_bus import EventBus
from engine.market_match import calculate_new_price
from policy.policy_engine import PolicyEngine
from simulation_runner import BufferedIntent, SimulationRunner


logger = logging.getLogger("civitas.engine.simulation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


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
        simulation_runner: Optional[SimulationRunner] = None,
        runner_symbol: str = "A_SHARE_IDX",
        steps_per_day: int = 10,
        evolution_interval_days: int = 5,
        bankruptcy_threshold: float = 0.1,
        low_return_threshold: float = -0.2,
        mutation_rate: float = 0.2,
        mutation_scale: float = 0.15,
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

        logger.info(
            "Initialized MarketEnvironment: agents=%s, price=%.2f, isolated=%s",
            len(self.agents),
            self.current_price,
            self.use_isolated_matching,
        )

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
        return graph

    def close(self) -> None:
        """Release subprocess resources if this environment owns the runner."""
        if not self.use_isolated_matching:
            return
        if self._owns_runner:
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

        async def process_agent(agent: TradingAgent):
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
            return await agent.generate_trading_decision(market_data, retrieved_context)

        logger.info("Collecting agent decisions: %s agents", len(self.agents))
        agent_actions = await asyncio.gather(*(process_agent(agent) for agent in self.agents))

        # 5) trading intent
        self.last_stage_order.append("trading intent")
        buy_volume = 0.0
        sell_volume = 0.0
        for action in agent_actions:
            action_type = str(getattr(action, "action", "HOLD")).upper()
            amount = float(getattr(action, "amount", 0.0) or 0.0)
            if action_type == "BUY":
                buy_volume += amount
            elif action_type == "SELL":
                sell_volume += amount

        # 6) IPC matching
        self.last_stage_order.append("IPC matching")
        old_price = self.current_price
        trade_count = 0
        buffered_intents = 0
        matching_mode = "impact_model"

        if self.use_isolated_matching:
            try:
                matching_mode = "isolated_ipc"
                for idx, action in enumerate(agent_actions):
                    action_type = str(getattr(action, "action", "HOLD")).upper()
                    amount = float(getattr(action, "amount", 0.0) or 0.0)
                    if action_type not in {"BUY", "SELL"} or amount <= 0:
                        continue

                    target_price = getattr(action, "target_price", None)
                    intent_price = float(target_price) if target_price else float(self.current_price)
                    intent = BufferedIntent(
                        intent_id=str(uuid.uuid4()),
                        agent_id=getattr(self.agents[idx], "agent_id", f"agent_{idx}"),
                        side=action_type.lower(),
                        quantity=max(1, int(amount)),
                        price=max(0.01, intent_price),
                        symbol=self.runner_symbol,
                        activate_step=self.simulation_time,
                    )
                    ack = self.simulation_runner.submit_intent(intent)
                    buffered_intents = max(buffered_intents, int(ack.get("buffer_size", 0)))

                snapshot = self.simulation_runner.advance_time(1)
                trade_count = int(snapshot.get("trade_count", 0))
                if snapshot.get("last_price") is not None:
                    self.current_price = float(snapshot["last_price"])
                buffered_intents = int(snapshot.get("buffer_size", buffered_intents))
            except Exception as exc:
                logger.exception("isolated matching failed; fallback to impact model: %s", exc)
                matching_mode = "fallback_impact_model"
                self.current_price = calculate_new_price(buy_volume, sell_volume, self.current_price)
        else:
            self.current_price = calculate_new_price(buy_volume, sell_volume, self.current_price)

        # 7) metrics update
        self.last_stage_order.append("metrics update")
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
            "trade_count": trade_count,
            "buffered_intents": buffered_intents,
            "stage_order": list(self.last_stage_order),
            "macro_state": self.macro_state.to_dict(),
            "social_mean_sentiment": contagion.mean_sentiment,
            "policy_transmission_chain": policy_chain,
            "macro_context": macro_context.to_payload(),
        }
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
            return None
        shock = self.government.compile_policy_text(policy_text, tick=self.simulation_time)
        self.event_bus.publish(
            event_type="policy_compiled",
            stage="policy",
            tick=self.simulation_time,
            payload=shock.to_dict(),
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
        """Apply selection + mutation to maintain adaptive population."""
        if not self.agents:
            return {"status": "skipped", "reason": "no_agents"}

        to_remove: List[TradingAgent] = []
        returns: Dict[str, float] = {}
        freed_capital = 0.0

        for agent in self.agents:
            agent_id = agent.agent_id
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
        top_agents = sorted(self.agents, key=lambda a: returns.get(a.agent_id, 0.0), reverse=True)

        if freed_capital > 0 and top_agents:
            scores = [max(returns.get(a.agent_id, 0.0), -0.5) for a in top_agents]
            exps = [math.exp(s) for s in scores]
            total = sum(exps) if exps else 1.0
            for agent, weight in zip(top_agents, exps):
                grant = freed_capital * (weight / total)
                agent.cash = float(getattr(agent, "cash", 0.0)) + grant

        mutants: List[TradingAgent] = []
        if to_remove and top_agents:
            n_mutants = max(1, int(len(to_remove) * self.mutation_rate))
            budget = freed_capital * 0.2 if freed_capital > 0 else 0.0
            per_capital = budget / n_mutants if n_mutants > 0 else 0.0

            for _ in range(n_mutants):
                parent = random.choice(top_agents)
                parent_persona = parent.persona
                new_risk = float(
                    np.clip(
                        parent_persona.risk_tolerance + random.uniform(-self.mutation_scale, self.mutation_scale),
                        0.0,
                        1.0,
                    )
                )
                new_persona = Persona(
                    risk_tolerance=new_risk,
                    cognitive_bias=parent_persona.cognitive_bias,
                    investment_horizon=parent_persona.investment_horizon,
                    policy_sensitivity=parent_persona.policy_sensitivity,
                )
                new_agent = TradingAgent(agent_id=f"Mutant_{uuid.uuid4().hex[:8]}", persona=new_persona)
                new_agent.cash = max(0.0, per_capital)
                mutants.append(new_agent)

        if mutants:
            self.agents.extend(mutants)
            for mutant in mutants:
                self.social_graph.ensure_node(mutant.agent_id)
                existing = [node_id for node_id in self.social_graph.nodes.keys() if node_id != mutant.agent_id]
                if existing:
                    peer = random.choice(existing)
                    self.social_graph.add_edge(mutant.agent_id, peer, bidirectional=True)

        return {
            "status": "ok",
            "removed": [a.agent_id for a in to_remove],
            "mutants": [a.agent_id for a in mutants],
            "freed_capital": freed_capital,
            "survivors": len(self.agents),
        }


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
