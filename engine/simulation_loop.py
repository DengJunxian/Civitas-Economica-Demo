"""
Civitas-Economica 核心仿真循环 (Simulation Loop)

重构目标：
1. 将慢速 LLM 决策与快速撮合引擎解耦。
2. 通过 SimulationRunner 将撮合放到独立进程，避免主线程阻塞。
3. 保留旧的价格冲击模型作为降级路径，保证可用性。
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from agents.trading_agent_core import GLOBAL_CONFIG, TradingAgent, Persona
from engine.market_match import calculate_new_price
from policy.policy_engine import PolicyEngine
from simulation_runner import BufferedIntent, SimulationRunner

logger = logging.getLogger("civitas.engine.simulation")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)


class MarketEnvironment:
    """
    宏观仿真控制器。

    关键改动：
    - LLM 只生成交易意图，不直接执行撮合。
    - 意图先进入 IPC 缓冲，再由独立 OASIS Runner 按离散时间执行。
    """

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
    ):
        self.agents = agents
        self.simulation_time = 0
        self.current_price = float(GLOBAL_CONFIG.get("initial_market_price", 100.0))
        self.price_history: List[float] = [self.current_price]
        self.policy_engine = PolicyEngine()
        self.use_isolated_matching = use_isolated_matching
        self.runner_symbol = runner_symbol
        self.last_step_report: Dict[str, Any] = {}
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

    def close(self) -> None:
        """显式释放子进程资源。"""
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
        """
        提供给外部系统的“宏观扰动注入”接口。
        对应 FinEvo 的 Perturbation：定期向系统注入黑天鹅冲击。
        """
        if policy_text:
            self._policy_shocks.append(str(policy_text))

    def _consume_policy_shock(self) -> Optional[str]:
        if not self._policy_shocks:
            return None
        return self._policy_shocks.pop(0)

    async def simulation_step(self) -> Dict[str, Any]:
        """
        执行单个仿真步。

        返回结构化报告，便于 UI/调度器直接消费。
        """
        self.simulation_time += 1
        logger.info("--- Start simulation tick %s ---", self.simulation_time)

        new_policy = self._consume_policy_shock() or self.policy_engine.emit_policy(self.simulation_time)
        if new_policy:
            logger.warning("[Policy Broadcast] %s", new_policy)
            for agent in self.agents:
                # 兼容部分轻量测试替身 Agent，不强制要求实现完整 memory_bank
                if not hasattr(agent, "memory_bank"):
                    continue
                try:
                    memory_strength = agent.memory_bank.calculate_memory_strength(new_policy, agent.persona)
                    agent.memory_bank.add_memory(
                        timestamp=self.simulation_time,
                        content=new_policy,
                        content_embedding=None,
                        memory_strength=memory_strength,
                    )
                except Exception as exc:
                    logger.debug("memory update skipped for agent=%s, reason=%s", getattr(agent, "agent_id", "?"), exc)

        market_data = {
            "tick": self.simulation_time,
            "current_price": self.current_price,
            "latest_broadcast": new_policy if new_policy else "market_stable",
            "price_history": list(self.price_history),
            "symbol": self.runner_symbol,
        }

        async def process_agent(agent: TradingAgent):
            query_text = f"在价格 {self.current_price:.2f} 下的应对策略及近期冲击回顾"
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

        buy_volume = 0.0
        sell_volume = 0.0
        for action in agent_actions:
            action_type = str(getattr(action, "action", "HOLD")).upper()
            amount = float(getattr(action, "amount", 0.0) or 0.0)
            if action_type == "BUY":
                buy_volume += amount
            elif action_type == "SELL":
                sell_volume += amount

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

                    # 交易意图异步入缓冲；撮合只在独立进程中发生
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

        self.price_history.append(self.current_price)
        if len(self.price_history) > 500:
            self.price_history = self.price_history[-500:]

        price_change = ((self.current_price - old_price) / old_price) * 100 if old_price > 0 else 0.0
        logger.info(
            "[Macro State] Tick %s settled | BuyVol=%.2f | SellVol=%.2f | NewPrice=%.2f (%+.2f%%)",
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
        }

        self._update_wealth_history()

        if self.simulation_time % self.steps_per_day == 0:
            self._day_count += 1
            if self._day_count % self.evolution_interval_days == 0:
                evo_report = self._evolution_step()
                self.last_step_report["evolution"] = evo_report

        return dict(self.last_step_report)

    def _update_wealth_history(self) -> None:
        """记录每个 Agent 的财富轨迹（现金 + 持仓市值）。"""
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
        """
        FinEvo 演化步骤：选择（Selection）/ 创新（Innovation）/ 扰动（Perturbation）。
        """
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

        survivors = [a for a in self.agents if a not in to_remove]
        self.agents = survivors

        top_agents = sorted(survivors, key=lambda a: returns.get(a.agent_id, 0.0), reverse=True)
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
                new_risk = float(np.clip(parent_persona.risk_tolerance + random.uniform(-self.mutation_scale, self.mutation_scale), 0.0, 1.0))
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

        return {
            "status": "ok",
            "removed": [a.agent_id for a in to_remove],
            "mutants": [a.agent_id for a in mutants],
            "freed_capital": freed_capital,
            "survivors": len(self.agents),
        }
