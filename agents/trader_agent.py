# file: agents/trader_agent.py
"""
TraderAgent 实现 — 基于认知闭环的交易智能体

实现 BaseAgent 定义的 Perceive-Reason-Decide-Act 闭环。
核心特性:
1. 心理画像驱动: 风险厌恶、自信程度、关注广度等个性化参数
2. 模拟 DeepSeek R1 推理: 生成类人思维链 (Chain of Thought)
3. 结构化决策: 输出标准 Limit Order

作者: Civitas Economica Team
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Optional, Any, Tuple

from agents.base_agent import BaseAgent, MarketSnapshot
from agents.brain import DeepSeekBrain
from core.types import Order, OrderSide, OrderType, OrderStatus
from agents.persona import Persona, RiskAppetite # [NEW]
from core.society.network import SocialGraph # [NEW]

class TraderAgent(BaseAgent):
    """
    TraderAgent — 具备完整认知闭环的交易智能体
    """

    def __init__(
        self,
        agent_id: str,
        cash_balance: float = 100_000.0,
        portfolio: Optional[Dict[str, int]] = None,
        psychology_profile: Optional[Dict[str, float]] = None,
        model_router: Optional[Any] = None, # Support Router
        persona: Optional[Persona] = None # [NEW] Persona
    ):
        super().__init__(agent_id, cash_balance, portfolio, psychology_profile)
        
        # [NEW] Persona Integration
        self.persona = persona if persona else Persona(name=agent_id)
        
        # [NEW] Social Network Integration
        self.social_node_id: Optional[int] = None
        self.social_graph: Optional[SocialGraph] = None
        
        # Map Persona to legacy psychology profile for compatibility
        self.profile = {
            "risk_aversion": self._map_persona_to_risk_aversion(),
            "confidence_level": 0.5 + (self.persona.overconfidence * 0.5),
            "attention_span": 3 + int(self.persona.patience * 2),
            "loss_sensitivity": self.persona.loss_aversion,
        }
        if psychology_profile:
            self.profile.update(psychology_profile)

        # Initialize Brain
        self.brain = DeepSeekBrain(
            agent_id=agent_id,
            persona={
                "risk_preference": self.persona.risk_appetite.value,
                "loss_aversion": self.profile.get("loss_sensitivity", 1.5)
            },
            model_router=model_router
        )
        # Sync initial confidence
        self.brain.state.confidence = self.profile.get("confidence_level", 0.5) * 100

    def _map_persona_to_risk_aversion(self) -> float:
        mapping = {
            RiskAppetite.CONSERVATIVE: 0.8,
            RiskAppetite.BALANCED: 0.5,
            RiskAppetite.AGGRESSIVE: 0.3,
            RiskAppetite.GAMBLER: 0.1
        }
        return mapping.get(self.persona.risk_appetite, 0.5)

    def bind_social_node(self, node_id: int, graph: SocialGraph):
        """Bind this agent to a node in the Social Graph."""
        self.social_node_id = node_id
        self.social_graph = graph
        # Sync agent ID to graph node
        if node_id in graph.agents:
            graph.agents[node_id].agent_id = self.agent_id

    async def perceive(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
    ) -> Dict[str, Any]:
        """
        感知阶段: 过滤信息
        """
        # 1. 过滤新闻 (基于 attention_span)
        span = int(self.profile.get("attention_span", 3))
        observed_news = public_news[:span] if public_news else []
        
        portfolio_value = self.get_total_value({market_snapshot.symbol: market_snapshot.last_price})
        initial = getattr(self, 'initial_cash', 100000)
        pnl = portfolio_value - initial
        pnl_pct = pnl / initial if initial > 0 else 0

        return {
            "snapshot": market_snapshot,
            "news": observed_news,
            "portfolio_value": portfolio_value,
            "cash": self.cash_balance,
            "pnl_pct": pnl_pct,
            "timestamp": market_snapshot.timestamp
        }

    async def reason(
        self,
        perceived_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        推理阶段: 调用 DeepSeek Brain (Enhanced with Emotion & Social)
        """
        # 计算当前情绪状态
        self.update_emotional_state(perceived_data)
        
        # 获取社交信号
        social_signal = self.perceive_social_signal(perceived_data)
        
        # 调用增强版推理方法 (reason_and_act logic integrated here)
        return await self.reason_and_act(perceived_data, self.emotional_state, social_signal)

    async def reason_and_act(
        self,
        perceived_data: Dict[str, Any],
        emotional_state: str,
        social_signal: str
    ) -> Dict[str, Any]:
        """
        核心认知方法：结合市场感知、情绪、社交信号进行推理决策
        此方法对应 User Request Prompt 3 中的 "reason_and_act"
        """
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        news = perceived_data["news"]
        
        # 1. Construct Market State
        market_state = {
            "price": snapshot.last_price,
            "trend": getattr(snapshot, "market_trend", "震荡"), 
            "panic_level": getattr(snapshot, "panic_level", 0.5), 
            "news": "; ".join(news) if news else "无重大新闻"
        }
        
        # Map numeric trend to string for Brain
        trend_val = snapshot.market_trend
        if trend_val > 0.02: market_state["trend"] = "上涨"
        elif trend_val < -0.02: market_state["trend"] = "下跌"
        else: market_state["trend"] = "震荡"

        # 2. Construct Account State
        account_state = {
            "cash": perceived_data["cash"],
            "market_value": perceived_data["portfolio_value"] - perceived_data["cash"],
            "pnl_pct": perceived_data["pnl_pct"]
        }

        # 3. Call Brain with Enhanced Context
        try:
            decision_output = await self.brain.think_async(
                market_state=market_state, 
                account_state=account_state,
                emotional_state=emotional_state,
                social_signal=social_signal
            )
            
            # Inject symbol/timestamp for decide phase
            decision_output["symbol"] = snapshot.symbol
            decision_output["timestamp"] = snapshot.timestamp
            
            return decision_output
            
        except Exception as e:
            print(f"Agent {self.agent_id} Brain Error: {e}")
            return {
                "decision": {"action": "HOLD"},
                "reasoning": f"Brain Failure: {e}",
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp
            }

    def update_emotional_state(self, perceived_data: Dict[str, Any]):
        """根据 PnL 和 市场波动更新情绪 (受 Persona 影响)"""
        pnl_pct = perceived_data.get("pnl_pct", 0)
        snapshot = perceived_data.get("snapshot")
        panic_level = getattr(snapshot, "panic_level", 0)
        
        # Adjust sensitivity based on Persona
        loss_threshold = -0.10 / (self.persona.loss_aversion / 2.25) # More averse -> smaller threshold (easier to regret)
        panic_threshold = 0.6 * self.persona.patience # More patient -> higher threshold
        
        if pnl_pct < loss_threshold:
            self.emotional_state = "Regretful" # 悔恨 (大亏)
        elif pnl_pct < loss_threshold / 2:
            self.emotional_state = "Anxious"   # 焦虑 (小亏)
        elif pnl_pct > 0.10:
            self.emotional_state = "Greedy"    # 贪婪 (大赚)
        elif pnl_pct > 0.05:
            self.emotional_state = "Confident" # 自信 (小赚)
        elif panic_level > panic_threshold:
            self.emotional_state = "Fearful"   # 恐惧 (市场恐慌)
        else:
            self.emotional_state = "Neutral"   # 中性
            
    def perceive_social_signal(self, perceived_data: Dict[str, Any]) -> str:
        """
        感知社交信号 (集成 SocialNetwork)
        """
        # 1. 优先使用真实社交网络
        if self.social_graph and self.social_node_id is not None:
            bearish_ratio = self.social_graph.get_bearish_ratio(self.social_node_id)
            
            # Thresholds based on conformity
            panic_threshold = 1.0 - (self.persona.conformity * 0.8) # High conformity -> Low threshold (0.2)
            
            if bearish_ratio > panic_threshold:
                return f"Panic Alert! {bearish_ratio:.0%} of neighbors are selling!"
            elif bearish_ratio > panic_threshold * 0.6:
                return "Neighbors are getting nervous."
                
        # 2. 只有在没有网络连接时，才回退到市场趋势代理
        snapshot = perceived_data.get("snapshot")
        trend = getattr(snapshot, "market_trend", 0)
        
        if trend > 0.05:
            return "Everyone is buying! (FOMO)"
        elif trend < -0.05:
            return "Everyone is panic selling!"
        else:
            return "Market is quiet."

    async def decide(
        self,
        reasoning_output: Dict[str, Any],
    ) -> Optional[Order]:
        """
        决策阶段: 将 Brain 的输出转化为 Order 对象
        """
        decision = reasoning_output.get("decision", {})
        action = decision.get("action", "HOLD")
        
        if action == "HOLD" or not action:
            return None
            
        qty = decision.get("qty", 0)
        # Brain might return JSON with "price" or "limit_price"
        price = decision.get("price", 0.0)
        
        symbol = reasoning_output.get("symbol", "UNKNOWN")
        timestamp = reasoning_output.get("timestamp", time.time())
        side = None

        if action == "BUY":
            side = OrderSide.BUY
        elif action == "SELL":
            side = OrderSide.SELL
        else:
            return None
            
        if qty <= 0:
            return None
            
        # 资金/持仓 预检查 (虽然 OrderBook 也会查，但 Agent 应有自我认知)
        if side == OrderSide.BUY:
            cost = price * qty
            if cost > self.cash_balance:
                # 资金不足，尝试调整
                if price > 0:
                    qty = int(self.cash_balance / price) // 100 * 100
                else:
                    qty = 0
                if qty <= 0:
                    return None
        elif side == OrderSide.SELL:
            holding = self.portfolio.get(symbol, 0)
            if qty > holding:
                qty = holding
                if qty <= 0:
                    return None
        
        # 生成 Order
        order = Order(
            symbol=symbol,
            price=price,
            quantity=qty,
            side=side,
            order_type=OrderType.LIMIT,
            agent_id=self.agent_id,
            timestamp=timestamp
        )
        return order

    async def update_memory(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        """
        更新记忆
        """
        # Updates both Vector Memory (Brain) and Local List
        
        # 1. Update Brain's State (Confidence, PnL)
        pnl = outcome.get("pnl", 0.0)
        pnl_pct = 0.0 # Need total capital to calc pct, simplified here
        self.brain.state.update_after_trade(pnl, pnl_pct)
        
        # 2. Update Vector Memory if significant
        if abs(pnl) > 1000:
             content = f"Decision: {decision}. Outcome: {outcome}"
             score = 1.0 if pnl > 0 else -1.0
             self.brain.memory.add_memory(content, score)

    # ------------------------------------------
    # 额外方法 (针对 TraderAgent 特定请求)
    # ------------------------------------------
    
    def get_psychology_description(self) -> str:
        """返回心理画像描述"""
        return (
            f"Risk: {self.profile.get('risk_aversion', 0.5):.2f}, "
            f"Conf: {self.brain.state.confidence:.1f}"
        )
