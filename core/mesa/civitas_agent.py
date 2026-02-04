# file: core/mesa/civitas_agent.py
"""
Mesa Agent 适配器

将 CognitiveAgent 包装为 Mesa Agent，实现与 Mesa 调度器的兼容。
"""

from mesa import Agent
from typing import Optional, Dict, Any

from agents.cognition.cognitive_agent import CognitiveAgent
from agents.cognition.utility import InvestorType


class CivitasAgent(Agent):
    """
    Civitas Mesa Agent
    
    包装 CognitiveAgent，使其兼容 Mesa 的调度和数据收集机制。
    
    Mesa 3.0+ API:
    - step(): 执行决策逻辑
    - advance(): 应用决策结果 (用于 SimultaneousActivation 模式)
    
    Attributes:
        cognitive_agent: 底层认知 Agent
        pending_action: 待执行的动作 (step 产生, advance 执行)
        wealth: 总资产 (用于 DataCollector)
        position: 持仓量
        sentiment: 情绪状态 [-1, 1]
    """
    
    def __init__(
        self,
        model: "CivitasModel",
        investor_type: InvestorType = InvestorType.NORMAL,
        initial_cash: float = 100000.0,
        use_local_reasoner: bool = True
    ):
        """
        初始化 Mesa Agent
        
        Args:
            model: Mesa Model 实例
            investor_type: 投资者类型
            initial_cash: 初始资金
            use_local_reasoner: 是否使用本地推理器 (避免 API 调用)
        """
        super().__init__(model)
        
        # 创建底层认知 Agent
        self.cognitive_agent = CognitiveAgent(
            agent_id=f"agent_{self.unique_id}",
            investor_type=investor_type,
            use_local_reasoner=use_local_reasoner
        )
        
        # 账户状态
        self.cash = initial_cash
        self.position = 0  # 持仓股数
        self.avg_cost = 0.0  # 平均成本
        
        # 待执行动作 (用于 advance 阶段)
        self.pending_action: Optional[Dict] = None
        
        # 可观测状态 (用于 DataCollector)
        self.wealth = initial_cash
        self.pnl_pct = 0.0
        self.sentiment = 0.0  # [-1, 1]
        self.confidence = 0.5
    
    def step(self) -> None:
        """
        决策阶段 - 获取市场数据并做出决策
        
        在 SimultaneousActivation 模式下，所有 Agent 先执行 step()，
        然后再统一执行 advance()。
        """
        # 获取市场状态
        market_state = self.model.get_market_state()
        
        # 计算账户状态
        current_price = market_state.get("price", 0)
        market_value = self.position * current_price
        self.wealth = self.cash + market_value
        
        if self.avg_cost > 0 and self.position > 0:
            self.pnl_pct = (current_price - self.avg_cost) / self.avg_cost
        else:
            self.pnl_pct = 0.0
        
        account_state = {
            "cash": self.cash,
            "market_value": market_value,
            "pnl_pct": self.pnl_pct,
            "avg_cost": self.avg_cost
        }
        
        # 做出决策
        decision = self.cognitive_agent.make_decision(market_state, account_state)
        
        # 更新情绪状态
        self.sentiment = decision.greed_level - decision.fear_level
        self.confidence = self.cognitive_agent.confidence_tracker.confidence
        
        # 保存待执行动作
        self.pending_action = {
            "action": decision.final_action,
            "quantity": decision.final_quantity,
            "price": current_price
        }
    
    def advance(self) -> None:
        """
        执行阶段 - 应用决策结果
        
        在 SimultaneousActivation 模式下，所有 Agent 的 step() 执行完毕后，
        统一执行 advance() 以更新状态。
        """
        if self.pending_action is None:
            return
        
        action = self.pending_action["action"]
        quantity = self.pending_action["quantity"]
        price = self.pending_action["price"]
        
        if action == "BUY" and quantity > 0:
            cost = quantity * price
            if self.cash >= cost:
                # 更新平均成本
                total_cost = self.avg_cost * self.position + cost
                self.position += quantity
                self.avg_cost = total_cost / self.position if self.position > 0 else 0
                self.cash -= cost
                
        elif action == "SELL" and quantity > 0:
            sell_qty = min(quantity, self.position)
            if sell_qty > 0:
                revenue = sell_qty * price
                self.position -= sell_qty
                self.cash += revenue
                
                # 记录交易结果
                pnl = (price - self.avg_cost) / self.avg_cost if self.avg_cost > 0 else 0
                self.cognitive_agent.record_outcome(
                    pnl=pnl,
                    market_summary=f"卖出 {sell_qty} 股 @ {price:.2f}"
                )
                
                if self.position == 0:
                    self.avg_cost = 0.0
        
        # 清空待执行动作
        self.pending_action = None
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取可序列化的状态 (用于 DataCollector)
        """
        return {
            "unique_id": self.unique_id,
            "wealth": self.wealth,
            "cash": self.cash,
            "position": self.position,
            "pnl_pct": self.pnl_pct,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "investor_type": self.cognitive_agent.investor_type.value
        }
