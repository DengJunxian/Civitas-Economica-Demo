# file: core/mesa/civitas_agent.py
"""
Mesa Agent 适配器

将 CognitiveAgent 包装为 Mesa Agent，实现与 Mesa 调度器的兼容。
"""

from mesa import Agent
from typing import Optional, Dict, Any

from agents.cognition.cognitive_agent import CognitiveAgent
from agents.cognition.utility import InvestorType
from config import GLOBAL_CONFIG


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
        
        # 账户状态 - 使用 Portfolio 管理 T+1 和 批次成本
        from core.account import Portfolio
        self.portfolio = Portfolio(initial_cash=initial_cash)
        
        # 待执行动作 (用于 advance 阶段)
        self.pending_action: Optional[Dict] = None
        
        # 可观测状态 (用于 DataCollector)
        self.wealth = initial_cash
        self.pnl_pct = 0.0
        self.sentiment = 0.0  # [-1, 1]
        self.confidence = 0.5

    @property
    def position(self) -> int:
        """From Portfolio"""
        return self.portfolio.get_total_holdings_qty("SH000001")
        
    @property
    def id(self) -> str:
        """Compatibility with StratifiedPopulation"""
        return str(self.unique_id)
    
    async def prepare_decision(self, market_state: Dict) -> Tuple[Optional[List[Dict]], Dict]:
        """
        [阶段 1] 只有准备 Prompt 的过程
        
        返回: (Messages for LLM, Account State Snapshot)
        """
        # 1. 获取当前持仓和现金
        ticker = self.model.symbol
        # position = self.portfolio.positions.get(ticker, 0) # This returns Position object
        pos_obj = self.portfolio.positions.get(ticker)
        total_qty = pos_obj.quantity if pos_obj else 0
        
        # 2. 计算当前浮盈
        # market_state should contain current price
        current_price = market_state.get("price", 0)
        avg_cost = pos_obj.average_cost if pos_obj else 0
        market_value = total_qty * current_price
        
        if total_qty > 0 and avg_cost > 0:
            self.pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
        else:
            self.pnl_pct = 0.0
            
        # 3. 准备账户状态快照
        # 使用仿真时间进行 T+1 检查
        sim_time = self.model.current_dt if hasattr(self.model, "current_dt") else pd.Timestamp.now()
        
        account_state = {
            "cash": self.portfolio.available_cash,
            "market_value": market_value,
            "pnl_pct": self.pnl_pct,
            "avg_cost": avg_cost,
            "position": total_qty,
            "sellable": self.portfolio.get_sellable_qty(ticker, sim_time)
        }
        
        # 4. 调用 CognitiveAgent 生成 Prompt (或本地决策)
        # Use async version for memory retrieval
        prompt = await self.cognitive_agent.prepare_decision_prompt_async(market_state, account_state)
        return prompt, account_state

    def finalize_decision(self, result: Any, market_state: Dict, account_state: Dict) -> None:
        """
        [阶段 2] 决策应用
        
        接收推理结果 (ReasoningResult or Decision)，更新状态并设置 pending_action
        """
        current_price = market_state.get("price", 0)
        
        # 1. 生成最终 CognitiveDecision 对象
        # 如果 result 是 ReasoningResult (LLM)，调用 finalize_decision_from_result
        # 如果 result 是 CognitiveDecision (Local?), 适配之
        
        # 统一: 假设传入的是 ReasoningResult (无论 Local 还是 LLM)
        decision_obj = self.cognitive_agent.finalize_decision_from_result(result, market_state, account_state)
        
        # 2. 更新情绪状态
        # 确保 decision_obj 有这些属性 (CognitiveDecision dataclass vs simple Decision dataclass)
        # 注意: llm_brain.Decision 有 greed/fear_level. CognitiveDecision 也有.
        # Check what finalize_decision_from_result returns (currently raw_decision which is llm_brain.Decision)
        
        self.sentiment = getattr(decision_obj, "greed_level", 0.0) - getattr(decision_obj, "fear_level", 0.0)
        self.confidence = getattr(decision_obj, "confidence", 0.5) # simple Decision has confidence
        
        # 3. 设置待执行动作
        self.pending_action = {
            "action": getattr(decision_obj, "action", getattr(decision_obj, "final_action", "HOLD")),
            "quantity": getattr(decision_obj, "quantity", getattr(decision_obj, "final_quantity", 0)),
            "price": current_price # 市价单暂定为当前价，限价单逻辑后续细化
        }
        
    def step(self) -> None:
        """[Deprecated] 请使用 prepare_decision 和 finalize_decision"""
        # 为了兼容旧的 Mesa 调用 (如果还在用)，保留同步逻辑
        import warnings
        warnings.warn("CivitasAgent.step() is deprecated. Use prepare/finalize flow.", DeprecationWarning)
        pass

    def advance(self) -> None:
        """
        执行阶段 - 生成订单并提交给撮合引擎
        """
        if self.pending_action is None:
            return
        
        action = self.pending_action["action"]
        quantity = self.pending_action["quantity"]
        price = self.pending_action["price"]
        ticker = "SH000001"
        
        # 忽略无效动作
        if quantity <= 0 or action not in ["BUY", "SELL"]:
            self.pending_action = None
            return

        # T+1 和 资金 预检查 (虽然 Engine 也会查，但 Agent 应避免提交无效单)
        if action == "SELL":
            if self.portfolio.withdrawable_cash > 20000: # 某种保留逻辑?
                 # 使用仿真时间
                sim_time = self.model.current_dt if hasattr(self.model, "current_dt") else pd.Timestamp.now()
                sellable = self.portfolio.get_sellable_qty(ticker, sim_time)
                if quantity > sellable:
                    quantity = sellable # 自动调整为最大可卖
                    if quantity == 0:
                        self.pending_action = None
                        return
        elif action == "BUY":
            cost_per_share = price * (1 + GLOBAL_CONFIG.TAX_RATE_COMMISSION)
            
            # 防御性检查: 避免除以零
            if cost_per_share <= 0:
                 # 价格异常 (0 or negative)，放弃买入
                 self.pending_action = None
                 return
                 
            cost = cost_per_share * quantity
            if cost > self.portfolio.available_cash:
                # 简单调整数量
                max_afford = self.portfolio.available_cash / cost_per_share
                quantity = int(max_afford // 100 * 100) # 手数取整
                if quantity == 0:
                    self.pending_action = None
                    return

        # 1. 生成订单对象
        from core.market_engine import Order
        
        side = 'buy' if action == 'BUY' else 'sell'
        

        order = Order(
            agent_id=str(self.unique_id),
            side=side,
            price=price,
            quantity=int(quantity),
            timestamp=self.model.clock.timestamp
        )
        # Note: Order.create now handles ID generation
        
        # 2. 提交订单
        if hasattr(self.model, "submit_order"):
            trades = self.model.submit_order(order)
        else:
            trades = []

        # 3. 更新账户
        self._process_trades(trades)
        
        # 清空
        self.pending_action = None

    def _process_trades(self, trades: list) -> None:
        """处理成交记录"""
        import pandas as pd
        ticker = "SH000001"
        ts = pd.Timestamp.now()
        
        for trade in trades:
            # Re-infer side from context or check Trade details if available
            # Let's rely on self.pending_action['action'] because these trades come from THAT action.
            # (Assuming submit_order is synchronous and exclusive to that order)
             
            # If pending_action is None (unexpected), try to guess
            current_action = self.pending_action["action"] if self.pending_action else None
            
            if current_action == "BUY":
                # I Bought
                self.portfolio.buy(
                    ticker=ticker,
                    price=trade.price,
                    qty=trade.quantity,
                    time=ts,
                    transaction_cost=trade.buyer_fee
                )
            elif current_action == "SELL":
                # I Sold
                self.portfolio.sell(
                    ticker=ticker,
                    price=trade.price,
                    qty=trade.quantity,
                    time=ts,
                    transaction_cost=trade.seller_fee + trade.seller_tax
                )
                
                # Record Outcome
                # For PnL recording, we need a snapshot of avg_cost from BEFORE the sale?
                # Actually Portfolio handles accounting, but doesn't track "Trade PnL" explicitly for memory.
                # We can approximate or just use the agent's tracked avg_cost (which is updated via Portfolio now?)
                # Wait, CivitasAgent.step() calculates PnL pct.
                # For "Memory", we want realized PnL of *this specific trade*.
                # Simplified: use (Price - AvgCost) / AvgCost
                
                # We need to access the portfolio's avg cost for this ticker
                # But Portfolio doesn't expose a simple "Avg Cost" property easily without calculation
                # Let's calculate it from the batches *before* the sell? Too late now.
                # Use current step's self.avg_cost (calculated in step())
                
                if hasattr(self, 'avg_cost') and self.avg_cost > 0:
                    pnl_val = (trade.price - self.avg_cost) * trade.quantity
                    pnl_pct = (trade.price - self.avg_cost) / self.avg_cost
                    
                    if hasattr(self.cognitive_agent, "record_outcome"):
                         self.cognitive_agent.record_outcome(
                            pnl=pnl_pct,
                            market_summary=f"卖出 {trade.quantity} 股 @ {trade.price:.2f}"
                        )

    def get_state(self) -> Dict[str, Any]:
        """
        获取可序列化的状态 (用于 DataCollector)
        """
        ticker = "SH000001"
        pos = self.portfolio.get_total_holdings_qty(ticker)
        return {
            "unique_id": self.unique_id,
            "wealth": self.wealth,
            "cash": self.portfolio.available_cash,
            "position": pos,
            "pnl_pct": self.pnl_pct,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "investor_type": self.cognitive_agent.investor_type.value
        }
