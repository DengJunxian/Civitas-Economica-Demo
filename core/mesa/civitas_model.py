# file: core/mesa/civitas_model.py
"""
Mesa Model 编排器

使用 Mesa 框架管理 Agent 群体、市场状态和数据收集。
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any

from core.society.network import SocialGraph, InformationDiffusion # [NEW]
from agents.persona import Persona, PersonaGenerator, RiskAppetite # [NEW]
from core.mesa.civitas_agent import CivitasAgent
from agents.cognition.utility import InvestorType
import time
from core.market_engine import Order, OrderType
from core.time_manager import SimulationClock
from core.utils import truncate_text

class CivitasModel(Model):
    """
    Civitas Mesa Model
    
    编排整个模拟系统：
    - 管理 Agent 群体
    - 控制时钟步进
    - 收集微观数据
    - 计算宏观指标 (如 CSAD)
    
    Mesa 3.0+ API:
    - 使用 self.agents (AgentSet) 替代 self.schedule
    - step(): 调用 self.agents.do("step") 和 self.agents.do("advance")
    
    Attributes:
        n_agents: Agent 数量
        current_price: 当前市场价格
        datacollector: Mesa 数据收集器
        clock: 仿真时钟
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        panic_ratio: float = 0.3,
        quant_ratio: float = 0.1,
        initial_price: float = 3000.0,
        seed: Optional[int] = None,
        model_router: Optional[Any] = None
    ):
        """
        初始化模型
        
        Args:
            n_agents: Agent 数量
            panic_ratio: 恐慌型散户比例
            quant_ratio: 量化投资者比例
            initial_price: 初始价格
            seed: 随机种子
            model_router: 模型路由器 (用于 Policy 和 LLM)
        """
        super().__init__(seed=seed)
        
        self.n_agents = n_agents
        self.shared_router = model_router
        
        # 0. 初始化仿真时钟
        self.clock = SimulationClock()
        
        # [NEW] 初始化社会网络与传播引擎
        # k must be <= n and even for Watts-Strogatz
        graph_k = min(10, n_agents - 1)
        if graph_k % 2 != 0:
            graph_k = max(2, graph_k - 1)
        self.social_graph = SocialGraph(n_agents=n_agents, k=graph_k, p=0.1, seed=seed)
        self.diffusion = InformationDiffusion(self.social_graph)
        
        # 1. 初始化市场引擎 (单一真实之源)
        from core.market_engine import MarketDataManager
        # 注意: 我们不需要在这里传 api_key，MarketDataManager 会自己从 Config 或 Env 获取
        self.market_manager = MarketDataManager(api_key_or_router=model_router, load_real_data=True, clock=self.clock)
        
        # 覆写初始价格 (如果有指定)
        if hasattr(self.market_manager.engine, 'last_price'):
            pass
            
        self.current_price = self.market_manager.engine.last_price
        self.price_history: List[float] = [self.current_price]
        
        # 市场状态
        self.trend = "震荡"
        
        # 从 Engine 获取状态
        self.panic_level = self.market_manager.panic_level
        self.volatility = 0.02 # 依然保留作为参考，或者从 Engine 计算历史波动率
        self.csad = 0.0
        
        # 步骤指标
        self.last_step_trades_count = 0
        self.last_smart_sentiment = 0.0
        
        # 创建异质性 Agent 群体 (Tier 1)
        self._create_agents(panic_ratio, quant_ratio)
        
        # 初始化 Tier 2 (Vectorized Population)
        from agents.population import StratifiedPopulation
        # 传入 Tier 1 Agent 列表供 Tier 2 关注
        self.population = StratifiedPopulation(
            n_smart=0,  # Tier 1 由 Mesa 管理
            n_vectorized=5000, # 默认 5000 散户
            smart_agents=list(self.agents)
        )
        
        # 初始化数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Price": "current_price",
                "CSAD": "csad",
                "Panic_Level": "panic_level",
                "Avg_Wealth": lambda m: np.mean([a.wealth for a in m.agents]),
                "Avg_Sentiment": lambda m: np.mean([a.sentiment for a in m.agents]),
                "Volume": lambda m: m.market_manager.sim_candles[-1].volume if m.market_manager.sim_candles else 0,
                "Infected_Count": lambda m: m.diffusion.history[-1]['infected'] if m.diffusion.history else 0 # [NEW]
            },
            agent_reporters={
                "Wealth": "wealth",
                "Position": "position",
                "PnL_Pct": "pnl_pct",
                "Sentiment": "sentiment",
                "Confidence": "confidence",
            }
        )
    
    def _create_agents(self, panic_ratio: float, quant_ratio: float) -> None:
        """创建异质性 Agent 群体 (集成 Persona 和 Social Network)"""
        n_panic = int(self.n_agents * panic_ratio)
        n_quant = int(self.n_agents * quant_ratio)
        n_normal = self.n_agents - n_panic - n_quant
        
        # Generate generic personas first
        personas = PersonaGenerator.generate_distribution(self.n_agents)
        
        # Assign Agents
        idx = 0
        
        # 1. Panic Retail (High Conformity, Low Patience)
        for _ in range(n_panic):
            p = personas[idx]
            p.risk_appetite = RiskAppetite.GAMBLER
            p.conformity = 0.9
            p.patience = 0.1
            p.loss_aversion = 3.0
            
            self._create_single_agent(idx, InvestorType.PANIC_RETAIL, p)
            idx += 1
            
        # 2. Quants (Low Conformity, Disciplined)
        for _ in range(n_quant):
            p = personas[idx]
            p.risk_appetite = RiskAppetite.BALANCED
            p.conformity = 0.05
            p.influence = 0.8 # Quants might lead trend
            p.loss_aversion = 1.0 # Neutral to loss
            
            self._create_single_agent(idx, InvestorType.DISCIPLINED_QUANT, p, cash=1_000_000)
            idx += 1
            
        # 3. Normal Agents
        for _ in range(n_normal):
            p = personas[idx]
            self._create_single_agent(idx, InvestorType.NORMAL, p)
            idx += 1
            
    def _create_single_agent(self, idx: int, inv_type: InvestorType, persona: Persona, cash: float = None) -> CivitasAgent:
        if cash is None:
            cash = np.random.uniform(50000, 500000)
            
        agent = CivitasAgent(
            model=self,
            investor_type=inv_type,
            initial_cash=cash,
            use_local_reasoner=True,
            persona=persona
        )
        
        # Bind to Social Network
        agent.core.bind_social_node(idx, self.social_graph)
        return agent
    
    def get_market_state(self) -> Dict[str, Any]:
        """获取当前市场状态 (供 Agent 决策使用)"""
        # 从 Engine 获取最新深度和价格
        # 模拟部分信息透明度
        return {
            "price": self.current_price,
            "trend": self.trend, # 仍需维护或通过 technically analysis 计算
            "panic_level": self.panic_level,
            "volatility": self.volatility,
            "csad": self.csad,
            "news": truncate_text(self.market_manager.current_news, 500),
            "dates": self.market_manager.history_candles[-1].timestamp if self.market_manager.history_candles else "2024-01-01",
            "infected_ratio": self.diffusion.history[-1]['infected'] / self.n_agents if self.diffusion.history else 0 # [NEW]
        }
    
    @property
    def current_dt(self) -> Any:
        # 返回当前仿真时间 (pd.Timestamp)
        if self.market_manager.sim_candles:
            ts = self.market_manager.sim_candles[-1].timestamp
        elif self.market_manager.history_candles:
            ts = self.market_manager.history_candles[-1].timestamp
        else:
            ts = "2024-01-01"
        
        import pandas as pd
        return pd.Timestamp(ts)

    def set_policy(self, name: str, param: str, value) -> None:
        """
        动态设置监管策略参数
        
        Args:
            name: 策略名称 ("circuit_breaker" | "tax")
            param: 参数名 (e.g. "threshold_pct", "rate", "active")
            value: 参数值
        """
        self.market_manager.policy_manager.set_policy_param(name, param, value)
    
    def get_policy_status(self) -> Dict[str, Any]:
        """获取当前策略状态"""
        pm = self.market_manager.policy_manager
        cb = pm.policies.get("circuit_breaker")
        tax = pm.policies.get("tax")
        return {
            "circuit_breaker": {
                "active": cb.active if cb else False,
                "threshold": cb.threshold_pct if cb else 0,
                "is_halted": cb.is_halted if cb else False,
            },
            "transaction_tax": {
                "active": tax.active if tax else False,
                "rate": tax.rate if tax else 0,
            }
        }

    def step(self) -> None:
        """
        [已废弃] 同步 Step 方法
        请使用 async_step() 代替。
        """
        raise NotImplementedError("CivitasModel.step() is deprecated. Please use await model.async_step() instead.")

    async def async_step(self) -> None:
        """
        执行一个模拟步骤 (异步版)
        
        1. 收集 Agent 提示词
        2. 并行调用 LLM (Brain)
        3. 分发结果并执行交易
        4. 市场结算
        """
        # [NEW] Update Circuit Breaker reference price at start of each step
        cb = self.market_manager.policy_manager.policies.get("circuit_breaker")
        if cb:
            cb.update_reference_price(self.current_price)
        
        # [NEW] Update Social Network Dynamics
        if self.diffusion:
            self.diffusion.update_sentiment_propagation()
        
        # 阶段 1: 市场感知
        market_state = self.get_market_state()
        snapshot = self.market_manager.get_market_snapshot() # Get snapshot object
        # Ensure snapshot has required fields for Agent
        
        news = [market_state["news"]] if market_state["news"] else []
        
        # 阶段 2: Agent 并行决策 (Tier 1)
        # Gather all agent.async_act coroutines
        agent_tasks = []
        active_agents = []
        
        for agent in self.agents:
            if isinstance(agent, CivitasAgent):
                agent_tasks.append(agent.async_act(snapshot, news))
                active_agents.append(agent)
        
        # Concurrent execution
        orders = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # 阶段 3: 订单提交
        smart_actions = []
        for agent, order_or_err in zip(active_agents, orders):
            if isinstance(order_or_err, Exception):
                print(f"[Model Error] Agent {agent.unique_id} act failed: {order_or_err}")
                smart_actions.append(0)
                continue
                
            order = order_or_err
            if order:
                self.market_manager.submit_agent_order(order)
                # Record action for Tier 2 sentiment
                # Order side can be string or Enum
                side_str = str(order.side).lower()
                if "buy" in side_str: smart_actions.append(1)
                elif "sell" in side_str: smart_actions.append(-1)
                else: smart_actions.append(0)
            else:
                smart_actions.append(0)

        # 更新 Smart Sentiment 指标
        self.last_smart_sentiment = np.mean(smart_actions) if smart_actions else 0.0

        # 阶段 5: 执行交易 (Tier 1 Advance)
        # Mesa's step/advance structure is less relevant here as we handled actions explicitly
        # But we call it to maintain compatibility if any other logic exists in agents
        self.agents.do("advance")
        
        # 阶段 5.5: Tier 2 (Vectorized) 决策与执行
        # A. 更新情绪
        trend_signal = 1.0 if self.trend == "上涨" else -1.0
        if self.trend == "下跌": trend_signal = -1.0
        elif self.trend == "震荡": trend_signal = 0.0
        
        self.population.update_tier2_sentiment(smart_actions, trend_signal)
        
        # B. 生成决策
        v_actions, v_qtys, v_prices = self.population.generate_tier2_decisions(self.current_price)
        
        # C. 批量下单 (采样部分以减轻撮合压力)
        active_indices = np.where(v_actions != 0)[0]
        if len(active_indices) > 0:
            # 限制最大单号量，防止阻塞
            sample_size = min(len(active_indices), 500)
            chosen_idx = np.random.choice(active_indices, sample_size, replace=False)
            
            ts = self.clock.timestamp
            for idx in chosen_idx:
                side = 'buy' if v_actions[idx] == 1 else 'sell'
                order = Order(
                    symbol=self.market_manager.engine.symbol,
                    price=float(v_prices[idx]),
                    quantity=int(v_qtys[idx]),
                    side=side,
                    order_type=OrderType.LIMIT, 
                    agent_id=f"Vec_{idx}",
                    timestamp=ts
                )
                self.market_manager.submit_agent_order(order)
                
        # 阶段 6: 市场结算
        # 获取本 Step 所有成交 (Tier 1 + Tier 2)
        step_trades = self.market_manager.engine.flush_step_trades()
        
        # 更新成交量指标
        self.last_step_trades_count = len(step_trades)
        
        # 同步 Tier 2 成交状态
        vec_indices = []
        vec_prices = []
        vec_qtys = []
        vec_dirs = []
        
        for t in step_trades:
            # Check buyer (Standardized to buyer_agent_id)
            buyer_id = getattr(t, 'buyer_agent_id', getattr(t, 'buy_agent_id', None))
            seller_id = getattr(t, 'seller_agent_id', getattr(t, 'sell_agent_id', None))
            
            # 检查买方
            if buyer_id and buyer_id.startswith("Vec_"):
                idx = int(buyer_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(1) # Buy
            # 检查卖方
            if seller_id and seller_id.startswith("Vec_"):
                idx = int(seller_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(-1) # Sell
                
        if vec_indices:
            self.population.sync_tier2_execution(
                np.array(vec_indices),
                np.array(vec_prices),
                np.array(vec_qtys),
                np.array(vec_dirs)
            )

        last_date = "2024-01-01"
        if self.market_manager.sim_candles:
            last_date = self.market_manager.sim_candles[-1].timestamp
        elif self.market_manager.history_candles:
            last_date = self.market_manager.history_candles[-1].timestamp
            
        # 传入 Trades 生成 K 线
        current_candle = self.market_manager.finalize_step(self.steps, last_date, trades=step_trades)
        
        # 更新 Model 状态
        self.current_price = current_candle.close
        self.price_history.append(self.current_price)
        self._calculate_csad()
        self._update_trend()
        
        # 收集数据
        self.datacollector.collect(self)
        
        # 推进仿真时钟
        self.clock.tick()
        self.steps += 1
        
    def _update_trend(self):
        """简单趋势判断"""
        if len(self.price_history) >= 5:
            recent = self.price_history[-5:]
            if recent[-1] > recent[0] * 1.02:
                self.trend = "上涨"
            elif recent[-1] < recent[0] * 0.98:
                self.trend = "下跌"
            else:
                self.trend = "震荡"

    def _calculate_csad(self) -> None:
        """计算截面绝对偏差 (CSAD)"""
        # 使用 Agent 的 Sentiment 作为代理
        sentiments = [a.sentiment for a in self.agents]
        if not sentiments:
            self.csad = 0.0
            return
        
        avg_sentiment = np.mean(sentiments)
        self.csad = np.mean(np.abs(np.array(sentiments) - avg_sentiment))
        
    async def run_simulation(self, n_steps: int = 100) -> None:
        """运行完整模拟 (异步)"""
        for _ in range(n_steps):
            await self.async_step()
    
    def get_results(self) -> Dict[str, Any]:
        """获取模拟结果"""
        model_data = self.datacollector.get_model_vars_dataframe()
        agent_data = self.datacollector.get_agent_vars_dataframe()
        
        return {
            "model_data": model_data,
            "agent_data": agent_data,
            "final_price": self.current_price,
            "price_history": self.price_history,
            "n_steps": self.steps
        }


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mesa Model 测试")
    print("=" * 60)
    
    async def main():
        # 创建模型
        model = CivitasModel(n_agents=50, seed=42)
        print(f"创建 {model.n_agents} 个 Agent")
        
        # 运行 20 步
        print("\n[运行模拟 - 20 步]")
        for i in range(20):
            await model.async_step()
            if i % 5 == 0:
                print(f"  Step {model.steps}: Price={model.current_price:.2f}, "
                      f"CSAD={model.csad:.4f}, Panic={model.panic_level:.2f}")
        
        # 获取结果
        results = model.get_results()
        print(f"\n[结果]")
        print(f"  最终价格: {results['final_price']:.2f}")
        print(f"  Model DataFrame shape: {results['model_data'].shape}")
        print(f"  Agent DataFrame shape: {results['agent_data'].shape}")
    
    import asyncio
    asyncio.run(main())
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
