# file: core/mesa/civitas_model.py
"""
Mesa Model 编排器

使用 Mesa 框架管理 Agent 群体、市场状态和数据收集。
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np
from typing import Dict, List, Optional, Any

from core.mesa.civitas_agent import CivitasAgent
from agents.cognition.utility import InvestorType
import time
from core.market_engine import Order
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
        
        # 1. 初始化市场引擎 (单一真实之源)
        from core.market_engine import MarketDataManager
        # 注意: 我们不需要在这里传 api_key，MarketDataManager 会自己从 Config 或 Env 获取
        self.market_manager = MarketDataManager(api_key_or_router=model_router, load_real_data=True, clock=self.clock)
        
        # 覆写初始价格 (如果有指定)
        if hasattr(self.market_manager.engine, 'last_price'):
            # 如果加载了历史数据，引擎价格可能已经不同。
            # 如果强制指定 initial_price，我们需要 reset engine?
            # 简单起见，我们信任 MarketLoader 加载的最后价格作为起点，
            # 除非 initial_price 显式传入且不同于默认。
            # 这里我们让 engine 决定价格。
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
                "Volume": lambda m: m.market_manager.sim_candles[-1].volume if m.market_manager.sim_candles else 0
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
        """创建异质性 Agent 群体"""
        n_panic = int(self.n_agents * panic_ratio)
        n_quant = int(self.n_agents * quant_ratio)
        n_normal = self.n_agents - n_panic - n_quant
        
        # 恐慌型散户
        for _ in range(n_panic):
            CivitasAgent(
                model=self,
                investor_type=InvestorType.PANIC_RETAIL,
                initial_cash=np.random.uniform(50000, 200000),
                use_local_reasoner=True
            )
        
        # 量化投资者
        for _ in range(n_quant):
            CivitasAgent(
                model=self,
                investor_type=InvestorType.DISCIPLINED_QUANT,
                initial_cash=np.random.uniform(500000, 2000000),
                use_local_reasoner=True
            )
        
        # 普通投资者
        for _ in range(n_normal):
            CivitasAgent(
                model=self,
                investor_type=InvestorType.NORMAL,
                initial_cash=np.random.uniform(100000, 500000),
                use_local_reasoner=True
            )
    
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
            "dates": self.market_manager.history_candles[-1].timestamp if self.market_manager.history_candles else "2024-01-01"
        }
    
    @property
    def current_dt(self) -> Any:
        # 返回当前仿真时间 (pd.Timestamp)
        # 优先使用 sim_candles 的最后一根 K 线时间，或者 history_candles
        # 如果都没有，返回一个默认起始时间
        if self.market_manager.sim_candles:
            ts = self.market_manager.sim_candles[-1].timestamp
        elif self.market_manager.history_candles:
            ts = self.market_manager.history_candles[-1].timestamp
        else:
            ts = "2024-01-01"
        
        import pandas as pd
        return pd.Timestamp(ts)

    def submit_order(self, order) -> List[Any]:
        """
        Agent 提交的一级入口
        
        Args:
           order: core.market_engine.Order 对象
           
        Returns:
           List[Trade]: 该订单产生的成交记录 (作为 Taker)
        """
        # 转发给 MarketManager
        trades = self.market_manager.submit_agent_order(order)
        return trades

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
        2. 并行调用 LLM
        3. 分发结果并执行交易
        4. 市场结算
        """
        from agents.cognition.llm_brain import ReasoningResult, Decision
        
        # 阶段 1: 收集 Prompt & 识别 Local Agents
        llm_agents = []
        prompts = []
        local_agents = []
        
        # 准备市场状态缓存 (避免重复计算)
        market_state = self.get_market_state()
        
        for agent in self.agents:
            if not isinstance(agent, CivitasAgent):
                continue
                
            # 使用新拆分的 API: prepare_decision
            prompt, acc_state = agent.prepare_decision(market_state)
            
            if prompt:
                llm_agents.append((agent, acc_state))
                prompts.append(prompt)
            else:
                local_agents.append((agent, acc_state))
        
        # 阶段 2: 并行推理 (LLM)
        results = []
        if prompts:
            # 懒加载 Router
            if not hasattr(self, 'shared_router'):
                from core.model_router import ModelRouter
                from config import GLOBAL_CONFIG
                self.shared_router = ModelRouter(
                    deepseek_key=GLOBAL_CONFIG.DEEPSEEK_API_KEY,
                    hunyuan_key=GLOBAL_CONFIG.HUNYUAN_API_KEY,
                    zhipu_key=GLOBAL_CONFIG.ZHIPU_API_KEY
                )

            # 获取 Priority
            priority = self.shared_router.get_model_priority("SMART")
            
            import asyncio
            # 创建 Tasks
            tasks = [
                self.shared_router.call_with_fallback(msgs, priority)
                for msgs in prompts
            ]
            
            # 并发执行
            raw_results = await asyncio.gather(*tasks)
            
            # Parse results
            for (content, reasoning, model_name), (agent, acc_state) in zip(raw_results, llm_agents):
                # Reuse agent's reasoner to parse
                decision = agent.cognitive_agent.reasoner._parse_response(content)
                res = ReasoningResult(decision, content, reasoning, model_name)
                
                results.append((agent, res, acc_state))

        # 阶段 3: 处理 Local Agent
        for agent, acc_state in local_agents:
            # Local Reasoner call (Sync & Fast)
            res = agent.cognitive_agent.reasoner.derive_decision(market_state, acc_state)
            results.append((agent, res, acc_state))
            
        # 阶段 4: 分发决策 & 记录状态
        for agent, result, acc_state in results:
            # 使用新 API: finalize_decision
            agent.finalize_decision(result, market_state, acc_state)

        # 阶段 4.5: 收集 Smart Agent 动作 (用于 Tier 2 影响)
        smart_actions = []
        for agent in self.agents:
            if isinstance(agent, CivitasAgent) and agent.pending_action:
                act = agent.pending_action.get("action", "HOLD")
                if act == "BUY":
                    smart_actions.append(1)
                elif act == "SELL":
                    smart_actions.append(-1)
                else:
                    smart_actions.append(0)
            else:
                smart_actions.append(0)
        
        # 更新 Smart Sentiment 指标
        self.last_smart_sentiment = np.mean(smart_actions) if smart_actions else 0.0

        # 阶段 5: 执行交易 (Tier 1 Advance)
        self.agents.do("advance")
        
        # 阶段 5.5: Tier 2 (Vectorized) 决策与执行
        # A. 更新情绪
        trend_signal = 1.0 if self.trend == "上涨" else -1.0
        if self.trend == "下跌": trend_signal = -1.0
        elif self.trend == "震荡": trend_signal = 0.0
        
        self.population.update_tier2_sentiment(smart_actions, trend_signal)
        
        
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
                    price=float(v_prices[idx]),
                    quantity=int(v_qtys[idx]),
                    agent_id=f"Vec_{idx}",
                    side=side,
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
            # 检查买方
            if t.buy_agent_id.startswith("Vec_"):
                idx = int(t.buy_agent_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(1) # Buy
            # 检查卖方
            if t.sell_agent_id.startswith("Vec_"):
                idx = int(t.sell_agent_id.split("_")[1])
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
        
        # 同步给 MarketManager 以影响恐慌因子
        # MarketManager 内部也有 calculate_csad 逻辑，我们这里做轻量级计算即可
        # 也可以把 Agent returns 传给 Manager
        pass
        
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
