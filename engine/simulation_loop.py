"""
Civitas-Economica 核心仿真循环 (Simulation Loop)
"""

import asyncio
import logging
from typing import List

from agents.trading_agent_core import TradingAgent, GLOBAL_CONFIG
from policy.policy_engine import PolicyEngine
from engine.market_match import calculate_new_price

logger = logging.getLogger("civitas.engine.simulation")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(sh)


class MarketEnvironment:
    """
    全球市场宏观环境与仿真控制器。
    负责时间推进、政策广播、并发收集代理决策并计算聚合价格。
    """

    def __init__(self, agents: List[TradingAgent]):
        """
        初始化大环境。
        
        Args:
            agents (List[TradingAgent]): 仿真空间内的所有交易代理实体
        """
        self.agents = agents
        self.simulation_time = 0
        self.current_price = GLOBAL_CONFIG.get("initial_market_price", 100.0)
        self.policy_engine = PolicyEngine()
        
        logger.info(f"Initialized Market Environment with {len(agents)} agents. Initial Price: {self.current_price}")

    async def simulation_step(self) -> None:
        """
        执行单步核心仿真循环 (The Core Simulation Loop)。
        
        因果关系架构展现:
        政策发射 -> 代理接收并查阅记忆 -> 代理根据 Persona 过滤与 LLM 决策 -> 聚合的群体订单冲击市场价格
        """
        # 1. 推进全局仿真时间 (Increment the global simulation_time)
        self.simulation_time += 1
        logger.info(f"--- 开启第 {self.simulation_time} 步宏观仿真 ---")

        # 2. 宏观政策引擎发射事件 (Emit macro policy events)
        new_policy = self.policy_engine.emit_policy(self.simulation_time)
        
        if new_policy:
            logger.warning(f"[宏观政策广播] 新政策发布: {new_policy}")
            # 广播并让所有代理存入各自的 Ebbinghaus 认知库
            for agent in self.agents:
                # 给代理计算对该事件特有的话题记忆强度 S
                s_strength = agent.memory_bank.calculate_memory_strength(new_policy, agent.persona)
                agent.memory_bank.add_memory(
                    timestamp=self.simulation_time,
                    content=new_policy,
                    content_embedding=None,  # 依赖 ChromaDB 默认嵌入模型
                    memory_strength=s_strength
                )
                
        # 组装当前市场公开的截面数据
        market_data = {
            "tick": self.simulation_time,
            "current_price": self.current_price,
            "latest_broadcast": new_policy if new_policy else "市场平静"
        }

        # 3. 异步并发收集所有代理的决策
        async def process_agent(agent: TradingAgent):
            # (Crucial step): 在决策前，每个代理必须采用当前时间查询自己的艾宾浩斯记忆库以获取未被遗忘的政策。
            query_text = f"在价格 {self.current_price} 下我的应对策略及对近况的回顾。"
            
            retrieved_context = agent.memory_bank.retrieve_context(
                current_simulation_time=self.simulation_time,
                query_embedding=None,
                query_text=query_text,
                top_k=3
            )
            
            # 使用提取出的未遗忘政策上下文（retrieved_context），结合该智能体特有的人格 (Persona) 交给 LLM 决策
            # 这将被放入 Agent 的 LLM prompt 中 (见 agents/trading_agent_core.py)
            trade_action = await agent.generate_trading_decision(market_data, retrieved_context)
            return trade_action

        # asyncio.gather concurrently asks all agents for their TradeAction
        logger.info(f"并发收集 {len(self.agents)} 名代理的 LLM 交易决议...")
        agent_actions = await asyncio.gather(*(process_agent(agent) for agent in self.agents))
        
        # 4. 聚合所有订单结果 TradeActions
        buy_volume = 0.0
        sell_volume = 0.0
        
        for action in agent_actions:
            if action.action == "BUY":
                buy_volume += action.amount
            elif action.action == "SELL":
                sell_volume += action.amount
                
        # 5. 运行简化的撮合引擎/价格冲击模型 (Run the matching engine)
        old_price = self.current_price
        self.current_price = calculate_new_price(buy_volume, sell_volume, self.current_price)
        
        # 6. 输出并记录宏观事件状态 (Log the macro state)
        price_change = ((self.current_price - old_price) / old_price) * 100
        logger.info(f"[Macro State] Tick {self.simulation_time} 结算完成 | "
                    f"Buy Vol: {buy_volume:.2f} | Sell Vol: {sell_volume:.2f} | "
                    f"New Price: {self.current_price:.2f} ({price_change:+.2f}%)")
