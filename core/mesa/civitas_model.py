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
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        panic_ratio: float = 0.3,
        quant_ratio: float = 0.1,
        initial_price: float = 3000.0,
        seed: Optional[int] = None
    ):
        """
        初始化模型
        
        Args:
            n_agents: Agent 数量
            panic_ratio: 恐慌型散户比例
            quant_ratio: 量化投资者比例
            initial_price: 初始价格
            seed: 随机种子
        """
        super().__init__(seed=seed)
        
        self.n_agents = n_agents
        self.current_price = initial_price
        self.price_history: List[float] = [initial_price]
        
        # 市场状态
        self.trend = "震荡"
        self.panic_level = 0.3
        self.volatility = 0.02
        self.csad = 0.0
        
        # 创建异质性 Agent 群体
        self._create_agents(panic_ratio, quant_ratio)
        
        # 初始化数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Price": "current_price",
                "CSAD": "csad",
                "Panic_Level": "panic_level",
                "Avg_Wealth": lambda m: np.mean([a.wealth for a in m.agents]),
                "Avg_Sentiment": lambda m: np.mean([a.sentiment for a in m.agents]),
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
        return {
            "price": self.current_price,
            "trend": self.trend,
            "panic_level": self.panic_level,
            "volatility": self.volatility,
            "csad": self.csad,
            "news": "市场平稳运行"
        }
    
    def _update_market(self) -> None:
        """更新市场状态 (基于 Agent 行为)"""
        # 统计买卖意向
        buy_pressure = 0
        sell_pressure = 0
        
        for agent in self.agents:
            if agent.pending_action:
                if agent.pending_action["action"] == "BUY":
                    buy_pressure += agent.pending_action["quantity"]
                elif agent.pending_action["action"] == "SELL":
                    sell_pressure += agent.pending_action["quantity"]
        
        # 价格变动
        net_pressure = (buy_pressure - sell_pressure) / max(1, buy_pressure + sell_pressure)
        price_change = net_pressure * self.volatility * self.current_price
        self.current_price = max(1.0, self.current_price + price_change)
        self.price_history.append(self.current_price)
        
        # 更新趋势
        if len(self.price_history) >= 5:
            recent = self.price_history[-5:]
            if recent[-1] > recent[0] * 1.01:
                self.trend = "上涨"
            elif recent[-1] < recent[0] * 0.99:
                self.trend = "下跌"
            else:
                self.trend = "震荡"
        
        # 计算 CSAD (羊群效应指标)
        self._calculate_csad()
    
    def _calculate_csad(self) -> None:
        """计算截面绝对偏差 (CSAD)"""
        if len(self.price_history) < 2:
            self.csad = 0.0
            return
        
        # 使用 Agent 的情绪作为代理变量
        sentiments = [a.sentiment for a in self.agents]
        if not sentiments:
            self.csad = 0.0
            return
        
        avg_sentiment = np.mean(sentiments)
        self.csad = np.mean(np.abs(np.array(sentiments) - avg_sentiment))
        
        # 更新恐慌指数
        bearish_ratio = sum(1 for s in sentiments if s < -0.2) / len(sentiments)
        self.panic_level = bearish_ratio * 0.7 + self.panic_level * 0.3
    
    def step(self) -> None:
        """
        执行一个模拟步骤
        
        Mesa 3.0+ 的 SimultaneousActivation 模式：
        1. 所有 Agent 执行 step() (决策)
        2. 更新市场状态
        3. 所有 Agent 执行 advance() (执行)
        4. 收集数据
        """
        # 阶段 1: 所有 Agent 决策
        self.agents.do("step")
        
        # 阶段 2: 更新市场
        self._update_market()
        
        # 阶段 3: 所有 Agent 执行
        self.agents.do("advance")
        
        # 阶段 4: 收集数据
        self.datacollector.collect(self)
    
    def run_simulation(self, n_steps: int = 100) -> None:
        """运行完整模拟"""
        for _ in range(n_steps):
            self.step()
    
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
    
    # 创建模型
    model = CivitasModel(n_agents=50, seed=42)
    print(f"创建 {model.n_agents} 个 Agent")
    
    # 运行 20 步
    print("\n[运行模拟 - 20 步]")
    for i in range(20):
        model.step()
        if i % 5 == 0:
            print(f"  Step {model.steps}: Price={model.current_price:.2f}, "
                  f"CSAD={model.csad:.4f}, Panic={model.panic_level:.2f}")
    
    # 获取结果
    results = model.get_results()
    print(f"\n[结果]")
    print(f"  最终价格: {results['final_price']:.2f}")
    print(f"  Model DataFrame shape: {results['model_data'].shape}")
    print(f"  Agent DataFrame shape: {results['agent_data'].shape}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
