# file: core/counterfactual.py
"""
反事实推理模块 (Counterfactual Reasoning)

支持"平行宇宙"实验：在同一历史时刻分叉两个仿真环境，
比较不同政策下的市场表现差异。
"""

import copy
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UniverseSnapshot:
    """宇宙快照：保存仿真状态"""
    timestamp: datetime
    candles: List[Any]
    agent_states: Dict[str, Any]
    population_state: np.ndarray
    matcher_state: Dict[str, Any]
    policy: Optional[str] = None
    
    
@dataclass
class CounterfactualResult:
    """反事实实验结果"""
    universe_a_name: str
    universe_b_name: str
    fork_point: int  # 分叉点（K线索引）
    
    # 对比指标
    final_price_a: float
    final_price_b: float
    price_divergence: float  # 价格分歧度
    
    volatility_a: float
    volatility_b: float
    
    cumulative_return_a: float
    cumulative_return_b: float
    
    # 推理分析
    analysis: str = ""
    

class ParallelUniverseEngine:
    """
    平行宇宙仿真引擎
    
    核心功能：
    1. 在指定时刻创建仿真状态快照
    2. 分叉为两个独立的仿真环境
    3. 分别注入不同政策，运行仿真
    4. 生成差异分析报告
    """
    
    def __init__(self):
        self.snapshots: Dict[str, UniverseSnapshot] = {}
        self.universe_a_history: List[float] = []
        self.universe_b_history: List[float] = []
    
    def create_snapshot(self, name: str, controller) -> UniverseSnapshot:
        """
        创建仿真状态快照
        
        Args:
            name: 快照名称
            controller: SimulationController 实例
            
        Returns:
            快照对象
        """
        snapshot = UniverseSnapshot(
            timestamp=datetime.now(),
            candles=[copy.deepcopy(c) for c in controller.market.candles],
            agent_states={
                agent.id: {
                    "cash": controller.smart_portfolios[agent.id].available_cash,
                    "confidence": agent.brain.state.confidence,
                    "consecutive_losses": agent.brain.state.consecutive_losses
                }
                for agent in controller.population.smart_agents
            },
            population_state=controller.population.state.copy(),
            matcher_state={
                "last_price": controller.matcher.last_price,
                "prev_close": controller.matcher.prev_close,
                "total_volume": controller.matcher.total_volume
            },
            policy=getattr(controller.market, 'current_news', None)
        )
        
        self.snapshots[name] = snapshot
        return snapshot
    
    def fork_universe(self, snapshot_name: str) -> Tuple[Any, Any]:
        """
        从快照分叉两个宇宙
        
        注意：这是一个概念性接口。在实际实现中，
        需要完整重建 SimulationController 的状态。
        
        Args:
            snapshot_name: 快照名称
            
        Returns:
            (universe_a_state, universe_b_state) 两个独立的状态副本
        """
        if snapshot_name not in self.snapshots:
            raise ValueError(f"快照 '{snapshot_name}' 不存在")
        
        snapshot = self.snapshots[snapshot_name]
        
        # 创建两个独立的状态副本
        universe_a = copy.deepcopy(snapshot)
        universe_b = copy.deepcopy(snapshot)
        
        return universe_a, universe_b
    
    def run_counterfactual_experiment(
        self,
        base_prices: List[float],
        policy_a: Optional[str],
        policy_b: str,
        n_steps: int = 30,
        base_volatility: float = 0.02
    ) -> CounterfactualResult:
        """
        运行反事实实验（简化版）
        
        模拟两种政策下的价格演化差异。
        
        Args:
            base_prices: 基准价格序列
            policy_a: 原始政策（可为None表示无政策）
            policy_b: 反事实政策
            n_steps: 仿真步数
            base_volatility: 基础波动率
            
        Returns:
            实验结果
        """
        if not base_prices:
            raise ValueError("基准价格序列不能为空")
        
        fork_price = base_prices[-1]
        
        # 宇宙A：维持原状（较低波动）
        np.random.seed(42)
        universe_a_prices = [fork_price]
        for _ in range(n_steps):
            shock = np.random.normal(0, base_volatility)
            new_price = universe_a_prices[-1] * (1 + shock)
            universe_a_prices.append(new_price)
        
        # 宇宙B：注入政策（引入方向性偏移）
        np.random.seed(42)  # 相同随机种子确保可比性
        universe_b_prices = [fork_price]
        
        # 模拟政策效应（根据政策关键词判断方向）
        policy_bias = 0.001  # 默认轻微看多
        if policy_b:
            policy_lower = policy_b.lower()
            if any(word in policy_lower for word in ['降', '减', '下调', '放松']):
                policy_bias = 0.003  # 利好政策
            elif any(word in policy_lower for word in ['涨', '加', '上调', '收紧']):
                policy_bias = -0.002  # 利空政策
        
        for i in range(n_steps):
            shock = np.random.normal(policy_bias, base_volatility * 1.2)
            # 政策效应随时间衰减
            decay = 1 / (1 + i * 0.1)
            effective_shock = shock * decay + np.random.normal(0, base_volatility) * (1 - decay)
            new_price = universe_b_prices[-1] * (1 + effective_shock)
            universe_b_prices.append(new_price)
        
        self.universe_a_history = universe_a_prices
        self.universe_b_history = universe_b_prices
        
        # 计算指标
        def calc_return(prices: List[float]) -> float:
            return (prices[-1] - prices[0]) / prices[0]
        
        def calc_volatility(prices: List[float]) -> float:
            returns = np.diff(prices) / np.array(prices[:-1])
            return np.std(returns) * np.sqrt(252)  # 年化
        
        result = CounterfactualResult(
            universe_a_name=policy_a or "维持原状",
            universe_b_name=policy_b,
            fork_point=len(base_prices) - 1,
            final_price_a=universe_a_prices[-1],
            final_price_b=universe_b_prices[-1],
            price_divergence=abs(universe_a_prices[-1] - universe_b_prices[-1]) / fork_price,
            volatility_a=calc_volatility(universe_a_prices),
            volatility_b=calc_volatility(universe_b_prices),
            cumulative_return_a=calc_return(universe_a_prices),
            cumulative_return_b=calc_return(universe_b_prices)
        )
        
        # 生成分析
        result.analysis = self._generate_analysis(result, policy_b)
        
        return result
    
    def _generate_analysis(self, result: CounterfactualResult, policy: str) -> str:
        """生成差异分析报告"""
        lines = [
            "## 反事实推理分析报告",
            "",
            f"**分叉点**: 第 {result.fork_point} 个交易日",
            "",
            "### 价格对比",
            f"| 指标 | {result.universe_a_name} | {result.universe_b_name} |",
            "|------|------------------------|------------------------|",
            f"| 最终价格 | {result.final_price_a:.2f} | {result.final_price_b:.2f} |",
            f"| 累计收益 | {result.cumulative_return_a:+.2%} | {result.cumulative_return_b:+.2%} |",
            f"| 年化波动率 | {result.volatility_a:.2%} | {result.volatility_b:.2%} |",
            "",
            f"**价格分歧度**: {result.price_divergence:.2%}",
            "",
            "### 政策效应解读",
        ]
        
        if result.cumulative_return_b > result.cumulative_return_a:
            lines.append(f"注入政策 **「{policy}」** 后，模拟市场表现优于原状，")
            lines.append(f"累计收益提升 {(result.cumulative_return_b - result.cumulative_return_a):.2%}。")
        else:
            lines.append(f"注入政策 **「{policy}」** 后，模拟市场表现不及原状，")
            lines.append(f"累计收益下降 {(result.cumulative_return_a - result.cumulative_return_b):.2%}。")
        
        if result.volatility_b > result.volatility_a * 1.1:
            lines.append("")
            lines.append("⚠️ **波动率警告**: 政策导致市场波动显著加剧。")
        
        return "\n".join(lines)
    
    def get_comparison_data(self) -> Dict[str, List[float]]:
        """获取用于绘图的对比数据"""
        return {
            "universe_a": self.universe_a_history,
            "universe_b": self.universe_b_history
        }


# 使用示例
if __name__ == "__main__":
    engine = ParallelUniverseEngine()
    
    # 模拟历史价格
    base_prices = [3000 + i * 5 for i in range(30)]
    
    result = engine.run_counterfactual_experiment(
        base_prices=base_prices,
        policy_a=None,
        policy_b="印花税减半至0.025%",
        n_steps=20
    )
    
    print(result.analysis)
