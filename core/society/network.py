# file: core/society/network.py
"""
社会网络拓扑与情绪传播模块

使用 NetworkX 构建小世界网络 (Watts-Strogatz)，
并实现 SIR 模型的情绪传播动态。

核心功能:
1. SocialGraph: 小世界网络拓扑
2. InformationDiffusion: SIR 情绪传播引擎
3. 动态 λ 调整: 基于邻居情绪的羊群效应

参考:
- Watts-Strogatz (1998): 小世界网络
- Shiller (2000): Irrational Exuberance (羊群行为)
- SIR Model: 传染病学传播模型

作者: Civitas Economica Team
"""

import random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


# ==========================================
# 数据结构
# ==========================================

class SentimentState(str, Enum):
    """
    情绪状态 (SIR 模型)
    
    S - Susceptible: 易感/中性，可能被感染看空
    I - Infected: 感染/看空，会传播悲观情绪
    R - Recovered: 恢复/免疫，暂时不受影响
    B - Bullish: 看多 (扩展状态)
    """
    SUSCEPTIBLE = "S"  # 中性
    INFECTED = "I"     # 看空
    RECOVERED = "R"    # 免疫
    BULLISH = "B"      # 看多


@dataclass
class AgentNode:
    """
    社会网络中的 Agent 节点
    
    存储 Agent 的社会属性和动态状态。
    """
    node_id: int
    agent_id: str = ""
    
    # 情绪状态
    sentiment_state: SentimentState = SentimentState.SUSCEPTIBLE
    
    # 前景理论参数 (可动态调整)
    base_lambda: float = 2.25     # 基础损失厌恶系数
    lambda_coeff: float = 2.25    # 当前损失厌恶系数
    
    # 社会属性
    conformity: float = 0.5       # 从众性 [0, 1]，越高越容易被影响
    influence: float = 0.5        # 影响力 [0, 1]，越高越能影响他人
    
    # 状态持续时间
    state_duration: int = 0       # 当前状态持续的 tick 数
    
    # 历史
    sentiment_history: List[SentimentState] = field(default_factory=list)


# ==========================================
# 社会网络图
# ==========================================

class SocialGraph:
    """
    小世界社会网络
    
    使用 Watts-Strogatz 模型构建 Agent 之间的社交连接。
    
    特性:
    - 高聚类系数 (朋友的朋友也是朋友)
    - 短平均路径长度 (六度分隔)
    
    Attributes:
        graph: NetworkX 图对象
        agents: Agent 节点字典
    """
    
    def __init__(
        self,
        n_agents: int = 1000,
        k: int = 6,
        p: float = 0.3,
        seed: int = None
    ):
        """
        创建小世界网络
        
        Args:
            n_agents: Agent 数量
            k: 每个节点的初始邻居数 (必须为偶数)
            p: 重连概率 (p=0 为规则网络, p=1 为随机网络)
            seed: 随机种子
        """
        self.n_agents = n_agents
        self.k = k
        self.p = p
        
        # 创建 Watts-Strogatz 小世界网络
        self.graph = nx.watts_strogatz_graph(n_agents, k, p, seed=seed)
        
        # 初始化 Agent 节点
        self.agents: Dict[int, AgentNode] = {}
        self._init_agents()
        
        # 统计信息
        self.tick = 0
    
    def _init_agents(self) -> None:
        """初始化 Agent 节点，设置异质性属性"""
        for node_id in self.graph.nodes():
            # 从众性：正态分布，大部分人中等从众
            conformity = np.clip(np.random.normal(0.5, 0.15), 0.1, 0.9)
            
            # 影响力：幂律分布，少数人有高影响力
            influence = min(0.9, np.random.pareto(2) * 0.2)
            
            # 损失厌恶系数：正态分布
            base_lambda = np.clip(np.random.normal(2.25, 0.5), 1.0, 4.0)
            
            self.agents[node_id] = AgentNode(
                node_id=node_id,
                agent_id=f"agent_{node_id}",
                conformity=conformity,
                influence=influence,
                base_lambda=base_lambda,
                lambda_coeff=base_lambda
            )
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """获取邻居节点 ID"""
        return list(self.graph.neighbors(node_id))
    
    def get_neighbor_sentiments(self, node_id: int) -> Dict[SentimentState, int]:
        """统计邻居的情绪分布"""
        neighbors = self.get_neighbors(node_id)
        counts = {state: 0 for state in SentimentState}
        
        for n in neighbors:
            state = self.agents[n].sentiment_state
            counts[state] += 1
        
        return counts
    
    def get_bearish_ratio(self, node_id: int) -> float:
        """计算邻居中看空者的比例"""
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        
        bearish_count = sum(
            1 for n in neighbors
            if self.agents[n].sentiment_state == SentimentState.INFECTED
        )
        return bearish_count / len(neighbors)
    
    def get_bullish_ratio(self, node_id: int) -> float:
        """计算邻居中看多者的比例"""
        neighbors = self.get_neighbors(node_id)
        if not neighbors:
            return 0.0
        
        bullish_count = sum(
            1 for n in neighbors
            if self.agents[n].sentiment_state == SentimentState.BULLISH
        )
        return bullish_count / len(neighbors)
    
    def generate_social_summary(self, node_id: int) -> str:
        """
        生成朋友圈舆情摘要
        
        用于注入 LLM Prompt，模拟社交媒体对投资决策的影响。
        
        Returns:
            描述朋友圈情绪的中文字符串
        """
        sentiments = self.get_neighbor_sentiments(node_id)
        neighbors = self.get_neighbors(node_id)
        total = len(neighbors) if neighbors else 1
        
        bearish = sentiments.get(SentimentState.INFECTED, 0)
        bullish = sentiments.get(SentimentState.BULLISH, 0)
        neutral = sentiments.get(SentimentState.SUSCEPTIBLE, 0)
        
        parts = []
        if bearish > total * 0.5:
            parts.append(f"你的{bearish}个朋友中超过半数在看空！群里弥漫着悲观情绪。")
        elif bearish > total * 0.3:
            parts.append(f"朋友圈开始出现恐慌情绪，{bearish}人发帖说要割肉。")
        
        if bullish > total * 0.5:
            parts.append(f"你的{bullish}个朋友都在喊加仓！群里非常亢奋。")
        elif bullish > total * 0.3:
            parts.append(f"有{bullish}个朋友分享了盈利截图，气氛偏乐观。")
        
        if not parts:
            parts.append(f"朋友圈比较平静，{neutral}人在观望中。")
        
        return " ".join(parts)
    
    def get_most_influential_node(self) -> Optional[int]:
        """
        获取影响力最高的节点 ID (网络枢纽)
        
        Returns:
            影响力最高节点的 ID，若网络为空则返回 None
        """
        if not self.agents:
            return None
        return max(self.agents.items(), key=lambda x: x[1].influence)[0]

    def get_network_stats(self) -> Dict:
        """获取网络统计信息"""
        states = [a.sentiment_state for a in self.agents.values()]
        
        return {
            "total_nodes": self.n_agents,
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.n_agents,
            "clustering_coefficient": nx.average_clustering(self.graph),
            "sentiment_distribution": {
                "susceptible": sum(1 for s in states if s == SentimentState.SUSCEPTIBLE),
                "infected": sum(1 for s in states if s == SentimentState.INFECTED),
                "recovered": sum(1 for s in states if s == SentimentState.RECOVERED),
                "bullish": sum(1 for s in states if s == SentimentState.BULLISH)
            },
            "avg_lambda": np.mean([a.lambda_coeff for a in self.agents.values()])
        }


# ==========================================
# SIR 情绪传播引擎
# ==========================================

class InformationDiffusion:
    """
    信息/情绪传播引擎 (SIR 模型)
    
    模拟看空情绪在社交网络中的传播：
    
    传播机制:
    1. S → I: 中性 Agent 被看空邻居"感染"
       概率 = β * (看空邻居比例) * (从众性)
       
    2. I → R: 看空 Agent "恢复"中性
       概率 = γ
       
    3. R → S: 免疫 Agent 再次变得易感
       概率 = δ (损失免疫力)
    
    羊群效应:
    - 当邻居看空比例高时，动态增加 λ (损失厌恶)
    - 模拟恐慌情绪的自我强化
    """
    
    def __init__(
        self,
        social_graph: SocialGraph,
        beta: float = 0.3,      # 感染概率
        gamma: float = 0.1,     # 恢复概率
        delta: float = 0.05,    # 免疫衰减概率
        lambda_amplifier: float = 0.5  # λ 放大系数
    ):
        """
        初始化传播引擎
        
        Args:
            social_graph: 社会网络
            beta: 基础感染概率
            gamma: 恢复概率
            delta: 免疫衰减概率
            lambda_amplifier: λ 放大系数 (羊群效应强度)
        """
        self.graph = social_graph
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_amplifier = lambda_amplifier
        
        # 历史记录
        self.history: List[Dict] = []
    
    def update_sentiment_propagation(self) -> Dict:
        """
        执行一轮情绪传播更新
        
        Returns:
            本轮传播统计
        """
        self.graph.tick += 1
        
        # 统计变化
        new_infected = 0
        new_recovered = 0
        new_susceptible = 0
        
        # 收集状态变化 (避免在迭代中修改)
        state_changes: List[Tuple[int, SentimentState]] = []
        
        for node_id, agent in self.graph.agents.items():
            new_state = self._update_single_agent(agent)
            
            if new_state != agent.sentiment_state:
                state_changes.append((node_id, new_state))
                
                if new_state == SentimentState.INFECTED:
                    new_infected += 1
                elif new_state == SentimentState.RECOVERED:
                    new_recovered += 1
                elif new_state == SentimentState.SUSCEPTIBLE:
                    new_susceptible += 1
        
        # 应用状态变化
        for node_id, new_state in state_changes:
            agent = self.graph.agents[node_id]
            agent.sentiment_history.append(agent.sentiment_state)
            agent.sentiment_state = new_state
            agent.state_duration = 0
        
        # 更新持续时间
        for agent in self.graph.agents.values():
            agent.state_duration += 1
        
        # 更新 Lambda (羊群效应)
        self._update_lambda_by_herding()
        
        # 记录历史
        stats = {
            "tick": self.graph.tick,
            "new_infected": new_infected,
            "new_recovered": new_recovered,
            "new_susceptible": new_susceptible,
            **self.graph.get_network_stats()["sentiment_distribution"]
        }
        self.history.append(stats)
        
        return stats
    
    def _update_single_agent(self, agent: AgentNode) -> SentimentState:
        """
        更新单个 Agent 的情绪状态
        
        基于 SIR 模型的状态转换规则。
        """
        current_state = agent.sentiment_state
        
        if current_state == SentimentState.SUSCEPTIBLE:
            # S → I: 可能被感染
            bearish_ratio = self.graph.get_bearish_ratio(agent.node_id)
            
            # 感染概率 = β * 看空比例 * 从众性
            infection_prob = self.beta * bearish_ratio * agent.conformity
            
            if random.random() < infection_prob:
                return SentimentState.INFECTED
        
        elif current_state == SentimentState.INFECTED:
            # I → R: 可能恢复
            # 考虑状态持续时间 (恐慌持续越久越容易恢复)
            recovery_prob = self.gamma * (1 + agent.state_duration * 0.1)
            
            if random.random() < recovery_prob:
                return SentimentState.RECOVERED
        
        elif current_state == SentimentState.RECOVERED:
            # R → S: 可能再次易感
            if random.random() < self.delta:
                return SentimentState.SUSCEPTIBLE
        
        return current_state
    
    def _update_lambda_by_herding(self) -> None:
        """
        基于羊群效应动态调整 λ (损失厌恶系数)
        
        规则:
        - 邻居看空比例越高 → λ 越大 → 越恐惧
        - λ 范围: [base_lambda * 0.8, base_lambda * 2.0]
        """
        for node_id, agent in self.graph.agents.items():
            bearish_ratio = self.graph.get_bearish_ratio(node_id)
            
            # 羊群效应放大
            # λ = base_λ * (1 + amplifier * bearish_ratio)
            herding_factor = 1 + self.lambda_amplifier * bearish_ratio
            
            # 限制范围
            new_lambda = agent.base_lambda * herding_factor
            new_lambda = np.clip(new_lambda, agent.base_lambda * 0.8, agent.base_lambda * 2.0)
            
            agent.lambda_coeff = new_lambda
    
    def inject_panic(
        self,
        n_seeds: int = 10,
        method: str = "random"
    ) -> List[int]:
        """
        注入恐慌种子
        
        模拟外部冲击 (如政策变化、黑天鹅事件) 导致部分 Agent 变为看空。
        
        Args:
            n_seeds: 种子数量
            method: 选择方法
                - "random": 随机选择
                - "influential": 选择高影响力节点
                - "clustered": 选择同一社区的节点
                
        Returns:
            被感染的节点 ID 列表
        """
        if method == "random":
            seeds = random.sample(list(self.graph.agents.keys()), min(n_seeds, len(self.graph.agents)))
        
        elif method == "influential":
            # 按影响力排序，选择最高的
            sorted_agents = sorted(
                self.graph.agents.items(),
                key=lambda x: x[1].influence,
                reverse=True
            )
            seeds = [a[0] for a in sorted_agents[:n_seeds]]
        
        elif method == "clustered":
            # 选择一个随机节点及其邻居
            start_node = random.choice(list(self.graph.agents.keys()))
            neighbors = self.graph.get_neighbors(start_node)
            seeds = [start_node] + neighbors[:n_seeds-1]
        
        else:
            seeds = random.sample(list(self.graph.agents.keys()), min(n_seeds, len(self.graph.agents)))
        
        # 设置为感染状态
        for node_id in seeds:
            self.graph.agents[node_id].sentiment_state = SentimentState.INFECTED
            self.graph.agents[node_id].state_duration = 0
        
        return seeds
    
    def get_propagation_stats(self) -> Dict:
        """获取传播统计"""
        if not self.history:
            return {}
        
        infected_counts = [h["infected"] for h in self.history]
        
        return {
            "total_ticks": len(self.history),
            "peak_infected": max(infected_counts) if infected_counts else 0,
            "peak_tick": infected_counts.index(max(infected_counts)) if infected_counts else 0,
            "current_stats": self.history[-1] if self.history else {}
        }
    
    def simulate(
        self,
        n_ticks: int = 100,
        initial_infected: int = 10
    ) -> List[Dict]:
        """
        运行完整传播模拟
        
        Args:
            n_ticks: 模拟轮数
            initial_infected: 初始感染数
            
        Returns:
            每轮的统计列表
        """
        # 注入初始恐慌
        self.inject_panic(initial_infected, method="influential")
        
        # 运行模拟
        for _ in range(n_ticks):
            self.update_sentiment_propagation()
        
        return self.history



