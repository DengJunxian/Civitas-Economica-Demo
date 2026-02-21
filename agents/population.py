# file: agents/population.py

import numpy as np
import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import random

from config import GLOBAL_CONFIG
from agents.brain import DeepSeekBrain
from agents.debate_brain import DebateBrain

# --- Tier 1: 智能体定义 ---

@dataclass
class SmartAgent:
    """
    Tier 1 意见领袖 (Opinion Leader)
    拥有独立的 DeepSeek 大脑、记忆和完整账户状态。
    """
    id: str
    brain: DeepSeekBrain
    cash: float
    holdings: int
    cost_basis: float
    
    # 影响力系数 (决定能覆盖多少 Tier 2 节点)
    influence_factor: float = 1.0 
    
    @property
    def market_value(self) -> float:
        # 注: 需外部注入当前价格计算，此处仅为占位
        return 0.0

# --- Tier 2: 向量化群体 ---

class StratifiedPopulation:
    """
    分层智能体群体管理器
    
    架构设计:
    - Tier 1: List[SmartAgent] -> 复杂逻辑，低并发
    - Tier 2: Numpy Matrix -> 简单逻辑，高并发 (SIMD)
    """
    
    # 状态矩阵列索引定义
    IDX_CASH = 0
    IDX_HOLDINGS = 1
    IDX_COST = 2
    IDX_SENTIMENT = 3  # -1.0 (极度看空) ~ 1.0 (极度看多)
    IDX_COGNITIVE_TYPE = 4  # 认知类型: 0=技术派, 1=消息派, 2=跟风派
    IDX_CONFIDENCE = 5  # 信心指数: 0-100
    
    def __init__(self, n_smart: int = 50, n_vectorized: int = 9950, api_key: str = None, smart_agents: List[Any] = None):
        self.n_smart = n_smart
        self.n_vectorized = n_vectorized
        self._api_key = api_key
        
        # 1. 初始化 Tier 1 (Smart Agents)
        if smart_agents is not None:
             self.smart_agents = smart_agents
             self.n_smart = len(smart_agents)
             print(f"[*] 使用外部传入的 {self.n_smart} 个 Smart Agents")
        else:
            self.smart_agents: List[SmartAgent] = []
            self._init_smart_agents()
        
        # 2. 初始化 Tier 2 (Vectorized Matrix)
        # Shape: (N, 6) -> [Cash, Holdings, Cost, Sentiment, CognitiveType, Confidence]
        self.state = np.zeros((n_vectorized, 6), dtype=np.float32)
        self._init_vectorized_state()
        
        # 3. 构建社会网络 (Influence Topology)
        # 使用 Watts-Strogatz 小世界网络模拟"圈子"效应
        # 实际上我们构建一个二部图的简化版：每个 Tier 2 节点关注 1-3 个 Tier 1 节点
        self.influence_map = self._build_influence_network()
        
        # 4. 构建邻居网络（用于涌现式羊群效应）
        self.neighbor_network = self._build_neighbor_network()
        
        # 5. 构建Smart Agent之间的社交网络（大V之间互相影响）
        self.smart_social_network = self._build_smart_social_network()
        
        # 6. 上一轮Smart Agent决策记录（用于影响传递）
        self.last_smart_actions: Dict[str, Dict] = {}
        
    def _init_smart_agents(self):
        """初始化意见领袖"""
        print(f"[*] 初始化 {self.n_smart} 位 DeepSeek 智能体...")
        for i in range(self.n_smart):
            # 随机生成一些差异化的人设
            persona = {
                "risk_preference": random.choice(["激进", "稳健", "保守"]),
                "loss_aversion": random.uniform(1.5, 3.0)
            }
            # 前5个Agent默认使用DebateBrain（启用辩论功能）
            if i < 5:
                brain = DebateBrain(agent_id=f"Debate_{i}", persona=persona, api_key=self._api_key)
                agent = SmartAgent(
                    id=f"Debate_{i}",
                    brain=brain,
                    cash=GLOBAL_CONFIG.DEFAULT_CASH * random.uniform(10, 30),  # 辩论大V资金更多
                    holdings=int(random.uniform(5000, 80000)),
                    cost_basis=3000.0
                )
            else:
                agent = SmartAgent(
                    id=f"Smart_{i}",
                    brain=DeepSeekBrain(agent_id=f"Smart_{i}", persona=persona, api_key=self._api_key),
                    cash=GLOBAL_CONFIG.DEFAULT_CASH * random.uniform(5, 20),
                    holdings=int(random.uniform(1000, 50000)),
                    cost_basis=3000.0
                )
            
            # 设置重要性等级（用于混合调度）
            # 前10%为核心(2)，10-30%为重要(1)，其余为普通(0)
            if i < self.n_smart * 0.1:
                agent.brain.importance_level = 2  # 核心Agent
            elif i < self.n_smart * 0.3:
                agent.brain.importance_level = 1  # 重要Agent
            else:
                agent.brain.importance_level = 0  # 普通Agent
            self.smart_agents.append(agent)

    def get_agent_by_id(self, agent_id: str) -> Optional[SmartAgent]:
        """根据 ID 获取 SmartAgent"""
        for agent in self.smart_agents:
            if agent.id == agent_id:
                return agent
        return None

    def _init_vectorized_state(self):
        """初始化散户矩阵"""
        # 资金: 对数正态分布 (贫富差距)
        # 修正: 提高初始现金使之与持仓价值匹配，避免结构性卖压
        self.state[:, self.IDX_CASH] = np.random.lognormal(11.5, 0.8, self.n_vectorized)
        
        # 持仓: 随机分布 (降低初始持仓，避免卖压过大)
        self.state[:, self.IDX_HOLDINGS] = np.random.randint(0, 1000, self.n_vectorized)
        
        # 成本: 围绕 3000 点波动
        self.state[:, self.IDX_COST] = np.random.normal(3000, 200, self.n_vectorized)
        
        # 情绪: 初始微正偏好 (正常市场散户通常略偏乐观)
        # Beta(3,2) 均值=0.6, 映射到 [-1,1] 后均值=0.2, 适度偏多
        self.state[:, self.IDX_SENTIMENT] = np.random.beta(3, 2, self.n_vectorized) * 2 - 1
        
        # 认知类型: 0=技术派(20%), 1=消息派(30%), 2=跟风派(50%)
        type_probs = np.random.random(self.n_vectorized)
        self.state[:, self.IDX_COGNITIVE_TYPE] = np.where(
            type_probs < 0.2, 0, np.where(type_probs < 0.5, 1, 2)
        )
        
        # 信心指数: 正态分布，均值55 (略偏信心充足)
        self.state[:, self.IDX_CONFIDENCE] = np.clip(
            np.random.normal(55, 15, self.n_vectorized), 0, 100
        )
    
    def _build_neighbor_network(self, n_neighbors: int = 5) -> np.ndarray:
        """
        构建邻居网络（用于涌现式羊群效应）
        
        每个散户与周围若干节点形成邻居关系，
        模拟社交圈子内的情绪传染。
        
        Returns:
            邻居索引矩阵 (N, n_neighbors)
        """
        neighbors = np.zeros((self.n_vectorized, n_neighbors), dtype=np.int32)
        for i in range(self.n_vectorized):
            # 随机选择邻居（可重复选择同一节点表示更紧密的联系）
            candidates = np.random.randint(0, self.n_vectorized, n_neighbors * 2)
            # 排除自己
            candidates = candidates[candidates != i][:n_neighbors]
            # 补齐不足的
            while len(candidates) < n_neighbors:
                new_neighbor = np.random.randint(0, self.n_vectorized)
                if new_neighbor != i:
                    candidates = np.append(candidates, new_neighbor)
            neighbors[i] = candidates[:n_neighbors]
        return neighbors

    def _build_influence_network(self) -> np.ndarray:
        """
        构建影响图谱
        Returns:
            adjacency matrix (N_vec, N_smart) 的稠密表示或索引列表
            这里简化为: 每个 Tier 2 只有一个主要关注的 Tier 1 (Guru)
        """
        # 帕累托分布：少数大V拥有绝大多数粉丝
        weights = np.random.pareto(a=2.0, size=self.n_smart)
        weights /= weights.sum()
        
        # 为每个散户分配一个"带头大哥"
        guru_indices = np.random.choice(
            self.n_smart, 
            size=self.n_vectorized, 
            p=weights
        )
        return guru_indices
    
    def _build_smart_social_network(self) -> Dict[str, List[str]]:
        """
        构建Smart Agent之间的社交网络（小世界网络）
        
        每个大V关注2-4个其他大V，形成信息传递环路。
        使用随机图模拟社交媒体上的互关关系。
        
        Returns:
            Dict[agent_id, List[关注的agent_id]]
        """
        network = {}
        agent_ids = [a.id for a in self.smart_agents]
        
        for i, agent_id in enumerate(agent_ids):
            # 每个Agent关注2-4个其他Agent
            n_follow = random.randint(2, min(4, self.n_smart - 1))
            others = [aid for aid in agent_ids if aid != agent_id]
            
            # 权重：倾向于关注编号相近的（模拟圈子效应）
            weights = [1.0 / (1 + abs(j - i)) for j in range(len(others))]
            weights = [w / sum(weights) for w in weights]
            
            follows = random.choices(others, weights=weights, k=n_follow)
            network[agent_id] = list(set(follows))
        
        print(f"[OK] Smart Agent社交网络构建完成，平均关注数: {sum(len(v) for v in network.values()) / len(network):.1f}")
        return network
    
    def get_social_influence_context(self, agent_id: str) -> Dict:
        """
        获取某个Smart Agent的社交影响上下文
        
        返回该Agent所关注的其他Agent的最近决策信息，
        用于在prompt中注入社交影响因素。
        
        Args:
            agent_id: 目标Agent ID
            
        Returns:
            Dict: 包含社交影响信息的上下文
        """
        followed = self.smart_social_network.get(agent_id, [])
        
        influences = []
        for fid in followed:
            if fid in self.last_smart_actions:
                action_info = self.last_smart_actions[fid]
                influences.append({
                    "agent_id": fid,
                    "action": action_info.get("action", "HOLD"),
                    "confidence": action_info.get("confidence", 0.5),
                    "emotion": action_info.get("emotion_score", 0.0)
                })
        
        # 计算社交圈整体情绪
        if influences:
            avg_emotion = sum(i["emotion"] for i in influences) / len(influences)
            bullish_ratio = sum(1 for i in influences if i["action"] == "BUY") / len(influences)
            bearish_ratio = sum(1 for i in influences if i["action"] == "SELL") / len(influences)
        else:
            avg_emotion = 0.0
            bullish_ratio = 0.0
            bearish_ratio = 0.0
        
        return {
            "followed_agents": influences,
            "circle_emotion": avg_emotion,
            "circle_bullish_ratio": bullish_ratio,
            "circle_bearish_ratio": bearish_ratio,
            "influence_strength": len(influences) / max(len(followed), 1)
        }
    
    def record_smart_action(self, agent_id: str, decision: Dict):
        """
        记录Smart Agent的决策，用于下一轮的社交影响
        """
        self.last_smart_actions[agent_id] = {
            "action": decision.get("action", "HOLD"),
            "confidence": decision.get("confidence", 0.5),
            "emotion_score": decision.get("emotion_score", 0.0),
            "reasoning_summary": decision.get("reasoning", "")[:100]
        }

    def calculate_csad(self) -> float:
        """
        计算市场情绪的一致性 (Cross-Sectional Absolute Deviation)
        CSAD 越低，说明散户情绪越趋同，越容易发生羊群效应。
        """
        sentiments = self.state[:, self.IDX_SENTIMENT]
        mean_sentiment = np.mean(sentiments)
        # 绝对偏差的平均值
        csad = np.mean(np.abs(sentiments - mean_sentiment))
        return csad

    def update_tier2_sentiment(self, smart_actions: List[int], market_trend: float):
        """
        [向量化] 涌现式情绪传染更新
        
        采用基于邻居网络的涌现机制，替代硬编码阈值：
        - 每个散户观察其邻居的情绪状态
        - 当邻居恐慌比例超过阈值时，被"感染"
        - 不同认知类型对不同信号的响应权重不同
        
        Parameters:
            smart_actions: Tier 1 的操作列表 (1=Buy, -1=Sell, 0=Hold)
            market_trend: 市场趋势信号 (-1.0 ~ 1.0)
        """
        # 1. 获取来自大V的信号 (Local Signal)
        smart_acts = np.array(smart_actions)  # shape (N_smart,)
        guru_signals = smart_acts[self.influence_map]
        
        # 2. 计算邻居情绪状态（涌现式羊群效应核心）
        current_sentiment = self.state[:, self.IDX_SENTIMENT]
        neighbor_indices = self.neighbor_network  # (N, n_neighbors)
        
        # 获取每个节点的邻居情绪
        neighbor_sentiments = current_sentiment[neighbor_indices]  # (N, n_neighbors)
        
        # 计算邻居恐慌比例（情绪 < -0.3 视为恐慌）
        neighbor_panic_ratio = np.mean(neighbor_sentiments < -0.3, axis=1)  # (N,)
        
        # 计算邻居平均情绪（用于信号传递）
        neighbor_avg_sentiment = np.mean(neighbor_sentiments, axis=1)
        
        # 3. 基于认知类型的权重分配
        cognitive_types = self.state[:, self.IDX_COGNITIVE_TYPE]
        confidence = self.state[:, self.IDX_CONFIDENCE] / 100.0  # 归一化到 0-1
        
        # 基础权重
        w_personal = np.full(self.n_vectorized, 0.4)
        w_guru = np.full(self.n_vectorized, 0.2)
        w_neighbor = np.full(self.n_vectorized, 0.2)
        w_market = np.full(self.n_vectorized, 0.2)
        
        # 技术派 (type=0): 更依赖市场趋势，较少受邻居影响
        tech_mask = cognitive_types == 0
        w_personal[tech_mask] = 0.3
        w_market[tech_mask] = 0.4
        w_neighbor[tech_mask] = 0.1
        w_guru[tech_mask] = 0.2
        
        # 消息派 (type=1): 更依赖大V信号
        news_mask = cognitive_types == 1
        w_personal[news_mask] = 0.2
        w_guru[news_mask] = 0.4
        w_neighbor[news_mask] = 0.2
        w_market[news_mask] = 0.2
        
        # 跟风派 (type=2): 更容易受邻居影响
        herd_mask = cognitive_types == 2
        w_personal[herd_mask] = 0.1
        w_neighbor[herd_mask] = 0.5
        w_guru[herd_mask] = 0.2
        w_market[herd_mask] = 0.2
        
        # 4. 涌现式羊群效应：邻居恐慌时，权重动态调整
        # 当超过60%的邻居恐慌时，个人判断被大幅削弱
        panic_threshold = 0.6
        panic_mask = neighbor_panic_ratio > panic_threshold
        
        # 恐慌传染系数（信心低的更容易被传染）
        susceptibility = (1.0 - confidence) * 0.8 + 0.2  # 0.2 ~ 1.0
        
        # 调整权重
        w_personal[panic_mask] *= (1.0 - susceptibility[panic_mask] * 0.7)
        w_neighbor[panic_mask] += susceptibility[panic_mask] * 0.4
        
        # 归一化权重
        total_w = w_personal + w_guru + w_neighbor + w_market
        w_personal /= total_w
        w_guru /= total_w
        w_neighbor /= total_w
        w_market /= total_w
        
        # 5. 批量更新情绪 (Matrix Operation)
        noise = np.random.normal(0, 0.05, self.n_vectorized)
        
        new_sentiment = (
            w_personal * current_sentiment +
            w_guru * guru_signals +
            w_neighbor * neighbor_avg_sentiment +
            w_market * market_trend +
            noise
        )
        
        # 截断到 [-1, 1]
        self.state[:, self.IDX_SENTIMENT] = np.clip(new_sentiment, -1.0, 1.0)
        
        # 6. 更新信心指数（经历恐慌后信心下降）
        confidence_change = np.where(
            panic_mask,
            -5 * susceptibility,  # 恐慌中信心快速下降
            0.5  # 正常情况缓慢恢复
        )
        self.state[:, self.IDX_CONFIDENCE] = np.clip(
            self.state[:, self.IDX_CONFIDENCE] + confidence_change, 0, 100
        )

    def generate_tier2_decisions(self, current_price: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [向量化] 生成散户交易决策
        避免 10,000 次 if-else，直接用概率矩阵生成 mask。
        
        Returns:
            actions: (N,) {-1, 0, 1}
            quantities: (N,)
            prices: (N,) 挂单价格
        """
        sentiment = self.state[:, self.IDX_SENTIMENT]
        
        # 1. 生成买卖概率
        # 情绪 > 0.6 -> 高概率买入; 情绪 < -0.6 -> 高概率卖出
        prob_buy = 1 / (1 + np.exp(-5 * (sentiment - 0.3))) # Sigmoid shift
        prob_sell = 1 / (1 + np.exp(5 * (sentiment + 0.3)))
        
        # 随机骰子
        rng = np.random.random(self.n_vectorized)
        
        # 生成动作 Mask
        buy_mask = rng < prob_buy
        sell_mask = (rng > (1 - prob_sell)) & (~buy_mask) # 互斥
        
        actions = np.zeros(self.n_vectorized, dtype=int)
        actions[buy_mask] = 1
        actions[sell_mask] = -1
        
        # 2. 计算数量 (简单的资金比例法)
        quantities = np.zeros(self.n_vectorized, dtype=int)
        
        # 买入: 使用 20%~50% 可用资金
        buy_ratio = np.random.uniform(0.2, 0.5, size=self.n_vectorized)
        avail_cash = self.state[:, self.IDX_CASH]
        # 向量化计算: 资金 * 比例 / 单价 (向下取整)
        raw_buy_qty = (avail_cash * buy_ratio / current_price).astype(int)
        quantities[buy_mask] = raw_buy_qty[buy_mask]
        
        # 卖出: 使用 50%~100% 持仓
        sell_ratio = np.random.uniform(0.5, 1.0, size=self.n_vectorized)
        avail_holdings = self.state[:, self.IDX_HOLDINGS]
        raw_sell_qty = (avail_holdings * sell_ratio).astype(int)
        quantities[sell_mask] = raw_sell_qty[sell_mask]
        
        # 3. 过滤无效单 (数量为0)
        valid_mask = quantities > 0
        actions[~valid_mask] = 0
        quantities[~valid_mask] = 0
        
        # 4. 生成挂单价格 (在现价附近波动)
        # 散户通常挂市价或略好的价格
        price_noise = np.random.normal(0, 0.002, self.n_vectorized)
        order_prices = current_price * (1 + price_noise)
        
        return actions, quantities, order_prices

    def sync_tier2_execution(self, executed_indices: np.ndarray, executed_prices: np.ndarray, 
                             executed_qtys: np.ndarray, directions: np.ndarray):
        """
        [向量化] 成交回执处理
        当撮合引擎成交后，批量更新状态矩阵。
        """
        if len(executed_indices) == 0:
            return
            
        # 提取相关行
        subset_cash = self.state[executed_indices, self.IDX_CASH]
        subset_holdings = self.state[executed_indices, self.IDX_HOLDINGS]
        subset_cost = self.state[executed_indices, self.IDX_COST]
        
        cost_val = executed_prices * executed_qtys
        
        # 更新资金 (买入减，卖出加)
        delta_cash = -1 * directions * cost_val
        # FIX: 使用 add.at 处理重复索引 (同一Agent多笔成交)
        np.add.at(self.state[:, self.IDX_CASH], executed_indices, delta_cash)
        
        # 更新持仓
        delta_stock = directions * executed_qtys
        np.add.at(self.state[:, self.IDX_HOLDINGS], executed_indices, delta_stock)
        
        # 更新成本价 (仅买入时更新加权平均)
        buy_indices_local = (directions == 1)
        if np.any(buy_indices_local):
            # 获取全局索引
            g_idx = executed_indices[buy_indices_local]
            b_qty = executed_qtys[buy_indices_local]
            b_prc = executed_prices[buy_indices_local]
            
            # 原始持仓和成本
            old_qty = subset_holdings[buy_indices_local] # 注意: 这是更新前的数量吗? 不，上面已经 += delta了
            # 修正: 应该用更新前的数量。由于上面已经加了，这里要减回去算旧的
            cur_qty = self.state[g_idx, self.IDX_HOLDINGS]
            prev_qty = cur_qty - b_qty
            prev_cost = self.state[g_idx, self.IDX_COST]
            
            # 加权平均公式
            # (OldQty * OldCost + BuyQty * BuyPrice) / NewQty
            # 避免除以零
            denom = np.where(cur_qty > 0, cur_qty, 1.0)
            new_cost_basis = ((prev_qty * prev_cost) + (b_qty * b_prc)) / denom
            
            self.state[g_idx, self.IDX_COST] = new_cost_basis