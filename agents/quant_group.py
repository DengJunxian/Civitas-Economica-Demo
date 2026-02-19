# file: agents/quant_group.py
"""
量化策略 Agent 群体模块

实现持有相同 DeepSeek Prompt 的 Agent 群体，
用于研究量化交易对市场稳定性的影响。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
import random

from config import GLOBAL_CONFIG
from agents.brain import DeepSeekBrain, ThoughtRecord


@dataclass
class QuantStrategyGroup:
    """
    量化策略群体
    
    持有相同 DeepSeek Prompt（同一量化策略）的 Agent 群体，
    用于模拟量化交易对市场稳定性的影响。
    
    Attributes:
        group_id: 群体唯一标识
        strategy_name: 策略名称（用于展示）
        strategy_prompt: 共享的量化策略系统提示词
        agents: 群体内的 Agent 列表
    """
    group_id: str
    strategy_name: str
    strategy_prompt: str
    agents: List[DeepSeekBrain] = field(default_factory=list)
    
    # 集体行为监控
    collective_action: Optional[str] = None  # 最新的群体主导行为
    action_consensus: float = 0.0  # 行动一致性 (0-1)
    sell_pressure: float = 0.0  # 抛售压力 (0-1)
    
    # 历史记录
    action_history: List[Dict] = field(default_factory=list)
    
    # 阈值配置
    panic_sell_threshold: float = 0.7  # 超过70%卖出触发集体抛售警报
    
    def add_agent(self, agent_id: str, persona: Dict, api_key: Optional[str] = None, model_router: Optional[Any] = None):
        """
        添加一个使用共享策略的 Agent
        
        Args:
            agent_id: Agent 唯一标识
            persona: Agent 人格设定
            api_key: DeepSeek API 密钥
            model_router: 模型路由器
        """
        brain = DeepSeekBrain(
            agent_id=f"{self.group_id}_{agent_id}",
            persona=persona,
            api_key=api_key,
            model_router=model_router
        )
        # 注入共享策略提示词
        brain._shared_strategy_prompt = self.strategy_prompt
        self.agents.append(brain)
    
    async def get_group_decisions_async(
        self, 
        market_state: Dict, 
        account_states: Dict[str, Dict],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict]:
        """
        获取群体所有成员的决策 (异步)
        """
        import asyncio
        decisions = []
        total = len(self.agents)
        
        # 批量异步任务
        tasks = []
        
        for i, agent in enumerate(self.agents):
            acct = account_states.get(agent.agent_id, {
                "cash": GLOBAL_CONFIG.DEFAULT_CASH,
                "market_value": 0,
                "pnl_pct": 0
            })
            
            # 注入共享策略到市场状态
            enhanced_market_state = market_state.copy()
            enhanced_market_state['quant_strategy'] = self.strategy_prompt
            
            # 优先使用 think_async
            tasks.append(agent.think_async(enhanced_market_state, acct))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, res in enumerate(results):
             if isinstance(res, Exception):
                 print(f"Quant Agent Error: {res}")
                 continue
             
             res['agent_id'] = self.agents[i].agent_id
             decisions.append(res)
             
             if progress_callback:
                progress_callback(i + 1, total, self.agents[i].agent_id)
        
        # 分析集体行为
        self._analyze_collective_behavior(decisions)
        
        return decisions
    
    def _analyze_collective_behavior(self, decisions: List[Dict]):
        """
        分析群体集体行为
        
        检测是否发生集体抛售等异常行为
        """
        if not decisions:
            return
        
        actions = [d['decision'].get('action', 'HOLD') for d in decisions]
        total = len(actions)
        
        buy_count = actions.count('BUY')
        sell_count = actions.count('SELL')
        hold_count = actions.count('HOLD')
        
        # 计算各行为占比
        buy_ratio = buy_count / total
        sell_ratio = sell_count / total
        
        # 确定主导行为
        if sell_ratio >= self.panic_sell_threshold:
            self.collective_action = 'PANIC_SELL'
            self.sell_pressure = sell_ratio
        elif buy_ratio > 0.5:
            self.collective_action = 'BUY'
        elif sell_ratio > 0.5:
            self.collective_action = 'SELL'
        else:
            self.collective_action = 'MIXED'
        
        # 计算一致性（使用 Herfindahl 指数）
        self.action_consensus = buy_ratio**2 + sell_ratio**2 + (hold_count/total)**2
        
        # 记录历史
        self.action_history.append({
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'consensus': self.action_consensus,
            'collective_action': self.collective_action
        })
        
        # 保留最近100条记录
        if len(self.action_history) > 100:
            self.action_history.pop(0)
    
    def is_panic_selling(self) -> bool:
        """检测是否正在发生集体抛售"""
        return self.collective_action == 'PANIC_SELL'
    
    def get_tier2_signal(self) -> float:
        """
        生成传递给 Tier 2 散户的信号
        
        Returns:
            float: -1.0 (强烈看空) ~ 1.0 (强烈看多)
        """
        if not self.action_history:
            return 0.0
        
        latest = self.action_history[-1]
        
        # 基于买卖比例计算信号
        signal = latest['buy_ratio'] - latest['sell_ratio']
        
        # 如果正在集体抛售，放大负面信号
        if self.is_panic_selling():
            signal = min(-0.8, signal * 1.5)
        
        return max(-1.0, min(1.0, signal))
    
    def get_emotion_distribution(self) -> Dict[str, int]:
        """
        获取群体情绪分布
        
        Returns:
            Dict: {'greedy': count, 'neutral': count, 'fearful': count}
        """
        greedy = 0
        neutral = 0
        fearful = 0
        
        for agent in self.agents:
            history = DeepSeekBrain.thought_history.get(agent.agent_id, [])
            if history:
                latest_emotion = history[-1].emotion_score
                if latest_emotion > 0.3:
                    greedy += 1
                elif latest_emotion < -0.3:
                    fearful += 1
                else:
                    neutral += 1
        
        return {
            'greedy': greedy,
            'neutral': neutral,
            'fearful': fearful
        }


class QuantGroupManager:
    """
    量化群体管理器
    
    管理多个量化策略群体，协调它们与市场的交互
    """
    
    # 预置策略模板
    STRATEGY_TEMPLATES = {
        'momentum': """
你是一个动量策略交易者。你的核心理念是"追涨杀跌"：
- 当市场趋势向上时，你倾向于买入
- 当市场趋势向下时，你倾向于卖出
- 你非常重视交易量和价格突破信号
- 你愿意承担较高风险以获取超额收益
        """,
        
        'mean_reversion': """
你是一个均值回归策略交易者。你的核心理念是"低买高卖"：
- 当价格大幅低于均值时，你认为是买入机会
- 当价格大幅高于均值时，你认为应该卖出
- 你相信极端行情会回归正常
- 你愿意逆势操作，但控制仓位
        """,
        
        'risk_parity': """
你是一个风险平价策略交易者。你的核心理念是"控制风险"：
- 你密切关注波动率和恐慌指数
- 当市场波动加剧时，你倾向于减仓
- 当市场平稳时，你逐步加仓
- 你极度厌恶大幅亏损
        """,
        
        'news_driven': """
你是一个消息驱动策略交易者。你的核心理念是"信息就是金钱"：
- 你非常重视政策消息和市场新闻
- 利好消息让你积极买入
- 利空消息让你迅速撤退
- 你的反应速度比其他投资者更快
        """
    }
    
    def __init__(self, api_key: Optional[str] = None, model_router: Optional[Any] = None):
        self.api_key = api_key
        self.model_router = model_router
        self.groups: Dict[str, QuantStrategyGroup] = {}

    def set_model_router(self, router: Any):
        """Later binding of model router"""
        self.model_router = router
        for group in self.groups.values():
            for agent in group.agents:
                if hasattr(agent, 'set_model_router'):
                    agent.set_model_router(router)
    
    def create_group(
        self, 
        group_id: str, 
        strategy_name: str,
        strategy_prompt: str,
        n_agents: int = 10,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> QuantStrategyGroup:
        """
        创建一个量化策略群体
        """
        group = QuantStrategyGroup(
            group_id=group_id,
            strategy_name=strategy_name,
            strategy_prompt=strategy_prompt
        )
        
        for i in range(n_agents):
            if progress_callback:
                progress_callback(i + 1, n_agents, f"Agent_{i}")
            
            # 为每个 Agent 生成略有差异的人格
            persona = {
                'risk_preference': random.choice(['激进', '稳健', '保守']),
                'loss_aversion': random.uniform(1.5, 3.0)
            }
            group.add_agent(f"Agent_{i}", persona, self.api_key, self.model_router)
        
        self.groups[group_id] = group
        return group
    
    def create_from_template(
        self, 
        group_id: str, 
        template_name: str,
        n_agents: int = 10,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Optional[QuantStrategyGroup]:
        """
        使用预置模板创建群体
        
        Args:
            group_id: 群体ID
            template_name: 模板名称 (momentum/mean_reversion/risk_parity/news_driven)
            n_agents: Agent 数量
            progress_callback: 进度回调
            
        Returns:
            创建的群体实例，模板不存在则返回 None
        """
        if template_name not in self.STRATEGY_TEMPLATES:
            return None
        
        strategy_prompt = self.STRATEGY_TEMPLATES[template_name]
        strategy_name = {
            'momentum': '动量追踪',
            'mean_reversion': '均值回归',
            'risk_parity': '风险平价',
            'news_driven': '消息驱动'
        }.get(template_name, template_name)
        
        return self.create_group(
            group_id, 
            strategy_name, 
            strategy_prompt, 
            n_agents,
            progress_callback
        )
    
    def get_all_signals(self) -> Dict[str, float]:
        """
        获取所有群体的 Tier 2 信号
        
        Returns:
            {group_id: signal}
        """
        return {
            gid: group.get_tier2_signal() 
            for gid, group in self.groups.items()
        }
    
    def detect_systemic_risk(self) -> Dict:
        """
        检测系统性风险
        
        当多个量化群体同时抛售时，可能引发系统性风险
        
        Returns:
            风险评估报告
        """
        panic_groups = [g for g in self.groups.values() if g.is_panic_selling()]
        total_groups = len(self.groups)
        
        if total_groups == 0:
            return {'risk_level': 'low', 'panic_ratio': 0, 'warning': None}
        
        panic_ratio = len(panic_groups) / total_groups
        
        if panic_ratio >= 0.5:
            risk_level = 'critical'
            warning = '⚠️ 系统性风险警告：超过50%的量化群体正在集体抛售！'
        elif panic_ratio >= 0.25:
            risk_level = 'high'
            warning = '⚡ 高风险警告：多个量化群体出现抛售行为'
        elif panic_ratio > 0:
            risk_level = 'medium'
            warning = '📊 注意：部分量化群体出现抛售倾向'
        else:
            risk_level = 'low'
            warning = None
        
        return {
            'risk_level': risk_level,
            'panic_ratio': panic_ratio,
            'panic_groups': [g.group_id for g in panic_groups],
            'warning': warning
        }
