# file: core/scheduler.py

import asyncio
import random
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from enum import Enum

from config import GLOBAL_CONFIG
from core.market_engine import MarketDataManager, Order, MatchingEngine
from core.validator import StylizedFactsValidator
from core.model_router import ModelRouter
from agents.population import StratifiedPopulation, SmartAgent
from core.account import Portfolio


class SimulationMode(Enum):
    """仿真模式枚举"""
    SMART = "SMART"  # 智能模式：自适应调度，≤15秒/天
    FAST = "FAST"    # 快速模式：仅对话模型，≤5秒/天
    DEEP = "DEEP"    # 深度模式：仅推理模型，≤30秒/天


from core.mesa.civitas_model import CivitasModel

class SimulationController:
    """
    异步仿真调度器 (The Matrix Controller)
    
    新增功能：
    1. 多模式调度 (SMART/FAST/DEEP)
    2. 天数计数与100天循环
    3. 多模型路由器集成
    4. 智能Agent采样
    """
    
    def __init__(
        self, 
        deepseek_key: str, 
        hunyuan_key: Optional[str] = None,
        zhipu_key: Optional[str] = None,
        mode: str = "SMART"
    ):
        """
        初始化仿真控制器
        
        Args:
            deepseek_key: DeepSeek API密钥（必需）
            hunyuan_key: 混元API密钥（可选）
            zhipu_key: 智谱API密钥（可选，快速模式专用）
            mode: 仿真模式 "SMART" | "FAST" | "DEEP"
        """
        # API密钥
        self.deepseek_key = deepseek_key
        self.hunyuan_key = hunyuan_key
        self.zhipu_key = zhipu_key
        
        # 仿真模式
        self.mode = SimulationMode(mode)
        
        # 初始化多模型路由器（包含智谱）
        self.model_router = ModelRouter(deepseek_key, hunyuan_key, zhipu_key)
        
        # 初始化核心组件
        # 使用 CivitasModel 接管 Market 和 Agent
        self.model = CivitasModel(
            n_agents=100, # 默认 100 个 Agents
            model_router=self.model_router,
            initial_price=3000.0
        )
        
        # 兼容性: MarketDataManager 由 Model 管理
        self.market = self.model.market_manager
        self.matcher = self.market.engine
        
        # 移除旧的 StratifiedPopulation 直接引用
        # self.population 由 self.model.population 接管 (对于 Tier 2)
        
        # 量化群体管理器 (可选注入)
        self.quant_manager = None
        
        # 智能体账户映射 (不再需要单独维护，Agent 自带 Portfolio)
        # 兼容旧代码访问 self.population.smart_agents
        # 但 CivitasModel.agents 是 AgentSet. 
        # 我们暂时保留 self.population 指向 Tier 2 Population 方便访问?
        # CivitasModel.population 是 StratifiedPopulation 实例
        # 我们可以 expose 它
        
        # 日志与状态
        
        # 日志与状态
        self.step_count = 0
        self.day_count = 0  # 仿真天数
        self.logs: List[str] = []
        self.latest_reasoning: Dict[str, str] = {}
        
        # 性能统计
        self.tick_times: List[float] = []
    
    def set_mode(self, mode: str):
        """切换仿真模式"""
        self.mode = SimulationMode(mode)
        print(f"[Scheduler] 仿真模式切换为: {self.mode.value}")
    
    def get_time_budget(self) -> float:
        """获取当前模式的时间预算"""
        if self.mode == SimulationMode.FAST:
            return GLOBAL_CONFIG.TIME_BUDGET_FAST
        elif self.mode == SimulationMode.DEEP:
            return GLOBAL_CONFIG.TIME_BUDGET_DEEP
        else:
            return GLOBAL_CONFIG.TIME_BUDGET_SMART
    
    def get_deep_agent_count(self) -> int:
        """获取需要深度思考的Agent数量"""
        if self.mode == SimulationMode.FAST:
            return GLOBAL_CONFIG.MAX_DEEP_AGENTS_FAST
        elif self.mode == SimulationMode.DEEP:
            return GLOBAL_CONFIG.MAX_DEEP_AGENTS_DEEP
        else:
            return GLOBAL_CONFIG.MAX_DEEP_AGENTS_SMART
    
    def get_model_priority(self) -> List[str]:
        """获取当前模式的模型优先级"""
        return self.model_router.get_model_priority(self.mode.value)

    async def run_tick(self) -> Dict[str, Any]:
        """
        执行单个仿真步 (One Tick)
        通过 CivitasModel 驱动
        """
        tick_start = time.time()
        
        # 1. 执行 Model Step
        await self.model.async_step()
        
        # 2. 同步状态
        self.step_count = self.model.steps
        self.day_count = self.model.steps
        
        # 3. 记录性能
        
        tick_time = time.time() - tick_start
        self.tick_times.append(tick_time)
        if len(self.tick_times) > 100:
            self.tick_times.pop(0)
            
        # 4. 构造返回结果 (UI 需要)
        # 从 Model 获取最新指标
        
        # 注意: apply_policy_async 现在需要更新 model.market_manager.policy
        # 由于 self.market 指向 self.model.market_manager，所以应该没问题
        
        return {
            "candle": self.market.candles[-1], 
            "trades": self.model.last_step_trades_count,
            "csad": self.model.csad,
            "smart_sentiment": self.model.last_smart_sentiment,
            "day_count": self.day_count,
            "tick_time": tick_time,
            "avg_tick_time": np.mean(self.tick_times) if self.tick_times else 0.0
        }

    def validate_simulation(self) -> Dict[str, Any]:
        """
        执行典型事实验证
        
        在仿真结束后调用，验证仿真市场是否表现出真实市场的统计特征。
        
        Returns:
            验证结果字典，包含各项指标的通过情况
        """
        validator = StylizedFactsValidator()
        
        # 提取仿真阶段的K线数据
        simulated_candles = [c for c in self.market.candles if c.is_simulated]
        
        if len(simulated_candles) < 30:
            return {
                "error": "仿真数据不足，需要至少30个交易日",
                "passed": 0,
                "total": 0
            }
        
        prices = [c.close for c in simulated_candles]
        volumes = [c.volume for c in simulated_candles]
        
        return validator.run_full_validation(prices, volumes)

    def finalize_market_step(self):
        """辅助方法，用于桥接 MarketDataManager 和 MatchingEngine"""
        pass