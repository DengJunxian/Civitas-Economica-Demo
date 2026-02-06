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
        self.market = MarketDataManager(self.model_router, load_real_data=True)
        # 修正：直接使用 MarketDataManager 的 engine，避免实例不一致
        self.matcher = self.market.engine
        self.population = StratifiedPopulation(
            n_smart=50, n_vectorized=5000, api_key=deepseek_key
        )
        
        # 量化群体管理器 (可选注入)
        self.quant_manager = None
        
        # 为所有Smart Agent设置模型路由器
        for agent in self.population.smart_agents:
            agent.brain.set_model_router(self.model_router)
        
        # 智能体账户映射
        self.smart_portfolios: Dict[str, Portfolio] = {
            agent.id: Portfolio(initial_cash=agent.cash) 
            for agent in self.population.smart_agents
        }
        
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
        支持多模式调度和智能Agent采样
        """
        tick_start = time.time()
        current_price = self.market.engine.last_price
        timestamp = pd.Timestamp(self.market.candles[-1].timestamp)
        time_budget = self.get_time_budget()
        
        time_budget = self.get_time_budget()
        
        # --- 0. 更新 Prev Close (用于计算当日涨跌幅) ---
        # 在每日交易开始前，将 prev_close 设置为昨日收盘价
        # 注意：market.candles[-1] 是最近一根已完成的K线（即昨日）
        if self.market.candles:
            last_candle = self.market.candles[-1]
            self.matcher.update_prev_close(last_candle.close)

    def ensure_market_engine(self):
        """确保 Engine 引用正确"""
        if self.matcher != self.market.engine:
            self.matcher = self.market.engine

    async def apply_policy_async(self, policy_text: str) -> Dict:
        """异步应用政策 (在 Worker Loop 中调用)"""
        print(f"[Scheduled] 正在分析政策: {policy_text[:10]}...")
        self.ensure_market_engine()
        
        # 1. 异步调用 Interpreter
        result = await self.market.interpreter.interpret(policy_text)
        
        # 2. 更新 Policy 对象 (Thread-safe enough for simple attrs)
        self.market.policy.update(result)
        
        # 3. 记录
        self.market.current_news = result.get('initial_news', '政策发布')
        self.logs.append(f"[Policy] {self.market.current_news}")
        
        return result

        # --- Phase 1: 智能Agent采样和混合调度 ---
        n_deep = self.get_deep_agent_count()
        
        # 混合调度策略：按重要性排序，重要Agent用Reasoner，次要用GLM
        all_agents = sorted(
            self.population.smart_agents,
            key=lambda a: a.brain.importance_level,
            reverse=True  # 重要性高的在前
        )
        
        # 核心Agent（importance >= 2）：始终用Reasoner
        # 重要Agent（importance == 1）：根据配额用Reasoner
        # 普通Agent（importance == 0）：用GLM快速模型
        core_agents = [a for a in all_agents if a.brain.importance_level >= 2]
        important_agents = [a for a in all_agents if a.brain.importance_level == 1]
        normal_agents = [a for a in all_agents if a.brain.importance_level == 0]
        
        # 分配Reasoner配额：优先核心Agent，然后重要Agent
        n_reasoner_budget = n_deep
        reasoner_agents = core_agents[:n_reasoner_budget]
        remaining_budget = n_reasoner_budget - len(reasoner_agents)
        if remaining_budget > 0:
            reasoner_agents.extend(important_agents[:remaining_budget])
        
        # 剩余Agent用GLM
        glm_agents = [a for a in all_agents if a not in reasoner_agents][:10]  # 最多10个
        
        # 构建市场上下文
        market_ctx = {
            "price": current_price,
            "trend": "上涨" if self.market.candles[-1].close > self.market.candles[-1].open else "下跌",
            "panic_level": self.market.panic_level,
            "news": self.market.current_news,
            "policy_description": self.market.policy.description,
            "liquidity_injection": self.market.policy.liquidity_injection,
            "policy_reasoning": getattr(self.market.interpreter, 'last_reasoning', None)
        }
        
        # 构造深度思考任务（Reasoner模型）
        deep_tasks = []
        reasoner_priority = self.model_router.get_model_priority("DEEP")  # DeepSeek优先
        per_agent_budget = time_budget / max(len(reasoner_agents), 1) * 0.8
        
        for agent in reasoner_agents:
            pf = self.smart_portfolios[agent.id]
            acct_ctx = {
                "cash": pf.available_cash,
                "market_value": pf.get_market_value({"A_SHARE_IDX": current_price}),
                "pnl_pct": 0.0
            }
            deep_tasks.append(agent.brain.think_async(
                market_ctx, acct_ctx,
                model_priority=reasoner_priority,
                timeout_budget=per_agent_budget
            ))
        
        # 重命名变量以保持兼容性
        deep_agents = reasoner_agents
        fast_agents = glm_agents
        
        # 并发执行深度思考
        deep_decisions = await asyncio.gather(*deep_tasks, return_exceptions=True)
        
        # 快速模式：使用GLM-4-FlashX API调用（异步，高速）
        fast_tasks = []
        fast_model_priority = self.model_router.get_model_priority("FAST")  # GLM优先
        fast_budget = 5.0  # 快速模式每Agent预算5秒（增加容错）
        
        for agent in fast_agents:
            pf = self.smart_portfolios[agent.id]
            acct_ctx = {
                "cash": pf.available_cash,
                "market_value": pf.get_market_value({"A_SHARE_IDX": current_price}),
                "pnl_pct": 0.0
            }
            fast_tasks.append(agent.brain.think_async(
                market_ctx, acct_ctx,
                model_priority=fast_model_priority,
                timeout_budget=fast_budget
            ))
        
        # 并发执行快速思考
        fast_decisions = await asyncio.gather(*fast_tasks, return_exceptions=True)
        
        # 合并所有决策
        all_decisions = []
        all_agents = []
        
        # 处理深度决策
        for agent, result in zip(deep_agents, deep_decisions):
            if isinstance(result, Exception):
                # 异常情况使用fallback
                result = {"decision": {"action": "HOLD", "qty": 0}, "reasoning": f"错误: {result}"}
            all_decisions.append(result)
            all_agents.append(agent)
        
        # 处理快速决策
        for agent, result in zip(fast_agents, fast_decisions):
            if isinstance(result, Exception):
                # 异常情况使用fallback
                result = {"decision": {"action": "HOLD", "qty": 0}, "reasoning": f"GLM错误: {result}"}
            all_decisions.append(result)
            all_agents.append(agent)
        
        # 处理所有决策
        smart_actions = []
        
        for agent, result in zip(all_agents, all_decisions):
            # 1. 记录思维链 (用于 UI "Cognitive Lens")
            self.latest_reasoning[agent.id] = result.get('reasoning', '')
            
            # 2. 记录决策到社交网络（用于下一轮的影响传递）
            d = result.get('decision', {})
            self.population.record_smart_action(agent.id, {
                "action": d.get("action", "HOLD"),
                "confidence": d.get("confidence", 0.5),
                "emotion_score": result.get("emotion_score", 0.0),
                "reasoning": result.get("reasoning", "")
            })
            
            # 3. 提交订单
            action_code = 0
            
            if d.get('action') in ['BUY', 'SELL']:
                qty = int(d.get('qty', 0))
                price = float(d.get('price', current_price))
                side = d['action'].lower()
                
                if qty > 0:
                    pf = self.smart_portfolios[agent.id]
                    try:
                        if side == 'buy':
                            if pf.available_cash >= qty * price:
                                self.matcher.submit_order(Order(price, qty, agent.id, side, timestamp.timestamp()))
                                action_code = 1
                        elif side == 'sell':
                            if pf.get_sellable_qty("A_SHARE_IDX", timestamp) >= qty:
                                self.matcher.submit_order(Order(price, qty, agent.id, side, timestamp.timestamp()))
                                action_code = -1
                    except ValueError as e:
                        self.logs.append(f"[Error] Agent {agent.id}: {e}")
            
            smart_actions.append(action_code)

        # 补齐剩余 Smart Agent 的动作 (未思考的默认为 0)
        full_smart_actions = smart_actions + [0] * (len(self.population.smart_agents) - len(smart_actions))
        
        # --- Phase 2: Tier 2 (Vectorized) 群体演化 ---
        
        # 情绪传染
        trend_signal = 1.0 if market_ctx['trend'] == "上涨" else -1.0
        self.population.update_tier2_sentiment(full_smart_actions, trend_signal)
        
        # 批量生成决策
        v_actions, v_qtys, v_prices = self.population.generate_tier2_decisions(current_price)
        
        # 批量下单 (采样部分以减轻撮合压力)
        active_indices = np.where(v_actions != 0)[0]
        if len(active_indices) > 0:
            sample_size = min(len(active_indices), 500)
            chosen_idx = np.random.choice(active_indices, sample_size, replace=False)
            
            for idx in chosen_idx:
                side = 'buy' if v_actions[idx] == 1 else 'sell'
                order = Order(
                    price=float(v_prices[idx]),
                    quantity=int(v_qtys[idx]),
                    agent_id=f"Vec_{idx}",
                    side=side,
                    timestamp=timestamp.timestamp()
                )
                self.matcher.submit_order(order)

        # --- Phase 3: 量化群体决策 (如果存在) ---
        if self.quant_manager:
            # 获取所有量化群体的决策
            quant_decisions = self.quant_manager.get_group_decisions(
                market_ctx,  # 复用之前的 market_ctx
                {agent.id: {"cash": 1000000} for group in self.quant_manager.groups.values() for agent in group.agents} # 简化账户
            )
            
            for d in quant_decisions:
                decision = d.get('decision', {})
                if decision.get('action') in ['BUY', 'SELL']:
                    qty = int(decision.get('qty', 0))
                    price = float(decision.get('price', current_price))
                    side = decision['action'].lower()
                    agent_id = d.get('agent_id')
                    
                    if qty > 0:
                        self.matcher.submit_order(Order(price, qty, agent_id, side, timestamp.timestamp()))

        # --- Phase 4: 撮合与结算 ---
        trades = self.matcher.flush_step_trades() # 直接获取并清空
        
        # 同步 Tier 2 散户成交状态
        # 需要从成交记录中提取 "Vec_" 开头的条目并汇总
        vec_indices = []
        vec_prices = []
        vec_qtys = []
        vec_dirs = []
        
        for t in trades:
            # 检查买方是否为散户
            if t.buy_agent_id.startswith("Vec_"):
                idx = int(t.buy_agent_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(1) # Buy
                
            # 检查卖方是否为散户
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
            
        # 更新 K 线
        last_date = self.market.candles[-1].timestamp if self.market.candles else "2024-01-01"
        new_candle = self.market.finalize_step(self.step_count, last_date)
        
        # 日终结算
        for pf in self.smart_portfolios.values():
            pf.settle()
        
        # 更新计数器
        self.step_count += 1
        self.day_count += 1
        
        # 记录性能
        tick_time = time.time() - tick_start
        self.tick_times.append(tick_time)
        if len(self.tick_times) > 100:
            self.tick_times.pop(0)
        
        return {
            "candle": new_candle,
            "trades": len(trades),
            "csad": self.population.calculate_csad(),
            "smart_sentiment": np.mean(full_smart_actions),
            "day_count": self.day_count,
            "tick_time": tick_time,
            "avg_tick_time": np.mean(self.tick_times) if self.tick_times else 0
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