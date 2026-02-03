# file: core/regulatory_sandbox.py
"""
监管沙盒模块

实现中国 A 股特色监管机制：
1. 程序化交易新规 (2024) - 高频交易限制
2. 国家金融稳定基金 - "国家队"救市机制
3. 熔断机制 - 极端行情暂停交易
4. 异常交易监控 - 撤单率、报单速度等
"""

import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

from config import GLOBAL_CONFIG


class TradingRestriction(Enum):
    """交易限制类型"""
    NONE = "none"                    # 无限制
    WARNING = "warning"              # 警告
    RATE_LIMIT = "rate_limit"        # 限流
    LIQUIDITY_FEE = "liquidity_fee"  # 流动性费用
    SUSPENDED = "suspended"          # 停止交易


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: float
    agent_id: str
    order_type: str  # limit, market, cancel
    price: float
    qty: int
    executed: bool = True


@dataclass
class AgentTradeStats:
    """Agent 交易统计"""
    agent_id: str
    order_count_1s: int = 0         # 1秒内报单数
    order_count_today: int = 0       # 当日报单数
    cancel_count_today: int = 0      # 当日撤单数
    high_freq_warnings: int = 0      # 高频警告次数
    current_restriction: TradingRestriction = TradingRestriction.NONE
    liquidity_fee_paid: float = 0.0  # 已缴流动性费用


class ProgrammaticTradingRegulator:
    """
    程序化交易监管模块
    
    实现《证券市场程序化交易管理规定（试行）》(2024)
    
    核心规则：
    1. 1秒内申报超过300笔 → 触发高频交易限制
    2. 撤单率超过50% → 加收流量费
    3. 累计违规3次 → 暂停交易
    """
    
    # 监管参数
    HFT_ORDER_LIMIT_1S = 300       # 1秒内报单上限
    HFT_ORDER_LIMIT_1M = 5000      # 1分钟内报单上限
    CANCEL_RATE_LIMIT = 0.5       # 撤单率上限 50%
    LIQUIDITY_FEE_RATE = 0.0005   # 流动性费率 0.05%
    MAX_WARNINGS = 3              # 最大警告次数
    
    def __init__(self):
        self.agent_stats: Dict[str, AgentTradeStats] = {}
        self.trade_records: Dict[str, List[TradeRecord]] = defaultdict(list)
        self.violations: List[Dict] = []
        
        # 时间窗口控制
        self._last_cleanup = time.time()
    
    def register_order(
        self, 
        agent_id: str, 
        order_type: str,
        price: float,
        qty: int
    ) -> Tuple[bool, str]:
        """
        注册订单并检查合规性
        
        Args:
            agent_id: Agent ID
            order_type: 订单类型 (limit/market/cancel)
            price: 价格
            qty: 数量
            
        Returns:
            (是否允许, 原因)
        """
        current_time = time.time()
        
        # 初始化统计
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = AgentTradeStats(agent_id=agent_id)
        
        stats = self.agent_stats[agent_id]
        
        # 检查是否被暂停
        if stats.current_restriction == TradingRestriction.SUSPENDED:
            return False, "⛔ 交易权限已被暂停"
        
        # 记录交易
        record = TradeRecord(
            timestamp=current_time,
            agent_id=agent_id,
            order_type=order_type,
            price=price,
            qty=qty
        )
        self.trade_records[agent_id].append(record)
        
        # 更新统计
        if order_type == 'cancel':
            stats.cancel_count_today += 1
        else:
            stats.order_count_today += 1
        
        # 计算1秒内报单数
        one_second_ago = current_time - 1
        recent_orders = [
            r for r in self.trade_records[agent_id]
            if r.timestamp > one_second_ago and r.order_type != 'cancel'
        ]
        stats.order_count_1s = len(recent_orders)
        
        # 检查高频交易
        if stats.order_count_1s > self.HFT_ORDER_LIMIT_1S:
            stats.high_freq_warnings += 1
            self._record_violation(agent_id, "高频交易", 
                                   f"1秒内报单{stats.order_count_1s}笔，超过{self.HFT_ORDER_LIMIT_1S}限制")
            
            if stats.high_freq_warnings >= self.MAX_WARNINGS:
                stats.current_restriction = TradingRestriction.SUSPENDED
                return False, "⛔ 高频交易违规次数过多，交易权限已暂停"
            else:
                stats.current_restriction = TradingRestriction.RATE_LIMIT
                return False, f"⚠️ 高频交易警告 ({stats.high_freq_warnings}/{self.MAX_WARNINGS})"
        
        # 检查撤单率
        if stats.order_count_today > 10:  # 至少10笔才计算
            cancel_rate = stats.cancel_count_today / stats.order_count_today
            if cancel_rate > self.CANCEL_RATE_LIMIT:
                fee = price * qty * self.LIQUIDITY_FEE_RATE
                stats.liquidity_fee_paid += fee
                stats.current_restriction = TradingRestriction.LIQUIDITY_FEE
                return True, f"💰 撤单率过高({cancel_rate:.1%})，加收流动性费 ¥{fee:.2f}"
        
        # 定期清理历史记录
        if current_time - self._last_cleanup > 60:
            self._cleanup_old_records()
        
        return True, "✅ 正常"
    
    def _record_violation(self, agent_id: str, violation_type: str, detail: str):
        """记录违规"""
        self.violations.append({
            "timestamp": time.time(),
            "agent_id": agent_id,
            "type": violation_type,
            "detail": detail
        })
    
    def _cleanup_old_records(self):
        """清理超过1分钟的交易记录"""
        cutoff = time.time() - 60
        for agent_id in self.trade_records:
            self.trade_records[agent_id] = [
                r for r in self.trade_records[agent_id]
                if r.timestamp > cutoff
            ]
        self._last_cleanup = time.time()
    
    def reset_daily_stats(self):
        """重置每日统计"""
        for stats in self.agent_stats.values():
            stats.order_count_today = 0
            stats.cancel_count_today = 0
            if stats.current_restriction != TradingRestriction.SUSPENDED:
                stats.current_restriction = TradingRestriction.NONE
    
    def get_violation_report(self) -> List[Dict]:
        """获取违规报告"""
        return self.violations[-50:]  # 最近50条


@dataclass
class StabilityFundConfig:
    """稳定基金配置"""
    total_capital: float = 500_000_000_000  # 5000亿
    single_intervention_max: float = 50_000_000_000  # 单次最大500亿
    panic_threshold: float = 0.7  # 恐慌指数触发阈值
    price_drop_threshold: float = -0.05  # 跌幅触发阈值 -5%
    cooldown_ticks: int = 10  # 干预冷却期


class NationalStabilityFund:
    """
    国家金融稳定基金
    
    模拟"国家队"救市行为
    
    干预触发条件：
    1. 全市场恐慌指数 > 0.7
    2. 主要指数跌幅 > 5%
    3. 上一次干预已过冷却期
    
    干预方式：
    1. 大量买入蓝筹股/ETF
    2. 发布稳定市场声明（影响情绪）
    """
    
    def __init__(self, config: Optional[StabilityFundConfig] = None):
        self.config = config or StabilityFundConfig()
        
        # 状态
        self.available_capital = self.config.total_capital
        self.deployed_capital = 0.0
        self.intervention_count = 0
        self.last_intervention_tick = -100
        
        # 历史记录
        self.intervention_history: List[Dict] = []
        
        # 持仓
        self.holdings: Dict[str, int] = {}  # symbol -> qty
    
    def should_intervene(
        self, 
        panic_level: float, 
        price_change: float,
        current_tick: int
    ) -> bool:
        """
        判断是否应该入场干预
        
        Args:
            panic_level: 当前恐慌指数 (0-1)
            price_change: 当日涨跌幅
            current_tick: 当前 tick
            
        Returns:
            是否应该干预
        """
        # 冷却期检查
        if current_tick - self.last_intervention_tick < self.config.cooldown_ticks:
            return False
        
        # 资金检查
        if self.available_capital < self.config.single_intervention_max * 0.1:
            return False
        
        # 触发条件
        panic_triggered = panic_level > self.config.panic_threshold
        drop_triggered = price_change < self.config.price_drop_threshold
        
        return panic_triggered or drop_triggered
    
    def intervene(
        self, 
        current_tick: int,
        price: float,
        panic_level: float
    ) -> Dict:
        """
        执行干预
        
        Returns:
            干预详情
        """
        # 计算干预力度（恐慌越高，干预越大）
        intensity = min(1.0, (panic_level - 0.5) * 2)
        capital_to_deploy = self.config.single_intervention_max * intensity
        capital_to_deploy = min(capital_to_deploy, self.available_capital)
        
        # 计算买入数量
        qty = int(capital_to_deploy / price)
        
        # 更新状态
        self.available_capital -= capital_to_deploy
        self.deployed_capital += capital_to_deploy
        self.intervention_count += 1
        self.last_intervention_tick = current_tick
        
        # 更新持仓
        self.holdings['INDEX'] = self.holdings.get('INDEX', 0) + qty
        
        # 记录历史
        intervention_record = {
            "tick": current_tick,
            "capital": capital_to_deploy,
            "qty": qty,
            "price": price,
            "panic_level": panic_level,
            "statement": self._generate_statement(capital_to_deploy)
        }
        self.intervention_history.append(intervention_record)
        
        return intervention_record
    
    def _generate_statement(self, capital: float) -> str:
        """生成官方声明"""
        capital_yi = capital / 100_000_000  # 转换为亿
        
        statements = [
            f"国家金融稳定基金已出手 {capital_yi:.0f} 亿元维护市场稳定。",
            f"相关部门密切关注市场动态，已投入 {capital_yi:.0f} 亿元支持市场流动性。",
            f"稳定基金增持 {capital_yi:.0f} 亿元，彰显对市场长期健康发展的信心。"
        ]
        
        return statements[self.intervention_count % len(statements)]
    
    def get_tier2_calming_effect(self) -> float:
        """
        获取对 Tier 2 散户的安抚效果
        
        国家队入场后，散户情绪会得到一定程度的安抚
        
        Returns:
            安抚效果 (0-1)
        """
        if not self.intervention_history:
            return 0.0
        
        last = self.intervention_history[-1]
        capital_ratio = last['capital'] / self.config.single_intervention_max
        
        # 效果随时间衰减
        # 这里简化处理，实际应该考虑 tick 差异
        return capital_ratio * 0.5
    
    def get_status_report(self) -> Dict:
        """获取状态报告"""
        return {
            "可用资金": f"¥{self.available_capital/1e8:.0f} 亿",
            "已投入资金": f"¥{self.deployed_capital/1e8:.0f} 亿",
            "干预次数": self.intervention_count,
            "当前持仓市值": sum(self.holdings.values()) * 3000  # 假设价格3000
        }


class CircuitBreaker:
    """
    熔断机制
    
    A股熔断规则（历史规则，已废止，但可用于研究）：
    - 一级熔断：涨跌 ±5% 暂停15分钟
    - 二级熔断：涨跌 ±7% 暂停至收盘
    """
    
    LEVEL1_THRESHOLD = 0.05   # 5%
    LEVEL2_THRESHOLD = 0.07   # 7%
    LEVEL1_HALT_TICKS = 15    # 模拟15分钟
    
    def __init__(self):
        self.is_halted = False
        self.halt_level = 0
        self.halt_start_tick = 0
        self.halt_history: List[Dict] = []
    
    def check_and_halt(
        self, 
        price_change: float, 
        current_tick: int
    ) -> Tuple[bool, str]:
        """
        检查是否触发熔断
        
        Args:
            price_change: 当日涨跌幅
            current_tick: 当前 tick
            
        Returns:
            (是否熔断, 消息)
        """
        # 如果已经熔断，检查是否可以恢复
        if self.is_halted:
            if self.halt_level == 1:
                # 一级熔断：检查是否已过15分钟
                if current_tick - self.halt_start_tick >= self.LEVEL1_HALT_TICKS:
                    self.is_halted = False
                    self.halt_level = 0
                    return False, "🔔 一级熔断解除，交易恢复"
                else:
                    remaining = self.LEVEL1_HALT_TICKS - (current_tick - self.halt_start_tick)
                    return True, f"🔴 一级熔断中，剩余 {remaining} tick"
            else:
                # 二级熔断：全天暂停
                return True, "🔴🔴 二级熔断，今日交易暂停"
        
        # 检查是否触发新的熔断
        abs_change = abs(price_change)
        
        if abs_change >= self.LEVEL2_THRESHOLD:
            self.is_halted = True
            self.halt_level = 2
            self.halt_start_tick = current_tick
            self.halt_history.append({
                "tick": current_tick,
                "level": 2,
                "price_change": price_change
            })
            return True, f"🔴🔴 触发二级熔断！涨跌幅 {price_change:+.2%}"
        
        elif abs_change >= self.LEVEL1_THRESHOLD:
            self.is_halted = True
            self.halt_level = 1
            self.halt_start_tick = current_tick
            self.halt_history.append({
                "tick": current_tick,
                "level": 1,
                "price_change": price_change
            })
            return True, f"🔴 触发一级熔断！涨跌幅 {price_change:+.2%}，暂停15分钟"
        
        return False, ""
    
    def reset(self):
        """重置熔断状态（新的一天）"""
        self.is_halted = False
        self.halt_level = 0
        self.halt_start_tick = 0


class RegulatoryModule:
    """
    监管综合模块
    
    整合所有监管功能
    """
    
    def __init__(self):
        self.trading_regulator = ProgrammaticTradingRegulator()
        self.stability_fund = NationalStabilityFund()
        self.circuit_breaker = CircuitBreaker()
        
        # 回调函数
        self._on_intervention: Optional[Callable[[Dict], None]] = None
        self._on_circuit_break: Optional[Callable[[str], None]] = None
    
    def set_intervention_callback(self, callback: Callable[[Dict], None]):
        """设置干预回调"""
        self._on_intervention = callback
    
    def set_circuit_break_callback(self, callback: Callable[[str], None]):
        """设置熔断回调"""
        self._on_circuit_break = callback
    
    def process_tick(
        self,
        current_tick: int,
        price: float,
        prev_close: float,
        panic_level: float
    ) -> Dict:
        """
        处理每个 tick 的监管逻辑
        
        Returns:
            监管状态字典
        """
        price_change = (price - prev_close) / prev_close if prev_close else 0
        
        result = {
            "tick": current_tick,
            "circuit_break": None,
            "intervention": None,
            "trading_halted": False
        }
        
        # 1. 检查熔断
        halted, halt_msg = self.circuit_breaker.check_and_halt(price_change, current_tick)
        if halted:
            result["circuit_break"] = halt_msg
            result["trading_halted"] = True
            if self._on_circuit_break:
                self._on_circuit_break(halt_msg)
        
        # 2. 检查是否需要国家队入场（熔断时不干预）
        if not halted and self.stability_fund.should_intervene(
            panic_level, price_change, current_tick
        ):
            intervention = self.stability_fund.intervene(
                current_tick, price, panic_level
            )
            result["intervention"] = intervention
            if self._on_intervention:
                self._on_intervention(intervention)
        
        return result
    
    def get_comprehensive_report(self) -> Dict:
        """获取综合监管报告"""
        return {
            "程序化交易监管": {
                "监控Agent数": len(self.trading_regulator.agent_stats),
                "今日违规次数": len(self.trading_regulator.violations)
            },
            "国家稳定基金": self.stability_fund.get_status_report(),
            "熔断状态": {
                "当前状态": "熔断中" if self.circuit_breaker.is_halted else "正常",
                "历史熔断次数": len(self.circuit_breaker.halt_history)
            }
        }
