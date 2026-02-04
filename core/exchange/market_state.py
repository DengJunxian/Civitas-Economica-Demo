# file: core/exchange/market_state.py
"""
市场状态管理模块

参考 Microsoft MarS 架构设计：
- L1 市场快照 (Best Bid/Ask)
- VWAP 计算
- 仿真步进控制

作者: Civitas Economica Team
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from core.exchange.order_book import OrderBook, Order, Trade, OrderSide, OrderType


# ==========================================
# 市场状态数据结构
# ==========================================

@dataclass
class MarketState:
    """
    L1 市场快照
    
    包含实时市场行情的关键指标，用于 Agent 决策。
    
    Attributes:
        timestamp: 快照时间戳
        best_bid: 最优买价
        best_ask: 最优卖价
        last_price: 最新成交价
        vwap: 成交量加权平均价
        total_volume: 累计成交量
        bid_volume: 买一档挂单量
        ask_volume: 卖一档挂单量
        spread: 买卖价差
    """
    timestamp: float = 0.0
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    last_price: float = 0.0
    vwap: float = 0.0
    total_volume: int = 0
    bid_volume: int = 0
    ask_volume: int = 0
    spread: Optional[float] = None
    
    @property
    def mid_price(self) -> Optional[float]:
        """中间价"""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "timestamp": self.timestamp,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "last_price": self.last_price,
            "vwap": self.vwap,
            "total_volume": self.total_volume,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "spread": self.spread,
            "mid_price": self.mid_price
        }


@dataclass
class StepResult:
    """
    仿真步骤结果
    
    Attributes:
        step_id: 步骤编号
        market_state: 步骤结束时的市场状态
        trades: 本步骤产生的所有成交
        rejected_orders: 被拒绝的订单
    """
    step_id: int
    market_state: MarketState
    trades: List[Trade] = field(default_factory=list)
    rejected_orders: List[Order] = field(default_factory=list)


# ==========================================
# 交易所核心类
# ==========================================

class Exchange:
    """
    虚拟交易所
    
    负责管理订单簿、处理 Agent 动作、更新市场状态。
    遵循 Microsoft MarS 架构的 step() 仿真模式。
    
    Features:
    - 订单簿管理
    - 批量订单处理
    - VWAP 实时计算
    - 市场状态快照
    
    Attributes:
        order_book: 订单簿实例
        current_step: 当前仿真步骤
        _vwap_numerator: VWAP 计算分子累积
        _vwap_denominator: VWAP 计算分母累积
    """
    
    def __init__(
        self,
        symbol: str = "A_SHARE_IDX",
        prev_close: float = 3000.0
    ):
        """
        初始化交易所
        
        Args:
            symbol: 交易标的代码
            prev_close: 前收盘价
        """
        self.order_book = OrderBook(symbol=symbol, prev_close=prev_close)
        self.symbol = symbol
        self.current_step: int = 0
        
        # VWAP 计算状态
        self._vwap_numerator: float = 0.0  # Σ(price * volume)
        self._vwap_denominator: int = 0     # Σ(volume)
        
        # 历史记录
        self.state_history: List[MarketState] = []
        self.step_results: List[StepResult] = []
    
    # ------------------------------------------
    # 仿真步进
    # ------------------------------------------
    
    def step(self, actions: List[Order]) -> StepResult:
        """
        执行一个仿真步骤
        
        处理 Agent 的所有动作，更新市场状态。
        
        Args:
            actions: 本步骤的所有订单 (Agent 动作)
            
        Returns:
            步骤结果，包含市场状态和成交列表
        """
        self.current_step += 1
        all_trades: List[Trade] = []
        rejected_orders: List[Order] = []
        
        # 1. 处理所有订单
        for order in actions:
            trades = self.order_book.add_order(order)
            all_trades.extend(trades)
            
            if order.status.value == "rejected":
                rejected_orders.append(order)
        
        # 2. 更新 VWAP
        for trade in all_trades:
            self._vwap_numerator += trade.price * trade.quantity
            self._vwap_denominator += trade.quantity
        
        # 3. 生成市场状态快照
        market_state = self._create_market_state()
        self.state_history.append(market_state)
        
        # 4. 创建步骤结果
        result = StepResult(
            step_id=self.current_step,
            market_state=market_state,
            trades=all_trades,
            rejected_orders=rejected_orders
        )
        self.step_results.append(result)
        
        return result
    
    def _create_market_state(self) -> MarketState:
        """
        创建当前市场状态快照
        
        Returns:
            MarketState 对象
        """
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()
        
        # 计算档位挂单量
        bid_volume = 0
        ask_volume = 0
        depth = self.order_book.get_depth(1)
        if depth["bids"]:
            bid_volume = depth["bids"][0]["qty"]
        if depth["asks"]:
            ask_volume = depth["asks"][0]["qty"]
        
        # 计算 VWAP
        vwap = 0.0
        if self._vwap_denominator > 0:
            vwap = self._vwap_numerator / self._vwap_denominator
        
        return MarketState(
            timestamp=time.time(),
            best_bid=best_bid,
            best_ask=best_ask,
            last_price=self.order_book.last_price,
            vwap=vwap,
            total_volume=self.order_book.total_volume,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            spread=self.order_book.get_spread()
        )
    
    # ------------------------------------------
    # 订单操作接口
    # ------------------------------------------
    
    def submit_order(self, order: Order) -> List[Trade]:
        """
        提交单个订单
        
        供外部直接调用，不通过 step() 批量处理。
        
        Args:
            order: 订单对象
            
        Returns:
            成交列表
        """
        return self.order_book.add_order(order)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        Args:
            order_id: 订单 ID
            
        Returns:
            是否成功取消
        """
        return self.order_book.cancel_order(order_id)
    
    # ------------------------------------------
    # 市场数据查询
    # ------------------------------------------
    
    def get_market_state(self) -> MarketState:
        """获取当前市场状态"""
        return self._create_market_state()
    
    def get_order_book_depth(self, levels: int = 5) -> Dict:
        """获取订单簿深度"""
        return self.order_book.get_depth(levels)
    
    def get_vwap(self) -> float:
        """获取当前 VWAP"""
        if self._vwap_denominator == 0:
            return self.order_book.last_price
        return self._vwap_numerator / self._vwap_denominator
    
    # ------------------------------------------
    # 日终处理
    # ------------------------------------------
    
    def end_of_day(self, close_price: Optional[float] = None) -> None:
        """
        日终处理
        
        更新前收盘价，重置 VWAP。
        
        Args:
            close_price: 收盘价，默认使用最后成交价
        """
        if close_price is None:
            close_price = self.order_book.last_price
        
        self.order_book.update_prev_close(close_price)
        
        # 重置 VWAP
        self._vwap_numerator = 0.0
        self._vwap_denominator = 0
    
    def reset(self) -> None:
        """
        重置交易所状态
        
        清空订单簿和历史记录。
        """
        self.order_book.clear()
        self.current_step = 0
        self._vwap_numerator = 0.0
        self._vwap_denominator = 0
        self.state_history.clear()
        self.step_results.clear()


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exchange 仿真步进测试")
    print("=" * 60)
    
    # 创建交易所
    exchange = Exchange(symbol="SH000001", prev_close=3000.0)
    
    # 模拟 Agent 动作批次
    actions_step1 = [
        Order.create("noise_1", "SH000001", "sell", "limit", 3005.0, 100),
        Order.create("noise_2", "SH000001", "sell", "limit", 3008.0, 200),
        Order.create("noise_3", "SH000001", "buy", "limit", 2995.0, 150),
    ]
    
    # 步骤 1: 建立初始流动性
    result1 = exchange.step(actions_step1)
    print(f"\n步骤 1:")
    print(f"  成交数: {len(result1.trades)}")
    print(f"  市场状态: {result1.market_state.to_dict()}")
    
    # 步骤 2: 激进买单触发成交
    actions_step2 = [
        Order.create("quant_agent", "SH000001", "buy", "limit", 3010.0, 80),
    ]
    
    result2 = exchange.step(actions_step2)
    print(f"\n步骤 2:")
    print(f"  成交数: {len(result2.trades)}")
    if result2.trades:
        t = result2.trades[0]
        print(f"  成交价: {t.price}, 成交量: {t.quantity}")
    print(f"  VWAP: {exchange.get_vwap():.2f}")
    print(f"  市场状态: {result2.market_state.to_dict()}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
