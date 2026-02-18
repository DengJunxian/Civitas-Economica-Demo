# file: core/types.py
"""
统一的数据类型定义
Order, Trade, Candle 等核心数据结构
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import uuid
import time # Fallback for creation time if clock is not available, but should be avoided in simulation context

class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    """订单类型"""
    LIMIT = "limit"
    MARKET = "market"

class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """
    通用订单对象
    """
    symbol: str
    price: float
    quantity: int
    side: OrderSide
    order_type: OrderType
    agent_id: str
    timestamp: float # 仿真逻辑时间 (SimulationClock time)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    reason: str = "" # 订单原因 (如 MARGIN_CALL) 或 拒绝原因
    filled_qty: int = 0
    
    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty
    
    @property
    def is_filled(self) -> bool:
        return self.filled_qty >= self.quantity

    @property
    def value(self) -> float:
        """订单总价值"""
        return self.price * self.quantity

@dataclass
class Trade:
    """
    成交记录
    """
    trade_id: str
    price: float
    quantity: int
    maker_id: str
    taker_id: str
    maker_agent_id: str
    taker_agent_id: str
    buyer_agent_id: str # 明确的买方 ID
    seller_agent_id: str # 明确的卖方 ID
    timestamp: float # 仿真逻辑时间
    buyer_fee: float = 0.0
    seller_fee: float = 0.0
    seller_tax: float = 0.0
    
    @property
    def notional(self) -> float:
        return self.price * self.quantity
    
    @property
    def buyer_total_cost(self) -> float:
        return self.notional + self.buyer_fee
    
    @property
    def seller_net_proceeds(self) -> float:
        return self.notional - self.seller_fee - self.seller_tax

@dataclass
class Candle:
    """
    K线数据 (OHLCV)
    """
    symbol: str
    step: int # Simulation step index
    timestamp: str # YYYY-MM-DD HH:MM:SS
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float = 0.0
    is_simulated: bool = False
