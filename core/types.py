# file: core/types.py
"""
统一的核心数据类型定义。
包含订单、成交和K线等结构。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import uuid


class OrderSide(str, Enum):
    """订单方向。"""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型。"""

    LIMIT = "limit"
    MARKET = "market"


class OrderStatus(str, Enum):
    """订单状态。"""

    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """通用订单对象。"""

    symbol: str
    price: float
    quantity: int
    side: OrderSide
    order_type: OrderType
    agent_id: str
    timestamp: float
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    reason: str = ""
    filled_qty: int = 0

    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty

    @property
    def is_filled(self) -> bool:
        return self.filled_qty >= self.quantity

    @property
    def value(self) -> float:
        """订单总价值。"""
        return self.price * self.quantity

    @property
    def action(self) -> str:
        """
        兼容旧字段：部分旧测试与脚本使用 `order.action` 表示买卖方向。
        """
        return self.side.value.upper()

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        symbol: str,
        side: str | OrderSide,
        order_type: str | OrderType,
        price: float,
        quantity: int,
        timestamp: Optional[float] = None,
    ) -> "Order":
        """
        兼容旧接口的订单工厂方法。
        支持字符串或枚举输入，未传时间戳时使用当前时间。
        """
        side_enum = side if isinstance(side, OrderSide) else OrderSide(str(side).lower())
        type_enum = order_type if isinstance(order_type, OrderType) else OrderType(str(order_type).lower())
        return cls(
            symbol=str(symbol),
            price=float(price),
            quantity=int(quantity),
            side=side_enum,
            order_type=type_enum,
            agent_id=str(agent_id),
            timestamp=float(time.time() if timestamp is None else timestamp),
        )


@dataclass
class Trade:
    """成交记录。"""

    trade_id: str
    price: float
    quantity: int
    maker_id: str
    taker_id: str
    maker_agent_id: str
    taker_agent_id: str
    buyer_agent_id: str = ""
    seller_agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
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
    """K线数据（OHLCV）。"""

    symbol: str
    step: int
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float = 0.0
    is_simulated: bool = False
