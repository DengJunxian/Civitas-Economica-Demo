# file: core/exchange/order_book.py
"""
高保真中央限价订单簿 (CLOB) 撮合引擎

基于 sortedcontainers.SortedList 实现 O(log n) 的订单插入与删除。
遵循价格-时间优先级 (Price-Time Priority) 撮合规则。

参考架构:
- Microsoft MarS: https://arxiv.org/abs/2409.07486
- 真实交易所系统设计原则

作者: Civitas Economica Team
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from sortedcontainers import SortedList

from config import GLOBAL_CONFIG


# ==========================================
# 枚举类型定义
# ==========================================

class OrderSide(str, Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型"""
    LIMIT = "limit"      # 限价单
    MARKET = "market"    # 市价单


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"      # 待撮合
    PARTIAL = "partial"      # 部分成交
    FILLED = "filled"        # 完全成交
    CANCELLED = "cancelled"  # 已撤销
    REJECTED = "rejected"    # 被拒绝


# ==========================================
# 核心数据结构
# ==========================================

@dataclass
class Order:
    """
    高保真订单对象
    
    包含完整的订单生命周期追踪属性。
    支持限价单和市价单两种类型。
    
    Attributes:
        order_id: 唯一标识符 (UUID)
        agent_id: 下单代理的 ID
        timestamp: 订单创建时间戳 (用于时间优先级)
        symbol: 交易标的代码
        side: 买/卖方向
        order_type: 订单类型 (限价/市价)
        price: 委托价格 (市价单为 None 或 0)
        quantity: 委托数量
        status: 当前订单状态
        filled_qty: 已成交数量
    """
    order_id: str
    agent_id: str
    timestamp: float
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    
    @staticmethod
    def create(
        agent_id: str,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        quantity: int,
        order_id: Optional[str] = None
    ) -> "Order":
        """
        工厂方法: 创建新订单
        
        Args:
            agent_id: 代理 ID
            symbol: 交易标的
            side: 'buy' 或 'sell'
            order_type: 'limit' 或 'market'
            price: 委托价格
            quantity: 委托数量
            order_id: 可选的指定订单 ID (如果不传则自动生成 UUID)
            
        Returns:
            新创建的 Order 对象
        """
        return Order(
            order_id=order_id if order_id else str(uuid.uuid4()),
            agent_id=agent_id,
            timestamp=time.time(),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType(order_type),
            price=price,
            quantity=quantity
        )
    
    @property
    def remaining_qty(self) -> int:
        """剩余未成交数量"""
        return self.quantity - self.filled_qty
    
    @property
    def is_filled(self) -> bool:
        """是否完全成交"""
        return self.filled_qty >= self.quantity
    
    def __hash__(self):
        return hash(self.order_id)
    
    def __eq__(self, other):
        if isinstance(other, Order):
            return self.order_id == other.order_id
        return False


@dataclass
class Trade:
    """
    成交记录 (执行回报)
    
    包含完整的成交信息和费用计算。
    
    Attributes:
        trade_id: 成交唯一标识
        price: 成交价格 (由 Maker 决定)
        quantity: 成交数量
        maker_id: 被动方订单 ID
        taker_id: 主动方订单 ID
        maker_agent_id: 被动方代理 ID
        taker_agent_id: 主动方代理 ID
        timestamp: 成交时间戳
        buyer_fee: 买方佣金
        seller_fee: 卖方佣金
        seller_tax: 卖方印花税
    """
    trade_id: str
    price: float
    quantity: int
    maker_id: str
    taker_id: str
    maker_agent_id: str
    taker_agent_id: str
    timestamp: float
    buyer_fee: float = 0.0
    seller_fee: float = 0.0
    seller_tax: float = 0.0
    
    @property
    def notional(self) -> float:
        """成交金额 (名义价值)"""
        return self.price * self.quantity
    
    @property
    def buyer_total_cost(self) -> float:
        """买方总支出 = 成交金额 + 佣金"""
        return self.notional + self.buyer_fee
    
    @property
    def seller_net_proceeds(self) -> float:
        """卖方净收入 = 成交金额 - 佣金 - 印花税"""
        return self.notional - self.seller_fee - self.seller_tax


# ==========================================
# 订单簿实现
# ==========================================

class OrderBook:
    """
    中央限价订单簿 (Central Limit Order Book)
    
    使用 SortedList 实现高效的订单管理:
    - Bids (买单): 按价格降序、时间升序排列
    - Asks (卖单): 按价格升序、时间升序排列
    
    Features:
    - O(log n) 订单插入与删除
    - 价格-时间优先级撮合
    - 支持部分成交
    - 支持订单取消 (模拟撤单行为)
    - 涨跌停限制检查
    
    Attributes:
        symbol: 交易标的代码
        prev_close: 前收盘价 (用于涨跌停计算)
        bids: 买单队列
        asks: 卖单队列
        orders: 订单 ID 到订单对象的映射
    """
    
    def __init__(self, symbol: str = "A_SHARE_IDX", prev_close: float = 3000.0):
        """
        初始化订单簿
        
        Args:
            symbol: 交易标的代码
            prev_close: 前收盘价
        """
        self.symbol = symbol
        self.prev_close = prev_close
        
        # Bids: 价格降序 (高价优先), 同价时间升序 (先到先得)
        # key: (-price, timestamp) 确保高价优先、时间早优先
        self.bids: SortedList = SortedList(key=lambda o: (-o.price, o.timestamp))
        
        # Asks: 价格升序 (低价优先), 同价时间升序
        # key: (price, timestamp) 确保低价优先、时间早优先
        self.asks: SortedList = SortedList(key=lambda o: (o.price, o.timestamp))
        
        # 订单索引: order_id -> Order
        self.orders: Dict[str, Order] = {}
        
        # 统计信息
        self.last_price: float = prev_close
        self.total_volume: int = 0
        self.trades_history: List[Trade] = []
        
        # 当前步骤的成交缓冲
        self._step_trades: List[Trade] = []
    
    # ------------------------------------------
    # 价格限制检查
    # ------------------------------------------
    
    
    def _get_dynamic_limit(self) -> float:
        """根据证券代码获取动态涨跌幅限制"""
        limit = GLOBAL_CONFIG.PRICE_LIMIT
        sym = str(self.symbol)
        
        if sym.startswith("688") or sym.startswith("300"):
            limit = 0.20
        elif sym.startswith("8"):
            limit = 0.30
        elif "ST" in sym.upper():
            limit = 0.05
            
        return limit
    
    def _check_price_limit(self, price: float) -> bool:
        """
        检查价格是否在涨跌停范围内
        
        A 股主板: ±10% (默认)
        科创板/创业板: ±20%
        北交所: ±30%
        ST: ±5%
        
        Args:
            price: 待检查价格
            
        Returns:
            是否在有效范围内
        """
        limit = self._get_dynamic_limit()
        upper = self.prev_close * (1 + limit)
        lower = self.prev_close * (1 - limit)
        return round(lower, 2) <= round(price, 2) <= round(upper, 2)
    
    def get_limit_prices(self) -> Tuple[float, float]:
        """
        获取当日涨跌停价格
        
        Returns:
            (跌停价, 涨停价)
        """
        limit = self._get_dynamic_limit()
        lower = round(self.prev_close * (1 - limit), 2)
        upper = round(self.prev_close * (1 + limit), 2)
        return lower, upper
    
    # ------------------------------------------
    # 订单提交与撮合
    # ------------------------------------------
    
    def add_order(self, order: Order) -> List[Trade]:
        """
        添加订单并尝试撮合
        
        Args:
            order: 待添加的订单
            
        Returns:
            本次撮合产生的成交列表
        """
        # 1. 校验标的
        if order.symbol != self.symbol:
            order.status = OrderStatus.REJECTED
            return []
        
        # 2. 限价单价格检查
        if order.order_type == OrderType.LIMIT:
            if not self._check_price_limit(order.price):
                order.status = OrderStatus.REJECTED
                return []
        
        # 3. 市价单转换为激进限价单 (涨停/跌停价)
        if order.order_type == OrderType.MARKET:
            lower, upper = self.get_limit_prices()
            order.price = upper if order.side == OrderSide.BUY else lower
        
        # 4. 尝试撮合
        trades = self.match_order(order)
        
        # 5. 未完全成交则挂单
        if not order.is_filled and order.status != OrderStatus.CANCELLED:
            self._add_to_book(order)
        
        return trades
    
    def match_order(self, incoming: Order) -> List[Trade]:
        """
        订单撮合核心逻辑
        
        严格遵循价格-时间优先级 (Price-Time Priority):
        1. 价格优先: 买方出价高者先成交，卖方报价低者先成交
        2. 时间优先: 同价位先提交者先成交
        
        Args:
            incoming: 入场订单 (Taker)
            
        Returns:
            成交列表
        """
        trades: List[Trade] = []
        
        if incoming.side == OrderSide.BUY:
            # 买单 vs 卖单队列
            trades = self._match_against_asks(incoming)
        else:
            # 卖单 vs 买单队列
            trades = self._match_against_bids(incoming)
        
        return trades
    
    def _match_against_asks(self, buy_order: Order) -> List[Trade]:
        """
        买单与卖单队列撮合
        
        条件: 买入价 >= 最优卖价
        """
        trades: List[Trade] = []
        orders_to_remove: List[Order] = []
        
        for ask_order in self.asks:
            if buy_order.remaining_qty <= 0:
                break
            
            # 价格检查: 买入价 >= 卖出价
            if buy_order.price >= ask_order.price:
                trade = self._execute_trade(
                    buy_order=buy_order,
                    sell_order=ask_order,
                    taker=buy_order,
                    maker=ask_order
                )
                trades.append(trade)
                
                if ask_order.is_filled:
                    orders_to_remove.append(ask_order)
            else:
                # 价格不再满足，后续订单价格只会更高
                break
        
        # 移除已完全成交的订单
        for order in orders_to_remove:
            self.asks.remove(order)
            self.orders.pop(order.order_id, None)
        
        return trades
    
    def _match_against_bids(self, sell_order: Order) -> List[Trade]:
        """
        卖单与买单队列撮合
        
        条件: 卖出价 <= 最优买价
        """
        trades: List[Trade] = []
        orders_to_remove: List[Order] = []
        
        for bid_order in self.bids:
            if sell_order.remaining_qty <= 0:
                break
            
            # 价格检查: 卖出价 <= 买入价
            if sell_order.price <= bid_order.price:
                trade = self._execute_trade(
                    buy_order=bid_order,
                    sell_order=sell_order,
                    taker=sell_order,
                    maker=bid_order
                )
                trades.append(trade)
                
                if bid_order.is_filled:
                    orders_to_remove.append(bid_order)
            else:
                break
        
        # 移除已完全成交的订单
        for order in orders_to_remove:
            self.bids.remove(order)
            self.orders.pop(order.order_id, None)
        
        return trades
    
    def _execute_trade(
        self,
        buy_order: Order,
        sell_order: Order,
        taker: Order,
        maker: Order
    ) -> Trade:
        """
        执行成交
        
        成交价由 Maker (被动方) 决定。
        
        Args:
            buy_order: 买方订单
            sell_order: 卖方订单
            taker: 主动方 (入场订单)
            maker: 被动方 (挂单)
            
        Returns:
            成交记录
        """
        # 1. 确定成交量
        exec_qty = min(buy_order.remaining_qty, sell_order.remaining_qty)
        
        # 2. 确定成交价 (Maker's Price)
        exec_price = maker.price
        
        # 3. 更新订单状态
        buy_order.filled_qty += exec_qty
        sell_order.filled_qty += exec_qty
        
        for o in [buy_order, sell_order]:
            if o.is_filled:
                o.status = OrderStatus.FILLED
            elif o.filled_qty > 0:
                o.status = OrderStatus.PARTIAL
        
        # 4. 计算费用
        notional = exec_price * exec_qty
        comm_rate = GLOBAL_CONFIG.TAX_RATE_COMMISSION
        stamp_rate = GLOBAL_CONFIG.TAX_RATE_STAMP
        
        buyer_fee = notional * comm_rate
        seller_fee = notional * comm_rate
        seller_tax = notional * stamp_rate  # 印花税仅卖方
        
        # 5. 创建成交记录
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            price=exec_price,
            quantity=exec_qty,
            maker_id=maker.order_id,
            taker_id=taker.order_id,
            maker_agent_id=maker.agent_id,
            taker_agent_id=taker.agent_id,
            timestamp=time.time(),
            buyer_fee=buyer_fee,
            seller_fee=seller_fee,
            seller_tax=seller_tax
        )
        
        # 6. 更新统计
        self.last_price = exec_price
        self.total_volume += exec_qty
        self.trades_history.append(trade)
        self._step_trades.append(trade)
        
        return trade
    
    def _add_to_book(self, order: Order) -> None:
        """
        将订单加入订单簿
        """
        self.orders[order.order_id] = order
        
        if order.side == OrderSide.BUY:
            self.bids.add(order)
        else:
            self.asks.add(order)
    
    # ------------------------------------------
    # 订单取消
    # ------------------------------------------
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        模拟高频交易中的撤单行为，用于分析高频交易风险（如幌骗/Spoofing）。
        
        Args:
            order_id: 待取消的订单 ID
            
        Returns:
            是否成功取消
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # 只能取消未完全成交的订单
        if order.status == OrderStatus.FILLED:
            return False
        
        # 从订单簿中移除
        try:
            if order.side == OrderSide.BUY:
                self.bids.remove(order)
            else:
                self.asks.remove(order)
        except ValueError:
            pass  # 订单可能不在队列中
        
        # 更新状态
        order.status = OrderStatus.CANCELLED
        del self.orders[order_id]
        
        return True
    
    # ------------------------------------------
    # 市场数据查询
    # ------------------------------------------
    
    def get_best_bid(self) -> Optional[float]:
        """获取最优买价 (Best Bid)"""
        if not self.bids:
            return None
        return self.bids[0].price
    
    def get_best_ask(self) -> Optional[float]:
        """获取最优卖价 (Best Ask)"""
        if not self.asks:
            return None
        return self.asks[0].price
    
    def get_spread(self) -> Optional[float]:
        """
        获取买卖价差 (Bid-Ask Spread)
        
        Returns:
            价差，或 None 如果无法计算
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask - best_bid
    
    def get_mid_price(self) -> Optional[float]:
        """
        获取中间价 (Mid Price)
        
        Returns:
            (Best Bid + Best Ask) / 2
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2
    
    def get_depth(self, levels: int = 5) -> Dict:
        """
        获取市场深度 (L5 行情)
        
        Args:
            levels: 档位数量
            
        Returns:
            {
                "bids": [{"price": float, "qty": int}, ...],
                "asks": [{"price": float, "qty": int}, ...]
            }
        """
        bids_display = []
        for order in self.bids[:levels]:
            bids_display.append({
                "price": order.price,
                "qty": order.remaining_qty
            })
        
        asks_display = []
        for order in self.asks[:levels]:
            asks_display.append({
                "price": order.price,
                "qty": order.remaining_qty
            })
        
        return {"bids": bids_display, "asks": asks_display}
    
    # ------------------------------------------
    # 步骤管理
    # ------------------------------------------
    
    def flush_step_trades(self) -> List[Trade]:
        """
        获取并清空当前步骤的成交记录
        
        Returns:
            本步骤的成交列表
        """
        trades = self._step_trades[:]
        self._step_trades = []
        return trades
    
    def update_prev_close(self, close_price: float) -> None:
        """
        更新前收盘价
        
        在每日收盘后调用，用于下一日的涨跌停计算。
        
        Args:
            close_price: 收盘价
        """
        self.prev_close = close_price
    
    def clear(self) -> None:
        """
        清空订单簿
        
        保留前收盘价和统计信息。
        """
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("订单簿撮合引擎测试")
    print("=" * 60)
    
    # 创建订单簿
    ob = OrderBook(symbol="SH000001", prev_close=3000.0)
    
    # 1. 提交限价卖单
    sell_order = Order.create(
        agent_id="seller_alice",
        symbol="SH000001",
        side="sell",
        order_type="limit",
        price=3005.0,
        quantity=100
    )
    trades1 = ob.add_order(sell_order)
    print(f"\n[卖单] Alice 挂单: 价格 3005, 数量 100")
    print(f"  - 成交数: {len(trades1)}")
    print(f"  - 当前盘口: {ob.get_depth(1)}")
    
    # 2. 提交限价买单 (会与卖单交叉)
    buy_order = Order.create(
        agent_id="buyer_bob",
        symbol="SH000001",
        side="buy",
        order_type="limit",
        price=3010.0,
        quantity=60
    )
    trades2 = ob.add_order(buy_order)
    print(f"\n[买单] Bob 下单: 价格 3010, 数量 60")
    print(f"  - 成交数: {len(trades2)}")
    
    if trades2:
        t = trades2[0]
        print(f"  - 成交价: {t.price} (Maker 价格)")
        print(f"  - 成交量: {t.quantity}")
        print(f"  - 买方支付: {t.buyer_total_cost:.2f}")
        print(f"  - 卖方收入: {t.seller_net_proceeds:.2f}")
    
    print(f"  - 卖单剩余: {sell_order.remaining_qty}")
    print(f"  - 当前盘口: {ob.get_depth(1)}")
    
    # 3. 测试撤单
    print(f"\n[撤单] 取消剩余卖单...")
    result = ob.cancel_order(sell_order.order_id)
    print(f"  - 撤单结果: {'成功' if result else '失败'}")
    print(f"  - 当前盘口: {ob.get_depth(1)}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
