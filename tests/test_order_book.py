# file: tests/test_order_book.py
"""
订单簿撮合引擎单元测试

测试覆盖:
1. 限价单交叉撮合
2. 部分成交
3. 价格-时间优先级
4. 订单取消
5. 市价单处理

作者: Civitas Economica Team
"""

import pytest
import time

from core.exchange.order_book import (
    Order, Trade, OrderBook,
    OrderSide, OrderType, OrderStatus
)
from core.exchange.market_state import Exchange, MarketState


# ==========================================
# 辅助函数
# ==========================================

def create_test_order(
    side: str,
    price: float,
    quantity: int,
    agent_id: str = "test_agent",
    order_type: str = "limit"
) -> Order:
    """创建测试订单"""
    return Order.create(
        agent_id=agent_id,
        symbol="TEST",
        side=side,
        order_type=order_type,
        price=price,
        quantity=quantity
    )


# ==========================================
# OrderBook 测试类
# ==========================================

class TestOrderBook:
    """订单簿核心功能测试"""
    
    @pytest.fixture
    def order_book(self):
        """创建测试用订单簿"""
        return OrderBook(symbol="TEST", prev_close=100.0)
    
    # ------------------------------------------
    # 基础功能测试
    # ------------------------------------------
    
    def test_empty_order_book(self, order_book):
        """测试空订单簿"""
        assert order_book.get_best_bid() is None
        assert order_book.get_best_ask() is None
        assert order_book.get_spread() is None
        assert len(order_book.bids) == 0
        assert len(order_book.asks) == 0
    
    def test_add_single_buy_order(self, order_book):
        """测试添加单个买单"""
        order = create_test_order("buy", 99.0, 100)
        trades = order_book.add_order(order)
        
        assert len(trades) == 0  # 无对手方，不产生成交
        assert order_book.get_best_bid() == 99.0
        assert len(order_book.bids) == 1
    
    def test_add_single_sell_order(self, order_book):
        """测试添加单个卖单"""
        order = create_test_order("sell", 101.0, 100)
        trades = order_book.add_order(order)
        
        assert len(trades) == 0
        assert order_book.get_best_ask() == 101.0
        assert len(order_book.asks) == 1
    
    # ------------------------------------------
    # 撮合测试
    # ------------------------------------------
    
    def test_limit_buy_cross_limit_sell(self, order_book):
        """
        测试限价买单与限价卖单交叉撮合
        
        场景:
        1. Alice 挂卖单: 价格 100.50, 数量 100
        2. Bob 下买单: 价格 101.00, 数量 80
        
        预期:
        - 成交价格: 100.50 (Maker 价格)
        - 成交数量: 80
        - Alice 剩余: 20
        """
        # 1. Alice 挂卖单
        sell_order = create_test_order("sell", 100.50, 100, "alice")
        trades1 = order_book.add_order(sell_order)
        
        assert len(trades1) == 0
        assert order_book.get_best_ask() == 100.50
        
        # 2. Bob 下买单 (激进，会与 Alice 成交)
        buy_order = create_test_order("buy", 101.00, 80, "bob")
        trades2 = order_book.add_order(buy_order)
        
        # 验证成交
        assert len(trades2) == 1
        trade = trades2[0]
        
        # 验证成交价格 (Maker 价格)
        assert trade.price == 100.50
        
        # 验证成交数量
        assert trade.quantity == 80
        
        # 验证参与方
        assert trade.maker_agent_id == "alice"
        assert trade.taker_agent_id == "bob"
        
        # 验证订单状态
        assert sell_order.remaining_qty == 20
        assert sell_order.status == OrderStatus.PARTIAL
        assert buy_order.remaining_qty == 0
        assert buy_order.status == OrderStatus.FILLED
        
        # 验证盘口
        assert order_book.get_best_ask() == 100.50  # Alice 剩余挂单
        assert order_book.get_best_bid() is None    # Bob 已完全成交
    
    def test_partial_fill(self, order_book):
        """
        测试部分成交场景
        
        场景:
        1. 多个小卖单挂在不同价位
        2. 一个大买单扫单
        """
        # 挂多个卖单
        sells = [
            create_test_order("sell", 100.10, 30, "seller_1"),
            create_test_order("sell", 100.20, 40, "seller_2"),
            create_test_order("sell", 100.30, 50, "seller_3"),
        ]
        for s in sells:
            order_book.add_order(s)
        
        # 大买单
        big_buy = create_test_order("buy", 100.25, 60, "big_buyer")
        trades = order_book.add_order(big_buy)
        
        # 应该成交两笔
        assert len(trades) == 2
        
        # 第一笔: 与 100.10 的卖单成交 30
        assert trades[0].price == 100.10
        assert trades[0].quantity == 30
        
        # 第二笔: 与 100.20 的卖单部分成交 30
        assert trades[1].price == 100.20
        assert trades[1].quantity == 30
        
        # 买单完全成交
        assert big_buy.is_filled
        
        # 验证剩余盘口
        assert sells[1].remaining_qty == 10  # 40 - 30
        assert order_book.get_best_ask() == 100.20  # seller_2 剩余
    
    def test_price_time_priority(self, order_book):
        """
        测试价格-时间优先级
        
        场景: 同价位多个订单，先提交的先成交
        """
        # 添加两个同价位卖单，seller_1 先提交
        sell1 = create_test_order("sell", 100.00, 50, "seller_1")
        order_book.add_order(sell1)
        
        time.sleep(0.001)  # 确保时间戳不同
        
        sell2 = create_test_order("sell", 100.00, 50, "seller_2")
        order_book.add_order(sell2)
        
        # 下买单，只成交 50
        buy = create_test_order("buy", 100.00, 50, "buyer")
        trades = order_book.add_order(buy)
        
        # 应该与 seller_1 成交（时间优先）
        assert len(trades) == 1
        assert trades[0].maker_agent_id == "seller_1"
        
        # seller_1 完全成交，seller_2 仍挂在盘口
        assert sell1.is_filled
        assert not sell2.is_filled
        assert order_book.get_best_ask() == 100.00  # seller_2
    
    # ------------------------------------------
    # 撤单测试
    # ------------------------------------------
    
    def test_cancel_order_success(self, order_book):
        """测试成功撤单"""
        order = create_test_order("buy", 99.00, 100, "test")
        order_book.add_order(order)
        
        assert order_book.get_best_bid() == 99.00
        
        result = order_book.cancel_order(order.order_id)
        
        assert result is True
        assert order.status == OrderStatus.CANCELLED
        assert order_book.get_best_bid() is None
        assert order.order_id not in order_book.orders
    
    def test_cancel_order_not_found(self, order_book):
        """测试撤销不存在的订单"""
        result = order_book.cancel_order("non_existent_id")
        assert result is False
    
    def test_cancel_filled_order(self, order_book):
        """测试撤销已完全成交的订单"""
        # 先撮合一笔成交
        sell = create_test_order("sell", 100.00, 50, "seller")
        order_book.add_order(sell)
        
        buy = create_test_order("buy", 100.00, 50, "buyer")
        order_book.add_order(buy)
        
        # 尝试撤销已成交的买单
        assert buy.is_filled
        result = order_book.cancel_order(buy.order_id)
        
        assert result is False  # 已成交订单无法撤销
    
    # ------------------------------------------
    # 市价单测试
    # ------------------------------------------
    
    def test_market_order(self, order_book):
        """测试市价单 (转换为涨跌停价限价单)"""
        # 先挂卖单
        sell = create_test_order("sell", 100.00, 100, "seller")
        order_book.add_order(sell)
        
        # 市价买单
        market_buy = create_test_order("buy", 0, 50, "buyer", "market")
        assert market_buy.order_type == OrderType.MARKET
        
        trades = order_book.add_order(market_buy)
        
        # 应该立即成交
        assert len(trades) == 1
        assert trades[0].quantity == 50
        assert market_buy.is_filled
    
    # ------------------------------------------
    # 涨跌停测试
    # ------------------------------------------
    
    def test_price_limit_reject(self, order_book):
        """测试超出涨跌停限制的订单被拒绝"""
        # prev_close = 100.0, 涨停 = 110.0, 跌停 = 90.0
        
        # 超涨停买单
        buy_over_limit = create_test_order("buy", 115.0, 100, "test")
        trades = order_book.add_order(buy_over_limit)
        
        assert len(trades) == 0
        assert buy_over_limit.status == OrderStatus.REJECTED
        
        # 超跌停卖单
        sell_under_limit = create_test_order("sell", 85.0, 100, "test")
        trades = order_book.add_order(sell_under_limit)
        
        assert len(trades) == 0
        assert sell_under_limit.status == OrderStatus.REJECTED


# ==========================================
# Exchange 测试类
# ==========================================

class TestExchange:
    """交易所仿真步进测试"""
    
    @pytest.fixture
    def exchange(self):
        """创建测试用交易所"""
        return Exchange(symbol="TEST", prev_close=100.0)
    
    def test_step_empty_actions(self, exchange):
        """测试空操作步骤"""
        result = exchange.step([])
        
        assert result.step_id == 1
        assert len(result.trades) == 0
        assert result.market_state is not None
    
    def test_step_with_matching(self, exchange):
        """测试带撮合的步骤"""
        actions = [
            create_test_order("sell", 100.50, 100, "seller"),
            create_test_order("buy", 101.00, 80, "buyer"),
        ]
        
        result = exchange.step(actions)
        
        assert result.step_id == 1
        assert len(result.trades) == 1
        assert result.trades[0].quantity == 80
    
    def test_vwap_calculation(self, exchange):
        """测试 VWAP 计算"""
        # 步骤 1: 100.00 成交 50
        actions1 = [
            create_test_order("sell", 100.00, 50, "s1"),
            create_test_order("buy", 100.00, 50, "b1"),
        ]
        exchange.step(actions1)
        
        # 步骤 2: 102.00 成交 100
        actions2 = [
            create_test_order("sell", 102.00, 100, "s2"),
            create_test_order("buy", 102.00, 100, "b2"),
        ]
        exchange.step(actions2)
        
        # VWAP = (100*50 + 102*100) / (50+100) = 15200 / 150 ≈ 101.33
        vwap = exchange.get_vwap()
        expected_vwap = (100.00 * 50 + 102.00 * 100) / 150
        
        assert abs(vwap - expected_vwap) < 0.01
    
    def test_end_of_day(self, exchange):
        """测试日终处理"""
        # 产生一些成交
        actions = [
            create_test_order("sell", 105.00, 100, "seller"),
            create_test_order("buy", 105.00, 100, "buyer"),
        ]
        exchange.step(actions)
        
        # 日终
        exchange.end_of_day(close_price=105.00)
        
        # VWAP 应该重置
        assert exchange._vwap_denominator == 0
        
        # prev_close 应该更新
        assert exchange.order_book.prev_close == 105.00


# ==========================================
# Trade 费用计算测试
# ==========================================

class TestTradeFees:
    """成交费用计算测试"""
    
    def test_trade_buyer_cost(self):
        """测试买方总支出计算"""
        trade = Trade(
            trade_id="test",
            price=100.0,
            quantity=100,
            maker_id="m",
            taker_id="t",
            maker_agent_id="maker",
            taker_agent_id="taker",
            timestamp=time.time(),
            buyer_fee=3.0,
            seller_fee=3.0,
            seller_tax=10.0
        )
        
        # 买方支付 = 100 * 100 + 3 = 10003
        assert trade.buyer_total_cost == 10003.0
    
    def test_trade_seller_proceeds(self):
        """测试卖方净收入计算"""
        trade = Trade(
            trade_id="test",
            price=100.0,
            quantity=100,
            maker_id="m",
            taker_id="t",
            maker_agent_id="maker",
            taker_agent_id="taker",
            timestamp=time.time(),
            buyer_fee=3.0,
            seller_fee=3.0,
            seller_tax=10.0
        )
        
        # 卖方收入 = 100 * 100 - 3 - 10 = 9987
        assert trade.seller_net_proceeds == 9987.0


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
