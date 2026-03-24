import pytest

import core.market_engine as market_engine_module
from core.exchange.order_book import OrderBook
from core.market_engine import MatchingEngine
from core.types import ExecutionPlan, Order, OrderSide, OrderStatus, OrderType


def create_order(side: OrderSide, price: float, qty: int, order_type: OrderType = OrderType.LIMIT) -> Order:
    return Order.create(
        agent_id="test_agent",
        symbol="TEST",
        side=side,
        order_type=order_type,
        price=price,
        quantity=qty,
        timestamp=1000.0,
    )


def test_market_order_does_not_rest():
    book = OrderBook(symbol="TEST", prev_close=100.0)
    book.add_order(create_order(OrderSide.SELL, 100.0, 5))

    market_buy = create_order(OrderSide.BUY, 0.0, 10, OrderType.MARKET)
    trades = book.add_order(market_buy)

    assert sum(t.quantity for t in trades) == 5
    assert market_buy.status == OrderStatus.PARTIAL
    assert book.get_best_bid() is None


def test_ioc_partially_fills_without_resting():
    book = OrderBook(symbol="TEST", prev_close=100.0)
    book.add_order(create_order(OrderSide.SELL, 100.0, 5))

    ioc_buy = create_order(OrderSide.BUY, 100.0, 10, OrderType.IOC)
    trades = book.add_order(ioc_buy)

    assert sum(t.quantity for t in trades) == 5
    assert ioc_buy.remaining_qty == 5
    assert ioc_buy.status == OrderStatus.PARTIAL
    assert book.get_best_bid() is None


def test_fok_rejects_when_liquidity_is_insufficient():
    book = OrderBook(symbol="TEST", prev_close=100.0)
    book.add_order(create_order(OrderSide.SELL, 100.0, 5))

    fok_buy = create_order(OrderSide.BUY, 100.0, 10, OrderType.FOK)
    trades = book.add_order(fok_buy)

    assert trades == []
    assert fok_buy.status == OrderStatus.REJECTED
    assert fok_buy.order_id not in book.orders


def test_post_only_rejects_if_it_would_take_liquidity():
    book = OrderBook(symbol="TEST", prev_close=100.0)
    book.add_order(create_order(OrderSide.SELL, 100.0, 5))

    post_only_buy = create_order(OrderSide.BUY, 101.0, 5, OrderType.POST_ONLY)
    trades = book.add_order(post_only_buy)

    assert trades == []
    assert post_only_buy.status == OrderStatus.REJECTED
    assert post_only_buy.order_id not in book.orders


def test_twap_like_execution_plan_is_split_into_child_orders(monkeypatch):
    monkeypatch.setattr(market_engine_module, "USE_CPP_LOB", False)
    engine = MatchingEngine(symbol="TEST", prev_close=100.0)

    for _ in range(3):
        engine.lob.add_order(create_order(OrderSide.SELL, 100.0, 30))

    plan = ExecutionPlan(
        symbol="TEST",
        agent_id="buyer",
        action="BUY",
        side=OrderSide.BUY,
        target_qty=90,
        urgency=0.8,
        order_type=OrderType.LIMIT,
        max_slippage=0.02,
        participation_rate=0.2,
        slicing_rule="twap-like",
        cancel_replace_policy="cancel-replace",
        time_horizon=3,
        price=100.0,
        child_order_schedule=[30, 30, 30],
        seed=11,
        snapshot_info={"symbol": "TEST", "last_price": 100.0, "timestamp": 1000.0},
    )

    trades = engine.submit_order(plan)

    assert sum(t.quantity for t in trades) == 90
    assert len(trades) == 3
    assert engine.lob.trade_count == 3
    assert engine.lob.submitted_orders == 6
