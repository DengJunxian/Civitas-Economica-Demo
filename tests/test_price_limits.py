from __future__ import annotations

from datetime import datetime, timedelta

from core.exchange.order_book import OrderBook
from core.types import Order, OrderSide, OrderStatus, OrderType


def _order(side: OrderSide, price: float, qty: int, ts: float, *, metadata=None) -> Order:
    return Order.create(
        agent_id=f"{side.value}_{price}",
        symbol="TEST",
        side=side,
        order_type=OrderType.LIMIT,
        price=price,
        quantity=qty,
        timestamp=ts,
        metadata=dict(metadata or {}),
    )


def test_a_share_price_limit_rejects_orders_beyond_limit():
    book = OrderBook(symbol="TEST", prev_close=100.0, feature_flags={"market_rules_v1": True})
    _, upper = book.get_limit_prices()
    too_high = _order(OrderSide.BUY, upper + 0.01, 100, datetime(2026, 3, 23, 9, 30).timestamp())
    trades = book.add_order(too_high)
    assert trades == []
    assert too_high.status == OrderStatus.REJECTED

    inside = _order(OrderSide.BUY, upper, 100, datetime(2026, 3, 23, 9, 31).timestamp())
    assert book.add_order(inside) == []
    assert inside.status == OrderStatus.PENDING


def test_t_plus_one_sell_constraint_rejects_same_day_sale():
    ts = datetime(2026, 3, 23, 10, 0).timestamp()
    book = OrderBook(
        symbol="TEST",
        prev_close=100.0,
        market_rules={"t_plus_one": True},
        feature_flags={"market_rules_v1": True},
    )
    sell = _order(OrderSide.SELL, 100.0, 100, ts, metadata={"position_acquired_ts": ts - 60})
    assert book.add_order(sell) == []
    assert sell.status == OrderStatus.REJECTED
    assert sell.reason == "t+1 sell restriction"

    next_day_sell = _order(
        OrderSide.SELL,
        100.0,
        100,
        (datetime(2026, 3, 24, 10, 0)).timestamp(),
        metadata={"position_acquired_ts": ts},
    )
    assert book.add_order(next_day_sell) == []
    assert next_day_sell.status == OrderStatus.PENDING


def test_lot_size_validation_rejects_non_board_lot():
    book = OrderBook(
        symbol="TEST",
        prev_close=100.0,
        market_rules={"board_lot": 100, "enforce_board_lot": True, "allow_odd_lots": False},
        feature_flags={"market_rules_v1": True},
    )
    odd = _order(OrderSide.BUY, 100.0, 50, datetime(2026, 3, 23, 9, 30).timestamp())
    assert book.add_order(odd) == []
    assert odd.status == OrderStatus.REJECTED
    assert "lot" in odd.reason

    round_lot = _order(OrderSide.BUY, 100.0, 200, datetime(2026, 3, 23, 9, 31).timestamp())
    assert book.add_order(round_lot) == []
    assert round_lot.status == OrderStatus.PENDING
