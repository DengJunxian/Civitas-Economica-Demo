import math

import pytest

from core.exchange.market_rules import build_market_rule_schema
from core.exchange.order_book import OrderBook
from core.types import Order, OrderSide, OrderStatus, OrderType

pytest.importorskip("_civitas_lob")
from core.exchange.order_book_cpp import OrderBookCPP  # noqa: E402


RULE_OVERRIDES = {
    "min_price_tick": 0.01,
    "min_trade_unit": 10,
    "enforce_min_trade_unit": True,
    "timestamp_precision": "microsecond",
}
FEATURE_FLAGS = {"market_rules_v1": True, "strict_queue_timestamps": True}


def _schema():
    return build_market_rule_schema("TEST", 100.0, overrides=RULE_OVERRIDES, feature_flags=FEATURE_FLAGS)


def _order(*, agent_id: str, side: OrderSide, price: float, qty: int, ts: float, order_type: OrderType = OrderType.LIMIT) -> Order:
    return Order.create(
        agent_id=agent_id,
        symbol="TEST",
        side=side,
        order_type=order_type,
        price=price,
        quantity=qty,
        timestamp=ts,
    )


def _make_books():
    schema = _schema()
    return (
        OrderBook(symbol="TEST", prev_close=100.0, market_rules=schema),
        OrderBookCPP(symbol="TEST", prev_close=100.0, market_rules=schema),
    )


def _depth_signature(book):
    depth = book.get_depth(levels=5)
    return {
        side: [(round(level["price"], 6), int(level["qty"])) for level in depth[side]]
        for side in ("bids", "asks")
    }


def _trade_signature(trade):
    return {
        "price": round(trade.price, 6),
        "quantity": int(trade.quantity),
        "buyer_fee": round(trade.buyer_fee, 6),
        "seller_fee": round(trade.seller_fee, 6),
        "seller_tax": round(trade.seller_tax, 6),
        "timestamp": round(trade.timestamp, 6),
    }


def test_python_cpp_limit_cross_trade_parity():
    py_book, cpp_book = _make_books()
    orders = [
        _order(agent_id="seller", side=OrderSide.SELL, price=100.0, qty=100, ts=1000.001000),
        _order(agent_id="buyer", side=OrderSide.BUY, price=100.0, qty=40, ts=1000.002000),
    ]

    py_book.add_order(orders[0])
    cpp_book.add_order(_order(agent_id="seller", side=OrderSide.SELL, price=100.0, qty=100, ts=1000.001000))
    py_trades = py_book.add_order(orders[1])
    cpp_trades = cpp_book.add_order(_order(agent_id="buyer", side=OrderSide.BUY, price=100.0, qty=40, ts=1000.002000))

    assert [_trade_signature(t) for t in py_trades] == [_trade_signature(t) for t in cpp_trades]
    assert _depth_signature(py_book) == _depth_signature(cpp_book)
    assert py_book.last_price == cpp_book.last_price == 100.0


def test_python_cpp_ioc_remainder_does_not_rest():
    py_book, cpp_book = _make_books()
    for book in (py_book, cpp_book):
        book.add_order(_order(agent_id="seller", side=OrderSide.SELL, price=100.0, qty=30, ts=1001.000000))

    py_ioc = _order(agent_id="buyer", side=OrderSide.BUY, price=100.0, qty=50, ts=1001.100000, order_type=OrderType.IOC)
    cpp_ioc = _order(agent_id="buyer", side=OrderSide.BUY, price=100.0, qty=50, ts=1001.100000, order_type=OrderType.IOC)

    py_trades = py_book.add_order(py_ioc)
    cpp_trades = cpp_book.add_order(cpp_ioc)

    assert [_trade_signature(t) for t in py_trades] == [_trade_signature(t) for t in cpp_trades]
    assert py_ioc.status == cpp_ioc.status == OrderStatus.PARTIAL
    assert py_ioc.remaining_qty == cpp_ioc.remaining_qty == 20
    assert _depth_signature(py_book) == _depth_signature(cpp_book) == {"bids": [], "asks": []}


def test_python_cpp_post_only_and_fok_regressions_match():
    py_book, cpp_book = _make_books()
    for book in (py_book, cpp_book):
        book.add_order(_order(agent_id="seller", side=OrderSide.SELL, price=100.0, qty=50, ts=1002.000000))

    py_post_only = _order(agent_id="buyer", side=OrderSide.BUY, price=101.0, qty=20, ts=1002.100000, order_type=OrderType.POST_ONLY)
    cpp_post_only = _order(agent_id="buyer", side=OrderSide.BUY, price=101.0, qty=20, ts=1002.100000, order_type=OrderType.POST_ONLY)
    py_fok = _order(agent_id="buyer", side=OrderSide.BUY, price=100.0, qty=60, ts=1002.200000, order_type=OrderType.FOK)
    cpp_fok = _order(agent_id="buyer", side=OrderSide.BUY, price=100.0, qty=60, ts=1002.200000, order_type=OrderType.FOK)

    assert py_book.add_order(py_post_only) == cpp_book.add_order(cpp_post_only) == []
    assert py_post_only.status == cpp_post_only.status == OrderStatus.REJECTED
    assert py_book.add_order(py_fok) == cpp_book.add_order(cpp_fok) == []
    assert py_fok.status == cpp_fok.status == OrderStatus.REJECTED
    assert _depth_signature(py_book) == _depth_signature(cpp_book) == {"bids": [], "asks": [(100.0, 50)]}
