from __future__ import annotations

from datetime import datetime

import core.market_engine as market_engine_module
from core.exchange.a_share_session import call_auction_match, find_price_maximizing_volume_and_minimizing_imbalance
from core.exchange.market_kernel import MarketKernel, MarketKernelConfig
from core.market_engine import MatchingEngine
from core.types import Order, OrderSide, OrderType


def _ts(hour: int, minute: int) -> float:
    return datetime(2026, 3, 23, hour, minute).timestamp()


def _order(agent: str, side: OrderSide, price: float, qty: int, ts: float) -> Order:
    return Order.create(
        agent_id=agent,
        symbol="TEST",
        side=side,
        order_type=OrderType.LIMIT,
        price=price,
        quantity=qty,
        timestamp=ts,
    )


def _kernel(monkeypatch) -> MarketKernel:
    monkeypatch.setattr(market_engine_module, "USE_CPP_LOB", False)
    return MarketKernel(
        symbol="TEST",
        prev_close=100.0,
        matching_engine=MatchingEngine(symbol="TEST", prev_close=100.0),
        config=MarketKernelConfig(
            seed=11,
            order_latency_ticks=0,
            feature_flags={"market_kernel_v1": True, "a_share_session_v1": True, "market_rules_v1": True},
        ),
    )


def test_midday_break_queues_orders_without_matching(monkeypatch):
    kernel = _kernel(monkeypatch)
    sell = _order("seller", OrderSide.SELL, 100.0, 100, _ts(9, 30))
    assert kernel.submit_order(sell, current_timestamp=_ts(9, 30)) == []

    buy = _order("buyer", OrderSide.BUY, 101.0, 100, _ts(11, 45))
    assert kernel.submit_order(buy, current_timestamp=_ts(11, 45)) == []
    assert kernel.advance_to(_ts(12, 0)) == []
    assert kernel.get_canonical_trade_tape() == []

    trades = kernel.advance_to(_ts(13, 0))
    assert sum(trade.quantity for trade in trades) == 100
    assert kernel.get_canonical_trade_tape()[-1].phase == "continuous"


def test_opening_call_auction_max_volume_then_imbalance_rule():
    orders = [
        _order("buy_100", OrderSide.BUY, 100.0, 100, _ts(9, 16)),
        _order("buy_101", OrderSide.BUY, 101.0, 100, _ts(9, 17)),
        _order("sell_99", OrderSide.SELL, 99.0, 100, _ts(9, 16)),
        _order("sell_100", OrderSide.SELL, 100.0, 100, _ts(9, 18)),
    ]
    price, meta = find_price_maximizing_volume_and_minimizing_imbalance(orders, prev_close=100.0)
    assert price == 100.0
    assert meta["match_volume"] == 200

    trades = call_auction_match(
        orders,
        price=price,
        timestamp=_ts(9, 25),
        commission_rate=0.0,
        stamp_duty_rate=0.0,
    )
    assert sum(trade.quantity for trade in trades) == 200
    assert all(trade.price == 100.0 for trade in trades)


def test_same_price_fifo_in_call_auction():
    first = _order("seller_1", OrderSide.SELL, 100.0, 100, _ts(9, 16))
    second = _order("seller_2", OrderSide.SELL, 100.0, 100, _ts(9, 17))
    buy = _order("buyer", OrderSide.BUY, 100.0, 100, _ts(9, 18))
    trades = call_auction_match(
        [second, first, buy],
        price=100.0,
        timestamp=_ts(9, 25),
        commission_rate=0.0,
        stamp_duty_rate=0.0,
    )
    assert len(trades) == 1
    assert trades[0].seller_agent_id == "seller_1"
    assert first.is_filled
    assert second.remaining_qty == 100
