from datetime import datetime

import core.market_engine as market_engine_module
from core.exchange.market_kernel import MarketKernel, MarketKernelConfig
from core.market_engine import MatchingEngine
from core.types import Order, OrderSide, OrderType


def _ts(year: int, month: int, day: int, hour: int, minute: int) -> float:
    return datetime(year, month, day, hour, minute).timestamp()


def _make_order(*, agent_id: str, side: OrderSide, price: float, quantity: int, timestamp: float) -> Order:
    return Order.create(
        agent_id=agent_id,
        symbol="TEST",
        side=side,
        order_type=OrderType.LIMIT,
        price=price,
        quantity=quantity,
        timestamp=timestamp,
    )


def _make_kernel(monkeypatch, prev_close: float = 100.0) -> MarketKernel:
    monkeypatch.setattr(market_engine_module, "USE_CPP_LOB", False)
    engine = MatchingEngine(symbol="TEST", prev_close=prev_close)
    config = MarketKernelConfig(
        seed=7,
        board_lot=100,
        allow_odd_lots=True,
        enforce_board_lot=False,
        order_latency_ticks=0,
        cancel_latency_ticks=0,
        modify_latency_ticks=0,
        call_auction_start="09:15",
        call_auction_end="09:25",
        midday_break_start="11:30",
        midday_break_end="13:00",
        continuous_start="09:30",
        continuous_end="15:00",
        feature_flags={"market_kernel_v1": True},
    )
    return MarketKernel(symbol="TEST", prev_close=prev_close, matching_engine=engine, config=config)


def test_kernel_halt_resume_and_latency_defers_execution_until_release(monkeypatch):
    kernel = _make_kernel(monkeypatch)
    t0 = _ts(2026, 3, 23, 9, 30)

    sell = _make_order(agent_id="seller", side=OrderSide.SELL, price=100.0, quantity=10, timestamp=t0)
    assert kernel.submit_order(sell, current_timestamp=t0) == []

    kernel.inject_halt(timestamp=_ts(2026, 3, 23, 9, 31), reason="halt test")
    kernel.advance_to(_ts(2026, 3, 23, 9, 31))

    buy = _make_order(agent_id="buyer", side=OrderSide.BUY, price=101.0, quantity=10, timestamp=t0 + 60)
    assert kernel.submit_order(buy, current_timestamp=t0 + 60, latency_ticks=2) == []
    assert kernel.flush_step_trade_tape() == []

    kernel.inject_resume(timestamp=_ts(2026, 3, 23, 9, 33), reason="resume test")
    trades = kernel.advance_to(_ts(2026, 3, 23, 9, 35))

    assert sum(trade.quantity for trade in trades) == 10
    tape = kernel.get_trade_tape()
    assert tape
    assert tape[-1].latency_ticks == 2
    assert tape[-1].queue_position >= 0
    assert tape[-1].phase in {"continuous", "call_auction"}


def test_kernel_cancel_and_modify_events_follow_event_queue_semantics(monkeypatch):
    kernel = _make_kernel(monkeypatch)
    t0 = _ts(2026, 3, 23, 9, 30)

    resting_buy = _make_order(agent_id="buyer", side=OrderSide.BUY, price=99.0, quantity=10, timestamp=t0)
    assert kernel.submit_order(resting_buy, current_timestamp=t0) == []

    assert kernel.cancel_order(resting_buy.order_id, current_timestamp=t0 + 10) == []
    kernel.advance_to(t0 + 10)

    late_sell = _make_order(agent_id="seller", side=OrderSide.SELL, price=99.0, quantity=10, timestamp=t0 + 20)
    assert kernel.submit_order(late_sell, current_timestamp=t0 + 20) == []
    assert kernel.advance_to(t0 + 20) == []

    matching_engine = kernel.engine
    assert matching_engine.last_price == 100.0

    kernel2 = _make_kernel(monkeypatch)
    resting_buy_2 = _make_order(agent_id="buyer2", side=OrderSide.BUY, price=99.0, quantity=10, timestamp=t0 + 40)
    assert kernel2.submit_order(resting_buy_2, current_timestamp=t0 + 40) == []
    kernel2.advance_to(t0 + 40)

    sell_2 = _make_order(agent_id="seller2", side=OrderSide.SELL, price=100.0, quantity=10, timestamp=t0 + 41)
    assert kernel2.submit_order(sell_2, current_timestamp=t0 + 41) == []

    modified = kernel2.modify_order(
        resting_buy_2.order_id,
        price=101.0,
        quantity=10,
        current_timestamp=t0 + 50,
    )
    assert sum(trade.quantity for trade in modified) == 10
    trades = kernel2.advance_to(t0 + 50)
    assert trades == []
    assert kernel2.get_trade_tape()[-1].event_type == "arrival"


def test_kernel_call_auction_flushes_into_trade_tape(monkeypatch):
    kernel = _make_kernel(monkeypatch)
    t_call = _ts(2026, 3, 23, 9, 20)
    t_open = _ts(2026, 3, 23, 9, 30)

    buy = _make_order(agent_id="buyer", side=OrderSide.BUY, price=100.0, quantity=10, timestamp=t_call)
    sell = _make_order(agent_id="seller", side=OrderSide.SELL, price=100.0, quantity=10, timestamp=t_call + 1)

    assert kernel.submit_order(buy, current_timestamp=t_call) == []
    assert kernel.submit_order(sell, current_timestamp=t_call + 1) == []
    assert kernel.get_trade_tape() == []

    trades = kernel.advance_to(t_open)
    assert trades
    assert sum(trade.quantity for trade in trades) == 10

    tape = kernel.get_trade_tape()
    assert tape
    assert any(entry.event_type == "auction_flush" for entry in tape)
    assert all(entry.phase == "call_auction" for entry in tape if entry.event_type == "auction_flush")
