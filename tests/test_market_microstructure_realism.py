from __future__ import annotations

from core.exchange.order_book import OrderBook
from core.market_metrics import MarketMetrics
from core.types import Order


def test_market_microstructure_metrics_and_order_book_realism_interfaces():
    book = OrderBook(symbol="TEST", prev_close=100.0)

    # Seed resting liquidity on both sides.
    book.add_order(
        Order.create(
            agent_id="bid-maker",
            symbol="TEST",
            side="buy",
            order_type="limit",
            price=99.0,
            quantity=300,
            timestamp=1.0,
        )
    )
    book.add_order(
        Order.create(
            agent_id="ask-maker",
            symbol="TEST",
            side="sell",
            order_type="limit",
            price=101.0,
            quantity=350,
            timestamp=2.0,
        )
    )

    trades = book.add_order(
        Order.create(
            agent_id="aggressive-buyer",
            symbol="TEST",
            side="buy",
            order_type="market",
            price=0.0,
            quantity=120,
            timestamp=3.0,
        )
    )

    snapshot = {
        "best_bid": book.get_best_bid(),
        "best_ask": book.get_best_ask(),
        "depth": book.get_depth(5),
        "trade_count": len(trades),
        "last_price": float(book.last_price),
    }
    trade_tape = [{"price": t.price, "quantity": t.quantity} for t in trades]
    metrics = MarketMetrics.compute(
        snapshot=snapshot,
        trade_tape=trade_tape,
        role_flows={"retail_general": 120.0, "market_maker": -120.0},
    )

    sweep = book.sweep_cost_estimate("buy", 100, levels=5)
    ofi = book.order_flow_imbalance(levels=5)
    curve = book.impact_curve_snapshot(order_sizes=[50, 120], levels=5)

    assert metrics["trade_count"] >= 1.0
    assert metrics["spread"] >= 0.0
    assert metrics["depth_bid_total"] >= 0.0
    assert metrics["depth_ask_total"] >= 0.0
    assert -1.0 <= metrics["herding_proxy"] <= 1.0

    assert sweep["requested_qty"] == 100.0
    assert 0.0 <= sweep["fill_ratio"] <= 1.0
    assert -1.0 <= ofi <= 1.0
    assert len(curve["buy"]) == 2
    assert len(curve["sell"]) == 2

