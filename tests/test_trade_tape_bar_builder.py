from core.exchange.bar_builder import TradeTapeBarBuilder, TradeTapeEntry
from core.types import Trade


def _trade(trade_id: str, price: float, qty: int, buyer_fee: float, seller_fee: float, seller_tax: float) -> Trade:
    return Trade(
        trade_id=trade_id,
        price=price,
        quantity=qty,
        maker_id=f"maker-{trade_id}",
        taker_id=f"taker-{trade_id}",
        maker_agent_id=f"maker-agent-{trade_id}",
        taker_agent_id=f"taker-agent-{trade_id}",
        buyer_agent_id=f"buyer-agent-{trade_id}",
        seller_agent_id=f"seller-agent-{trade_id}",
        buyer_fee=buyer_fee,
        seller_fee=seller_fee,
        seller_tax=seller_tax,
    )


def test_trade_tape_builder_generates_ohlcv_bars_and_replay_metrics():
    builder = TradeTapeBarBuilder(
        seed=123,
        config_hash="cfg-hash",
        feature_flags={"market_kernel_v1": True},
        snapshot_info={"symbol": "TEST", "source": "unit-test"},
        bar_interval_seconds=60,
    )

    tape = [
        TradeTapeEntry(trade=_trade("t1", 100.0, 10, 0.5, 0.5, 1.0), tick=1, queue_position=3, latency_ticks=2, market_timestamp=1.0),
        TradeTapeEntry(trade=_trade("t2", 101.0, 20, 0.5, 0.5, 1.0), tick=1, queue_position=1, latency_ticks=1, market_timestamp=2.0),
        TradeTapeEntry(trade=_trade("t3", 99.5, 20, 0.25, 0.25, 0.5), tick=2, queue_position=7, latency_ticks=4, market_timestamp=61.0),
    ]

    bars = builder.build_bars_from_trade_tape(tape, symbol="TEST", prev_close=99.0)
    metrics = builder.build_replay_metrics(tape, bars)

    assert len(bars) == 2
    assert bars[0].open == 100.0
    assert bars[0].high == 101.0
    assert bars[0].low == 100.0
    assert bars[0].close == 101.0
    assert bars[0].volume == 30
    assert bars[0].amount == 3020.0
    assert getattr(bars[0], "metadata")["trade_count"] == 2

    assert bars[1].open == 99.5
    assert bars[1].close == 99.5
    assert bars[1].volume == 20

    assert metrics["trade_count"] == 3
    assert metrics["bar_count"] == 2
    assert metrics["total_volume"] == 50
    assert metrics["fee_total"] == 2.5
    assert metrics["tax_total"] == 2.5
    assert metrics["seed"] == 123
    assert metrics["config_hash"] == "cfg-hash"
    assert metrics["snapshot_info"]["symbol"] == "TEST"
    assert abs(metrics["average_queue_position"] - ((3 + 1 + 7) / 3)) < 1e-12
    assert abs(metrics["average_latency_ticks"] - ((2 + 1 + 4) / 3)) < 1e-12


def test_trade_tape_builder_is_reproducible_for_same_seed_and_snapshot():
    tape = [
        TradeTapeEntry(trade=_trade("t1", 10.0, 5, 0.0, 0.0, 0.0), tick=1, queue_position=0, latency_ticks=0, market_timestamp=1.0),
        TradeTapeEntry(trade=_trade("t2", 11.0, 5, 0.0, 0.0, 0.0), tick=2, queue_position=0, latency_ticks=0, market_timestamp=2.0),
    ]

    builder_a = TradeTapeBarBuilder(seed=42, config_hash="same", snapshot_info={"window": "a"})
    builder_b = TradeTapeBarBuilder(seed=42, config_hash="same", snapshot_info={"window": "a"})

    metrics_a = builder_a.build_replay_metrics(tape, builder_a.build_bars_from_trade_tape(tape, symbol="TEST", prev_close=10.0))
    metrics_b = builder_b.build_replay_metrics(tape, builder_b.build_bars_from_trade_tape(tape, symbol="TEST", prev_close=10.0))

    assert metrics_a == metrics_b
