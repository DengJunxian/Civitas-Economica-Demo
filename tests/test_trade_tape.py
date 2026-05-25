from __future__ import annotations

from datetime import datetime

from core.exchange.trade_tape import TradeTape, aggregate_trade_tape_to_bars
from core.types import Trade


def _trade(trade_id: str, price: float, qty: int, ts: float) -> Trade:
    return Trade(
        trade_id=trade_id,
        price=price,
        quantity=qty,
        maker_id=f"maker_{trade_id}",
        taker_id=f"taker_{trade_id}",
        maker_agent_id=f"maker_agent_{trade_id}",
        taker_agent_id=f"taker_agent_{trade_id}",
        buyer_agent_id=f"buyer_{trade_id}",
        seller_agent_id=f"seller_{trade_id}",
        timestamp=ts,
    )


def test_trade_tape_seeded_replay_is_deterministic():
    ts = datetime(2026, 3, 23, 9, 30).timestamp()
    tape_a = TradeTape(symbol="TEST", seed=123, config_hash="cfg", data_snapshot_hash="snap")
    tape_b = TradeTape(symbol="TEST", seed=123, config_hash="cfg", data_snapshot_hash="snap")
    for tape in (tape_a, tape_b):
        tape.append_trade(_trade("engine_random_a", 100.0, 100, ts), tick=1, trading_day="2026-03-23", phase="continuous")
        tape.append_trade(_trade("engine_random_b", 101.0, 200, ts + 30), tick=2, trading_day="2026-03-23", phase="continuous")

    assert tape_a.to_dicts() == tape_b.to_dicts()
    assert tape_a.hash() == tape_b.hash()
    assert tape_a.records[0].trade_id.startswith("tt_")


def test_ohlcv_bars_are_aggregated_from_tape_only_without_empty_buckets():
    base = datetime(2026, 3, 23, 9, 30).timestamp()
    tape = TradeTape(symbol="TEST", seed=1)
    tape.append_trade(_trade("t1", 100.0, 100, base), tick=1, trading_day="2026-03-23", phase="continuous")
    tape.append_trade(_trade("t2", 102.0, 100, base + 20), tick=2, trading_day="2026-03-23", phase="continuous")
    tape.append_trade(_trade("t3", 101.0, 50, base + 360), tick=3, trading_day="2026-03-23", phase="continuous")

    one_min = aggregate_trade_tape_to_bars(tape.records, freq="1m", symbol="TEST")
    five_min = aggregate_trade_tape_to_bars(tape.records, freq="5m", symbol="TEST")
    one_day = aggregate_trade_tape_to_bars(tape.records, freq="1d", symbol="TEST")

    assert len(one_min) == 2
    assert one_min[0].open == 100.0
    assert one_min[0].high == 102.0
    assert one_min[0].low == 100.0
    assert one_min[0].close == 102.0
    assert one_min[0].volume == 200
    assert getattr(one_min[0], "metadata")["source"] == "trade_tape"
    assert len(five_min) == 2
    assert len(one_day) == 1
    assert one_day[0].volume == 250
