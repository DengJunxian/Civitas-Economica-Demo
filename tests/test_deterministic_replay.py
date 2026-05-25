from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from core.experiment_registry import (
    append_experiment_record,
    attach_experiment_to_report,
    create_experiment_record,
    create_experiment_record_from_hashes,
    load_experiment_registry,
)
from core.exchange.bar_builder import TradeTapeBarBuilder
from core.exchange.trade_tape import TradeTape
from core.reproducibility import replay_signature
from core.types import Trade


def _build_tape(seed: int) -> TradeTape:
    tape = TradeTape(symbol="sh000001", seed=seed, config_hash="cfg", data_snapshot_hash="snap")
    trades = [
        Trade(trade_id="raw-1", price=3000.0, quantity=100, maker_id="s1", taker_id="b1", maker_agent_id="seller", taker_agent_id="buyer", buyer_agent_id="buyer", seller_agent_id="seller", timestamp=1.0),
        Trade(trade_id="raw-2", price=3002.5, quantity=120, maker_id="s2", taker_id="b2", maker_agent_id="seller", taker_agent_id="buyer", buyer_agent_id="buyer", seller_agent_id="seller", timestamp=2.0),
        Trade(trade_id="raw-3", price=2998.0, quantity=80, maker_id="s3", taker_id="b3", maker_agent_id="seller", taker_agent_id="buyer", buyer_agent_id="buyer", seller_agent_id="seller", timestamp=61.0),
    ]
    for idx, trade in enumerate(trades, start=1):
        tape.append_trade(
            trade,
            tick=idx,
            trading_day="2026-01-02",
            phase="continuous",
            market_timestamp=trade.timestamp,
            metadata={"buy_order_id": f"b{idx}", "sell_order_id": f"s{idx}", "aggressor_side": "buy"},
        )
    return tape


def test_trade_tape_replay_signature_is_deterministic_for_same_seed() -> None:
    first = _build_tape(42)
    second = _build_tape(42)
    builder = TradeTapeBarBuilder(seed=42, config_hash="cfg", snapshot_info={"snapshot": "snap"})

    first_bars = builder.build_bars_from_canonical_tape(first.records, symbol="sh000001", prev_close=2999.0)
    second_bars = builder.build_bars_from_canonical_tape(second.records, symbol="sh000001", prev_close=2999.0)

    assert first.hash() == second.hash()
    assert replay_signature([asdict(bar) for bar in first_bars]) == replay_signature([asdict(bar) for bar in second_bars])


def test_trade_tape_replay_signature_changes_when_seed_changes() -> None:
    assert _build_tape(42).hash() != _build_tape(43).hash()


def test_experiment_registry_roundtrip_and_report_experiment_id(tmp_path: Path) -> None:
    record = create_experiment_record(
        module="deterministic_replay",
        scenario_name="unit replay",
        config={"mode": "SMART"},
        dataset={"snapshot": "snap", "rows": 3},
        parameters={"risk": 0.5},
        seed=42,
        benchmark_symbol="sh000001",
        metrics={"path_rmse": 0.01},
    )
    registry_path = append_experiment_record(record, tmp_path / "registry.jsonl")
    loaded = load_experiment_registry(registry_path)
    report = attach_experiment_to_report({"title": "unit report"}, experiment_id=record.experiment_id)

    assert loaded[0].experiment_id == record.experiment_id
    assert loaded[0].config_hash == record.config_hash
    assert loaded[0].data_snapshot_hash == record.data_snapshot_hash
    assert loaded[0].parameter_set_id == record.parameter_set_id
    assert report["report_experiment_id"] == record.experiment_id


def test_registry_record_from_hashes_is_replay_stable() -> None:
    left = create_experiment_record_from_hashes(
        module="policy_lab",
        config_hash="cfg",
        data_snapshot_hash="snap",
        parameter_set_id="pset",
        seed=7,
    )
    right = create_experiment_record_from_hashes(
        module="policy_lab",
        config_hash="cfg",
        data_snapshot_hash="snap",
        parameter_set_id="pset",
        seed=7,
    )

    assert left.experiment_id == right.experiment_id
