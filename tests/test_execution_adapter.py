from __future__ import annotations

from agents.execution_adapter import ExecutionAdapter
from core.types import ExecutionPlan, OrderSide, OrderType


def test_execution_adapter_preserves_explicit_limit_price():
    adapter = ExecutionAdapter(default_symbol="TEST")
    plan = ExecutionPlan(
        symbol="TEST",
        agent_id="agent-1",
        action="BUY",
        side=OrderSide.BUY,
        target_qty=100,
        urgency=0.5,
        order_type=OrderType.LIMIT,
        max_slippage=0.02,
        participation_rate=0.3,
        slicing_rule="single",
        cancel_replace_policy="none",
        time_horizon=1,
        price=100.0,
        timestamp=1.0,
    )

    intents = adapter.compile(plan, market_state={"last_price": 100.0}, step=5)
    assert len(intents) == 1
    assert intents[0].price == 100.0
    assert intents[0].side == "buy"
    assert intents[0].symbol == "TEST"


def test_execution_adapter_compile_batch_supports_slicing_schedule():
    adapter = ExecutionAdapter(default_symbol="TEST")
    plan = ExecutionPlan(
        symbol="TEST",
        agent_id="agent-2",
        action="SELL",
        side=OrderSide.SELL,
        target_qty=300,
        urgency=0.7,
        order_type=OrderType.MARKET,
        max_slippage=0.01,
        participation_rate=0.2,
        slicing_rule="twap-like",
        cancel_replace_policy="none",
        time_horizon=3,
        price=100.0,
        timestamp=2.0,
    )

    intents = adapter.compile_batch([plan], market_state={"last_price": 100.0}, step=10)
    assert len(intents) == 3
    assert sum(intent.quantity for intent in intents) == 300
    assert [intent.activate_step for intent in intents] == [10, 11, 12]
    assert all(intent.side == "sell" for intent in intents)


def test_execution_adapter_legacy_hold_action_returns_none_plan():
    adapter = ExecutionAdapter(default_symbol="TEST")
    plan = adapter.plan_from_legacy_action(
        agent_id="legacy",
        symbol="TEST",
        action="HOLD",
        amount=0.0,
        target_price=100.0,
        step=1,
        market_state={"last_price": 100.0},
    )
    assert plan is None

