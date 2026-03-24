import pytest
from unittest.mock import AsyncMock

from agents.base_agent import MarketSnapshot
from agents.trader_agent import TraderAgent
from core.types import ExecutionPlan, Order, OrderSide, OrderType


def make_snapshot(price: float = 10.0) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="TEST",
        last_price=price,
        best_bid=price - 0.1,
        best_ask=price + 0.1,
        market_trend=0.0,
        panic_level=0.0,
        timestamp=1000.0,
        mid_price=price,
        bid_ask_spread=0.2,
    )


@pytest.mark.asyncio
async def test_legacy_mode_returns_order():
    agent = TraderAgent("legacy_agent", cash_balance=20_000)
    agent.brain.think_async = AsyncMock(
        return_value={
            "decision": {"action": "BUY", "qty": 100, "price": 10.0},
            "reasoning": "legacy",
        }
    )

    order = await agent.act(make_snapshot(), ["news"])

    assert isinstance(order, Order)
    assert order.order_type == OrderType.LIMIT
    assert order.side == OrderSide.BUY
    assert order.quantity == 100
    assert order.metadata["feature_flag_execution_plan"] is False


@pytest.mark.asyncio
async def test_execution_plan_mode_returns_plan_with_metadata():
    agent = TraderAgent(
        "plan_agent",
        cash_balance=50_000,
        execution_plan_enabled=True,
        execution_seed=17,
    )
    agent.current_trading_intent = 0.9
    agent.brain.think_async = AsyncMock(
        return_value={
            "decision": {
                "action": "BUY",
                "qty": 600,
                "price": 10.0,
                "slicing_rule": "twap-like",
                "time_horizon": 3,
                "cancel_replace_policy": "cancel-replace",
            },
            "reasoning": "plan",
        }
    )

    plan = await agent.act(make_snapshot(), ["news"])

    assert isinstance(plan, ExecutionPlan)
    assert plan.side == OrderSide.BUY
    assert plan.target_qty == 600
    assert plan.slicing_rule == "twap-like"
    assert plan.time_horizon == 3
    assert plan.config_hash
    assert plan.metadata["execution_seed"] == 17
    assert plan.metadata["snapshot_info"]["symbol"] == "TEST"


@pytest.mark.asyncio
async def test_execution_plan_hash_is_reproducible():
    agent_a = TraderAgent(
        "hash_agent",
        cash_balance=50_000,
        portfolio={"TEST": 500},
        execution_plan_enabled=True,
        execution_seed=7,
    )
    agent_b = TraderAgent(
        "hash_agent",
        cash_balance=50_000,
        portfolio={"TEST": 500},
        execution_plan_enabled=True,
        execution_seed=7,
    )

    reasoning = {
        "decision": {"action": "SELL", "qty": 300, "price": 9.5},
        "symbol": "TEST",
        "timestamp": 1000.0,
    }

    plan_a = await agent_a.decide(reasoning)
    plan_b = await agent_b.decide(reasoning)

    assert isinstance(plan_a, ExecutionPlan)
    assert isinstance(plan_b, ExecutionPlan)
    assert plan_a.config_hash == plan_b.config_hash
    assert plan_a.metadata["snapshot_info"]["last_price"] == 9.5


@pytest.mark.asyncio
async def test_intent_execution_split_feature_flag_separates_intent_from_execution():
    agent = TraderAgent(
        "split_agent",
        cash_balance=80_000,
        execution_plan_enabled=True,
        psychology_profile={"feature_flags": {"trader_intent_execution_split_v1": True}},
    )
    agent.current_trading_intent = 0.85
    agent.brain.think_async = AsyncMock(
        return_value={
            "decision": {
                "action": "BUY",
                "qty": 1200,
                "price": 10.0,
                "urgency": 0.9,
                "order_type": "market",
            },
            "reasoning": "split",
        }
    )

    plan = await agent.act(make_snapshot(), ["policy easing"])

    assert isinstance(plan, ExecutionPlan)
    assert plan.metadata["feature_flag_trader_intent_execution_split_v1"] is True
    assert plan.metadata["intent_trace"]["desired_qty"] == 1200
    assert plan.metadata["execution_trace"]["schema_version"] == "intent_execution_split_v1"
    assert plan.metadata["snapshot_info"]["persona"]["constraints"]["schema_version"] == "persona_constraints_v1"
    assert plan.target_qty <= 1200
