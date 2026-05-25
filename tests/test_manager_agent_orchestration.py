import pytest

from agents.base_agent import MarketSnapshot
from agents.manager_agent import ExecutionIntent, ManagerAgent
from core.types import OrderSide


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        symbol="TEST",
        last_price=100.0,
        best_bid=99.5,
        best_ask=100.5,
        mid_price=100.0,
        bid_ask_spread=1.0,
        market_trend=0.25,
        panic_level=0.05,
        timestamp=1234.0,
    )


@pytest.mark.asyncio
async def test_manager_agent_emits_execution_intent_with_metadata():
    manager = ManagerAgent(
        "manager_001",
        seed=7,
        feature_flags={
            "manager_orchestration_v1": True,
            "manager_debate_v1": True,
            "manager_external_analysts_v1": False,
        },
    )

    reports = {
        "news": {"sentiment_score": 0.6},
        "quant": {"momentum": 0.4, "herding_intensity": 0.2},
        "risk": {"cvar": -0.01, "max_drawdown": 0.05},
    }

    intent = await manager.act(
        _snapshot(),
        ["positive headline"],
        analyst_reports=reports,
        portfolio_value=100_000.0,
    )

    assert isinstance(intent, ExecutionIntent)
    assert intent.action == "BUY"
    assert intent.side == "buy"
    assert intent.approved is True
    assert intent.target_qty > 0
    assert intent.metadata["seed"] == 7
    assert "config_hash" in intent.metadata
    assert intent.metadata["snapshot"]["symbol"] == "TEST"
    assert intent.debate["enabled"] is True

    order = intent.to_order()
    assert order is not None
    assert order.side == OrderSide.BUY


def test_manager_agent_summary_is_reproducible():
    manager = ManagerAgent("manager_002", seed=11)
    summary = manager.summary()

    assert summary["agent_id"] == "manager_002"
    assert summary["seed"] == 11
    assert summary["config_hash"]
    assert "feature_flags" in summary
