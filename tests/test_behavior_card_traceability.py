import pytest

from agents.base_agent import MarketSnapshot
from agents.trader_agent import TraderAgent


class DummyRouter:
    pass


@pytest.mark.asyncio
async def test_behavior_card_survives_fast_path_and_is_traceable(monkeypatch):
    monkeypatch.setenv("CIVITAS_LAYERED_MEMORY_V1", "1")

    agent = TraderAgent(
        agent_id="trace_agent",
        cash_balance=100_000.0,
        portfolio={"000001": 0},
        use_llm=False,
        model_router=DummyRouter(),
        psychology_profile={"institution_type": "pension_fund"},
    )

    snapshot = MarketSnapshot(
        symbol="000001",
        last_price=10.0,
        market_trend=0.08,
        panic_level=0.10,
        policy_description="降准",
        policy_news="流动性支持",
        text_sentiment_score=0.20,
        text_policy_shock=0.40,
        text_regime_bias="easing",
        timestamp=1.0,
    )
    perceived_data = {
        "snapshot": snapshot,
        "cash": 100_000.0,
        "portfolio_value": 100_000.0,
        "pnl_pct": 0.0,
        "behavioral_state": {},
        "analyst_reports": {},
    }

    result = await agent.reason_and_act(perceived_data, "Neutral", "Market is quiet.")
    behavior_card = result["behavior_card"]

    for key in [
        "current_belief",
        "key_memories",
        "current_constraints",
        "current_risk_budget",
        "current_narrative",
    ]:
        assert key in behavior_card

    assert behavior_card["metadata"]["institution_type"] == "pension_fund"
    assert behavior_card["config_hash"]
    assert result["behavior_context"]["enabled"] is True
    initial_budget = behavior_card["current_risk_budget"]

    for _ in range(3):
        await agent.update_memory(
            result["decision"],
            {"pnl": -20_000.0, "pnl_pct": -0.20, "status": "FILLED"},
        )

    result_after_losses = await agent.reason_and_act(perceived_data, "Fearful", "panic selling")
    assert result_after_losses["behavior_card"]["current_risk_budget"] < initial_budget
    assert result_after_losses["behavior_card"]["current_belief"] != behavior_card["current_belief"]
    assert result_after_losses["behavior_card"]["policy_delay_remaining"] >= 0

