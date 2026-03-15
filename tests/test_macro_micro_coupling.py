import pytest

from engine.simulation_loop import MarketEnvironment


class _DummyAction:
    def __init__(self, action: str = "HOLD", amount: float = 0.0, target_price: float = 100.0):
        self.action = action
        self.amount = amount
        self.target_price = target_price


class _CaptureAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.persona = object()
        self.last_macro_context = None

    async def generate_trading_decision(self, market_data, retrieved_context):
        self.last_macro_context = market_data.get("macro_context")
        return _DummyAction("HOLD", 0.0, market_data.get("current_price", 100.0))


@pytest.mark.asyncio
async def test_macro_micro_coupling_stage_order_and_context_injection() -> None:
    agent = _CaptureAgent("cap_agent")
    env = MarketEnvironment([agent], use_isolated_matching=False, runner_symbol="TEST")

    try:
        env.schedule_policy_shock("流动性注入并下调印花税")
        report = await env.simulation_step()
    finally:
        env.close()

    assert report["stage_order"] == [
        "policy",
        "macro update",
        "social contagion",
        "agent cognition",
        "trading intent",
        "IPC matching",
        "metrics update",
    ]

    assert isinstance(agent.last_macro_context, dict)
    macro_state = agent.last_macro_context["macro_state"]
    for key in (
        "inflation",
        "unemployment",
        "wage_growth",
        "credit_spread",
        "liquidity_index",
        "policy_rate",
        "fiscal_stimulus",
        "sentiment_index",
    ):
        assert key in macro_state

    chain = report["policy_transmission_chain"]
    assert chain["policy"] != ""
    assert "macro_variables" in chain
    assert "social_sentiment" in chain
    assert "industry_agent" in chain
    assert "market_microstructure" in chain
