import pytest

from engine.market_match import calculate_new_price
from engine.simulation_loop import MarketEnvironment


class _DummyMemoryBank:
    def calculate_memory_strength(self, _content, _persona):
        return 1.0

    def add_memory(self, **_kwargs):
        return None

    def retrieve_context(self, **_kwargs):
        return ""


class _DummyAction:
    def __init__(self, action, amount, target_price=None):
        self.action = action
        self.amount = amount
        self.target_price = target_price


class _DummyAgent:
    def __init__(self, agent_id, action, amount, target_price=100.0):
        self.agent_id = agent_id
        self.persona = object()
        self.memory_bank = _DummyMemoryBank()
        self._action = _DummyAction(action=action, amount=amount, target_price=target_price)

    async def generate_trading_decision(self, market_data, retrieved_context):
        assert "current_price" in market_data
        assert isinstance(retrieved_context, str)
        return self._action


@pytest.mark.asyncio
async def test_simulation_step_uses_isolated_matching_runner():
    # 先挂卖单，再下买单，确保在同一个离散步内可成交
    agents = [
        _DummyAgent(agent_id="seller", action="SELL", amount=100, target_price=100.0),
        _DummyAgent(agent_id="buyer", action="BUY", amount=100, target_price=100.0),
    ]
    env = MarketEnvironment(agents, use_isolated_matching=True, runner_symbol="TEST")

    try:
        report = await env.simulation_step()
        assert report["matching_mode"] == "isolated_ipc"
        assert report["trade_count"] == 1
        assert report["new_price"] == pytest.approx(100.0)
        assert report["buffered_intents"] == 0
    finally:
        env.close()


@pytest.mark.asyncio
async def test_simulation_step_falls_back_to_impact_model_when_disabled():
    agents = [
        _DummyAgent(agent_id="buyer_only", action="BUY", amount=100, target_price=100.0),
        _DummyAgent(agent_id="holder", action="HOLD", amount=0, target_price=100.0),
    ]
    env = MarketEnvironment(agents, use_isolated_matching=False, runner_symbol="TEST")

    try:
        report = await env.simulation_step()
        expected = calculate_new_price(100.0, 0.0, 100.0)
        assert report["matching_mode"] == "impact_model"
        assert report["trade_count"] == 0
        assert report["new_price"] == pytest.approx(expected)
    finally:
        env.close()

