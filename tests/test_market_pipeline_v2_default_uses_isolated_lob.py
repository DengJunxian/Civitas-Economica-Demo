from __future__ import annotations

import pytest

from engine.simulation_loop import MarketEnvironment


class _Action:
    def __init__(self, action: str, amount: float, target_price: float) -> None:
        self.action = action
        self.amount = amount
        self.target_price = target_price


class _Agent:
    def __init__(self, agent_id: str, action: str, amount: float, target_price: float) -> None:
        self.agent_id = agent_id
        self.persona = object()
        self.memory_bank = None
        self._decision = _Action(action=action, amount=amount, target_price=target_price)

    async def generate_trading_decision(self, market_data, retrieved_context):
        assert "current_price" in market_data
        assert isinstance(retrieved_context, str)
        return self._decision


@pytest.mark.asyncio
async def test_market_pipeline_v2_defaults_to_isolated_lob_matching():
    agents = [
        _Agent(agent_id="seller", action="SELL", amount=80, target_price=100.0),
        _Agent(agent_id="buyer", action="BUY", amount=80, target_price=100.0),
    ]
    env = MarketEnvironment(agents, runner_symbol="TEST")
    try:
        assert env.market_pipeline_v2 is True
        assert env.use_isolated_matching is True
        report = await env.simulation_step()
        assert report["pipeline_version"] == "v2"
        assert report["matching_mode"] == "isolated_ipc"
        assert report["trade_count"] >= 1
    finally:
        env.close()

