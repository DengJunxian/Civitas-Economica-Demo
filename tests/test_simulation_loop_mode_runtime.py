from __future__ import annotations

import pytest

from engine.simulation_loop import MarketEnvironment


class _Action:
    def __init__(self, action: str, amount: float, target_price: float) -> None:
        self.action = action
        self.amount = amount
        self.target_price = target_price


class _ModeAwareAgent:
    def __init__(self, agent_id: str, action: str, amount: float, target_price: float) -> None:
        self.agent_id = agent_id
        self.persona = object()
        self.memory_bank = None
        self.use_llm = False
        self._decision = _Action(action=action, amount=amount, target_price=target_price)

    async def generate_trading_decision(self, market_data, retrieved_context):
        assert "macro_context" in market_data
        return self._decision


@pytest.mark.asyncio
async def test_simulation_loop_reports_deep_mode_runtime_profile() -> None:
    agents = [
        _ModeAwareAgent(agent_id="seller", action="SELL", amount=50, target_price=100.0),
        _ModeAwareAgent(agent_id="buyer", action="BUY", amount=50, target_price=100.0),
    ]
    env = MarketEnvironment(
        agents,
        runner_symbol="TEST",
        simulation_mode="DEEP",
        llm_primary=True,
        deep_reasoning_pause_s=0.0,
    )
    try:
        assert agents[0].use_llm is True
        report = await env.simulation_step()
    finally:
        env.close()

    assert report["simulation_mode"] == "DEEP"
    assert report["mode_runtime"]["mode"] == "DEEP"
    assert report["thinking_stats"]["llm_primary"] is True
