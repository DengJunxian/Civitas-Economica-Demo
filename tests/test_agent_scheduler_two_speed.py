from __future__ import annotations

import pytest

from agents.fast_agent_kernel import FastAgentKernel
from engine.agent_scheduler import AgentScheduler, AgentSchedulerConfig
from engine.simulation_loop import MarketEnvironment


class _Persona:
    def __init__(self, archetype_key: str = "retail_general", risk_tolerance: float = 0.5) -> None:
        self.archetype_key = archetype_key
        self.risk_tolerance = risk_tolerance


class _Agent:
    def __init__(self, agent_id: str, archetype_key: str = "retail_general", *, use_llm: bool = False) -> None:
        self.agent_id = agent_id
        self.persona = _Persona(archetype_key)
        self.psychology_profile = {"institution_type": archetype_key}
        self.use_llm = use_llm
        self.portfolio = {"A_SHARE_IDX": 100}


@pytest.mark.asyncio
async def test_fast_slow_agent_scheduler_outputs_consistent_intents():
    agents = [
        _Agent("strategic", "mutual_fund", use_llm=True),
        *[_Agent(f"retail_{idx}", "retail_momentum_chaser") for idx in range(7)],
    ]
    scheduler = AgentScheduler(config=AgentSchedulerConfig(max_slow_agents=2, small_world_slow_threshold=2))

    async def process_agent(agent):
        return {"action": "BUY", "amount": 100.0, "target_price": 100.0, "agent_id": agent.agent_id}

    result = await scheduler.collect_actions(
        agents,
        process_agent=process_agent,
        tick=1,
        current_price=100.0,
        runner_symbol="A_SHARE_IDX",
        event_digest={"active_events": [{"event_id": "news_1"}], "aggregate_impact": {"sentiment_impact": 0.20}},
        macro_state={"sentiment_index": 0.68, "regime": "risk_on"},
    )

    assert len(result.actions) == len(agents)
    assert result.stats["architecture"] == "two_speed_v1"
    assert result.stats["slow_agent_count"] >= 1
    assert result.stats["fast_agent_count"] >= 1
    assert result.stats["batch_count"] == 1
    assert result.stats["cohort_count"] >= 1


def test_belief_cache_reuses_identical_event_archetype_pairs():
    agents = [_Agent(f"retail_{idx}", "retail_general") for idx in range(3)]
    env = MarketEnvironment(agents, runner_symbol="A_SHARE_IDX")
    try:
        package = env.government.compile_policy_package("降准释放流动性并稳定市场预期", tick=1, intensity=1.0)
        env._last_policy_package = package
        beliefs = env._batch_interpret_beliefs_cached(
            market_state_payload={"tick": 1, "regime": "neutral", "symbols": ["A_SHARE_IDX"]},
            event_digest=env.runtime_event_queue.digest_for_time(1),
        )

        assert len(beliefs) == len(agents)
        assert env._last_belief_cache_stats["belief_cache_miss_count"] == 1
        assert env._last_belief_cache_stats["belief_cache_hit_count"] == 2
        assert env._last_belief_cache_stats["belief_cache_hit_rate"] > 0
    finally:
        env.close()


def test_cohort_fast_agents_change_order_flow_under_news_shock():
    agents = [_Agent(f"retail_{idx}", "retail_momentum_chaser") for idx in range(10)]
    kernel = FastAgentKernel()

    calm_actions, _ = kernel.decide(
        agents,
        current_price=100.0,
        event_digest={"aggregate_impact": {"sentiment_impact": 0.0}, "by_type": {}},
        macro_state={"sentiment_index": 0.50},
        tick=1,
    )
    shock_actions, shock_cohorts = kernel.decide(
        agents,
        current_price=100.0,
        event_digest={
            "aggregate_impact": {"sentiment_impact": -0.30, "funding_stress": 0.15},
            "by_type": {"rumor": [{"event_id": "rumor_1"}]},
        },
        macro_state={"sentiment_index": 0.36},
        tick=2,
    )

    calm_sell = sum(1 for item in calm_actions if item.action == "SELL")
    shock_sell = sum(1 for item in shock_actions if item.action == "SELL")
    assert shock_sell > calm_sell
    assert any(cohort.action == "SELL" for cohort in shock_cohorts)

