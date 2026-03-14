import pytest

from regulator_agent import (
    SimulationControllerEnvAdapter,
    RegulatoryAction,
    RegulatorAgent,
    RewardFunction,
    ToyRegulatoryMarketEnv,
    run_regulatory_optimization,
    training_summary_to_dict,
)


def test_action_space_contains_required_controls():
    actions = RegulatorAgent.build_default_action_space()
    assert len(actions) > 10
    sample = actions[0]
    assert isinstance(sample, RegulatoryAction)
    assert hasattr(sample, "stamp_tax_rate")
    assert hasattr(sample, "reserve_cut_bps")
    assert hasattr(sample, "policy_rate_cut_bps")
    assert hasattr(sample, "rumor_refute_strength")


def test_reward_function_penalizes_crash_and_rewards_stability():
    reward_fn = RewardFunction()
    env = ToyRegulatoryMarketEnv(reward_fn=reward_fn)
    obs = env.reset(seed=1)
    # 构造更糟糕状态：更低稳定度、更高崩盘损失
    bad_obs = type(obs)(
        step=obs.step,
        macro_stability=max(0.0, obs.macro_stability - 0.3),
        crash_loss_rate=min(1.0, obs.crash_loss_rate + 0.3),
        volatility=min(1.0, obs.volatility + 0.1),
        panic_level=min(1.0, obs.panic_level + 0.1),
        agent_count=obs.agent_count,
        extras={},
    )
    assert reward_fn(obs, None) > reward_fn(bad_obs, None)


def test_training_loop_returns_best_regulatory_bundle():
    env = ToyRegulatoryMarketEnv(agent_count=5000, episode_steps=32)
    agent = RegulatorAgent(seed=7)
    summary = agent.train(env=env, episodes=40, max_steps_per_episode=32)

    assert summary.episodes == 40
    assert summary.best_action is not None
    assert isinstance(summary.best_action, RegulatoryAction)
    assert len(summary.top_actions) > 0
    assert summary.q_states > 0


def test_run_regulatory_optimization_entry():
    summary = run_regulatory_optimization(episodes=20, max_steps_per_episode=16, use_toy_env=True)
    assert summary.best_action is not None
    assert summary.best_action_score == summary.top_actions[0][1]


def test_simulation_controller_adapter_with_mock_controller():
    class _MockPolicyManager:
        def __init__(self):
            self.params = {}

        def set_policy_param(self, name, param, value):
            self.params[(name, param)] = value

    class _MockPolicy:
        def __init__(self):
            self.liquidity_injection = 0.0
            self.risk_free_rate = 0.02

    class _MockMarket:
        def __init__(self):
            self.policy_manager = _MockPolicyManager()
            self.policy = _MockPolicy()
            self.panic_level = 0.5
            self.current_news = ""

    class _MockModel:
        def __init__(self):
            self.price_history = [100.0, 99.5, 100.2]
            self.agents = [1, 2, 3]

        async def async_step(self):
            self.price_history.append(self.price_history[-1] * 1.001)

    class _MockController:
        def __init__(self):
            self.market = _MockMarket()
            self.model = _MockModel()
            self.mode = "FAST"

        async def run_tick(self):
            await self.model.async_step()
            self.market.panic_level = max(0.0, self.market.panic_level - 0.01)
            return {}

    env = SimulationControllerEnvAdapter(controller_factory=_MockController, episode_steps=5)
    obs = env.reset()
    assert obs.agent_count == 3

    action = RegulatoryAction(
        stamp_tax_rate=0.0008,
        reserve_cut_bps=25,
        policy_rate_cut_bps=10,
        rumor_refute_strength=0.6,
    )
    next_obs, reward, done, _ = env.step(action)

    assert isinstance(reward, float)
    assert next_obs.step == 1
    assert done is False


def test_training_summary_serialization():
    summary = run_regulatory_optimization(episodes=8, max_steps_per_episode=8, use_toy_env=True)
    payload = training_summary_to_dict(summary)

    assert "best_action" in payload
    assert "top_actions" in payload
    assert payload["episodes"] == 8
    assert isinstance(payload["best_action"]["stamp_tax_rate"], float)


@pytest.mark.asyncio
async def test_coro_bridge_works_inside_running_event_loop():
    async def _echo() -> int:
        return 7

    result = SimulationControllerEnvAdapter._run_coro(_echo())
    assert result == 7
