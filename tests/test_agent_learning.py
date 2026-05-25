from __future__ import annotations

import pandas as pd

from agents.fast_agent_kernel import FastAgentKernel
from agents.learning import (
    HeuristicPolicyHead,
    ImitationDataset,
    MARLEnvironment,
    PersonaPrior,
    RegimeRouter,
    kl_to_prior,
)


class _Persona:
    archetype_key = "retail_momentum_chaser"
    risk_tolerance = 0.7
    loss_aversion = 1.8
    policy_channel_sensitivity = 0.6
    rumor_sensitivity = 0.8
    turnover_target = 3.0


class _Agent:
    def __init__(self, agent_id: str, archetype_key: str = "retail_momentum_chaser") -> None:
        self.agent_id = agent_id
        self.persona = _Persona()
        self.persona.archetype_key = archetype_key
        self.psychology_profile = {"institution_type": archetype_key}
        self.use_llm = False


def test_persona_prior_is_regularizer_not_final_decision():
    prior = PersonaPrior.from_persona(_Persona(), agent_group="retail")
    head = HeuristicPolicyHead()
    route = RegimeRouter().route({"sentiment_index": 0.72, "volatility": 0.02}, agent_group="retail", persona_prior=prior)
    output = head.decide(
        {"sentiment_index": 0.72, "momentum": 0.01, "price": 100.0, "news_heat": 0.3},
        agent_group="retail",
        persona_prior=prior,
        route=route,
    )
    assert output.action in {"BUY", "HOLD", "SELL"}
    assert output.metadata["slow_model_called"] is False
    assert output.regularization["kl_to_persona_prior"] >= 0.0
    assert kl_to_prior(output.action_distribution, prior) == output.regularization["kl_to_persona_prior"]


def test_fast_agent_kernel_uses_policy_head_without_slow_model_calls():
    agents = [_Agent(f"retail_{idx}") for idx in range(5)]
    actions, cohorts = FastAgentKernel().decide(
        agents,
        current_price=100.0,
        event_digest={"aggregate_impact": {"sentiment_impact": 0.25}, "by_type": {"major_news": [{"event_id": "n"}]}},
        macro_state={"sentiment_index": 0.70, "volatility": 0.02},
        tick=1,
    )
    assert len(actions) == len(agents)
    assert cohorts
    assert all(action.metadata.get("slow_model_called") is False for action in actions)
    assert all("policy_head_distribution" in action.metadata for action in actions)
    assert any(action.action == "BUY" for action in actions)


def test_imitation_dataset_and_marl_interface_smoke():
    dataset = ImitationDataset.from_bars(
        pd.DataFrame(
            {
                "close": [100.0, 101.0, 100.5, 102.0],
                "volume": [1000, 1100, 900, 1200],
            }
        )
    )
    assert len(dataset.samples) == 3
    assert sum(dataset.action_counts().values()) == 3

    env = MARLEnvironment(["a", "b"], initial_state={"price": 100.0}, horizon=2)
    obs = env.reset(seed=3)
    assert set(obs) == {"a", "b"}
    step = env.step({"a": {"action": "BUY", "target_qty": 100}, "b": {"action": "SELL", "target_qty": 50}})
    assert set(step.rewards) == {"a", "b"}
    assert step.info["net_flow"] == 50
