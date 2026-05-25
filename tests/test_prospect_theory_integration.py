# file: tests/test_prospect_theory_integration.py
"""前景理论与社交模块集成冒烟测试。"""

from __future__ import annotations

from agents.cognition.utility import ConfidenceTracker, InvestorType, ProspectTheory, ProspectValue
from agents.cognition.cognitive_agent import CognitiveAgent
from agents.brain import DeepSeekBrain
from core.society.network import SentimentState, SocialGraph
from agents.trader_agent import TraderAgent


def test_prospect_theory_value_sign_and_aversion() -> None:
    pt = ProspectTheory(investor_type=InvestorType.NORMAL)
    val_loss = pt.calculate_value(-0.08)
    val_gain = pt.calculate_value(0.08)

    assert val_loss < 0
    assert val_gain > 0
    assert abs(val_loss) > abs(val_gain)


def test_prospect_theory_full_result_type() -> None:
    pt = ProspectTheory(investor_type=InvestorType.PANIC_RETAIL)
    full = pt.calculate_full(-0.05)

    assert isinstance(full, ProspectValue)
    assert full.subjective_value < 0


def test_confidence_tracker_returns_description() -> None:
    tracker = ConfidenceTracker()
    tracker.update(-0.03)
    tracker.update(0.04)

    desc = tracker.get_description()
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_cognitive_agent_reference_point_logic() -> None:
    agent = CognitiveAgent(
        "test_agent",
        InvestorType.NORMAL,
        use_local_reasoner=True,
        reference_point=10.0,
        risk_aversion_lambda=2.5,
    )

    assert agent.calculate_psychological_value(9.5) < 0
    assert agent.calculate_psychological_value(10.5) > 0
    assert agent.calculate_psychological_value(10.0) == 0


def test_brain_prompt_is_non_empty() -> None:
    brain = DeepSeekBrain(
        "retail_001",
        {
            "agent_type": "retail",
            "loss_aversion": 3.0,
            "risk_preference": "aggressive",
        },
    )
    prompt = brain._build_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_social_graph_summary_and_ratio() -> None:
    graph = SocialGraph(n_agents=20, k=4, p=0.3, seed=42)
    graph.agents[1].sentiment_state = SentimentState.BULLISH
    graph.agents[2].sentiment_state = SentimentState.BULLISH

    ratio = graph.get_bullish_ratio(0)
    summary = graph.generate_social_summary(0)

    assert isinstance(ratio, float)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_trader_social_methods() -> None:
    agent = TraderAgent("social_test_agent")

    opinion = agent.share_opinion()
    assert "agent_id" in opinion
    assert "sentiment" in opinion

    result = agent.receive_opinion(
        [
            {"sentiment": "bearish"},
            {"sentiment": "bearish"},
            {"sentiment": "bullish"},
            {"sentiment": "neutral"},
        ]
    )
    assert isinstance(result, str)

    summary = agent.get_social_summary()
    assert isinstance(summary, str)
