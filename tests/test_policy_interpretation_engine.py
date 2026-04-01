from __future__ import annotations

from types import SimpleNamespace

from agents.persona import Persona
from policy.interpretation_engine import PolicyInterpretationEngine
from policy.structured import StructuredPolicyParser


class _Agent:
    def __init__(self, persona: object, portfolio: dict | None = None) -> None:
        self.persona = persona
        self.portfolio = portfolio or {}


def test_policy_interpretation_engine_returns_well_formed_belief():
    parser = StructuredPolicyParser(seed=7)
    pkg = parser.parse("rate cut with liquidity easing support", tick=1)
    engine = PolicyInterpretationEngine(default_symbols=["INDEX"])

    persona = Persona.from_archetype("retail_day_trader")
    belief = engine.interpret(
        pkg,
        persona,
        market_state={"symbols": ["INDEX"], "prices": {"INDEX": 100.0}, "last_price": 100.0},
    )

    assert "INDEX" in belief.expected_return
    assert "INDEX" in belief.expected_risk
    assert "INDEX" in belief.liquidity_score
    assert 0.0 < belief.confidence <= 1.0
    assert belief.latency_bars >= 0
    assert isinstance(belief.disagreement_tags, list)
    assert "persona_key" in belief.metadata


def test_policy_interpretation_engine_batch_interpret_aligns_with_agents():
    parser = StructuredPolicyParser(seed=3)
    pkg = parser.parse("fiscal stimulus for growth sectors", tick=2)
    engine = PolicyInterpretationEngine(default_symbols=["INDEX"])

    retail = Persona.from_archetype("retail_day_trader")
    maker = Persona.from_archetype("market_maker")
    agents = [
        _Agent(retail, portfolio={"INDEX": 10}),
        _Agent(maker, portfolio={"INDEX": 5}),
    ]
    beliefs = engine.batch_interpret(
        pkg,
        agents,
        market_state={"symbols": ["INDEX"], "prices": {"INDEX": 120.0}, "last_price": 120.0},
    )

    assert len(beliefs) == 2
    assert beliefs[0].metadata.get("persona_key") != ""
    assert beliefs[1].metadata.get("persona_key") != ""
    # Two personas should not collapse to the exact same belief payload.
    assert beliefs[0].expected_return != beliefs[1].expected_return


def test_policy_interpretation_engine_applies_lagged_pass_through_over_time():
    parser = StructuredPolicyParser(seed=11)
    pkg = parser.parse("rate cut with liquidity easing support", tick=2)
    engine = PolicyInterpretationEngine(default_symbols=["INDEX"])
    persona = Persona.from_archetype("retail_day_trader")

    early = engine.interpret(
        pkg,
        persona,
        market_state={"tick": 2, "symbols": ["INDEX"], "prices": {"INDEX": 100.0}, "last_price": 100.0},
    )
    later = engine.interpret(
        pkg,
        persona,
        market_state={"tick": 8, "symbols": ["INDEX"], "prices": {"INDEX": 100.0}, "last_price": 100.0},
    )

    assert later.metadata["bars_since_policy"] > early.metadata["bars_since_policy"]
    assert later.metadata["effective_pass_through"] >= early.metadata["effective_pass_through"]
    assert abs(later.expected_return["INDEX"]) >= abs(early.expected_return["INDEX"])


def test_policy_interpretation_engine_channel_weights_vary_by_persona():
    parser = StructuredPolicyParser(seed=13)
    pkg = parser.parse("rate cut with liquidity easing support", tick=1)
    engine = PolicyInterpretationEngine(default_symbols=["INDEX"])

    rumor_sensitive = Persona.from_archetype("retail_day_trader")
    policy_sensitive = Persona.from_archetype("policy_capital")

    rumor_belief = engine.interpret(
        pkg,
        rumor_sensitive,
        market_state={"tick": 4, "symbols": ["INDEX"], "prices": {"INDEX": 100.0}, "last_price": 100.0},
    )
    policy_belief = engine.interpret(
        pkg,
        policy_sensitive,
        market_state={"tick": 4, "symbols": ["INDEX"], "prices": {"INDEX": 100.0}, "last_price": 100.0},
    )

    rumor_weight = rumor_belief.metadata["channel_weights"].get("liquidity", 0.0)
    policy_weight = policy_belief.metadata["channel_weights"].get("liquidity", 0.0)

    assert rumor_weight != policy_weight
