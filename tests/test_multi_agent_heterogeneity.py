from __future__ import annotations

from types import SimpleNamespace

from policy.interpretation_engine import PolicyInterpretationEngine
from policy.structured import StructuredPolicyParser


def _persona(key: str) -> object:
    return SimpleNamespace(
        archetype_key=key,
        archetype=None,
        mutable_state=SimpleNamespace(policy_reaction_delay=0),
    )


def test_same_policy_produces_heterogeneous_agent_reactions():
    parser = StructuredPolicyParser(seed=21)
    pkg = parser.parse("custom neutral policy template", tick=1, policy_type_hint="unclassified")

    # Force a deterministic heterogeneity pattern for this test:
    pkg.market_effects["market_bias"] = 0.0
    pkg.sector_effects["defensive"] = 0.0
    pkg.agent_class_effects.update(
        {
            "retail_general": 0.35,
            "market_maker": -0.30,
            "long_term_institution": 0.10,
        }
    )

    engine = PolicyInterpretationEngine(default_symbols=["INDEX"])
    retail_belief = engine.interpret(pkg, _persona("retail_general"), market_state={"symbols": ["INDEX"]})
    maker_belief = engine.interpret(pkg, _persona("market_maker"), market_state={"symbols": ["INDEX"]})
    institution_belief = engine.interpret(pkg, _persona("long_term_institution"), market_state={"symbols": ["INDEX"]})

    retail_alpha = float(retail_belief.expected_return["INDEX"])
    maker_alpha = float(maker_belief.expected_return["INDEX"])
    institution_alpha = float(institution_belief.expected_return["INDEX"])

    assert retail_alpha > 0.0
    assert maker_alpha < 0.0
    assert len({round(retail_alpha, 6), round(maker_alpha, 6), round(institution_alpha, 6)}) >= 2

