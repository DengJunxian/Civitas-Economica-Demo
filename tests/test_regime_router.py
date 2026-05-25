from __future__ import annotations

from agents.learning import PersonaPrior, RegimeRouter


def test_regime_router_changes_parameters_across_market_states():
    router = RegimeRouter()
    prior = PersonaPrior.from_persona(agent_group="retail")
    risk_on = router.route({"sentiment_index": 0.70, "volatility": 0.02, "drawdown": 0.0}, agent_group="retail", persona_prior=prior)
    stress = router.route({"sentiment_index": 0.30, "volatility": 0.08, "drawdown": -0.10}, agent_group="retail", persona_prior=prior)

    assert risk_on.regime == "risk_on"
    assert stress.regime == "stress"
    assert risk_on.leverage_multiplier > stress.leverage_multiplier
    assert risk_on.risk_budget > stress.risk_budget


def test_policy_capital_routes_to_stabilizing_style_under_stress():
    router = RegimeRouter()
    route = router.route(
        {"sentiment_index": 0.25, "volatility": 0.09, "drawdown": -0.12},
        agent_group="policy_capital",
    )
    assert route.regime == "stress"
    assert route.execution_style == "stabilizing"
    assert route.slow_thinking_allowed is True
    assert route.policy_sensitivity > 0.7
