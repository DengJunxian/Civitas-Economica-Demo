from dataclasses import dataclass

from agents.persona import InvestmentHorizon, Persona, RiskAppetite
from agents.population import AgentProtocol, PopulationEngine, StratifiedPopulation


@dataclass
class DummyTraderLikeAgent:
    agent_id: str
    cash_balance: float
    persona: Persona


def test_persona_signature_is_stable():
    persona = Persona(
        name="Alpha",
        risk_appetite=RiskAppetite.BALANCED,
        investment_horizon=InvestmentHorizon.MEDIUM_TERM,
    )
    clone = Persona(
        name="Alpha",
        risk_appetite=RiskAppetite.BALANCED,
        investment_horizon=InvestmentHorizon.MEDIUM_TERM,
    )

    assert persona.stable_signature() == clone.stable_signature()


def test_population_engine_protocol_and_snapshot_metadata():
    pop = StratifiedPopulation(
        n_smart=0,
        n_vectorized=4,
        smart_agents=[],
        seed=11,
        feature_flags={"population_protocol_v1": True},
    )

    assert isinstance(pop, PopulationEngine)
    snapshot = pop.snapshot_metadata()
    assert snapshot["seed"] == 11
    assert snapshot["config_hash"]
    assert snapshot["smart_agent_count"] == 0
    assert snapshot["compat_agent_count"] == 0

    agent = DummyTraderLikeAgent(
        agent_id="trader_like_001",
        cash_balance=1000.0,
        persona=Persona(name="Compat"),
    )
    assert isinstance(agent, AgentProtocol)

    pop.register_agent(agent)
    assert pop.get_agent_by_id("trader_like_001") is agent

    snapshot = pop.snapshot_metadata()
    assert snapshot["compat_agent_count"] == 1
    assert any(a.agent_id == "trader_like_001" for a in pop.iter_agents())
