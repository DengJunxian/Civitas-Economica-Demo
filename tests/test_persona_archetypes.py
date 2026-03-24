import json
from pathlib import Path

from agents.persona import InvestmentHorizon, Persona, PersonaGenerator, RiskAppetite, get_archetype, list_archetype_keys


def test_archetype_registry_covers_expected_roles():
    keys = set(list_archetype_keys())
    expected = {
        "retail_day_trader",
        "retail_swing",
        "mutual_fund",
        "pension_fund",
        "insurer",
        "prop_desk",
        "market_maker",
        "state_stabilization_fund",
        "rumor_trader",
        "etf_arbitrageur",
    }
    assert expected.issubset(keys)
    archetype = get_archetype("mutual_fund")
    assert archetype.mandate
    assert archetype.benchmark
    assert archetype.turnover_target > 0


def test_persona_exposes_archetype_and_mutable_state_layers():
    persona = Persona.from_archetype("retail_day_trader", name="Day_001")
    data = persona.to_dict()

    assert persona.archetype.key == "retail_day_trader"
    assert persona.mutable_state is not None
    assert persona.risk_appetite == RiskAppetite.AGGRESSIVE
    assert persona.investment_horizon == InvestmentHorizon.SHORT_TERM
    assert data["archetype"]["key"] == "retail_day_trader"
    assert "mutable_state" in data
    assert persona.stable_signature()

    legacy = Persona(name="Legacy", risk_appetite=RiskAppetite.GAMBLER)
    assert legacy.archetype is not None
    assert legacy.risk_appetite == RiskAppetite.GAMBLER
    assert legacy.to_dict()["archetype"]["key"]


def test_composition_driven_generation_uses_weights(tmp_path: Path):
    composition = {
        "version": "1.0",
        "default_regime": "custom",
        "archetypes": ["market_maker", "retail_swing"],
        "regimes": {
            "custom": {
                "weights": {
                    "market_maker": 1.0,
                    "retail_swing": 0.0,
                }
            }
        },
    }
    comp_path = tmp_path / "market_composition.json"
    comp_path.write_text(json.dumps(composition, ensure_ascii=False), encoding="utf-8")

    personas = PersonaGenerator.generate_distribution(6, regime="custom", composition_path=comp_path, seed=42)
    assert {p.archetype.key for p in personas} == {"market_maker"}

    report = PersonaGenerator.build_market_distribution_report(
        personas,
        regime="custom",
        seed=42,
        composition_path=comp_path,
    )
    assert report["counts"]["market_maker"] == 6
    assert report["averages"]["holding_period_days"] > 0
    assert "market_maker" in report["markdown"]
