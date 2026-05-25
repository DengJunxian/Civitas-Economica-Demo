from pathlib import Path

from agents.persona import Persona, PersonaGenerator
from agents.population import StratifiedPopulation


def test_market_distribution_report_includes_reproducibility_metadata(tmp_path: Path):
    personas = [
        Persona.from_archetype("state_stabilization_fund", name="SSF_001"),
        Persona.from_archetype("pension_fund", name="PF_001"),
        Persona.from_archetype("mutual_fund", name="MF_001"),
    ]
    report = PersonaGenerator.build_market_distribution_report(
        personas,
        regime="stabilization",
        seed=11,
        composition_path=Path("data") / "market_composition.yaml",
    )

    assert report["feature_flag"] is True
    assert report["seed"] == 11
    assert report["counts"]["state_stabilization_fund"] == 1
    assert report["counts"]["pension_fund"] == 1
    assert report["counts"]["mutual_fund"] == 1
    assert report["snapshot_id"]
    assert report["config_hash"]
    assert "Average constraints" in report["markdown"]


def test_population_wraps_market_distribution_report():
    pop = StratifiedPopulation(n_smart=0, n_vectorized=8, market_regime="bull")
    report = pop.build_market_distribution_report([
        Persona.from_archetype("retail_day_trader", name="RDT_001"),
        Persona.from_archetype("retail_swing", name="RS_001"),
    ])

    assert report["regime"] == "bull"
    assert report["counts"]["retail_day_trader"] == 1
    assert report["counts"]["retail_swing"] == 1
    assert pop.market_distribution_report["config_hash"] == report["config_hash"]
    assert "retail_day_trader" in pop.render_market_distribution_report()
