from __future__ import annotations

from regulator_agent import ToyRegulatoryMarketEnv, run_regulatory_closed_loop
import regulator_agent


def test_regulator_closed_loop_outputs_ab_pareto_and_recommendation():
    result = run_regulatory_closed_loop(
        episodes=20,
        max_steps_per_episode=8,
        seed=123,
        top_k=3,
        use_toy_env=True,
    )
    assert "training_summary" in result
    assert "counterfactual_ab" in result
    assert "pareto_frontier" in result
    assert "recommendation" in result
    assert isinstance(result["pareto_frontier"], list)
    assert result["counterfactual_ab"]["baseline"]
    assert result["recommendation"]["evidence_chain"]


def test_regulator_closed_loop_reproducible_hash():
    a = run_regulatory_closed_loop(episodes=10, max_steps_per_episode=6, seed=77, top_k=2, use_toy_env=True)
    b = run_regulatory_closed_loop(episodes=10, max_steps_per_episode=6, seed=77, top_k=2, use_toy_env=True)
    assert a["reproducibility"]["config_hash"] == b["reproducibility"]["config_hash"]


def test_regulator_closed_loop_defaults_to_real_env_then_falls_back(monkeypatch):
    def _boom(**kwargs):
        def _factory():
            raise RuntimeError("real env unavailable in test")
        return _factory

    monkeypatch.setattr(regulator_agent, "build_default_real_env_factory", _boom)
    result = run_regulatory_closed_loop(
        episodes=6,
        max_steps_per_episode=4,
        seed=9,
        top_k=2,
        use_toy_env=None,
    )
    env_selection = result["reproducibility"]["env_selection"]
    assert env_selection["selected_path"] == "toy_fallback"
    assert env_selection["fallback_used"] is True
