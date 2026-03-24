from __future__ import annotations

from regulator_agent import run_regulatory_closed_loop


def test_regulator_closed_loop_outputs_ab_and_pareto():
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
    assert isinstance(result["pareto_frontier"], list)
    assert result["counterfactual_ab"]["baseline"]


def test_regulator_closed_loop_reproducible_hash():
    a = run_regulatory_closed_loop(episodes=10, max_steps_per_episode=6, seed=77, top_k=2, use_toy_env=True)
    b = run_regulatory_closed_loop(episodes=10, max_steps_per_episode=6, seed=77, top_k=2, use_toy_env=True)
    assert a["reproducibility"]["config_hash"] == b["reproducibility"]["config_hash"]
