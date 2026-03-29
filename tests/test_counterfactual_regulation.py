from __future__ import annotations

from regulator_agent import run_regulatory_closed_loop


def test_counterfactual_regulation_improves_tail_risk_or_reward():
    result = run_regulatory_closed_loop(
        episodes=18,
        max_steps_per_episode=8,
        seed=11,
        top_k=3,
        use_toy_env=True,
    )

    baseline = result["counterfactual_ab"]["baseline"]
    candidates = result["counterfactual_ab"]["candidates"]
    deltas = result["counterfactual_ab"]["deltas"]
    assert candidates
    assert deltas

    best = max(candidates, key=lambda row: float(row.get("avg_reward", 0.0)))
    assert float(best["avg_reward"]) >= float(baseline.get("avg_reward", 0.0))
    assert (
        float(best.get("crash_risk", 1.0)) <= float(baseline.get("crash_risk", 1.0))
        or float(best.get("macro_stability", 0.0)) >= float(baseline.get("macro_stability", 0.0))
    )

    recommendation = result.get("recommendation", {})
    assert isinstance(recommendation.get("evidence_chain", []), list)
    assert recommendation.get("scorecard")

