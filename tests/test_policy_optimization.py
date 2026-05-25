from __future__ import annotations

import numpy as np

from core.optimization import bayesian_search, nsga_search
from regulator_agent import run_regulatory_closed_loop


PARAMETER_SPACE = {
    "stamp_tax_rate": (0.0003, 0.0010),
    "reserve_cut_bps": (0.0, 50.0),
    "policy_rate_cut_bps": (0.0, 25.0),
    "rumor_refute_strength": (0.0, 1.0),
    "stabilization_capital": (0.0, 0.8),
    "stabilization_timing": (0.0, 6.0),
}


def _cheap_policy_simulator(params):
    target = {
        "stamp_tax_rate": 0.0005,
        "reserve_cut_bps": 25.0,
        "policy_rate_cut_bps": 10.0,
        "rumor_refute_strength": 0.60,
        "stabilization_capital": 0.20,
        "stabilization_timing": 1.0,
    }
    scales = {
        "stamp_tax_rate": 0.0007,
        "reserve_cut_bps": 50.0,
        "policy_rate_cut_bps": 25.0,
        "rumor_refute_strength": 1.0,
        "stabilization_capital": 0.8,
        "stabilization_timing": 6.0,
    }
    distance = sum(abs(float(params[k]) - target[k]) / scales[k] for k in target) / len(target)
    quality = float(np.clip(1.0 - distance, 0.0, 1.0))
    return {
        "avg_reward": quality,
        "episode_reward": quality * 10.0,
        "macro_stability": 0.35 + 0.55 * quality,
        "crash_risk": 0.25 * (1.0 - quality),
        "max_drawdown": 0.25 * (1.0 - quality),
        "volatility": 0.20 * (1.0 - quality),
        "liquidity": 0.40 + 0.45 * quality,
        "intervention_cost": 0.20 + 0.25 * float(params["stabilization_capital"]),
        "welfare_confidence": 0.30 + 0.60 * quality,
        "financing_function": 0.35 + 0.55 * quality,
        "fairness_compliance": 0.30 + 0.60 * quality,
        "panic": 0.50 * (1.0 - quality),
        "market_function_score": 0.35 + 0.55 * quality,
    }


def test_bayesian_optimizer_beats_manual_rule_on_fixed_window():
    rule_params = {
        "stamp_tax_rate": 0.0010,
        "reserve_cut_bps": 0.0,
        "policy_rate_cut_bps": 0.0,
        "rumor_refute_strength": 0.2,
        "stabilization_capital": 0.0,
        "stabilization_timing": 6.0,
    }
    baseline_score = _cheap_policy_simulator(rule_params)["avg_reward"]
    result = bayesian_search(
        simulator=_cheap_policy_simulator,
        parameter_space=PARAMETER_SPACE,
        n_iter=12,
        seed=5,
        initial_params=[rule_params],
    )
    assert result.best
    assert result.best["metrics"]["avg_reward"] >= baseline_score
    assert result.pareto_frontier


def test_nsga_search_outputs_pareto_frontier_and_constraints():
    result = nsga_search(
        simulator=_cheap_policy_simulator,
        parameter_space=PARAMETER_SPACE,
        population_size=8,
        generations=2,
        seed=4,
    )
    payload = result.to_dict()
    assert payload["pareto_frontier"]
    assert payload["best"]["objectives"]["financial_stability_score"] >= 0.0
    assert "violations" in payload["best"]["constraints"]


def test_regulator_closed_loop_reports_q_bo_nsga_and_rule_baseline():
    result = run_regulatory_closed_loop(
        episodes=8,
        max_steps_per_episode=4,
        seed=13,
        top_k=2,
        use_toy_env=True,
    )
    report = result["optimization_report"]
    methods = report["method_outputs"]
    assert methods["q_learning"]
    assert methods["bayesian_optimization"]["evaluations"]
    assert methods["nsga_ii"]["pareto_frontier"]
    assert methods["rule_baseline"]
    assert "constraint_violations" in report
    assert "final_recommendation_text" in report

    validation = result["blackbox_optimization"]["validation"]
    if validation["stable_win_count"] < validation["required_windows"]:
        assert result["default_production_path"] != "bayesian_blackbox"
