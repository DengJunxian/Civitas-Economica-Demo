from __future__ import annotations

import pytest

from core.types import ExecutionPlan, OrderSide, OrderType
from core.market_metrics import MarketMetrics
from engine.simulation_loop import MarketEnvironment
from policy.interpretation_engine import AgentBelief


class _HoldAction:
    action = "HOLD"
    amount = 0.0
    target_price = 100.0


class _HoldAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.persona = object()
        self.memory_bank = None

    async def generate_trading_decision(self, market_data, retrieved_context):
        _ = market_data, retrieved_context
        return _HoldAction()


class _PlanAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.persona = object()


def test_market_metrics_scorecard_detects_stress_on_thin_cancel_heavy_book() -> None:
    calm_snapshot = {
        "best_bid": 99.9,
        "best_ask": 100.1,
        "last_price": 100.0,
        "trade_count": 4,
        "cancel_count": 0,
        "depth": {
            "bids": [{"price": 99.9, "qty": 800}, {"price": 99.8, "qty": 500}],
            "asks": [{"price": 100.1, "qty": 780}, {"price": 100.2, "qty": 520}],
        },
        "trades": [
            {"price": 100.0, "quantity": 100},
            {"price": 100.02, "quantity": 100},
            {"price": 99.98, "quantity": 100},
            {"price": 100.01, "quantity": 100},
        ],
    }
    stress_snapshot = {
        "best_bid": 98.0,
        "best_ask": 102.0,
        "last_price": 101.5,
        "trade_count": 2,
        "cancel_count": 6,
        "depth": {
            "bids": [{"price": 98.0, "qty": 80}],
            "asks": [{"price": 102.0, "qty": 70}],
        },
        "trades": [
            {"price": 101.8, "quantity": 40},
            {"price": 101.2, "quantity": 35},
        ],
    }

    calm_metrics = MarketMetrics.compute(snapshot=calm_snapshot, trade_tape=calm_snapshot["trades"])
    stress_metrics = MarketMetrics.compute(snapshot=stress_snapshot, trade_tape=stress_snapshot["trades"])

    calm_score = MarketMetrics.scorecard(snapshot=calm_snapshot, metrics=calm_metrics)
    stress_score = MarketMetrics.scorecard(
        snapshot=stress_snapshot,
        metrics=stress_metrics,
        policy_input={"policy_text": "突发负面谣言"},
        policy_chain={
            "macro_variables": {"liquidity_index": 0.84, "sentiment_index": 0.28},
            "market_microstructure": {"buy_volume": 20.0, "sell_volume": 180.0},
        },
    )

    assert stress_metrics["cancel_to_trade_ratio"] > calm_metrics["cancel_to_trade_ratio"]
    assert stress_metrics["slippage_bps"] > calm_metrics["slippage_bps"]
    assert stress_score["execution_friction_score"] > calm_score["execution_friction_score"]
    assert stress_score["liquidity_thinness"] > calm_score["liquidity_thinness"]
    assert stress_score["flags"]["stress_cancels"] is True


@pytest.mark.asyncio
async def test_market_environment_emits_policy_realism_diagnostics() -> None:
    env = MarketEnvironment(
        [_HoldAgent("hold_agent")],
        use_isolated_matching=False,
        runner_symbol="TEST",
        enable_random_policy_events=False,
    )
    env.schedule_policy_shock("流动性注入并下调印花税")

    try:
        report = await env.simulation_step()
    finally:
        env.close()

    diagnostics = report["realism_diagnostics"]
    assert diagnostics["flags"]["has_policy"] is True
    assert diagnostics["transmission_detected"] is True
    assert diagnostics["policy_pass_through_ratio"] > 0.0
    assert "microstructure_score" in diagnostics
    assert "execution_friction_score" in diagnostics
    chain = report["transmission_chain"]
    assert chain["policy_signal"]["policy_text"]
    assert "agent_sentiment" in chain
    assert "order_flow" in chain
    assert "matching_result" in chain
    assert "index_move" in chain


@pytest.mark.asyncio
async def test_market_environment_policy_realism_is_higher_with_policy_than_without() -> None:
    policy_env = MarketEnvironment(
        [_HoldAgent("policy_agent")],
        use_isolated_matching=False,
        runner_symbol="TEST",
        enable_random_policy_events=False,
    )
    no_policy_env = MarketEnvironment(
        [_HoldAgent("baseline_agent")],
        use_isolated_matching=False,
        runner_symbol="TEST",
        enable_random_policy_events=False,
    )
    policy_env.schedule_policy_shock("流动性注入并下调印花税")

    try:
        policy_report = await policy_env.simulation_step()
        baseline_report = await no_policy_env.simulation_step()
    finally:
        policy_env.close()
        no_policy_env.close()

    assert baseline_report["realism_diagnostics"]["flags"]["has_policy"] is False
    assert baseline_report["realism_diagnostics"]["policy_pass_through_ratio"] == 0.0
    assert policy_report["realism_diagnostics"]["policy_pass_through_ratio"] > baseline_report["realism_diagnostics"]["policy_pass_through_ratio"]


def test_execution_plan_overlay_applies_belief_pass_through_to_direct_plans() -> None:
    env = MarketEnvironment([], use_isolated_matching=False, runner_symbol="TEST", enable_random_policy_events=False)
    base_plan = ExecutionPlan(
        symbol="TEST",
        agent_id="plan-agent",
        action="BUY",
        side=OrderSide.BUY,
        target_qty=1000,
        urgency=0.8,
        order_type=OrderType.IOC,
        participation_rate=0.30,
        slicing_rule="single",
        time_horizon=1,
        price=10.0,
    )
    weak_belief = AgentBelief(
        expected_return={"TEST": 0.08},
        expected_risk={"TEST": 0.70},
        liquidity_score={"TEST": 0.20},
        confidence=0.70,
        latency_bars=3,
        disagreement_tags=["lagged_policy_pass_through"],
        metadata={"effective_pass_through": 0.20},
    )

    try:
        resolved = env._resolve_execution_plan(
            agent=_PlanAgent("plan-agent"),
            action=base_plan,
            belief=weak_belief,
            market_state_payload={"symbol": "TEST", "last_price": 10.0, "prices": {"TEST": 10.0}},
        )
    finally:
        env.close()

    assert isinstance(resolved, ExecutionPlan)
    assert resolved.target_qty < 1000
    assert resolved.order_type == OrderType.LIMIT
    assert resolved.time_horizon >= 2
    assert "belief_overlay" in resolved.metadata


def test_execution_plan_overlay_skips_double_adjustment_for_belief_native_plans() -> None:
    env = MarketEnvironment([], use_isolated_matching=False, runner_symbol="TEST", enable_random_policy_events=False)
    native_plan = ExecutionPlan(
        symbol="TEST",
        agent_id="native-plan-agent",
        action="SELL",
        side=OrderSide.SELL,
        target_qty=600,
        urgency=0.9,
        order_type=OrderType.MARKET,
        participation_rate=0.40,
        slicing_rule="single",
        time_horizon=1,
        price=10.0,
        metadata={"belief_execution": {"conviction": 0.9}},
    )
    belief = AgentBelief(
        expected_return={"TEST": -0.10},
        expected_risk={"TEST": 0.80},
        liquidity_score={"TEST": 0.15},
        confidence=0.80,
        latency_bars=1,
        disagreement_tags=[],
        metadata={"effective_pass_through": 0.95},
    )

    try:
        resolved = env._resolve_execution_plan(
            agent=_PlanAgent("native-plan-agent"),
            action=native_plan,
            belief=belief,
            market_state_payload={"symbol": "TEST", "last_price": 10.0, "prices": {"TEST": 10.0}},
        )
    finally:
        env.close()

    assert isinstance(resolved, ExecutionPlan)
    assert resolved.target_qty == 600
    assert resolved.order_type == OrderType.MARKET
    assert "belief_overlay" not in resolved.metadata
