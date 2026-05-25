import json
import uuid
from pathlib import Path

import pandas as pd
import pytest

from core.exchange.evolution import EvolutionOperators, StrategyGenome
from core.regulatory_sandbox import MarketAbuseSandbox
from engine.simulation_loop import MarketEnvironment
from simulation_runner import BufferedIntent, SimulationRunner


class _Action:
    def __init__(self, action: str, amount: float, target_price: float = 100.0):
        self.action = action
        self.amount = amount
        self.target_price = target_price


class _DummyAgent:
    def __init__(self, agent_id: str, action: str = "HOLD", amount: float = 0.0):
        self.agent_id = agent_id
        self.persona = type("PersonaStub", (), {"risk_tolerance": 0.5})()
        self.memory_bank = None
        self._action = _Action(action, amount)

    async def generate_trading_decision(self, market_data, retrieved_context):
        _ = market_data, retrieved_context
        return self._action


def test_strategy_genome_operators_cover_selection_crossover_mutation_diffusion() -> None:
    ops = EvolutionOperators(mutation_rate=0.4, mutation_scale=0.2, seed=1)
    g1 = StrategyGenome.random()
    g2 = StrategyGenome.random()

    child = ops.crossover(g1, g2)
    mutated = ops.mutation(child)

    selected = ops.selection({"a": 0.2, "b": 0.1, "c": -0.3}, survival_rate=0.5)
    diffused = ops.local_diffusion(
        {"a": g1, "b": g2},
        {"a": ["b"], "b": ["a"]},
        strength=0.1,
    )

    assert len(selected) >= 1
    assert 0.0 <= mutated.order_aggressiveness <= 1.0
    assert -0.5 <= mutated.stop_loss_threshold <= -0.005
    assert set(diffused.keys()) == {"a", "b"}


def test_abuse_sandbox_detects_spoofing_sentiment_and_cancellation() -> None:
    sandbox = MarketAbuseSandbox(
        spoofing_size_threshold=1000,
        spoofing_max_lifetime=3,
        sentiment_burst_threshold=0.9,
        cancellation_ratio_threshold=0.6,
        lookback_window=5,
    )

    for idx in range(5):
        oid = f"order_{idx}"
        sandbox.register_submission(
            agent_id="spoofing_agent",
            order_id=oid,
            side="buy",
            qty=1500,
            price=100.0,
            tick=1,
            tag="spoof_layer",
        )
        if idx < 4:
            sandbox.register_cancellation(
                agent_id="spoofing_agent",
                target_order_id=oid,
                tick=2,
                successful=True,
            )

    sandbox.register_sentiment(agent_id="rumor_manipulator_agent", sentiment_delta=0.5, tick=2, source="rumor")
    sandbox.register_sentiment(agent_id="rumor_manipulator_agent", sentiment_delta=-0.45, tick=2, source="rumor")
    result = sandbox.detect(2)
    event_types = {item["type"] for item in result["events"]}

    assert "spoofing_like_pattern" in event_types
    assert "sentiment_manipulation_burst" in event_types
    assert "abnormal_order_cancellation" in event_types


@pytest.mark.asyncio
async def test_hybrid_replay_abuse_exports_and_ecology_metrics(tmp_path: Path) -> None:
    agents = [
        _DummyAgent("buyer", "BUY", 120.0),
        _DummyAgent("seller", "SELL", 100.0),
    ]
    env = MarketEnvironment(
        agents,
        use_isolated_matching=False,
        runner_symbol="TEST",
        hybrid_replay=True,
        exogenous_backdrop=[
            {"step": 1, "price": 101.0, "volume": 1_200_000},
            {"step": 2, "price": 100.5, "volume": 1_000_000},
            {"step": 3, "price": 102.0, "volume": 1_500_000},
        ],
        hybrid_backdrop_weight=0.4,
        enable_abuse_agents=True,
    )
    env.stylized_facts_report_path = tmp_path / "stylized_facts_report.json"
    env.ecology_metrics_path = tmp_path / "ecology_metrics.csv"
    env.market_abuse_report_path = tmp_path / "market_abuse_report.json"
    env.intervention_effect_report_path = tmp_path / "intervention_effect_report.json"

    try:
        report = await env.simulation_step()
        await env.simulation_step()
    finally:
        env.close()

    assert report["hybrid_replay"]["enabled"] is True
    assert report["hybrid_replay"]["point"] is not None
    assert Path(report["ecology_metrics_path"]).exists()
    assert Path(report["market_abuse_report_path"]).exists()
    assert Path(report["intervention_effect_report_path"]).exists()

    ecology = pd.read_csv(tmp_path / "ecology_metrics.csv")
    assert {"entropy", "hhi", "modularity", "phase_changes", "coalition_persistence"} <= set(ecology.columns)

    abuse = json.loads((tmp_path / "market_abuse_report.json").read_text(encoding="utf-8"))
    assert "events" in abuse


def test_parameter_sensitivity_scan_outputs_csv(tmp_path: Path) -> None:
    env = MarketEnvironment([_DummyAgent("scan_agent")], use_isolated_matching=False, runner_symbol="TEST")
    try:
        out = env.run_parameter_sensitivity_scan(output_csv=tmp_path / "parameter_sensitivity.csv")
    finally:
        env.close()

    assert out.exists()
    df = pd.read_csv(out)
    assert {"loss_aversion", "reference_adaptivity", "edge_weight", "avg_risk_appetite"} <= set(df.columns)
    assert len(df) >= 3


def test_simulation_runner_supports_cancel_intents() -> None:
    runner = SimulationRunner(response_timeout=2.0, prev_close=100.0, symbol="TEST")
    runner.start()
    try:
        order_id = str(uuid.uuid4())
        submit = BufferedIntent(
            intent_id=order_id,
            agent_id="spoofing_agent",
            side="buy",
            quantity=1000,
            price=99.0,
            symbol="TEST",
            intent_type="order",
            activate_step=1,
        )
        cancel = BufferedIntent(
            intent_id=str(uuid.uuid4()),
            agent_id="spoofing_agent",
            side="cancel",
            quantity=0,
            price=0.0,
            symbol="TEST",
            intent_type="cancel",
            activate_step=2,
            metadata={"target_order_id": order_id},
        )
        runner.submit_intent(submit)
        runner.submit_intent(cancel)
        first = runner.advance_time(1)
        second = runner.advance_time(1)
        assert first["current_step"] == 1
        assert second["cancel_count"] >= 1
        assert second["activity_stats"]["cancelled_orders"] >= 1
    finally:
        runner.stop()
