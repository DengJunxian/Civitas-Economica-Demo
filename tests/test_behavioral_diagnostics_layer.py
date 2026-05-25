import json
from pathlib import Path

import numpy as np
import pytest

from core.behavioral_finance import (
    DispositionCounter,
    StylizedFactsTracker,
    behavioral_update_step,
    initialize_reference_points,
)
from core.macro.state import MacroState
from core.social.contagion import SocialContagionEngine
from core.social.graph_state import SocialGraphState
from engine.simulation_loop import MarketEnvironment


def test_reference_point_update_pipeline_changes_risk_and_intent() -> None:
    refs = initialize_reference_points(100.0, peer_anchor=99.0, policy_anchor=101.0)
    step = behavioral_update_step(
        sentiment=-0.6,
        current_price=92.0,
        reference_points=refs,
        base_risk_appetite=0.6,
        peer_anchor=95.0,
        policy_anchor=97.0,
        policy_shock=-0.5,
        loss_aversion=2.5,
    )
    assert step.reference_points.purchase_anchor > 0
    assert step.reference_points.recent_high_anchor >= step.reference_points.purchase_anchor
    assert 0.0 <= step.risk_appetite <= 1.0
    assert -1.0 <= step.trading_intent <= 1.0
    assert step.loss_aversion_intensity >= 2.5
    assert step.trading_intent < 0.0


def test_contagion_propagation_uses_multi_edge_channels() -> None:
    graph = SocialGraphState.ring(["a", "b", "c"])
    graph.set_sentiment("a", -0.8)
    graph.set_sentiment("b", 0.2)
    graph.set_sentiment("c", 0.2)
    # b is strongly tied to a, c is weakly tied to a
    graph.set_edge_profile(
        "b",
        "a",
        trust_edge=0.95,
        position_similarity_edge=0.90,
        news_exposure_edge=0.90,
        institution_affiliation_edge=1.0,
    )
    graph.set_edge_profile(
        "c",
        "a",
        trust_edge=0.10,
        position_similarity_edge=0.10,
        news_exposure_edge=0.10,
        institution_affiliation_edge=0.0,
    )

    snap = SocialContagionEngine().step(graph, MacroState(sentiment_index=0.5), rumor_shock=0.0)
    assert "trust_edge" in snap.edge_channel_means
    assert snap.node_sentiment["b"] < snap.node_sentiment["c"]


def test_metrics_calculation_and_report_dump(tmp_path: Path) -> None:
    tracker = StylizedFactsTracker()
    prices = [100.0, 102.0, 99.0, 105.0, 103.0, 108.0]
    returns = np.diff(prices) / np.array(prices[:-1], dtype=float)
    csad_series = [0.020, 0.018, 0.024, 0.017, 0.021]
    cross_rows = [
        [0.01, 0.011, 0.009],
        [0.015, 0.012, 0.010],
        [-0.020, -0.018, -0.022],
        [0.025, 0.020, 0.019],
        [-0.010, -0.008, -0.012],
    ]

    for i in range(len(returns)):
        tracker.record_step(
            price=prices[i + 1],
            market_return=float(returns[i]),
            csad=csad_series[i],
            cross_returns=cross_rows[i],
            is_all_time_high=prices[i + 1] >= max(prices[: i + 2]),
            loss_aversion_intensity=2.0 + 0.1 * i,
        )

    counter = DispositionCounter(realized_gains=4, realized_losses=1, paper_gains=2, paper_losses=3)
    tracker.disposition = counter

    report = tracker.report()
    for key in (
        "csad",
        "pgr_plr",
        "volatility_clustering",
        "drawdown_distribution",
        "all_time_high_effect",
        "volatility clustering",
        "drawdown distribution",
        "all_time_high effect",
    ):
        assert key in report

    out = tracker.save_json(tmp_path / "stylized_facts_report.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "pgr_plr" in payload and "csad" in payload


class _HoldAction:
    action = "HOLD"
    amount = 0.0
    target_price = 100.0


class _DummyAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.persona = type("PersonaStub", (), {"risk_tolerance": 0.5})()
        self.memory_bank = None

    async def generate_trading_decision(self, market_data, retrieved_context):
        _ = market_data, retrieved_context
        return _HoldAction()


@pytest.mark.asyncio
async def test_simulation_step_emits_stylized_facts_report(tmp_path: Path) -> None:
    env = MarketEnvironment([_DummyAgent("diag_agent")], use_isolated_matching=False, runner_symbol="TEST")
    env.stylized_facts_report_path = tmp_path / "stylized_facts_report.json"
    try:
        report = await env.simulation_step()
    finally:
        env.close()
    assert Path(report["stylized_facts_report_path"]).exists()
