"""Tests for structured evidence schema validation and decision trace export."""

from __future__ import annotations

import json
from pathlib import Path

from core.validator import (
    aggregate_analyst_cards,
    coerce_analyst_card,
    export_decision_trace,
    validate_analyst_card,
)


def test_validate_analyst_card_detects_invalid_payload() -> None:
    invalid_card = {
        "thesis": "bad payload",
        "evidence": [{"type": "unknown_type", "content": "x", "weight": 1.4}],
        "time_horizon": "yearly",
        "risk_tags": "not_a_list",
        "confidence": 1.2,
        "counterarguments": "not_a_list",
        "recommended_action": "all_in",
    }
    is_valid, errors = validate_analyst_card(invalid_card)
    assert is_valid is False
    assert any(err.startswith("invalid_") for err in errors)


def test_coerce_then_validate_card_is_schema_compliant() -> None:
    raw_card = {
        "analyst_id": "alpha",
        "thesis": "policy support may stabilize sentiment",
        "evidence": [{"type": "macro", "content": "liquidity injection", "weight": 0.9}],
        "time_horizon": "weekly",
        "risk_tags": ["liquidity"],
        "confidence": 0.66,
        "counterarguments": ["transmission lag"],
        "recommended_action": "buy",
    }
    normalized = coerce_analyst_card(raw_card, analyst_id="alpha")
    is_valid, errors = validate_analyst_card(normalized)
    assert is_valid is True
    assert errors == []


def test_aggregate_analyst_cards_contains_calibration_metrics() -> None:
    cards = [
        coerce_analyst_card(
            {
                "analyst_id": "a1",
                "thesis": "bullish",
                "evidence": [{"type": "price", "content": "momentum up", "weight": 0.8}],
                "time_horizon": "intraday",
                "risk_tags": ["overcrowding"],
                "confidence": 0.72,
                "counterarguments": ["could reverse"],
                "recommended_action": "buy",
            },
            analyst_id="a1",
        ),
        coerce_analyst_card(
            {
                "analyst_id": "a2",
                "thesis": "risk-off",
                "evidence": [{"type": "risk", "content": "vol spike", "weight": 0.8}],
                "time_horizon": "swing",
                "risk_tags": ["panic"],
                "confidence": 0.68,
                "counterarguments": ["policy may calm market"],
                "recommended_action": "reduce_risk",
            },
            analyst_id="a2",
        ),
    ]
    final_card = aggregate_analyst_cards(cards, risk_alert={"level": "medium"})
    assert "calibration" in final_card
    assert "brier_like_score" in final_card["calibration"]
    assert "confidence_drift" in final_card["calibration"]
    assert "contradiction_matrix" in final_card


def test_export_decision_trace_writes_json(tmp_path: Path) -> None:
    payload = {
        "agent_id": "test_agent",
        "analyst_cards": [],
        "manager_final_card": {"recommended_action": "hold"},
        "decision": {"action": "HOLD", "qty": 0, "price": 100.0},
    }
    trace_dir = tmp_path / "decision_trace"
    output_path = export_decision_trace(payload, trace_dir=str(trace_dir))
    exported = Path(output_path)
    assert exported.exists()
    data = json.loads(exported.read_text(encoding="utf-8"))
    assert data["agent_id"] == "test_agent"
    assert data["decision"]["action"] == "HOLD"
