import pytest

from agents.cognition.layered_memory import LayeredMemory


def test_layered_memory_reacts_to_consecutive_losses():
    memory = LayeredMemory(
        agent_id="mf_1",
        seed=7,
        enabled=True,
        institution_type="mutual_fund",
    )

    market_state = {
        "price": 10.0,
        "last_price": 10.0,
        "panic_level": 0.15,
        "policy_description": "降息 降准 流动性支持",
        "policy_news": "liquidity support",
        "text_policy_shock": 0.35,
        "text_regime_bias": "easing",
        "news_source": "policy",
    }
    account_state = {"pnl_pct": 0.0, "market_value": 100_000.0}
    decision = {"action": "BUY", "qty": 2_000, "price": 10.0}

    first = memory.apply_decision_overlay(
        decision,
        market_state=market_state,
        account_state=account_state,
        emotional_state="Neutral",
        social_signal="Market is quiet.",
        snapshot_info={"price": 10.0, "policy_description": "降息"},
    )
    first_card = first["behavior_card"]

    for _ in range(3):
        memory.record_outcome(
            decision=first["decision"],
            outcome={"pnl": -20_000.0, "pnl_pct": -0.20, "source": "policy"},
            market_state=market_state,
            account_state=account_state,
        )

    second = memory.apply_decision_overlay(
        decision,
        market_state=market_state,
        account_state=account_state,
        emotional_state="Fearful",
        social_signal="panic selling",
        snapshot_info={"price": 10.0, "policy_description": "降息"},
    )
    second_card = second["behavior_card"]

    assert second_card["current_risk_budget"] < first_card["current_risk_budget"]
    assert second_card["current_belief"] != first_card["current_belief"]
    assert second_card["attention_fatigue"] >= first_card["attention_fatigue"]
    assert second_card["config_hash"] == first_card["config_hash"]
    assert second_card["seed"] == 7
    assert second_card["memory_layers"]["short_term"]
    assert "long_term" in second_card["memory_layers"]
    assert second_card["metadata"]["behavior_version"] == "layered_memory_v2"


def test_institution_constraints_are_stable_and_traceable():
    memory = LayeredMemory(
        agent_id="mm_1",
        seed=11,
        enabled=True,
        institution_type="market_maker",
    )

    market_state = {
        "price": 12.0,
        "last_price": 12.0,
        "panic_level": 0.05,
        "policy_description": "正常交易",
        "policy_news": "",
        "text_policy_shock": 0.0,
        "text_regime_bias": "neutral",
        "news_source": "market",
    }
    account_state = {"pnl_pct": 0.0, "market_value": 50_000.0}
    result = memory.apply_decision_overlay(
        {"action": "BUY", "qty": 500, "price": 12.0},
        market_state=market_state,
        account_state=account_state,
        emotional_state="Neutral",
        social_signal="steady flow",
        snapshot_info={"price": 12.0, "policy_description": "正常交易"},
    )

    constraints = result["behavior_card"]["current_constraints"]
    assert constraints["institution_type"] == "market_maker"
    assert constraints["inventory_limit"] <= 0.05
    assert result["behavior_card"]["current_narrative"]
    assert result["behavior_card"]["memory_layers"]["long_term"]["institution_type"] == "market_maker"
    assert result["behavior_card"]["metadata"]["behavior_version"] == "layered_memory_v2"
