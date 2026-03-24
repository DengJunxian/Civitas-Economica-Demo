import json
from datetime import datetime

from core.exchange.market_rules import SCHEMA_VERSION, build_market_rule_schema, resolve_market_rule_schema


def _ts(hour: int, minute: int) -> float:
    return datetime(2026, 3, 23, hour, minute).timestamp()


def test_market_rule_schema_is_serializable_and_round_trips():
    schema = build_market_rule_schema(
        "600000",
        100.0,
        overrides={
            "min_price_tick": 0.01,
            "min_trade_unit": 100,
            "enforce_min_trade_unit": True,
            "t_plus_one": True,
        },
        feature_flags={"market_rules_v1": True, "session_rules_v1": True},
    )

    payload = schema.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    restored = resolve_market_rule_schema(symbol="600000", prev_close=100.0, overrides=json.loads(encoded))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert restored.to_dict() == payload
    assert restored.feature_flags["market_rules_v1"] is True
    assert restored.t_plus_one is True


def test_market_rule_schema_applies_symbol_specific_price_limits():
    star_schema = build_market_rule_schema("688001", 100.0)
    growth_schema = build_market_rule_schema("300750", 100.0)
    st_schema = build_market_rule_schema("ST600001", 100.0)
    main_schema = build_market_rule_schema("600000", 100.0)

    assert star_schema.price_limit_pct == 0.20
    assert growth_schema.price_limit_pct == 0.20
    assert st_schema.price_limit_pct == 0.05
    assert main_schema.price_limit_pct == 0.10


def test_market_rule_schema_exposes_structured_sessions():
    schema = build_market_rule_schema(
        "600000",
        100.0,
        feature_flags={"session_rules_v1": True},
    )

    names = [session.name for session in schema.sessions]
    phases = [session.phase for session in schema.sessions]

    assert names[:4] == ["opening_call_auction", "continuous_am", "midday_break", "continuous_pm"]
    assert "closed" in phases
    assert schema.find_session(_ts(9, 20)).phase == "call_auction"
    assert schema.find_session(_ts(11, 45)).phase == "midday_break"
    assert schema.find_session(_ts(14, 0)).phase == "continuous"
