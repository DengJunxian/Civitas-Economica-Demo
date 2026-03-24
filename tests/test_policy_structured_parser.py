from core.macro.government import GovernmentAgent
from policy.structured import StructuredPolicyParser


def test_structured_parser_maps_common_policy_texts_to_channels() -> None:
    parser = StructuredPolicyParser(seed=42)

    cases = [
        (
            "宣布降准并降低政策利率，稳定市场预期",
            {"liquidity", "rate", "credit_spread"},
            {"parser_mode": "structured"},
        ),
        (
            "调整印花税并释放流动性，提升交易活跃度",
            {"tax_frictions", "liquidity"},
            {"parser_mode": "structured"},
        ),
        (
            "辟谣并维稳发声，压制恐慌情绪",
            {"rumor_suppression", "sentiment_confidence"},
            {"parser_mode": "structured"},
        ),
    ]

    for idx, (policy_text, expected_channels, expected_meta) in enumerate(cases, start=1):
        package = parser.parse(
            policy_text,
            tick=idx,
            market_regime="risk_off",
            snapshot_info={"window": "2026-03", "case": idx},
        )

        assert package.event.raw_text == policy_text
        assert package.uncertainty.parser_mode == expected_meta["parser_mode"]
        assert package.metadata["config_hash"]
        assert package.metadata["snapshot_info"]["window"] == "2026-03"
        assert expected_channels.issubset({channel.name for channel in package.channels})
        assert package.top_layers()["sector"]
        assert package.top_layers()["factor"]
        assert package.top_layers()["agent_class"]


def test_government_agent_can_disable_structured_parser() -> None:
    gov = GovernmentAgent(feature_flags={"structured_policy_parser_v1": False})

    package = gov.compile_policy_package(
        "未分类政策文本，只用于验证兼容层",
        tick=9,
        snapshot_info={"window": "legacy-mode"},
    )

    assert package.parser_version == "legacy_policy_compiler"
    assert package.uncertainty.parser_mode == "legacy"
    assert package.metadata["config_hash"]
    assert package.metadata["snapshot_info"]["window"] == "legacy-mode"
