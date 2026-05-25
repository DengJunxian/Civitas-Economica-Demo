from core.macro.government import GovernmentAgent


def test_policy_package_exposes_three_layer_transmission_outputs() -> None:
    gov = GovernmentAgent()
    text = "宣布降准并压降信用利差，释放流动性，稳定市场预期"

    package = gov.compile_policy_package(
        text,
        tick=3,
        market_regime="risk_off",
        snapshot_info={"source": "unit-test", "window": "2026-03"},
    )

    assert package.event.raw_text == text
    assert package.sector_effects
    assert package.factor_effects
    assert package.agent_class_effects
    assert package.market_effects
    assert package.policy_schema["schema_version"] == "policy_schema_v1"
    assert package.transmission_graph["schema_version"] == "transmission_graph_v1"
    assert package.transmission_graph["network_layers"]["information_network"]
    assert package.explanation["why_this_happened"]["channels"]
    assert package.metadata["config_hash"]
    assert package.metadata["snapshot_info"]["source"] == "unit-test"

    layers = package.top_layers()
    assert set(layers.keys()) == {"sector", "factor", "agent_class"}
    assert layers["sector"]
    assert layers["factor"]
    assert layers["agent_class"]

    shock = gov.compile_policy_text(
        text,
        tick=3,
        market_regime="risk_off",
        snapshot_info={"source": "unit-test", "window": "2026-03"},
    )

    assert shock.policy_text == text
    assert shock.policy_rate_delta <= 0.0
    assert shock.liquidity_injection >= 0.0
    assert shock.metadata["policy_package"]["event"]["raw_text"] == text
    assert shock.metadata["reproducibility"]["config_hash"] == package.metadata["config_hash"]
    assert shock.metadata["policy_package"]["transmission_graph"]["primary_path"]


def test_policy_package_reproducibility_metadata_is_stable() -> None:
    gov = GovernmentAgent()
    text = "印花税下调并释放流动性，提升市场风险偏好"

    first = gov.compile_policy_package(text, tick=11, snapshot_info={"window": "A"})
    second = gov.compile_policy_package(text, tick=11, snapshot_info={"window": "A"})

    assert first.metadata["config_hash"] == second.metadata["config_hash"]
    assert first.metadata["snapshot_info"]["window"] == "A"
    assert second.metadata["snapshot_info"]["window"] == "A"
    assert first.policy_schema["action_channels"] == second.policy_schema["action_channels"]
