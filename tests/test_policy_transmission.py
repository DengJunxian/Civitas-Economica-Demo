from core.macro.government import GovernmentAgent
from core.macro.state import MacroState
from core.world.event_bus import EventBus


def test_policy_compiler_compiles_tax_cut_and_liquidity_injection() -> None:
    gov = GovernmentAgent()
    bus = EventBus()
    initial = MacroState()

    text = "宣布印花税下调，并同步进行流动性注入 5000 亿"
    shock = gov.compile_policy_text(text, tick=1)
    bus.publish(event_type="policy_compiled", stage="policy", tick=1, payload=shock.to_dict())

    assert shock.policy_text == text
    assert shock.stamp_tax_delta < 0.0
    assert shock.liquidity_injection > 0.0

    events = bus.consume_stage("policy")
    assert len(events) == 1
    assert events[0].payload["policy_text"] == text

    updated = MacroState.from_mapping(initial.to_dict())
    gov.apply_policy_shock(updated, shock)
    assert updated.liquidity_index > initial.liquidity_index
    assert updated.sentiment_index >= initial.sentiment_index


def test_policy_compiler_handles_negative_rumor() -> None:
    gov = GovernmentAgent()
    initial = MacroState(sentiment_index=0.6, liquidity_index=1.2)

    shock = gov.compile_policy_text("突发负面谣言引发恐慌", tick=2)
    updated = MacroState.from_mapping(initial.to_dict())
    gov.apply_policy_shock(updated, shock)

    assert shock.rumor_shock < 0.0
    assert updated.sentiment_index < initial.sentiment_index
    assert updated.credit_spread >= initial.credit_spread
