from pathlib import Path

from core.competition_compliance import (
    COMPETITION_MODE_FLAG,
    competition_mode_enabled,
    write_competition_compliance_artifacts,
)
from core.competition_demo import (
    LIVE_MODE,
    DEMO_MODE,
    REQUIRED_SCENARIOS,
    list_competition_scenarios,
    load_competition_scenario,
    bootstrap_competition_demo,
    advance_competition_demo,
    replay_next_narration,
    switch_runtime_mode,
    build_competition_metrics_figure,
)
from core.ui_text import display_scenario_name, translate_ui_payload


def test_competition_demo_bootstrap_without_api_key():
    state = {}
    scenario = bootstrap_competition_demo(state, "tax_cut_liquidity_boost", auto_play=True)

    assert scenario.name == "tax_cut_liquidity_boost"
    assert state["runtime_mode"] == DEMO_MODE
    assert state["competition_mode"] == "COMPETITION_DEMO_MODE"
    assert state["is_running"] is True
    assert state["demo_autoplay"] is True
    assert state["market_history"] == []
    assert state["csad_history"] == []


def test_competition_demo_scenarios_are_loadable():
    scenarios = list_competition_scenarios()
    for required in REQUIRED_SCENARIOS:
        assert required in scenarios
        loaded = load_competition_scenario(required)
        assert loaded.name == required
        assert not loaded.metrics.empty
        assert isinstance(loaded.analyst_manager_output, dict)
        assert isinstance(loaded.narration, list)


def test_competition_demo_filters_incomplete_scenarios():
    scenarios = list_competition_scenarios()
    assert "hybrid_replay_abuse_minimal" not in scenarios


def test_competition_demo_bootstrap_fallback_to_first_valid():
    state = {}
    scenario = bootstrap_competition_demo(state, "unknown_scene_name", auto_play=False)
    assert scenario.name in REQUIRED_SCENARIOS
    assert state["demo_scenario_name"] == scenario.name


def test_competition_demo_metrics_figure_renderable():
    state = {}
    scenario = bootstrap_competition_demo(state, "rumor_panic_selloff", auto_play=False)

    advance_result = advance_competition_demo(state, steps=4)
    assert advance_result["advanced"] == 4

    fig = build_competition_metrics_figure(
        scenario.metrics,
        upto_step=state.get("demo_last_step"),
    )
    assert len(fig.data) >= 2
    assert fig.data[0].name == "指数收盘"
    assert fig.to_dict().get("data")


def test_ui_text_helpers_translate_frontend_content():
    assert display_scenario_name("tax_cut_liquidity_boost") == "减税与流动性提振"
    payload = translate_ui_payload(
        {
            "analyst_outputs": {
                "news_analyst": {
                    "headline": "Tax cut + liquidity support announced",
                    "signal": "risk_on",
                }
            }
        }
    )
    assert "分析师输出" in payload


def test_competition_demo_narration_replay():
    state = {}
    bootstrap_competition_demo(state, "regulator_stabilization_intervention", auto_play=False)

    first = replay_next_narration(state)
    assert first is not None
    assert "text" in first
    assert state["demo_narration_cursor"] == 1

    while replay_next_narration(state):
        pass

    assert replay_next_narration(state) is None


def test_competition_demo_mode_switching_no_error():
    state = {"is_running": True, "demo_autoplay": True}

    switch_runtime_mode(state, DEMO_MODE)
    assert state["runtime_mode"] == DEMO_MODE
    assert state["competition_mode"] == "COMPETITION_DEMO_MODE"

    switch_runtime_mode(state, LIVE_MODE)
    assert state["runtime_mode"] == LIVE_MODE
    assert state["competition_mode"] == ""
    assert state["is_running"] is False
    assert state["demo_autoplay"] is False


def test_competition_mode_gate_and_compliance_export(tmp_path: Path):
    assert competition_mode_enabled(feature_flags={COMPETITION_MODE_FLAG: True}, requested_mode=True) is True
    sample = tmp_path / "config.py"
    sample.write_text("API_KEY='x'\\nMODEL='deepseek-chat'\\n", encoding="utf-8")
    artifacts = write_competition_compliance_artifacts(
        root_dir=tmp_path,
        project_root=tmp_path,
        feature_flags={COMPETITION_MODE_FLAG: True},
        materials_context={"scenario": "demo"},
    )
    assert artifacts["manifest_path"].exists()
    assert artifacts["technical_route_template_path"].exists()
    assert "tools" in artifacts["manifest"]
