from core.agent_replay import AgentReplayEngine
from core.backtester import BacktestConfig, FactorBacktestEngine, HistoricalBacktester
from ui.history_replay import _resolve_history_workspace, _select_replay_engine


def test_history_replay_selector_falls_back_when_feature_flag_is_off():
    cfg = BacktestConfig(feature_flags={"agent_replay": False})
    engine, resolved_mode, reason = _select_replay_engine(cfg, "agent", cfg.feature_flags)

    assert isinstance(engine, FactorBacktestEngine)
    assert resolved_mode == "factor"
    assert reason


def test_history_replay_selector_uses_agent_engine_when_enabled():
    cfg = BacktestConfig(feature_flags={"agent_replay": True})
    engine, resolved_mode, reason = _select_replay_engine(cfg, "agent", cfg.feature_flags)

    assert isinstance(engine, AgentReplayEngine)
    assert resolved_mode == "agent"
    assert reason == ""


def test_history_replay_selector_keeps_factor_fallback_with_new_flags():
    cfg = BacktestConfig(
        feature_flags={
            "agent_replay": False,
            "history_replay_event_driven_v2": True,
            "history_replay_rolling_calibration_v1": True,
        }
    )
    engine, resolved_mode, reason = _select_replay_engine(cfg, "agent", cfg.feature_flags)

    assert isinstance(engine, FactorBacktestEngine)
    assert resolved_mode == "factor"
    assert "falling back" in reason.lower()


def test_historical_backtester_keeps_factor_compatibility():
    assert issubclass(HistoricalBacktester, FactorBacktestEngine)


def test_history_workspace_defaults_to_factor():
    assert _resolve_history_workspace(None) == "factor"
    assert _resolve_history_workspace("factor") == "factor"


def test_history_workspace_respects_agent_entry_mode():
    assert _resolve_history_workspace("agent") == "agent"
    assert _resolve_history_workspace(" AGENT ") == "agent"
