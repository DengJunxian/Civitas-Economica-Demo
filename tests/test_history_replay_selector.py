from core.agent_replay import AgentReplayEngine
from core.backtester import BacktestConfig, FactorBacktestEngine, HistoricalBacktester
from ui.history_replay import _select_replay_engine


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


def test_historical_backtester_keeps_factor_compatibility():
    assert issubclass(HistoricalBacktester, FactorBacktestEngine)
