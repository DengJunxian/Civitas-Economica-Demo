from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from core.exchange.trade_tape import TradeTape
from core.types import Trade
from ui import dashboard
from ui.components.event_marker_layer import add_event_marker_layer, event_marker_counts, normalize_event_markers
from ui.components.replay_scrubber import build_replay_timeline
from ui.components.repro_meta import build_experiment_registry_entry, build_reproducibility_meta
from ui.components.scorecard_panel import mock_scorecard, render_scorecard_panel, scorecard_summary_frames


def _trade(trade_id: str, price: float, qty: int, ts: float) -> Trade:
    return Trade(
        trade_id=trade_id,
        price=price,
        quantity=qty,
        maker_id=f"maker_{trade_id}",
        taker_id=f"taker_{trade_id}",
        maker_agent_id=f"maker_agent_{trade_id}",
        taker_agent_id=f"taker_agent_{trade_id}",
        buyer_agent_id=f"buyer_{trade_id}",
        seller_agent_id=f"seller_{trade_id}",
        timestamp=ts,
    )


def test_streamlit_research_pages_import_without_runtime_services() -> None:
    import app  # noqa: F401
    import ui.backtest_panel  # noqa: F401
    import ui.behavioral_diagnostics  # noqa: F401
    import ui.dashboard  # noqa: F401
    import ui.policy_lab  # noqa: F401
    import ui.regulator_optimization  # noqa: F401
    import ui.reporting  # noqa: F401


def test_kline_component_uses_trade_tape_aggregated_ohlcv() -> None:
    ts = datetime(2026, 3, 23, 9, 30).timestamp()
    tape = TradeTape(symbol="sh000001", seed=7)
    tape.append_trade(_trade("t1", 3000.0, 100, ts), tick=1, trading_day="2026-03-23", phase="continuous")
    tape.append_trade(_trade("t2", 3012.0, 200, ts + 10), tick=1, trading_day="2026-03-23", phase="continuous")
    tape.append_trade(_trade("t3", 2996.0, 100, ts + 20), tick=1, trading_day="2026-03-23", phase="continuous")
    tape.append_trade(_trade("t4", 3008.0, 300, ts + 30), tick=1, trading_day="2026-03-23", phase="continuous")

    fig, frame = dashboard.build_trade_tape_kline_figure(
        tape.records,
        symbol="sh000001",
        benchmark_symbol="sh000001",
        events=[{"event_type": "policy", "title": "政策", "effective_day": 1}],
        selected_indicators=["MACD", "RSI", "BOLL"],
    )

    assert not frame.empty
    assert frame.iloc[0]["source"] == "trade_tape"
    assert frame.iloc[0]["open"] == 3000.0
    assert frame.iloc[0]["high"] == 3012.0
    assert frame.iloc[0]["low"] == 2996.0
    assert frame.iloc[0]["close"] == 3008.0
    assert any(getattr(trace, "type", "") == "candlestick" for trace in fig.data)


def test_event_marker_layer_normalizes_policy_news_regulator() -> None:
    events = [
        {"event_type": "policy", "title": "政策", "effective_day": 1},
        {"event_type": "major_news", "title": "新闻", "effective_day": 2},
        {"event_type": "regulatory_action", "title": "监管", "effective_day": 3},
    ]
    frame = normalize_event_markers(events, time_values=["d1", "d2", "d3"])
    counts = event_marker_counts(events)
    fig = add_event_marker_layer(go.Figure(), events, time_values=["d1", "d2", "d3"], y_default=1.0)

    assert set(frame["kind"]) == {"policy", "news", "regulator"}
    assert counts == {"policy": 1, "news": 1, "regulator": 1}
    assert {"政策事件", "新闻事件", "监管动作"} <= {trace.name for trace in fig.data}


def test_replay_timeline_contains_required_process_context() -> None:
    market = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "time": ["d1", "d2", "d3"],
            "open": [3000.0, 3002.0, 3001.0],
            "high": [3005.0, 3006.0, 3004.0],
            "low": [2998.0, 2999.0, 2996.0],
            "close": [3002.0, 3001.0, 3003.0],
            "volume": [1000, 1100, 1200],
            "panic_level": [0.2, 0.3, 0.25],
            "csad": [0.06, 0.07, 0.065],
        }
    )
    events = [
        {"event_type": "policy", "title": "政策", "effective_day": 1},
        {"event_type": "major_news", "title": "新闻", "effective_day": 2},
        {"event_type": "regulatory_action", "title": "监管", "effective_day": 3},
    ]

    timeline = build_replay_timeline(market, events=events)
    third = timeline[-1]

    assert third["policy"]
    assert third["news"]
    assert third["regulator"]
    assert {"agent_belief", "order_flow", "order_book", "trades", "kline"} <= set(third.keys())


def test_registry_repro_and_scorecard_mock_are_available() -> None:
    registry = build_experiment_registry_entry(
        scenario_name="baseline",
        config_hash="cfg",
        data_snapshot_id="snap",
        seed=123,
        selected_benchmark="sh000001",
        status="ready",
    )
    repro = build_reproducibility_meta(
        data_snapshot_hash="snap",
        config_hash="cfg",
        random_seed=123,
        llm_provider_chain=["offline_fallback"],
        calibration_parameter_set_id="calib",
    )
    metrics, flags = scorecard_summary_frames(mock_scorecard())

    assert registry["experiment_id"]
    assert registry["selected_benchmark"] == "sh000001"
    assert repro["random_seed"] == 123
    assert not metrics.empty
    assert not flags.empty
    render_scorecard_panel(mock_scorecard(), key_prefix="test_scorecard_panel")
