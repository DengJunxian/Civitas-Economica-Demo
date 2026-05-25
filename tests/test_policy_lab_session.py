from __future__ import annotations

import pandas as pd

from core.policy_session import PolicySession, PolicySessionConfig
from core.runtime_mode import resolve_runtime_mode_profile
from ui.policy_lab import (
    _build_agent_fmri_rows,
    _build_policy_demo_briefing,
    _build_policy_demo_cards,
    _build_policy_market_pulse,
    _policy_session_backdrop_rows,
    _policy_session_advance,
    _policy_session_enqueue,
    _policy_session_maybe_autoplay,
    _policy_session_new,
    _policy_session_policy_package,
    _policy_session_report_payload,
    _session_frame_to_market_frame,
    _policy_session_status_text,
)


def _reference_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=12, freq="B")
    close = pd.Series([3000.0, 3008.0, 3012.0, 3006.0, 3020.0, 3028.0, 3033.0, 3026.0, 3038.0, 3042.0, 3040.0, 3048.0])
    volume = pd.Series([1_000_000 + idx * 15_000 for idx in range(len(dates))], dtype=float)
    return pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "open": close - 4.0,
            "high": close + 6.0,
            "low": close - 8.0,
            "close": close,
            "volume": volume,
        }
    )


def test_policy_session_new_starts_idle_with_initial_policy() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="下调印花税并释放流动性",
        policy_type="市场稳定",
        total_days=100,
        intensity=1.2,
        effective_day=1,
        half_life_days=30,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )

    assert session["status"] == "idle"
    assert session["total_days"] == 100
    assert session["current_day"] == 0
    assert session["policy_events"]
    assert _policy_session_status_text(session["status"]) == "未开始"


def test_policy_session_backdrop_rows_use_reference_history() -> None:
    rows = _policy_session_backdrop_rows(_reference_frame(), total_days=5)

    assert rows
    assert rows[0]["step"] == 1.0
    assert rows[-1]["price"] > 0.0
    assert all(item["price"] > 0.0 for item in rows)


def test_policy_session_resolve_day_prices_damps_extreme_raw_move_in_smart_mode() -> None:
    session = PolicySession(
        environment=object(),
        config=PolicySessionConfig(simulation_mode="SMART"),
    )

    old_price, close_price = session._resolve_day_prices(
        report={
            "old_price": 100.0,
            "new_price": 108.0,
            "buy_volume": 5000.0,
            "sell_volume": 1000.0,
            "macro_state": {"sentiment_index": 0.82},
            "behavioral_diagnostics": {"csad": 0.06},
            "policy_input": {"policy_intensity": 1.0},
        },
        active_timeline=[{}],
    )

    realized = (close_price - old_price) / old_price
    assert old_price == 100.0
    assert 0.0 < realized <= 0.0076


def test_policy_session_advance_builds_daily_frame_and_decays_policy() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="下调印花税并释放流动性",
        policy_type="市场稳定",
        total_days=5,
        intensity=1.2,
        effective_day=1,
        half_life_days=30,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"

    _policy_session_advance(session, 2)

    frame = pd.DataFrame(session["frame_rows"])
    assert session["current_day"] == 2
    assert not frame.empty
    assert frame.iloc[-1]["step"] == 2
    assert 0.0 < float(session["policy_events"][0]["remaining_effect"]) < 1.0
    assert session["summary"]["return_pct"] != 0.0
    assert session["summary"]["policy_signal_avg"] != 0.0


def test_policy_session_enqueue_adds_future_policy_and_updates_timeline() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="下调印花税并释放流动性",
        policy_type="市场稳定",
        total_days=10,
        intensity=1.0,
        effective_day=1,
        half_life_days=20,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    _policy_session_advance(session, 1)

    event = _policy_session_enqueue(
        session,
        policy_name="追加政策",
        policy_text="加大财政刺激力度",
        policy_type="财政刺激",
        effective_day=3,
        intensity=0.8,
        half_life_days=15,
        rumor_noise=True,
    )

    assert event["policy_name"] == "追加政策"
    assert len(session["policy_events"]) == 2
    assert any(item["当前状态"] == "待生效" for item in session["policy_timeline"])


def test_policy_session_report_payload_contains_chinese_summary_and_package() -> None:
    session = _policy_session_new(
        policy_name="测试政策",
        policy_text="下调印花税并释放流动性",
        policy_type="市场稳定",
        total_days=8,
        intensity=1.1,
        effective_day=1,
        half_life_days=30,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    _policy_session_advance(session, 3)

    package = _policy_session_policy_package(session)
    payload = _policy_session_report_payload(session, resolve_runtime_mode_profile("SMART"))

    assert package["event"]["raw_text"]
    assert payload["title"].startswith("政策试验台")
    assert payload["session"]["status"] == "running"
    assert payload["timeline"]
    assert payload["narrative"]


def test_policy_session_autoplay_advances_until_complete() -> None:
    session = _policy_session_new(
        policy_name="自动推进测试",
        policy_text="下调印花税并释放流动性支持。",
        policy_type="市场稳定",
        total_days=2,
        intensity=1.0,
        effective_day=1,
        half_life_days=10,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    session["autoplay"] = {
        "enabled": True,
        "step_days": 1,
        "interval_seconds": 0.0,
        "last_wallclock_ts": 0.0,
    }

    progressed = 0
    while _policy_session_maybe_autoplay(session, now_ts=float(progressed + 1), min_interval_seconds=0.0):
        progressed += 1

    frame = pd.DataFrame(session["frame_rows"])
    assert progressed == 2
    assert session["current_day"] == 2
    assert session["status"] == "completed"
    assert session["autoplay"]["enabled"] is False
    assert len(frame) == 2


def test_policy_session_autoplay_keeps_summary_in_sync() -> None:
    session = _policy_session_new(
        policy_name="自动推进同步测试",
        policy_text="发布稳市政策。",
        policy_type="市场稳定",
        total_days=2,
        intensity=1.0,
        effective_day=1,
        half_life_days=20,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"
    session["autoplay"] = {
        "enabled": True,
        "step_days": 1,
        "interval_seconds": 0.0,
        "last_wallclock_ts": 0.0,
    }

    while _policy_session_maybe_autoplay(session, now_ts=float(session.get("current_day", 0) + 1), min_interval_seconds=0.0):
        pass

    frame = pd.DataFrame(session["frame_rows"])
    assert session["current_day"] == 2
    assert len(frame) == 2
    assert float(session["summary"]["最新收盘价"]) == float(frame.iloc[-1]["close"])


def test_policy_session_snapshot_contains_transmission_chain() -> None:
    session = _policy_session_new(
        policy_name="传导链测试",
        policy_text="提高流动性支持，稳定市场预期。",
        policy_type="市场稳定",
        total_days=5,
        intensity=1.0,
        effective_day=1,
        half_life_days=15,
        rumor_noise=False,
        index_label="上证指数（000001）",
        index_symbol="sh000001",
        reference_frame=_reference_frame(),
        runtime_profile=resolve_runtime_mode_profile("SMART"),
    )
    session["status"] = "running"

    _policy_session_advance(session, 1)

    latest = session.get("latest_step_report", {})
    chain = latest.get("transmission_chain", {})

    assert "policy_signal" in chain
    assert "agent_sentiment" in chain
    assert "order_flow" in chain
    assert "matching_result" in chain
    assert "index_move" in chain


def test_policy_lab_demo_cards_cover_reference_three_stage_story() -> None:
    cards = _build_policy_demo_cards(
        policy_text="下调印花税并释放流动性。",
        latest_step_report={
            "transmission_chain": {
                "policy_signal": {"strength": 1.2, "policy_text": "下调印花税并释放流动性。"},
                "agent_sentiment": {"social_mean": -0.1, "committee_enabled": True},
                "order_flow": {"buy_volume": 1200.0, "sell_volume": 900.0},
                "matching_result": {"trade_count": 18, "last_price": 3050.0},
                "index_move": {"return_pct": 0.8, "new_price": 3050.0},
            }
        },
        session_summary={"policy_signal_avg": 0.4, "最新收盘价": 3050.0},
    )

    assert [card["phase"] for card in cards] == ["政策注入", "情绪扩散", "撮合落地"]
    assert all(card["summary"] for card in cards)


def test_policy_lab_market_frame_smooths_extreme_close_and_volume_moves() -> None:
    frame = pd.DataFrame(
        {
            "交易日序号": [1, 2, 3],
            "交易日": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "收盘价": [100.0, 116.0, 96.0],
            "总买量": [1000.0, 9000.0, 1500.0],
            "总卖量": [800.0, 8500.0, 1200.0],
            "羊群度": [0.05, 0.07, 0.06],
            "恐慌度": [0.20, 0.38, 0.28],
            "成交笔数": [20, 60, 24],
            "活跃政策数": [1, 1, 1],
        }
    )

    market = _session_frame_to_market_frame(frame, anchor_close=3000.0)
    returns = market["close"].pct_change().fillna(0.0)

    assert len(market) == 3
    assert float(market.iloc[0]["close"]) == 3000.0
    assert float(returns.abs().max()) <= 0.0076
    assert float(market["volume"].max()) < 20000.0


def test_policy_lab_agent_fmri_rows_capture_agent_stance() -> None:
    session = {
        "policy_text": "发布稳市政策并配套流动性支持。",
        "index_symbol": "sh000001",
        "last_close": 3025.0,
        "last_panic": 0.32,
        "llm_agent_count": 3,
        "latest_step_report": {
            "transmission_chain": {
                "agent_sentiment": {"panic_level": 0.32},
                "order_flow": {"buy_volume": 1800.0, "sell_volume": 900.0},
                "matching_result": {"last_price": 3025.0},
            }
        },
    }
    package_dict = {"agent_class_effects": {"mutual_fund": 0.42, "quant_arbitrage": -0.21}}

    rows = _build_agent_fmri_rows(session, package_dict)

    assert len(rows) == 2
    assert rows[0]["agent"] == "mutual_fund"
    assert rows[0]["decision"]["action"] == "BUY"
    assert rows[0]["decision_label"].startswith("BUY")
    assert rows[1]["decision"]["action"] == "SELL"
    assert rows[0]["history"]


def test_policy_lab_briefing_and_pulse_detect_market_stress() -> None:
    session = {
        "status": "running",
        "current_day": 3,
        "total_days": 10,
        "mode_text": "标准推演模式",
        "policy_text": "发布稳市政策并配套流动性支持。",
        "policy_events": [{"status": "active"}],
        "autoplay": {"enabled": True},
        "latest_step_report": {
            "price_change_pct": -1.6,
            "transmission_chain": {
                "policy_signal": {"strength": 1.1, "policy_text": "发布稳市政策并配套流动性支持。"},
                "agent_sentiment": {"social_mean": -0.45, "panic_level": 0.72},
                "order_flow": {"buy_volume": 800.0, "sell_volume": 2100.0, "imbalance": -1300.0},
                "matching_result": {"trade_count": 24, "matching_mode": "v2"},
                "index_move": {"return_pct": -1.6, "new_price": 2980.0},
            },
        },
    }
    summary = {"return_pct": -0.022, "max_panic": 0.72, "avg_panic": 0.58}

    briefing = _build_policy_demo_briefing(session, summary)
    pulse = _build_policy_market_pulse(session, summary)

    assert briefing["alert"] in {"高波动预警", "市场异动提示"}
    assert briefing["tone"] in {"risk", "watch"}
    assert pulse[0]["value"] == "卖盘主导"
    assert pulse[-1]["value"] == "下行重估"
