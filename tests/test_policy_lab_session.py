from __future__ import annotations

import pandas as pd

from core.runtime_mode import resolve_runtime_mode_profile
from ui.policy_lab import (
    _policy_session_advance,
    _policy_session_enqueue,
    _policy_session_new,
    _policy_session_policy_package,
    _policy_session_report_payload,
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
    assert payload["title"].startswith("政策实验台")
    assert payload["session"]["status"] == "running"
    assert payload["timeline"]
    assert payload["narrative"]
