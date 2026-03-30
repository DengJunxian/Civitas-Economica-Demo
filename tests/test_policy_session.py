from __future__ import annotations

import pytest

from core.policy_session import PolicySession


class _SessionHoldAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.persona = type("PersonaStub", (), {"risk_tolerance": 0.5})()
        self.memory_bank = None

    async def generate_trading_decision(self, market_data, retrieved_context):
        _ = market_data, retrieved_context
        return type("HoldAction", (), {"action": "HOLD", "amount": 0.0, "target_price": 100.0})()


@pytest.mark.asyncio
async def test_policy_session_supports_base_policy_enqueue_decay_and_report_payload() -> None:
    session = PolicySession.create(
        agents=[_SessionHoldAgent("a1"), _SessionHoldAgent("a2")],
        total_days=6,
        base_policy="下调印花税并注入流动性",
        start_day=1,
        half_life_days=1.0,
        enable_random_policy_events=False,
        use_isolated_matching=False,
        steps_per_day=1,
    )

    first = await session.advance_async(1)
    frame1 = first["frame"]
    assert first["current_day"] == 1
    assert len(frame1) == 1
    assert frame1.iloc[0]["活跃政策数"] == 1
    first_strength = float(frame1.iloc[0]["政策强度"])
    assert first_strength > 0.0

    session.enqueue_policy(
        "发布辟谣并释放稳定预期信号",
        effective_day=3,
        strength=0.9,
        half_life_days=2.0,
        rumor_noise=True,
        label="追加政策",
    )

    second = await session.advance_async(2)
    frame2 = second["frame"]
    assert second["current_day"] == 3
    assert len(frame2) == 3
    assert float(frame2.iloc[1]["政策强度"]) < first_strength
    assert second["active_policies"]
    assert any(item["是否谣言噪声"] for item in second["active_policies"])
    assert not second["queued_policies"]

    payload = second["report_payload"]
    assert "政策时间轴" in payload
    assert "会话摘要" in payload
    assert payload["会话摘要"]["当前交易日"] == 3

    report = session.generate_report(use_llm=False)
    assert "政策试验台会话报告" in report["报告正文"]
    assert "会话摘要" in report["报告数据"]
    assert report["是否使用大模型"] is False

    session.stop()
    stopped = await session.advance_async(1)
    assert stopped["current_day"] == 3
    assert stopped["status"] == "stopped"


@pytest.mark.asyncio
async def test_policy_session_can_disable_random_policy_events() -> None:
    session = PolicySession.create(
        agents=[_SessionHoldAgent("x1"), _SessionHoldAgent("x2")],
        total_days=5,
        base_policy="",
        enable_random_policy_events=False,
        use_isolated_matching=False,
        steps_per_day=1,
    )

    result = await session.advance_async(5)
    frame = result["frame"]
    assert result["current_day"] == 5
    assert len(frame) == 5
    assert all(str(text) == "" for text in frame["政策文本"].tolist())
    assert result["summary"]["自动随机政策事件"] is False
