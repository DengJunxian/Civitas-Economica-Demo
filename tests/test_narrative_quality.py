from __future__ import annotations

import re

from ui import narrative


def test_fallback_narrative_is_data_driven_and_not_banned() -> None:
    payload = {
        "panic_level": 0.82,
        "csad": 0.16,
        "max_drawdown": 0.09,
        "close": [3000, 3012, 2998],
    }
    text = narrative._fallback_narrative("行为金融指标", payload, "判断风险是否扩散")

    assert all(phrase not in text for phrase in narrative.BANNED_NARRATIVE_PHRASES)
    assert "恐慌度" in text or "CSAD" in text or "最大回撤" in text
    assert re.search(r"\d", text)


def test_low_value_narrative_detects_banned_phrase() -> None:
    assert narrative.is_low_value_narrative(
        "建议先看趋势方向，再结合波动和风险项判断执行节奏。",
        {"panic_level": 0.7},
    )


def test_low_value_narrative_detects_numberless_output_for_numeric_payload() -> None:
    assert narrative.is_low_value_narrative(
        "该模块说明风险、趋势、波动和节奏均出现变化，整体应保持审慎研判并跟踪后续表现。",
        {"panic_level": 0.7, "csad": 0.1, "max_drawdown": 0.08, "volatility": 0.03},
    )


def test_narrate_payload_falls_back_when_llm_is_low_value(monkeypatch) -> None:
    def low_value(*args, **kwargs):
        return "总体来看仍需持续观察。"

    monkeypatch.setattr(narrative, "_llm_narrative", low_value)
    text = narrative.narrate_payload(
        "综合拟真评分",
        {"strict_authenticity_score": 0.71, "demo_authenticity_score": 0.78, "trend_alignment": 0.83},
        context="解释历史验证可信度",
        cache_namespace="test_narrative_quality_cache",
    )

    assert "总体来看仍需持续观察" not in text
    assert "综合拟真评分" in text or "严格拟真评分" in text or "趋势一致性" in text
    assert re.search(r"\d", text)
