from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.ui_text import (
    MODE_DISPLAY_NAMES,
    VALUE_TRANSLATIONS,
    localize_dataframe_columns,
    zh_action_name,
    zh_label,
    zh_metric_name,
    zh_provider_name,
    zh_value,
    zh_world_name,
)


def test_core_ui_dictionary_covers_required_labels() -> None:
    assert zh_label("Research workbench") == "研究工作台"
    assert zh_label("benchmark selector") == "基准指数"
    assert zh_label("Scorecard") == "评估卡"
    assert zh_label("Composite Policy Score") == "政策综合评分"
    assert zh_label("relation_to_shanghai_index") == "与上证指数关系"
    assert zh_label("MACD signal") == "MACD 信号线"
    assert zh_label("BOLL upper") == "布林线上轨"
    assert zh_label("intervention_cost") == "干预成本"
    assert zh_value("self") == "核心基准"
    assert zh_value("strict") == "严格验证"
    assert zh_metric_name("early_warning_utility") == "预警价值"
    assert zh_world_name("early_intervention") == "提前介入"
    assert zh_action_name("BUY") == "买入"
    assert zh_provider_name("q_learning_baseline") == "强化学习基线"


def test_modes_no_longer_use_old_presentation_terms() -> None:
    assert MODE_DISPLAY_NAMES["DEMO_MODE"] == "场景推演"
    assert MODE_DISPLAY_NAMES["COMPETITION_DEMO_MODE"] == "综合展示"
    assert VALUE_TRANSLATIONS["demo_mode"] == "场景推演"


def test_localize_dataframe_columns_is_display_only() -> None:
    frame = pd.DataFrame(
        {
            "rank_score": [0.8],
            "relation_to_shanghai_index": ["self"],
            "default_production_path": ["q_learning_baseline"],
        }
    )
    localized = localize_dataframe_columns(frame)

    assert list(frame.columns) == ["rank_score", "relation_to_shanghai_index", "default_production_path"]
    assert list(localized.columns) == ["综合排序分", "与上证指数关系", "默认优化路径"]
    assert localized.iloc[0]["与上证指数关系"] == "核心基准"
    assert localized.iloc[0]["默认优化路径"] == "强化学习基线"


def test_primary_ui_copy_has_no_old_presentation_terms() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [
        "app.py",
        "core/ui_text.py",
        "ui/dashboard.py",
        "ui/policy_lab.py",
        "ui/history_replay.py",
        "ui/regulator_optimization.py",
        "ui/narrative.py",
        "ui/demo_wind_tunnel.py",
    ]
    banned = ["答辩演示", "比赛答辩", "展示校准", "不做展示校准", "保留展示校准", "答辩展示", "defense demo"]
    combined = "\n".join((root / file).read_text(encoding="utf-8", errors="ignore") for file in files)
    for phrase in banned:
        assert phrase not in combined
