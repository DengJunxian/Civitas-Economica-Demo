"""UI-facing text helpers for Chinese labels."""

from __future__ import annotations

from typing import Any


SCENARIO_DISPLAY_NAMES = {
    "tax_cut_liquidity_boost": "减税与流动性提振",
    "rumor_panic_selloff": "传言冲击与恐慌抛售",
    "regulator_stabilization_intervention": "监管维稳与市场修复",
}

MODE_DISPLAY_NAMES = {
    "LIVE_MODE": "实时联机",
    "DEMO_MODE": "答辩演示",
    "COMPETITION_DEMO_MODE": "比赛答辩",
}

RISK_ALERT_DISPLAY_NAMES = {
    "GREEN": "绿色",
    "YELLOW": "黄色",
    "RED": "红色",
}

KEY_TRANSLATIONS = {
    "scenario": "场景",
    "analyst_outputs": "分析师输出",
    "analyst_cards": "分析师卡片",
    "analyst_id": "分析师",
    "headline": "标题",
    "sentiment_score": "情绪得分",
    "key_event": "关键事件",
    "momentum": "动量",
    "herding_intensity": "羊群强度",
    "signal": "信号",
    "cvar": "条件风险价值",
    "max_drawdown": "最大回撤",
    "risk_level": "风险等级",
    "manager_decision": "经理决策",
    "manager_final_card": "经理最终卡",
    "stance": "策略立场",
    "allocation": "仓位分配",
    "execution_plan": "执行计划",
    "equity": "股票",
    "cash": "现金",
    "hedge": "对冲",
    "step": "步数",
    "close": "收盘价",
    "volume": "成交量",
    "panic_level": "风险热度",
    "csad": "羊群度",
    "contradiction_matrix": "矛盾矩阵",
    "contradiction_index": "矛盾指数",
    "analysts": "分析师列表",
    "matrix": "矩阵",
    "risk_alert": "风险预警",
    "risk_alerts": "风险告警",
    "calibration": "校准指标",
    "brier_like_score": "Brier 类分数",
    "confidence_drift": "置信漂移",
    "outcome_proxy": "结果代理值",
    "raw_confidence": "原始置信度",
    "calibrated_confidence": "校准后置信度",
    "time_horizon": "时间跨度",
    "counterarguments": "反方观点",
    "recommended_action": "建议动作",
    "confidence": "置信度",
    "risk_tags": "风险标签",
    "evidence": "证据",
    "type": "类型",
    "content": "内容",
    "weight": "权重",
    "thesis": "核心判断",
    "speaker": "角色",
    "text": "内容",
    "event": "事件",
    "level": "等级",
}

VALUE_TRANSLATIONS = {
    "news_analyst": "新闻分析师",
    "quant_analyst": "量化分析师",
    "risk_analyst": "风险分析师",
    "GREEN": "绿色",
    "YELLOW": "黄色",
    "RED": "红色",
    "risk_on": "风险偏好提升",
    "panic_sell": "恐慌抛售",
    "stabilizing": "趋于稳定",
    "moderate": "中等",
    "high": "高",
    "controlled": "可控",
    "RISK_ON_CONTROLLED": "进攻但受控",
    "DEFENSIVE_DELEVERAGING": "防御去杠杆",
    "STABILIZE_AND_REBALANCE": "稳市再平衡",
    "hold": "观望",
    "swing": "波段",
    "expert_replay": "专家复盘",
    "demo_mode": "答辩演示",
    "旁白": "旁白",
    "Tax cut + liquidity support announced": "减税与流动性支持政策公布",
    "Policy package reduces transaction friction": "政策组合降低了交易摩擦",
    "Unverified restructuring rumor spreads": "未经证实的重组传言迅速扩散",
    "Negative social amplification accelerates risk-off": "负面舆情扩散加速风险偏好收缩",
    "Regulator launches stabilization package": "监管部门推出市场稳定方案",
    "Market receives coordinated policy backstop": "市场获得协同政策托底",
    "Increase index exposure in three tranches": "分三批提高指数仓位",
    "Keep liquidity buffer to avoid chasing spikes": "保留流动性缓冲，避免追高",
    "Switch to defensive plan if regulation reverses": "若监管预期逆转，切换至防御方案",
    "Cut high-beta positions first": "优先削减高 Beta 仓位",
    "Increase cash and wait for volatility compression": "提高现金比例，等待波动收敛",
    "Activate emergency hedge against gap risk": "启动应急对冲，应对跳空风险",
    "Rebalance exposure toward liquid large caps": "将仓位再平衡至高流动性大盘蓝筹",
    "Reduce tail-risk hedge gradually as panic cools": "随着恐慌缓解，逐步降低尾部风险对冲",
    "Track policy persistence before adding risk": "确认政策持续性后再增加风险敞口",
}

TEXT_REPLACEMENTS = (
    ("A/B World Compare", "A/B 世界对照"),
    ("A/B Compare", "A/B 对照"),
    ("risk-on", "风险偏好提升"),
    ("risk-off", "风险偏好收缩"),
    ("legacy output", "历史输出"),
    ("large caps", "大盘蓝筹"),
    ("tail-risk hedge", "尾部风险对冲"),
    ("gap risk", "跳空风险"),
)


def display_scenario_name(name: str) -> str:
    return SCENARIO_DISPLAY_NAMES.get(name, name)


def display_runtime_mode(mode: str) -> str:
    return MODE_DISPLAY_NAMES.get(mode, mode)


def display_risk_alert(level: str) -> str:
    return RISK_ALERT_DISPLAY_NAMES.get(level, level)


def translate_display_text(text: str) -> str:
    translated = SCENARIO_DISPLAY_NAMES.get(text, VALUE_TRANSLATIONS.get(text, text))
    for source, target in TEXT_REPLACEMENTS:
        translated = translated.replace(source, target)
    return translated


def translate_ui_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            KEY_TRANSLATIONS.get(str(key), translate_display_text(str(key))): translate_ui_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [translate_ui_payload(item) for item in value]
    if isinstance(value, str):
        return translate_display_text(value)
    return value
