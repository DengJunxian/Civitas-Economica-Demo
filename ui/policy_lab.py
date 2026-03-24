"""Policy lab page focused on government-facing policy experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.macro.government import GovernmentAgent, PolicyShock
from core.event_store import EventRecord, EventStore, EventType
from policy.structured import PolicyPackage
from ui import dashboard as dashboard_ui
from ui.reporting import official_report_meta, write_report_artifacts


POLICY_TYPE_OPTIONS = {
    "税收调整": "tax",
    "流动性投放": "liquidity",
    "财政刺激": "fiscal",
    "监管收紧": "tightening",
    "市场稳定措施": "stabilization",
    "自定义政策": "custom",
}

TEMPLATE_LIBRARY_PATH = Path("data") / "policy_templates.json"
POLICY_REPORT_DIR = Path("outputs") / "policy_reports"
CONTROL_MODE_OPTIONS = ["不启用对照组", "无政策基线", "模板建议对照", "温和版本", "风险压力版本"]


@dataclass
class PolicyNarrativeCard:
    title: str
    summary: str
    bullets: List[str]
    tone: str = "neutral"


def _default_template_library() -> List[Dict[str, Any]]:
    return [
        {
            "id": "stamp-tax-liquidity",
            "category": "市场稳定",
            "title": "下调印花税并配套流动性支持",
            "policy_type": "税收调整",
            "policy_text": "阶段性下调证券交易印花税，并同步释放流动性支持，引导长期资金入市，稳定市场预期。",
            "policy_goal": "改善流动性，降低交易摩擦，稳定指数表现。",
            "suitable_departments": "财政、税务、证监、市场稳定资金团队",
            "recommended_intensity": 1.1,
            "recommended_duration": 30,
            "default_rumor_noise": False,
            "control_label": "维持现有税费与流动性安排",
            "control_text": "维持当前税费和流动性安排，不新增稳定市场措施。",
        }
    ]


def _load_policy_templates() -> List[Dict[str, Any]]:
    if TEMPLATE_LIBRARY_PATH.exists():
        try:
            payload = json.loads(TEMPLATE_LIBRARY_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return payload
        except Exception:
            pass
    return _default_template_library()


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _to_iso_safe(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.Timestamp.now(tz="UTC")
    return ts.isoformat()


def _policy_feature_flags(enable_structured_parser: bool = True) -> Dict[str, bool]:
    return {
        "structured_policy_parser_v1": bool(enable_structured_parser),
        "policy_transmission_layers_v1": True,
    }


def _compile_policy_bundle(
    policy_text: str,
    intensity: float,
    *,
    policy_type_hint: Optional[str] = None,
    market_regime: Optional[str] = None,
    enable_structured_parser: bool = True,
) -> Tuple[PolicyShock, PolicyPackage]:
    gov = GovernmentAgent(feature_flags=_policy_feature_flags(enable_structured_parser))
    package = gov.compile_policy_package(
        policy_text,
        tick=1,
        policy_type_hint=policy_type_hint,
        intensity=float(intensity),
        market_regime=market_regime,
        snapshot_info={
            "policy_text_length": len(policy_text or ""),
            "policy_type_hint": policy_type_hint or "",
            "parser_mode": "structured" if enable_structured_parser else "legacy",
        },
    )
    shock = PolicyShock(policy_id=package.event.policy_id, policy_text=policy_text, **package.to_policy_shock_fields())
    shock.metadata = {
        "policy_event": package.event.to_dict() if hasattr(package.event, "to_dict") else {
            "policy_id": package.event.policy_id,
            "raw_text": package.event.raw_text,
            "policy_type": package.event.policy_type,
            "policy_label": package.event.policy_label,
            "direction": package.event.direction,
            "intensity": package.event.intensity,
            "tick": package.event.tick,
            "matched_tokens": list(package.event.matched_tokens),
            "timestamp": float(package.event.timestamp),
            "source": package.event.source,
        },
        "policy_package": package.to_dict(),
        "reproducibility": {
            "seed": int(package.metadata.get("seed", 0)),
            "config_hash": str(package.metadata.get("config_hash", "")),
            "snapshot_info": dict(package.metadata.get("snapshot_info", {})),
        },
        "parser_mode": package.uncertainty.parser_mode,
        "feature_flags": dict(package.metadata.get("feature_flags", {})),
    }
    return shock, package


def _compile_scaled_shock(
    policy_text: str,
    intensity: float,
    *,
    enable_structured_parser: bool = True,
) -> PolicyShock:
    shock, _ = _compile_policy_bundle(policy_text, intensity, enable_structured_parser=enable_structured_parser)
    return shock


def _format_top_effects(effect_map: Dict[str, float], limit: int = 3) -> List[str]:
    ranked = sorted(effect_map.items(), key=lambda item: abs(float(item[1])), reverse=True)
    return [f"{name}: {float(value):+.3f}" for name, value in ranked[:limit]]


def _normalize_top_layer(layer_value: Any, limit: int = 3) -> List[str]:
    if isinstance(layer_value, dict):
        return _format_top_effects({str(key): float(value) for key, value in layer_value.items()}, limit=limit)
    if isinstance(layer_value, list):
        normalized: List[str] = []
        for item in layer_value[:limit]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append(f"{item[0]}: {float(item[1]):+.3f}")
            else:
                normalized.append(str(item))
        return normalized
    return []


def _render_policy_transmission_path(primary: Dict[str, Any]) -> None:
    package = primary.get("policy_package") or {}
    transmission_layers = primary.get("transmission_layers") or {}
    if not package:
        return

    event = package.get("event", {})
    channels = package.get("channels", [])
    uncertainty = package.get("uncertainty", {})
    reproducibility = primary.get("reproducibility", {})
    market_effects = transmission_layers.get("market", package.get("market_effects", {}))
    top_layers = transmission_layers.get("top_layers", package.get("top_layers", {}))

    channel_lines = []
    for channel in channels[:3]:
        if not isinstance(channel, dict):
            continue
        channel_lines.append(
            f"{channel.get('name', 'channel')} / {float(channel.get('impact', 0.0)):+.2f} / "
            f"lag {int(channel.get('lag_days', 0))}d"
        )
    if not channel_lines:
        channel_lines = ["暂无显式通道，已回落到 legacy parser"]

    sector_lines = _normalize_top_layer(top_layers.get("sector")) or _format_top_effects(transmission_layers.get("sector", package.get("sector_effects", {})))
    factor_lines = _normalize_top_layer(top_layers.get("factor")) or _format_top_effects(transmission_layers.get("factor", package.get("factor_effects", {})))
    agent_lines = _normalize_top_layer(top_layers.get("agent_class")) or _format_top_effects(transmission_layers.get("agent_class", package.get("agent_class_effects", {})))
    market_lines = [
        f"market_bias: {float(market_effects.get('market_bias', 0.0)):+.3f}",
        f"liquidity_bias: {float(market_effects.get('liquidity_bias', 0.0)):+.3f}",
        f"confidence_bias: {float(market_effects.get('confidence_bias', 0.0)):+.3f}",
        f"volatility_bias: {float(market_effects.get('volatility_bias', 0.0)):+.3f}",
    ]

    cols = st.columns(4)
    cards = [
        (
            "政策",
            f"{event.get('policy_label', 'Unknown')} / {event.get('policy_type', 'unclassified')}",
            [
                f"direction: {event.get('direction', 'neutral')}",
                f"intensity: {float(event.get('intensity', 0.0)):.2f}",
                f"confidence: {float(uncertainty.get('confidence', 0.0)):.2f}",
            ],
        ),
        (
            "传导通道",
            channels[0].get("name", "channel") if channels else "channel",
            channel_lines,
        ),
        (
            "主体反应",
            "sector / factor / agent-class",
            [
                f"sector: {sector_lines[0] if sector_lines else 'n/a'}",
                f"factor: {factor_lines[0] if factor_lines else 'n/a'}",
                f"agent: {agent_lines[0] if agent_lines else 'n/a'}",
            ],
        ),
        (
            "市场结果",
            "path fit / microstructure / behavior",
            market_lines
            + [
                f"seed: {reproducibility.get('seed', 0)}",
                f"config: {str(reproducibility.get('config_hash', ''))[:12]}",
            ],
        ),
    ]
    for idx, (title, summary, bullets) in enumerate(cards):
        items = "".join(f"<li>{item}</li>" for item in bullets)
        with cols[idx]:
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-label">{title}</div>
                  <div class="summary-value" style="font-size: 1.02em; font-weight: 600;">{summary}</div>
                  <ul class="story-card-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(
        "reproducibility: "
        f"seed={reproducibility.get('seed', 0)} | "
        f"config_hash={reproducibility.get('config_hash', '')} | "
        f"snapshot={json.dumps(reproducibility.get('snapshot_info', {}), ensure_ascii=False, sort_keys=True)}"
    )


def _apply_local_policy_keywords(policy_text: str, shock: PolicyShock) -> None:
    text = str(policy_text or "")
    lower = text.lower()

    def _contains(*tokens: str) -> bool:
        return any(token in text or token in lower for token in tokens)

    if _contains("印花税", "减税", "税率下调", "lower tax", "cut tax"):
        shock.stamp_tax_delta -= 0.0005
        shock.liquidity_injection += 0.06
        shock.sentiment_delta += 0.05

    if _contains("流动性", "注入", "降准", "降息", "liquidity", "rrr cut", "rate cut"):
        shock.liquidity_injection += 0.08
        shock.policy_rate_delta -= 0.001
        shock.sentiment_delta += 0.05

    if _contains("财政", "补贴", "刺激", "基建", "fiscal", "subsidy", "stimulus"):
        shock.fiscal_stimulus_delta += 0.05
        shock.sentiment_delta += 0.04

    if _contains("稳定市场", "托底", "平准", "长期资金入市", "stabilize", "backstop"):
        shock.liquidity_injection += 0.04
        shock.sentiment_delta += 0.06
        shock.credit_spread_delta -= 0.001

    if _contains("谣言", "风险", "恐慌", "挤兑", "panic", "rumor", "selloff"):
        shock.rumor_shock -= 0.16
        shock.sentiment_delta -= 0.08
        shock.credit_spread_delta += 0.0015

    if _contains("加息", "收紧", "tighten", "tightening"):
        shock.policy_rate_delta += 0.0012
        shock.sentiment_delta -= 0.06
        shock.credit_spread_delta += 0.0015


def _shock_score(shock: PolicyShock) -> float:
    return (
        shock.liquidity_injection * 1.3
        + shock.fiscal_stimulus_delta * 1.5
        - shock.policy_rate_delta * 60.0
        - shock.credit_spread_delta * 18.0
        - shock.stamp_tax_delta * 420.0
        + shock.sentiment_delta * 1.2
        + shock.rumor_shock * 1.6
    )


def _generate_policy_metrics(
    *,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    scenario_key: str,
) -> pd.DataFrame:
    shock = _compile_scaled_shock(policy_text, intensity)
    score = _shock_score(shock)
    seed = f"{scenario_key}|{policy_text}|{intensity}|{duration_days}|{rumor_noise}"
    rng = np.random.default_rng(_seed_from_text(seed))

    periods = max(10, int(duration_days))
    dates = pd.bdate_range(pd.Timestamp.today().normalize(), periods=periods)
    base_price = 3000.0
    price = base_price
    rows: List[Dict[str, float | int | str]] = []

    baseline_drift = 0.0004
    support_drift = np.clip(score * 0.0032, -0.008, 0.008)
    base_vol = 0.0035 + 0.005 * min(1.0, abs(shock.sentiment_delta) + abs(shock.rumor_shock))
    panic_floor = 0.18 + 0.18 * max(0.0, -shock.sentiment_delta)

    for idx, dt in enumerate(dates, start=1):
        decay = np.exp(-(idx - 1) / max(periods * 0.55, 1.0))
        rumor_pulse = 0.0
        if rumor_noise:
            rumor_pulse = shock.rumor_shock * 0.010 * np.exp(-(idx - 1) / max(periods * 0.25, 1.0))
        drift = baseline_drift + support_drift * decay + rumor_pulse
        noise = rng.normal(0.0, base_vol) * (0.85 + 0.25 * np.sin(idx / 3.0))
        ret = drift + noise

        prev_price = price
        price = max(1600.0, prev_price * (1.0 + ret))
        intraday_noise = abs(noise) * 0.8 + 0.0012
        high = max(price, prev_price) * (1.0 + intraday_noise)
        low = min(price, prev_price) * (1.0 - intraday_noise)

        panic = np.clip(
            panic_floor
            + max(0.0, -support_drift) * 9.0
            + max(0.0, rumor_pulse) * 8.0
            + max(0.0, -ret) * 11.0
            - max(0.0, support_drift) * 4.0
            + rng.normal(0.0, 0.015),
            0.08,
            0.86,
        )
        csad = np.clip(0.05 + 0.11 * panic + abs(ret) * 5.5, 0.04, 0.21)
        volume = 1_150_000 * (
            1.0
            + 0.5 * abs(score)
            + 0.22 * idx / periods
            + 0.35 * panic
            + max(0.0, shock.liquidity_injection) * 0.7
        )

        rows.append(
            {
                "step": idx,
                "time": dt.strftime("%Y-%m-%d"),
                "open": round(prev_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": round(volume, 2),
                "csad": round(float(csad), 4),
                "panic_level": round(float(panic), 4),
            }
        )

    return pd.DataFrame(rows)


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _render_chip_row(items: List[str], empty_text: str = "暂无") -> str:
    if not items:
        return f'<span class="tag-chip muted">{empty_text}</span>'
    return "".join(f'<span class="tag-chip">{item}</span>' for item in items)


def _compute_policy_summary(metrics: pd.DataFrame) -> Dict[str, float]:
    close = metrics["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    drawdown = close / close.cummax() - 1.0
    return {
        "return_pct": float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0),
        "avg_panic": float(metrics["panic_level"].mean()),
        "max_panic": float(metrics["panic_level"].max()),
        "avg_csad": float(metrics["csad"].mean()),
        "max_drawdown": float(abs(drawdown.min())),
        "avg_volume": float(metrics["volume"].mean()),
        "volatility": float(returns.std()),
    }


def _build_policy_cards(
    *,
    policy_title: str,
    policy_type: str,
    shock: PolicyShock,
    metrics: pd.DataFrame,
    control_summary: Optional[Dict[str, float]] = None,
) -> List[PolicyNarrativeCard]:
    start_close = float(metrics.iloc[0]["close"])
    end_close = float(metrics.iloc[-1]["close"])
    change = end_close / max(start_close, 1e-9) - 1.0
    avg_panic = float(metrics["panic_level"].mean())
    avg_csad = float(metrics["csad"].mean())
    score = _shock_score(shock)

    news_tone = "利好" if score >= 0 else "偏空"
    quant_view = "上行偏强" if change > 0.02 else "弱势承压" if change < -0.02 else "区间震荡"
    risk_view = "风险可控" if avg_panic < 0.32 else "需要重点盯防" if avg_panic < 0.52 else "高压预警"
    manager_action = "分批加仓" if change > 0.03 else "保持观察" if abs(change) < 0.015 else "先防守再观察"
    control_line = ""
    if control_summary is not None:
        diff = change - float(control_summary["return_pct"])
        control_line = f"相对对照组超额变化：{_format_percent(diff)}"

    return [
        PolicyNarrativeCard(
            title="新闻分析师",
            summary=f"{policy_title}在舆论层面会被市场解读为“{news_tone}”政策。",
            bullets=[
                f"政策类型：{policy_type}",
                f"情绪冲击：{shock.sentiment_delta:+.2f}，谣言扰动：{shock.rumor_shock:+.2f}",
                "传播速度会先影响成交情绪，再影响指数方向。",
            ],
            tone="good" if score >= 0 else "warn",
        ),
        PolicyNarrativeCard(
            title="量化分析师",
            summary=f"仿真结果显示市场大概率处于“{quant_view}”状态。",
            bullets=[
                f"区间累计变化：{_format_percent(change)}",
                f"平均羊群度：{avg_csad:.3f}",
                control_line or f"流动性注入强度：{shock.liquidity_injection:.2f}",
            ],
            tone="good" if change >= 0 else "warn",
        ),
        PolicyNarrativeCard(
            title="风险分析师",
            summary=f"当前市场处于“{risk_view}”区间。",
            bullets=[
                f"平均风险热度：{avg_panic:.2f}",
                f"信用利差冲击：{shock.credit_spread_delta:+.3f}",
                "若舆情再度放大，波动通常会先于价格扩张。",
            ],
            tone="bad" if avg_panic > 0.52 else "warn" if avg_panic > 0.32 else "good",
        ),
        PolicyNarrativeCard(
            title="经理最终决定",
            summary=f"建议采用“{manager_action}”，而不是一次性重仓。",
            bullets=[
                "先观察政策落地后的流动性改善是否持续。",
                "若风险热度重新升高，优先保护流动性缓冲。",
                "对外汇报时应强调：政策先改变情绪，再改变价格路径。",
            ],
            tone="good" if change >= 0 else "warn",
        ),
    ]


def _render_policy_cards(cards: List[PolicyNarrativeCard]) -> None:
    cols = st.columns(len(cards))
    tone_map = {
        "good": "#1d4d39",
        "warn": "#5b4213",
        "bad": "#5b1d24",
        "neutral": "#16243b",
    }
    for idx, card in enumerate(cards):
        bullets = "".join(f"<li>{item}</li>" for item in card.bullets)
        with cols[idx]:
            st.markdown(
                f"""
                <div class="story-card" style="border-top:3px solid {tone_map.get(card.tone, '#16243b')};">
                  <div class="story-card-title">{card.title}</div>
                  <div class="story-card-summary">{card.summary}</div>
                  <ul class="story-card-list">{bullets}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_flow_strip() -> None:
    st.markdown(
        """
        <div style="padding: 18px; border-radius: 8px; background: rgba(24,144,255,0.06); border: 1px solid rgba(24,144,255,0.3); box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <h4 style="margin-top:0; color: #1890ff; font-weight: 600;">⚙️ 系统底层推演链条</h4>
            <p style="color: #e2e8f0; line-height: 1.8; font-size: 15px; margin-bottom: 0;">
            1. <b>意图智能解析</b>：您输入的纯净政策文本，由 ManagerAgent 自动解析为主机底层所需的具体宏观参数（例如流动性释放倍数、利率降幅等）。<br>
            2. <b>网络情绪扩散</b>：宏观变化立刻投放进虚拟的微缩社交网络中，模拟因为新政出台引发的全民“情绪传染”与“羊群炒作”。<br>
            3. <b>微观主体调整</b>：大模型驱动的各行业虚拟公司、散户、机构庄家感知到热度与基本面的变化，各自调整风险偏好与投资组合。<br>
            4. <b>高频订单碰撞</b>：系统每秒将所有主体的交易冲动汇聚为海量限价单（LOB），在 C++ 高性能撮合引擎中完成毫无主观干预的毫秒级撮合，最后向上反馈生成顶部那根逼真的 K 线。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_operational_brief(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    primary = result["primary"]
    summary = primary["summary"]
    shock = primary["shock"]
    control = result.get("control")
    diff_text = "当前未启用对照组。"
    if control:
        diff_return = summary["return_pct"] - control["summary"]["return_pct"]
        diff_text = f"相对对照组超额变化：{_format_percent(diff_return)}"

    if summary["return_pct"] >= 0.03 and summary["avg_panic"] < 0.35:
        conclusion = "这项政策更像“稳预期 + 稳流动性”的组合，适合拿来做会前推演和对外口径准备。"
        action = "建议先做基线版与增强版两组对比，重点看情绪回暖是否可持续。"
    elif summary["return_pct"] >= 0:
        conclusion = "这项政策能提供一定支撑，但更适合做温和托底，不适合包装成强刺激。"
        action = "建议把冲击强度和仿真时长分成两档，再比较政策滞后效应。"
    else:
        conclusion = "这项政策在当前参数下更像压力测试输入，适合用来评估风险暴露，而不是直接作为稳市方案。"
        action = "建议补充配套流动性动作，或开启对照组评估是否存在副作用放大。"

    owner_map = {
        "税收调整": "财政、税务、资本市场研究部门",
        "流动性投放": "央行、金融监管、市场稳定资金团队",
        "财政刺激": "财政、发改、产业政策研究团队",
        "监管收紧": "金融监管、风控、法务与合规团队",
        "市场稳定措施": "监管协调组、稳定资金、交易监测团队",
        "自定义政策": "政策研究室、综合改革试点团队",
    }

    boundary_lines = [
        f"本轮窗口：{result['duration_days']} 个交易日，舆情扰动：{'已开启' if result['rumor_noise'] else '未开启'}",
        f"平均风险热度：{summary['avg_panic']:.2f}，平均羊群度：{summary['avg_csad']:.3f}",
        "结果适合做政策讨论、压力预演和汇报示意，不等同于正式价格预测。",
    ]

    return [
        {
            "title": "工程结论",
            "summary": conclusion,
            "lines": [
                f"区间累计变化：{_format_percent(summary['return_pct'])}",
                f"流动性注入：{shock['liquidity_injection']:+.2f}，情绪变化：{shock['sentiment_delta']:+.2f}",
                diff_text,
            ],
        },
        {
            "title": "适用部门",
            "summary": owner_map.get(result["policy_type"], "政策研究与跨部门会商团队"),
            "lines": [
                "适合放在政策会商前，用来快速比较不同政策版本的市场反应。",
                "也适合给答辩或汇报场景提供一张“为什么会这样走”的解释图。",
                action,
            ],
        },
        {
            "title": "运行边界",
            "summary": "当前结果体现的是一个可解释、可复盘的实验世界，不是实时预测引擎。",
            "lines": boundary_lines,
        },
    ]


def _render_operational_brief(cards: List[Dict[str, Any]]) -> None:
    cols = st.columns(len(cards))
    for idx, card in enumerate(cards):
        items = "".join(f"<li>{line}</li>" for line in card["lines"])
        with cols[idx]:
            st.markdown(
                f"""
                <div class="ops-card">
                  <div class="ops-card-title">{card['title']}</div>
                  <div class="ops-card-summary">{card['summary']}</div>
                  <ul class="ops-card-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_policy_header() -> None:
    st.markdown(
        """
        <div class="hero-panel">
          <div class="hero-kicker">Policy Sandbox</div>
          <h1>政府政策试验台</h1>
          <p>输入一项政策，用多智能体视角观察市场、情绪与风险如何联动变化。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    entries = [
        ("适用场景", "测试一项新政策如何改变指数、风险和情绪路径。"),
        ("主要输出", "仿真大盘、政策传导链、AI 文字解读、多智能体协作流程。"),
        ("工程价值", "把政策文本翻译成可讨论、可复盘、可导出的实验结果。"),
    ]
    for idx, (title, body) in enumerate(entries):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-label">{title}</div>
                  <div class="summary-note">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _select_template(templates: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = ["全部"] + sorted({str(item.get("category", "其他")) for item in templates})
    col1, col2 = st.columns([0.9, 1.4])
    with col1:
        category = st.selectbox("模板分类", options=categories, index=0)
    filtered = [item for item in templates if category == "全部" or item.get("category") == category]
    labels = {item["title"]: item for item in filtered}
    with col2:
        selected_label = st.selectbox("政策模板库", options=list(labels.keys()), index=0)
    selected = labels[selected_label]
    toolbox_count = len(templates)
    category_count = len({str(item.get("category", "其他")) for item in templates})
    st.caption(f"政策工具箱已内置 {toolbox_count} 条模板，覆盖 {category_count} 类政策工具。")

    meta_left, meta_right = st.columns([1.2, 1.0])
    with meta_left:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">模板目标</div>
              <div class="summary-value">{selected['title']}</div>
              <div class="summary-note">{selected['policy_goal']}</div>
              <div class="summary-label">适用阶段</div>
              <div class="summary-note">{selected.get('use_stage', '常规政策研判')}</div>
              <div class="summary-label">适用部门</div>
              <div class="summary-note">{selected['suitable_departments']}</div>
              <div class="summary-label">建议对照</div>
              <div class="summary-note">{selected['control_label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with meta_right:
        st.markdown(
            f"""
            <div class="summary-card toolbox-card">
              <div class="summary-label">关键工具</div>
              <div class="chip-row">{_render_chip_row(list(selected.get('key_tools', [])))}</div>
              <div class="summary-label">主要传导</div>
              <div class="chip-row">{_render_chip_row(list(selected.get('expected_channels', [])))}</div>
              <div class="summary-label">风险关注</div>
              <div class="chip-row">{_render_chip_row(list(selected.get('risk_focus', [])))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return selected


def _build_control_policy(
    *,
    template: Dict[str, Any],
    control_mode: str,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
) -> Optional[Dict[str, Any]]:
    if control_mode == "不启用对照组":
        return None
    if control_mode == "无政策基线":
        return {
            "label": "无政策基线",
            "policy_text": "维持现有政策安排，不新增刺激或收紧动作。",
            "intensity": 0.35,
            "duration_days": duration_days,
            "rumor_noise": False,
        }
    if control_mode == "模板建议对照":
        return {
            "label": str(template.get("control_label", "模板建议对照")),
            "policy_text": str(template.get("control_text", "维持当前政策安排，不新增政策动作。")),
            "intensity": max(0.4, float(template.get("recommended_intensity", intensity)) * 0.65),
            "duration_days": duration_days,
            "rumor_noise": bool(template.get("default_rumor_noise", False)),
        }
    if control_mode == "温和版本":
        return {
            "label": "温和版本",
            "policy_text": policy_text,
            "intensity": max(0.3, intensity * 0.6),
            "duration_days": duration_days,
            "rumor_noise": False,
        }
    return {
        "label": "风险压力版本",
        "policy_text": f"{policy_text} 同时伴随谣言扩散和风险偏好回落。",
        "intensity": max(0.8, intensity),
        "duration_days": duration_days,
        "rumor_noise": True,
    }


def _build_policy_run(
    *,
    label: str,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    enable_structured_parser: bool = True,
    event_store: Optional[EventStore] = None,
    dataset_version: str = "",
    snapshot_id: str = "",
    event_store_enabled: bool = False,
) -> Dict[str, Any]:
    shock, package = _compile_policy_bundle(
        policy_text,
        intensity,
        enable_structured_parser=enable_structured_parser,
    )
    package_dict = package.to_dict()
    reproducibility = {
        "seed": int(package.metadata.get("seed", 0)),
        "config_hash": str(package.metadata.get("config_hash", "")),
        "snapshot_info": dict(package.metadata.get("snapshot_info", {})),
    }
    transmission_layers = {
        "sector": package.sector_effects,
        "factor": package.factor_effects,
        "agent_class": package.agent_class_effects,
        "market": package.market_effects,
        "top_layers": package.top_layers(),
    }
    metrics = _generate_policy_metrics(
        policy_text=policy_text,
        intensity=intensity,
        duration_days=duration_days,
        rumor_noise=rumor_noise,
        scenario_key=label,
    )
    event_store_rows = 0
    if event_store_enabled and event_store is not None and dataset_version:
        policy_event = EventRecord(
            event_type=EventType.POLICY,
            timestamp=_to_iso_safe(metrics["time"].iloc[0]) if not metrics.empty else _to_iso_safe(None),
            visibility_time=_to_iso_safe(metrics["time"].iloc[0]) if not metrics.empty else _to_iso_safe(None),
            source="policy_lab",
            confidence=0.85,
            payload=package_dict.get("event", {}),
            metadata={
                "label": label,
                "parser_mode": package.uncertainty.parser_mode,
                "snapshot_id": snapshot_id,
            },
        )
        bar_events: List[EventRecord] = []
        for _, row in metrics.iterrows():
            ts = _to_iso_safe(row.get("time"))
            bar_events.append(
                EventRecord(
                    event_type=EventType.MARKET_BAR,
                    timestamp=ts,
                    visibility_time=ts,
                    source="policy_lab_simulation",
                    confidence=0.80,
                    payload={
                        "open": float(row.get("open", 0.0)),
                        "high": float(row.get("high", 0.0)),
                        "low": float(row.get("low", 0.0)),
                        "close": float(row.get("close", 0.0)),
                        "volume": float(row.get("volume", 0.0)),
                        "step": int(row.get("step", 0)),
                    },
                    metadata={"label": label, "snapshot_id": snapshot_id},
                )
            )
        event_store.append_events(
            dataset_version,
            [policy_event, *bar_events],
            seed=int(reproducibility.get("seed", 0)),
            config_hash=str(reproducibility.get("config_hash", "")),
            snapshot_id=snapshot_id,
        )
        event_store_rows = 1 + len(bar_events)
    return {
        "label": label,
        "policy_text": policy_text,
        "intensity": intensity,
        "duration_days": duration_days,
        "rumor_noise": rumor_noise,
        "shock": shock.to_dict(),
        "policy_event": package_dict["event"],
        "policy_package": package_dict,
        "transmission_layers": transmission_layers,
        "reproducibility": reproducibility,
        "parser_mode": package.uncertainty.parser_mode,
        "metrics": metrics,
        "summary": _compute_policy_summary(metrics),
        "event_store": {
            "enabled": bool(event_store_enabled),
            "dataset_version": dataset_version,
            "snapshot_id": snapshot_id,
            "rows_written": int(event_store_rows),
        },
    }


def _render_policy_comparison_chart(primary: Dict[str, Any], control: Optional[Dict[str, Any]]) -> None:
    if not control:
        return
    frame = pd.DataFrame(
        {
            "time": primary["metrics"]["time"],
            "main_close": primary["metrics"]["close"],
            "control_close": control["metrics"]["close"],
            "main_panic": primary["metrics"]["panic_level"],
            "control_panic": control["metrics"]["panic_level"],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["main_close"],
            mode="lines",
            name="当前政策方案",
            line=dict(color="#35d07f", width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["control_close"],
            mode="lines",
            name=control["label"],
            line=dict(color="#8ec5ff", width=2.3, dash="dash"),
        )
    )
    fig.update_layout(
        **dashboard_ui.PLOTLY_DARK_LAYOUT,
        title=dict(text="政策方案 vs 对照组", font=dict(color="#e2e8f0", size=18)),
        yaxis=dict(title=dict(text="指数点位", font=dict(color="#8aa0c2")), tickfont=dict(color="#e2e8f0")),
        xaxis=dict(title=dict(text="时间", font=dict(color="#8aa0c2")), tickfont=dict(color="#e2e8f0")),
        legend=dict(orientation="h", font=dict(color="#e2e8f0")),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True, key="policy_lab_compare_chart")
    dashboard_ui.export_plot_bundle(fig, frame, "policy_lab_compare", "policy_lab_compare")


def _render_control_delta_cards(primary: Dict[str, Any], control: Optional[Dict[str, Any]]) -> None:
    if not control:
        return
    main = primary["summary"]
    base = control["summary"]
    cards = [
        ("相对收益", _format_percent(main["return_pct"] - base["return_pct"]), "当前方案与对照组的累计收益差"),
        ("风险改善", _format_percent(base["max_drawdown"] - main["max_drawdown"]), "若为正值，说明当前方案回撤更小"),
        ("情绪改善", f"{base['avg_panic'] - main['avg_panic']:+.2f}", "若为正值，说明当前方案平均风险热度更低"),
        ("流动性改善", f"{main['avg_volume'] / max(base['avg_volume'], 1e-9) - 1.0:+.1%}", "当前方案相对对照组的平均成交活跃度"),
    ]
    cols = st.columns(len(cards))
    for idx, (title, value, note) in enumerate(cards):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-title">{title}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_policy_report(result: Dict[str, Any]) -> Dict[str, Any]:
    primary = result["primary"]
    control = result.get("control")
    cards = result["cards"]
    ops_cards = result["ops_cards"]
    report_meta = official_report_meta("policy_lab", result["policy_title"])
    template = result["template"]
    conclusion_card = ops_cards[0] if ops_cards else {"summary": "", "lines": []}
    boundary_card = ops_cards[-1] if ops_cards else {"summary": "", "lines": []}
    control_delta = None
    if control:
        control_delta = primary["summary"]["return_pct"] - control["summary"]["return_pct"]

    payload = {
        "report_meta": report_meta,
        "policy_title": result["policy_title"],
        "policy_type": result["policy_type"],
        "policy_text": result["policy_text"],
        "template_title": template["title"],
        "template_category": template["category"],
        "template_stage": template.get("use_stage"),
        "template_key_tools": template.get("key_tools", []),
        "template_channels": template.get("expected_channels", []),
        "template_risk_focus": template.get("risk_focus", []),
        "primary_summary": primary["summary"],
        "primary_shock": primary["shock"],
        "primary_policy_event": primary.get("policy_event", {}),
        "primary_policy_package": primary.get("policy_package", {}),
        "primary_transmission_layers": primary.get("transmission_layers", {}),
        "primary_reproducibility": primary.get("reproducibility", {}),
        "primary_parser_mode": primary.get("parser_mode", "legacy"),
        "structured_parser_enabled": bool(result.get("structured_parser_enabled", True)),
        "control_summary": control["summary"] if control else None,
        "control_label": control["label"] if control else None,
        "operational_brief": ops_cards,
        "ai_cards": [{"title": card.title, "summary": card.summary, "bullets": card.bullets} for card in cards],
    }
    markdown = [
        f"# 关于《{result['policy_title']}》仿真情况的汇报摘要",
        "",
        f"- 报告编号：{report_meta['report_no']}",
        f"- 生成日期：{report_meta['date_cn']}",
        f"- 报送对象：{report_meta['recipient']}",
        f"- 材料性质：{report_meta['classification']}",
        "",
        "## 一、事项概述",
        f"- 政策名称：{result['policy_title']}",
        f"- 政策类型：{result['policy_type']}",
        f"- 模板来源：{template['title']} / {template['category']}",
        f"- 适用阶段：{template.get('use_stage', '常规政策研判')}",
        f"- 仿真时长：{result['duration_days']} 个交易日",
        f"- 舆情扰动：{'开启' if result['rumor_noise'] else '关闭'}",
        f"- 对照组模式：{result['control_mode']}",
        "",
        "## 二、政策内容",
        result["policy_text"],
        "",
        "## 三、政策传导链路",
        f"- 解析模式：{primary.get('parser_mode', 'legacy')}",
        f"- 政策事件：{primary.get('policy_event', {}).get('policy_label', 'Unknown')} / {primary.get('policy_event', {}).get('policy_type', 'unclassified')}",
        f"- 传导通道：{', '.join([str(item.get('name', 'channel')) for item in primary.get('policy_package', {}).get('channels', [])]) or 'n/a'}",
        f"- 主体反应：{', '.join([f'{k}={v:+.3f}' for k, v in primary.get('transmission_layers', {}).get('agent_class', {}).items()]) or 'n/a'}",
        f"- 市场结果：{', '.join([f'{k}={v:+.3f}' for k, v in primary.get('transmission_layers', {}).get('market', {}).items()]) or 'n/a'}",
        "",
        "## 四、核心结论",
        f"- 总体判断：{conclusion_card['summary']}",
        f"- 市场累计变化：{_format_percent(primary['summary']['return_pct'])}",
        f"- 风险热度水平：{primary['summary']['avg_panic']:.2f}",
        f"- 最大回撤：{_format_percent(primary['summary']['max_drawdown'])}",
        f"- 工具传导重点：{', '.join(template.get('expected_channels', [])) or '市场情绪与流动性传导'}",
    ]
    if control and control_delta is not None:
        markdown.append(f"- 相对对照组超额变化：{_format_percent(control_delta)}")
    markdown.extend(
        [
            "",
            "## 五、仿真结果",
        ]
    )
    markdown.extend(
        [
        f"- 当前方案累计变化：{_format_percent(primary['summary']['return_pct'])}",
        f"- 最大回撤：{_format_percent(primary['summary']['max_drawdown'])}",
        f"- 平均风险热度：{primary['summary']['avg_panic']:.2f}",
        f"- 平均羊群度：{primary['summary']['avg_csad']:.3f}",
        f"- 平均成交活跃度：{primary['summary']['avg_volume']:.0f}",
        f"- 波动水平：{primary['summary']['volatility']:.2%}",
        ]
    )
    if control:
        markdown.extend(
            [
                "",
                "## 六、对照组比较",
                f"- 对照组：{control['label']}",
                f"- 对照组累计变化：{_format_percent(control['summary']['return_pct'])}",
                f"- 相对对照组超额变化：{_format_percent(primary['summary']['return_pct'] - control['summary']['return_pct'])}",
                f"- 对照组平均风险热度：{control['summary']['avg_panic']:.2f}",
                f"- 对照组最大回撤：{_format_percent(control['summary']['max_drawdown'])}",
            ]
        )
    markdown.extend(["", "## 七、AI 研判意见"])
    for card in cards:
        markdown.append(f"### {card.title}")
        markdown.append(f"- 研判结论：{card.summary}")
        for line in card.bullets:
            markdown.append(f"- {line}")
    markdown.extend(["", "## 八、风险提示与边界"])
    markdown.append(f"- {boundary_card['summary']}")
    for line in boundary_card["lines"]:
        markdown.append(f"- {line}")
    markdown.extend(["", "## 九、建议事项"])
    for card in ops_cards:
        markdown.append(f"- {card['title']}：{card['summary']}")
        for line in card["lines"]:
            markdown.append(f"- {line}")

    export_bundle = write_report_artifacts(
        root_dir=POLICY_REPORT_DIR,
        report_type="policy_lab",
        title=result["policy_title"],
        markdown_text="\n".join(markdown),
        payload=payload,
    )
    export_bundle["report_meta"] = report_meta
    return export_bundle


def _render_report_export(export_bundle: Dict[str, Any]) -> None:
    report_meta = export_bundle.get("report_meta", {})
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">已生成正式汇报摘要</div>
              <div class="summary-value">{report_meta.get('title', export_bundle['stem'])}</div>
              <div class="summary-note">报告编号：{report_meta.get('report_no', export_bundle['stem'])}</div>
              <div class="summary-note">报送对象：{report_meta.get('recipient', '内部研究与评估团队')}</div>
              <div class="summary-note">现已生成带封面、固定页眉页脚的 Markdown / Word / PDF / JSON，可直接用于会商材料、汇报初稿和归档附件。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        top_row = st.columns(2)
        bottom_row = st.columns(2)
        with top_row[0]:
            st.download_button(
                "下载 Word 汇报",
                data=export_bundle["docx_bytes"],
                file_name=f"{export_bundle['stem']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key=f"policy_docx_{export_bundle['stem']}",
            )
        with top_row[1]:
            st.download_button(
                "下载 PDF 汇报",
                data=export_bundle["pdf_bytes"],
                file_name=f"{export_bundle['stem']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"policy_pdf_{export_bundle['stem']}",
            )
        with bottom_row[0]:
            st.download_button(
                "下载 Markdown 汇报",
                data=export_bundle["markdown_text"],
                file_name=f"{export_bundle['stem']}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"policy_md_{export_bundle['stem']}",
            )
        with bottom_row[1]:
            st.download_button(
                "下载 JSON 附件",
                data=export_bundle["json_text"],
                file_name=f"{export_bundle['stem']}.json",
                mime="application/json",
                use_container_width=True,
                key=f"policy_json_{export_bundle['stem']}",
            )


def render_policy_lab() -> None:
    _render_policy_header()
    with st.expander("如何设计一条有工程价值的政策实验"):
        st.markdown(
            """
            - 先写清楚政策动作本身，例如“下调印花税并释放流动性支持”。
            - 再说明你关心的目标，是稳指数、降波动、提流动性还是压恐慌。
            - 最后务必给一个对照组，这样结果才适合拿去做会商和汇报。
            """
        )

    templates = _load_policy_templates()
    st.markdown("### 政策工具箱")
    selected_template = _select_template(templates)

    if "policy_lab_result" not in st.session_state:
        st.session_state.policy_lab_result = None

    with st.form("policy_lab_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            policy_title = st.text_input("政策名称", value=str(selected_template["title"]))
            policy_text = st.text_area("政策说明", value=str(selected_template["policy_text"]), height=120)
        with col2:
            policy_type = st.selectbox(
                "政策类型",
                options=list(POLICY_TYPE_OPTIONS.keys()),
                index=list(POLICY_TYPE_OPTIONS.keys()).index(str(selected_template["policy_type"])),
            )
            intensity = st.slider("冲击强度", min_value=0.2, max_value=1.6, value=float(selected_template.get("recommended_intensity", 1.0)), step=0.1)
            duration_days = st.slider("仿真时长（交易日）", min_value=12, max_value=60, value=int(selected_template.get("recommended_duration", 30)), step=6)
            rumor_noise = st.toggle("加入舆情扰动", value=bool(selected_template.get("default_rumor_noise", False)))
            structured_parser_enabled = st.toggle("启用结构化政策解析", value=True, help="默认优先 structured parser，关闭时回落到 legacy parser。")
            enable_event_store = st.toggle("Enable EventStore write", value=False)
            dataset_version = st.text_input("EventStore dataset_version", value="default")
            snapshot_id = st.text_input("Snapshot id (optional)", value="")

        st.markdown("#### 对照组设置")
        control_mode = st.selectbox("对照组模式", options=CONTROL_MODE_OPTIONS, index=2)
        submitted = st.form_submit_button("开始仿真", use_container_width=True, type="primary")

    if submitted:
        event_store = EventStore() if enable_event_store else None
        primary = _build_policy_run(
            label="primary",
            policy_text=policy_text,
            intensity=intensity,
            duration_days=duration_days,
            rumor_noise=rumor_noise,
            enable_structured_parser=structured_parser_enabled,
            event_store=event_store,
            dataset_version=dataset_version,
            snapshot_id=snapshot_id,
            event_store_enabled=bool(enable_event_store),
        )
        control_config = _build_control_policy(
            template=selected_template,
            control_mode=control_mode,
            policy_text=policy_text,
            intensity=intensity,
            duration_days=duration_days,
            rumor_noise=rumor_noise,
        )
        control = None
        if control_config is not None:
            control = _build_policy_run(
                label=str(control_config["label"]),
                policy_text=str(control_config["policy_text"]),
                intensity=float(control_config["intensity"]),
                duration_days=int(control_config["duration_days"]),
                rumor_noise=bool(control_config["rumor_noise"]),
                enable_structured_parser=structured_parser_enabled,
                event_store=event_store,
                dataset_version=dataset_version,
                snapshot_id=snapshot_id,
                event_store_enabled=bool(enable_event_store),
            )
        shock = _compile_scaled_shock(policy_text, intensity, enable_structured_parser=structured_parser_enabled)
        cards = _build_policy_cards(
            policy_title=policy_title,
            policy_type=policy_type,
            shock=shock,
            metrics=primary["metrics"],
            control_summary=control["summary"] if control else None,
        )
        result = {
            "template": selected_template,
            "policy_title": policy_title,
            "policy_text": policy_text,
            "policy_type": policy_type,
            "intensity": intensity,
            "duration_days": duration_days,
            "rumor_noise": rumor_noise,
            "structured_parser_enabled": structured_parser_enabled,
            "event_store_enabled": bool(enable_event_store),
            "dataset_version": dataset_version,
            "snapshot_id": snapshot_id,
            "control_mode": control_mode,
            "primary": primary,
            "control": control,
            "cards": cards,
        }
        result["ops_cards"] = _build_operational_brief(result)
        result["export_bundle"] = _build_policy_report(result)
        st.session_state.policy_lab_result = result

    result = st.session_state.policy_lab_result
    if not result:
        st.info("提示：请先在“参数配置”面板选定或输入您的政策方案，然后点击“开始仿真”。")
        st.markdown("### 历史大盘基准 (前30日走势)")
        dashboard_ui.render_empty_market_board(key_prefix="policy_lab_empty")
        return

    primary = result["primary"]
    control = result.get("control")
    metrics = primary["metrics"]
    current_step = int(metrics.iloc[-1]["step"])

    st.markdown("### 大盘走势与关键风险点全景")
    _render_policy_comparison_chart(primary, control)
    if control:
        _render_control_delta_cards(primary, control)

    st.markdown("### 政策 -> 传导通道 -> 主体反应 -> 市场结果")
    _render_policy_transmission_path(primary)

    # 渲染大盘主图及KPI（独占整宽）
    kpi = dashboard_ui.build_kpi_snapshot(metrics, current_step, regulation_hint="政策观察期")
    dashboard_ui.render_kpi_cards(kpi)
    dashboard_ui.render_market_overview(metrics, current_step, key_prefix="policy_lab_full")

    # 横向布局的“政策概览与成效对比”
    st.markdown("### 政策宏观摘要与回撤对照")
    primary_summary = primary["summary"]
    if control:
        c1, c2, c3 = st.columns(3)
        control_summary = control["summary"]
    else:
        c1, c2 = st.columns(2)
        c3 = None

    with c1:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">方案定性摘要</div>
              <div class="summary-value" style="font-size: 1.15em; color: #1890ff; font-weight: bold;">{result['policy_title']}</div>
              <div class="summary-note">模板来源：{result['template']['category']}</div>
              <div class="summary-note" style="margin-top: 5px;">简述：{result['policy_text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">当前运行期核心指标</div>
              <div class="summary-note">累计变动率：{_format_percent(primary_summary['return_pct'])}</div>
              <div class="summary-note">探测最大回撤：<span style="color: #f5222d;">{_format_percent(primary_summary['max_drawdown'])}</span></div>
              <div class="summary-note">盘面羊群共识指数：{primary_summary['avg_csad']:.3f}</div>
              <div class="summary-note">平均风险热度中枢：{primary_summary['avg_panic']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if c3:
        with c3:
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-label">平移对照组 ({control['label']})</div>
                  <div class="summary-note">累计变动率：{_format_percent(control_summary['return_pct'])}</div>
                  <div class="summary-note">最大回撤：{_format_percent(control_summary['max_drawdown'])}</div>
                  <div class="summary-note">平均风险热度中枢：{control_summary['avg_panic']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 工程应用提示")
    _render_operational_brief(result["ops_cards"])

    st.markdown("### AI 如何解读这项政策")
    _render_policy_cards(result["cards"])

    st.markdown("### 多智能体协作流程")
    _render_flow_strip()

    detail_left, detail_right = st.columns(2)
    with detail_left:
        primary_shock = primary["shock"]
        dashboard_ui.render_policy_transmission_chain(
            {
                "policy": result["policy_text"],
                "macro_variables": {
                    "inflation": 0.021,
                    "unemployment": 0.048 - primary_shock["fiscal_stimulus_delta"] * 0.05,
                    "wage_growth": 0.028 + primary_shock["fiscal_stimulus_delta"] * 0.04,
                    "credit_spread": 0.015 + primary_shock["credit_spread_delta"],
                    "liquidity_index": 1.0 + primary_shock["liquidity_injection"],
                    "policy_rate": 0.022 + primary_shock["policy_rate_delta"],
                    "fiscal_stimulus": primary_shock["fiscal_stimulus_delta"],
                    "sentiment_index": 0.55 + primary_shock["sentiment_delta"] + primary_shock["rumor_shock"],
                },
                "social_sentiment": {"mean": 0.35 + primary_shock["sentiment_delta"] + primary_shock["rumor_shock"]},
                "industry_agent": {
                    "avg_household_risk": 0.48 + primary_shock["sentiment_delta"] * 0.2,
                    "avg_firm_hiring": 0.12 + primary_shock["fiscal_stimulus_delta"] * 0.6,
                },
                "market_microstructure": {
                    "buy_volume": float(metrics["volume"].iloc[-1]) * (1.1 - float(metrics["panic_level"].iloc[-1])),
                    "sell_volume": float(metrics["volume"].iloc[-1]) * (0.45 + float(metrics["panic_level"].iloc[-1])),
                    "trade_count": int(float(metrics["volume"].iloc[-1]) / 1200),
                    "matching_mode": "policy_lab",
                },
            },
            key_prefix="policy_lab",
        )
    with detail_right:
        dashboard_ui.render_risk_event_timeline(metrics, key_prefix="policy_lab")

    st.markdown("### 导出汇报摘要")
    _render_report_export(result["export_bundle"])
