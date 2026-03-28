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

from core.event_store import EventRecord, EventStore, EventType
from core.macro.government import GovernmentAgent, PolicyShock
from policy.structured import PolicyPackage
from ui.reporting import official_report_meta, write_report_artifacts


POLICY_TYPE_OPTIONS = {
    "Tax Adjustment": "tax",
    "Liquidity Injection": "liquidity",
    "Fiscal Stimulus": "fiscal",
    "Regulatory Tightening": "tightening",
    "Market Stabilization": "stabilization",
    "Custom Policy": "custom",
}

TEMPLATE_LIBRARY_PATH = Path("data") / "policy_templates.json"
POLICY_REPORT_DIR = Path("outputs") / "policy_reports"
CONTROL_MODE_OPTIONS = [
    "No control arm",
    "No policy baseline",
    "Template recommendation control",
    "Mild variant",
    "Risk-stress variant",
]


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
            "category": "Market Stabilization",
            "title": "Cut stamp tax with liquidity support",
            "policy_type": "Tax Adjustment",
            "policy_text": "Reduce stamp tax and pair it with liquidity support to stabilize expectations.",
            "policy_goal": "Improve liquidity, reduce trading frictions, and stabilize index dynamics.",
            "suitable_departments": "Finance, Tax, Securities Regulator, Stabilization Fund",
            "recommended_intensity": 1.1,
            "recommended_duration": 30,
            "default_rumor_noise": False,
            "control_label": "Maintain current tax and liquidity setup",
            "control_text": "Keep current tax and liquidity arrangement without adding stabilization interventions.",
        },
        {
            "id": "targeted-fiscal-demand",
            "category": "Fiscal Support",
            "title": "Targeted fiscal expansion with sector focus",
            "policy_type": "Fiscal Stimulus",
            "policy_text": "Launch targeted fiscal spending for infrastructure and advanced manufacturing with phased implementation.",
            "policy_goal": "Stabilize growth expectations while preserving financial stability.",
            "suitable_departments": "Finance, Development and Reform, Industry, Local Government",
            "recommended_intensity": 1.0,
            "recommended_duration": 60,
            "default_rumor_noise": False,
            "control_label": "No targeted fiscal expansion",
            "control_text": "Keep fiscal stance unchanged as control arm.",
        },
        {
            "id": "rumor-refutation-stabilization",
            "category": "Expectation Management",
            "title": "Rumor refutation with stabilization statement",
            "policy_type": "Market Stabilization",
            "policy_text": "Issue official clarification to refute market rumors and release a coordinated stabilization communication package.",
            "policy_goal": "Reduce panic and suppress rumor-driven sell pressure.",
            "suitable_departments": "Regulator, Official Media, Exchange, Stability Fund",
            "recommended_intensity": 1.2,
            "recommended_duration": 20,
            "default_rumor_noise": True,
            "control_label": "No clarification response",
            "control_text": "Observe market dynamics without official clarification.",
        },
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


def _policy_feature_flags(enable_structured_parser: bool = True) -> Dict[str, bool]:
    return {
        "structured_policy_parser_v1": bool(enable_structured_parser),
        "policy_transmission_layers_v1": True,
        "policy_transmission_graph_v1": True,
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


def _generate_policy_metrics(*, policy_text: str, intensity: float, duration_days: int, rumor_noise: bool, scenario_key: str) -> pd.DataFrame:
    shock = _compile_scaled_shock(policy_text, intensity)
    score = _shock_score(shock)
    seed = f"{scenario_key}|{policy_text}|{intensity}|{duration_days}|{rumor_noise}"
    rng = np.random.default_rng(_seed_from_text(seed))

    periods = max(10, int(duration_days))
    dates = pd.bdate_range(pd.Timestamp.today().normalize(), periods=periods)
    price = 3000.0
    rows: List[Dict[str, float | int | str]] = []
    for idx, dt in enumerate(dates, start=1):
        drift = 0.0003 + np.clip(score, -1.0, 1.0) * 0.0025 * np.exp(-(idx - 1) / max(periods * 0.5, 1.0))
        rumor_term = (shock.rumor_shock * 0.008 if rumor_noise else 0.0) * np.exp(-(idx - 1) / max(periods * 0.25, 1.0))
        ret = drift + rumor_term + rng.normal(0.0, 0.004)
        prev = price
        price = max(1600.0, prev * (1.0 + ret))
        band = abs(ret) * 0.9 + 0.001
        high = max(prev, price) * (1 + band)
        low = min(prev, price) * (1 - band)
        panic = float(np.clip(0.2 + max(0.0, -ret) * 8.0 + max(0.0, rumor_term) * 6.0, 0.05, 0.95))
        csad = float(np.clip(0.05 + panic * 0.1 + abs(ret) * 4.5, 0.04, 0.22))
        volume = float(1_000_000 * (1 + 0.3 * abs(score) + 0.4 * panic))
        rows.append(
            {
                "step": idx,
                "time": dt.strftime("%Y-%m-%d"),
                "open": round(prev, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": round(volume, 2),
                "csad": round(csad, 4),
                "panic_level": round(panic, 4),
            }
        )
    return pd.DataFrame(rows)


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


def _build_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["time"], y=frame["close"], mode="lines+markers", name="指数收盘"))
    fig.add_trace(go.Bar(x=frame["time"], y=frame["volume"], name="成交量", opacity=0.25, yaxis="y2"))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="日期",
        yaxis_title="收盘价",
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def _policy_narrative_key(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    payload = {
        "policy_text": str(policy_text or ""),
        "summary": dict(summary or {}),
        "explanation": dict(package_dict.get("explanation", {}) or {}),
        "top_layers": dict(package_dict.get("top_layers", {}) or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _top_effect_text(items: List[Dict[str, Any]], limit: int = 3) -> str:
    if not items:
        return "暂无显著项"
    ranked = sorted(items, key=lambda item: abs(float(item.get("score", 0.0) or 0.0)), reverse=True)[:limit]
    parts: List[str] = []
    for item in ranked:
        name = str(item.get("name", "未命名"))
        score = float(item.get("score", 0.0) or 0.0)
        direction = "受益" if score >= 0 else "承压"
        parts.append(f"{name}（{direction}）")
    return "、".join(parts)


def _fallback_policy_narrative(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    explanation = dict(package_dict.get("explanation", {}) or {})
    headline = str(explanation.get("headline", "") or "政策通过多条传导链路影响市场预期与资金行为。")
    primary_path = [str(node) for node in explanation.get("primary_path", []) if str(node).strip()]
    path_text = " -> ".join(primary_path[:5]) if primary_path else "政策信号 -> 预期修复 -> 风险偏好变化 -> 价格反馈"
    lag_days = int(explanation.get("expected_lag_days", 0) or 0)
    side_effects = [str(item) for item in explanation.get("side_effects", []) if str(item).strip()]
    risk_tip = "、".join(side_effects[:2]) if side_effects else "短期波动可能放大，需关注情绪过冲。"
    return "\n".join(
        [
            f"**一句话结论**：{headline}",
            f"- 这项政策的核心目标是：{str(policy_text or '').strip()[:120]}",
            f"- 主要传导路径：{path_text}",
            f"- 重点受影响主体：{_top_effect_text(list(explanation.get('affected_agents', []) or []))}",
            f"- 重点受影响行业：{_top_effect_text(list(explanation.get('affected_sectors', []) or []))}",
            f"- 重点受影响因子：{_top_effect_text(list(explanation.get('affected_factors', []) or []))}",
            f"- 市场结果指向：{_top_effect_text(list(explanation.get('market_results', []) or []))}",
            f"- 仿真表现：收益率 {summary.get('return_pct', 0.0):.2%}，最大回撤 {summary.get('max_drawdown', 0.0):.2%}，波动率 {summary.get('volatility', 0.0):.4f}",
            f"- 预计传导时滞：约 {lag_days} 天；风险提示：{risk_tip}",
        ]
    )


def _llm_policy_narrative(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    try:
        from core.inference.api_backend import APIBackend
    except Exception:
        return ""
    snapshot = {
        "policy_text": str(policy_text or ""),
        "summary": dict(summary or {}),
        "explanation": dict(package_dict.get("explanation", {}) or {}),
        "policy_schema": dict(package_dict.get("policy_schema", {}) or {}),
        "top_layers": dict(package_dict.get("top_layers", {}) or {}),
    }
    prompt = "\n".join(
        [
            "请把下面的政策仿真结果写成面向评委的中文自然语言解读。",
            "要求：不要输出 JSON、代码块、键名；先给 1 句结论，再给 4-6 条要点，最后给 1 条风险提示和 2 个建议观察指标。",
            "请尽量使用简洁、可读、非技术化表达。",
            f"数据：{json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)}",
        ]
    )
    try:
        backend = APIBackend(model="deepseek-chat", max_tokens=520, temperature=0.3)
        response = str(
            backend.generate(
                prompt,
                system_prompt="你是政策仿真讲解专家，擅长把结构化结果翻译成评委一读就懂的自然语言。",
                timeout_budget=20.0,
            )
            or ""
        ).strip()
    except Exception:
        return ""
    if not response or response.startswith("[API Error]"):
        return ""
    if response.lstrip().startswith("{") or response.lstrip().startswith("["):
        return ""
    return response


def _get_policy_narrative(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    cache = st.session_state.setdefault("policy_lab_narrative_cache", {})
    key = _policy_narrative_key(policy_text, summary, package_dict)
    if key in cache:
        return str(cache[key])
    text = _llm_policy_narrative(policy_text, summary, package_dict)
    if not text:
        text = _fallback_policy_narrative(policy_text, summary, package_dict)
    cache[key] = text
    return text


def _persist_policy_event(
    *,
    selected_title: str,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
) -> None:
    timestamp = pd.Timestamp.utcnow().isoformat()
    record = EventRecord(
        timestamp=timestamp,
        visibility_time=timestamp,
        source="policy_lab_ui",
        confidence=1.0,
        event_type=EventType.POLICY,
        payload={
            "title": f"Policy scenario: {selected_title}",
            "policy_text": str(policy_text),
            "intensity": float(intensity),
            "duration_days": int(duration_days),
            "rumor_noise": bool(rumor_noise),
        },
        metadata={"module": "policy_lab"},
    )
    try:
        EventStore().append_events(dataset_version="policy_lab", events=[record])
    except Exception:
        # Event persistence is best-effort and should never block the demo flow.
        return


def render_policy_lab() -> None:
    st.subheader("政策实验台")

    templates = _load_policy_templates()
    template_map = {str(item.get("title", f"template-{idx}")): item for idx, item in enumerate(templates)}
    selected_title = st.selectbox("模板", options=list(template_map.keys()), index=0)
    selected = template_map[selected_title]

    policy_text = st.text_area("政策文本", value=str(selected.get("policy_text", "")), height=110)
    intensity = st.slider("政策强度", min_value=0.2, max_value=2.0, value=float(selected.get("recommended_intensity", 1.0)), step=0.1)
    duration_days = st.slider("持续天数", min_value=10, max_value=180, value=int(selected.get("recommended_duration", 30)), step=5)
    rumor_noise = st.checkbox("注入传言噪声", value=bool(selected.get("default_rumor_noise", False)))

    if st.button("运行政策场景", type="primary"):
        with st.spinner("正在运行政策仿真..."):
            frame = _generate_policy_metrics(
                policy_text=policy_text,
                intensity=float(intensity),
                duration_days=int(duration_days),
                rumor_noise=bool(rumor_noise),
                scenario_key=str(selected.get("id", selected_title)),
            )
            summary = _compute_policy_summary(frame)
            st.session_state.policy_lab_result = {
                "frame": frame,
                "summary": summary,
                "policy_text": policy_text,
                "template": selected,
            }
            _, package = _compile_policy_bundle(policy_text, float(intensity), policy_type_hint=str(selected.get("policy_type", "")))
            package_dict = package.to_dict()
            st.session_state.policy_lab_result["policy_package"] = package_dict

            report_payload = {
                "title": f"政策实验台 - {selected_title}",
                "summary": summary,
                "policy_text": policy_text,
                "metrics": frame.to_dict(orient="records"),
                "template": selected,
                "policy_schema": package_dict.get("policy_schema", {}),
                "transmission_graph": package_dict.get("transmission_graph", {}),
                "why_this_happened": package_dict.get("explanation", {}),
            }
            report_title = f"政策实验台 - {selected_title}"
            report_meta = official_report_meta("policy_lab", report_title)
            report_payload["report_meta"] = report_meta
            markdown_text = "\n".join(
                [
                    f"# {report_title}",
                    "",
                    f"- 报告编号：{report_meta['report_no']}",
                    f"- 生成日期：{report_meta['date_cn']}",
                    f"- 模板：{selected_title}",
                    f"- 政策强度：{float(intensity):.1f}",
                    f"- 持续天数：{int(duration_days)}",
                    f"- 传言噪声：{'是' if rumor_noise else '否'}",
                    "",
                    "## 核心指标",
                    f"- 收益率：{summary['return_pct']:.2%}",
                    f"- 平均恐慌度：{summary['avg_panic']:.4f}",
                    f"- 最大回撤：{summary['max_drawdown']:.2%}",
                    f"- 波动率：{summary['volatility']:.4f}",
                    "",
                    "## 政策文本",
                    policy_text,
                ]
            )
            bundle = write_report_artifacts(
                root_dir=POLICY_REPORT_DIR,
                report_type="policy_lab",
                title=report_title,
                markdown_text=markdown_text,
                payload=report_payload,
            )
            st.session_state.policy_lab_bundle = bundle

            _persist_policy_event(
                selected_title=selected_title,
                policy_text=policy_text,
                intensity=float(intensity),
                duration_days=int(duration_days),
                rumor_noise=bool(rumor_noise),
            )

    result = st.session_state.get("policy_lab_result")
    if not result:
        st.info("运行一个场景后，这里会生成政策传导结果和报告材料。")
        return

    frame = result["frame"]
    summary = result["summary"]

    cols = st.columns(4)
    cols[0].metric("收益率", f"{summary['return_pct'] * 100:.2f}%")
    cols[1].metric("平均恐慌度", f"{summary['avg_panic']:.3f}")
    cols[2].metric("最大回撤", f"{summary['max_drawdown'] * 100:.2f}%")
    cols[3].metric("波动率", f"{summary['volatility']:.4f}")

    st.plotly_chart(_build_chart(frame), use_container_width=True)

    package_dict = result.get("policy_package") or {}
    if package_dict:
        st.markdown("#### 成因解读（自然语言）")
        st.markdown(_get_policy_narrative(str(result.get("policy_text", "")), summary, package_dict))

    bundle = st.session_state.get("policy_lab_bundle")
    if bundle:
        st.caption(f"报告已导出：{bundle.get('json_path')}")
