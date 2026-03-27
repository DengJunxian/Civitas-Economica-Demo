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
from ui import dashboard as dashboard_ui
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
            report_meta = official_report_meta(module_name="policy_lab", title=f"政策实验台 - {selected_title}")
            bundle = write_report_artifacts(
                report_payload,
                meta=report_meta,
                output_dir=POLICY_REPORT_DIR,
                stem=f"policy_lab_{_seed_from_text(policy_text) % 10_000_000}",
            )
            st.session_state.policy_lab_bundle = bundle

            event_store = EventStore()
            event_store.append(
                EventRecord(
                    event_type=EventType.POLICY,
                    title=f"Policy scenario: {selected_title}",
                    payload={"policy_text": policy_text, "intensity": float(intensity), "duration_days": int(duration_days)},
                )
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
    dashboard_ui.render_market_snapshot(frame.rename(columns={"close": "price"}) if "price" not in frame.columns else frame)

    package_dict = result.get("policy_package") or {}
    explanation = package_dict.get("explanation", {})
    if package_dict:
        st.markdown("#### 成因解读")
        st.json(
            {
                "policy_schema": package_dict.get("policy_schema", {}),
                "transmission_graph": package_dict.get("transmission_graph", {}),
                "why_this_happened": explanation,
            },
            expanded=False,
        )

    bundle = st.session_state.get("policy_lab_bundle")
    if bundle:
        st.caption(f"报告已导出：{bundle.get('json_path')}")
