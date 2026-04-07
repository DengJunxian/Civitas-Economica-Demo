"""Dashboard widgets for competition-defense Streamlit UI."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import streamlit as st

from core.ui_text import display_risk_alert, translate_display_text, translate_ui_payload
from ui.narrative import narrate_payload


PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Microsoft YaHei"),
    margin=dict(l=20, r=20, t=48, b=20),
)


def _json_default(value: Any) -> Any:
    """
    统一处理导出 JSON 时的非常规类型，避免前端导出按钮触发序列化异常。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return value.to_dict()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    return str(value)


def metric_value_text(value: float, as_percent: bool = False) -> str:
    if as_percent:
        return f"{value:.2%}"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"


def export_plot_bundle(
    fig: go.Figure,
    data: Optional[pd.DataFrame],
    title_prefix: str,
    key_prefix: str,
    fig_json: Optional[Dict[str, Any]] = None,
) -> None:
    """Provide PNG/CSV/JSON export controls under every key chart."""
    c1, c2, c3 = st.columns(3)

    with c1:
        try:
            png_bytes = fig.to_image(format="png", width=1400, height=840, scale=2)
            st.download_button(
                "导出 PNG",
                data=png_bytes,
                file_name=f"{title_prefix}.png",
                mime="image/png",
                key=f"{key_prefix}_png",
                use_container_width=True,
            )
        except Exception:
            st.button("导出 PNG（需安装依赖）", key=f"{key_prefix}_png_disabled", disabled=True, use_container_width=True)
            st.caption("PNG 导出依赖未就绪，请安装 `kaleido` 后重启应用。")

    with c2:
        csv_text = ""
        if data is not None and not data.empty:
            csv_text = data.to_csv(index=False)
        st.download_button(
            "导出 CSV",
            data=csv_text,
            file_name=f"{title_prefix}.csv",
            mime="text/csv",
            key=f"{key_prefix}_csv",
            use_container_width=True,
            disabled=(data is None),
        )

    with c3:
        payload = fig_json if fig_json is not None else fig.to_dict()
        try:
            # Prefer Plotly's encoder for figure payloads that include typed arrays/internal wrappers.
            json_text = json.dumps(payload, ensure_ascii=False, indent=2, cls=PlotlyJSONEncoder)
        except Exception:
            try:
                json_text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            except Exception as exc:
                json_text = json.dumps(
                    {
                        "export_error": str(exc),
                        "payload_type": type(payload).__name__,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )
        st.download_button(
            "导出 JSON",
            data=json_text,
            file_name=f"{title_prefix}.json",
            mime="application/json",
            key=f"{key_prefix}_json",
            use_container_width=True,
        )


def render_kpi_cards(snapshot: Dict[str, Any]) -> None:
    cards = [
        ("流动性", snapshot.get("liquidity", 0.0), False, "越高越好"),
        ("波动率", snapshot.get("volatility", 0.0), True, "越低越稳"),
        ("羊群度", snapshot.get("herding", 0.0), False, "高值代表拥挤交易"),
        ("风险预警", snapshot.get("risk_alert", "GREEN"), None, "系统风险灯"),
        ("监管动作", snapshot.get("reg_action", "观察"), None, "当前干预状态"),
    ]
    cols = st.columns(len(cards))
    for idx, (label, value, percent, note) in enumerate(cards):
        with cols[idx]:
            if percent is True and isinstance(value, (int, float)):
                v = metric_value_text(float(value), as_percent=True)
            elif percent is False and isinstance(value, (int, float)):
                v = metric_value_text(float(value))
            else:
                if label == "风险预警":
                    v = display_risk_alert(str(value))
                else:
                    v = translate_display_text(str(value))
            st.markdown(
                f"""
                <div class='kpi-card'>
                  <div class='kpi-title'>{label}</div>
                  <div class='kpi-value'>{v}</div>
                  <div class='kpi-note'>{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_market_overview(metrics: pd.DataFrame, upto_step: Optional[int], key_prefix: str = "market", title: str = "市场主图与风险追踪") -> go.Figure:
    frame = metrics.copy()
    if upto_step is not None:
        frame = frame[frame["step"] <= int(upto_step)]
    
    if "open" not in frame.columns:
        frame["open"] = frame["close"].shift(1).fillna(frame["close"].iloc[0])
        noise = frame["close"].std() * 0.15 if len(frame) > 1 else 1.0
        frame["high"] = frame[["open", "close"]].max(axis=1) + np.abs(np.random.randn(len(frame))) * noise
        frame["low"] = frame[["open", "close"]].min(axis=1) - np.abs(np.random.randn(len(frame))) * noise

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=frame["time"],
        open=frame["open"],
        high=frame["high"],
        low=frame["low"],
        close=frame["close"],
        name="指数 K线",
        increasing_line_color="#f5222d",
        decreasing_line_color="#22c55e",
    ))
    fig.add_trace(go.Scatter(x=frame["time"], y=frame["panic_level"], mode="lines", name="风险热度", yaxis="y2", line=dict(color="#ff7a59", width=2, dash="dash")))
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title=dict(text=title, font=dict(color="#e2e8f0", size=18)),
        yaxis=dict(title=dict(text="指数点位", font=dict(color="#8aa0c2")), tickfont=dict(color="#e2e8f0")),
        yaxis2=dict(title=dict(text="风险热度", font=dict(color="#ff7a59")), overlaying="y", side="right", tickfont=dict(color="#ff7a59")),
        xaxis=dict(
            title=dict(text="", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0"),
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="全部")
                ]),
                bgcolor="#0a1931",
                font=dict(color="#e2e8f0")
            )
        ),
        legend=dict(orientation="h", font=dict(color="#e2e8f0")),
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_market_plot")
    export_plot_bundle(fig, frame, f"{key_prefix}_market_overview", f"{key_prefix}_market_overview")
    return fig


def render_empty_market_board(key_prefix: str = "empty") -> go.Figure:
    np.random.seed(42)  # 固定种子以保持每次渲染时的逼真“历史图”一致
    dates = pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=30, freq='D')
    close = 3200 + np.cumsum(np.random.randn(30) * 15)
    open_p = pd.Series(close).shift(1).fillna(close[0] - 5).values
    high = np.maximum(open_p, close) + np.abs(np.random.randn(30) * 10)
    low = np.minimum(open_p, close) - np.abs(np.random.randn(30) * 10)
    panic = np.clip(0.3 + np.cumsum(np.random.randn(30) * 0.05), 0.1, 0.9)
    
    frame = pd.DataFrame({
        "time": dates,
        "step": range(-30, 0),
        "open": open_p, "high": high, "low": low, "close": close,
        "panic_level": panic
    })
    return render_market_overview(frame, upto_step=0, key_prefix=key_prefix, title="市场主图与风险追踪 (历史前30日)")


def render_ab_world_compare(world_a: pd.DataFrame, world_b: pd.DataFrame, key_prefix: str = "ab") -> go.Figure:
    merged = world_a[["step", "time", "close"]].rename(columns={"close": "A世界"}).merge(
        world_b[["step", "close"]].rename(columns={"close": "B世界"}),
        on="step",
        how="left",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["time"], y=merged["A世界"], mode="lines", name="A世界（政策执行）", line=dict(color="#21c55d", width=2)))
    fig.add_trace(go.Scatter(x=merged["time"], y=merged["B世界"], mode="lines", name="B世界（政策缺失）", line=dict(color="#f97316", width=2)))
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="A/B 世界对照：政策有无的市场路径",
        yaxis=dict(title="指数点位"),
        legend=dict(orientation="h"),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_ab_plot")
    export_plot_bundle(fig, merged, f"{key_prefix}_ab_compare", f"{key_prefix}_ab_compare")
    return fig


def _build_lob_depth_steps(metrics: pd.DataFrame, max_steps: int = 12) -> Tuple[List[go.Frame], List[str]]:
    steps = []
    labels = []
    subset = metrics.head(max_steps)
    for _, row in subset.iterrows():
        mid = float(row["close"])
        spread = max(2.0, float(row["panic_level"]) * 25.0)
        prices_bid = np.linspace(mid - spread * 1.6, mid - spread * 0.1, 14)
        prices_ask = np.linspace(mid + spread * 0.1, mid + spread * 1.6, 14)
        bid_depth = np.linspace(240, 60, 14) * (1.0 + (0.3 - float(row["panic_level"])))
        ask_depth = np.linspace(70, 260, 14) * (1.0 + float(row["panic_level"]))
        frame = go.Frame(
            name=f"step_{int(row['step'])}",
            data=[
                go.Bar(x=prices_bid, y=bid_depth, name="买盘深度", marker_color="#16a34a"),
                go.Bar(x=prices_ask, y=ask_depth, name="卖盘深度", marker_color="#ef4444"),
            ],
        )
        steps.append(frame)
        labels.append(str(int(row["step"])))
    return steps, labels


def render_lob_depth_animation(metrics: pd.DataFrame, key_prefix: str = "lob") -> go.Figure:
    frames, labels = _build_lob_depth_steps(metrics)
    if not frames:
        fig = go.Figure()
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_lob_plot_empty")
        return fig

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="LOB 深度动画（买卖盘结构）",
        barmode="overlay",
        xaxis_title="价格",
        yaxis_title="挂单量",
        height=360,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "播放",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "暂停",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "method": "animate",
                        "label": label,
                        "args": [[f"step_{label}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for label in labels
                ],
                "currentvalue": {"prefix": "步骤 "},
            }
        ],
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_lob_plot")
    export_plot_bundle(fig, metrics.head(len(frames)), f"{key_prefix}_lob_depth", f"{key_prefix}_lob_depth")
    return fig


def render_social_network_heatmap(metrics: pd.DataFrame, key_prefix: str = "social") -> go.Figure:
    n = min(12, len(metrics))
    seed = np.clip(metrics["panic_level"].head(n).to_numpy(), 0.01, 0.99)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            base = 1.0 - abs(i - j) / max(1, n - 1)
            matrix[i, j] = base * (0.3 + 0.7 * seed[(i + j) % n])
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                colorscale="YlOrRd",
                zmin=0,
                zmax=1,
                colorbar=dict(title="传播强度"),
            )
        ]
    )
    fig.update_layout(**PLOTLY_DARK_LAYOUT, title="社会传播网络热图", xaxis_title="节点", yaxis_title="节点", height=360)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_social_plot")
    heatmap_df = pd.DataFrame(matrix)
    export_plot_bundle(fig, heatmap_df, f"{key_prefix}_social_heatmap", f"{key_prefix}_social_heatmap")
    return fig


def _policy_chain_strength(chain_payload: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not chain_payload:
        return {
            "policy_to_macro": 90.0,
            "macro_to_social": 75.0,
            "social_to_industry": 62.0,
            "industry_to_micro": 58.0,
        }

    macro = chain_payload.get("macro_variables", {})
    social = chain_payload.get("social_sentiment", {})
    industry = chain_payload.get("industry_agent", {})
    micro = chain_payload.get("market_microstructure", {})

    macro_shift = 0.0
    for key in ("inflation", "unemployment", "wage_growth", "credit_spread", "liquidity_index", "policy_rate", "fiscal_stimulus", "sentiment_index"):
        if key in macro:
            macro_shift += abs(float(macro[key]))

    social_mean = abs(float(social.get("mean", 0.0)))
    industry_shift = abs(float(industry.get("avg_household_risk", 0.0))) + abs(float(industry.get("avg_firm_hiring", 0.0)))
    buy = float(micro.get("buy_volume", 0.0))
    sell = float(micro.get("sell_volume", 0.0))
    imbalance = abs(buy - sell) / max(1.0, buy + sell)

    return {
        "policy_to_macro": float(np.clip(40 + macro_shift * 10, 10, 100)),
        "macro_to_social": float(np.clip(35 + social_mean * 70, 10, 100)),
        "social_to_industry": float(np.clip(30 + industry_shift * 60, 10, 100)),
        "industry_to_micro": float(np.clip(25 + imbalance * 90, 10, 100)),
    }


def render_policy_transmission_chain(chain_payload: Optional[Dict[str, Any]] = None, key_prefix: str = "policy_chain") -> go.Figure:
    policy_text = "政策输入"
    if chain_payload and chain_payload.get("policy"):
        policy_text = translate_display_text(str(chain_payload.get("policy")))
    labels = [policy_text[:24], "宏观变量", "社会情绪", "行业主体", "市场微结构"]
    strength = _policy_chain_strength(chain_payload)
    source = [0, 1, 2, 3]
    target = [1, 2, 3, 4]
    value = [
        strength["policy_to_macro"],
        strength["macro_to_social"],
        strength["social_to_industry"],
        strength["industry_to_micro"],
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, color=["#334155", "#2563eb", "#16a34a", "#f97316", "#ef4444"]),
                link=dict(source=source, target=target, value=value, color="rgba(148,163,184,0.35)"),
            )
        ]
    )
    fig.update_layout(**PLOTLY_DARK_LAYOUT, title="政策传导链：政策 -> 宏观变量 -> 社会情绪 -> 行业主体 -> 市场微结构", height=360)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chain_plot")
    sankey_df = pd.DataFrame({"source": source, "target": target, "value": value})
    export_plot_bundle(fig, sankey_df, f"{key_prefix}_policy_chain", f"{key_prefix}_policy_chain")
    return fig


def render_policy_sankey(key_prefix: str = "policy") -> go.Figure:
    """Backward-compatible wrapper."""
    return render_policy_transmission_chain(chain_payload=None, key_prefix=key_prefix)


def render_risk_event_timeline(metrics: pd.DataFrame, key_prefix: str = "risk") -> go.Figure:
    frame = metrics.copy()
    levels = pd.cut(frame["panic_level"], bins=[-1, 0.2, 0.35, 0.5, 2], labels=["低", "中", "高", "极高"])  # type: ignore[arg-type]
    events = []
    for _, row in frame.iterrows():
        severity = levels.iloc[int(row.name)]
        if severity in ("高", "极高"):
            events.append({
                "time": row["time"],
                "event": "风险预警" if severity == "高" else "触发监管评估",
                "level": str(severity),
                "value": float(row["panic_level"]),
            })
    if not events:
        events.append({"time": frame.iloc[-1]["time"], "event": "无高风险事件", "level": "低", "value": float(frame.iloc[-1]["panic_level"])})
    events_df = pd.DataFrame(events)

    y_map = {"低": 0, "中": 1, "高": 2, "极高": 3}
    fig = go.Figure(
        data=[
            go.Scatter(
                x=events_df["time"],
                y=events_df["level"].map(y_map),
                mode="markers+lines+text",
                text=events_df["event"],
                textposition="top center",
                marker=dict(size=12, color=events_df["value"], colorscale="Reds", cmin=0, cmax=1),
                name="风险事件",
            )
        ]
    )
    fig.update_layout(
        **PLOTLY_DARK_LAYOUT,
        title="风险事件时间轴",
        yaxis=dict(title="风险等级", tickvals=list(y_map.values()), ticktext=list(y_map.keys())),
        xaxis=dict(title="时间"),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_timeline_plot")
    export_plot_bundle(fig, events_df, f"{key_prefix}_timeline", f"{key_prefix}_timeline")
    return fig


def render_decision_evidence_flow(
    narration_items: Iterable[Dict[str, Any]],
    analyst_manager_output: Dict[str, Any],
) -> None:
    st.subheader("证据流")
    st.caption("展示结构化证据卡、矛盾矩阵、经理最终卡、风控告警与校准指标。")

    analyst_cards = analyst_manager_output.get("analyst_cards")
    if not isinstance(analyst_cards, list) or not analyst_cards:
        legacy_analyst = analyst_manager_output.get("analyst_outputs", {})
        analyst_cards = []
        if isinstance(legacy_analyst, dict):
            for name, payload in legacy_analyst.items():
                analyst_cards.append(
                    {
                        "analyst_id": str(name),
                        "thesis": f"{translate_display_text(str(name))} 历史输出",
                        "evidence": [{"type": "risk", "content": json.dumps(payload, ensure_ascii=False), "weight": 0.5}],
                        "time_horizon": "swing",
                        "risk_tags": [],
                        "confidence": 0.5,
                        "counterarguments": [],
                        "recommended_action": "hold",
                    }
                )

    manager_final_card = analyst_manager_output.get("manager_final_card")
    if not isinstance(manager_final_card, dict) or not manager_final_card:
        manager_final_card = analyst_manager_output.get("manager_decision", {}) or {}

    contradiction_matrix = analyst_manager_output.get("contradiction_matrix")
    if not isinstance(contradiction_matrix, dict) or not contradiction_matrix:
        contradiction_matrix = manager_final_card.get("contradiction_matrix", {}) if isinstance(manager_final_card, dict) else {}

    risk_alerts = analyst_manager_output.get("risk_alerts") or analyst_manager_output.get("risk_alert") or {}
    calibration = analyst_manager_output.get("calibration")
    if not isinstance(calibration, dict) or not calibration:
        calibration = manager_final_card.get("calibration", {}) if isinstance(manager_final_card, dict) else {}

    tab_cards, tab_matrix, tab_manager, tab_risk, tab_calib, tab_narration = st.tabs(
        ["分析师卡片", "矛盾矩阵", "经理最终卡", "风险告警", "校准指标", "叙事时间线"]
    )

    with tab_cards:
        if analyst_cards:
            for idx, card in enumerate(analyst_cards):
                label = f"{idx + 1}. {translate_display_text(str(card.get('analyst_id', f'analyst_{idx}')))}"
                with st.expander(label, expanded=(idx == 0)):
                    st.markdown(
                        narrate_payload(
                            f"{label}观点解读",
                            translate_ui_payload(card),
                            context="提炼分析师观点、证据与风险提示。",
                        )
                    )
        else:
            st.info("暂无分析师卡片。")

    with tab_matrix:
        analysts = contradiction_matrix.get("analysts", []) if isinstance(contradiction_matrix, dict) else []
        matrix = contradiction_matrix.get("matrix", []) if isinstance(contradiction_matrix, dict) else []
        if matrix and analysts:
            analyst_labels = [translate_display_text(str(item)) for item in analysts]
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=matrix,
                        x=analyst_labels,
                        y=analyst_labels,
                        zmin=0,
                        zmax=1,
                        colorscale="YlOrRd",
                        colorbar=dict(title="矛盾强度"),
                    )
                ]
            )
            fig.update_layout(**PLOTLY_DARK_LAYOUT, title="分析师矛盾矩阵", height=360)
            st.plotly_chart(fig, use_container_width=True, key="evidence_contradiction_matrix_plot")
            matrix_df = pd.DataFrame(matrix, columns=analyst_labels, index=analyst_labels)
            export_plot_bundle(fig, matrix_df.reset_index(), "evidence_contradiction_matrix", "evidence_contradiction_matrix")
            st.caption(f"矛盾指数：{float(contradiction_matrix.get('contradiction_index', 0.0)):.3f}")
        else:
            st.info("暂无矛盾矩阵数据。")

    with tab_manager:
        st.markdown(
            narrate_payload(
                "经理最终决策解读",
                translate_ui_payload(manager_final_card),
                context="解释仓位取向、执行节奏与核心判断。",
            )
        )

    with tab_risk:
        st.markdown(
            narrate_payload(
                "风险预警解读",
                translate_ui_payload(risk_alerts),
                context="说明风险等级、触发原因与建议动作。",
            )
        )

    with tab_calib:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Brier 类分数", f"{float(calibration.get('brier_like_score', 0.0)):.4f}")
        with c2:
            st.metric("置信漂移", f"{float(calibration.get('confidence_drift', 0.0)):.4f}")
        with c3:
            st.metric("结果代理值", f"{float(calibration.get('outcome_proxy', 0.0)):.2f}")
        if isinstance(manager_final_card, dict):
            st.caption(
                f"原始置信度={float(manager_final_card.get('raw_confidence', 0.0)):.3f}, "
                f"校准后置信度={float(manager_final_card.get('calibrated_confidence', 0.0)):.3f}"
            )

    with tab_narration:
        rows = []
        for item in narration_items:
            rows.append(
                {
                    "步骤": item.get("step"),
                    "角色": translate_display_text(str(item.get("speaker", "旁白"))),
                    "事件": translate_display_text(str(item.get("text", ""))),
                }
            )
        evidence_df = pd.DataFrame(rows)
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)


def build_kpi_snapshot(metrics: pd.DataFrame, step: int, regulation_hint: str = "观察") -> Dict[str, Any]:
    if metrics.empty:
        return {"liquidity": 0.0, "volatility": 0.0, "herding": 0.0, "risk_alert": "GREEN", "reg_action": regulation_hint}

    current = metrics[metrics["step"] <= step].tail(1)
    if current.empty:
        current = metrics.head(1)
    current_row = current.iloc[0]

    recent = metrics[metrics["step"] <= step].tail(5)
    if len(recent) >= 2:
        returns = recent["close"].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else 0.0
    else:
        volatility = 0.0

    liquidity = float(current_row.get("volume", 0.0)) / max(1.0, float(metrics["volume"].median()))
    herding = float(current_row.get("csad", 0.0))
    panic = float(current_row.get("panic_level", 0.0))

    if panic > 0.7:
        risk_alert = "RED"
    elif panic > 0.45:
        risk_alert = "YELLOW"
    else:
        risk_alert = "GREEN"

    return {
        "liquidity": liquidity,
        "volatility": volatility,
        "herding": herding,
        "risk_alert": risk_alert,
        "reg_action": regulation_hint,
    }

def render_orderflow_microstructure_panel(
    role_order_flows: Optional[Mapping[str, float]] = None,
    microstructure_metrics: Optional[Mapping[str, Any]] = None,
    *,
    key_prefix: str = "micro_panel",
) -> None:
    """Render role order-flow decomposition and microstructure summary."""
    st.subheader("角色订单流与微结构")
    st.caption("展示各类主体净买卖拆解，以及 spread/depth/impact/herding 等核心指标。")

    role_order_flows = dict(role_order_flows or {})
    if role_order_flows:
        orderflow_df = pd.DataFrame(
            [{"role": str(role), "net_flow": float(value)} for role, value in role_order_flows.items()]
        ).sort_values("net_flow", ascending=False)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=orderflow_df["role"],
                    y=orderflow_df["net_flow"],
                    marker_color=["#16a34a" if value >= 0 else "#ef4444" for value in orderflow_df["net_flow"]],
                    name="净买卖额",
                )
            ]
        )
        fig.update_layout(
            **PLOTLY_DARK_LAYOUT,
            title="角色净买卖拆解",
            xaxis_title="角色",
            yaxis_title="净买卖额",
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_role_flow_plot")
        export_plot_bundle(fig, orderflow_df, f"{key_prefix}_role_orderflow", f"{key_prefix}_role_orderflow")
    else:
        st.info("当前场景暂无角色订单流明细。")

    metrics = dict(microstructure_metrics or {})
    spread = float(metrics.get("spread", 0.0) or 0.0)
    spread_pct = float(metrics.get("spread_pct", 0.0) or 0.0)
    depth_imbalance = float(metrics.get("depth_imbalance", 0.0) or 0.0)
    impact = float(metrics.get("impact", metrics.get("impact_bps", 0.0)) or 0.0)
    herding_proxy = float(metrics.get("herding_proxy", 0.0) or 0.0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("买卖价差", f"{spread:.4f}")
    c2.metric("价差占比", f"{spread_pct:.2%}")
    c3.metric("深度失衡", f"{depth_imbalance:+.3f}")
    c4.metric("价格冲击", f"{impact:.4f}")
    c5.metric("羊群程度", f"{herding_proxy:.3f}")

def render_financial_health_dashboard(metrics_dict: Dict[str, Any], key_prefix: str = "health") -> None:
    st.markdown("### 📊 实时金融异常指标监测", help="通过速度表盘直观反应市场在恐慌、羊群效应、流动性枯竭方面的风险情况。")
    
    panic = float(metrics_dict.get("panic_level", 0.0))
    csad = float(metrics_dict.get("csad", 0.0))
    imb = float(metrics_dict.get("depth_imbalance", 0.0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=panic,
        title={'text': "恐慌指数", 'font': {'size': 14, 'color': '#8aa0c2'}},
        gauge={'axis': {'range': [0, 1]},
               'bar': {'color': "rgba(250, 140, 22, 0.8)", 'line': {'width': 0}},
               'steps': [
                   {'range': [0, 0.35], 'color': "rgba(34, 197, 94, 0.2)"},
                   {'range': [0.35, 0.7], 'color': "rgba(250, 140, 22, 0.2)"},
                   {'range': [0.7, 1.0], 'color': "rgba(245, 34, 45, 0.3)"}],
               'threshold': {'line': {'color': "#f5222d", 'width': 3}, 'thickness': 0.75, 'value': 0.8}},
        domain={'row': 0, 'column': 0}
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=csad,
        number={'valueformat': '.3f'},
        title={'text': "羊群效应（CSAD）", 'font': {'size': 14, 'color': '#8aa0c2'}},
        gauge={'axis': {'range': [0, 0.15]},
               'bar': {'color': "rgba(138, 43, 226, 0.8)", 'line': {'width': 0}},
               'steps': [
                   {'range': [0, 0.05], 'color': "rgba(34, 197, 94, 0.2)"},
                   {'range': [0.05, 0.1], 'color': "rgba(250, 140, 22, 0.2)"},
                   {'range': [0.1, 0.15], 'color': "rgba(245, 34, 45, 0.3)"}],
               },
        domain={'row': 0, 'column': 1}
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=imb,
        number={'valueformat': '+.2f'},
        title={'text': "深度失衡", 'font': {'size': 14, 'color': '#8aa0c2'}},
        gauge={'axis': {'range': [-1, 1]},
               'bar': {'color': "#1890ff", 'line': {'width': 0}},
               'steps': [
                   {'range': [-1, -0.4], 'color': "rgba(245, 34, 45, 0.2)"},
                   {'range': [-0.4, 0.4], 'color': "rgba(34, 197, 94, 0.2)"},
                   {'range': [0.4, 1.0], 'color': "rgba(24, 144, 255, 0.2)"}],
               },
        domain={'row': 0, 'column': 2}
    ))
    
    fig.update_layout(
        **{
            **dict(PLOTLY_DARK_LAYOUT),
            "grid": {'rows': 1, 'columns': 3, 'pattern': "independent"},
            "height": 220,
            "margin": dict(l=10, r=10, t=30, b=10),
        }
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_health_gauges")

def render_ai_insight_card(text: str) -> None:
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(15, 30, 55, 0.6), rgba(10, 20, 40, 0.9)); 
                    border-left: 5px solid #1890ff; 
                    border-radius: 8px; 
                    padding: 16px 20px; 
                    margin: 12px 0 24px 0;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                    backdrop-filter: blur(8px);
                    animation: insight-pulse 3.5s infinite;">
            <div style="color: #4da6ff; font-weight: 700; font-size: 14px; margin-bottom: 8px; letter-spacing: 1.5px; display: flex; align-items: center; justify-content: space-between;">
                <span>🧠 AI 专家前瞻与状态点评</span>
                <span style="font-size: 11px; font-weight: normal; background: rgba(24, 144, 255, 0.2); padding: 2px 8px; border-radius: 4px;">由 ModelRouter 实时驱动</span>
            </div>
            <div style="color: #e2e8f0; font-size: 15px; line-height: 1.6;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
