"""Defense-mode wind tunnel page for competition demo."""

from __future__ import annotations

import time
from typing import Any, Dict, MutableMapping, Optional, cast

import pandas as pd
import streamlit as st

from core.competition_demo import (
    DEMO_MODE,
    COMPETITION_DEMO_MODE,
    list_competition_scenarios,
    bootstrap_competition_demo,
    advance_competition_demo,
    replay_next_narration,
)
from core.ui_text import display_scenario_name, translate_display_text, translate_ui_payload
from ui import dashboard as dashboard_ui
from ui.narrative import narrate_payload


AUTO_PLAY_INTERVAL_SECONDS = 0.7


def _counterfactual_world(metrics: pd.DataFrame) -> pd.DataFrame:
    """Generate a deterministic B-world for A/B compare."""
    b = metrics.copy()
    if b.empty:
        return b
    base = float(b.iloc[0]["close"])
    for idx, row in b.iterrows():
        shock = 1.0 - 0.002 * float(idx)
        panic_drag = 1.0 - 0.18 * float(row.get("panic_level", 0.0))
        b.at[idx, "close"] = max(base * 0.82, float(row["close"]) * shock * panic_drag)
    return b


def _current_step() -> int:
    return int(st.session_state.get("demo_last_step", 0))


def _current_narration_text() -> str:
    latest = st.session_state.get("demo_last_narration")
    if not latest:
        return "等待自动讲解..."
    speaker = translate_display_text(str(latest.get("speaker", "旁白")))
    text = translate_display_text(str(latest.get("text", "")))
    return f"{speaker}: {text}"


def _build_policy_chain_payload(scenario: Any, current_step: int) -> Dict[str, Any]:
    frame = scenario.metrics[scenario.metrics["step"] <= current_step].tail(1)
    row = frame.iloc[0] if not frame.empty else scenario.metrics.iloc[-1]
    panic = float(row.get("panic_level", 0.0))
    csad = float(row.get("csad", 0.0))
    volume = float(row.get("volume", 0.0))
    narration = st.session_state.get("demo_last_narration") or {}
    policy_text = str(narration.get("text", scenario.name))

    return {
        "policy": policy_text,
        "macro_variables": {
            "inflation": 0.02 + 0.01 * panic,
            "unemployment": 0.05 + 0.04 * panic,
            "wage_growth": 0.03 - 0.015 * panic,
            "credit_spread": 0.015 + 0.020 * panic,
            "liquidity_index": max(0.2, 1.2 - panic),
            "policy_rate": 0.022,
            "fiscal_stimulus": 0.03 if "注入" in policy_text or "刺激" in policy_text else 0.0,
            "sentiment_index": max(0.0, min(1.0, 0.65 - 0.6 * panic)),
        },
        "social_sentiment": {
            "mean": 0.5 - panic,
            "stressed_nodes": [],
            "avg_news_exposure": min(1.0, 0.35 + panic),
            "avg_social_exposure": min(1.0, 0.45 + 0.6 * panic),
        },
        "industry_agent": {
            "avg_household_risk": 0.55 - 0.4 * panic,
            "avg_firm_hiring": 0.2 - 0.6 * panic,
            "sector_outlook": {
                "金融": 0.2 - panic * 0.5,
                "科技": 0.3 - panic * 0.4,
                "消费": 0.25 - panic * 0.45,
            },
        },
        "market_microstructure": {
            "buy_volume": volume * (1.0 - panic),
            "sell_volume": volume * (0.4 + panic),
            "trade_count": int(max(1, volume / 1000)),
            "matching_mode": "demo_mode",
            "price": float(row.get("close", 0.0)),
            "csad": csad,
        },
    }


def _render_three_stage_story(scenario: Any, current_step: int) -> None:
    st.subheader("三段式叙事：分析师 -> 经理 -> 市场")
    analyst = scenario.analyst_manager_output.get("analyst_outputs", {})
    manager = scenario.analyst_manager_output.get("manager_decision", {})
    market_row = scenario.metrics[scenario.metrics["step"] <= current_step].tail(1)
    market_data = market_row.iloc[0].to_dict() if not market_row.empty else {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1) 分析师")
        st.markdown(
            narrate_payload(
                "分析师阶段解读",
                translate_ui_payload(analyst),
                context="概括分析师观察到的信号、证据和市场判断。",
            )
        )
    with c2:
        st.markdown("### 2) 经理")
        st.markdown(
            narrate_payload(
                "经理阶段解读",
                translate_ui_payload(manager),
                context="说明经理的动作选择、仓位调整和执行逻辑。",
            )
        )
    with c3:
        st.markdown("### 3) 市场")
        st.markdown(
            narrate_payload(
                "市场反馈阶段解读",
                translate_ui_payload(
                    {
                        "step": market_data.get("step", 0),
                        "close": market_data.get("close", 0.0),
                        "volume": market_data.get("volume", 0.0),
                        "panic_level": market_data.get("panic_level", 0.0),
                        "csad": market_data.get("csad", 0.0),
                    }
                ),
                context="解释当前市场价格、成交与风险情绪反馈。",
            )
        )


def _handle_autoplay(scenario: Any) -> None:
    if st.session_state.get("runtime_mode") != DEMO_MODE:
        return
    if not st.session_state.get("demo_autoplay") or not st.session_state.get("is_running"):
        return

    last = float(st.session_state.get("_demo_autoplay_time", 0.0))
    now = time.time()
    if now - last < AUTO_PLAY_INTERVAL_SECONDS:
        return

    state = cast(MutableMapping[str, Any], st.session_state)
    result = advance_competition_demo(state, steps=1)
    hits = result.get("narration", [])
    if hits:
        st.session_state.demo_last_narration = hits[-1]
    else:
        replay_next_narration(state)
    st.session_state._demo_autoplay_time = now

    if not result.get("done", False):
        st.rerun()


def render_demo_tab(ctrl: Optional[Any] = None) -> None:
    st.markdown("## 答辩模式")
    st.caption("一键加载场景、自动播放时间线，5 分钟内完成全链路演示。")

    scenarios = list_competition_scenarios()
    if not scenarios:
        st.warning("未找到 demo_scenarios 场景目录。")
        return

    selected = st.selectbox(
        "场景选择",
        scenarios,
        index=scenarios.index(st.session_state.get("demo_scenario_name", scenarios[0]))
        if st.session_state.get("demo_scenario_name", scenarios[0]) in scenarios
        else 0,
        key="defense_scene_selector",
        format_func=display_scenario_name,
    )
    st.session_state.demo_scenario_name = selected

    control_cols = st.columns(5)
    with control_cols[0]:
        load_clicked = st.button("一键载入场景", use_container_width=True, type="primary")
    with control_cols[1]:
        play_clicked = st.button("自动播放", use_container_width=True)
    with control_cols[2]:
        pause_clicked = st.button("暂停", use_container_width=True)
    with control_cols[3]:
        step_clicked = st.button("下一步", use_container_width=True)
    with control_cols[4]:
        reset_clicked = st.button("重置", use_container_width=True)

    if load_clicked:
        try:
            state = cast(MutableMapping[str, Any], st.session_state)
            bootstrap_competition_demo(state, selected, auto_play=True)
            st.session_state.runtime_mode = DEMO_MODE
            st.session_state.competition_mode = COMPETITION_DEMO_MODE
            st.session_state._demo_autoplay_time = 0.0
            st.success(f"场景“{display_scenario_name(selected)}”已加载。")
            st.rerun()
        except Exception as exc:
            st.error(f"场景加载失败：{exc}")

    if play_clicked and st.session_state.get("demo_scenario") is not None:
        st.session_state.demo_autoplay = True
        st.session_state.is_running = True
        st.rerun()

    if pause_clicked:
        st.session_state.demo_autoplay = False
        st.session_state.is_running = False

    if step_clicked and st.session_state.get("demo_scenario") is not None:
        state = cast(MutableMapping[str, Any], st.session_state)
        result = advance_competition_demo(state, steps=1)
        hits = result.get("narration", [])
        if hits:
            st.session_state.demo_last_narration = hits[-1]
        else:
            replay_next_narration(state)
        st.rerun()

    if reset_clicked and st.session_state.get("demo_scenario") is not None:
        try:
            state = cast(MutableMapping[str, Any], st.session_state)
            bootstrap_competition_demo(state, selected, auto_play=False)
            st.session_state.runtime_mode = DEMO_MODE
            st.session_state.competition_mode = COMPETITION_DEMO_MODE
            st.session_state.is_running = False
            st.session_state.demo_autoplay = False
            st.rerun()
        except Exception as exc:
            st.error(f"场景重置失败：{exc}")

    scenario = st.session_state.get("demo_scenario")
    if scenario is None:
        st.info("点击“一键载入场景”开始答辩演示。")
        return

    _handle_autoplay(scenario)

    current_step = _current_step()
    if current_step <= 0:
        state = cast(MutableMapping[str, Any], st.session_state)
        preview = advance_competition_demo(state, steps=1)
        if preview.get("narration"):
            st.session_state.demo_last_narration = preview["narration"][-1]
        current_step = _current_step()

    metrics = scenario.metrics
    upto = metrics[metrics["step"] <= current_step]
    if upto.empty:
        upto = metrics.head(1)

    regulation_hint = "干预中" if upto.iloc[-1]["panic_level"] > 0.45 else "观察"
    kpi = dashboard_ui.build_kpi_snapshot(metrics, current_step, regulation_hint=regulation_hint)
    dashboard_ui.render_kpi_cards(kpi)

    st.info(f"自动讲解：{_current_narration_text()}")
    st.progress(min(1.0, current_step / max(1, len(metrics))))

    _render_three_stage_story(scenario, current_step)

    st.markdown("---")
    st.subheader("A/B 世界对照")
    world_b = _counterfactual_world(metrics)
    fig_market = dashboard_ui.render_market_overview(metrics, current_step, key_prefix="defense")
    fig_ab = dashboard_ui.render_ab_world_compare(metrics, world_b, key_prefix="defense")

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_lob = dashboard_ui.render_lob_depth_animation(upto, key_prefix="defense")
        fig_heatmap = dashboard_ui.render_social_network_heatmap(upto, key_prefix="defense")
    with chart_col2:
        chain_payload = _build_policy_chain_payload(scenario, current_step)
        fig_sankey = dashboard_ui.render_policy_transmission_chain(chain_payload, key_prefix="defense")
        fig_timeline = dashboard_ui.render_risk_event_timeline(upto, key_prefix="defense")

    st.markdown("---")
    dashboard_ui.render_decision_evidence_flow(
        narration_items=scenario.narration,
        analyst_manager_output=scenario.analyst_manager_output,
    )

    st.session_state.last_demo_figures = {
        "market_overview": fig_market.to_dict(),
        "ab_compare": fig_ab.to_dict(),
        "lob_depth": fig_lob.to_dict(),
        "social_heatmap": fig_heatmap.to_dict(),
        "policy_sankey": fig_sankey.to_dict(),
        "risk_timeline": fig_timeline.to_dict(),
    }
