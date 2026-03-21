# file: app.py
"""Civitas front-end in Streamlit: Policy Lab / History Replay / Advanced Analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, MutableMapping, cast

import pandas as pd
import streamlit as st

from core.competition_demo import (
    LIVE_MODE,
    DEMO_MODE,
    COMPETITION_DEMO_MODE,
    REQUIRED_SCENARIOS,
    list_competition_scenarios,
    load_competition_scenario,
    bootstrap_competition_demo,
)
from core.ui_text import display_runtime_mode, display_scenario_name
from ui.backtest_panel import render_backtest_panel
from ui.behavioral_diagnostics import render_behavioral_diagnostics
from ui.demo_wind_tunnel import render_demo_tab
from ui.history_replay import render_history_replay
from ui.policy_lab import render_policy_lab
from ui import dashboard as dashboard_ui


st.set_page_config(
    page_title="Civitas Policy Sandbox | 政策试验台",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


ENTRY_POINTS = ["政策试验台", "历史政策回放", "高级分析", "系统说明"]
ENTRY_DESCRIPTIONS = {
    "政策试验台": "输入新政策，观察市场、风险与情绪如何联动变化。",
    "历史政策回放": "选取真实历史区间，把仿真走势与真实市场放在一起比较。",
    "高级分析": "查看 AI 证据流、行为金融诊断、答辩演示与研究参数。",
    "系统说明": "快速了解架构、数据流与工程实现边界。",
}
ENTRY_PURPOSE = {
    "政策试验台": "输入政策并运行多智能体仿真",
    "历史政策回放": "验证政策机制是否像真实市场",
    "高级分析": "专家追问与研究调参",
    "系统说明": "了解系统如何工作",
}
THEME_PATH = Path("theme") / "competition_dark.css"
MATERIALS_ROOT = Path("outputs") / "competition_materials"


def _load_theme() -> None:
    if THEME_PATH.exists():
        css = THEME_PATH.read_text(encoding="utf-8")
    else:
        css = """
        .stApp { background: #070b14; color: #d8e2f2; }
        .kpi-card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 12px; }
        .kpi-title { font-size: 12px; color: #94a3b8; }
        .kpi-value { font-size: 24px; font-weight: 700; color: #e2e8f0; }
        .kpi-note { font-size: 12px; color: #64748b; }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "entry": "政策试验台",
        "controller": None,
        "runtime_mode": LIVE_MODE,
        "competition_mode": "",
        "market_history": [],
        "csad_history": [],
        "demo_scenario_name": REQUIRED_SCENARIOS[0],
        "demo_scenario": None,
        "demo_step_cursor": 0,
        "demo_narration_cursor": 0,
        "demo_last_narration": None,
        "demo_last_step": 0,
        "demo_last_panic_level": 0.0,
        "demo_autoplay": False,
        "is_running": False,
        "last_demo_figures": {},
        "materials_last_export": None,
        "policy_lab_result": None,
        "history_replay_result": None,
        "simulation_mode": "SMART",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_top_entry_selector() -> None:
    st.markdown("## Civitas Policy Sandbox")
    st.caption("首页入口：政策试验台 / 历史政策回放 / 高级分析 / 系统说明")

    cols = st.columns(len(ENTRY_POINTS))
    for idx, entry in enumerate(ENTRY_POINTS):
        with cols[idx]:
            if st.button(
                entry,
                key=f"entry_{entry}",
                width="stretch",
                type="primary" if st.session_state.entry == entry else "secondary",
            ):
                st.session_state.entry = entry
            st.caption(ENTRY_DESCRIPTIONS[entry])

    st.markdown(
        f"""
        <div class='mode-pill'>当前页面：{st.session_state.entry} | 用途：{ENTRY_PURPOSE.get(st.session_state.entry, "")}</div>
        """,
        unsafe_allow_html=True,
    )


def _ensure_demo_loaded() -> None:
    if st.session_state.get("demo_scenario") is None:
        name = st.session_state.get("demo_scenario_name", REQUIRED_SCENARIOS[0])
        try:
            state = cast(MutableMapping[str, Any], st.session_state)
            bootstrap_competition_demo(state, name, auto_play=False)
            st.session_state.runtime_mode = DEMO_MODE
            st.session_state.competition_mode = COMPETITION_DEMO_MODE
        except Exception as exc:
            st.session_state.demo_scenario = None
            st.error(f"场景加载失败：{exc}")


def _material_summary_from_metrics(metrics: pd.DataFrame) -> Dict[str, float]:
    if metrics.empty:
        return {"return_pct": 0.0, "volatility": 0.0, "panic_max": 0.0, "herding_avg": 0.0}
    ret = (float(metrics.iloc[-1]["close"]) - float(metrics.iloc[0]["close"])) / max(1.0, float(metrics.iloc[0]["close"]))
    vol = float(metrics["close"].pct_change().std()) if len(metrics) > 1 else 0.0
    panic_max = float(metrics["panic_level"].max())
    herding_avg = float(metrics["csad"].mean())
    return {
        "return_pct": ret,
        "volatility": vol,
        "panic_max": panic_max,
        "herding_avg": herding_avg,
    }


def _generate_competition_materials() -> Dict[str, Path]:
    scenario = st.session_state.get("demo_scenario")
    if scenario is None:
        scenario_name = st.session_state.get("demo_scenario_name", REQUIRED_SCENARIOS[0])
        scenario = load_competition_scenario(scenario_name)

    MATERIALS_ROOT.mkdir(parents=True, exist_ok=True)
    figures_dir = MATERIALS_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = scenario.metrics
    stat = _material_summary_from_metrics(metrics)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    comparison_md = (
        "## UI 重构前后对比\n"
        "- 前：入口较分散，演示链路跨多个标签页，评委视角不连续。\n"
        "- 后：首页收敛为五入口（含行为金融诊断页），答辩链路一键加载+自动时间线+三段叙事。\n"
        "- 前：思维链以原始文本为主。\n"
        "- 后：改为“决策证据流”，仅展示结构化推理产物。\n"
        "- 前：图表样式不统一。\n"
        "- 后：统一深色主题、中文标题、图表均支持 PNG/CSV/JSON 导出。\n"
    )

    competition_summary = f"""# competition_summary

生成时间：{now}
场景：{display_scenario_name(scenario.name)}

## 核心结论
- 累计收益：{stat['return_pct']:.2%}
- 波动率：{stat['volatility']:.2%}
- 最大风险热度：{stat['panic_max']:.2f}
- 平均羊群度（CSAD）：{stat['herding_avg']:.3f}

## 叙事主线
1. 分析师汇总多源证据并形成结构化判断。
2. 经理将证据映射为可执行仓位与风险约束。
3. 市场在流动性、波动率、羊群效应上给出反馈。

{comparison_md}
"""

    design_outline = f"""# design_outline

## 信息架构
- 首页五入口：答辩模式 / 专家模式 / 历史回测 / 行为金融诊断 / 系统说明
- 答辩模式：场景加载、自动时间线、KPI 大卡、三段叙事、A/B 世界对照
- 专家模式：决策证据流 + 多图联动

## 关键前端组件
- LOB 深度动画
- 社会传播网络热图
- 政策传导桑基图
- 风险事件时间轴

## 约束达成
- DEMO_MODE 无需 API Key 亦可完整演示
- 5 分钟内可跑完完整答辩链路（默认 12 步）
- 图表统一深色风格、中文标题，支持 PNG/CSV/JSON 导出

{comparison_md}
"""

    demo_script_10min = f"""# demo_script_10min

## 0:00 - 1:00 开场
- 打开首页，说明五入口结构和当前演示目标。

## 1:00 - 4:00 答辩模式
- 一键加载场景 `{display_scenario_name(scenario.name)}`。
- 启动自动播放，展示 KPI 变化与三段式叙事。
- 强调 A/B 世界对照中的政策差异路径。

## 4:00 - 7:30 专家模式
- 进入“决策证据流”，展示结构化证据而非原始 CoT。
- 展示 LOB 深度动画、传播热图、桑基图、风险时间轴。

## 7:30 - 9:00 历史回测
- 切入回测页，说明与仿真页的数据闭环。

## 9:00 - 10:00 总结
- 回答评委：为何该系统在无 API Key 条件下仍可稳定完整展示。
"""

    figures_index = {
        "generated_at": now,
        "scenario": scenario.name,
        "scenario_display": display_scenario_name(scenario.name),
        "figures": st.session_state.get("last_demo_figures", {}),
    }

    file_map = {
        "competition_summary.md": MATERIALS_ROOT / "competition_summary.md",
        "design_outline.md": MATERIALS_ROOT / "design_outline.md",
        "demo_script_10min.md": MATERIALS_ROOT / "demo_script_10min.md",
        "figures/index.json": figures_dir / "index.json",
    }

    file_map["competition_summary.md"].write_text(competition_summary, encoding="utf-8")
    file_map["design_outline.md"].write_text(design_outline, encoding="utf-8")
    file_map["demo_script_10min.md"].write_text(demo_script_10min, encoding="utf-8")
    file_map["figures/index.json"].write_text(json.dumps(figures_index, ensure_ascii=False, indent=2), encoding="utf-8")

    return file_map


def _render_sidebar_global() -> None:
    with st.sidebar:
        st.markdown("### 快速说明")
        st.info(f"当前页面：{st.session_state.entry}")
        
        st.markdown("---")
        st.markdown("### 仿真模式设置")
        sim_mode_display = {"SMART": "智能模式 (GLM-4 + Chat)", "DEEP": "深度模式 (Reasoner + Chat)"}
        selected_mode_key = st.radio(
            "选择 LLM 调度策略",
            options=["SMART", "DEEP"],
            index=0 if st.session_state.simulation_mode == "SMART" else 1,
            format_func=lambda x: sim_mode_display.get(x, x),
            help="智能模式优先使用 GLM；深度模式优先使用 DeepSeek Reasoner 推理模型。"
        )
        if selected_mode_key != st.session_state.simulation_mode:
            st.session_state.simulation_mode = selected_mode_key
            st.toast(f"已切换至 {sim_mode_display[selected_mode_key]}")

        st.markdown("---")
        st.caption("默认先做政策实验或历史回放。需要专家级细节时，再进入“高级分析”。")

        if st.session_state.entry == "高级分析":
            scenarios = list_competition_scenarios()
            if scenarios:
                if st.session_state.demo_scenario_name not in scenarios:
                    st.session_state.demo_scenario_name = scenarios[0]
                st.session_state.demo_scenario_name = st.selectbox(
                    "高级分析默认场景",
                    options=scenarios,
                    index=scenarios.index(st.session_state.demo_scenario_name),
                    format_func=display_scenario_name,
                )
            else:
                st.warning("未发现可用答辩场景，请检查 demo_scenarios 内容完整性。")

            if st.button("生成答辩材料", width="stretch"):
                _ensure_demo_loaded()
                if st.session_state.get("demo_scenario") is None:
                    st.error("当前没有可用场景，无法生成比赛材料。")
                else:
                    try:
                        file_map = _generate_competition_materials()
                        st.session_state.materials_last_export = {k: str(v) for k, v in file_map.items()}
                        st.success("比赛材料已生成。")
                    except Exception as exc:
                        st.error(f"比赛材料生成失败：{exc}")

            if st.session_state.materials_last_export:
                st.caption("最近生成文件")
                st.json(st.session_state.materials_last_export)


def _render_ai_decision_tab() -> None:
    st.markdown("### AI 决策解读")
    st.caption("这里保留专家级视角，用于追问时查看结构化证据，而非普通用户默认入口。")
    _ensure_demo_loaded()
    scenario = st.session_state.demo_scenario
    metrics = scenario.metrics
    step = int(st.session_state.get("demo_last_step", 0))
    if step <= 0:
        step = int(metrics.iloc[min(3, len(metrics) - 1)]["step"])

    kpi = dashboard_ui.build_kpi_snapshot(metrics, step, regulation_hint="专家复盘")
    dashboard_ui.render_kpi_cards(kpi)
    dashboard_ui.render_decision_evidence_flow(scenario.narration, scenario.analyst_manager_output)

    upto = metrics[metrics["step"] <= step]
    latest = upto.tail(1).iloc[0] if not upto.empty else metrics.tail(1).iloc[0]
    chain_payload = {
        "policy": f"专家复盘 第 {int(latest['step'])} 步",
        "macro_variables": {
            "inflation": 0.02 + float(latest["panic_level"]) * 0.01,
            "unemployment": 0.05 + float(latest["panic_level"]) * 0.03,
            "wage_growth": 0.03 - float(latest["panic_level"]) * 0.01,
            "credit_spread": 0.015 + float(latest["panic_level"]) * 0.02,
            "liquidity_index": max(0.2, 1.15 - float(latest["panic_level"])),
            "policy_rate": 0.022,
            "fiscal_stimulus": 0.02,
            "sentiment_index": max(0.0, min(1.0, 0.65 - float(latest["panic_level"]) * 0.55)),
        },
        "social_sentiment": {"mean": 0.5 - float(latest["panic_level"])},
        "industry_agent": {
            "avg_household_risk": 0.50 - float(latest["panic_level"]) * 0.3,
            "avg_firm_hiring": 0.15 - float(latest["panic_level"]) * 0.5,
        },
        "market_microstructure": {
            "buy_volume": float(latest["volume"]) * (1.0 - float(latest["panic_level"])),
            "sell_volume": float(latest["volume"]) * (0.45 + float(latest["panic_level"])),
            "trade_count": int(max(1.0, float(latest["volume"]) / 800)),
            "matching_mode": "expert_replay",
        },
    }
    c1, c2 = st.columns(2)
    with c1:
        dashboard_ui.render_social_network_heatmap(upto, key_prefix="advanced_ai")
        dashboard_ui.render_lob_depth_animation(upto, key_prefix="advanced_ai")
    with c2:
        dashboard_ui.render_policy_transmission_chain(chain_payload, key_prefix="advanced_ai")
        dashboard_ui.render_risk_event_timeline(upto, key_prefix="advanced_ai")


def _render_advanced_analysis() -> None:
    st.session_state.runtime_mode = DEMO_MODE
    st.markdown("## 高级分析")
    st.caption("这里保留专家追问、答辩演示、行为金融诊断和研究参数面板。")
    tab1, tab2, tab3, tab4 = st.tabs(["AI 决策解读", "市场行为分析", "答辩演示", "研究参数"])

    with tab1:
        _render_ai_decision_tab()
    with tab2:
        render_behavioral_diagnostics()
    with tab3:
        st.session_state.competition_mode = COMPETITION_DEMO_MODE
        render_demo_tab(ctrl=st.session_state.get("controller"))
    with tab4:
        st.session_state.runtime_mode = LIVE_MODE
        render_backtest_panel(ctrl=st.session_state.get("controller"))


def _render_system_guide() -> None:
    st.markdown("## 系统说明")
    st.markdown(
        """
### 当前产品定位
- 主入口已经切换为“政策试验台”和“历史政策回放”。
- “高级分析”保留专家级证据流、行为金融诊断与答辩演示能力。
- 页面默认展示业务语言和结果图，而不是原始结构化对象。

### 架构保持不变
- 前端：Streamlit
- 调度：IPC + Python 控制层
- 撮合：C++ LOB 引擎
- 历史回放：统一回测引擎 + 真实市场数据提供器

### AI 与多智能体如何工作
- 政策文本先被编译成结构化冲击。
- 新闻分析、量化分析、风险分析共同形成结构化判断。
- 经理代理聚合意见、校准置信度并给出执行建议。
- 仿真结果以指数、风险热度、流动性和行为金融指标返回前端。
        """
    )


def main() -> None:
    _load_theme()
    _init_state()
    _render_sidebar_global()
    _render_top_entry_selector()

    entry = st.session_state.entry
    if entry == "政策试验台":
        st.session_state.runtime_mode = DEMO_MODE
        render_policy_lab()
    elif entry == "历史政策回放":
        st.session_state.runtime_mode = LIVE_MODE
        render_history_replay()
    elif entry == "高级分析":
        _render_advanced_analysis()
    else:
        _render_system_guide()


if __name__ == "__main__":
    main()
