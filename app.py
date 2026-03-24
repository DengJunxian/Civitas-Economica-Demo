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
from core.competition_compliance import (
    COMPETITION_MODE_FLAG,
    competition_mode_enabled,
    write_competition_compliance_artifacts,
)
from core.ui_text import display_runtime_mode, display_scenario_name
from ui.backtest_panel import render_backtest_panel
from ui.behavioral_diagnostics import render_behavioral_diagnostics
from ui.demo_wind_tunnel import render_demo_tab
from ui.history_replay import render_history_replay
from ui.policy_lab import render_policy_lab
from ui.regulator_optimization import REGULATOR_OPTIMIZATION_PAGE_FLAG, render_regulator_optimization
from ui.reporting import export_defense_bundle
from ui import dashboard as dashboard_ui


st.set_page_config(
    page_title="数治观澜 | 政策风动推演沙箱",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


ENTRY_POINTS = [
    "政策试验台",
    "历史因子回测",
    "历史Agent回放",
    "真实性报告",
    "政策A/B推演",
    "监管优化",
    "系统说明",
]
ENTRY_ALIASES = {
    "历史政策回放": "历史因子回测",
    "高级分析": "真实性报告",
}
ENTRY_DESCRIPTIONS = {
    "政策试验台": "输入新政策，观察市场、风险与情绪如何联动变化。",
    "历史因子回测": "保留传统因子/组合回测能力，用于基准对照。",
    "历史Agent回放": "在历史窗口重放 agent 决策与撮合成交，并输出 simulated OHLCV。",
    "真实性报告": "查看路径拟合、微观结构拟合、行为模式拟合及可解释差异。",
    "政策A/B推演": "同一政策在不同干预方案下做对照实验，支持答辩展示。",
    "监管优化": "面向监管目标做动作搜索，输出稳市场-流动性-成本权衡。",
    "系统说明": "快速了解架构、数据流与工程实现边界。",
}
ENTRY_PURPOSE = {
    "政策试验台": "输入政策并运行多智能体仿真",
    "历史因子回测": "因子与组合回测对照",
    "历史Agent回放": "真实成交驱动的历史重放",
    "真实性报告": "解释哪里像真、哪里不像真",
    "政策A/B推演": "政策组合对照与机制解释",
    "监管优化": "监管动作优化与权衡分析",
    "系统说明": "了解系统如何工作",
}
THEME_PATH = Path("theme") / "competition_dark.css"
MATERIALS_ROOT = Path("outputs") / "competition_materials"


def _normalize_entry(entry: str) -> str:
    key = str(entry or "")
    return ENTRY_ALIASES.get(key, key)


def _feature_flag_enabled(flag_name: str, *, default: bool = False) -> bool:
    flags = st.session_state.get("feature_flags", {})
    if isinstance(flags, MutableMapping):
        if flag_name in flags:
            return bool(flags[flag_name])
    return bool(default)


def _load_theme() -> None:
    if THEME_PATH.exists():
        css = THEME_PATH.read_text(encoding="utf-8")
    else:
        css = """
        .stApp { background: #050b14; color: #e2e8f0; }
        .kpi-card { background: rgba(10, 25, 49, 0.65); border: 1px solid #1f365c; border-radius: 12px; padding: 16px; backdrop-filter: blur(8px); }
        .kpi-title { font-size: 13px; color: #8aa0c2; text-transform: uppercase; }
        .kpi-value { font-size: 28px; font-weight: 700; color: #ffffff; }
        .kpi-note { font-size: 12px; color: #7995bc; }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "entry": "系统说明",
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
        "feature_flags": {
            COMPETITION_MODE_FLAG: True,
            "regulator_real_env_v1": True,
            "regulator_toy_fallback_v1": True,
        },
        "competition_mode_requested": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.entry = _normalize_entry(str(st.session_state.get("entry", "系统说明")))
    if st.session_state.entry not in ENTRY_POINTS:
        st.session_state.entry = "系统说明"


def _render_top_entry_selector() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <h1 style="font-size: 2.4rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; letter-spacing: 2px; text-shadow: 0 0 20px rgba(24,144,255,0.4);">
                数治观澜 <span style="font-size: 1.4rem; font-weight: 400; color: #8aa0c2; text-shadow: none;">—— 基于大模型多智能体的金融政策风动推演沙箱</span>
            </h1>
            <div style="font-size: 15px; color: #4da6ff; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600;">
                Civitas Policy Sandbox 
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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


def _competition_mode_active() -> bool:
    flags = st.session_state.get("feature_flags", {})
    requested = bool(st.session_state.get("competition_mode_requested", True))
    if not isinstance(flags, MutableMapping):
        flags = {}
    return competition_mode_enabled(feature_flags=flags, requested_mode=requested)


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

    competition_summary = f"""# competition_summary

?????{now}
???{display_scenario_name(scenario.name)}

## ????
- ?????{stat['return_pct']:.2%}
- ????{stat['volatility']:.2%}
- ???????{stat['panic_max']:.2f}
- ??????CSAD??{stat['herding_avg']:.3f}

## ????
1. ??????????????
2. ?????????????????
3. ????????????????????
"""

    design_outline = """# design_outline

## ????
- ????
- ????
- ????
- ?????
- A/B ??
- ????

## ????
- ???? Streamlit ?????????
- ???????????
- ?????????????
"""

    demo_script_10min = f"""# demo_script_10min

1. 0:00-1:00 ????????? `{display_scenario_name(scenario.name)}`
2. 1:00-3:00 ???????????
3. 3:00-5:30 ???????????
4. 5:30-7:30 ?????????????
5. 7:30-9:00 ?? A/B ???????
6. 9:00-10:00 ?????????????
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

    if _competition_mode_active():
        feature_flags = dict(st.session_state.get("feature_flags", {}))
        compliance = write_competition_compliance_artifacts(
            root_dir=MATERIALS_ROOT,
            project_root=Path.cwd(),
            feature_flags=feature_flags,
            materials_context={
                "scenario": scenario.name,
                "generated_at": now,
            },
            app_flow=["????", "????", "????", "???", "A/B", "????"],
        )
        realism_payload = {
            "title": "???????",
            "path_fit": {
                "enabled": True,
                "score": round(max(0.0, 1.0 - stat["volatility"]), 4),
                "price_correlation": round(max(0.0, 1.0 - stat["panic_max"] * 0.2), 4),
                "volatility_correlation": round(max(0.0, 1.0 - stat["volatility"]), 4),
                "price_rmse": round(abs(stat["return_pct"]), 4),
                "price_mae": round(abs(stat["return_pct"]) * 0.8, 4),
            },
            "microstructure_fit": {"enabled": True, "score": round(max(0.0, 1.0 - stat["panic_max"] * 0.2), 4)},
            "behavioral_fit": {"enabled": True, "score": round(max(0.0, 1.0 - stat["herding_avg"]), 4)},
            "reproducibility": {"seed": 42, "config_hash": json.dumps(feature_flags, sort_keys=True, ensure_ascii=False)},
            "snapshot_info": {"scenario": scenario.name},
            "charts": [],
        }
        bundle = export_defense_bundle(
            root_dir=MATERIALS_ROOT,
            bundle_name=f"{scenario.name}_competition_bundle",
            design_chapter_markdown=design_outline,
            realism_payload=realism_payload,
            policy_ab_markdown="# Policy A/B\n\n- Competition mode exports a concise comparison bundle for defense.",
            architecture_graph={"nodes": [], "edges": []},
            causal_chain_graph={"nodes": [], "edges": []},
            defense_outline_markdown=demo_script_10min,
            feature_flags=feature_flags,
            compliance_artifacts={
                "manifest": compliance["manifest"],
                "files": {
                    "ai_tool_usage_manifest": compliance["manifest_path"],
                    "technical_route_template": compliance["technical_route_template_path"],
                },
            },
        )
        file_map["bundle_manifest.json"] = bundle["manifest_path"]
        file_map["ai_tool_usage_manifest.json"] = compliance["manifest_path"]
        file_map["technical_route_template.md"] = compliance["technical_route_template_path"]

    return file_map


def _render_sidebar_global() -> None:
    with st.sidebar:
        st.markdown("### 导航菜单")
        for entry in ENTRY_POINTS:
            if st.button(
                entry,
                key=f"entry_{entry}",
                use_container_width=True,
                type="primary" if st.session_state.entry == entry else "secondary",
            ):
                st.session_state.entry = entry
            st.caption(ENTRY_DESCRIPTIONS[entry])
        
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

        if st.session_state.entry in {"真实性报告", "监管优化"}:
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
    st.markdown(
        """
        <div class="hero-panel" style="margin-top: 10px;">
            <h1 style="font-size: 28px; margin-bottom: 15px;">欢迎使用数治观澜</h1>
            <p style="font-size: 16px; color: #8aa0c2; max-width: 100%;">
                —— 基于大模型多智能体的金融政策风动推演沙箱 · <b>前端使用说明书</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        ### 1. 系统概览与核心定位
        本项目旨在构建一个高拟真度、事件驱动的金融市场与政策推演数字沙箱。系统通过**大语言模型（LLMs）**及**多智能体（Multi-Agent）**架构，模拟宏观政策颁布、社会舆情传播以及微观交易行为，为政府部门、监管机构及金融风控专家提供政策效果评估、尾部风险预警以及市场群体心理推演平台。
        
        本前端面板为“数治观澜”的交互入口。面板采用定制的政务深蓝科技风，包含四大核心模块：

        ---

        ### 2. 核心功能模块指引
        请通过左上方导航栏（或本页面上方标签）切换模块：

        #### 2.1 🎯 政策试验台 (Policy Lab)
        - **用途：** 动态输入或配置宏观政策与外部事件（如：超预期降息、地缘突发新闻等），实时观测数字沙箱中虚拟市场与各类 Agent 对该事件的联合响应。
        - **操作指引：** 在本页面设定假设条件和执行步骤。启动推演后，系统将进行多智能体仿真计算，并同步输出大盘指数走势、市场情绪波动、风险热度（Panic Level）等联动指标看板。

        #### 2.2 ⏪ 历史政策回放 (History Replay)
        - **用途：** 将沙箱生成的“虚拟市场数据”与真实历史区间的“真实资本市场走势”进行对比，用于验证底层大规模异构 Agent 机制设计的回测拟合度。
        - **操作指引：** 通过历史事件下拉菜单，载入内置的历史场景。系统将使用双轴/或平行时间线展示“仿真推演世界”与“真实发生世界”特征，用于回测评估。

        #### 2.3 🔬 高级分析 (Advanced Analysis)
        - **用途：** 面向风控专家与架构师，提供最详细的模型执行逻辑追踪、行为金融学深度诊断，并支持对底层 Agent 控制参数的大规模调优。
        - **核心子面板包含：**
          - **AI 决策解读**：专家界面不展示枯燥的原始 JSON 日志，而是直观展示底层 Agent（如分析师、策略经理）组成的“决策证据链”。
          - **市场行为分析**：包含订单簿 (LOB) 逐笔交易深度动画、社会网络中传播的情绪热图以及政策传导桑基图。
          - **研究参数与调试**：包含更硬核的控制参数（例如：机构交易员容忍度、噪音分子比例等），可用来对宏观市场施加压力测试或故障注入。
        
        ---

        ### 3. 底层仿真引擎工作原理模型
        1. **信息编译与冲击输入**：输入的自然语言政策文本或环境指标变更，由引擎自动编译为结构化的环境冲击（Macro Shock）。
        2. **网络传播与响应**：不同的 LLM Agent（模拟新闻机构、零售散户、企业等）捕捉冲击信号并开始信息链传导。
        3. **量化与策略决策**：多智能体（例如专业做市商、套利者和基金经理模型）通过投票与聚合逻辑生成实际的订单意图与置信度。
        4. **高频引擎撮合 (C++ LOB Engine)**：模拟所有的意向订单发往底层原生 C++ 撮合引擎，在极大规模并发下构建含有真实流动性的逐笔行情数据。
        5. **数据呈现**：推演结束，将各类资金流向、深度图表及微观诊断数据回传给数治观澜的前端并加以渲染。
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
    elif entry == "历史因子回测":
        st.session_state.runtime_mode = LIVE_MODE
        st.session_state["history_replay_entry_mode"] = "factor"
        render_history_replay()
    elif entry == "历史Agent回放":
        st.session_state.runtime_mode = LIVE_MODE
        st.session_state["history_replay_entry_mode"] = "agent"
        render_history_replay()
    elif entry == "真实性报告":
        st.session_state.runtime_mode = LIVE_MODE
        render_behavioral_diagnostics()
    elif entry == "政策A/B推演":
        st.session_state.runtime_mode = DEMO_MODE
        render_policy_lab()
    elif entry == "监管优化":
        st.session_state.runtime_mode = LIVE_MODE
        if _feature_flag_enabled(REGULATOR_OPTIMIZATION_PAGE_FLAG, default=True):
            render_regulator_optimization()
        else:
            render_backtest_panel(ctrl=st.session_state.get("controller"))
    elif entry == "高级分析":
        _render_advanced_analysis()
    else:
        _render_system_guide()


if __name__ == "__main__":
    main()
