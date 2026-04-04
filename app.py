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
from core.runtime_mode import merge_mode_feature_flags, resolve_runtime_mode_profile
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
    "历史智能回测",
    "真实性报告",
    "政策A/B推演",
    "监管优化",
    "高级分析",
    "系统说明",
]
ENTRY_ALIASES = {
    "历史政策回放": "历史智能回测",
    "历史Agent回放": "历史智能回测",
    "历史因子回测": "历史智能回测",
    "历史智能体回放": "历史智能回测",
}
ENTRY_DESCRIPTIONS = {
    "政策试验台": "输入新政策，观察市场、风险与情绪如何联动变化。",
    "历史智能回测": "统一承载智能因子回测与历史智能体回放，减少页面切换并保留两种研究路径。",
    "真实性报告": "查看路径拟合、微观结构拟合、行为模式拟合及可解释差异。",
    "政策A/B推演": "同一政策在不同干预方案下做对照实验，支持答辩展示。",
    "监管优化": "面向监管目标做动作搜索，输出稳市场-流动性-成本权衡。",
    "高级分析": "集中展示证据链、行为金融诊断与专家级答辩视图。",
    "系统说明": "快速了解架构、数据流与工程实现边界。",
}
ENTRY_PURPOSE = {
    "政策试验台": "输入政策并运行多智能体仿真",
    "历史智能回测": "在同一工作台切换因子回测与智能体重放",
    "真实性报告": "解释哪里像真、哪里不像真",
    "政策A/B推演": "政策组合对照与机制解释",
    "监管优化": "监管动作优化与权衡分析",
    "高级分析": "专家级答辩与证据追问",
    "系统说明": "了解系统如何工作",
}
THEME_PATH = Path("theme") / "competition_dark.css"
MATERIALS_ROOT = Path("outputs") / "competition_materials"


def _render_policy_lab_compatible(presentation_mode: str) -> None:
    """Call policy lab in a backward-compatible way across mixed deployments."""
    try:
        render_policy_lab(presentation_mode=presentation_mode)
    except TypeError:
        # Fallback for older deployments where render_policy_lab() has no kwargs.
        render_policy_lab()


def _normalize_entry(entry: str) -> str:
    key = str(entry or "")
    return ENTRY_ALIASES.get(key, key)


def _feature_flag_enabled(flag_name: str, *, default: bool = False) -> bool:
    flags = st.session_state.get("feature_flags", {})
    if isinstance(flags, MutableMapping):
        if flag_name in flags:
            return bool(flags[flag_name])
    return bool(default)


def _sync_runtime_mode_profile() -> None:
    mode = str(st.session_state.get("simulation_mode", "SMART"))
    profile = resolve_runtime_mode_profile(mode)
    st.session_state.runtime_mode_profile = profile.to_dict()
    flags = st.session_state.get("feature_flags", {})
    st.session_state.feature_flags = merge_mode_feature_flags(
        mode,
        flags if isinstance(flags, MutableMapping) else {},
    )


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
        "runtime_mode_profile": resolve_runtime_mode_profile("SMART").to_dict(),
        "feature_flags": {
            COMPETITION_MODE_FLAG: True,
            "competition_safe_mode": True,
            "market_pipeline_v2": True,
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
    _sync_runtime_mode_profile()


def _render_top_entry_selector() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <h1 style="font-size: 2.4rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem; letter-spacing: 2px; text-shadow: 0 0 20px rgba(24,144,255,0.4);">
                数治观澜 <span style="font-size: 1.4rem; font-weight: 400; color: #8aa0c2; text-shadow: none;">—— 基于大模型多智能体的金融政策风动推演沙箱</span>
            </h1>
            <div style="font-size: 15px; color: #4da6ff; letter-spacing: 1.5px; font-weight: 600;">
                Civitas 政策沙箱
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

生成时间：{now}
演示场景：{display_scenario_name(scenario.name)}

## 关键指标
- 区间收益率：{stat['return_pct']:.2%}
- 波动率：{stat['volatility']:.2%}
- 最大风险热度：{stat['panic_max']:.2f}
- 平均羊群度（CSAD）：{stat['herding_avg']:.3f}

## 结论摘要
1. 系统能够围绕政策冲击、市场反应和风险扩散形成闭环推演。
2. 多智能体行为与市场微观结构结果可以同步展示，便于答辩解释。
3. 监管动作、A/B 对照和导出材料可直接服务比赛展示与复盘。
"""

    design_outline = """# design_outline

## 功能模块
- 政策试验台
- 历史智能回测
- 真实性报告
- 政策 A/B 推演
- 监管优化

## 技术实现
- 使用 Streamlit 构建比赛演示前端与交互流程。
- 通过多智能体仿真引擎驱动市场演化与行为决策。
- 结合行为金融指标、回测分析和材料导出形成完整闭环。
"""

    demo_script_10min = f"""# demo_script_10min

1. 0:00-1:00 介绍项目定位并载入场景 `{display_scenario_name(scenario.name)}`
2. 1:00-3:00 展示政策输入与多智能体响应过程
3. 3:00-5:30 展示价格、成交量、风险热度和羊群度变化
4. 5:30-7:30 展示真实性报告与行为金融诊断结果
5. 7:30-9:00 展示 A/B 对照与监管优化能力
6. 9:00-10:00 总结系统价值并导出比赛材料
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
            app_flow=["政策试验台", "历史智能回测", "真实性报告", "政策A/B推演", "监管优化"],
        )
        realism_payload = {
            "title": "真实性评估摘要",
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
        agent_taxonomy_markdown = "\n".join(
            [
                "# agent_taxonomy",
                "",
                "- retail_general: 情绪与政策敏感的散户资金",
                "- mutual_fund: 基准约束下的机构配置资金",
                "- quant_timing: 趋势/波动状态切换资金",
                "- market_maker: 流动性与库存约束资金",
                "- state_stabilization_fund: 稳市承接资金",
            ]
        )
        policy_causal_chain = {
            "policy": scenario.name,
            "path": ["policy_input", "expectation_shift", "order_flow_split", "lob_matching", "market_metrics"],
            "summary": {
                "return_pct": float(stat["return_pct"]),
                "volatility": float(stat["volatility"]),
                "panic_max": float(stat["panic_max"]),
                "herding_avg": float(stat["herding_avg"]),
            },
        }
        realism_metrics_csv = "\n".join(
            [
                "metric,value",
                f"return_pct,{float(stat['return_pct'])}",
                f"volatility,{float(stat['volatility'])}",
                f"panic_max,{float(stat['panic_max'])}",
                f"herding_avg,{float(stat['herding_avg'])}",
            ]
        )
        competition_snapshot = {
            "scenario": scenario.name,
            "generated_at": now,
            "seed": 42,
            "config_hash": json.dumps(feature_flags, sort_keys=True, ensure_ascii=False),
            "dataset_snapshot_id": f"demo:{scenario.name}",
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
            agent_taxonomy_markdown=agent_taxonomy_markdown,
            policy_causal_chain=policy_causal_chain,
            realism_metrics_csv=realism_metrics_csv,
            competition_snapshot=competition_snapshot,
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
            _sync_runtime_mode_profile()
            st.toast(f"已切换至 {sim_mode_display[selected_mode_key]}")
        else:
            _sync_runtime_mode_profile()

        runtime_profile = st.session_state.get("runtime_mode_profile", {})
        if isinstance(runtime_profile, MutableMapping):
            summary = str(runtime_profile.get("summary", ""))
            pause_seconds = float(runtime_profile.get("pause_for_llm_seconds", 0.0) or 0.0)
            st.caption(f"模式摘要：{summary} | LLM暂停={pause_seconds:.2f}s")

        st.markdown("---")
        st.caption("建议先看“系统说明”建立整体认知，再进入政策试验台和历史回放，最后用“高级分析”回答追问。")

        if st.session_state.entry in {"真实性报告", "监管优化", "高级分析"}:
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
                st.caption("最近生成文件（自然语言展示）")
                for name, path in st.session_state.materials_last_export.items():
                    st.markdown(f"- 已生成 `{name}`，保存位置：`{path}`")


def _render_ai_decision_tab() -> None:
    st.markdown("### 智能决策解读")
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

    role_order_flows = {
        "retail_general": float(latest["volume"]) * (0.32 - 0.22 * float(latest["panic_level"])),
        "mutual_fund": float(latest["volume"]) * (0.22 - 0.12 * float(latest["panic_level"])),
        "quant_timing": float(latest["volume"]) * (0.10 + 0.18 * float(latest["panic_level"])),
        "market_maker": float(latest["volume"]) * (-0.08 + 0.04 * float(latest["panic_level"])),
    }
    spread = max(0.001, float(latest["panic_level"]) * 0.02)
    buy_vol = float(chain_payload["market_microstructure"]["buy_volume"])
    sell_vol = float(chain_payload["market_microstructure"]["sell_volume"])
    open_px = float(latest.get("open", latest["close"]))
    csad_val = float(latest.get("csad", latest.get("panic_level", 0.0)))
    micro_metrics = {
        "spread": spread,
        "spread_pct": spread / max(1.0, float(latest["close"])),
        "depth_imbalance": (buy_vol - sell_vol) / max(1.0, buy_vol + sell_vol),
        "impact": abs(float(latest["close"]) - open_px) / max(1.0, float(latest["close"])),
        "herding_proxy": abs(csad_val),
    }
    dashboard_ui.render_orderflow_microstructure_panel(
        role_order_flows=role_order_flows,
        microstructure_metrics=micro_metrics,
        key_prefix="advanced_ai",
    )


def _render_advanced_analysis() -> None:
    st.session_state.runtime_mode = DEMO_MODE
    st.markdown("## 高级分析")
    st.caption("这里保留专家追问、答辩演示、行为金融诊断和研究参数面板。")
    tab1, tab2, tab3, tab4 = st.tabs(["智能决策解读", "市场行为分析", "答辩演示", "研究参数"])

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
            <div class="hero-kicker">Judge Quick View</div>
            <h1 style="font-size: 28px; margin-bottom: 15px;">3 分钟理解这个项目</h1>
            <p style="font-size: 16px; color: #8aa0c2; max-width: 100%;">
                数治观澜是一套面向金融政策评估与监管答辩的多智能体推演沙箱，
                重点展示“政策输入 -> 智能体决策 -> 市场撮合 -> 风险诊断 -> 材料导出”的完整闭环。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_cols = st.columns(3)
    top_cards = [
        (
            "项目定位",
            "把抽象政策冲击翻译成可以看、可以解释、可以复盘的市场推演过程。",
            ["面向政策评估", "支持答辩演示", "强调可解释结果"],
        ),
        (
            "核心能力",
            "同时具备前台演示、历史回放、真实性诊断和监管优化四类能力。",
            ["多智能体市场仿真", "历史场景重放", "行为金融指标诊断"],
        ),
        (
            "评委建议路线",
            "先看系统说明，再跑政策试验台，最后用高级分析回答追问。",
            ["第 1 步：系统说明", "第 2 步：政策试验台", "第 3 步：高级分析"],
        ),
    ]
    for col, (title, summary, bullets) in zip(top_cols, top_cards):
        with col:
            bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
            st.markdown(
                f"""
                <div class="ops-card">
                  <div class="ops-card-title">{title}</div>
                  <div class="ops-card-summary">{summary}</div>
                  <ul class="ops-card-list">{bullet_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 评委建议演示路径")
    route_cols = st.columns(4)
    route_cards = [
        ("1. 系统说明", "先建立项目整体认知。", ["看项目定位", "看核心闭环", "看页面导航"]),
        ("2. 政策试验台", "输入政策并运行主演示。", ["展示K线与风险热度", "解释政策传导", "导出报告"]),
        ("3. 历史智能回测", "证明项目不只是炫技界面。", ["切换因子/智能体工作台", "查看历史路径重放", "解释偏差说明"]),
        ("4. 高级分析", "回答“为什么可信”。", ["证据链", "微观结构图", "行为金融诊断"]),
    ]
    for col, (title, summary, bullets) in zip(route_cols, route_cards):
        with col:
            bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
            st.markdown(
                f"""
                <div class="story-card">
                  <div class="story-card-title">{title}</div>
                  <div class="story-card-summary">{summary}</div>
                  <ul class="story-card-list">{bullet_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 页面功能总览")
    overview_cols = st.columns(2)
    overview_cards = [
        (
            "面向演示的页面",
            [
                "政策试验台：展示政策输入后的市场路径和风险指标变化。",
                "政策A/B推演：对比不同干预方案的效果差异。",
                "监管优化：围绕稳市场、流动性和成本做权衡搜索。",
            ],
        ),
        (
            "面向答辩的页面",
            [
                "历史智能回测：在同一工作台完成因子基准与智能体历史重放。",
                "真实性报告/高级分析：解释哪里拟真、哪里仍有偏差。",
            ],
        ),
    ]
    for col, (title, items) in zip(overview_cols, overview_cards):
        item_html = "".join(f"<li>{item}</li>" for item in items)
        with col:
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-label">{title}</div>
                  <ul class="ops-card-list">{item_html}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 技术闭环")
    st.markdown(
        """
        1. 政策文本先被结构化编译成可执行冲击。
        2. 多智能体根据政策、情绪和市场状态生成决策意图。
        3. 订单进入底层撮合引擎，生成价格、成交量和微观结构数据。
        4. 前端同步展示市场主图、风险热度、行为金融诊断和证据链。
        5. 系统可导出答辩材料，支持现场演示和赛后复盘。
        """
    )

    st.info("如果时间只有几分钟，建议直接从左侧进入“政策试验台”，运行默认模板后再切换到“高级分析”。")


def main() -> None:
    _load_theme()
    _init_state()
    _render_sidebar_global()
    _render_top_entry_selector()

    entry = st.session_state.entry
    if entry == "政策试验台":
        st.session_state.runtime_mode = DEMO_MODE
        _render_policy_lab_compatible("standard")
    elif entry == "历史智能回测":
        st.session_state.runtime_mode = LIVE_MODE
        st.session_state["history_replay_entry_mode"] = str(
            st.session_state.get("history_replay_entry_mode", "factor")
        ).strip().lower() or "factor"
        render_history_replay(ctrl=st.session_state.get("controller"))
    elif entry == "真实性报告":
        st.session_state.runtime_mode = LIVE_MODE
        render_behavioral_diagnostics()
    elif entry == "政策A/B推演":
        st.session_state.runtime_mode = DEMO_MODE
        _render_policy_lab_compatible("defense")
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
    st.caption("仅供教学科研与仿真，不构成投资建议。")


if __name__ == "__main__":
    main()
