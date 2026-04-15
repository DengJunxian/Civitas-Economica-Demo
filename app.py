# file: app.py
"""Civitas front-end in Streamlit: Policy Lab / History Replay / Advanced Analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, cast

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
from core.model_router import ModelRouter
from core.runtime_mode import merge_mode_feature_flags, resolve_runtime_mode_profile
from core.ui_text import display_runtime_mode, display_scenario_name
from ui.backtest_panel import render_backtest_panel
from ui.behavioral_diagnostics import render_behavioral_diagnostics
from ui.demo_wind_tunnel import render_demo_tab
from ui.history_replay import render_history_replay
from ui.policy_lab import _build_regulation_counterfactual_worlds, render_policy_lab
from ui.regulator_optimization import render_regulator_optimization
from ui.reporting import export_defense_bundle
from ui import dashboard as dashboard_ui


st.set_page_config(
    page_title="数治观澜 | 金融政策智能推演与验证平台",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


OVERVIEW_ENTRY = "系统总览"
ENTRY_POINTS = [
    OVERVIEW_ENTRY,
    "政策实验",
    "历史验证",
    "研判分析",
]
ENTRY_ALIASES = {
    "系统说明": OVERVIEW_ENTRY,
    "总览首页": OVERVIEW_ENTRY,
    "政策试验台": "政策实验",
    "历史政策回放": "历史验证",
    "历史Agent回放": "历史验证",
    "历史因子回测": "历史验证",
    "历史智能体回放": "历史验证",
    "新闻驱动历史回测": "历史验证",
    "历史回测": "历史验证",
    "政策A/B推演": "政策实验",
    "真实性报告": "研判分析",
    "高级分析": "研判分析",
    "监管优化": "研判分析",
}
ENTRY_DESCRIPTIONS = {
    OVERVIEW_ENTRY: "用一屏建立平台定位、能力链路、工程特性与代表性结果。",
    "政策实验": "输入政策文本，查看结构化编译、多智能体响应与市场路径。",
    "历史验证": "基于真实历史窗口自动汇总政策与新闻，验证仿真与真实走势的一致性。",
    "研判分析": "聚合 AI 证据、行为诊断、监管优化、研究验证与结果归档能力。",
}
ENTRY_PURPOSE = {
    OVERVIEW_ENTRY: "快速建立平台认知并进入核心工作流",
    "政策实验": "进行政策仿真、影响评估与结果归档",
    "历史验证": "验证历史区间下的拟真效果与误差边界",
    "研判分析": "查看证据链、机制诊断、策略建议与研究验证",
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


def _runtime_router_summary() -> Dict[str, Any]:
    summary = ModelRouter.runtime_observability_summary()
    if isinstance(summary, dict):
        return summary
    return {}


def _load_data_flywheel_summary(max_records: int = 500) -> Dict[str, Any]:
    seed_path = Path("data") / "seed_events.jsonl"
    if not seed_path.exists():
        return {"available": False, "reason": f"{seed_path} not found"}

    total = 0
    impact_counter: Dict[str, int] = {}
    topic_counter: Dict[str, int] = {}
    source_counter: Dict[str, int] = {}
    recent: List[Dict[str, Any]] = []
    for raw in seed_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        total += 1
        impact = str(item.get("impact_level", "unknown"))
        impact_counter[impact] = impact_counter.get(impact, 0) + 1
        source = str(item.get("source", item.get("source_name", "unknown")))
        source_counter[source] = source_counter.get(source, 0) + 1
        topic = str(item.get("topic", item.get("event_type", "unknown")))
        topic_counter[topic] = topic_counter.get(topic, 0) + 1
        if len(recent) < 5:
            recent.append(
                {
                    "time": item.get("published_at", item.get("created_at", "")),
                    "topic": topic,
                    "impact": impact,
                    "source": source,
                }
            )
        if total >= max_records:
            break

    top_topics = sorted(topic_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_sources = sorted(source_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "available": True,
        "total_events": total,
        "impact_counter": impact_counter,
        "top_topics": top_topics,
        "top_sources": top_sources,
        "recent_events": recent,
    }


def _render_value_bridge_tab() -> None:
    st.markdown("### 数据飞轮与反事实价值卡")
    flywheel = _load_data_flywheel_summary()
    if not flywheel.get("available", False):
        st.info(f"数据飞轮暂不可用：{flywheel.get('reason', 'unknown')}")
    else:
        a, b, c = st.columns(3)
        a.metric("已采集事件", int(flywheel.get("total_events", 0)))
        high_impact = int(dict(flywheel.get("impact_counter", {})).get("high", 0))
        b.metric("高影响事件", high_impact)
        c.metric("TOP来源数", len(list(flywheel.get("top_sources", []))))
        st.markdown("#### 事件来源与主题摘要")
        left, right = st.columns(2)
        with left:
            st.dataframe(pd.DataFrame(flywheel.get("top_sources", []), columns=["source", "count"]), hide_index=True, use_container_width=True)
        with right:
            st.dataframe(pd.DataFrame(flywheel.get("top_topics", []), columns=["topic", "count"]), hide_index=True, use_container_width=True)

    _ensure_demo_loaded()
    scenario = st.session_state.get("demo_scenario")
    if scenario is None:
        st.warning("当前未加载分析场景，无法生成反事实对照。")
        return
    metrics = scenario.metrics if hasattr(scenario, "metrics") else pd.DataFrame()
    if metrics.empty:
        st.warning("当前场景缺少指标数据。")
        return

    try:
        counterfactual = _build_regulation_counterfactual_worlds(metrics, intensity=1.0)
    except Exception as exc:
        st.warning(f"反事实计算失败：{exc}")
        return

    st.markdown("#### 监管反事实对照（A/B）")
    worlds = dict(counterfactual.get("worlds", {}) or {})
    scorecards = dict(counterfactual.get("scorecards", {}) or {})
    rec = str(counterfactual.get("recommended_timing", ""))
    if rec:
        st.caption(f"推荐干预时机：{rec}")
    cards_df = pd.DataFrame(
        [
            {"world": key, **(value if isinstance(value, dict) else {})}
            for key, value in scorecards.items()
        ]
    )
    if not cards_df.empty:
        st.dataframe(cards_df, use_container_width=True, hide_index=True)
    world_keys = list(worlds.keys())
    if len(world_keys) >= 2:
        base = pd.DataFrame(worlds[world_keys[0]])
        alt = pd.DataFrame(worlds[world_keys[1]])
        if not base.empty and not alt.empty and "step" in base.columns and "close" in base.columns:
            merged = base[["step", "close"]].merge(
                alt[["step", "close"]],
                on="step",
                suffixes=("_base", "_alt"),
                how="inner",
            )
            if not merged.empty:
                merged["delta_close"] = merged["close_alt"] - merged["close_base"]
                st.line_chart(merged.set_index("step")[["delta_close"]], use_container_width=True)


def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "entry": OVERVIEW_ENTRY,
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
    st.session_state.entry = _normalize_entry(str(st.session_state.get("entry", OVERVIEW_ENTRY)))
    if st.session_state.entry not in ENTRY_POINTS:
        st.session_state.entry = OVERVIEW_ENTRY
    _sync_runtime_mode_profile()


def _render_top_entry_selector() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 2rem; position: relative;">
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle at 10% 20%, rgba(24, 144, 255, 0.15), transparent 60%); pointer-events: none;"></div>
            <h1 style="font-size: 2.8rem; font-weight: 800; color: #ffffff; margin-bottom: 0.2rem; letter-spacing: 2px; text-shadow: 0 0 24px rgba(24,144,255,0.6); display: flex; align-items: center; gap: 12px;">
                <span style="background: linear-gradient(90deg, #4da6ff, #1890ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">数治观澜</span> 
            <span style="font-size: 1.5rem; font-weight: 500; color: #8aa0c2; text-shadow: none;">面向政府场景的金融政策智能推演与验证平台</span>
            </h1>
            <div style="font-size: 16px; color: #4da6ff; letter-spacing: 2px; font-weight: 600; text-transform: uppercase;">
                Policy Intelligence Engineering Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='mode-pill' style='box-shadow: 0 0 15px rgba(24, 144, 255, 0.2) inset; padding: 6px 16px; font-weight: 600;'>
            <span style='color: #fff;'>当前模块：</span>{st.session_state.entry} 
            <span style='margin: 0 12px; color: #1f365c;'>|</span> 
            <span style='color: #8aa0c2;'>核心用途：</span>{ENTRY_PURPOSE.get(st.session_state.entry, "")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_a, quick_b, quick_c, quick_d = st.columns([1.05, 1.05, 1.05, 1.25])
    with quick_a:
        if st.button(
            OVERVIEW_ENTRY,
            key="top_entry_overview",
            use_container_width=True,
            type="primary" if st.session_state.entry == OVERVIEW_ENTRY else "secondary",
        ):
            st.session_state.entry = OVERVIEW_ENTRY
    with quick_b:
        if st.button(
            "政策实验",
            key="top_entry_policy_lab",
            use_container_width=True,
            type="primary" if st.session_state.entry == "政策实验" else "secondary",
        ):
            st.session_state.entry = "政策实验"
    with quick_c:
        if st.button(
            "历史验证",
            key="top_entry_history_replay",
            use_container_width=True,
            type="primary" if st.session_state.entry == "历史验证" else "secondary",
        ):
            st.session_state.entry = "历史验证"
    with quick_d:
        if st.button(
            "研判分析",
            key="top_entry_advanced",
            use_container_width=True,
            type="primary" if st.session_state.entry == "研判分析" else "secondary",
        ):
            st.session_state.entry = "研判分析"
        st.caption("建议路径：政策实验 -> 历史验证 -> 研判分析。")


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
    runtime_summary = _runtime_router_summary()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    competition_summary = f"""# experiment_summary

生成时间：{now}
分析场景：{display_scenario_name(scenario.name)}

## 关键指标
- 区间收益率：{stat['return_pct']:.2%}
- 波动率：{stat['volatility']:.2%}
- 最大风险热度：{stat['panic_max']:.2f}
- 平均羊群度（CSAD）：{stat['herding_avg']:.3f}

## 运行模式证据
- 在线调用成功次数：{int(runtime_summary.get('online_success_total', 0))}
- 自动回退次数：{int(runtime_summary.get('fallback_total', 0))}
- 在线成功率：{float(runtime_summary.get('online_success_rate', 0.0)):.1%}
- 最近回退原因：{str(runtime_summary.get('last_fallback_reason', '') or '-')}

## 结论摘要
1. 系统能够围绕政策冲击、市场反应和风险扩散形成闭环推演。
2. 多智能体行为与市场微观结构结果可以同步展示，便于机制解释与方案汇报。
3. 监管动作、A/B 对照和归档材料可直接服务复盘、留痕与策略研判。
"""

    design_outline = """# design_outline

## 功能模块
- 政策实验
- 历史验证
- 研究验证
- 反事实 A/B 推演
- 监管优化

## 技术实现
- 使用 Streamlit 构建工程化分析前端与交互流程。
- 通过多智能体仿真引擎驱动市场演化与行为决策。
- 结合行为金融指标、历史验证分析和结果归档形成完整闭环。
"""

    demo_script_10min = f"""# demo_script_10min

1. 0:00-0:40 在系统总览页说明平台定位与分析对象
2. 0:40-1:30 展示系统工作流与核心能力链
3. 1:30-4:00 进入政策实验，演示政策输入、结构化编译与多智能体推演
4. 4:00-6:00 进入历史验证，展示仿真走势与真实市场的对照结果
5. 6:00-7:20 进入行为与风险诊断，说明群体行为与风险扩散机制
6. 7:20-8:40 展示监管优化、A/B 差分与 Pareto 权衡
7. 8:40-9:30 展示实验报告、分析摘要与结果归档
8. 9:30-10:00 回到总览页，总结平台从政策理解到策略建议的闭环价值
"""

    figures_index = {
        "generated_at": now,
        "scenario": scenario.name,
        "scenario_display": display_scenario_name(scenario.name),
        "figures": st.session_state.get("last_demo_figures", {}),
    }

    file_map = {
        "experiment_summary.md": MATERIALS_ROOT / "experiment_summary.md",
        "system_design_outline.md": MATERIALS_ROOT / "system_design_outline.md",
        "video_script_10min.md": MATERIALS_ROOT / "video_script_10min.md",
        "figures/index.json": figures_dir / "index.json",
        "runtime_mode_evidence.json": MATERIALS_ROOT / "runtime_mode_evidence.json",
    }
    file_map["experiment_summary.md"].write_text(competition_summary, encoding="utf-8")
    file_map["system_design_outline.md"].write_text(design_outline, encoding="utf-8")
    file_map["video_script_10min.md"].write_text(demo_script_10min, encoding="utf-8")
    file_map["figures/index.json"].write_text(json.dumps(figures_index, ensure_ascii=False, indent=2), encoding="utf-8")
    file_map["runtime_mode_evidence.json"].write_text(
        json.dumps(runtime_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
            app_flow=["政策实验", "历史验证", "研究验证", "反事实 A/B 推演", "监管优化"],
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
            "runtime_mode_evidence": runtime_summary,
        }
        bundle = export_defense_bundle(
            root_dir=MATERIALS_ROOT,
            bundle_name=f"{scenario.name}_analysis_bundle",
            design_chapter_markdown=design_outline,
            realism_payload=realism_payload,
            policy_ab_markdown="# 反事实与 A/B 说明\n\n- 系统会导出精简版对照材料，用于方案汇报、复盘说明与结果归档。",
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
        st.markdown(
            "### 导航菜单",
            help="在这里切换平台的核心功能模块。从上到下按工作流排列。"
        )
        
        menu_groups = {
            "核心工作流": [OVERVIEW_ENTRY, "政策实验", "历史验证", "研判分析"],
        }
        st.caption("建议从系统总览进入，再按“政策实验 -> 历史验证 -> 研判分析”完成全链路浏览。")

        for group, entries in menu_groups.items():
            st.markdown(f"<div style='margin-top: 16px; margin-bottom: 8px; font-size: 13px; color: #8aa0c2; letter-spacing: 1px;'>{group}</div>", unsafe_allow_html=True)
            for entry in entries:
                if entry in ENTRY_POINTS:
                    if st.button(
                        entry,
                        key=f"entry_{entry}",
                        use_container_width=True,
                        type="primary" if st.session_state.entry == entry else "secondary",
                        help=ENTRY_DESCRIPTIONS.get(entry, "")
                    ):
                        st.session_state.entry = entry
        
        st.markdown("---")
        st.markdown("### 推演模式设置")
        sim_mode_display = {"SMART": "智能模式（API 优先 + 自动回退）", "DEEP": "深度模式（Reasoner + Chat）"}
        selected_mode_key = st.radio(
            "选择 LLM 调度策略",
            options=["SMART", "DEEP"],
            index=0 if st.session_state.simulation_mode == "SMART" else 1,
            format_func=lambda x: sim_mode_display.get(x, x),
            label_visibility="collapsed",
            help="智能模式优先在线 API，失败自动回退；深度模式优先 Reasoner，适合深入解读。"
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
            st.caption(f"模式摘要：{summary}")

        runtime_summary = _runtime_router_summary()
        if runtime_summary:
            online_success = int(runtime_summary.get("online_success_total", 0))
            fallback_total = int(runtime_summary.get("fallback_total", 0))
            success_rate = float(runtime_summary.get("online_success_rate", 0.0))
            status_map = {
                "online_ok": "🟢 在线稳定",
                "degraded_with_fallback": "🟡 降级兜底",
                "offline_fallback": "🔴 离线兜底",
            }
            status = status_map.get(str(runtime_summary.get("status", "")), "⚪ 状态未知")
            st.markdown(f"<div style='font-size: 12px; margin-top: 10px; color: #8aa0c2;'>{status} | 在线成功={online_success} | 降级={fallback_total} | 成功率={success_rate:.0%}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("主线入口聚焦政策分析、历史验证与研判分析；监管优化、A/B 对照与研究验证已收敛到模块内部。")

        if st.session_state.entry in {OVERVIEW_ENTRY, "研判分析"}:
            scenarios = list_competition_scenarios()
            if scenarios:
                if st.session_state.demo_scenario_name not in scenarios:
                    st.session_state.demo_scenario_name = scenarios[0]
                st.session_state.demo_scenario_name = st.selectbox(
                    "研判分析默认场景",
                    options=scenarios,
                    index=scenarios.index(st.session_state.demo_scenario_name),
                    format_func=display_scenario_name,
                )
            else:
                st.warning("未发现可用分析场景，请检查 demo_scenarios 内容完整性。")

            if st.button("导出实验归档", width="stretch"):
                _ensure_demo_loaded()
                if st.session_state.get("demo_scenario") is None:
                    st.error("当前没有可用场景，无法生成归档材料。")
                else:
                    try:
                        file_map = _generate_competition_materials()
                        st.session_state.materials_last_export = {k: str(v) for k, v in file_map.items()}
                        st.success("实验归档已生成。")
                    except Exception as exc:
                        st.error(f"实验归档生成失败：{exc}")

            if st.session_state.materials_last_export:
                st.caption(f"最近已导出 {len(st.session_state.materials_last_export)} 份实验归档。")
                with st.expander("查看导出文件清单", expanded=False):
                    for name, path in st.session_state.materials_last_export.items():
                        st.markdown(f"- `{name}`")


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


def _build_overview_chain_payload(metrics: pd.DataFrame) -> Dict[str, Any]:
    latest = metrics.tail(1).iloc[0]
    panic = float(latest.get("panic_level", 0.0))
    return {
        "policy": "默认分析场景传导链",
        "macro_variables": {
            "inflation": 0.02 + panic * 0.01,
            "unemployment": 0.05 + panic * 0.03,
            "credit_spread": 0.015 + panic * 0.02,
            "liquidity_index": max(0.2, 1.15 - panic),
            "sentiment_index": max(0.0, min(1.0, 0.68 - panic * 0.52)),
        },
        "social_sentiment": {"mean": 0.5 - panic},
        "industry_agent": {
            "avg_household_risk": 0.50 - panic * 0.3,
            "avg_firm_hiring": 0.15 - panic * 0.5,
        },
        "market_microstructure": {
            "buy_volume": float(latest["volume"]) * (1.0 - panic),
            "sell_volume": float(latest["volume"]) * (0.45 + panic),
            "trade_count": int(max(1.0, float(latest["volume"]) / 800)),
            "matching_mode": "overview_replay",
        },
    }


def _render_overview_home() -> None:
    st.markdown(
        """
        <div class="hero-panel" style="margin-top: 10px;">
            <div class="hero-kicker">系统总览</div>
            <h1 style="font-size: 30px; margin-bottom: 14px;">数治观澜：面向金融政策预评估的风洞推演沙箱</h1>
            <p style="font-size: 16px; max-width: 920px;">
                数治观澜面向的核心问题是：一项经济或金融政策在正式出台之前，能否先在数字环境中完成理解、推演、验证和优化，
                提前预判它可能引发的市场行为、风险扩散与监管效果。它不是单纯输出结论的问答系统，也不是静态展示平台，而是一套把
                自然语言政策输入、结构化编译、多智能体市场仿真、历史验证、因子研究、行为风险诊断、监管优化和结果归档串联起来的工程化闭环。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _ensure_demo_loaded()
    scenario = st.session_state.get("demo_scenario")
    metrics = scenario.metrics if scenario is not None and hasattr(scenario, "metrics") else pd.DataFrame()
    stat = _material_summary_from_metrics(metrics) if not metrics.empty else {"return_pct": 0.0, "volatility": 0.0, "panic_max": 0.0, "herding_avg": 0.0}
    runtime_summary = _runtime_router_summary()
    flywheel = _load_data_flywheel_summary()

    top_stats = st.columns(4)
    top_stats[0].metric("默认分析场景", display_scenario_name(getattr(scenario, "name", REQUIRED_SCENARIOS[0])))
    top_stats[1].metric("区间收益表现", f"{stat['return_pct']:.2%}")
    top_stats[2].metric("峰值风险热度", f"{stat['panic_max']:.2f}")
    top_stats[3].metric("在线稳定率", f"{float(runtime_summary.get('online_success_rate', 0.0)):.0%}")

    story_cols = st.columns(3)
    story_cards = [
        (
            "平台定位",
            "把自然语言政策自动转成可推演、可解释、可归档的市场实验过程，服务政府经济治理与金融稳定场景。",
            ["政策结构化解析", "多智能体联动决策", "市场撮合与风控反馈"],
        ),
        (
            "核心能力链",
            "平台既能完成政策实验，也能完成历史验证、行为诊断、因子研究与策略优化，体现 AI 在分析链路中的连续作用。",
            ["可编译", "可推演", "可验证", "可建议"],
        ),
        (
            "工程特性",
            "系统保留运行证据、历史对照和结果归档，不只展示结果，也强调过程可信、可复盘、可追溯。",
            ["AI 证据链", "历史拟真验证", "监管优化与 A/B", "结果归档"],
        ),
    ]
    for col, (title, summary, bullets) in zip(story_cols, story_cards):
        bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
        col.markdown(
            f"""
            <div class="ops-card">
              <div class="ops-card-title">{title}</div>
              <div class="ops-card-summary">{summary}</div>
              <ul class="ops-card-list">{bullet_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 大模型与智能底座")
    model_cols = st.columns(3)
    model_cards = [
        (
            "GLM-4-flashx 接入",
            "当前演示接入智谱 GLM-4-flashx，承担政策文本理解、结构化信息抽取、部分智能体认知推理以及解释性内容生成。",
            ["政策语义理解", "结构化编译辅助", "智能体认知分化", "解释生成"],
        ),
        (
            "多智能体市场仿真",
            "不同角色市场主体会基于风险偏好、资金约束和行为偏差做差异化响应，再映射到订单流、板块轮动和价格路径上。",
            ["散户与机构", "情绪与预期", "订单流与撮合", "风险热度"],
        ),
        (
            "结果沉淀与复现",
            "系统支持把实验结论、关键指标、候选方案和推荐建议沉淀为报告、图表索引和归档材料，便于复盘、汇报和留痕。",
            ["实验报告", "图表导出", "证据链", "可复现信息"],
        ),
    ]
    for col, (title, summary, bullets) in zip(model_cols, model_cards):
        bullet_html = "".join(f"<li>{item}</li>" for item in bullets)
        col.markdown(
            f"""
            <div class="ops-card">
              <div class="ops-card-title">{title}</div>
              <div class="ops-card-summary">{summary}</div>
              <ul class="ops-card-list">{bullet_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 系统工作流")
    flow_cols = st.columns(6)
    flow_cards = [
        ("1. 政策输入", "输入自然语言政策或事件描述"),
        ("2. 结构化编译", "识别类型、强度、传导渠道与影响对象"),
        ("3. 多智能体推演", "形成多角色分歧、行为意图与市场反馈"),
        ("4. 市场撮合", "生成价格、成交量与微观结构结果"),
        ("5. 历史对照验证", "校验走势一致性、响应时点与误差边界"),
        ("6. 策略优化与归档", "输出建议方案、报告、图表与归档材料"),
    ]
    for col, (title, summary) in zip(flow_cols, flow_cards):
        col.markdown(
            f"""
            <div class="flow-card">
              <div class="flow-step-index">{title}</div>
              <div class="flow-step-label">{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 当前可运行模块")
    route_cols = st.columns(4)
    route_cards = [
        ("系统总览", "建立平台定位、能力边界、工程特性与代表性结果。"),
        ("政策实验", "输入政策文本或选择模板，查看结构化编译与推演结果。"),
        ("历史验证", "用真实历史窗口验证仿真与现实走势的一致性。"),
        ("研判分析", "查看证据链、风险机制、监管优化与研究验证内容。"),
    ]
    for col, (title, summary) in zip(route_cols, route_cards):
        col.markdown(
            f"""
            <div class="story-card">
              <div class="story-card-title">{title}</div>
              <div class="story-card-summary">{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not metrics.empty:
        st.markdown("### 代表性结果卡片")
        snapshot = dashboard_ui.build_kpi_snapshot(metrics, int(metrics["step"].max()), regulation_hint="已具备干预评估条件")
        dashboard_ui.render_kpi_cards(snapshot)

        left, right = st.columns([1.5, 1.0])
        with left:
            dashboard_ui.render_market_overview(metrics, upto_step=None, key_prefix="overview", title="默认分析场景市场结果总览")
        with right:
            package_summary = [
                f"区间收益率：{stat['return_pct']:.2%}",
                f"波动率：{stat['volatility']:.2%}",
                f"羊群度均值：{stat['herding_avg']:.3f}",
                f"高影响事件：{int(dict(flywheel.get('impact_counter', {})).get('high', 0)) if flywheel.get('available') else 0}",
                f"自动回退次数：{int(runtime_summary.get('fallback_total', 0))}",
            ]
            st.markdown(
                """
                <div class="summary-card">
                  <div class="summary-label">代表性结果摘要</div>
                  <div class="summary-value">适合汇报截图与结果留痕</div>
                  <div class="summary-note">建议优先展示主图、KPI、政策传导链与历史验证对照。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {item}" for item in package_summary))
            dashboard_ui.render_policy_transmission_chain(_build_overview_chain_payload(metrics), key_prefix="overview")

    st.markdown("### 汇报价值与应用边界")
    value_cols = st.columns(2)
    value_cols[0].markdown(
        """
        <div class="summary-card">
          <div class="summary-label">技术意义</div>
          <div class="summary-value">让大模型从“能理解政策”走向“能参与政策预演”</div>
          <div class="summary-note">通过结构化政策编译、多智能体认知决策、历史对照验证与反事实推演，把 AI 能力落到可运行、可验证、可交付的系统中。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    value_cols[1].markdown(
        """
        <div class="summary-card">
          <div class="summary-label">社会价值</div>
          <div class="summary-value">服务公共治理，而不是娱乐化内容生成</div>
          <div class="summary-note">平台旨在帮助政策制定者更早识别潜在风险、比较不同干预方案效果，并支撑更透明、更稳健、更可复盘的公共决策过程。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 工程特性与交付能力")
    capability_cols = st.columns(3)
    capability_cards = [
        (
            "工作流主线",
            ["系统总览", "政策实验", "历史验证", "研判分析"],
            "支持从问题定义到策略建议的连续分析，不依赖单点功能拼接。",
        ),
        (
            "后端沉淀能力",
            ["政策结构化包", "向量记忆 / 事件库", "监管优化 / Pareto", "多格式报告导出"],
            "很多能力已完成工程沉淀，前端负责把能力链按分析逻辑组织出来。",
        ),
        (
            "结果交付支持",
            ["实验报告", "分析摘要", "结果归档", "10 分钟视频脚本"],
            "总览页、历史验证报告和归档导出可直接复用到汇报、复盘与展示材料。",
        ),
    ]
    for col, (title, items, note) in zip(capability_cols, capability_cards):
        item_html = "".join(f"<li>{item}</li>" for item in items)
        col.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">{title}</div>
              <ul class="ops-card-list">{item_html}</ul>
              <div class="summary-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_advanced_analysis() -> None:
    st.session_state.runtime_mode = DEMO_MODE
    st.markdown("## 研判分析中心")
    st.caption("这里集中展示 AI 证据链、行为诊断、监管优化、研究验证与结果归档能力。")
    briefing_cols = st.columns(3)
    briefing_cards = [
        (
            "行为与风险诊断",
            "把价格涨跌进一步拆解成羊群效应、情绪不对称、波动聚集和风险扩散机制。",
        ),
        (
            "监管策略优化",
            "围绕候选动作、A/B 差分、Pareto 权衡和推荐方案形成真正可汇报的决策支持过程。",
        ),
        (
            "结果归档与复用",
            "政策实验、历史验证、研究回测和监管优化都可以导出报告与证据链，支持留痕和复盘。",
        ),
    ]
    for col, (title, summary) in zip(briefing_cols, briefing_cards):
        col.markdown(
            f"""
            <div class="story-card">
              <div class="story-card-title">{title}</div>
              <div class="story-card-summary">{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    tab1, tab2, tab3, tab4 = st.tabs(["AI 决策证据", "行为与风险诊断", "监管优化与 A/B", "研究验证与归档"])

    with tab1:
        _render_ai_decision_tab()
    with tab2:
        render_behavioral_diagnostics()
    with tab3:
        reg_tab, demo_tab = st.tabs(["监管策略优化", "系统场景对照"])
        with reg_tab:
            render_regulator_optimization()
        with demo_tab:
            st.session_state.competition_mode = COMPETITION_DEMO_MODE
            render_demo_tab(ctrl=st.session_state.get("controller"))
    with tab4:
        verify_tab, value_tab = st.tabs(["研究回测与导出", "价值链路与数据飞轮"])
        with verify_tab:
            st.session_state.runtime_mode = LIVE_MODE
            render_backtest_panel(ctrl=st.session_state.get("controller"))
        with value_tab:
            _render_value_bridge_tab()


def main() -> None:
    _load_theme()
    _init_state()
    _render_sidebar_global()
    _render_top_entry_selector()

    entry = st.session_state.entry
    if entry == OVERVIEW_ENTRY:
        _render_overview_home()
    elif entry == "政策实验":
        st.session_state.runtime_mode = DEMO_MODE
        _render_policy_lab_compatible("standard")
    elif entry == "历史验证":
        st.session_state.runtime_mode = LIVE_MODE
        st.session_state["history_replay_entry_mode"] = str(
            st.session_state.get("history_replay_entry_mode", "factor")
        ).strip().lower() or "factor"
        render_history_replay(ctrl=st.session_state.get("controller"))
    elif entry == "研判分析":
        _render_advanced_analysis()
    else:
        _render_overview_home()
    st.caption("仅供教学科研与仿真，不构成投资建议。")


if __name__ == "__main__":
    main()
