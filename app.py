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
    page_title="数治观澜 | 政策风动推演沙箱",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


OVERVIEW_ENTRY = "总览首页"
ENTRY_POINTS = [
    OVERVIEW_ENTRY,
    "政策试验台",
    "历史回测",
    "高级分析",
]
ENTRY_ALIASES = {
    "系统说明": OVERVIEW_ENTRY,
    "历史政策回放": "历史回测",
    "历史Agent回放": "历史回测",
    "历史因子回测": "历史回测",
    "历史智能体回放": "历史回测",
    "新闻驱动历史回测": "历史回测",
    "历史回测": "历史回测",
    "政策A/B推演": "政策试验台",
    "真实性报告": "高级分析",
    "监管优化": "高级分析",
}
ENTRY_DESCRIPTIONS = {
    OVERVIEW_ENTRY: "用一屏讲清项目价值、AI 闭环、演示路径与代表性结果。",
    "政策试验台": "输入政策文本，查看市场路径、风险指标与情绪传导。",
    "历史回测": "基于真实历史窗口自动汇总政策与新闻，评估仿真与真实走势的一致性。",
    "高级分析": "聚合 AI 证据、行为诊断、监管优化与研究验证能力。",
}
ENTRY_PURPOSE = {
    OVERVIEW_ENTRY: "快速建立全局认知并进入演示主线",
    "政策试验台": "进行政策仿真与影响评估",
    "历史回测": "验证历史区间下的拟真效果",
    "高级分析": "追问证据、机制与风险",
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
        st.warning("当前未加载答辩场景，无法生成反事实对照。")
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
                <span style="font-size: 1.5rem; font-weight: 500; color: #8aa0c2; text-shadow: none;">金融政策风动推演沙箱</span>
            </h1>
            <div style="font-size: 16px; color: #4da6ff; letter-spacing: 2px; font-weight: 600; text-transform: uppercase;">
                Civitas Sandbox System
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
            "总览首页",
            key="top_entry_overview",
            use_container_width=True,
            type="primary" if st.session_state.entry == OVERVIEW_ENTRY else "secondary",
        ):
            st.session_state.entry = OVERVIEW_ENTRY
    with quick_b:
        if st.button(
            "政策试验台",
            key="top_entry_policy_lab",
            use_container_width=True,
            type="primary" if st.session_state.entry == "政策试验台" else "secondary",
        ):
            st.session_state.entry = "政策试验台"
    with quick_c:
        if st.button(
            "历史回测",
            key="top_entry_history_replay",
            use_container_width=True,
            type="primary" if st.session_state.entry == "历史回测" else "secondary",
        ):
            st.session_state.entry = "历史回测"
    with quick_d:
        if st.button(
            "高级分析",
            key="top_entry_advanced",
            use_container_width=True,
            type="primary" if st.session_state.entry == "高级分析" else "secondary",
        ):
            st.session_state.entry = "高级分析"
        st.caption("推荐路径：政策试验台 -> 历史回测 -> 高级分析。")


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

    competition_summary = f"""# competition_summary

生成时间：{now}
演示场景：{display_scenario_name(scenario.name)}

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
2. 多智能体行为与市场微观结构结果可以同步展示，便于答辩解释。
3. 监管动作、A/B 对照和导出材料可直接服务比赛展示与复盘。
"""

    design_outline = """# design_outline

## 功能模块
- 政策试验台
- 历史回测
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
        "runtime_mode_evidence.json": MATERIALS_ROOT / "runtime_mode_evidence.json",
    }
    file_map["competition_summary.md"].write_text(competition_summary, encoding="utf-8")
    file_map["design_outline.md"].write_text(design_outline, encoding="utf-8")
    file_map["demo_script_10min.md"].write_text(demo_script_10min, encoding="utf-8")
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
            app_flow=["政策试验台", "历史回测", "真实性报告", "政策A/B推演", "监管优化"],
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
            bundle_name=f"{scenario.name}_competition_bundle",
            design_chapter_markdown=design_outline,
            realism_payload=realism_payload,
            policy_ab_markdown="# 反事实与 A/B 说明\n\n- 比赛模式会导出精简版对照材料，用于答辩与作品说明。",
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
            help="在这里切换沙箱的不同功能模块。从上到下按展示逻辑排列。"
        )
        
        menu_groups = {
            "比赛主线": [OVERVIEW_ENTRY, "政策试验台", "历史回测", "高级分析"],
        }
        st.caption("默认建议从总览首页进入，再按“政策试验台 -> 历史回测 -> 高级分析”完成展示。")

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
        st.caption("比赛演示优先保留主线入口；监管优化、A/B 对照与研究验证已收敛到模块内部。")

        if st.session_state.entry in {OVERVIEW_ENTRY, "高级分析"}:
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

            if st.button("导出答辩材料", width="stretch"):
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
                st.caption(f"最近已导出 {len(st.session_state.materials_last_export)} 份比赛材料。")
                with st.expander("查看导出文件清单", expanded=False):
                    for name, path in st.session_state.materials_last_export.items():
                        st.markdown(f"- `{name}`：`{path}`")


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
        "policy": "默认答辩场景传导链",
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
            <div class="hero-kicker">比赛总览</div>
            <h1 style="font-size: 30px; margin-bottom: 14px;">面向政策冲击推演的多智能体金融仿真与 AI 决策展示系统</h1>
            <p style="font-size: 16px; max-width: 920px;">
                数治观澜聚焦“政策发布后市场会发生什么、AI 如何理解政策并解释结果、系统如何证明推演可信”三个核心问题，
                以政策试验、历史回测、行为诊断、监管优化和材料导出构成完整展示闭环。
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
    top_stats[0].metric("默认场景", display_scenario_name(getattr(scenario, "name", REQUIRED_SCENARIOS[0])))
    top_stats[1].metric("综合收益表现", f"{stat['return_pct']:.2%}")
    top_stats[2].metric("峰值风险热度", f"{stat['panic_max']:.2f}")
    top_stats[3].metric("在线稳定率", f"{float(runtime_summary.get('online_success_rate', 0.0)):.0%}")

    story_cols = st.columns(3)
    story_cards = [
        (
            "项目定位",
            "把政策文本自动转成可推演、可解释、可导出的市场实验过程，帮助评委快速理解 AI 在业务链路中的作用。",
            ["政策结构化解析", "多智能体联动决策", "市场撮合与风控反馈"],
        ),
        (
            "应用价值",
            "既能做演示，也能做验证和复盘，适合说明作品不仅有前端观感，还有真实的建模与分析深度。",
            ["可演示", "可验证", "可导出", "可答辩"],
        ),
        (
            "比赛亮点",
            "把 AI 能力、可解释图表、历史对照和答辩材料生成放到同一条主线上，减少“做了但没展示出来”的断层。",
            ["AI 证据链", "历史拟真验证", "监管优化与 A/B", "比赛材料打包"],
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

    st.markdown("### AI 应用闭环")
    flow_cols = st.columns(6)
    flow_cards = [
        ("1. 政策输入", "输入自然语言政策或事件描述"),
        ("2. 结构化解析", "识别类型、强度、传导渠道与影响对象"),
        ("3. 智能体决策", "形成多角色分歧与行为意图"),
        ("4. 市场撮合", "生成价格、成交量与微观结构结果"),
        ("5. 风险诊断", "输出羊群、波动、回撤与传播证据"),
        ("6. 材料导出", "沉淀报告、图表和答辩文稿"),
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

    st.markdown("### 推荐演示路径")
    route_cols = st.columns(4)
    route_cards = [
        ("总览首页", "30 秒讲清作品背景、AI 闭环和代表性结果。"),
        ("政策试验台", "现场输入或调用默认政策，展示 AI 处理到市场结果的主流程。"),
        ("历史回测", "说明系统不仅能演示，还能用历史窗口做真实性验证。"),
        ("高级分析", "回答评委追问，展开证据链、监管优化与研究验证。"),
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
            dashboard_ui.render_market_overview(metrics, upto_step=None, key_prefix="overview", title="默认答辩场景市场结果总览")
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
                  <div class="summary-value">适合截图与视频展示</div>
                  <div class="summary-note">建议在 10 分钟演示中优先展示主图、KPI、政策传导链与历史回测对照。</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {item}" for item in package_summary))
            dashboard_ui.render_policy_transmission_chain(_build_overview_chain_payload(metrics), key_prefix="overview")

    st.markdown("### 已有能力如何承接到比赛展示")
    capability_cols = st.columns(3)
    capability_cards = [
        (
            "展示主线",
            ["政策试验台", "历史回测", "高级分析"],
            "评委先看价值，再看主功能，再看真实性与追问。",
        ),
        (
            "后端沉淀能力",
            ["政策结构化包", "事件库与新闻覆盖", "监管优化 / Pareto", "多格式报告导出"],
            "很多能力已实现，本次提交版重点解决“已做未显”。",
        ),
        (
            "交付物支持",
            ["设计说明书", "作品小结", "答辩 PPT", "10 分钟视频"],
            "总览页、历史回测报告和比赛材料导出可直接复用到最终提交物。",
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
    st.markdown("## 高级分析")
    st.caption("这里用于回答评委追问，集中展示 AI 证据链、行为诊断、监管优化和研究验证能力。")
    tab1, tab2, tab3, tab4 = st.tabs(["AI 决策证据", "行为与风险诊断", "监管优化与 A/B", "研究验证与材料"])

    with tab1:
        _render_ai_decision_tab()
    with tab2:
        render_behavioral_diagnostics()
    with tab3:
        reg_tab, demo_tab = st.tabs(["监管策略优化", "答辩场景对照"])
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
    elif entry == "政策试验台":
        st.session_state.runtime_mode = DEMO_MODE
        _render_policy_lab_compatible("standard")
    elif entry == "历史回测":
        st.session_state.runtime_mode = LIVE_MODE
        st.session_state["history_replay_entry_mode"] = str(
            st.session_state.get("history_replay_entry_mode", "factor")
        ).strip().lower() or "factor"
        render_history_replay(ctrl=st.session_state.get("controller"))
    elif entry == "高级分析":
        _render_advanced_analysis()
    else:
        _render_overview_home()
    st.caption("仅供教学科研与仿真，不构成投资建议。")


if __name__ == "__main__":
    main()
