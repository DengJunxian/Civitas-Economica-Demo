"""Dedicated regulator optimization page with A/B and Pareto visualization."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from regulator_agent import run_regulatory_closed_loop
from ui.components.optimization_report_panel import render_optimization_report_panel
from ui.components.repro_meta import build_reproducibility_meta, render_reproducibility_panel
from ui.dashboard import render_discovered_metrics_panel
from ui.narrative import narrate_payload, render_narrative_block


REGULATOR_OPTIMIZATION_PAGE_FLAG = "regulator_optimization_page_v1"
_REGULATOR_RESULT_STATE_KEY = "regulator_optimization_result"


def _safe_rows(items: Any) -> List[Dict[str, Any]]:
    if isinstance(items, list):
        return [dict(x) for x in items if isinstance(x, dict)]
    return []


def _build_regulator_result_frames(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    counterfactual = result.get("counterfactual_ab", {}) if isinstance(result, dict) else {}
    recommendation = result.get("recommendation", {}) if isinstance(result, dict) else {}
    baseline = counterfactual.get("baseline", {}) if isinstance(counterfactual, dict) else {}
    candidates = _safe_rows(counterfactual.get("candidates"))
    deltas = _safe_rows(counterfactual.get("deltas"))
    pareto = _safe_rows(result.get("pareto_frontier"))
    evidence = _safe_rows(recommendation.get("evidence_chain"))

    baseline_df = pd.DataFrame([baseline]) if isinstance(baseline, dict) and baseline else pd.DataFrame()
    candidates_df = pd.DataFrame(candidates)
    deltas_df = pd.DataFrame(deltas)
    pareto_df = pd.DataFrame(pareto)
    recommendation_df = (
        pd.DataFrame([recommendation.get("scorecard", {})])
        if isinstance(recommendation, dict) and recommendation.get("scorecard")
        else pd.DataFrame()
    )
    evidence_df = pd.DataFrame(evidence)

    for frame in (baseline_df, candidates_df, deltas_df, pareto_df, recommendation_df, evidence_df):
        for col in (
            "macro_stability",
            "liquidity",
            "intervention_cost",
            "avg_reward",
            "financing_function",
            "fairness_compliance",
            "composite_score",
            "delta",
        ):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    return {
        "baseline": baseline_df,
        "candidates": candidates_df,
        "deltas": deltas_df,
        "pareto": pareto_df,
        "recommendation": recommendation_df,
        "evidence": evidence_df,
    }


def render_regulator_optimization() -> None:
    st.markdown("## 监管策略优化")
    st.caption("独立监管优化页：默认优先真实环境，输出 A/B、Pareto、推荐方案与证据链。")

    with st.form("regulator_optimization_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            episodes = st.slider("训练轮数", min_value=10, max_value=400, value=120, step=10)
            max_steps = st.slider("每轮最大步数", min_value=4, max_value=96, value=24, step=4)
            top_k = st.slider("候选动作数量", min_value=1, max_value=5, value=3, step=1)
        with right:
            seed = int(st.number_input("随机种子", min_value=0, max_value=2_147_483_647, value=42, step=1))
            use_toy_env = st.toggle("直接使用简化环境", value=False)
            st.caption("默认优先真实环境；真实环境初始化失败时，自动回退到简化环境。")
        submitted = st.form_submit_button("运行监管优化", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("正在运行监管闭环优化..."):
            result = run_regulatory_closed_loop(
                episodes=int(episodes),
                max_steps_per_episode=int(max_steps),
                seed=int(seed),
                top_k=int(top_k),
                use_toy_env=bool(use_toy_env),
            )
            st.session_state[_REGULATOR_RESULT_STATE_KEY] = result

    result = st.session_state.get(_REGULATOR_RESULT_STATE_KEY)
    if not isinstance(result, dict):
        st.info("先运行一次优化以查看 A/B、Pareto 和推荐证据链。")
        return

    summary = result.get("training_summary", {}) if isinstance(result, dict) else {}
    reproducibility = result.get("reproducibility", {}) if isinstance(result, dict) else {}
    recommendation = result.get("recommendation", {}) if isinstance(result, dict) else {}
    blackbox = result.get("blackbox_optimization", {}) if isinstance(result, dict) else {}
    opt_report = result.get("optimization_report", {}) if isinstance(result, dict) else {}
    frames = _build_regulator_result_frames(result)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("平均回合收益", f"{float(summary.get('avg_episode_reward', 0.0)):.4f}")
    c2.metric("最优动作得分", f"{float(summary.get('best_action_score', 0.0)):.4f}")
    c3.metric("帕累托点数", str(len(frames["pareto"])))
    c4.metric("Q 状态数", str(int(summary.get("q_states", 0))))
    c5.metric("默认路径", str(result.get("default_production_path", "q_learning_baseline")))

    st.markdown("### 发现目标约束")
    render_discovered_metrics_panel(result.get("discovered_objectives", {}), key_prefix="regulator_objectives")

    st.caption(
        "可复现信息："
        f"seed={reproducibility.get('seed', 0)} | "
        f"config_hash={reproducibility.get('config_hash', '')} | "
        f"训练轮数={reproducibility.get('episodes', 0)} | "
        f"max_steps={reproducibility.get('max_steps_per_episode', 0)}"
    )
    env_selection = reproducibility.get("env_selection", {}) if isinstance(reproducibility, dict) else {}
    if isinstance(env_selection, dict) and env_selection:
        st.caption(
            "环境信息："
            f"path={env_selection.get('selected_path', '')} | "
            f"fallback={env_selection.get('fallback_used', False)}"
        )
    render_reproducibility_panel(
        build_reproducibility_meta(
            data_snapshot_hash=str(reproducibility.get("data_snapshot_hash", reproducibility.get("dataset_snapshot_id", "regulator_env_snapshot"))),
            config_hash=str(reproducibility.get("config_hash", "")),
            random_seed=int(reproducibility.get("seed", 42) or 42),
            llm_provider_chain=["regulator_q_learning", "bayesian_optimization", "nsga_ii"],
            calibration_parameter_set_id=str(reproducibility.get("calibration_parameter_set_id", "regulator_default_v1")),
            extra={"env_selection": env_selection},
        ),
        key_prefix="regulator_repro",
    )

    st.markdown("### 黑箱多目标优化")
    if isinstance(blackbox, dict) and blackbox:
        bo_best = dict(blackbox.get("bayesian_optimization", {}).get("best", {}) or {})
        validation = dict(blackbox.get("validation", {}) or {})
        report_best = dict(opt_report.get("best_solution", {}) or {})
        st.caption(
            "BO 用于静态政策包搜索；NSGA-II 输出多目标 Pareto；晋升默认路径需在两个固定 replay window 同时优于规则 baseline。"
        )
        cols = st.columns(4)
        cols[0].metric("BO 最优分", f"{float(bo_best.get('score', 0.0)):.4f}")
        cols[1].metric("稳定胜场", f"{int(validation.get('stable_win_count', 0))}/{int(validation.get('required_windows', 2))}")
        cols[2].metric("晋升默认", str(bool(validation.get("promote_blackbox_default", False))))
        cols[3].metric("NSGA 点数", str(len(blackbox.get("nsga_ii", {}).get("pareto_frontier", []) or [])))
        if report_best:
            st.dataframe(pd.DataFrame([report_best.get("metrics", {})]), use_container_width=True, hide_index=True)
        if validation.get("windows"):
            st.dataframe(pd.DataFrame(validation.get("windows", [])), use_container_width=True, hide_index=True)
        nsga_pareto = blackbox.get("nsga_ii", {}).get("pareto_frontier", [])
        if nsga_pareto:
            pareto_rows = []
            for row in nsga_pareto:
                if not isinstance(row, dict):
                    continue
                metrics = dict(row.get("metrics", {}) or {})
                objectives = dict(row.get("objectives", {}) or {})
                pareto_rows.append({"score": row.get("score", 0.0), **metrics, **objectives})
            if pareto_rows:
                st.dataframe(pd.DataFrame(pareto_rows).head(20), use_container_width=True, hide_index=True)
        render_narrative_block(
            "黑箱优化报告解读",
            {
                "best_solution": opt_report.get("best_solution", {}),
                "constraint_violations": opt_report.get("constraint_violations", {}),
                "validation": opt_report.get("validation", {}),
                "final_recommendation_text": opt_report.get("final_recommendation_text", ""),
            },
            context="请解释 BO、NSGA-II、规则 baseline 与 Q-learning 的关系，并说明为什么当前默认生产路径可以或不可以晋升。",
            cache_namespace="regulator_opt_narrative_cache",
        )
    else:
        st.info("暂无黑箱优化输出。")

    render_optimization_report_panel(result, key_prefix="regulator_optimization_report")

    st.markdown("### 反事实对照（A/B）")
    left, right = st.columns(2)
    with left:
        if frames["baseline"].empty:
            st.info("暂无基线结果。")
        else:
            st.dataframe(frames["baseline"], use_container_width=True, hide_index=True)
    with right:
        if frames["deltas"].empty:
            st.info("暂无候选动作差分。")
        else:
            st.dataframe(frames["deltas"], use_container_width=True, hide_index=True)
    render_narrative_block(
        "监管 A/B 对照解读",
        {
            "baseline": frames["baseline"].to_dict(orient="records"),
            "deltas": frames["deltas"].head(10).to_dict(orient="records"),
        },
        context="请解释基线方案与候选动作之间的主要差异，说明哪些指标改善最大、代价最大。",
        cache_namespace="regulator_opt_narrative_cache",
    )

    st.markdown("### 帕累托前沿")
    pareto_df = frames["pareto"]
    if pareto_df.empty:
        st.info("暂无 Pareto 数据。")
    else:
        hover_cols = [c for c in ("action_description", "action_signature", "avg_reward") if c in pareto_df.columns]
        fig = px.scatter(
            pareto_df,
            x="intervention_cost",
            y="macro_stability",
            size="liquidity",
            color="avg_reward" if "avg_reward" in pareto_df.columns else None,
            hover_data=hover_cols,
            title="帕累托前沿（稳定性 vs 成本，气泡大小=流动性）",
        )
        fig.update_layout(height=480, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)
        render_narrative_block(
            "Pareto 前沿解读",
            pareto_df.head(12).to_dict(orient="records"),
            context="请解释稳定性、成本与流动性之间的权衡关系，并指出推荐方案为何位于合理区间。",
            cache_namespace="regulator_opt_narrative_cache",
        )

    st.markdown("### 推荐方案与证据")
    left, right = st.columns(2)
    with left:
        st.markdown(
            narrate_payload(
                "推荐方案解读",
                {
                    "action_description": recommendation.get("action_description", ""),
                    "tradeoff_summary": recommendation.get("tradeoff_summary", ""),
                    "side_effects": recommendation.get("side_effects", []),
                },
                context="解释推荐动作、收益-成本权衡和可能副作用。",
            )
        )
        if not frames["recommendation"].empty:
            st.dataframe(frames["recommendation"], use_container_width=True, hide_index=True)
    with right:
        if not frames["evidence"].empty:
            st.dataframe(frames["evidence"], use_container_width=True, hide_index=True)
        else:
            st.info("暂无推荐证据。")

    if not frames["candidates"].empty:
        st.markdown("### 候选方案")
        st.dataframe(frames["candidates"], use_container_width=True, hide_index=True)
        render_narrative_block(
            "候选方案清单解读",
            frames["candidates"].head(12).to_dict(orient="records"),
            context="请概括候选方案的分布特征，并指出适合重点汇报的 1 到 2 个候选动作。",
            cache_namespace="regulator_opt_narrative_cache",
        )

    export_cols = st.columns(2)
    export_cols[0].download_button(
        "下载监管优化结果 JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="regulator_optimization_result.json",
        mime="application/json",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "下载帕累托数据 CSV",
        data=frames["pareto"].to_csv(index=False).encode("utf-8"),
        file_name="regulator_pareto.csv",
        mime="text/csv",
        use_container_width=True,
    )


__all__ = ["REGULATOR_OPTIMIZATION_PAGE_FLAG", "_build_regulator_result_frames", "render_regulator_optimization"]
