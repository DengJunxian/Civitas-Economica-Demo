"""History replay page focused on policy playback versus real market data."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtester import BacktestConfig, BacktestResult, HistoricalBacktester
from ui import dashboard as dashboard_ui
from ui.policy_lab import _compile_scaled_shock, _load_policy_templates
from ui.reporting import official_report_meta, write_report_artifacts


INDEX_OPTIONS = {
    "上证指数 (sh000001)": "sh000001",
    "沪深300 (sh000300)": "sh000300",
    "深证成指 (sz399001)": "sz399001",
    "创业板指 (sz399006)": "sz399006",
}

BACKGROUND_TEMPLATES = {
    "宽松期": 0.10,
    "中性环境": 0.00,
    "紧缩期": -0.12,
    "风险事件期": -0.20,
    "政策托底期": 0.16,
}

HISTORY_REPORT_DIR = Path("outputs") / "history_reports"


def _default_window() -> tuple[date, date]:
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=240)
    return start, end


def _compile_policy_score(policy_text: str, strength: float, background: str) -> float:
    shock = _compile_scaled_shock(policy_text, 1.0)
    base = (
        shock.liquidity_injection * 1.2
        + shock.fiscal_stimulus_delta * 1.4
        - shock.policy_rate_delta * 50.0
        - shock.credit_spread_delta * 20.0
        - shock.stamp_tax_delta * 420.0
        + shock.sentiment_delta * 1.2
        + shock.rumor_shock * 1.7
    )
    return float(np.clip(base * strength + BACKGROUND_TEMPLATES.get(background, 0.0), -1.0, 1.0))


def _build_replay_metrics(result: BacktestResult) -> Dict[str, float]:
    if len(result.real_prices) < 3 or len(result.simulated_prices) < 3:
        return {
            "trend_alignment": 0.0,
            "turning_point_match": 0.0,
            "drawdown_gap": 0.0,
            "vol_similarity": 0.0,
            "response_gap": 0.0,
        }

    real = np.asarray(result.real_prices, dtype=float)
    sim = np.asarray(result.simulated_prices, dtype=float)
    real_ret = np.diff(real) / np.maximum(real[:-1], 1e-12)
    sim_ret = np.diff(sim) / np.maximum(sim[:-1], 1e-12)

    sign_match = float(np.mean(np.sign(real_ret) == np.sign(sim_ret)))

    real_turn = np.sign(np.diff(np.sign(real_ret), prepend=real_ret[0]))
    sim_turn = np.sign(np.diff(np.sign(sim_ret), prepend=sim_ret[0]))
    turning_point_match = float(np.mean(real_turn == sim_turn))

    def _max_drawdown(prices: np.ndarray) -> float:
        peaks = np.maximum.accumulate(prices)
        dd = prices / np.maximum(peaks, 1e-12) - 1.0
        return float(abs(dd.min()))

    drawdown_gap = abs(_max_drawdown(real) - _max_drawdown(sim))
    vol_similarity = float(np.clip(result.volatility_correlation, 0.0, 1.0))

    threshold = 0.015
    real_days = next((idx for idx, value in enumerate(np.cumsum(real_ret), start=1) if abs(value) >= threshold), len(real_ret))
    sim_days = next((idx for idx, value in enumerate(np.cumsum(sim_ret), start=1) if abs(value) >= threshold), len(sim_ret))
    response_gap = float(abs(real_days - sim_days))

    return {
        "trend_alignment": sign_match,
        "turning_point_match": turning_point_match,
        "drawdown_gap": drawdown_gap,
        "vol_similarity": vol_similarity,
        "response_gap": response_gap,
    }


def _build_bias_explanation(metrics: Dict[str, float], policy_name: str) -> str:
    if metrics["trend_alignment"] >= 0.65 and metrics["drawdown_gap"] <= 0.05:
        return f"这次“{policy_name}”历史回放整体可信：大方向与真实市场接近，但局部波动仍保留了模拟世界的偏差。"
    if metrics["trend_alignment"] < 0.5:
        return f"这次“{policy_name}”回放的方向一致度偏弱，说明当前政策强度或背景模板与真实历史仍有偏差。"
    if metrics["response_gap"] > 8:
        return f"这次回放的政策反应速度和真实市场有明显时差，建议继续调节政策强度或背景模板。"
    return f"这次“{policy_name}”回放在趋势上基本可用，但回撤和波动节奏仍有优化空间。"


def _compute_result_summary(result: BacktestResult) -> Dict[str, float]:
    return {
        "total_return": float(result.total_return),
        "max_drawdown": float(result.max_drawdown),
        "excess_return": float(result.excess_return),
        "price_correlation": float(result.price_correlation),
        "volatility_correlation": float(result.volatility_correlation),
        "price_rmse": float(result.price_rmse),
    }


def _run_backtest_with_progress(config: BacktestConfig, start_ratio: float, end_ratio: float, progress, status, label: str) -> BacktestResult:
    backtester = HistoricalBacktester(config)

    def _on_progress(cur: int, total: int, msg: str) -> None:
        ratio = cur / max(total, 1)
        progress.progress(start_ratio + (end_ratio - start_ratio) * ratio)
        status.caption(f"{label}：{msg}")

    return backtester.run_backtest(progress_callback=_on_progress)


def _render_comparison_chart(result: BacktestResult, baseline: Optional[BacktestResult] = None) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result.dates),
            "real": result.real_prices,
            "simulated": result.simulated_prices,
        }
    )
    if baseline and baseline.simulated_prices:
        frame["baseline"] = baseline.simulated_prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["real"], mode="lines", name="真实指数", line=dict(color="#8ec5ff", width=2.4)))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["simulated"], mode="lines", name="当前政策仿真", line=dict(color="#35d07f", width=2.4)))
    if "baseline" in frame:
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["baseline"], mode="lines", name="无政策基线", line=dict(color="#f59e0b", width=2.2, dash="dash")))
    if not frame.empty:
        fig.add_vline(x=frame["date"].iloc[0], line_color="#f59e0b", line_dash="dot")
    fig.update_layout(
        **dashboard_ui.PLOTLY_DARK_LAYOUT,
        title="真实走势 vs 仿真走势",
        yaxis=dict(title="指数点位"),
        xaxis=dict(title="时间"),
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True, key="history_replay_compare_chart")
    dashboard_ui.export_plot_bundle(fig, frame, "history_replay_compare", "history_replay_compare")


def _render_replay_cards(result: BacktestResult, metrics: Dict[str, float]) -> None:
    cols = st.columns(5)
    cards = [
        ("趋势一致度", f"{metrics['trend_alignment']:.0%}", "越高说明方向越接近"),
        ("拐点匹配度", f"{metrics['turning_point_match']:.0%}", "观察节奏是否像真实市场"),
        ("回撤差异", f"{metrics['drawdown_gap']:.2%}", "越低越自然"),
        ("波动相似度", f"{metrics['vol_similarity']:.0%}", "比较波动阶段"),
        ("反应时差", f"{metrics['response_gap']:.0f}天", "政策传导快慢差异"),
    ]
    for idx, (title, value, note) in enumerate(cards):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-title">{title}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_baseline_delta_cards(result: BacktestResult, baseline: Optional[BacktestResult]) -> None:
    if not baseline:
        return
    cards = [
        ("相对基线收益", f"{result.total_return - baseline.total_return:+.2%}", "当前政策仿真相对无政策基线的收益差"),
        ("回撤改善", f"{baseline.max_drawdown - result.max_drawdown:+.2%}", "若为正值，说明当前政策回撤更小"),
        ("超额收益", f"{result.excess_return:+.2%}", "当前政策仿真相对真实基准的超额收益"),
        ("校准提升", f"{result.volatility_correlation - baseline.volatility_correlation:+.0%}", "当前政策对波动阶段复现是否优于基线"),
    ]
    cols = st.columns(len(cards))
    for idx, (title, value, note) in enumerate(cards):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-title">{title}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_agent_readout(policy_text: str, result: BacktestResult, metrics: Dict[str, float]) -> None:
    excess = result.excess_return
    panic_hint = "更稳" if metrics["drawdown_gap"] < 0.04 else "更激进"
    cards = [
        {
            "title": "新闻分析师",
            "summary": "这段历史被系统视为一个由政策与成交情绪共同驱动的阶段。",
            "lines": [
                f"输入政策：{policy_text}",
                "系统会把政策文本先编译成结构化冲击，再进入仿真。",
                "因此 AI 不是装饰，而是决定了冲击方向和情绪偏置。",
            ],
        },
        {
            "title": "量化分析师",
            "summary": "仿真曲线已经在趋势和波动节奏上与真实走势建立了对照。",
            "lines": [
                f"趋势一致度：{metrics['trend_alignment']:.0%}",
                f"波动相似度：{metrics['vol_similarity']:.0%}",
                "目标是像真的，不是逐点贴线。",
            ],
        },
        {
            "title": "风险分析师",
            "summary": f"这一段回放表现为{panic_hint}的政策路径。",
            "lines": [
                f"回撤差异：{metrics['drawdown_gap']:.2%}",
                f"反应时差：{metrics['response_gap']:.0f}天",
                "如果时差过大，通常说明政策强度或背景模板选得不对。",
            ],
        },
        {
            "title": "经理最终判断",
            "summary": "这条仿真曲线可用于解释政策机制，不应被包装成对历史的机械复制。",
            "lines": [
                f"相对基准超额：{excess:.2%}",
                "展示时建议强调拐点、波动阶段和政策后反应。",
                "若被追问偏差，重点解释机制差异而非误差本身。",
            ],
        },
    ]

    cols = st.columns(4)
    for idx, card in enumerate(cards):
        items = "".join(f"<li>{line}</li>" for line in card["lines"])
        with cols[idx]:
            st.markdown(
                f"""
                <div class="story-card">
                  <div class="story-card-title">{card['title']}</div>
                  <div class="story-card-summary">{card['summary']}</div>
                  <ul class="story-card-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_replay_brief(bundle: Dict[str, Any], metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    trend_alignment = metrics["trend_alignment"]
    turning_point_match = metrics["turning_point_match"]
    drawdown_gap = metrics["drawdown_gap"]
    response_gap = metrics["response_gap"]
    vol_similarity = metrics["vol_similarity"]

    if trend_alignment >= 0.65 and drawdown_gap <= 0.05:
        conclusion = "这组参数已经具备展示和复盘价值，适合解释政策机制为什么大致成立。"
    elif trend_alignment >= 0.5:
        conclusion = "这组参数可以用于内部研判，但还应继续调校节奏和波动阶段。"
    else:
        conclusion = "这组参数更适合暴露模型偏差，不适合直接作为对外展示版本。"

    if response_gap > 8:
        next_step = "优先调整仿真强度或历史背景模板，先把政策反应时差拉回合理区间。"
    elif turning_point_match < 0.55:
        next_step = "优先调整观察窗口和调仓节奏，让关键拐点更接近真实历史阶段。"
    else:
        next_step = "可以继续做不同政策版本的 A/B 对照，比较哪种叙事更像真实市场。"

    lines = [
        {
            "title": "回放结论",
            "summary": conclusion,
            "lines": [
                f"趋势一致度：{trend_alignment:.0%}",
                f"拐点匹配度：{turning_point_match:.0%}",
                "这页重点证明机制是否可信，不追求逐日逐点复刻历史价格。",
            ],
        },
        {
            "title": "工程用途",
            "summary": "适合做历史验证、参数校准和向业务方解释模型边界。",
            "lines": [
                "先用它验证政策逻辑是否成立，再决定是否进入更细的参数调参。",
                "可用于答辩时说明：为什么仿真线看起来像真的，但又不会假到完全贴线。",
                f"当前波动相似度：{vol_similarity:.0%}，回撤差异：{drawdown_gap:.2%}",
            ],
        },
        {
            "title": "下一步动作",
            "summary": next_step,
            "lines": [
                f"历史背景模板：{bundle['background']}，仿真强度：{bundle['strength']:.1f}",
                "若要更贴近真实市场，应先校准节奏，再校准波动，而不是直接追求曲线重合。",
                "参数调好后，再拿去和政策试验台的新政策结果做前后呼应。",
            ],
        },
    ]
    if bundle.get("baseline_result") is not None:
        lines[1]["lines"].append("当前页面已加入“无政策基线”，可直接解释政策是否比自然市场路径更有效。")
    return lines


def _render_replay_brief(cards: List[Dict[str, Any]]) -> None:
    cols = st.columns(len(cards))
    for idx, card in enumerate(cards):
        items = "".join(f"<li>{line}</li>" for line in card["lines"])
        with cols[idx]:
            st.markdown(
                f"""
                <div class="ops-card">
                  <div class="ops-card-title">{card['title']}</div>
                  <div class="ops-card-summary">{card['summary']}</div>
                  <ul class="ops-card-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_history_report(bundle: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    result: BacktestResult = bundle["result"]
    baseline: Optional[BacktestResult] = bundle.get("baseline_result")
    report_meta = official_report_meta("history_replay", bundle["policy_name"])
    replay_cards = bundle.get("replay_cards", [])
    summary_card = replay_cards[0] if replay_cards else {"summary": "", "lines": []}
    payload = {
        "report_meta": report_meta,
        "policy_name": bundle["policy_name"],
        "policy_text": bundle["policy_text"],
        "background": bundle["background"],
        "strength": bundle["strength"],
        "symbol_label": bundle["symbol_label"],
        "start_date": bundle["start_date"],
        "end_date": bundle["end_date"],
        "metrics": metrics,
        "result_summary": _compute_result_summary(result),
        "baseline_summary": _compute_result_summary(baseline) if baseline else None,
        "replay_brief": replay_cards,
    }
    markdown = [
        f"# 关于《{bundle['policy_name']}》历史回放情况的汇报摘要",
        "",
        f"- 报告编号：{report_meta['report_no']}",
        f"- 生成日期：{report_meta['date_cn']}",
        f"- 报送对象：{report_meta['recipient']}",
        f"- 材料性质：{report_meta['classification']}",
        "",
        "## 一、历史背景",
        f"- 对照指数：{bundle['symbol_label']}",
        f"- 历史区间：{bundle['start_date']} 至 {bundle['end_date']}",
        f"- 背景模板：{bundle['background']}",
        f"- 仿真强度：{bundle['strength']:.1f}",
        "",
        "## 二、政策内容",
        bundle["policy_text"],
        "",
        "## 三、回放结论",
        f"- 总体判断：{summary_card['summary']}",
        f"- 偏差解释：{_build_bias_explanation(metrics, bundle['policy_name'])}",
        "",
        "## 四、与真实市场对照",
        f"- 趋势一致度：{metrics['trend_alignment']:.0%}",
        f"- 拐点匹配度：{metrics['turning_point_match']:.0%}",
        f"- 回撤差异：{metrics['drawdown_gap']:.2%}",
        f"- 波动相似度：{metrics['vol_similarity']:.0%}",
        f"- 反应时差：{metrics['response_gap']:.0f}天",
        "",
        "## 五、工程判读",
    ]
    for card in replay_cards:
        markdown.append(f"- {card['title']}：{card['summary']}")
        for line in card["lines"]:
            markdown.append(f"- {line}")
    if baseline:
        markdown.extend(
            [
                "",
                "## 六、无政策基线比较",
                f"- 基线收益：{baseline.total_return:.2%}",
                f"- 当前方案相对基线收益：{result.total_return - baseline.total_return:+.2%}",
                f"- 当前方案相对基线回撤改善：{baseline.max_drawdown - result.max_drawdown:+.2%}",
            ]
        )
    markdown.extend(
        [
            "",
            "## 七、风险与边界",
            "- 本回放用于复现政策作用机制与市场反应路径，不用于逐点复制真实历史价格。",
            "- 若趋势一致度不足或反应时差过大，应优先调整政策强度和历史背景模板，再校准节奏。",
            "- 展示时建议强调趋势、拐点与风险阶段，而非单日价格误差。",
            "",
            "## 八、建议事项",
            "- 若用于正式汇报，建议同时附上无政策基线和关键拐点解释。",
            "- 若用于模型校准，建议围绕趋势一致度、回撤差异和反应时差继续调参。",
        ]
    )

    export_bundle = write_report_artifacts(
        root_dir=HISTORY_REPORT_DIR,
        report_type="history_replay",
        title=bundle["policy_name"],
        markdown_text="\n".join(markdown),
        payload=payload,
    )
    export_bundle["report_meta"] = report_meta
    return export_bundle


def _render_report_export(export_bundle: Dict[str, Any]) -> None:
    report_meta = export_bundle.get("report_meta", {})
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">已生成正式历史回放摘要</div>
              <div class="summary-value">{report_meta.get('title', export_bundle['stem'])}</div>
              <div class="summary-note">报告编号：{report_meta.get('report_no', export_bundle['stem'])}</div>
              <div class="summary-note">报送对象：{report_meta.get('recipient', '政策评估组、历史复盘与校准团队')}</div>
              <div class="summary-note">现已生成带封面、固定页眉页脚的 Markdown / Word / PDF / JSON，可直接用于答辩材料、正式汇报和参数校准归档。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        top_row = st.columns(2)
        bottom_row = st.columns(2)
        with top_row[0]:
            st.download_button(
                "下载 Word 汇报",
                data=export_bundle["docx_bytes"],
                file_name=f"{export_bundle['stem']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key=f"history_docx_{export_bundle['stem']}",
            )
        with top_row[1]:
            st.download_button(
                "下载 PDF 汇报",
                data=export_bundle["pdf_bytes"],
                file_name=f"{export_bundle['stem']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"history_pdf_{export_bundle['stem']}",
            )
        with bottom_row[0]:
            st.download_button(
                "下载 Markdown 汇报",
                data=export_bundle["markdown_text"],
                file_name=f"{export_bundle['stem']}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"history_md_{export_bundle['stem']}",
            )
        with bottom_row[1]:
            st.download_button(
                "下载 JSON 附件",
                data=export_bundle["json_text"],
                file_name=f"{export_bundle['stem']}.json",
                mime="application/json",
                use_container_width=True,
                key=f"history_json_{export_bundle['stem']}",
            )


def render_history_replay() -> None:
    st.markdown(
        """
        <div class="hero-panel">
          <div class="hero-kicker">History Playback</div>
          <h1>历史政策回放</h1>
          <p>选择真实历史区间，回放政策影响，并把仿真走势与真实市场走势放在同一张图上比较。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    entries = [
        ("这页是干什么的", "把政策影响放回真实历史区间里，检验仿真结果是不是“像真的”。"),
        ("你会看到什么", "真实走势、仿真走势、无政策基线、拐点匹配度和 AI 偏差解释。"),
        ("展示时怎么讲", "强调机制复现，不强调逐点贴线，避免看起来像事后抄答案。"),
    ]
    for idx, (title, body) in enumerate(entries):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-label">{title}</div>
                  <div class="summary-note">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("历史回放的推荐使用方法"):
        st.markdown(
            """
            - 先选一个真实政策密集发生的历史阶段。
            - 再填写当时的政策描述和市场背景模板。
            - 最后观察三件事：趋势方向、拐点节奏、政策后的反应速度。
            """
        )

    if "history_replay_result" not in st.session_state:
        st.session_state.history_replay_result = None

    template_map = {item["title"]: item for item in _load_policy_templates()}
    selected_template_label = st.selectbox("快速载入模板", options=list(template_map.keys()), index=0)
    selected_template = template_map[selected_template_label]
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-label">当前模板</div>
          <div class="summary-value">{selected_template['title']}</div>
          <div class="summary-note">{selected_template['policy_goal']}</div>
          <div class="summary-label">建议观察重点</div>
          <div class="summary-note">{'、'.join(selected_template.get('expected_channels', []))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_start, default_end = _default_window()
    with st.form("history_replay_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            start_date = st.date_input("开始日期", value=default_start)
            end_date = st.date_input("结束日期", value=default_end)
            symbol_label = st.selectbox("对照指数", options=list(INDEX_OPTIONS.keys()), index=1)
            policy_name = st.text_input("政策事件名称", value=f"{selected_template['title']} 历史回放")
            policy_text = st.text_area("政策说明", value=str(selected_template["policy_text"]), height=110)
        with col2:
            background = st.selectbox("历史背景模板", options=list(BACKGROUND_TEMPLATES.keys()), index=1)
            strength = st.slider("仿真强度", min_value=0.3, max_value=1.6, value=float(selected_template.get("recommended_intensity", 1.0)), step=0.1)
            rebalance_frequency = st.select_slider("调仓节奏", options=[1, 2, 3, 5, 10], value=5)
            lookback = st.slider("观察窗口", min_value=10, max_value=80, value=20, step=5)
            enable_baseline = st.toggle("加入无政策基线", value=True)
        submitted = st.form_submit_button("运行历史回放", use_container_width=True, type="primary")

    if submitted:
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            return

        progress = st.progress(0.0)
        status = st.empty()
        policy_score = _compile_policy_score(policy_text, strength, background)
        config = BacktestConfig(
            symbol=INDEX_OPTIONS[symbol_label],
            benchmark_symbol=INDEX_OPTIONS[symbol_label],
            start_date=str(start_date),
            end_date=str(end_date),
            period_days=0,
            strategy_name="portfolio_system",
            lookback=int(lookback),
            rebalance_frequency=int(rebalance_frequency),
            max_position=1.0,
            policy_shock=policy_score,
            sentiment_weight=0.55,
            civitas_factor_weight=0.45,
        )
        try:
            result = _run_backtest_with_progress(config, 0.0, 0.7 if enable_baseline else 1.0, progress, status, "当前政策")
            baseline_result = None
            if enable_baseline:
                baseline_config = BacktestConfig(**{**config.__dict__, "policy_shock": 0.0})
                baseline_result = _run_backtest_with_progress(baseline_config, 0.7, 1.0, progress, status, "无政策基线")
        finally:
            progress.empty()
            status.empty()

        bundle = {
            "config": config,
            "policy_name": policy_name,
            "policy_text": policy_text,
            "background": background,
            "strength": strength,
            "symbol_label": symbol_label,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "result": result,
            "baseline_result": baseline_result,
            "metrics": _build_replay_metrics(result),
        }
        bundle["replay_cards"] = _build_replay_brief(bundle, bundle["metrics"])
        bundle["export_bundle"] = _build_history_report(bundle, bundle["metrics"])
        st.session_state.history_replay_result = bundle

    bundle = st.session_state.history_replay_result
    if not bundle:
        st.info("先选择一个历史区间和政策，再点击“运行历史回放”。")
        return

    result: BacktestResult = bundle["result"]
    if not result or not result.real_prices:
        st.warning("这次历史回放没有得到有效结果。请调整日期区间或指数后重试。")
        return

    metrics = bundle["metrics"]
    baseline = bundle.get("baseline_result")
    st.markdown("### 真实走势 vs 仿真走势")
    _render_comparison_chart(result, baseline)
    _render_replay_cards(result, metrics)
    _render_baseline_delta_cards(result, baseline)

    st.markdown("### 工程判读")
    _render_replay_brief(bundle["replay_cards"])

    st.markdown("### 偏差解释")
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-value">{_build_bias_explanation(metrics, bundle['policy_name'])}</div>
          <div class="summary-note">这套历史回放强调政策机制复现，而不是逐日逐点复制真实价格。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### AI 如何理解这段历史")
    _render_agent_readout(bundle["policy_text"], result, metrics)

    st.markdown("### 导出汇报摘要")
    _render_report_export(bundle["export_bundle"])
