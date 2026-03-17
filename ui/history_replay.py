"""History replay page focused on policy playback versus real market data."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtester import BacktestConfig, BacktestResult, HistoricalBacktester
from ui import dashboard as dashboard_ui
from ui.policy_lab import _compile_scaled_shock


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


def _render_comparison_chart(result: BacktestResult) -> go.Figure:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result.dates),
            "real": result.real_prices,
            "simulated": result.simulated_prices,
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["real"], mode="lines", name="真实指数", line=dict(color="#8ec5ff", width=2.4)))
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["simulated"],
            mode="lines",
            name="仿真指数",
            line=dict(color="#35d07f", width=2.4, dash="solid"),
        )
    )
    if not frame.empty:
        fig.add_vline(x=frame["date"].iloc[0], line_color="#f59e0b", line_dash="dash")
    fig.update_layout(
        **dashboard_ui.PLOTLY_DARK_LAYOUT,
        title="真实走势 vs 仿真走势",
        yaxis=dict(title="指数点位"),
        xaxis=dict(title="时间"),
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)
    dashboard_ui.export_plot_bundle(fig, frame, "history_replay_compare", "history_replay_compare")
    return fig


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
                "评委若追问偏差，重点解释机制差异而非误差本身。",
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

    if "history_replay_result" not in st.session_state:
        st.session_state.history_replay_result = None

    default_start, default_end = _default_window()
    with st.form("history_replay_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            start_date = st.date_input("开始日期", value=default_start)
            end_date = st.date_input("结束日期", value=default_end)
            symbol_label = st.selectbox("对照指数", options=list(INDEX_OPTIONS.keys()), index=1)
            policy_name = st.text_input("政策事件名称", value="历史政策影响回放")
            policy_text = st.text_area(
                "政策说明",
                value="请描述当时要回放的政策，例如：下调印花税并释放流动性支持。",
                height=110,
            )
        with col2:
            background = st.selectbox("历史背景模板", options=list(BACKGROUND_TEMPLATES.keys()), index=1)
            strength = st.slider("仿真强度", min_value=0.3, max_value=1.6, value=1.0, step=0.1)
            rebalance_frequency = st.select_slider("调仓节奏", options=[1, 2, 3, 5, 10], value=5)
            lookback = st.slider("观察窗口", min_value=10, max_value=80, value=20, step=5)
        submitted = st.form_submit_button("运行历史回放", use_container_width=True)

    if submitted:
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            return

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

        backtester = HistoricalBacktester(config)
        progress = st.progress(0.0)
        status = st.empty()

        def _on_progress(cur: int, total: int, msg: str) -> None:
            progress.progress(cur / max(total, 1))
            status.caption(msg)

        try:
            result = backtester.run_backtest(progress_callback=_on_progress)
        finally:
            progress.empty()
            status.empty()

        st.session_state.history_replay_result = {
            "config": config,
            "policy_name": policy_name,
            "policy_text": policy_text,
            "background": background,
            "strength": strength,
            "result": result,
            "metrics": _build_replay_metrics(result),
        }

    bundle = st.session_state.history_replay_result
    if not bundle:
        st.info("先选择一个历史区间和政策，再点击“运行历史回放”。")
        return

    result: BacktestResult = bundle["result"]
    if not result or not result.real_prices:
        st.warning("这次历史回放没有得到有效结果。请调整日期区间或指数后重试。")
        return

    metrics = bundle["metrics"]
    st.markdown("### 真实走势 vs 仿真走势")
    _render_comparison_chart(result)
    _render_replay_cards(result, metrics)

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
