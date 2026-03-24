"""History replay page for factor backtest and agent replay."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.agent_replay import AgentReplayEngine
from core.backtester import BacktestConfig, BacktestResult, FactorBacktestEngine
from core.event_store import EventStore
from ui import dashboard as dashboard_ui
from ui.policy_lab import _compile_scaled_shock, _load_policy_templates
from ui.reporting import official_report_meta, write_report_artifacts


INDEX_OPTIONS = {
    "上海指数 (sh000001)": "sh000001",
    "沪深300 (sh000300)": "sh000300",
    "深证成指 (sz399001)": "sz399001",
    "创业板指 (sz399006)": "sz399006",
}

BACKGROUND_TEMPLATES = {
    "宽松": 0.10,
    "中性": 0.00,
    "紧缩": -0.12,
    "风险事件": -0.20,
    "政策托底": 0.16,
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
        return f"{policy_name}: path shape is close to the historical series."
    if metrics["trend_alignment"] < 0.5:
        return f"{policy_name}: direction alignment is weak, so the replay is likely over/under-reacting."
    if metrics["response_gap"] > 8:
        return f"{policy_name}: response timing is lagging the historical move."
    return f"{policy_name}: usable for policy mechanism comparison, but not a day-by-day clone."


def _compute_result_summary(result: BacktestResult) -> Dict[str, float]:
    return {
        "total_return": float(result.total_return),
        "max_drawdown": float(result.max_drawdown),
        "excess_return": float(result.excess_return),
        "price_correlation": float(result.price_correlation),
        "volatility_correlation": float(result.volatility_correlation),
        "price_rmse": float(result.price_rmse),
    }


def _select_replay_engine(config: BacktestConfig, engine_mode: str, feature_flags: Dict[str, bool]) -> tuple[Any, str, str]:
    agent_enabled = bool(feature_flags.get("agent_replay", False))
    if engine_mode == "agent" and agent_enabled:
        return AgentReplayEngine(config), "agent", ""
    if engine_mode == "agent" and not agent_enabled:
        return FactorBacktestEngine(config), "factor", "Agent replay is disabled by the feature flag; falling back to factor mode."
    return FactorBacktestEngine(config), "factor", ""


def _run_engine_with_progress(engine: Any, start_ratio: float, end_ratio: float, progress, status, label: str) -> BacktestResult:
    def _on_progress(cur: int, total: int, msg: str) -> None:
        ratio = cur / max(total, 1)
        progress.progress(start_ratio + (end_ratio - start_ratio) * ratio)
        status.caption(f"{label}: {msg}")

    return engine.run_backtest(progress_callback=_on_progress)


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
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["real"], mode="lines", name="Real", line=dict(color="#8ec5ff", width=2.4)))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["simulated"], mode="lines", name="Simulated", line=dict(color="#35d07f", width=2.4)))
    if "baseline" in frame:
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["baseline"], mode="lines", name="Baseline", line=dict(color="#f59e0b", width=2.2, dash="dash")))
    if not frame.empty:
        fig.add_vline(x=frame["date"].iloc[0], line_color="#f59e0b", line_dash="dot")
    fig.update_layout(
        **dashboard_ui.PLOTLY_DARK_LAYOUT,
        title="Real vs Simulated",
        yaxis=dict(title="Index level"),
        xaxis=dict(title="Date"),
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True, key="history_replay_compare_chart")
    dashboard_ui.export_plot_bundle(fig, frame, "history_replay_compare", "history_replay_compare")


def _render_metric_cards(result: BacktestResult, metrics: Dict[str, float]) -> None:
    cols = st.columns(5)
    cards = [
        ("Trend alignment", f"{metrics['trend_alignment']:.0%}", "Direction match"),
        ("Turning points", f"{metrics['turning_point_match']:.0%}", "Regime flips"),
        ("Drawdown gap", f"{metrics['drawdown_gap']:.2%}", "Closer is better"),
        ("Vol similarity", f"{metrics['vol_similarity']:.0%}", "Volatility regime"),
        ("Response lag", f"{metrics['response_gap']:.0f}d", "Timing offset"),
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
        ("Relative return", f"{result.total_return - baseline.total_return:+.2%}", "Against no-policy baseline"),
        ("Drawdown improvement", f"{baseline.max_drawdown - result.max_drawdown:+.2%}", "Positive is better"),
        ("Excess return", f"{result.excess_return:+.2%}", "vs benchmark"),
        ("Vol calibration lift", f"{result.volatility_correlation - baseline.volatility_correlation:+.0%}", "Vol regime fit"),
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


def _build_replay_brief(bundle: Dict[str, Any], metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    engine_mode = bundle.get("engine_mode", "factor")
    return [
        {
            "title": "Path fit",
            "summary": "Checks the line shape and timing against the real series.",
            "lines": [
                f"Mode: {engine_mode}",
                f"Trend alignment: {metrics['trend_alignment']:.0%}",
                f"Turning-point F1 proxy: {metrics['turning_point_match']:.0%}",
            ],
        },
        {
            "title": "Microstructure fit",
            "summary": "Focuses on OHLCV consistency and trade-tape behavior.",
            "lines": [
                f"Vol similarity: {metrics['vol_similarity']:.0%}",
                f"Drawdown gap: {metrics['drawdown_gap']:.2%}",
                "Agent mode uses trade-tape close for simulated prices.",
            ],
        },
        {
            "title": "Behavioral fit",
            "summary": "Explains whether the replay reacts like a policy-driven market.",
            "lines": [
                f"Response lag: {metrics['response_gap']:.0f} days",
                f"Feature flags: {bundle.get('feature_flags', {})}",
                "This is for comparison and defense, not a pixel-perfect clone.",
            ],
        },
    ]


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


def _render_agent_readout(policy_text: str, result: BacktestResult, metrics: Dict[str, float]) -> None:
    cards = [
        (
            "Policy analysis",
            "The policy text is mapped into a scalar shock that drives the replay.",
            [
                f"Policy text length: {len(policy_text)}",
                f"Trend alignment: {metrics['trend_alignment']:.0%}",
            ],
        ),
        (
            "Quant analysis",
            "The replay is judged on path fit, timing, and volatility regime.",
            [
                f"Price correlation: {result.price_correlation:.3f}",
                f"Volatility correlation: {result.volatility_correlation:.3f}",
            ],
        ),
        (
            "Risk analysis",
            "A stronger replay should avoid large drawdown and response gaps.",
            [
                f"Drawdown gap: {metrics['drawdown_gap']:.2%}",
                f"Response lag: {metrics['response_gap']:.0f}d",
            ],
        ),
        (
            "Final review",
            "Use the report to explain what fits and what still diverges.",
            [
                f"Simulated prices: {len(result.simulated_prices)}",
                f"Trade tape rows: {len(result.trade_log)}",
            ],
        ),
    ]

    cols = st.columns(4)
    for idx, card in enumerate(cards):
        items = "".join(f"<li>{line}</li>" for line in card[2])
        with cols[idx]:
            st.markdown(
                f"""
                <div class="story-card">
                  <div class="story-card-title">{card[0]}</div>
                  <div class="story-card-summary">{card[1]}</div>
                  <ul class="story-card-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_history_report(bundle: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    result: BacktestResult = bundle["result"]
    baseline: Optional[BacktestResult] = bundle.get("baseline_result")
    report_meta = official_report_meta("history_replay", bundle["policy_name"])
    payload = {
        "report_meta": report_meta,
        "policy_name": bundle["policy_name"],
        "policy_text": bundle["policy_text"],
        "background": bundle["background"],
        "strength": bundle["strength"],
        "symbol_label": bundle["symbol_label"],
        "start_date": bundle["start_date"],
        "end_date": bundle["end_date"],
        "engine_mode": bundle.get("engine_mode", "factor"),
        "feature_flags": bundle.get("feature_flags", {}),
        "metrics": metrics,
        "result_summary": _compute_result_summary(result),
        "baseline_summary": _compute_result_summary(baseline) if baseline else None,
        "replay_brief": bundle.get("replay_cards", []),
        "simulated_bars": result.simulated_bars,
    }
    markdown = [
        f"# History replay report: {bundle['policy_name']}",
        "",
        f"- Report no: {report_meta['report_no']}",
        f"- Date: {report_meta['date_cn']}",
        f"- Engine mode: {bundle.get('engine_mode', 'factor')}",
        f"- Feature flags: {bundle.get('feature_flags', {})}",
        "",
        "## Replay summary",
        f"- Symbol: {bundle['symbol_label']}",
        f"- Date window: {bundle['start_date']} to {bundle['end_date']}",
        f"- Backdrop: {bundle['background']}",
        f"- Intensity: {bundle['strength']:.1f}",
        f"- Explanation: {_build_bias_explanation(metrics, bundle['policy_name'])}",
        "",
        "## Metrics",
        f"- Trend alignment: {metrics['trend_alignment']:.0%}",
        f"- Turning points: {metrics['turning_point_match']:.0%}",
        f"- Drawdown gap: {metrics['drawdown_gap']:.2%}",
        f"- Vol similarity: {metrics['vol_similarity']:.0%}",
        f"- Response lag: {metrics['response_gap']:.0f} days",
    ]
    if baseline:
        markdown.extend(
            [
                "",
                "## Baseline comparison",
                f"- Baseline return: {baseline.total_return:.2%}",
                f"- Relative return: {result.total_return - baseline.total_return:+.2%}",
                f"- Relative drawdown: {baseline.max_drawdown - result.max_drawdown:+.2%}",
            ]
        )

    markdown.extend(
        [
            "",
            "## Risks",
            "- Replay result is intended for defense, comparison, and calibration.",
            "- Simulated prices in agent mode come from the trade-tape close, not equity scaling.",
            "- All reproducibility info is recorded in the payload metadata.",
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
              <div class="summary-label">Report generated</div>
              <div class="summary-value">{report_meta.get('title', export_bundle['stem'])}</div>
              <div class="summary-note">No: {report_meta.get('report_no', export_bundle['stem'])}</div>
              <div class="summary-note">Recipient: {report_meta.get('recipient', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        top_row = st.columns(2)
        bottom_row = st.columns(2)
        with top_row[0]:
            st.download_button(
                "Download Word",
                data=export_bundle["docx_bytes"],
                file_name=f"{export_bundle['stem']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key=f"history_docx_{export_bundle['stem']}",
            )
        with top_row[1]:
            st.download_button(
                "Download PDF",
                data=export_bundle["pdf_bytes"],
                file_name=f"{export_bundle['stem']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"history_pdf_{export_bundle['stem']}",
            )
        with bottom_row[0]:
            st.download_button(
                "Download Markdown",
                data=export_bundle["markdown_text"],
                file_name=f"{export_bundle['stem']}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"history_md_{export_bundle['stem']}",
            )
        with bottom_row[1]:
            st.download_button(
                "Download JSON",
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
          <h1>政策因子回测/历史对照回测</h1>
          <p>Factor mode stays compatible with the old UI. Agent replay is behind a feature flag and uses trade-tape closes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "history_replay_result" not in st.session_state:
        st.session_state.history_replay_result = None

    template_map = {item["title"]: item for item in _load_policy_templates()}
    selected_template_label = st.selectbox("Template", options=list(template_map.keys()), index=0)
    selected_template = template_map[selected_template_label]
    entry_mode = str(st.session_state.get("history_replay_entry_mode", "factor")).strip().lower()
    default_engine_mode = "agent" if entry_mode == "agent" else "factor"

    default_start, default_end = _default_window()
    with st.form("history_replay_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            start_date = st.date_input("Start date", value=default_start)
            end_date = st.date_input("End date", value=default_end)
            symbol_label = st.selectbox("Index", options=list(INDEX_OPTIONS.keys()), index=1)
            policy_name = st.text_input("Policy name", value=f"{selected_template['title']} history replay")
            policy_text = st.text_area("Policy text", value=str(selected_template["policy_text"]), height=110)
        with col2:
            background = st.selectbox("Backdrop", options=list(BACKGROUND_TEMPLATES.keys()), index=1)
            strength = st.slider(
                "Replay intensity",
                min_value=0.3,
                max_value=1.6,
                value=float(selected_template.get("recommended_intensity", 1.0)),
                step=0.1,
            )
            rebalance_frequency = st.select_slider("Rebalance cadence", options=[1, 2, 3, 5, 10], value=5)
            lookback = st.slider("Lookback window", min_value=10, max_value=80, value=20, step=5)
            engine_mode = st.radio(
                "Engine mode",
                options=["factor", "agent"],
                horizontal=True,
                index=1 if default_engine_mode == "agent" else 0,
            )
            enable_agent_replay = st.toggle("Enable agent replay feature flag", value=False)
            enable_event_store = st.toggle("Enable EventStore feature flag", value=False)
            enable_baseline = st.toggle("Include factor baseline", value=True)
            dataset_version = st.text_input("EventStore dataset_version", value="default")
            scenario_id = st.text_input("Scenario id (optional)", value="")
            snapshot_id = st.text_input("Snapshot id (optional)", value="")
        submitted = st.form_submit_button("Run history replay", use_container_width=True, type="primary")

    if submitted:
        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
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
            random_seed=42,
            feature_flags={
                "agent_replay": bool(enable_agent_replay),
                "event_store_v1": bool(enable_event_store),
            },
        )
        engine, resolved_mode, fallback_reason = _select_replay_engine(config, engine_mode, config.feature_flags)
        if isinstance(engine, AgentReplayEngine) and bool(enable_event_store):
            engine.configure_event_store(
                event_store=EventStore(),
                dataset_version=dataset_version,
                snapshot_id=snapshot_id,
                scenario_id=scenario_id,
            )
        if fallback_reason:
            st.warning(fallback_reason)

        baseline_result = None
        try:
            result = _run_engine_with_progress(
                engine,
                0.0,
                0.7 if enable_baseline and resolved_mode == "factor" else 1.0,
                progress,
                status,
                "factor backtest" if resolved_mode == "factor" else "agent replay",
            )
            if enable_baseline and resolved_mode == "factor":
                baseline_config = BacktestConfig(**{**config.__dict__, "policy_shock": 0.0})
                baseline_engine = FactorBacktestEngine(baseline_config)
                baseline_result = _run_engine_with_progress(
                    baseline_engine,
                    0.7,
                    1.0,
                    progress,
                    status,
                    "no-policy baseline",
                )
        finally:
            progress.empty()
            status.empty()

        bundle = {
            "config": config,
            "engine_mode": resolved_mode,
            "policy_name": policy_name,
            "policy_text": policy_text,
            "background": background,
            "strength": strength,
            "symbol_label": symbol_label,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "feature_flags": dict(config.feature_flags),
            "result": result,
            "baseline_result": baseline_result,
            "metrics": _build_replay_metrics(result),
        }
        bundle["replay_cards"] = _build_replay_brief(bundle, bundle["metrics"])
        bundle["export_bundle"] = _build_history_report(bundle, bundle["metrics"])
        st.session_state.history_replay_result = bundle

    bundle = st.session_state.history_replay_result
    if not bundle:
        st.info("Choose a time window and replay mode, then run the replay.")
        return

    result: BacktestResult = bundle["result"]
    if not result or not result.real_prices:
        st.warning("No usable replay result was produced. Adjust the date window and try again.")
        return

    metrics = bundle["metrics"]
    baseline = bundle.get("baseline_result")
    st.markdown(f"### {bundle.get('engine_mode', 'factor').title()} comparison")
    _render_comparison_chart(result, baseline if bundle.get("engine_mode") == "factor" else None)
    _render_metric_cards(result, metrics)
    _render_baseline_delta_cards(result, baseline if bundle.get("engine_mode") == "factor" else None)

    st.markdown("### Replay brief")
    _render_replay_brief(bundle["replay_cards"])

    st.markdown("### Bias explanation")
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-value">{_build_bias_explanation(metrics, bundle['policy_name'])}</div>
          <div class="summary-note">Agent replay uses the trade-tape close as the simulated price source.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Model readout")
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-label">Mode</div>
          <div class="summary-value">{bundle.get('engine_mode', 'factor')}</div>
          <div class="summary-note">Config hash: {result.metadata.get('config_hash', '')}</div>
          <div class="summary-note">Snapshot: {result.metadata.get('data_snapshot', {}).get('snapshot_id', '')}</div>
          <div class="summary-note">Price source: {result.metadata.get('simulated_price_source', 'equity_curve_scaled')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_agent_readout(bundle["policy_text"], result, metrics)

    st.markdown("### Report export")
    _render_report_export(bundle["export_bundle"])
