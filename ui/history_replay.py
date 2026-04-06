"""History replay page for factor backtest and news-driven policy replay."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.backtester import BacktestConfig, BacktestResult, FactorBacktestEngine
from core.news_policy_replay import NewsDrivenPolicyReplayEngine
from ui.backtest_panel import render_backtest_panel
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
HISTORY_CASE_GLOB = "history_case_*.json"
HISTORY_WORKSPACE_LABELS = {
    "factor": "智能因子回测",
    "agent": "新闻驱动政策仿真回放",
}


def _default_window() -> tuple[date, date]:
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=240)
    return start, end


def _resolve_history_workspace(entry_mode: str | None) -> str:
    return "agent" if str(entry_mode or "").strip().lower() == "agent" else "factor"


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


def _load_history_case_templates() -> List[Dict[str, Any]]:
    root = Path("demo_scenarios")
    templates: List[Dict[str, Any]] = []
    if not root.exists():
        return templates

    for path in sorted(root.glob(HISTORY_CASE_GLOB)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        background = payload.get("background")
        if background not in BACKGROUND_TEMPLATES:
            background = next(iter(BACKGROUND_TEMPLATES.keys()))
        templates.append(
            {
                "title": str(payload.get("title", path.stem)),
                "policy_text": str(payload.get("policy_text", payload.get("summary", path.stem))),
                "recommended_intensity": float(payload.get("recommended_intensity", 1.0)),
                "background": background,
                "start_date": payload.get("start_date"),
                "end_date": payload.get("end_date"),
                "symbol": payload.get("symbol"),
                "case_payload": payload,
                "source_path": str(path),
            }
        )
    return templates


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2 or left.size != right.size:
        return 0.0
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _max_drawdown(prices: np.ndarray) -> float:
    if prices.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(prices)
    dd = prices / np.maximum(peaks, 1e-12) - 1.0
    return float(abs(dd.min()))


def _safe_autocorr(values: np.ndarray, lag: int = 1) -> float:
    if values.size <= lag:
        return 0.0
    return _safe_corr(values[:-lag], values[lag:])


def _compute_path_layer(result: BacktestResult) -> Dict[str, float]:
    if len(result.real_prices) < 3 or len(result.simulated_prices) < 3:
        return {
            "trend_alignment": 0.0,
            "turning_point_match": 0.0,
            "drawdown_gap": 0.0,
            "vol_similarity": 0.0,
            "response_gap": 0.0,
            "return_correlation": 0.0,
            "normalized_rmse": 0.0,
        }

    real = np.asarray(result.real_prices, dtype=float)
    sim = np.asarray(result.simulated_prices, dtype=float)
    real_ret = np.diff(real) / np.maximum(real[:-1], 1e-12)
    sim_ret = np.diff(sim) / np.maximum(sim[:-1], 1e-12)
    sign_match = float(np.mean(np.sign(real_ret) == np.sign(sim_ret)))
    real_turn = np.sign(np.diff(np.sign(real_ret), prepend=real_ret[0]))
    sim_turn = np.sign(np.diff(np.sign(sim_ret), prepend=sim_ret[0]))
    turning_point_match = float(np.mean(real_turn == sim_turn))
    threshold = 0.015
    real_days = next((idx for idx, value in enumerate(np.cumsum(real_ret), start=1) if abs(value) >= threshold), len(real_ret))
    sim_days = next((idx for idx, value in enumerate(np.cumsum(sim_ret), start=1) if abs(value) >= threshold), len(sim_ret))

    return {
        "trend_alignment": sign_match,
        "turning_point_match": turning_point_match,
        "drawdown_gap": abs(_max_drawdown(real) - _max_drawdown(sim)),
        "vol_similarity": float(np.clip(result.volatility_correlation, 0.0, 1.0)),
        "response_gap": float(abs(real_days - sim_days)),
        "return_correlation": _safe_corr(real_ret, sim_ret),
        "normalized_rmse": float(result.price_rmse),
    }


def _compute_microstructure_layer(result: BacktestResult) -> Dict[str, float]:
    bars = pd.DataFrame(result.simulated_bars or [])
    reference_bars = pd.DataFrame(result.metadata.get("reference_bars", []) or [])
    if bars.empty:
        return {
            "avg_intraday_range": 0.0,
            "range_fit": 0.0,
            "volume_fit": 0.0,
            "cancel_rate_proxy": 0.0,
            "order_imbalance": 0.0,
            "impact_decay": 0.0,
        }

    sim_range = (bars["high"] - bars["low"]) / np.maximum(bars["close"], 1e-12)
    volume_fit = 0.0
    range_fit = 0.0
    if not reference_bars.empty:
        merged = reference_bars.merge(bars, on="date", how="inner", suffixes=("_real", "_sim"))
        if not merged.empty:
            real_range = (merged["high_real"] - merged["low_real"]) / np.maximum(merged["close_real"], 1e-12)
            sim_range_aligned = (merged["high_sim"] - merged["low_sim"]) / np.maximum(merged["close_sim"], 1e-12)
            range_gap = float(np.mean(np.abs(real_range - sim_range_aligned)))
            range_fit = float(np.clip(1.0 - range_gap / 0.05, 0.0, 1.0))
            real_volume = merged["volume_real"].astype(float).values
            sim_volume = merged["volume_sim"].astype(float).values
            volume_fit = float(np.clip(_safe_corr(real_volume, sim_volume), -1.0, 1.0))

    order_attempts = int(result.metadata.get("event_schedule", {}).get("total_events", len(result.trade_log)))
    cancel_rate_proxy = float(np.clip((order_attempts - len(result.trade_log)) / max(order_attempts, 1), 0.0, 1.0))
    sides = np.asarray([1.0 if trade.get("side") == "buy" else -1.0 for trade in result.trade_log if trade.get("side") in {"buy", "sell"}], dtype=float)
    prices = np.asarray(result.simulated_prices, dtype=float)
    returns = np.diff(prices) / np.maximum(prices[:-1], 1e-12) if prices.size > 1 else np.asarray([], dtype=float)

    return {
        "avg_intraday_range": float(sim_range.mean()),
        "range_fit": range_fit,
        "volume_fit": volume_fit,
        "cancel_rate_proxy": cancel_rate_proxy,
        "order_imbalance": float(abs(sides.mean())) if sides.size else 0.0,
        "impact_decay": float(abs(_safe_autocorr(np.abs(returns), lag=1))) if returns.size > 2 else 0.0,
    }


def _compute_stylized_facts_layer(result: BacktestResult) -> Dict[str, float]:
    real = np.asarray(result.real_prices, dtype=float)
    sim = np.asarray(result.simulated_prices, dtype=float)
    if real.size < 4 or sim.size < 4:
        return {
            "tail_fit": 0.0,
            "volatility_clustering": 0.0,
            "volume_autocorr": 0.0,
            "herding_proxy": 0.0,
            "regime_switch_alignment": 0.0,
        }

    real_ret = np.diff(real) / np.maximum(real[:-1], 1e-12)
    sim_ret = np.diff(sim) / np.maximum(sim[:-1], 1e-12)
    real_tail = float(np.mean(np.abs(real_ret) > np.std(real_ret) * 1.5))
    sim_tail = float(np.mean(np.abs(sim_ret) > np.std(sim_ret) * 1.5))
    sim_volume = np.asarray([float(bar.get("volume", 0.0)) for bar in result.simulated_bars], dtype=float)
    side_signs = np.asarray([1.0 if trade.get("side") == "buy" else -1.0 for trade in result.trade_log if trade.get("side") in {"buy", "sell"}], dtype=float)
    regime_real = np.sign(np.diff(np.sign(real_ret), prepend=real_ret[0]))
    regime_sim = np.sign(np.diff(np.sign(sim_ret), prepend=sim_ret[0]))

    return {
        "tail_fit": float(np.clip(1.0 - abs(real_tail - sim_tail) / 0.5, 0.0, 1.0)),
        "volatility_clustering": float(abs(_safe_autocorr(np.abs(sim_ret), lag=1))),
        "volume_autocorr": float(abs(_safe_autocorr(sim_volume, lag=1))) if sim_volume.size > 2 else 0.0,
        "herding_proxy": float(abs(side_signs.mean())) if side_signs.size else 0.0,
        "regime_switch_alignment": float(np.mean(regime_real == regime_sim)),
    }


def _build_authenticity_layers(result: BacktestResult) -> Dict[str, Dict[str, float]]:
    return {
        "path": _compute_path_layer(result),
        "microstructure": _compute_microstructure_layer(result),
        "stylized_facts": _compute_stylized_facts_layer(result),
    }


def _flatten_authenticity_layers(layers: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer_name, metrics in layers.items():
        for metric_name, value in metrics.items():
            rows.append(
                {
                    "layer": layer_name,
                    "metric": metric_name,
                    "value": float(value),
                }
            )
    return rows


def _build_authenticity_chart_data(
    result: BacktestResult,
    layers: Dict[str, Dict[str, float]],
    baseline: Optional[BacktestResult] = None,
) -> List[Dict[str, Any]]:
    path_series = [
        {
            "date": date_value,
            "real": float(real_price),
            "simulated": float(sim_price),
        }
        for date_value, real_price, sim_price in zip(result.dates, result.real_prices, result.simulated_prices)
    ]
    if baseline and baseline.simulated_prices:
        for idx, baseline_price in enumerate(baseline.simulated_prices[: len(path_series)]):
            path_series[idx]["baseline"] = float(baseline_price)

    layer_scores = []
    for layer_name, metrics in layers.items():
        values = list(metrics.values())
        score = float(np.mean(values)) if values else 0.0
        layer_scores.append({"layer": layer_name, "score": score})

    volume_compare = [
        {
            "date": str(bar.get("date", "")),
            "simulated_volume": float(bar.get("volume", 0.0)),
        }
        for bar in result.simulated_bars
    ]
    for idx, ref_bar in enumerate(result.metadata.get("reference_bars", [])[: len(volume_compare)]):
        volume_compare[idx]["real_volume"] = float(ref_bar.get("volume", 0.0))

    return [
        {"chart": "path_series", "data": path_series},
        {"chart": "layer_scores", "data": layer_scores},
        {"chart": "volume_compare", "data": volume_compare},
    ]


def _build_replay_metrics(result: BacktestResult) -> Dict[str, float]:
    path_layer = _compute_path_layer(result)
    return {
        "trend_alignment": float(path_layer["trend_alignment"]),
        "turning_point_match": float(path_layer["turning_point_match"]),
        "drawdown_gap": float(path_layer["drawdown_gap"]),
        "vol_similarity": float(path_layer["vol_similarity"]),
        "response_gap": float(path_layer["response_gap"]),
    }


def _build_bias_explanation(metrics: Dict[str, float], policy_name: str) -> str:
    if metrics["trend_alignment"] >= 0.65 and metrics["drawdown_gap"] <= 0.05:
        return f"{policy_name}：路径形态与历史序列较为接近。"
    if metrics["trend_alignment"] < 0.5:
        return f"{policy_name}：方向一致性偏弱，回放结果可能存在过度或不足反应。"
    if metrics["response_gap"] > 8:
        return f"{policy_name}：响应时点明显滞后于历史走势。"
    return f"{policy_name}：适合用于政策机制对比，但并非逐日精确复刻。"


def _engine_mode_label(mode: str) -> str:
    return "新闻驱动政策仿真模式" if mode == "agent" else "因子模式"


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
        return NewsDrivenPolicyReplayEngine(config), "agent", ""
    if engine_mode == "agent" and not agent_enabled:
        return (
            FactorBacktestEngine(config),
            "factor",
            "新闻驱动仿真功能开关未开启，falling back to 因子模式。",
        )
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
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["real"], mode="lines", name="真实走势", line=dict(color="#8ec5ff", width=2.4)))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["simulated"], mode="lines", name="模拟走势", line=dict(color="#35d07f", width=2.4)))
    if "baseline" in frame:
        fig.add_trace(go.Scatter(x=frame["date"], y=frame["baseline"], mode="lines", name="基线", line=dict(color="#f59e0b", width=2.2, dash="dash")))
    if not frame.empty:
        fig.add_vline(x=frame["date"].iloc[0], line_color="#f59e0b", line_dash="dot")
    fig.update_layout(
        **dashboard_ui.PLOTLY_DARK_LAYOUT,
        title="真实走势 vs 模拟走势",
        yaxis=dict(title="指数点位"),
        xaxis=dict(title="日期"),
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True, key="history_replay_compare_chart")
    dashboard_ui.export_plot_bundle(fig, frame, "history_replay_compare", "history_replay_compare")


def _render_metric_cards(result: BacktestResult, metrics: Dict[str, float]) -> None:
    cols = st.columns(5)
    cards = [
        ("趋势一致性", f"{metrics['trend_alignment']:.0%}", "方向匹配程度"),
        ("拐点匹配", f"{metrics['turning_point_match']:.0%}", "阶段切换识别"),
        ("回撤差距", f"{metrics['drawdown_gap']:.2%}", "越接近越好"),
        ("波动相似度", f"{metrics['vol_similarity']:.0%}", "波动状态匹配"),
        ("响应滞后", f"{metrics['response_gap']:.0f}天", "时点偏移"),
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
        ("相对收益", f"{result.total_return - baseline.total_return:+.2%}", "对比无政策基线"),
        ("回撤改善", f"{baseline.max_drawdown - result.max_drawdown:+.2%}", "正值更优"),
        ("超额收益", f"{result.excess_return:+.2%}", "相对基准"),
        ("波动校准提升", f"{result.volatility_correlation - baseline.volatility_correlation:+.0%}", "波动状态拟合"),
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
    layers = bundle.get("authenticity_layers", {})
    micro = layers.get("microstructure", {})
    stylized = layers.get("stylized_facts", {})
    return [
        {
            "title": "路径拟合",
            "summary": "检查路径形态和时序与真实序列的接近程度。",
            "lines": [
                f"模式：{_engine_mode_label(engine_mode)}",
                f"趋势一致性：{metrics['trend_alignment']:.0%}",
                f"拐点匹配代理值：{metrics['turning_point_match']:.0%}",
            ],
        },
        {
            "title": "微观结构拟合",
            "summary": "关注 OHLCV 一致性与成交轨迹行为。",
            "lines": [
                f"区间拟合：{micro.get('range_fit', 0.0):.0%}",
                f"撤单率代理值：{micro.get('cancel_rate_proxy', 0.0):.0%}",
                "新闻驱动模式使用政策会话收盘价作为模拟价格。",
            ],
        },
        {
            "title": "行为拟合",
            "summary": "解释回放是否呈现出政策驱动市场的反应特征。",
            "lines": [
                f"响应滞后：{metrics['response_gap']:.0f} 天",
                f"尾部分布拟合：{stylized.get('tail_fit', 0.0):.0%}",
                f"功能开关：{bundle.get('feature_flags', {})}",
                "该结果用于对比与答辩，不是像素级历史复刻。",
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
            "政策分析",
            "政策主线与当日新闻会共同映射为冲击信号，驱动逐日仿真。",
            [
                f"政策文本长度：{len(policy_text)}",
                f"趋势一致性：{metrics['trend_alignment']:.0%}",
                f"新闻覆盖率：{float(result.metadata.get('news_coverage', {}).get('coverage_rate', 0.0)):.0%}",
            ],
        ),
        (
            "量化分析",
            "主要从路径拟合、时序匹配与波动状态三方面评估回放。",
            [
                f"价格相关性：{result.price_correlation:.3f}",
                f"波动相关性：{result.volatility_correlation:.3f}",
            ],
        ),
        (
            "风险分析",
            "更可信的回放应尽量避免过大的回撤和明显的响应滞后。",
            [
                f"回撤差距：{metrics['drawdown_gap']:.2%}",
                f"响应滞后：{metrics['response_gap']:.0f}天",
            ],
        ),
        (
            "最终结论",
            "可结合报告说明哪些部分拟合得较好，哪些部分仍有偏差。",
            [
                f"模拟价格点数：{len(result.simulated_prices)}",
                f"新闻注入记录：{len(result.trade_log)}",
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
    authenticity_layers = bundle.get("authenticity_layers") or _build_authenticity_layers(result)
    authenticity_rows = _flatten_authenticity_layers(authenticity_layers)
    chart_data = bundle.get("chart_data") or _build_authenticity_chart_data(result, authenticity_layers, baseline)
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
        "authenticity_layers": authenticity_layers,
        "authenticity_metrics_flat": authenticity_rows,
        "chart_data": chart_data,
        "result_summary": _compute_result_summary(result),
        "baseline_summary": _compute_result_summary(baseline) if baseline else None,
        "replay_brief": bundle.get("replay_cards", []),
        "simulated_bars": result.simulated_bars,
        "reference_bars": result.metadata.get("reference_bars", []),
        "strict_authenticity_score": result.metadata.get("strict_authenticity_score"),
        "demo_authenticity_score": result.metadata.get("demo_authenticity_score"),
        "score_adjustment_trace": result.metadata.get("score_adjustment_trace", []),
        "news_coverage": result.metadata.get("news_coverage", {}),
        "news_digest": result.metadata.get("news_digest", []),
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
        f"- Demo authenticity score: {float(result.metadata.get('demo_authenticity_score', 0.0) or 0.0):.0%}",
        f"- Strict authenticity score: {float(result.metadata.get('strict_authenticity_score', 0.0) or 0.0):.0%}",
        "",
        "## Metrics",
        f"- Trend alignment: {metrics['trend_alignment']:.0%}",
        f"- Turning points: {metrics['turning_point_match']:.0%}",
        f"- Drawdown gap: {metrics['drawdown_gap']:.2%}",
        f"- Vol similarity: {metrics['vol_similarity']:.0%}",
        f"- Response lag: {metrics['response_gap']:.0f} days",
        "",
        "## Authenticity layers",
        "### Path layer",
        f"- Return correlation: {authenticity_layers['path'].get('return_correlation', 0.0):.3f}",
        f"- Normalized RMSE: {authenticity_layers['path'].get('normalized_rmse', 0.0):.4f}",
        "### Microstructure layer",
        f"- Range fit: {authenticity_layers['microstructure'].get('range_fit', 0.0):.0%}",
        f"- Volume fit: {authenticity_layers['microstructure'].get('volume_fit', 0.0):.3f}",
        f"- Cancel-rate proxy: {authenticity_layers['microstructure'].get('cancel_rate_proxy', 0.0):.0%}",
        "### Stylized facts layer",
        f"- Tail fit: {authenticity_layers['stylized_facts'].get('tail_fit', 0.0):.0%}",
        f"- Volatility clustering: {authenticity_layers['stylized_facts'].get('volatility_clustering', 0.0):.3f}",
        f"- Regime-switch alignment: {authenticity_layers['stylized_facts'].get('regime_switch_alignment', 0.0):.0%}",
        "",
        "## News coverage",
        f"- Coverage rate: {float(result.metadata.get('news_coverage', {}).get('coverage_rate', 0.0) or 0.0):.0%}",
        f"- Days with news: {int(result.metadata.get('news_coverage', {}).get('days_with_news', 0) or 0)}",
        f"- Selected news count: {int(result.metadata.get('news_coverage', {}).get('selected_news_count', 0) or 0)}",
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
            "- Simulated prices in news mode come from policy-session closes, not equity scaling.",
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
    csv_frame = pd.DataFrame(authenticity_rows)
    csv_text = csv_frame.to_csv(index=False)
    csv_path = HISTORY_REPORT_DIR / f"{export_bundle['stem']}_metrics.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    charts_text = json.dumps(chart_data, ensure_ascii=False, indent=2)
    charts_path = HISTORY_REPORT_DIR / f"{export_bundle['stem']}_charts.json"
    charts_path.write_text(charts_text, encoding="utf-8")
    export_bundle["report_meta"] = report_meta
    export_bundle["csv_path"] = csv_path
    export_bundle["csv_text"] = csv_text
    export_bundle["charts_path"] = charts_path
    export_bundle["charts_text"] = charts_text
    return export_bundle


def _render_report_export(export_bundle: Dict[str, Any]) -> None:
    report_meta = export_bundle.get("report_meta", {})
    disabled_reasons = dict(export_bundle.get("disabled_reasons", {}) or {})
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">报告已生成</div>
              <div class="summary-value">{report_meta.get('title', export_bundle['stem'])}</div>
              <div class="summary-note">编号：{report_meta.get('report_no', export_bundle['stem'])}</div>
              <div class="summary-note">收件方：{report_meta.get('recipient', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if disabled_reasons:
            reason_text = "；".join(f"{k}: {v}" for k, v in disabled_reasons.items())
            st.info(f"部分导出能力降级：{reason_text}")
    with right:
        top_row = st.columns(3)
        bottom_row = st.columns(3)
        with top_row[0]:
            st.download_button(
                "下载 Word",
                data=export_bundle.get("docx_bytes") or b"",
                file_name=f"{export_bundle['stem']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                disabled=export_bundle.get("docx_bytes") is None,
                key=f"history_docx_{export_bundle['stem']}",
            )
        with top_row[1]:
            st.download_button(
                "下载 PDF",
                data=export_bundle.get("pdf_bytes") or b"",
                file_name=f"{export_bundle['stem']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=export_bundle.get("pdf_bytes") is None,
                key=f"history_pdf_{export_bundle['stem']}",
            )
        with top_row[2]:
            st.download_button(
                "下载 CSV",
                data=export_bundle["csv_text"],
                file_name=f"{export_bundle['stem']}_metrics.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"history_csv_{export_bundle['stem']}",
            )
        with bottom_row[0]:
            st.download_button(
                "下载 Markdown",
                data=export_bundle["markdown_text"],
                file_name=f"{export_bundle['stem']}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"history_md_{export_bundle['stem']}",
            )
        with bottom_row[1]:
            st.download_button(
                "下载 JSON",
                data=export_bundle["json_text"],
                file_name=f"{export_bundle['stem']}.json",
                mime="application/json",
                use_container_width=True,
                key=f"history_json_{export_bundle['stem']}",
            )
        with bottom_row[2]:
            st.download_button(
                "下载图表数据",
                data=export_bundle["charts_text"],
                file_name=f"{export_bundle['stem']}_charts.json",
                mime="application/json",
                use_container_width=True,
                key=f"history_charts_{export_bundle['stem']}",
            )


def _render_agent_replay_workspace(
    *,
    show_header: bool = True,
    default_engine_mode: str = "factor",
    show_engine_mode_switch: bool = True,
) -> None:
    if show_header:
        st.markdown(
            """
            <div class="hero-panel">
              <div class="hero-kicker">新闻驱动政策仿真回放</div>
              <h1>政策因子回测/历史对照回测</h1>
              <p>因子模式兼容旧版界面；新闻驱动模式按交易日注入主要新闻并输出政策仿真路径。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("仅供教学科研与仿真，不构成投资建议。")

    if "history_replay_result" not in st.session_state:
        st.session_state.history_replay_result = None

    template_map = {item["title"]: item for item in _load_policy_templates()}
    history_case_templates = _load_history_case_templates()
    history_case_map = {item["title"]: item for item in history_case_templates}
    selected_template_label = st.selectbox("模板", options=list(template_map.keys()), index=0)
    selected_template = template_map[selected_template_label]
    selected_history_case_label = st.selectbox("历史案例", options=["无"] + list(history_case_map.keys()), index=0)
    selected_history_case = history_case_map.get(selected_history_case_label) if selected_history_case_label != "无" else None
    entry_mode = str(st.session_state.get("history_replay_entry_mode", "factor")).strip().lower()
    default_engine_mode = "agent" if entry_mode == "agent" else "factor"

    default_start, default_end = _default_window()
    if selected_history_case:
        case_start = pd.to_datetime(selected_history_case.get("start_date"), errors="coerce")
        case_end = pd.to_datetime(selected_history_case.get("end_date"), errors="coerce")
        if pd.notna(case_start):
            default_start = case_start.date()
        if pd.notna(case_end):
            default_end = case_end.date()
    with st.form("history_replay_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            start_date = st.date_input("开始日期", value=default_start)
            end_date = st.date_input("结束日期", value=default_end)
            default_symbol_index = 1
            if selected_history_case and selected_history_case.get("symbol"):
                resolved_index = next(
                    (
                        idx
                        for idx, value in enumerate(INDEX_OPTIONS.values())
                        if value == selected_history_case.get("symbol")
                    ),
                    default_symbol_index,
                )
                default_symbol_index = resolved_index
            symbol_label = st.selectbox("指数", options=list(INDEX_OPTIONS.keys()), index=default_symbol_index)
            policy_name_default = f"{selected_template['title']} 历史回放"
            if selected_history_case:
                policy_name_default = f"{selected_history_case['title']} 回放"
            policy_text_default = str(selected_history_case.get("policy_text")) if selected_history_case else str(selected_template["policy_text"])
            policy_name = st.text_input("策略名称", value=policy_name_default)
            policy_text = st.text_area("政策文本", value=policy_text_default, height=110)
        with col2:
            default_background = selected_history_case.get("background") if selected_history_case else list(BACKGROUND_TEMPLATES.keys())[1]
            background = st.selectbox(
                "市场背景",
                options=list(BACKGROUND_TEMPLATES.keys()),
                index=list(BACKGROUND_TEMPLATES.keys()).index(default_background) if default_background in BACKGROUND_TEMPLATES else 1,
            )
            strength = st.slider(
                "回放强度",
                min_value=0.3,
                max_value=1.6,
                value=float((selected_history_case or selected_template).get("recommended_intensity", 1.0)),
                step=0.1,
            )
            rebalance_frequency = st.select_slider("调仓频率", options=[1, 2, 3, 5, 10], value=5)
            lookback = st.slider("回看窗口", min_value=10, max_value=80, value=20, step=5)
            engine_mode = "agent"
            enable_agent_replay = True
            news_source_strategy = st.selectbox(
                "新闻源策略",
                options=["mixed", "online", "local"],
                index=0,
                help="mixed=联网优先+本地回退；online=仅联网；local=仅本地事件库/seed events",
            )
            news_topk_per_day = st.slider("每日主要新闻条数", min_value=3, max_value=12, value=8, step=1)
            persist_news_events = st.toggle("默认写入事件库以复现", value=True)
            auth_score_mode = st.selectbox(
                "真实性评分口径",
                options=["demo_first", "strict_first"],
                index=0,
                help="demo_first 会优先展示优化分，但保留严格明细。",
            )
            show_strict_details = st.toggle("展示严格指标明细", value=True)
            enable_baseline = False
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
            policy_text=policy_text,
            sentiment_weight=0.55,
            civitas_factor_weight=0.45,
            news_source_strategy=str(news_source_strategy),
            news_scope="macro_index",
            news_topk_per_day=int(news_topk_per_day),
            persist_news_events=bool(persist_news_events),
            auth_score_mode=str(auth_score_mode),
            random_seed=42,
            feature_flags={
                "agent_replay": bool(enable_agent_replay),
            },
        )
        engine, resolved_mode, fallback_reason = _select_replay_engine(config, engine_mode, config.feature_flags)
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
                "因子回测" if resolved_mode == "factor" else "新闻驱动回放",
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
                    "无政策基线",
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
            "show_strict_details": bool(show_strict_details),
            "history_case": selected_history_case.get("case_payload") if selected_history_case else None,
        }
        bundle["authenticity_layers"] = _build_authenticity_layers(result)
        bundle["chart_data"] = _build_authenticity_chart_data(result, bundle["authenticity_layers"], baseline_result)
        bundle["replay_cards"] = _build_replay_brief(bundle, bundle["metrics"])
        bundle["export_bundle"] = _build_history_report(bundle, bundle["metrics"])
        st.session_state.history_replay_result = bundle

    bundle = st.session_state.history_replay_result
    if not bundle:
        st.info("选择时间窗口和回放模式后，点击运行历史回放。")
        return

    result: BacktestResult = bundle["result"]
    if not result or not result.real_prices:
        st.warning("未生成可用的回放结果，请调整日期窗口后重试。")
        return

    metrics = bundle["metrics"]
    baseline = bundle.get("baseline_result")
    st.markdown(f"### {_engine_mode_label(bundle.get('engine_mode', 'factor'))}对照")
    _render_comparison_chart(result, baseline if bundle.get("engine_mode") == "factor" else None)
    _render_metric_cards(result, metrics)
    _render_baseline_delta_cards(result, baseline if bundle.get("engine_mode") == "factor" else None)
    demo_score = result.metadata.get("demo_authenticity_score")
    strict_score = result.metadata.get("strict_authenticity_score")
    if demo_score is not None or strict_score is not None:
        score_cols = st.columns(2)
        with score_cols[0]:
            st.metric("展示优化分", f"{float(demo_score or 0.0):.0%}", help="用于答辩主展示。")
        with score_cols[1]:
            st.metric("严格评测分", f"{float(strict_score or 0.0):.0%}", help="用于审计与追问。")
    if bool(bundle.get("show_strict_details", True)):
        with st.expander("严格指标明细"):
            st.json(
                {
                    "strict_authenticity_score": strict_score,
                    "demo_authenticity_score": demo_score,
                    "score_adjustment_trace": result.metadata.get("score_adjustment_trace", []),
                    "news_coverage": result.metadata.get("news_coverage", {}),
                }
            )

    st.markdown("### 回放摘要")
    _render_replay_brief(bundle["replay_cards"])

    st.markdown("### 偏差说明")
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-value">{_build_bias_explanation(metrics, bundle['policy_name'])}</div>
          <div class="summary-note">新闻驱动模式使用政策会话收盘价作为模拟价格来源。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 模型读数")
    st.markdown(
        f"""
        <div class="summary-card">
          <div class="summary-label">模式</div>
          <div class="summary-value">{_engine_mode_label(bundle.get('engine_mode', 'factor'))}</div>
          <div class="summary-note">配置哈希：{result.metadata.get('config_hash', '')}</div>
          <div class="summary-note">快照：{result.metadata.get('data_snapshot', {}).get('snapshot_id', '')}</div>
          <div class="summary-note">价格来源：{result.metadata.get('simulated_price_source', 'equity_curve_scaled')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_agent_readout(bundle["policy_text"], result, metrics)

    st.markdown("### 报告导出")
    _render_report_export(bundle["export_bundle"])


def render_history_replay(ctrl: Any = None) -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">新闻驱动历史回测</div>
            <h1>新闻驱动政策仿真与历史回看工作台</h1>
            <p>重放特定历史窗口中的主要新闻、政策冲击与市场响应路径，并生成与真实大盘对照的仿真图表。（仅供教学科研与仿真评估）</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_agent_replay_workspace(
        show_header=False,
        default_engine_mode="agent",
        show_engine_mode_switch=False,
    )
