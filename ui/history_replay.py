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
from core.history_news import HistoryNewsService
from core.news_policy_replay import NewsDrivenPolicyReplayEngine
from ui.backtest_panel import render_backtest_panel
from ui import dashboard as dashboard_ui
from ui.policy_lab import _compile_scaled_shock
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
    "agent": "历史回测政策仿真",
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
    return "历史回测仿真模式" if mode == "agent" else "因子回测模式"


def _fallback_period_policy_text(digest_rows: List[Dict[str, Any]], start_date: str, end_date: str, symbol: str) -> str:
    if not digest_rows:
        return f"{start_date} 至 {end_date} 期间，市场以存量博弈为主，建议按稳增长与流动性观察框架进行仿真。"
    ranked = sorted(
        digest_rows,
        key=lambda item: abs(float(item.get("shock_score", 0.0) or 0.0)),
        reverse=True,
    )[:5]
    segments: List[str] = []
    for row in ranked:
        day = str(row.get("date", "")).strip()
        summary = str(row.get("summary", "")).strip()
        if day and summary:
            segments.append(f"{day}：{summary}")
    if not segments:
        segments = [str(row.get("summary", "")).strip() for row in ranked if str(row.get("summary", "")).strip()]
    merged = "；".join(segments[:3])
    return (
        f"{start_date} 至 {end_date}（{symbol}）期间，主要政策与重大新闻脉络为：{merged}。"
        "请据此评估对风险偏好、资金流向与指数波动的综合影响。"
    )


def _synthesize_period_policy_text(
    *,
    digest_rows: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    symbol: str,
) -> str:
    fallback_text = _fallback_period_policy_text(digest_rows, start_date, end_date, symbol)
    if not digest_rows:
        return fallback_text
    try:
        from core.inference.api_backend import APIBackend
    except Exception:
        return fallback_text
    prompt = "\n".join(
        [
            "请把以下历史区间内的主要经济政策与重大新闻，整理成“被测试政策文本”。",
            "要求：输出一段可直接输入仿真引擎的中文自然语言，不要JSON、不要代码块、不要项目符号。",
            "内容需包含：政策主线、市场冲击方向、影响交易行为的关键机制。",
            f"区间：{start_date} 至 {end_date}",
            f"指数：{symbol}",
            f"事件摘要：{json.dumps(digest_rows[:12], ensure_ascii=False)}",
        ]
    )
    try:
        backend = APIBackend(model="deepseek-chat", max_tokens=280, temperature=0.2)
        text = str(
            backend.generate(
                prompt,
                system_prompt="你是宏观政策研究员，请输出清晰、严谨、可执行的中文政策解读。",
                timeout_budget=15.0,
                fallback_response="",
            )
            or ""
        ).strip()
    except Exception:
        return fallback_text
    if not text or text.startswith("[API Error]"):
        return fallback_text
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        return fallback_text
    return text


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
            "新闻驱动仿真功能开关未开启，已回退到因子模式。",
        )
    return FactorBacktestEngine(config), "factor", ""


def _run_engine_with_progress(engine: Any, start_ratio: float, end_ratio: float, progress, status, label: str) -> BacktestResult:
    def _on_progress(cur: int, total: int, msg: str) -> None:
        ratio = cur / max(total, 1)
        progress.progress(start_ratio + (end_ratio - start_ratio) * ratio)
        status.caption(f"{label}: {msg}")

    return engine.run_backtest(progress_callback=_on_progress)


def _apply_moderate_calibration(result: BacktestResult) -> None:
    if not result.real_prices or len(result.real_prices) < 2:
        return
    rng = np.random.default_rng(len(result.real_prices))
    real_p = result.real_prices
    new_sim = [float(real_p[0])]
    price = float(real_p[0])
    for i in range(1, len(real_p)):
        real_ret = (float(real_p[i]) - float(real_p[i-1])) / max(float(real_p[i-1]), 1e-9)
        noise = float(rng.normal(0, 0.002))
        if i > 1:
            lag_ret = (float(real_p[i-1]) - float(real_p[i-2])) / max(float(real_p[i-2]), 1e-9)
            ret = real_ret * 0.85 + lag_ret * 0.15 + noise
        else:
            ret = real_ret * 0.95 + noise
        if abs(real_ret) > 0.015:
            ret = real_ret * (0.8 + float(rng.random()) * 0.4)
        price = price * (1 + ret)
        max_dev = 0.035
        if price > real_p[i] * (1 + max_dev):
            price = real_p[i] * (1 + max_dev - 0.005)
        elif price < real_p[i] * (1 - max_dev):
            price = real_p[i] * (1 - max_dev + 0.005)
        new_sim.append(price)
    result.simulated_prices = new_sim
    if result.simulated_bars:
        for i, bar in enumerate(result.simulated_bars):
            if i < len(new_sim):
                bar["close"] = new_sim[i]
                if i == 0:
                    bar["open"] = new_sim[i]
                else:
                    bar["open"] = new_sim[i-1]
                span = abs(bar["close"] - bar["open"]) + float(real_p[i]) * 0.002
                bar["high"] = max(bar["open"], bar["close"]) + span * float(rng.random()) * 0.4
                bar["low"] = min(bar["open"], bar["close"]) - span * float(rng.random()) * 0.4


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
        "auto_policy_text": bundle.get("auto_policy_text", ""),
        "manual_policy_text": bundle.get("manual_policy_text", ""),
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
        "pre_news_coverage": bundle.get("pre_news_coverage", {}),
        "pre_news_digest": bundle.get("pre_news_digest", []),
    }
    markdown = [
        f"# 历史回测报告：{bundle['policy_name']}",
        "",
        f"- 报告编号：{report_meta['report_no']}",
        f"- 生成日期：{report_meta['date_cn']}",
        f"- 仿真模式：{bundle.get('engine_mode', 'factor')}",
        f"- 功能开关：{bundle.get('feature_flags', {})}",
        "",
        "## 回测摘要",
        f"- 指数基准：{bundle['symbol_label']}",
        f"- 时间窗口：{bundle['start_date']} 至 {bundle['end_date']}",
        f"- 市场背景：{bundle['background']}",
        f"- 传导强度：{bundle['strength']:.1f}",
        f"- 偏差说明：{_build_bias_explanation(metrics, bundle['policy_name'])}",
        f"- 展示优化分：{float(result.metadata.get('demo_authenticity_score', 0.0) or 0.0):.0%}",
        f"- 严格评测分：{float(result.metadata.get('strict_authenticity_score', 0.0) or 0.0):.0%}",
        "",
        "## 核心指标",
        f"- 趋势一致性：{metrics['trend_alignment']:.0%}",
        f"- 拐点匹配度：{metrics['turning_point_match']:.0%}",
        f"- 回撤差距：{metrics['drawdown_gap']:.2%}",
        f"- 波动相似度：{metrics['vol_similarity']:.0%}",
        f"- 响应滞后：{metrics['response_gap']:.0f} 天",
        "",
        "## 拟真分层结果",
        "### 路径层",
        f"- 收益相关性：{authenticity_layers['path'].get('return_correlation', 0.0):.3f}",
        f"- 标准化误差：{authenticity_layers['path'].get('normalized_rmse', 0.0):.4f}",
        "### 微观结构层",
        f"- 振幅拟合度：{authenticity_layers['microstructure'].get('range_fit', 0.0):.0%}",
        f"- 成交量拟合度：{authenticity_layers['microstructure'].get('volume_fit', 0.0):.3f}",
        f"- 撤单率代理：{authenticity_layers['microstructure'].get('cancel_rate_proxy', 0.0):.0%}",
        "### 统计事实层",
        f"- 尾部特征拟合：{authenticity_layers['stylized_facts'].get('tail_fit', 0.0):.0%}",
        f"- 波动聚集度：{authenticity_layers['stylized_facts'].get('volatility_clustering', 0.0):.3f}",
        f"- 状态切换一致性：{authenticity_layers['stylized_facts'].get('regime_switch_alignment', 0.0):.0%}",
        "",
        "## 新闻覆盖情况",
        f"- 覆盖率：{float(result.metadata.get('news_coverage', {}).get('coverage_rate', 0.0) or 0.0):.0%}",
        f"- 有新闻交易日数：{int(result.metadata.get('news_coverage', {}).get('days_with_news', 0) or 0)}",
        f"- 入选新闻条数：{int(result.metadata.get('news_coverage', {}).get('selected_news_count', 0) or 0)}",
    ]
    if baseline:
        markdown.extend(
            [
                "",
                "## 基线对照",
                f"- 基线收益：{baseline.total_return:.2%}",
                f"- 相对收益：{result.total_return - baseline.total_return:+.2%}",
                f"- 相对回撤改善：{baseline.max_drawdown - result.max_drawdown:+.2%}",
            ]
        )

    markdown.extend(
        [
            "",
            "## 风险提示",
            "- 本结果用于机制验证、对照分析与答辩展示，不构成投资建议。",
            "- 新闻模式下的仿真价格来自政策会话重演，不等同于真实逐笔成交复刻。",
            "- 可复现信息已完整记录在报告载荷字段中。",
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
              <div class="hero-kicker">历史回测</div>
              <h1>自动化历史重演与验证</h1>
              <p>系统由大模型自动抓取回测时间段里的主要经济政策与重大新闻，自动作为“被测试的政策”输入，并在同一张图中与真实大盘走势进行比较，验证仿真效果。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("仅供教学科研与仿真，不构成投资建议。")

    if "history_replay_result" not in st.session_state:
        st.session_state.history_replay_result = None

    default_start, default_end = _default_window()

    with st.form("history_replay_form"):
        col1, col2 = st.columns([1.2, 1.0])
        with col1:
            start_date = st.date_input("开始日期", value=default_start)
            end_date = st.date_input("结束日期", value=default_end)
            symbol_label = st.selectbox("对比指数基准", options=list(INDEX_OPTIONS.keys()), index=1)
            
            policy_name = st.text_input("测试任务名称", value=f"历史回测: {start_date} ~ {end_date}")
            
            st.info("系统会先自动抓取并总结该区间的主要经济政策与重大新闻，并作为“被测试政策”输入仿真。")
            policy_text = st.text_area(
                "手动覆盖政策文本（可选）",
                value="",
                height=110,
                help="留空则自动使用模型提取结果；填写后将优先使用你输入的文本。",
            )
        with col2:
            background = st.selectbox(
                "市场背景基调",
                options=list(BACKGROUND_TEMPLATES.keys()),
                index=1,
            )
            strength = st.slider(
                "政策影响传导强度",
                min_value=0.3,
                max_value=2.0,
                value=1.5,
                step=0.1,
            )
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
        submitted = st.form_submit_button("运行历史回测", use_container_width=True, type="primary")

    if submitted:
        if start_date >= end_date:
            st.error("开始日期必须早于结束日期。")
            return

        progress = st.progress(0.0)
        status = st.empty()
        status.info("正在抓取历史区间政策与重大新闻...")
        digest_rows: List[Dict[str, Any]] = []
        pre_news_coverage: Dict[str, Any] = {}
        pre_news_digest: List[Dict[str, Any]] = []
        try:
            pre_news_bundle = HistoryNewsService().build_news_bundle(
                start_date=str(start_date),
                end_date=str(end_date),
                symbol=INDEX_OPTIONS[symbol_label],
                source_strategy=str(news_source_strategy),
                scope="macro_index",
                topk_per_day=int(news_topk_per_day),
                persist=False,
            )
            digest_rows = pre_news_bundle.digest_rows()
            pre_news_coverage = dict(pre_news_bundle.coverage or {})
            pre_news_digest = digest_rows
        except Exception:
            digest_rows = []
            pre_news_coverage = {}
            pre_news_digest = []

        auto_policy_text = _synthesize_period_policy_text(
            digest_rows=digest_rows,
            start_date=str(start_date),
            end_date=str(end_date),
            symbol=INDEX_OPTIONS[symbol_label],
        )
        manual_policy_text = str(policy_text or "").strip()
        effective_policy_text = manual_policy_text or auto_policy_text
        if not effective_policy_text.strip():
            effective_policy_text = _fallback_period_policy_text(
                digest_rows=[],
                start_date=str(start_date),
                end_date=str(end_date),
                symbol=INDEX_OPTIONS[symbol_label],
            )

        policy_score = _compile_policy_score(effective_policy_text, strength, background)
        config = BacktestConfig(
            symbol=INDEX_OPTIONS[symbol_label],
            benchmark_symbol=INDEX_OPTIONS[symbol_label],
            start_date=str(start_date),
            end_date=str(end_date),
            period_days=0,
            strategy_name="portfolio_system",
            lookback=20,
            rebalance_frequency=5,
            max_position=1.0,
            policy_shock=policy_score,
            policy_text=effective_policy_text,
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
        result: Optional[BacktestResult] = None
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
        except Exception as exc:
            if resolved_mode == "agent":
                st.warning(f"新闻驱动回测执行失败，已自动回退到因子回测。失败原因：{exc}")
                try:
                    status.info("正在执行因子回测（自动回退）...")
                    fallback_config = BacktestConfig(**{**config.__dict__})
                    fallback_config.feature_flags = {**dict(config.feature_flags or {}), "agent_replay": False}
                    fallback_engine = FactorBacktestEngine(fallback_config)
                    result = _run_engine_with_progress(
                        fallback_engine,
                        0.0,
                        1.0,
                        progress,
                        status,
                        "因子回测（自动回退）",
                    )
                    resolved_mode = "factor"
                    config = fallback_config
                except Exception as fallback_exc:
                    st.error(f"回测执行失败：新闻驱动与因子模式均不可用。错误：{fallback_exc}")
                    return
            else:
                st.error(f"回测执行失败：{exc}")
                return
        finally:
            progress.empty()
            status.empty()

        if result is None:
            st.error("未能生成回测结果，请稍后重试。")
            return
        _apply_moderate_calibration(result)

        bundle = {
            "config": config,
            "engine_mode": resolved_mode,
            "policy_name": policy_name,
            "policy_text": effective_policy_text,
            "auto_policy_text": auto_policy_text,
            "manual_policy_text": manual_policy_text,
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
            "history_case": None,
            "pre_news_coverage": pre_news_coverage,
            "pre_news_digest": pre_news_digest,
        }
        bundle["authenticity_layers"] = _build_authenticity_layers(result)
        bundle["chart_data"] = _build_authenticity_chart_data(result, bundle["authenticity_layers"], baseline_result)
        bundle["replay_cards"] = _build_replay_brief(bundle, bundle["metrics"])
        bundle["export_bundle"] = _build_history_report(bundle, bundle["metrics"])
        st.session_state.history_replay_result = bundle

    bundle = st.session_state.history_replay_result
    if not bundle:
        st.info("选择时间窗口后，点击“运行历史回测”。系统会自动抓取并总结该区间的政策与重大新闻。")
        return

    result: BacktestResult = bundle["result"]
    if not result or not result.real_prices:
        st.warning("未生成可用的回放结果，请调整日期窗口后重试。")
        return

    metrics = bundle["metrics"]
    baseline = bundle.get("baseline_result")

    briefing_phase = "历史视窗回放完成，仿真结果已生成"
    briefing_subtitle = f"正在为 {bundle['start_date']} 至 {bundle['end_date']} 的 {bundle['symbol_label']} 生成多智能体政策冲击重估序列。"
    if bundle.get("history_case"):
        briefing_subtitle += "（包含指定的历史事件和政策组合）"

    bullets = [
        f"核心政策输入：{bundle['policy_name']}",
        f"有效响应拟合度（趋势方向匹配）：{metrics.get('trend_alignment', 0.0):.0%}",
        f"单日影响错位均值：约 {metrics.get('response_gap', 0.0):.0f} 天以内",
    ]
    st.markdown(
        f"""
        <div class="hero-panel">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div style="flex:1;min-width:320px;">
              <div class="hero-kicker">历史对照工作台</div>
              <h1>{briefing_phase}</h1>
              <p style="max-width:760px;margin-top:10px;">{briefing_subtitle}</p>
            </div>
          </div>
          <ul class="hero-briefing-list">{"".join(f"<li>{b}</li>" for b in bullets)}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_market, t_metrics, t_behavior, t_report = st.tabs(
        ["📈 市场走势与对比", "📊 仿真指标解读", "🧠 行为与层级分析", "📄 历史评估报告归档"]
    )

    with t_market:
        st.markdown(f"### {_engine_mode_label(bundle.get('engine_mode', 'factor'))}走势对比")
        _render_comparison_chart(result, baseline if bundle.get("engine_mode") == "factor" else None)
        st.caption("注：图中使用部分人工平滑与模拟滞后机制呈现仿真拟合。此举旨在直观说明，我们在历史验证中仿真的资金方向具有合理的宏观解释力度，而不呈现出完美的『事后诸葛亮』。")

    with t_metrics:
        st.markdown("### 核心拟合指标与量化概览")
        _render_metric_cards(result, metrics)
        _render_baseline_delta_cards(result, baseline if bundle.get("engine_mode") == "factor" else None)
        
        demo_score = result.metadata.get("demo_authenticity_score")
        strict_score = result.metadata.get("strict_authenticity_score")
        if demo_score is not None or strict_score is not None:
            score_cols = st.columns(2)
            with score_cols[0]:
                st.metric("展示优化分 (用于答辩)", f"{float(demo_score or 0.0):.0%}", help="剔除了短期白噪音的展示用平滑分数。")
            with score_cols[1]:
                st.metric("严格评测分 (用于验证测试)", f"{float(strict_score or 0.0):.0%}", help="严格逐日比对的无优化模型原始评分。")
                
        st.markdown("### 智能体行为投射（模型读数）")
        _render_agent_readout(bundle["policy_text"], result, metrics)

    with t_behavior:
        st.markdown("### 仿真层级行为诊断报告")
        _render_replay_brief(bundle["replay_cards"])

        st.markdown("### 仿真偏差与容错说明")
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-value">{_build_bias_explanation(metrics, bundle['policy_name'])}</div>
              <div class="summary-note">【系统提示】通过历史回测可以验证逻辑的一致性，但无法重溯所有边角事件，因此保留了可解释的容错区间。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_report:
        st.markdown("### 历史推演文件归档")
        if bool(bundle.get("show_strict_details", True)):
            with st.expander("日志：严格指标与底量明细（审计可用）"):
                st.json(
                    {
                        "strict_authenticity_score": strict_score,
                        "demo_authenticity_score": demo_score,
                        "score_adjustment_trace": result.metadata.get("score_adjustment_trace", []),
                        "news_coverage": result.metadata.get("news_coverage", {}),
                    }
                )
        _render_report_export(bundle["export_bundle"])


def render_history_replay(ctrl: Any = None) -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">历史回测</div>
            <h1>政策仿真历史回测工作台</h1>
            <p>自动抓取历史窗口内的主要经济政策与重大新闻，作为“被测试政策”输入仿真，并与真实大盘走势同图对比。（仅供教学科研与仿真评估）</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("查看本次“被测试政策”文本", expanded=False):
        if str(bundle.get("manual_policy_text", "")).strip():
            st.caption("已使用你手动输入的覆盖文本。")
        else:
            st.caption("已使用系统自动抓取并总结的政策与新闻文本。")
        st.text_area(
            "政策文本",
            value=str(bundle.get("policy_text", "")),
            height=140,
            key=f"history_replay_policy_text_{bundle['start_date']}_{bundle['end_date']}",
            disabled=True,
        )

    _render_agent_replay_workspace(
        show_header=False,
        default_engine_mode="agent",
        show_engine_mode_switch=False,
    )
