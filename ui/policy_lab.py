"""Policy lab page focused on government-facing policy experiments."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.data.market_data_provider import MarketDataProvider, MarketDataQuery
from core.event_store import EventRecord, EventStore, EventType
from core.macro.government import GovernmentAgent, PolicyShock
from core.runtime_mode import RuntimeModeProfile, resolve_runtime_mode_profile
from policy.structured import PolicyPackage
from ui.reporting import official_report_meta, write_report_artifacts


POLICY_TYPE_OPTIONS = {
    "Tax Adjustment": "tax",
    "Liquidity Injection": "liquidity",
    "Fiscal Stimulus": "fiscal",
    "Regulatory Tightening": "tightening",
    "Market Stabilization": "stabilization",
    "Custom Policy": "custom",
}

TEMPLATE_LIBRARY_PATH = Path("data") / "policy_templates.json"
POLICY_REPORT_DIR = Path("outputs") / "policy_reports"
CONTROL_MODE_OPTIONS = [
    "No control arm",
    "No policy baseline",
    "Template recommendation control",
    "Mild variant",
    "Risk-stress variant",
]
INDEX_BENCHMARK_OPTIONS = {
    "上证指数（000001）": "sh000001",
    "深证成指（399001）": "sz399001",
    "创业板指（399006）": "sz399006",
}
CHART_MODE_OPTIONS = ["日K", "分时"]


@dataclass
class PolicyNarrativeCard:
    title: str
    summary: str
    bullets: List[str]
    tone: str = "neutral"


def _resolve_runtime_profile() -> RuntimeModeProfile:
    mode = str(st.session_state.get("simulation_mode", "SMART")).strip().upper()
    return resolve_runtime_mode_profile(mode)


def _run_async(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _default_template_library() -> List[Dict[str, Any]]:
    return [
        {
            "id": "stamp-tax-liquidity",
            "category": "Market Stabilization",
            "title": "Cut stamp tax with liquidity support",
            "policy_type": "Tax Adjustment",
            "policy_text": "Reduce stamp tax and pair it with liquidity support to stabilize expectations.",
            "policy_goal": "Improve liquidity, reduce trading frictions, and stabilize index dynamics.",
            "suitable_departments": "Finance, Tax, Securities Regulator, Stabilization Fund",
            "recommended_intensity": 1.1,
            "recommended_duration": 30,
            "default_rumor_noise": False,
            "control_label": "Maintain current tax and liquidity setup",
            "control_text": "Keep current tax and liquidity arrangement without adding stabilization interventions.",
        },
        {
            "id": "targeted-fiscal-demand",
            "category": "Fiscal Support",
            "title": "Targeted fiscal expansion with sector focus",
            "policy_type": "Fiscal Stimulus",
            "policy_text": "Launch targeted fiscal spending for infrastructure and advanced manufacturing with phased implementation.",
            "policy_goal": "Stabilize growth expectations while preserving financial stability.",
            "suitable_departments": "Finance, Development and Reform, Industry, Local Government",
            "recommended_intensity": 1.0,
            "recommended_duration": 60,
            "default_rumor_noise": False,
            "control_label": "No targeted fiscal expansion",
            "control_text": "Keep fiscal stance unchanged as control arm.",
        },
        {
            "id": "rumor-refutation-stabilization",
            "category": "Expectation Management",
            "title": "Rumor refutation with stabilization statement",
            "policy_type": "Market Stabilization",
            "policy_text": "Issue official clarification to refute market rumors and release a coordinated stabilization communication package.",
            "policy_goal": "Reduce panic and suppress rumor-driven sell pressure.",
            "suitable_departments": "Regulator, Official Media, Exchange, Stability Fund",
            "recommended_intensity": 1.2,
            "recommended_duration": 20,
            "default_rumor_noise": True,
            "control_label": "No clarification response",
            "control_text": "Observe market dynamics without official clarification.",
        },
    ]


def _load_policy_templates() -> List[Dict[str, Any]]:
    if TEMPLATE_LIBRARY_PATH.exists():
        try:
            payload = json.loads(TEMPLATE_LIBRARY_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                return payload
        except Exception:
            pass
    return _default_template_library()


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _policy_feature_flags(enable_structured_parser: bool = True) -> Dict[str, bool]:
    return {
        "structured_policy_parser_v1": bool(enable_structured_parser),
        "policy_transmission_layers_v1": True,
        "policy_transmission_graph_v1": True,
    }


def _compile_policy_bundle(
    policy_text: str,
    intensity: float,
    *,
    policy_type_hint: Optional[str] = None,
    market_regime: Optional[str] = None,
    enable_structured_parser: bool = True,
) -> Tuple[PolicyShock, PolicyPackage]:
    gov = GovernmentAgent(feature_flags=_policy_feature_flags(enable_structured_parser))
    package = gov.compile_policy_package(
        policy_text,
        tick=1,
        policy_type_hint=policy_type_hint,
        intensity=float(intensity),
        market_regime=market_regime,
        snapshot_info={
            "policy_text_length": len(policy_text or ""),
            "policy_type_hint": policy_type_hint or "",
            "parser_mode": "structured" if enable_structured_parser else "legacy",
        },
    )
    shock = PolicyShock(policy_id=package.event.policy_id, policy_text=policy_text, **package.to_policy_shock_fields())
    shock.metadata = {
        "policy_event": package.event.to_dict() if hasattr(package.event, "to_dict") else {
            "policy_id": package.event.policy_id,
            "raw_text": package.event.raw_text,
            "policy_type": package.event.policy_type,
        },
        "policy_package": package.to_dict(),
        "reproducibility": {
            "seed": int(package.metadata.get("seed", 0)),
            "config_hash": str(package.metadata.get("config_hash", "")),
            "snapshot_info": dict(package.metadata.get("snapshot_info", {})),
        },
        "parser_mode": package.uncertainty.parser_mode,
        "feature_flags": dict(package.metadata.get("feature_flags", {})),
    }
    return shock, package


def _compile_scaled_shock(
    policy_text: str,
    intensity: float,
    *,
    enable_structured_parser: bool = True,
) -> PolicyShock:
    shock, _ = _compile_policy_bundle(policy_text, intensity, enable_structured_parser=enable_structured_parser)
    return shock


def _shock_score(shock: PolicyShock) -> float:
    return (
        shock.liquidity_injection * 1.3
        + shock.fiscal_stimulus_delta * 1.5
        - shock.policy_rate_delta * 60.0
        - shock.credit_spread_delta * 18.0
        - shock.stamp_tax_delta * 420.0
        + shock.sentiment_delta * 1.2
        + shock.rumor_shock * 1.6
    )


def _policy_anchor_date() -> pd.Timestamp:
    key = "policy_lab_open_date"
    if key not in st.session_state:
        st.session_state[key] = pd.Timestamp.now(tz="Asia/Shanghai").normalize().strftime("%Y-%m-%d")
    return pd.to_datetime(st.session_state[key], errors="coerce").normalize()


def _load_real_index_history(index_symbol: str, lookback_days: int, end_date: pd.Timestamp) -> pd.DataFrame:
    start_date = (pd.Timestamp(end_date).normalize() - pd.Timedelta(days=max(int(lookback_days) * 3, 120))).strftime("%Y-%m-%d")
    query = MarketDataQuery(
        symbol=str(index_symbol),
        interval="1d",
        start=start_date,
        end=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        period_days=max(int(lookback_days), 30),
        adjust="",
        market="CN",
    )
    try:
        provider = MarketDataProvider()
        frame = provider.get_ohlcv(query, use_cache=True, freeze_snapshot=False)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return frame
    out = frame.sort_values("datetime").tail(max(int(lookback_days), 10)).copy()
    out["time"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["step"] = np.arange(1, len(out) + 1)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    if out.empty:
        return out
    if out["volume"].isna().all():
        out["volume"] = 1_000_000.0
    out["volume"] = out["volume"].fillna(out["volume"].median())
    return out[["step", "time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _generate_policy_metrics(
    *,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    scenario_key: str,
    market_history: Optional[pd.DataFrame] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    shock = _compile_scaled_shock(policy_text, intensity)
    score = _shock_score(shock)
    seed = f"{scenario_key}|{policy_text}|{intensity}|{duration_days}|{rumor_noise}"
    rng = np.random.default_rng(_seed_from_text(seed))

    periods = max(10, int(duration_days))
    rows: List[Dict[str, float | int | str]] = []
    history = market_history.copy() if isinstance(market_history, pd.DataFrame) and not market_history.empty else pd.DataFrame()
    if not history.empty:
        history = history.tail(periods).reset_index(drop=True)
        base_returns = history["close"].pct_change().fillna(0.0).astype(float)
        price = float(history.iloc[0]["close"])
        for idx, row in history.iterrows():
            step = idx + 1
            base_ret = float(base_returns.iloc[idx])
            drift = 0.0002 + np.clip(score, -1.0, 1.0) * 0.0020 * np.exp(-(step - 1) / max(periods * 0.6, 1.0))
            rumor_term = (shock.rumor_shock * 0.004 if rumor_noise else 0.0) * np.exp(-(step - 1) / max(periods * 0.25, 1.0))
            ret = base_ret + drift + rumor_term + rng.normal(0.0, 0.0015)
            prev = price
            price = max(1600.0, prev * (1.0 + ret))
            panic = float(np.clip(0.18 + max(0.0, -ret) * 7.5 + max(0.0, rumor_term) * 6.0, 0.05, 0.95))
            csad = float(np.clip(0.05 + panic * 0.09 + abs(ret) * 4.0, 0.04, 0.22))
            base_high = float(row.get("high", max(prev, price)))
            base_low = float(row.get("low", min(prev, price)))
            high = max(base_high, prev, price) * (1 + abs(rng.normal(0.0, 0.0012)))
            low = min(base_low, prev, price) * (1 - abs(rng.normal(0.0, 0.0012)))
            base_volume = float(row.get("volume", 1_000_000.0) or 1_000_000.0)
            volume = float(base_volume * (1 + 0.25 * abs(score) + 0.35 * panic))
            rows.append(
                {
                    "step": step,
                    "time": str(row.get("time", "")),
                    "open": round(prev, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(price, 2),
                    "volume": round(volume, 2),
                    "csad": round(csad, 4),
                    "panic_level": round(panic, 4),
                }
            )
        return pd.DataFrame(rows)

    end_anchor = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_anchor, periods=periods)
    price = 3000.0
    for idx, dt in enumerate(dates, start=1):
        drift = 0.0003 + np.clip(score, -1.0, 1.0) * 0.0025 * np.exp(-(idx - 1) / max(periods * 0.5, 1.0))
        rumor_term = (shock.rumor_shock * 0.008 if rumor_noise else 0.0) * np.exp(-(idx - 1) / max(periods * 0.25, 1.0))
        ret = drift + rumor_term + rng.normal(0.0, 0.004)
        prev = price
        price = max(1600.0, prev * (1.0 + ret))
        band = abs(ret) * 0.9 + 0.001
        high = max(prev, price) * (1 + band)
        low = min(prev, price) * (1 - band)
        panic = float(np.clip(0.2 + max(0.0, -ret) * 8.0 + max(0.0, rumor_term) * 6.0, 0.05, 0.95))
        csad = float(np.clip(0.05 + panic * 0.1 + abs(ret) * 4.5, 0.04, 0.22))
        volume = float(1_000_000 * (1 + 0.3 * abs(score) + 0.4 * panic))
        rows.append(
            {
                "step": idx,
                "time": dt.strftime("%Y-%m-%d"),
                "open": round(prev, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": round(volume, 2),
                "csad": round(csad, 4),
                "panic_level": round(panic, 4),
            }
        )
    return pd.DataFrame(rows)


def _run_policy_committee_review(policy_text: str, runtime_profile: RuntimeModeProfile) -> Dict[str, Any]:
    if not runtime_profile.enable_policy_committee or not runtime_profile.use_live_api:
        return {}
    try:
        from config import GLOBAL_CONFIG
        from core.policy_committee import PolicyCommittee
    except Exception as exc:
        return {"error": f"committee_import_error: {exc}"}

    if not (GLOBAL_CONFIG.DEEPSEEK_API_KEY or GLOBAL_CONFIG.ZHIPU_API_KEY):
        return {"error": "committee_skipped_no_api_key"}

    try:
        committee = PolicyCommittee(api_key=GLOBAL_CONFIG.DEEPSEEK_API_KEY)
        result = committee.interpret(policy_text)
        return {
            "parameters": dict(result.parameters or {}),
            "compliance_passed": bool(result.compliance.passed),
            "violations": list(result.compliance.violations or []),
            "warnings": list(result.compliance.warnings or []),
            "reasoning_chain": list(result.reasoning_chain or []),
            "final_state": dict(result.final_state or {}),
        }
    except Exception as exc:
        return {"error": f"committee_runtime_error: {exc}"}


def _generate_policy_metrics_deep(
    *,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    runtime_profile: RuntimeModeProfile,
    end_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    from agents.persona import Persona
    from agents.trader_agent import TraderAgent
    from engine.simulation_loop import MarketEnvironment

    symbol = "A_SHARE_IDX"
    role_specs: List[Tuple[str, float]] = [
        ("retail_day_trader", 240_000.0),
        ("retail_swing", 320_000.0),
        ("retail_momentum_chaser", 200_000.0),
        ("mutual_fund", 1_800_000.0),
        ("quant_arbitrage", 1_400_000.0),
        ("market_maker", 2_200_000.0),
    ]
    llm_cutoff = max(3, int(round(len(role_specs) * 0.67)))
    agents: List[TraderAgent] = []
    for idx, (archetype_key, cash) in enumerate(role_specs):
        persona = Persona.from_archetype(archetype_key, name=f"{archetype_key}_{idx:02d}")
        use_llm = bool(runtime_profile.llm_primary and idx < llm_cutoff)
        agent = TraderAgent(
            agent_id=f"deep_{idx:02d}",
            cash_balance=float(cash),
            portfolio={symbol: int(200 + 40 * idx)},
            psychology_profile={
                "feature_flags": {"trader_intent_execution_split_v1": True},
                "institution_type": archetype_key,
            },
            persona=persona,
            use_llm=use_llm,
            model_priority=list(runtime_profile.model_priority),
            execution_plan_enabled=True,
        )
        agents.append(agent)

    env = MarketEnvironment(
        agents,
        use_isolated_matching=True,
        market_pipeline_v2=bool(runtime_profile.market_pipeline_v2),
        runner_symbol=symbol,
        simulation_mode=runtime_profile.mode,
        llm_primary=bool(runtime_profile.llm_primary),
        deep_reasoning_pause_s=float(runtime_profile.pause_for_llm_seconds),
        model_priority=list(runtime_profile.model_priority),
        enable_policy_committee=bool(runtime_profile.enable_policy_committee),
    )

    committee_report = _run_policy_committee_review(policy_text, runtime_profile)
    policy_payload = f"{policy_text} [policy_intensity={float(intensity):.2f}]"
    env.schedule_policy_shock(policy_payload)
    if float(intensity) >= 1.1:
        env.schedule_policy_shock("政策力度超预期，资金对冲与追价行为同步放大。")
    if rumor_noise:
        env.schedule_policy_shock("谣言扩散导致恐慌交易升温，监管机构发布澄清提示。")

    steps = max(8, min(int(max(6.0, duration_days * max(0.6, min(1.2, float(intensity))))), 18))
    anchor = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=anchor, periods=steps)
    rows: List[Dict[str, Any]] = []
    last_close = float(env.current_price)
    latest_report: Dict[str, Any] = {}
    try:
        for idx in range(steps):
            latest_report = dict(_run_async(env.simulation_step()) or {})
            close = float(latest_report.get("new_price", last_close) or last_close)
            open_price = float(latest_report.get("old_price", last_close) or last_close)
            spread_proxy = abs(float(latest_report.get("price_change_pct", 0.0) or 0.0)) / 100.0
            high = max(open_price, close) * (1.0 + 0.001 + spread_proxy * 0.35)
            low = min(open_price, close) * (1.0 - 0.001 - spread_proxy * 0.35)
            volume = float(latest_report.get("buy_volume", 0.0) or 0.0) + float(
                latest_report.get("sell_volume", 0.0) or 0.0
            )
            macro_state = dict(latest_report.get("macro_state", {}) or {})
            sentiment = float(macro_state.get("sentiment_index", 0.5) or 0.5)
            panic = float(np.clip(1.0 - sentiment + spread_proxy * 5.0, 0.05, 0.95))
            diagnostics = dict(latest_report.get("behavioral_diagnostics", {}) or {})
            csad = float(diagnostics.get("csad", 0.05) or 0.05)
            rows.append(
                {
                    "step": idx + 1,
                    "time": dates[idx].strftime("%Y-%m-%d"),
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(max(0.01, low), 2),
                    "close": round(close, 2),
                    "volume": round(max(volume, 1.0), 2),
                    "csad": round(max(csad, 0.0), 4),
                    "panic_level": round(panic, 4),
                }
            )
            last_close = close
    finally:
        env.close()

    thinking_stats = dict(latest_report.get("thinking_stats", {}) or {})
    if not thinking_stats:
        thinking_stats = {
            "fast_count": int(sum(int(getattr(agent, "fast_think_count", 0) or 0) for agent in agents)),
            "slow_count": int(sum(int(getattr(agent, "slow_think_count", 0) or 0) for agent in agents)),
        }
    deep_meta: Dict[str, Any] = {
        "mode_profile": runtime_profile.to_dict(),
        "llm_agent_count": int(sum(1 for agent in agents if bool(getattr(agent, "use_llm", False)))),
        "agent_count": len(agents),
        "thinking_stats": thinking_stats,
        "committee_report": committee_report,
        "latest_step_report": latest_report,
    }
    baseline = pd.DataFrame(rows)
    deep_meta["counterfactual_regulation"] = _build_regulation_counterfactual_worlds(
        baseline,
        intensity=float(intensity),
    )
    return baseline, deep_meta


def _apply_regulatory_intervention_worldline(
    frame: pd.DataFrame,
    *,
    intervention_step: int,
    intensity: float,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy().reset_index(drop=True)
    prev_close = float(out.iloc[0]["open"])
    for idx, row in out.iterrows():
        step = int(row["step"])
        close = float(row["close"])
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        panic = float(row["panic_level"])
        csad = float(row["csad"])
        volume = float(row["volume"])
        if step >= intervention_step:
            phase = step - intervention_step
            relief = np.exp(-phase / 8.0)
            close *= 1.0 + (0.0022 * float(intensity) * relief)
            panic *= 1.0 - (0.24 * relief)
            csad *= 1.0 - (0.18 * relief)
            volume *= 1.0 + (0.06 * relief)
        open_price = prev_close
        high = max(high, open_price, close) * 1.0015
        low = min(low, open_price, close) * 0.9985
        out.at[idx, "open"] = round(open_price, 2)
        out.at[idx, "high"] = round(max(high, close), 2)
        out.at[idx, "low"] = round(max(0.01, min(low, close)), 2)
        out.at[idx, "close"] = round(close, 2)
        out.at[idx, "panic_level"] = round(float(np.clip(panic, 0.02, 0.98)), 4)
        out.at[idx, "csad"] = round(float(max(csad, 0.0)), 4)
        out.at[idx, "volume"] = round(float(max(volume, 1.0)), 2)
        prev_close = float(out.at[idx, "close"])
    return out


def _worldline_scorecard(frame: pd.DataFrame) -> Dict[str, float]:
    summary = _compute_policy_summary(frame)
    return {
        "return_pct": float(summary["return_pct"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "avg_panic": float(summary["avg_panic"]),
        "max_panic": float(summary["max_panic"]),
        "volatility": float(summary["volatility"]),
    }


def _build_regulation_counterfactual_worlds(frame: pd.DataFrame, *, intensity: float) -> Dict[str, Any]:
    if frame.empty:
        return {}
    steps = len(frame)
    early_step = max(2, int(round(steps * 0.25)))
    late_step = max(3, int(round(steps * 0.70)))
    no_intervention = frame.copy()
    early = _apply_regulatory_intervention_worldline(
        frame,
        intervention_step=early_step,
        intensity=float(intensity),
    )
    late = _apply_regulatory_intervention_worldline(
        frame,
        intervention_step=late_step,
        intensity=float(intensity),
    )
    scorecards = {
        "no_intervention": _worldline_scorecard(no_intervention),
        "early_intervention": _worldline_scorecard(early),
        "late_intervention": _worldline_scorecard(late),
    }
    ranking = sorted(
        scorecards.items(),
        key=lambda item: (
            float(item[1]["max_panic"]) * 0.45
            + float(item[1]["max_drawdown"]) * 0.35
            + float(item[1]["volatility"]) * 0.20
            - float(item[1]["return_pct"]) * 0.15
        ),
    )
    return {
        "intervention_steps": {"early": early_step, "late": late_step},
        "recommended_timing": str(ranking[0][0]) if ranking else "no_intervention",
        "scorecards": scorecards,
        "worlds": {
            "no_intervention": no_intervention.to_dict(orient="records"),
            "early_intervention": early.to_dict(orient="records"),
            "late_intervention": late.to_dict(orient="records"),
        },
    }


def _compute_policy_summary(metrics: pd.DataFrame) -> Dict[str, float]:
    close = metrics["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    drawdown = close / close.cummax() - 1.0
    return {
        "return_pct": float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0),
        "avg_panic": float(metrics["panic_level"].mean()),
        "max_panic": float(metrics["panic_level"].max()),
        "avg_csad": float(metrics["csad"].mean()),
        "max_drawdown": float(abs(drawdown.min())),
        "avg_volume": float(metrics["volume"].mean()),
        "volatility": float(returns.std()),
    }


def _build_chart(frame: pd.DataFrame, *, chart_title: str = "指数日K（东方财富风格）", mode: str = "日K") -> go.Figure:
    chart = frame.copy()
    chart["ma5"] = chart["close"].rolling(5).mean()
    chart["ma10"] = chart["close"].rolling(10).mean()
    up_mask = (chart["close"] >= chart["open"]).tolist()
    volume_colors = ["#d9383a" if is_up else "#18a058" for is_up in up_mask]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.74, 0.26])
    if mode == "分时":
        chart["avg_price"] = chart["close"].expanding().mean()
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=chart["close"],
                mode="lines",
                name="分时",
                line=dict(color="#0066cc", width=1.8),
                hovertemplate="时间=%{x}<br>价格=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=chart["avg_price"],
                mode="lines",
                name="均价",
                line=dict(color="#f39c12", width=1.2, dash="dot"),
                hovertemplate="时间=%{x}<br>均价=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=chart["time"],
                open=chart["open"],
                high=chart["high"],
                low=chart["low"],
                close=chart["close"],
                name="日K",
                increasing_line_color="#d9383a",
                decreasing_line_color="#18a058",
                increasing_fillcolor="#d9383a",
                decreasing_fillcolor="#18a058",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=chart["ma5"],
                mode="lines",
                name="MA5",
                line=dict(color="#f39c12", width=1.4),
                hovertemplate="时间=%{x}<br>MA5=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=chart["ma10"],
                mode="lines",
                name="MA10",
                line=dict(color="#2980b9", width=1.4),
                hovertemplate="时间=%{x}<br>MA10=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=chart["time"],
            y=chart["volume"],
            name="成交量",
            marker_color=volume_colors,
            opacity=0.8,
            hovertemplate="时间=%{x}<br>成交量=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=chart_title, x=0.01, xanchor="left", font=dict(size=16)),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", y=1.04, x=0.0),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=0,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#9aa0a6",
        spikethickness=1,
    )
    fig.update_yaxes(
        title_text="指数点位",
        row=1,
        col=1,
        side="right",
        gridcolor="#ededed",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#9aa0a6",
        spikethickness=1,
    )
    fig.update_yaxes(
        title_text="成交量",
        row=2,
        col=1,
        side="right",
        gridcolor="#f3f3f3",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#9aa0a6",
        spikethickness=1,
    )
    return fig


def _render_quote_banner(frame: pd.DataFrame, *, index_label: str, index_symbol: str, source_text: str, history_end: str) -> None:
    if frame.empty:
        return
    latest = frame.iloc[-1]
    prev_close = float(frame.iloc[-2]["close"]) if len(frame) > 1 else float(latest["open"])
    last_close = float(latest["close"])
    change = last_close - prev_close
    pct = change / prev_close if abs(prev_close) > 1e-9 else 0.0
    color = "#d9383a" if change >= 0 else "#18a058"
    sign = "+" if change >= 0 else ""
    st.markdown(
        (
            "<div style='border:1px solid #e6e6e6;border-radius:8px;padding:8px 12px;background:#fff;'>"
            f"<div style='font-size:13px;color:#666;'>{index_label}（{index_symbol}） | {source_text} | 截止 {history_end}</div>"
            f"<div style='font-size:28px;font-weight:700;color:{color};line-height:1.2'>{last_close:.2f}</div>"
            f"<div style='font-size:14px;color:{color};'>{sign}{change:.2f} ({sign}{pct:.2%})</div>"
            f"<div style='font-size:12px;color:#666;margin-top:2px;'>"
            f"开 {float(latest['open']):.2f} / 高 {float(latest['high']):.2f} / 低 {float(latest['low']):.2f} / 量 {float(latest['volume']):,.0f}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def _policy_narrative_key(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    payload = {
        "policy_text": str(policy_text or ""),
        "summary": dict(summary or {}),
        "explanation": dict(package_dict.get("explanation", {}) or {}),
        "top_layers": dict(package_dict.get("top_layers", {}) or {}),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _top_effect_text(items: List[Dict[str, Any]], limit: int = 3) -> str:
    if not items:
        return "暂无显著项"
    ranked = sorted(items, key=lambda item: abs(float(item.get("score", 0.0) or 0.0)), reverse=True)[:limit]
    parts: List[str] = []
    for item in ranked:
        name = str(item.get("name", "未命名"))
        score = float(item.get("score", 0.0) or 0.0)
        direction = "受益" if score >= 0 else "承压"
        parts.append(f"{name}（{direction}）")
    return "、".join(parts)


def _fallback_policy_narrative(policy_text: str, summary: Dict[str, float], package_dict: Dict[str, Any]) -> str:
    explanation = dict(package_dict.get("explanation", {}) or {})
    headline = str(explanation.get("headline", "") or "政策通过多条传导链路影响市场预期与资金行为。")
    primary_path = [str(node) for node in explanation.get("primary_path", []) if str(node).strip()]
    path_text = " -> ".join(primary_path[:5]) if primary_path else "政策信号 -> 预期修复 -> 风险偏好变化 -> 价格反馈"
    lag_days = int(explanation.get("expected_lag_days", 0) or 0)
    side_effects = [str(item) for item in explanation.get("side_effects", []) if str(item).strip()]
    risk_tip = "、".join(side_effects[:2]) if side_effects else "短期波动可能放大，需关注情绪过冲。"
    return "\n".join(
        [
            f"**一句话结论**：{headline}",
            f"- 这项政策的核心目标是：{str(policy_text or '').strip()[:120]}",
            f"- 主要传导路径：{path_text}",
            f"- 重点受影响主体：{_top_effect_text(list(explanation.get('affected_agents', []) or []))}",
            f"- 重点受影响行业：{_top_effect_text(list(explanation.get('affected_sectors', []) or []))}",
            f"- 重点受影响因子：{_top_effect_text(list(explanation.get('affected_factors', []) or []))}",
            f"- 市场结果指向：{_top_effect_text(list(explanation.get('market_results', []) or []))}",
            f"- 仿真表现：收益率 {summary.get('return_pct', 0.0):.2%}，最大回撤 {summary.get('max_drawdown', 0.0):.2%}，波动率 {summary.get('volatility', 0.0):.4f}",
            f"- 预计传导时滞：约 {lag_days} 天；风险提示：{risk_tip}",
        ]
    )


def _llm_policy_narrative(
    policy_text: str,
    summary: Dict[str, float],
    package_dict: Dict[str, Any],
    runtime_profile: RuntimeModeProfile,
) -> str:
    if not runtime_profile.use_live_api:
        return ""
    try:
        from core.inference.api_backend import APIBackend
    except Exception:
        return ""
    snapshot = {
        "policy_text": str(policy_text or ""),
        "summary": dict(summary or {}),
        "explanation": dict(package_dict.get("explanation", {}) or {}),
        "policy_schema": dict(package_dict.get("policy_schema", {}) or {}),
        "top_layers": dict(package_dict.get("top_layers", {}) or {}),
    }
    prompt = "\n".join(
        [
            "请把下面的政策仿真结果写成面向评委的中文自然语言解读。",
            "要求体现：快思考/慢思考切换、政策委员会防幻觉、订单簿撮合反馈。",
            "要求：不要输出 JSON、代码块、键名，必须严格按格式输出。",
            "【一句话结论】",
            "【政策如何影响市场】(3-4条)",
            "【这组结果为什么可信】(2条)",
            "【风险提示】(1条)",
            "【建议评委关注的指标】(2条)",
            "语言请简洁、可直接朗读。",
            f"数据：{json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str)}",
        ]
    )
    try:
        model_name = str(runtime_profile.model_priority[0]) if runtime_profile.model_priority else "deepseek-chat"
        backend = APIBackend(model=model_name, max_tokens=520, temperature=0.3)
        response = str(
            backend.generate(
                prompt,
                system_prompt=(
                    "你是政策风洞答辩讲解专家，强调DeepSeek/智谱双路路由、"
                    "快慢思考触发、委员会防幻觉与可解释市场传导。"
                ),
                timeout_budget=20.0,
            )
            or ""
        ).strip()
    except Exception:
        return ""
    if not response or response.startswith("[API Error]"):
        return ""
    if response.lstrip().startswith("{") or response.lstrip().startswith("["):
        return ""
    return response


def _get_policy_narrative(
    policy_text: str,
    summary: Dict[str, float],
    package_dict: Dict[str, Any],
    runtime_profile: RuntimeModeProfile,
) -> str:
    cache = st.session_state.setdefault("policy_lab_narrative_cache", {})
    key = f"{runtime_profile.mode}:{_policy_narrative_key(policy_text, summary, package_dict)}"
    if key in cache:
        return str(cache[key])
    text = _llm_policy_narrative(policy_text, summary, package_dict, runtime_profile)
    if not text:
        text = _fallback_policy_narrative(policy_text, summary, package_dict)
    cache[key] = text
    return text


def _persist_policy_event(
    *,
    selected_title: str,
    policy_text: str,
    intensity: float,
    duration_days: int,
    rumor_noise: bool,
    index_symbol: str = "",
    data_source: str = "",
) -> None:
    timestamp = pd.Timestamp.utcnow().isoformat()
    record = EventRecord(
        timestamp=timestamp,
        visibility_time=timestamp,
        source="policy_lab_ui",
        confidence=1.0,
        event_type=EventType.POLICY,
        payload={
            "title": f"Policy scenario: {selected_title}",
            "policy_text": str(policy_text),
            "intensity": float(intensity),
            "duration_days": int(duration_days),
            "rumor_noise": bool(rumor_noise),
            "index_symbol": str(index_symbol or ""),
            "data_source": str(data_source or ""),
        },
        metadata={"module": "policy_lab"},
    )
    try:
        EventStore().append_events(dataset_version="policy_lab", events=[record])
    except Exception:
        # Event persistence is best-effort and should never block the demo flow.
        return


def _render_agent_disagreement_chart(package_dict: Dict[str, Any]) -> None:
    agent_effects = dict(package_dict.get("agent_class_effects", {}) or {})
    if not agent_effects:
        return
    frame = (
        pd.DataFrame({"agent": list(agent_effects.keys()), "score": [float(v) for v in agent_effects.values()]})
        .sort_values("score", key=lambda series: series.abs(), ascending=False)
        .head(12)
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=frame["agent"],
                y=frame["score"],
                marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in frame["score"]],
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title="投资者政策解读分歧",
        xaxis_title="角色",
        yaxis_title="影响得分",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_role_orderflow_waterfall(frame: pd.DataFrame, package_dict: Dict[str, Any]) -> None:
    if frame.empty:
        return
    agent_effects = dict(package_dict.get("agent_class_effects", {}) or {})
    if not agent_effects:
        return
    scale = float(frame["volume"].mean() / max(len(agent_effects), 1))
    rows: List[Dict[str, Any]] = []
    for role, score in agent_effects.items():
        rows.append({"role": str(role), "net_flow": float(score) * scale})
    flow_df = pd.DataFrame(rows).sort_values("net_flow")
    fig = go.Figure(
        go.Waterfall(
            x=flow_df["role"],
            y=flow_df["net_flow"],
            measure=["relative"] * len(flow_df),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#16a34a"}},
            decreasing={"marker": {"color": "#ef4444"}},
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="角色订单流拆解（估计）",
        yaxis_title="净买卖量（估计）",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_sector_rotation_heatmap(package_dict: Dict[str, Any]) -> None:
    sector_effects = dict(package_dict.get("sector_effects", {}) or {})
    if not sector_effects:
        return
    ordered = sorted(sector_effects.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:12]
    labels = [str(name) for name, _ in ordered]
    values = [float(value) for _, value in ordered]
    z = np.array([values])
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=labels,
                y=["sector_heat"],
                colorscale="RdYlGn",
                zmid=0.0,
                colorbar=dict(title="影响"),
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title="板块热度轮动（政策冲击）",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_regulation_counterfactual_panel(counterfactual: Dict[str, Any]) -> None:
    worlds = dict(counterfactual.get("worlds", {}) or {})
    no_df = pd.DataFrame(worlds.get("no_intervention", []) or [])
    early_df = pd.DataFrame(worlds.get("early_intervention", []) or [])
    late_df = pd.DataFrame(worlds.get("late_intervention", []) or [])
    if no_df.empty or early_df.empty or late_df.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=no_df["time"], y=no_df["close"], mode="lines", name="不介入", line=dict(color="#6b7280", width=2)))
    fig.add_trace(go.Scatter(x=early_df["time"], y=early_df["close"], mode="lines", name="提前介入", line=dict(color="#16a34a", width=2.6)))
    fig.add_trace(go.Scatter(x=late_df["time"], y=late_df["close"], mode="lines", name="延后介入", line=dict(color="#f97316", width=2.6)))
    fig.update_layout(
        template="plotly_white",
        title="监管时点反事实世界线（DEEP）",
        xaxis_title="时间",
        yaxis_title="价格",
        margin=dict(l=20, r=20, t=40, b=20),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

    scorecards = dict(counterfactual.get("scorecards", {}) or {})
    rows: List[Dict[str, Any]] = []
    for world_name, metrics in scorecards.items():
        rows.append(
            {
                "world": world_name,
                "return_pct": float(metrics.get("return_pct", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "avg_panic": float(metrics.get("avg_panic", 0.0)),
                "max_panic": float(metrics.get("max_panic", 0.0)),
                "volatility": float(metrics.get("volatility", 0.0)),
            }
        )
    if rows:
        score_df = pd.DataFrame(rows)
        st.dataframe(score_df, use_container_width=True, hide_index=True)
    recommended = str(counterfactual.get("recommended_timing", ""))
    steps = dict(counterfactual.get("intervention_steps", {}) or {})
    st.caption(
        f"推荐时点：{recommended} | 提前介入步={steps.get('early', '-')}, 延后介入步={steps.get('late', '-')}"
    )


def render_policy_lab() -> None:
    st.subheader("政策实验台")
    st.caption("仅供教学科研与仿真，不构成投资建议。")
    runtime_profile = _resolve_runtime_profile()
    st.caption(f"当前模式：{runtime_profile.label} | {runtime_profile.summary}")

    templates = _load_policy_templates()
    template_map = {str(item.get("title", f"template-{idx}")): item for idx, item in enumerate(templates)}
    selected_title = st.selectbox("模板", options=list(template_map.keys()), index=0)
    selected = template_map[selected_title]

    policy_text = st.text_area("政策文本", value=str(selected.get("policy_text", "")), height=110)
    intensity = st.slider("政策强度", min_value=0.2, max_value=2.0, value=float(selected.get("recommended_intensity", 1.0)), step=0.1)
    duration_days = st.slider("持续天数", min_value=10, max_value=180, value=int(selected.get("recommended_duration", 30)), step=5)
    rumor_noise = st.checkbox("注入传言噪声", value=bool(selected.get("default_rumor_noise", False)))
    index_label = st.selectbox("真实指数基准", options=list(INDEX_BENCHMARK_OPTIONS.keys()), index=0)
    history_window_days = st.slider("真实基准回看天数", min_value=60, max_value=360, value=180, step=20)

    if st.button("运行政策场景", type="primary"):
        with st.spinner("正在运行政策仿真..."):
            index_symbol = INDEX_BENCHMARK_OPTIONS[index_label]
            open_anchor = _policy_anchor_date()
            history_end = open_anchor - pd.Timedelta(days=1)
            real_history = _load_real_index_history(index_symbol, max(int(history_window_days), int(duration_days)), history_end)
            if real_history.empty:
                st.error("未获取到真实指数数据，请检查网络或数据源配置后重试。")
                return
            data_source = "real_index"
            deep_meta: Dict[str, Any] = {}
            if runtime_profile.mode == "DEEP":
                frame, deep_meta = _generate_policy_metrics_deep(
                    policy_text=policy_text,
                    intensity=float(intensity),
                    duration_days=int(duration_days),
                    rumor_noise=bool(rumor_noise),
                    runtime_profile=runtime_profile,
                    end_date=history_end,
                )
                data_source = "deep_mode_simulation"
            else:
                frame = _generate_policy_metrics(
                    policy_text=policy_text,
                    intensity=float(intensity),
                    duration_days=int(duration_days),
                    rumor_noise=bool(rumor_noise),
                    scenario_key=str(selected.get("id", selected_title)),
                    market_history=real_history,
                    end_date=history_end,
                )
            summary = _compute_policy_summary(frame)
            st.session_state.policy_lab_result = {
                "frame": frame,
                "summary": summary,
                "policy_text": policy_text,
                "template": selected,
                "index_label": index_label,
                "index_symbol": index_symbol,
                "data_source": data_source,
                "history_end": history_end.strftime("%Y-%m-%d"),
                "runtime_mode": runtime_profile.mode,
                "runtime_profile": runtime_profile.to_dict(),
                "deep_mode_meta": deep_meta,
            }
            _, package = _compile_policy_bundle(policy_text, float(intensity), policy_type_hint=str(selected.get("policy_type", "")))
            package_dict = package.to_dict()
            st.session_state.policy_lab_result["policy_package"] = package_dict

            report_payload = {
                "title": f"政策实验台 - {selected_title}",
                "summary": summary,
                "policy_text": policy_text,
                "metrics": frame.to_dict(orient="records"),
                "template": selected,
                "policy_schema": package_dict.get("policy_schema", {}),
                "transmission_graph": package_dict.get("transmission_graph", {}),
                "why_this_happened": package_dict.get("explanation", {}),
                "index_symbol": index_symbol,
                "data_source": data_source,
                "history_end": history_end.strftime("%Y-%m-%d"),
                "runtime_mode": runtime_profile.mode,
                "runtime_profile": runtime_profile.to_dict(),
                "deep_mode_meta": deep_meta,
            }
            report_title = f"政策实验台 - {selected_title}"
            report_meta = official_report_meta("policy_lab", report_title)
            report_payload["report_meta"] = report_meta
            metric_start = str(frame.iloc[0]["time"]) if not frame.empty else ""
            metric_end = str(frame.iloc[-1]["time"]) if not frame.empty else ""
            markdown_text = "\n".join(
                [
                    f"# {report_title}",
                    "",
                    f"- 报告编号：{report_meta['report_no']}",
                    f"- 生成日期：{report_meta['date_cn']}",
                    f"- 模板：{selected_title}",
                    f"- 政策强度：{float(intensity):.1f}",
                    f"- 持续天数：{int(duration_days)}",
                    f"- 传言噪声：{'是' if rumor_noise else '否'}",
                    f"- 指数基准：{index_label}（{index_symbol}）",
                    f"- 运行模式：{runtime_profile.mode}",
                    f"- 数据来源：{'真实指数' if data_source == 'real_index' else ('深度模式多智能体仿真' if data_source == 'deep_mode_simulation' else '仿真回退')}",
                    f"- 数据区间：{metric_start} 至 {metric_end}",
                    f"- 截止日期：{history_end.strftime('%Y-%m-%d')}（软件开启前一日）",
                    "",
                    "## 核心指标",
                    f"- 收益率：{summary['return_pct']:.2%}",
                    f"- 平均恐慌度：{summary['avg_panic']:.4f}",
                    f"- 最大回撤：{summary['max_drawdown']:.2%}",
                    f"- 波动率：{summary['volatility']:.4f}",
                    "",
                    "## 政策文本",
                    policy_text,
                ]
            )
            bundle = write_report_artifacts(
                root_dir=POLICY_REPORT_DIR,
                report_type="policy_lab",
                title=report_title,
                markdown_text=markdown_text,
                payload=report_payload,
            )
            st.session_state.policy_lab_bundle = bundle

            _persist_policy_event(
                selected_title=selected_title,
                policy_text=policy_text,
                intensity=float(intensity),
                duration_days=int(duration_days),
                rumor_noise=bool(rumor_noise),
                index_symbol=index_symbol,
                data_source=data_source,
            )

    result = st.session_state.get("policy_lab_result")
    if not result:
        st.info("运行一个场景后，这里会生成政策传导结果和报告材料。")
        return

    frame = result["frame"]
    summary = result["summary"]

    cols = st.columns(4)
    cols[0].metric("收益率", f"{summary['return_pct'] * 100:.2f}%")
    cols[1].metric("平均恐慌度", f"{summary['avg_panic']:.3f}")
    cols[2].metric("最大回撤", f"{summary['max_drawdown'] * 100:.2f}%")
    cols[3].metric("波动率", f"{summary['volatility']:.4f}")

    index_label = str(result.get("index_label", "指数基准"))
    index_symbol = str(result.get("index_symbol", ""))
    data_source = str(result.get("data_source", "synthetic_fallback"))
    if data_source == "real_index":
        source_text = "真实指数数据"
    elif data_source == "deep_mode_simulation":
        source_text = "深度模式多智能体仿真"
    else:
        source_text = "仿真回退数据"
    history_end = str(result.get("history_end", ""))
    chart_mode = st.radio("图表模式", options=CHART_MODE_OPTIONS, horizontal=True, index=0, key="policy_lab_chart_mode")
    if not frame.empty:
        _render_quote_banner(
            frame,
            index_label=index_label,
            index_symbol=index_symbol,
            source_text=source_text,
            history_end=history_end,
        )
        st.caption(f"区间：{frame.iloc[0]['time']} 至 {frame.iloc[-1]['time']}")
    st.plotly_chart(
        _build_chart(frame, chart_title=f"{index_label} {chart_mode}（东方财富风格）", mode=chart_mode),
        use_container_width=True,
    )

    package_dict = result.get("policy_package") or {}
    result_runtime_profile = resolve_runtime_mode_profile(str(result.get("runtime_mode", "SMART")))
    if package_dict:
        c1, c2 = st.columns(2)
        with c1:
            _render_agent_disagreement_chart(package_dict)
        with c2:
            _render_role_orderflow_waterfall(frame, package_dict)
        _render_sector_rotation_heatmap(package_dict)
        st.markdown("#### 成因解读（自然语言）")
        st.markdown(
            _get_policy_narrative(
                str(result.get("policy_text", "")),
                summary,
                package_dict,
                result_runtime_profile,
            )
        )

    deep_mode_meta = result.get("deep_mode_meta") or {}
    if isinstance(deep_mode_meta, dict) and deep_mode_meta:
        st.markdown("#### 深度模式诊断")
        stats = dict(deep_mode_meta.get("thinking_stats", {}) or {})
        llm_agent_count = int(deep_mode_meta.get("llm_agent_count", 0) or 0)
        metric_cols = st.columns(3)
        metric_cols[0].metric("LLM主驱动代理数", f"{llm_agent_count}")
        metric_cols[1].metric("快思考次数", f"{int(stats.get('fast_count', 0) or 0)}")
        metric_cols[2].metric("慢思考次数", f"{int(stats.get('slow_count', 0) or 0)}")
        committee_report = dict(deep_mode_meta.get("committee_report", {}) or {})
        if committee_report:
            with st.expander("对抗式防幻觉委员会输出", expanded=False):
                st.json(committee_report)
        counterfactual = dict(deep_mode_meta.get("counterfactual_regulation", {}) or {})
        if counterfactual:
            _render_regulation_counterfactual_panel(counterfactual)

    bundle = st.session_state.get("policy_lab_bundle")
    if bundle:
        st.caption(f"报告已导出：{bundle.get('json_path')}")
