"""Policy lab page focused on government-facing policy experiments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
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
from core.policy_session import PolicySession
from core.runtime_mode import RuntimeModeProfile, resolve_runtime_mode_profile
from policy.structured import PolicyPackage
from ui.reporting import official_report_meta, write_report_artifacts


POLICY_TYPE_OPTIONS = {
    "税制调整": "tax",
    "流动性投放": "liquidity",
    "财政刺激": "fiscal",
    "监管收紧": "tightening",
    "市场稳定": "stabilization",
    "自定义政策": "custom",
}

TEMPLATE_LIBRARY_PATH = Path("data") / "policy_templates.json"
POLICY_REPORT_DIR = Path("outputs") / "policy_reports"
CONTROL_MODE_OPTIONS = [
    "不设置对照组",
    "无政策基线",
    "模板推荐对照组",
    "温和变体",
    "风险压力变体",
]
INDEX_BENCHMARK_OPTIONS = {
    "上证指数（000001）": "sh000001",
    "深证成指（399001）": "sz399001",
    "创业板指（399006）": "sz399006",
}
CHART_MODE_OPTIONS = ["日K", "分时"]
POLICY_SESSION_STATUS_LABELS = {
    "idle": "未开始",
    "running": "仿真中",
    "paused": "已暂停",
    "stopped": "已停止",
    "completed": "已完成",
}


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
            "category": "市场稳定",
            "title": "下调印花税并配套流动性支持",
            "policy_type": "税制调整",
            "policy_text": "下调印花税，并配合流动性支持政策，稳定市场预期。",
            "policy_goal": "提升市场流动性，降低交易摩擦，稳定指数走势。",
            "suitable_departments": "财政、税务、证券监管、平准基金",
            "recommended_intensity": 1.1,
            "recommended_duration": 30,
            "default_rumor_noise": False,
            "control_label": "维持现有税率与流动性安排",
            "control_text": "不新增稳定市场措施，保持当前税制与流动性安排。",
        },
        {
            "id": "targeted-fiscal-demand",
            "category": "财政支持",
            "title": "面向重点行业的定向财政扩张",
            "policy_type": "财政刺激",
            "policy_text": "分阶段推出面向基建和先进制造业的定向财政支出计划。",
            "policy_goal": "在维护金融稳定的同时，稳定增长预期。",
            "suitable_departments": "财政、发改、工信、地方政府",
            "recommended_intensity": 1.0,
            "recommended_duration": 60,
            "default_rumor_noise": False,
            "control_label": "不实施定向财政扩张",
            "control_text": "保持当前财政取向不变，作为对照组。",
        },
        {
            "id": "rumor-refutation-stabilization",
            "category": "预期管理",
            "title": "辟谣澄清并同步发布稳市声明",
            "policy_type": "市场稳定",
            "policy_text": "发布官方澄清公告，辟谣市场传闻，并推出协同稳市沟通方案。",
            "policy_goal": "降低市场恐慌，抑制谣言驱动的抛压。",
            "suitable_departments": "监管机构、官方媒体、交易所、稳定基金",
            "recommended_intensity": 1.2,
            "recommended_duration": 20,
            "default_rumor_noise": True,
            "control_label": "不进行公开澄清",
            "control_text": "不发布官方回应，仅观察市场自然演化。",
        },
    ]


def _runtime_mode_text(runtime_profile: RuntimeModeProfile) -> str:
    mode_map = {
        "SMART": "标准推演模式",
        "DEEP": "深度多智能体模式",
        "LIVE": "实时演示模式",
        "DEMO": "答辩演示模式",
    }
    mode = str(getattr(runtime_profile, "mode", "") or "").upper()
    label = mode_map.get(mode, mode or "未知模式")
    summary = str(getattr(runtime_profile, "summary", "") or "").strip()
    return f"{label}｜{summary}" if summary else label


def _data_source_text(data_source: str) -> str:
    mapping = {
        "real_index": "真实指数数据",
        "deep_mode_simulation": "深度模式多智能体仿真",
        "synthetic_fallback": "仿真回退数据",
    }
    return mapping.get(str(data_source or ""), "仿真回退数据")


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


def _build_policy_lab_agents(runtime_profile: RuntimeModeProfile) -> List[Any]:
    from agents.persona import Persona
    from agents.trader_agent import TraderAgent

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
    return agents


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
    from engine.simulation_loop import MarketEnvironment

    symbol = "A_SHARE_IDX"
    agents = _build_policy_lab_agents(runtime_profile)

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


def _policy_session_status_text(status: str) -> str:
    return POLICY_SESSION_STATUS_LABELS.get(str(status or "").strip().lower(), "未开始")


def _policy_session_reference_profile(reference_frame: Optional[pd.DataFrame]) -> Dict[str, Any]:
    fallback = {
        "start_close": 3000.0,
        "start_volume": 1_000_000.0,
        "mean_return": 0.0003,
        "volatility": 0.0080,
        "base_panic": 0.18,
        "base_csad": 0.06,
        "history_end": "",
    }
    if reference_frame is None or reference_frame.empty:
        return fallback

    frame = reference_frame.copy()
    close = pd.to_numeric(frame.get("close"), errors="coerce").dropna() if "close" in frame else pd.Series(dtype=float)
    if close.empty:
        return fallback
    returns = close.pct_change().dropna()
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").dropna() if "volume" in frame else pd.Series(dtype=float)
    avg_abs_return = float(np.mean(np.abs(returns))) if not returns.empty else 0.0015
    history_end = str(frame.iloc[-1].get("time", "")) if "time" in frame and not frame.empty else ""
    return {
        "start_close": float(close.iloc[-1]),
        "start_volume": float(volume.tail(10).mean()) if not volume.empty else 1_000_000.0,
        "mean_return": float(returns.mean()) if not returns.empty else fallback["mean_return"],
        "volatility": float(max(returns.std() if not returns.empty else fallback["volatility"], 0.003)),
        "base_panic": float(np.clip(0.12 + avg_abs_return * 40.0, 0.05, 0.35)),
        "base_csad": float(np.clip(0.05 + avg_abs_return * 18.0, 0.03, 0.16)),
        "history_end": history_end,
    }


def _policy_session_bday(calendar_start: str, day_number: int) -> str:
    start = pd.Timestamp(calendar_start).normalize()
    return pd.bdate_range(start=start, periods=max(1, int(day_number)))[-1].strftime("%Y-%m-%d")


def _policy_session_effect_score(policy_text: str, intensity: float) -> Tuple[float, PolicyShock]:
    shock = _compile_scaled_shock(policy_text, intensity)
    return _shock_score(shock), shock


def _policy_session_build_event(
    *,
    policy_name: str,
    policy_text: str,
    policy_type: str,
    intensity: float,
    effective_day: int,
    half_life_days: int,
    created_day: int,
    rumor_noise: bool,
) -> Dict[str, Any]:
    score, shock = _policy_session_effect_score(policy_text, intensity)
    _, package = _compile_policy_bundle(
        policy_text,
        float(intensity),
        policy_type_hint=policy_type,
    )
    return {
        "policy_id": package.event.policy_id,
        "policy_name": str(policy_name or "未命名政策"),
        "policy_text": str(policy_text or ""),
        "policy_type": str(policy_type or "自定义政策"),
        "intensity": float(intensity),
        "effective_day": max(1, int(effective_day)),
        "half_life_days": max(1, int(half_life_days)),
        "created_day": max(0, int(created_day)),
        "rumor_noise": bool(rumor_noise),
        "score": float(score),
        "shock": shock.to_dict(),
        "package": package.to_dict(),
        "remaining_effect": 0.0,
        "status": "queued",
    }


def _policy_session_empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "step",
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "panic_level",
            "csad",
            "政策压力",
            "活跃政策数",
            "待生效政策数",
            "买入量",
            "卖出量",
            "散户净流",
            "机构净流",
            "量化净流",
            "做市净流",
        ]
    )


def _policy_session_timeline_item_from_runner(item: Dict[str, Any]) -> Dict[str, Any]:
    state_raw = str(item.get("当前状态", "") or "").strip().lower()
    state_map = {
        "queued": "待生效",
        "active": "生效中",
        "fading": "影响减弱",
        "exhausted": "已淡出",
    }
    base_strength = float(item.get("基础强度", 0.0) or 0.0)
    current_strength = float(item.get("当前强度", 0.0) or 0.0)
    remaining = 0.0 if base_strength <= 0 else float(np.clip(current_strength / max(base_strength, 1e-9), 0.0, 1.0))
    name = str(item.get("政策标签", item.get("政策文本", "未命名政策")))
    policy_type = str(dict(item.get("元数据", {}) or {}).get("policy_type", "自定义政策"))
    policy_text = str(item.get("政策文本", ""))
    return {
        "政策名称": str(item.get("政策标签", item.get("政策文本", "未命名政策"))),
        "政策类型": policy_type,
        "生效交易日": int(item.get("生效日", 1) or 1),
        "当前状态": state_map.get(state_raw, state_raw or "待生效"),
        "初始强度": base_strength,
        "剩余影响": remaining,
        "传言噪声": "是" if bool(item.get("是否谣言噪声", False)) else "否",
        "政策文本": policy_text,
        "status": state_raw,
        "policy_name": name,
        "policy_type": policy_type,
        "policy_text": policy_text,
        "remaining_effect": remaining,
    }


def _policy_session_sync_from_runner(session: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    runner = session.get("_runner")
    if not isinstance(runner, PolicySession):
        return session

    if snapshot is None:
        snapshot = runner.advance(0)

    raw_frame = snapshot.get("frame", pd.DataFrame())
    if not isinstance(raw_frame, pd.DataFrame):
        raw_frame = pd.DataFrame(raw_frame or [])
    display_frame = _session_frame_to_market_frame(raw_frame, anchor_close=float(session.get("index_anchor_close", 0.0)))
    active_items = snapshot.get("active_policies", []) if isinstance(snapshot.get("active_policies"), list) else []
    queued_items = snapshot.get("queued_policies", []) if isinstance(snapshot.get("queued_policies"), list) else []
    policy_events = [_policy_session_timeline_item_from_runner(item) for item in [*active_items, *queued_items]]
    summary_payload = snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
    session["status"] = str(snapshot.get("status", session.get("status", "idle")))
    session["current_day"] = int(snapshot.get("current_day", session.get("current_day", 0)))
    session["frame_rows"] = display_frame.to_dict(orient="records")
    session["raw_frame_rows"] = raw_frame.to_dict(orient="records")
    session["policy_events"] = policy_events
    session["policy_timeline"] = policy_events
    session["latest_step_report"] = dict(snapshot.get("last_step_report", {}) or {})
    display_latest_close = float(display_frame["close"].iloc[-1]) if not display_frame.empty else float(session.get("last_close", 0.0) or 0.0)
    session["last_close"] = display_latest_close
    session["summary"] = {
        "return_pct": float(summary_payload.get("累计收益率", 0.0) or 0.0),
        "max_drawdown": float(summary_payload.get("最大回撤", 0.0) or 0.0),
        "avg_panic": float(display_frame["panic_level"].mean()) if not display_frame.empty else 0.0,
        "max_panic": float(display_frame["panic_level"].max()) if not display_frame.empty else 0.0,
        "volatility": float(display_frame["close"].pct_change().fillna(0.0).std()) if len(display_frame) > 1 else 0.0,
        "avg_csad": float(display_frame["csad"].mean()) if not display_frame.empty else 0.0,
        "avg_volume": float(display_frame["volume"].mean()) if not display_frame.empty else 0.0,
        "policy_signal_avg": float(len(active_items)),
        "active_policy_max": float(len(active_items)),
        "latest_close": display_latest_close,
        "最新收盘价": display_latest_close,
        "累计收益率": float(summary_payload.get("累计收益率", 0.0) or 0.0),
    }
    session["_snapshot"] = snapshot
    return session


def _policy_session_backdrop_rows(reference_frame: Optional[pd.DataFrame], total_days: int) -> List[Dict[str, float]]:
    if reference_frame is None or reference_frame.empty:
        return []
    frame = reference_frame.copy().tail(max(int(total_days), 10)).reset_index(drop=True)
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    rows: List[Dict[str, float]] = []
    for idx in range(len(frame)):
        price = float(close.iloc[idx]) if idx < len(close) and pd.notna(close.iloc[idx]) else 0.0
        if price <= 0.0:
            continue
        rows.append(
            {
                "step": float(idx + 1),
                "price": price,
                "close": price,
                "volume": float(volume.iloc[idx]) if idx < len(volume) and pd.notna(volume.iloc[idx]) else 0.0,
            }
        )
    return rows


def _policy_session_new(
    *,
    policy_name: str,
    policy_text: str,
    policy_type: str,
    total_days: int,
    intensity: float,
    effective_day: int,
    half_life_days: int,
    rumor_noise: bool,
    index_label: str,
    index_symbol: str,
    reference_frame: Optional[pd.DataFrame],
    runtime_profile: RuntimeModeProfile,
) -> Dict[str, Any]:
    reference = _policy_session_reference_profile(reference_frame)
    history_end = reference.get("history_end") or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    calendar_start = pd.bdate_range(start=pd.Timestamp(history_end).normalize(), periods=2)[-1].strftime("%Y-%m-%d")
    agents = _build_policy_lab_agents(runtime_profile)
    backdrop_rows = _policy_session_backdrop_rows(reference_frame, max(1, int(total_days)))
    use_backdrop = bool(backdrop_rows) and str(runtime_profile.mode or "SMART").strip().upper() == "SMART"
    runner = PolicySession.create(
        agents=agents,
        total_days=max(1, int(total_days)),
        base_policy="",
        start_date=calendar_start,
        half_life_days=max(1, int(half_life_days)),
        enable_random_policy_events=False,
        simulation_mode=runtime_profile.mode,
        use_isolated_matching=True,
        market_pipeline_v2=bool(runtime_profile.market_pipeline_v2),
        llm_primary=bool(runtime_profile.llm_primary),
        deep_reasoning_pause_s=float(runtime_profile.pause_for_llm_seconds),
        enable_policy_committee=bool(runtime_profile.enable_policy_committee),
        runner_symbol="A_SHARE_IDX",
        steps_per_day=1,
        model_priority=list(runtime_profile.model_priority),
        hybrid_replay=use_backdrop,
        exogenous_backdrop=backdrop_rows if use_backdrop else None,
        hybrid_backdrop_weight=0.82 if use_backdrop else 0.0,
    )
    runner.enqueue_policy(
        policy_text,
        effective_day=max(1, int(effective_day)),
        strength=float(intensity),
        half_life_days=max(1, int(half_life_days)),
        rumor_noise=bool(rumor_noise),
        label=str(policy_name or "基础政策"),
        source="base_policy",
        metadata={"policy_type": str(policy_type or "自定义政策")},
    )
    session = {
        "session_id": f"policy-session-{_seed_from_text(policy_name + policy_text + str(total_days)) & 0xFFFFFFFF:08x}",
        "status": "idle",
        "total_days": max(1, int(total_days)),
        "current_day": 0,
        "calendar_start": calendar_start,
        "index_label": str(index_label),
        "index_symbol": str(index_symbol),
        "policy_name": str(policy_name),
        "policy_text": str(policy_text),
        "policy_type": str(policy_type or "自定义政策"),
        "intensity": float(intensity),
        "effective_day": max(1, int(effective_day)),
        "half_life_days": max(1, int(half_life_days)),
        "rumor_noise": bool(rumor_noise),
        "policy_events": [],
        "frame_rows": [],
        "summary": {},
        "report_bundle": None,
        "report_payload": None,
        "reference_profile": reference,
        "runtime_profile": runtime_profile.to_dict(),
        "mode_text": _runtime_mode_text(runtime_profile),
        "last_close": float(reference["start_close"]),
        "last_return": float(reference["mean_return"]),
        "last_panic": float(reference["base_panic"]),
        "last_csad": float(reference["base_csad"]),
        "last_volume": float(reference["start_volume"]),
        "policy_package": None,
        "policy_timeline": [],
        "index_anchor_close": float(reference["start_close"]),
        "autoplay": {
            "enabled": False,
            "step_days": 1,
            "interval_seconds": 0.8,
            "last_wallclock_ts": 0.0,
        },
        "latest_step_report": {},
        "_runner": runner,
        "llm_agent_count": int(sum(1 for agent in agents if bool(getattr(agent, "use_llm", False)))),
    }
    timeline = [plan.to_timeline_row(0) for plan in runner.policies]
    session["policy_events"] = [_policy_session_timeline_item_from_runner(item) for item in timeline]
    session["policy_timeline"] = list(session["policy_events"])
    return session


def _policy_session_autoplay_state(session: Dict[str, Any]) -> Dict[str, Any]:
    autoplay = session.setdefault(
        "autoplay",
        {
            "enabled": False,
            "step_days": 1,
            "interval_seconds": 0.8,
            "last_wallclock_ts": 0.0,
        },
    )
    autoplay["enabled"] = bool(autoplay.get("enabled", False))
    autoplay["step_days"] = max(1, int(autoplay.get("step_days", 1) or 1))
    autoplay["interval_seconds"] = max(0.0, float(autoplay.get("interval_seconds", 0.8) or 0.0))
    autoplay["last_wallclock_ts"] = float(autoplay.get("last_wallclock_ts", 0.0) or 0.0)
    return autoplay


def _policy_session_disable_autoplay(session: Dict[str, Any]) -> None:
    autoplay = _policy_session_autoplay_state(session)
    autoplay["enabled"] = False


def _policy_session_maybe_autoplay(
    session: Dict[str, Any],
    *,
    now_ts: Optional[float] = None,
    min_interval_seconds: Optional[float] = None,
) -> bool:
    autoplay = _policy_session_autoplay_state(session)
    if not autoplay.get("enabled", False):
        return False

    status = str(session.get("status", "idle")).lower()
    if status not in {"running", "paused"}:
        autoplay["enabled"] = False
        return False

    total_days = int(session.get("total_days", 0) or 0)
    current_day = int(session.get("current_day", 0) or 0)
    if total_days > 0 and current_day >= total_days:
        session["status"] = "completed"
        autoplay["enabled"] = False
        return False

    interval_seconds = float(
        autoplay.get("interval_seconds", 0.8) if min_interval_seconds is None else min_interval_seconds
    )
    now_value = float(time.time() if now_ts is None else now_ts)
    last_ts = float(autoplay.get("last_wallclock_ts", 0.0) or 0.0)
    if last_ts > 0.0 and now_value - last_ts < interval_seconds:
        return False

    before = int(session.get("current_day", 0) or 0)
    _policy_session_advance(session, int(autoplay.get("step_days", 1) or 1))
    autoplay["last_wallclock_ts"] = now_value
    if int(session.get("current_day", 0) or 0) >= int(session.get("total_days", 0) or 0):
        autoplay["enabled"] = False
    return int(session.get("current_day", 0) or 0) > before


def _build_policy_demo_cards(
    *,
    policy_text: str,
    latest_step_report: Dict[str, Any],
    session_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    summary = dict(session_summary or {})
    latest = dict(latest_step_report or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    policy_signal = dict(chain.get("policy_signal", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    index_move = dict(chain.get("index_move", {}) or {})

    signal_strength = float(policy_signal.get("strength", summary.get("policy_signal_avg", 0.0)) or 0.0)
    social_mean = float(sentiment.get("social_mean", 0.0) or 0.0)
    buy_volume = float(order_flow.get("buy_volume", 0.0) or 0.0)
    sell_volume = float(order_flow.get("sell_volume", 0.0) or 0.0)
    trade_count = int(matching.get("trade_count", latest.get("trade_count", 0)) or 0)
    latest_close = float(index_move.get("new_price", summary.get("最新收盘价", summary.get("latest_close", 0.0))) or 0.0)
    return_pct = float(index_move.get("return_pct", latest.get("price_change_pct", 0.0)) or 0.0)

    return [
        {
            "phase": "政策注入",
            "summary": (str(policy_signal.get("policy_text", "") or policy_text).strip() or "等待政策输入")[:72],
            "detail": f"信号强度 {signal_strength:+.2f}｜来源 {str(policy_signal.get('source', 'policy_session'))}",
        },
        {
            "phase": "情绪扩散",
            "summary": f"情绪均值 {social_mean:+.2f}，多智能体开始重估政策影响。",
            "detail": f"买量 {buy_volume:.0f}｜卖量 {sell_volume:.0f}",
        },
        {
            "phase": "撮合落地",
            "summary": f"最新点位 {latest_close:.2f}，单日变化 {return_pct:+.2f}%。",
            "detail": f"成交 {trade_count} 笔｜撮合模式 {str(matching.get('matching_mode', latest.get('matching_mode', 'session')))}",
        },
    ]


def _build_policy_demo_briefing(session: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = dict(summary or {})
    latest = dict(session.get("latest_step_report", {}) or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    policy_signal = dict(chain.get("policy_signal", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    index_move = dict(chain.get("index_move", {}) or {})

    current_day = int(session.get("current_day", 0) or 0)
    total_days = int(session.get("total_days", 0) or 0)
    signal_strength = float(policy_signal.get("strength", summary.get("policy_signal_avg", 0.0)) or 0.0)
    panic_level = float(sentiment.get("panic_level", summary.get("max_panic", 0.0)) or 0.0)
    return_pct = float(index_move.get("return_pct", latest.get("price_change_pct", 0.0)) or 0.0)
    trade_count = int(matching.get("trade_count", latest.get("trade_count", 0)) or 0)
    imbalance = float(order_flow.get("imbalance", 0.0) or 0.0)
    autoplay = dict(session.get("autoplay", {}) or {})

    if current_day <= 1 and trade_count <= 0:
        phase = "政策刚注入，预期正在形成"
    elif panic_level >= 0.55 or abs(imbalance) >= 600:
        phase = "情绪扩散加速，订单簿开始失衡"
    elif trade_count > 0:
        phase = "撮合落地，市场正在重估政策价格"
    else:
        phase = "会话推进中，继续观察资金与情绪传导"

    if panic_level >= 0.65 or return_pct <= -1.2:
        tone = "risk"
        alert = "高波动预警"
    elif abs(return_pct) >= 0.8 or abs(imbalance) >= 400:
        tone = "watch"
        alert = "市场异动提示"
    else:
        tone = "calm"
        alert = "稳态观察"

    bullets = [
        f"交易日进度 {current_day}/{total_days}，当前模式 {str(session.get('mode_text', '会话仿真'))}",
        f"政策强度 {signal_strength:+.2f}，活跃政策 {sum(1 for item in session.get('policy_events', []) if item.get('status') == 'active')} 项",
        f"买量 {float(order_flow.get('buy_volume', 0.0) or 0.0):.0f} / 卖量 {float(order_flow.get('sell_volume', 0.0) or 0.0):.0f}，成交 {trade_count} 笔",
    ]
    chips = [
        f"状态：{_policy_session_status_text(str(session.get('status', 'idle')))}",
        f"自动演示：{'开启' if bool(autoplay.get('enabled', False)) else '关闭'}",
        f"最新收益：{summary.get('return_pct', 0.0):+.2%}",
        f"恐慌度：{panic_level:.2f}",
    ]
    subtitle = str(policy_signal.get("policy_text", "") or session.get("policy_text", "")).strip()
    if len(subtitle) > 88:
        subtitle = subtitle[:88] + "…"
    return {
        "kicker": "Policy Wind Tunnel",
        "phase": phase,
        "subtitle": subtitle or "等待政策输入后启动会话仿真。",
        "alert": alert,
        "tone": tone,
        "bullets": bullets,
        "chips": chips,
    }


def _build_policy_market_pulse(
    session: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    summary = dict(summary or {})
    latest = dict(session.get("latest_step_report", {}) or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    index_move = dict(chain.get("index_move", {}) or {})

    buy_volume = float(order_flow.get("buy_volume", 0.0) or 0.0)
    sell_volume = float(order_flow.get("sell_volume", 0.0) or 0.0)
    imbalance = buy_volume - sell_volume
    leader = "买盘主导" if imbalance >= 0 else "卖盘主导"
    pulse_color = "up" if imbalance >= 0 else "down"
    panic_level = float(sentiment.get("panic_level", summary.get("avg_panic", 0.0)) or 0.0)
    return_pct = float(index_move.get("return_pct", latest.get("price_change_pct", 0.0)) or 0.0)
    direction = "上行重估" if return_pct >= 0 else "下行重估"

    return [
        {
            "label": "主导资金",
            "value": leader,
            "note": f"净差 {imbalance:+.0f}",
            "tone": pulse_color,
        },
        {
            "label": "情绪热度",
            "value": f"{panic_level:.2f}",
            "note": "高于 0.60 需重点解释" if panic_level >= 0.60 else "仍在可控区间",
            "tone": "watch" if panic_level >= 0.60 else "calm",
        },
        {
            "label": "撮合反馈",
            "value": f"{int(matching.get('trade_count', latest.get('trade_count', 0)) or 0)} 笔",
            "note": f"模式 {str(matching.get('matching_mode', latest.get('matching_mode', 'session')))}",
            "tone": "watch",
        },
        {
            "label": "指数动作",
            "value": direction,
            "note": f"{return_pct:+.2f}%",
            "tone": "up" if return_pct >= 0 else "down",
        },
    ]


def _policy_session_effect_for_event(event: Dict[str, Any], day: int) -> float:
    effective_day = int(event.get("effective_day", 1))
    if day < effective_day:
        return 0.0
    half_life = max(1, int(event.get("half_life_days", 30)))
    age = max(0, int(day) - effective_day)
    decay = float(np.exp(-age / half_life))
    event["remaining_effect"] = float(decay)
    if decay >= 0.35:
        event["status"] = "active"
    elif decay >= 0.08:
        event["status"] = "fading"
    else:
        event["status"] = "exhausted"
    return float(event.get("score", 0.0)) * float(event.get("intensity", 1.0)) * decay


def _policy_session_row(session: Dict[str, Any], day: int) -> Dict[str, Any]:
    reference = dict(session.get("reference_profile", {}) or {})
    prev_close = float(session.get("last_close", reference.get("start_close", 3000.0)))
    prev_return = float(session.get("last_return", reference.get("mean_return", 0.0003)))
    prev_panic = float(session.get("last_panic", reference.get("base_panic", 0.18)))
    prev_csad = float(session.get("last_csad", reference.get("base_csad", 0.06)))
    prev_volume = float(session.get("last_volume", reference.get("start_volume", 1_000_000.0)))
    active_events = 0
    queued_events = 0
    policy_pressure = 0.0
    rumor_pressure = 0.0
    package_parts: List[str] = []
    for event in session.get("policy_events", []):
        package_parts.append(str(event.get("policy_text", "")))
        if day < int(event.get("effective_day", 1)):
            queued_events += 1
            continue
        active_events += 1
        if bool(event.get("rumor_noise", False)):
            rumor_pressure += 0.12 * float(event.get("intensity", 1.0))
        policy_pressure += _policy_session_effect_for_event(event, day)

    policy_signal = float(np.tanh(policy_pressure / 8.0))
    rumor_signal = float(np.tanh(rumor_pressure / 2.0))
    seed = _seed_from_text(
        f"{session.get('session_id', 'policy-session')}|{day}|{policy_signal:.4f}|{prev_close:.2f}|{len(package_parts)}"
    )
    rng = np.random.default_rng(seed)
    noise = float(rng.normal(0.0, max(float(reference.get("volatility", 0.0080)) * 0.35, 0.0015)))

    retail_flow = policy_signal * 0.55 + rumor_signal * 0.40 - prev_panic * 0.10 + noise * 8.0
    institution_flow = policy_signal * 0.42 + prev_return * 0.80 - prev_panic * 0.06 + noise * 4.0
    quant_flow = policy_signal * 0.28 - prev_return * 0.75 + noise * 3.0
    maker_flow = -abs(policy_signal) * 0.18 - prev_panic * 0.08 - noise * 2.0
    net_flow = retail_flow + institution_flow + quant_flow + maker_flow

    daily_return = (
        float(reference.get("mean_return", 0.0003))
        + 0.0045 * policy_signal
        + 0.0022 * net_flow
        - 0.0011 * prev_panic
        + noise
    )
    close = max(1.0, prev_close * (1.0 + daily_return))
    amplitude = max(abs(daily_return) * 1.8, float(reference.get("volatility", 0.0080)) * 1.5, 0.0025)
    high = max(prev_close, close) * (1.0 + amplitude * (0.55 + max(policy_signal, 0.0) * 0.25))
    low = min(prev_close, close) * (1.0 - amplitude * (0.55 + max(-policy_signal, 0.0) * 0.25))
    panic = float(
        np.clip(
            0.55 * prev_panic
            + max(0.0, -daily_return) * 12.0
            + abs(rumor_signal) * 0.20
            + abs(policy_signal) * 0.08,
            0.03,
            0.98,
        )
    )
    csad = float(np.clip(0.62 * prev_csad + abs(daily_return) * 4.8 + panic * 0.05, 0.03, 0.30))
    volume = float(prev_volume * (1.0 + abs(policy_signal) * 0.18 + panic * 0.22 + abs(net_flow) * 0.08))
    day_date = _policy_session_bday(session.get("calendar_start", pd.Timestamp.today().normalize().strftime("%Y-%m-%d")), day)
    return {
        "step": int(day),
        "time": day_date,
        "open": round(prev_close, 2),
        "high": round(high, 2),
        "low": round(max(0.01, low), 2),
        "close": round(close, 2),
        "volume": round(volume, 2),
        "panic_level": round(panic, 4),
        "csad": round(csad, 4),
        "政策压力": round(policy_signal, 4),
        "活跃政策数": int(active_events),
        "待生效政策数": int(queued_events),
        "买入量": round(max(0.0, retail_flow + institution_flow + quant_flow), 2),
        "卖出量": round(max(0.0, -maker_flow - min(net_flow, 0.0)), 2),
        "散户净流": round(retail_flow, 4),
        "机构净流": round(institution_flow, 4),
        "量化净流": round(quant_flow, 4),
        "做市净流": round(maker_flow, 4),
    }


def _policy_session_refresh_summary(session: Dict[str, Any]) -> Dict[str, float]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        _policy_session_sync_from_runner(session)
        return dict(session.get("summary", {}) or {})
    frame = _policy_session_frame(session)
    if frame.empty:
        summary = {
            "return_pct": 0.0,
            "max_drawdown": 0.0,
            "avg_panic": 0.0,
            "max_panic": 0.0,
            "volatility": 0.0,
            "avg_csad": 0.0,
            "avg_volume": 0.0,
            "policy_signal_avg": 0.0,
            "active_policy_max": 0.0,
        }
        session["summary"] = summary
        return summary
    summary = _compute_policy_summary(frame.rename(columns={"panic_level": "panic_level", "csad": "csad"}))
    summary["policy_signal_avg"] = float(frame["政策压力"].mean()) if "政策压力" in frame else 0.0
    summary["active_policy_max"] = float(frame["活跃政策数"].max()) if "活跃政策数" in frame else 0.0
    summary["latest_close"] = float(frame.iloc[-1]["close"])
    summary["最新收盘价"] = float(frame.iloc[-1]["close"])
    session["summary"] = summary
    return summary


def _policy_session_frame(session: Dict[str, Any]) -> pd.DataFrame:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        snapshot = session.get("_snapshot", {})
        raw_frame = snapshot.get("frame", pd.DataFrame()) if isinstance(snapshot, dict) else pd.DataFrame()
        if not isinstance(raw_frame, pd.DataFrame):
            raw_frame = pd.DataFrame(raw_frame or [])
        return _session_frame_to_market_frame(raw_frame, anchor_close=float(session.get("index_anchor_close", 0.0)))
    rows = list(session.get("frame_rows", []) or [])
    if not rows:
        return _policy_session_empty_frame()
    frame = pd.DataFrame(rows)
    return frame.sort_values("step").reset_index(drop=True)


def _policy_session_timeline(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        _policy_session_sync_from_runner(session)
        return list(session.get("policy_timeline", []) or [])
    current_day = int(session.get("current_day", 0))
    rows: List[Dict[str, Any]] = []
    for event in session.get("policy_events", []):
        effective_day = int(event.get("effective_day", 1))
        if current_day < effective_day:
            state = "待生效"
            remaining = 0.0
        else:
            age = current_day - effective_day
            half_life = max(1, int(event.get("half_life_days", 30)))
            remaining = float(np.exp(-age / half_life))
            if remaining >= 0.35:
                state = "生效中"
            elif remaining >= 0.08:
                state = "影响减弱"
            else:
                state = "已淡出"
        rows.append(
            {
                "政策名称": str(event.get("policy_name", "")),
                "政策类型": str(event.get("policy_type", "")),
                "生效交易日": effective_day,
                "当前状态": state,
                "初始强度": float(event.get("intensity", 0.0)),
                "剩余影响": round(remaining, 4),
                "传言噪声": "是" if bool(event.get("rumor_noise", False)) else "否",
            }
        )
    return rows


def _policy_session_enqueue(
    session: Dict[str, Any],
    *,
    policy_name: str,
    policy_text: str,
    policy_type: str,
    effective_day: int,
    intensity: float,
    half_life_days: int,
    rumor_noise: bool,
    ) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        runner.enqueue_policy(
            policy_text,
            effective_day=max(1, int(effective_day)),
            strength=float(intensity),
            half_life_days=float(half_life_days),
            rumor_noise=bool(rumor_noise),
            label=str(policy_name or "追加政策"),
            source="append_policy",
            metadata={"policy_type": str(policy_type or "自定义政策")},
        )
        _policy_session_sync_from_runner(session)
        event = next((item for item in session.get("policy_timeline", []) if item.get("政策名称") == str(policy_name or "追加政策")), None)
        return dict(event or {})

    created_day = int(session.get("current_day", 0))
    event = _policy_session_build_event(
        policy_name=policy_name,
        policy_text=policy_text,
        policy_type=policy_type,
        intensity=intensity,
        effective_day=max(1, int(effective_day)),
        half_life_days=half_life_days,
        created_day=created_day,
        rumor_noise=rumor_noise,
    )
    session.setdefault("policy_events", []).append(event)
    session["report_bundle"] = None
    session["report_payload"] = None
    session["policy_timeline"] = _policy_session_timeline(session)
    return event


def _policy_session_advance(session: Dict[str, Any], days: int) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        if str(session.get("status", "idle")) == "paused":
            runner.resume()
        snapshot = runner.advance(days=max(0, int(days)))
        _policy_session_sync_from_runner(session, snapshot)
        if str(session.get("status", "idle")).lower() == "completed":
            _policy_session_disable_autoplay(session)
        return session
    if str(session.get("status", "idle")) not in {"running", "paused"}:
        return session
    if str(session.get("status", "idle")) == "paused":
        session["status"] = "running"
    total_days = int(session.get("total_days", 100))
    steps = max(0, int(days))
    for _ in range(steps):
        current_day = int(session.get("current_day", 0))
        if current_day >= total_days:
            session["status"] = "completed"
            break
        next_day = current_day + 1
        row = _policy_session_row(session, next_day)
        session.setdefault("frame_rows", []).append(row)
        session["current_day"] = next_day
        session["last_close"] = float(row["close"])
        session["last_return"] = float((row["close"] - row["open"]) / max(row["open"], 1e-9))
        session["last_panic"] = float(row["panic_level"])
        session["last_csad"] = float(row["csad"])
        session["last_volume"] = float(row["volume"])
        session["latest_step_report"] = {
            "tick": next_day,
            "day_count": next_day,
            "buy_volume": float(row["买入量"]),
            "sell_volume": float(row["卖出量"]),
            "old_price": float(row["open"]),
            "new_price": float(row["close"]),
            "price_change_pct": float((row["close"] - row["open"]) / max(row["open"], 1e-9) * 100.0),
            "trade_count": int(max(row["买入量"], row["卖出量"]) // 100),
            "matching_mode": "session_fallback",
            "transmission_chain": {
                "policy_signal": {
                    "policy_text": str(session.get("policy_text", "")),
                    "strength": float(row["政策压力"]),
                    "source": "policy_session_fallback",
                },
                "agent_sentiment": {
                    "social_mean": float(0.5 - row["panic_level"]),
                    "panic_level": float(row["panic_level"]),
                    "committee_enabled": False,
                },
                "order_flow": {
                    "buy_volume": float(row["买入量"]),
                    "sell_volume": float(row["卖出量"]),
                    "imbalance": float(row["买入量"] - row["卖出量"]),
                },
                "matching_result": {
                    "trade_count": int(max(row["买入量"], row["卖出量"]) // 100),
                    "last_price": float(row["close"]),
                    "matching_mode": "session_fallback",
                },
                "index_move": {
                    "old_price": float(row["open"]),
                    "new_price": float(row["close"]),
                    "return_pct": float((row["close"] - row["open"]) / max(row["open"], 1e-9) * 100.0),
                },
            },
        }
        if next_day >= total_days:
            session["status"] = "completed"
            break
    if int(session.get("current_day", 0)) >= total_days:
        session["status"] = "completed"
        _policy_session_disable_autoplay(session)
    _policy_session_refresh_summary(session)
    session["policy_timeline"] = _policy_session_timeline(session)
    return session


def _policy_session_stop(session: Dict[str, Any]) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        runner.stop()
        snapshot = runner.advance(0)
        _policy_session_sync_from_runner(session, snapshot)
        _policy_session_disable_autoplay(session)
        return session
    if str(session.get("status", "idle")) in {"running", "paused"}:
        session["status"] = "stopped"
    _policy_session_disable_autoplay(session)
    return session


def _policy_session_report_payload(session: Dict[str, Any], runtime_profile: RuntimeModeProfile) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        payload = runner.build_report_payload()
        timeline = list(session.get("policy_timeline", []) or [])
        combined_policy_text = "\n".join(str(item.get("政策文本", "")) for item in timeline if str(item.get("政策文本", "")).strip())
        package_dict = _policy_session_policy_package(session)
        summary = dict(session.get("summary", {}) or {})
        narrative = _get_policy_narrative(
            combined_policy_text or str(session.get("policy_text", "")),
            summary,
            package_dict or {"explanation": {}, "policy_schema": {}, "top_layers": {}},
            runtime_profile,
        )
        return {
            "title": f"政策实验台 - {session.get('policy_name', '政策仿真')}",
            "summary": summary,
            "policy_name": session.get("policy_name", "政策仿真"),
            "policy_text": session.get("policy_text", ""),
            "policy_type": session.get("policy_type", "自定义政策"),
            "timeline": timeline,
            "frame": payload.get("日度结果", []),
            "policy_package": package_dict,
            "runtime_mode": runtime_profile.mode,
            "runtime_profile": runtime_profile.to_dict(),
            "narrative": narrative,
            "session": {
                "session_id": session.get("session_id", ""),
                "status": session.get("status", "idle"),
                "current_day": int(session.get("current_day", 0)),
                "total_days": int(session.get("total_days", 0)),
                "calendar_start": session.get("calendar_start", ""),
                "index_label": session.get("index_label", ""),
                "index_symbol": session.get("index_symbol", ""),
            },
            "runner_payload": payload,
        }
    frame = _policy_session_frame(session)
    summary = dict(session.get("summary", {}) or {}) or _policy_session_refresh_summary(session)
    combined_policy_text = "\n".join(
        str(event.get("policy_text", "")) for event in session.get("policy_events", []) if str(event.get("policy_text", "")).strip()
    )
    package_dict: Dict[str, Any] = {}
    if combined_policy_text.strip():
        _, package = _compile_policy_bundle(
            combined_policy_text,
            float(session.get("intensity", 1.0)),
            policy_type_hint=str(session.get("policy_type", "自定义政策")),
        )
        package_dict = package.to_dict()
        session["policy_package"] = package_dict
    narrative = _get_policy_narrative(
        combined_policy_text or str(session.get("policy_text", "")),
        summary,
        package_dict or {"explanation": {}, "policy_schema": {}, "top_layers": {}},
        runtime_profile,
    )
    return {
        "title": f"政策实验台 - {session.get('policy_name', '政策仿真')}",
        "summary": summary,
        "policy_name": session.get("policy_name", "政策仿真"),
        "policy_text": session.get("policy_text", ""),
        "policy_type": session.get("policy_type", "自定义政策"),
        "timeline": _policy_session_timeline(session),
        "frame": frame.to_dict(orient="records"),
        "policy_package": package_dict,
        "runtime_mode": runtime_profile.mode,
        "runtime_profile": runtime_profile.to_dict(),
        "narrative": narrative,
        "session": {
            "session_id": session.get("session_id", ""),
            "status": session.get("status", "idle"),
            "current_day": int(session.get("current_day", 0)),
            "total_days": int(session.get("total_days", 0)),
            "calendar_start": session.get("calendar_start", ""),
            "index_label": session.get("index_label", ""),
            "index_symbol": session.get("index_symbol", ""),
        },
    }


def _policy_session_generate_report(session: Dict[str, Any], runtime_profile: RuntimeModeProfile) -> Dict[str, Any]:
    runner = session.get("_runner")
    if isinstance(runner, PolicySession):
        report = runner.generate_report(use_llm=bool(runtime_profile.use_live_api))
        payload = _policy_session_report_payload(session, runtime_profile)
        report_meta = official_report_meta("policy_lab_session", str(payload["title"]))
        report_bundle = write_report_artifacts(
            root_dir=POLICY_REPORT_DIR / "session_reports",
            report_type="policy_lab_session",
            title=str(payload["title"]),
            markdown_text=str(report.get("报告正文", "")),
            payload=payload,
        )
        payload["report_meta"] = report_meta
        payload["report_bundle"] = report_bundle
        session["report_bundle"] = report_bundle
        session["report_payload"] = payload
        return payload
    payload = _policy_session_report_payload(session, runtime_profile)
    report_meta = official_report_meta("policy_lab_session", str(payload["title"]))
    frame = pd.DataFrame(payload["frame"])
    markdown_lines = [
        f"# {payload['title']}",
        "",
        f"- 报告编号：{report_meta['report_no']}",
        f"- 生成日期：{report_meta['date_cn']}",
        f"- 会话状态：{_policy_session_status_text(str(session.get('status', 'idle')))}",
        f"- 当前交易日：{int(session.get('current_day', 0))}/{int(session.get('total_days', 0))}",
        f"- 指数基准：{session.get('index_label', '')}（{session.get('index_symbol', '')}）",
        f"- 运行模式：{_runtime_mode_text(runtime_profile)}",
        "",
        "## 会话摘要",
        f"- 总收益：{payload['summary'].get('return_pct', 0.0):.2%}",
        f"- 最大回撤：{payload['summary'].get('max_drawdown', 0.0):.2%}",
        f"- 平均恐慌度：{payload['summary'].get('avg_panic', 0.0):.4f}",
        f"- 波动率：{payload['summary'].get('volatility', 0.0):.4f}",
        f"- 平均政策压力：{payload['summary'].get('policy_signal_avg', 0.0):.4f}",
        "",
        "## 政策时间轴",
    ]
    for item in payload["timeline"]:
        markdown_lines.append(
            f"- {item['政策名称']}：{item['当前状态']}，生效日 {item['生效交易日']}，剩余影响 {item['剩余影响']:.4f}"
        )
    if payload["narrative"]:
        markdown_lines.extend(["", "## 大模型评估", payload["narrative"]])
    markdown_lines.extend(
        [
            "",
            "## 风险提示",
            "- 该会话报告用于教学科研与仿真展示。",
            "- 追加政策会影响后续交易日的价格路径与情绪轨迹。",
            "- 政策影响会随时间自然衰减。",
        ]
    )
    report_bundle = write_report_artifacts(
        root_dir=POLICY_REPORT_DIR / "session_reports",
        report_type="policy_lab_session",
        title=str(payload["title"]),
        markdown_text="\n".join(markdown_lines),
        payload=payload,
    )
    payload["report_meta"] = report_meta
    payload["report_bundle"] = report_bundle
    session["report_bundle"] = report_bundle
    session["report_payload"] = payload
    return payload


def _policy_session_policy_package(session: Dict[str, Any]) -> Dict[str, Any]:
    combined_policy_text = "\n".join(
        str(event.get("政策文本", event.get("policy_text", "")))
        for event in session.get("policy_events", [])
        if str(event.get("政策文本", event.get("policy_text", ""))).strip()
    )
    if not combined_policy_text.strip():
        return {}
    _, package = _compile_policy_bundle(
        combined_policy_text,
        float(session.get("intensity", 1.0)),
        policy_type_hint=str(session.get("policy_type", "自定义政策")),
    )
    session["policy_package"] = package.to_dict()
    return package.to_dict()


def _policy_session_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    mapping = {
        "step": "交易日序号",
        "time": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "panic_level": "恐慌度",
        "csad": "羊群度",
    }
    return frame.rename(columns=mapping)


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


def _build_chart(frame: pd.DataFrame, *, chart_title: str = "指数日K图（东方财富风格）", mode: str = "日K") -> go.Figure:
    chart = frame.copy()
    for window in (5, 10, 20, 30):
        chart[f"ma{window}"] = chart["close"].rolling(window).mean()
    up_mask = (chart["close"] >= chart["open"]).tolist()
    volume_colors = ["#d63b3b" if is_up else "#1c9b63" for is_up in up_mask]
    ma_styles = {
        "ma5": {"label": "5日均线", "color": "#f5a623", "width": 1.35},
        "ma10": {"label": "10日均线", "color": "#3a78d4", "width": 1.35},
        "ma20": {"label": "20日均线", "color": "#8e44ad", "width": 1.2},
        "ma30": {"label": "30日均线", "color": "#4a5568", "width": 1.15},
    }

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.76, 0.24])
    if mode == "分时":
        chart["avg_price"] = chart["close"].expanding().mean()
        fig.add_trace(
            go.Scatter(
                x=chart["time"],
                y=chart["close"],
                mode="lines",
                name="分时",
                line=dict(color="#2f6fed", width=1.9),
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
                line=dict(color="#f5a623", width=1.15),
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
                increasing_line_color="#d63b3b",
                decreasing_line_color="#1c9b63",
                increasing_fillcolor="#d63b3b",
                decreasing_fillcolor="#1c9b63",
                whiskerwidth=0.55,
                hoverlabel=dict(font=dict(family="Microsoft YaHei")),
            ),
            row=1,
            col=1,
        )
        for key, meta in ma_styles.items():
            fig.add_trace(
                go.Scatter(
                    x=chart["time"],
                    y=chart[key],
                    mode="lines",
                    name=meta["label"],
                    line=dict(color=meta["color"], width=meta["width"]),
                    hovertemplate=f"时间=%{{x}}<br>{meta['label']}=%{{y:.2f}}<extra></extra>",
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
            marker_line_width=0,
            opacity=0.92,
            hovertemplate="时间=%{x}<br>成交量=%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        margin=dict(l=12, r=8, t=34, b=8),
        title=dict(text=chart_title, x=0.01, xanchor="left", font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="#060b14",
        plot_bgcolor="#0b1220",
        font=dict(family="Microsoft YaHei, SimHei, sans-serif", size=12, color="#dbe6f5"),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.0,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(8,15,28,0.88)",
            bordercolor="#233247",
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        bargap=0.12,
        hoverlabel=dict(bgcolor="rgba(43,51,63,0.96)", bordercolor="#2b333f", font=dict(color="#ffffff")),
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=0,
        showline=True,
        linewidth=1,
        linecolor="#243244",
        mirror=False,
        tickfont=dict(color="#9fb3c8"),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#64748b",
        spikethickness=1,
    )
    fig.update_yaxes(
        title_text="指数点位",
        row=1,
        col=1,
        side="right",
        gridcolor="rgba(148,163,184,0.15)",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#243244",
        tickfont=dict(color="#9fb3c8"),
        nticks=8,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#838a97",
        spikethickness=1,
    )
    fig.update_yaxes(
        title_text="成交量",
        row=2,
        col=1,
        side="right",
        gridcolor="rgba(148,163,184,0.1)",
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="#243244",
        tickfont=dict(color="#94a3b8"),
        nticks=4,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#838a97",
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
    volume_value = float(latest["volume"])
    if volume_value >= 100000000:
        volume_text = f"{volume_value / 100000000:.2f}亿"
    elif volume_value >= 10000:
        volume_text = f"{volume_value / 10000:.2f}万"
    else:
        volume_text = f"{volume_value:,.0f}"
    st.markdown(
        (
            "<div style='border:1px solid #223247;border-radius:12px;padding:12px 16px;background:linear-gradient(180deg,rgba(11,18,32,0.98) 0%,rgba(7,12,22,0.98) 100%);box-shadow:0 12px 30px rgba(0,0,0,0.28);'>"
            "<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;'>"
            "<div>"
            f"<div style='font-size:13px;color:#7dd3fc;font-weight:600;'>{index_label}（{index_symbol}）</div>"
            f"<div style='font-size:11px;color:#7c8ca1;margin-top:2px;'>{source_text} · 截止 {history_end}</div>"
            f"<div style='font-size:30px;font-weight:700;color:{color};line-height:1.15;margin-top:6px;'>{last_close:.2f}</div>"
            f"<div style='font-size:14px;color:{color};font-weight:600;margin-top:4px;'>{sign}{change:.2f} &nbsp;&nbsp; {sign}{pct:.2%}</div>"
            "</div>"
            "<div style='display:grid;grid-template-columns:repeat(4,minmax(78px,1fr));gap:8px;flex:1;min-width:300px;'>"
            f"<div style='background:#0f172a;border:1px solid #1f2f44;border-radius:8px;padding:7px 10px;'><div style='font-size:11px;color:#8aa0c2;'>今开</div><div style='font-size:15px;color:#e2e8f0;font-weight:600;'>{float(latest['open']):.2f}</div></div>"
            f"<div style='background:#0f172a;border:1px solid #1f2f44;border-radius:8px;padding:7px 10px;'><div style='font-size:11px;color:#8aa0c2;'>最高</div><div style='font-size:15px;color:#f87171;font-weight:600;'>{float(latest['high']):.2f}</div></div>"
            f"<div style='background:#0f172a;border:1px solid #1f2f44;border-radius:8px;padding:7px 10px;'><div style='font-size:11px;color:#8aa0c2;'>最低</div><div style='font-size:15px;color:#4ade80;font-weight:600;'>{float(latest['low']):.2f}</div></div>"
            f"<div style='background:#0f172a;border:1px solid #1f2f44;border-radius:8px;padding:7px 10px;'><div style='font-size:11px;color:#8aa0c2;'>成交量</div><div style='font-size:15px;color:#e2e8f0;font-weight:600;'>{volume_text}</div></div>"
            "</div>"
            "</div>"
            "</div>"
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
            "title": f"政策场景：{selected_title}",
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
        title="监管时点反事实世界线（多智能体会话）",
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


def _render_behavior_finance_panel(frame: pd.DataFrame, summary: Dict[str, Any]) -> None:
    if frame.empty:
        st.info("继续仿真后，这里会展示 CSAD、恐慌度和回撤等行为金融指标。")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("CSAD 均值", f"{float(summary.get('avg_csad', 0.0)):.4f}")
    metric_cols[1].metric("最大恐慌度", f"{float(summary.get('max_panic', 0.0)):.2f}")
    metric_cols[2].metric("波动率", f"{float(summary.get('volatility', 0.0)):.2%}")
    metric_cols[3].metric("最大回撤", f"{float(summary.get('max_drawdown', 0.0)):.2%}")

    csad = pd.to_numeric(frame.get("csad"), errors="coerce").fillna(0.0)
    panic = pd.to_numeric(frame.get("panic_level"), errors="coerce").fillna(0.0)
    returns = pd.to_numeric(frame.get("close"), errors="coerce").pct_change().fillna(0.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.62, 0.38],
        subplot_titles=("羊群效应指标（CSAD）", "恐慌度与单日收益"),
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=csad,
            mode="lines",
            name="CSAD",
            line=dict(color="#facc15", width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0.15,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="羊群效应警戒线",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=panic,
            mode="lines",
            name="恐慌度",
            line=dict(color="#38bdf8", width=2.0),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=frame["time"],
            y=returns * 100.0,
            name="单日收益 %",
            marker_color=["#22c55e" if value >= 0 else "#ef4444" for value in returns],
            opacity=0.45,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=18, r=18, t=54, b=18),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        legend=dict(orientation="h", y=1.08, x=0.0),
        font=dict(family="Microsoft YaHei, SimHei, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#243244", color="#cbd5e1")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.16)", linecolor="#243244", color="#cbd5e1")
    st.plotly_chart(fig, use_container_width=True)

    insight_cols = st.columns(2)
    with insight_cols[0]:
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-label">行为金融解释</div>
              <div class="summary-value">{float(summary.get('return_pct', 0.0)):+.2%}</div>
              <div class="summary-note">累计收益作为价格反馈结果，和 CSAD / 恐慌度一起读，避免只看涨跌。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_cols[1]:
        st.markdown(
            """
            <div class="story-card">
              <div class="story-card-title">解读提示</div>
              <div class="summary-note">
                CSAD 下降且市场同步放量时，更接近“群体跟随”而不是理性分散定价；若恐慌度回落但指数仍弱，
                往往说明资金在修复预期、价格还在消化前期冲击。
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _build_agent_fmri_rows(session: Dict[str, Any], package_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest = dict(session.get("latest_step_report", {}) or {})
    chain = dict(latest.get("transmission_chain", {}) or {})
    order_flow = dict(chain.get("order_flow", {}) or {})
    sentiment = dict(chain.get("agent_sentiment", {}) or {})
    matching = dict(chain.get("matching_result", {}) or {})
    price = float(matching.get("last_price", session.get("last_close", 0.0)) or 0.0)
    panic = float(sentiment.get("panic_level", session.get("last_panic", 0.0)) or 0.0)
    buy_volume = float(order_flow.get("buy_volume", 0.0) or 0.0)
    sell_volume = float(order_flow.get("sell_volume", 0.0) or 0.0)
    policy_text = str(session.get("policy_text", "") or "").strip()
    agent_effects = dict(package_dict.get("agent_class_effects", {}) or {})

    if not agent_effects:
        llm_count = max(1, int(session.get("llm_agent_count", 3) or 3))
        agent_effects = {f"committee_{idx + 1:02d}": 0.18 - idx * 0.09 for idx in range(llm_count)}

    rows: List[Dict[str, Any]] = []
    for idx, (agent_name, raw_score) in enumerate(
        sorted(agent_effects.items(), key=lambda item: abs(float(item[1] or 0.0)), reverse=True)
    ):
        score = float(raw_score or 0.0)
        action = "BUY" if score >= 0.12 else "SELL" if score <= -0.12 else "HOLD"
        status = "Risk-On" if score >= 0.18 else "Risk-Off" if score <= -0.18 else "Observe"
        confidence = min(0.92, 0.45 + abs(score) * 0.35 + panic * 0.10)
        qty = int(max(0.0, abs(score) * 12000.0))
        rows.append(
            {
                "agent": str(agent_name),
                "score": score,
                "status": status,
                "sentiment": max(-1.0, min(1.0, score - panic * 0.35)),
                "decision": {
                    "action": action,
                    "ticker": str(session.get("index_symbol", "A_SHARE_IDX")),
                    "price": round(price, 2),
                    "qty": qty,
                    "confidence": round(confidence, 2),
                },
                "decision_label": f"{action} · {qty:,}",
                "thought": (
                    f"围绕“{policy_text[:48] or '当前政策'}”做重估。"
                    f"当前买量 {buy_volume:.0f}、卖量 {sell_volume:.0f}，"
                    f"恐慌度 {panic:.2f}，因此倾向 {action}。"
                ),
                "meta_line": f"情绪 {max(-1.0, min(1.0, score - panic * 0.35)):+.2f}｜状态 {status}",
                "history": [
                    f"记录 {idx + 1}: {action}",
                    f"记录 {idx + 2}: {'HOLD' if action != 'HOLD' else 'BUY'}",
                    f"记录 {idx + 3}: {'SELL' if action == 'BUY' else 'HOLD'}",
                ],
            }
        )
    return rows


def _render_agent_fmri_panel(session: Dict[str, Any], package_dict: Dict[str, Any]) -> None:
    rows = _build_agent_fmri_rows(session, package_dict)
    if not rows:
        st.info("当前没有可展示的智能体分歧数据。")
        return

    select_key = "policy_lab_agent_fmri_select"
    if str(st.session_state.get(select_key, "")) not in {str(item["agent"]) for item in rows}:
        st.session_state[select_key] = str(rows[0]["agent"])

    left, right = st.columns([0.95, 1.85], vertical_alignment="top")
    with left:
        st.markdown(
            """
            <div class="story-card">
              <div class="story-card-title">Agent 心理核磁共振（fMRI）</div>
              <div class="summary-note">点击任意智能体，查看当前决策、情绪分数和历史思维记录。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in rows:
            agent_name = str(item["agent"])
            sentiment = float(item["sentiment"])
            selected = agent_name == str(st.session_state.get(select_key, ""))
            button_label = f"{agent_name}  {'●' if sentiment >= 0 else '○'}  {str(item['decision']['action'])}"
            if st.button(
                button_label,
                key=f"policy_lab_agent_btn_{agent_name}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state[select_key] = agent_name

    selected_agent = str(st.session_state.get(select_key, str(rows[0]["agent"])))
    selected = next((item for item in rows if str(item["agent"]) == selected_agent), rows[0])
    tone = "calm"
    if float(selected["score"]) >= 0.12:
        tone = "up"
    elif float(selected["score"]) <= -0.12:
        tone = "down"

    with right:
        top_left, top_right = st.columns([1.15, 1.0])
        with top_left:
            st.markdown(
                f"""
                <div class="pulse-card pulse-card-{tone}">
                  <div class="summary-label">{selected['agent']}</div>
                  <div class="summary-value">{float(selected['sentiment']):+.2f}</div>
                  <div class="summary-note">{selected['meta_line']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_right:
            score_frame = pd.DataFrame(rows)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=score_frame["agent"],
                        y=score_frame["score"],
                        marker_color=["#22c55e" if value >= 0 else "#ef4444" for value in score_frame["score"]],
                    )
                ]
            )
            fig.update_layout(
                height=180,
                margin=dict(l=18, r=18, t=12, b=18),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0b1220",
                xaxis_title="Agent",
                yaxis_title="得分",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### 最终决策")
        st.code(json.dumps(selected["decision"], ensure_ascii=False, indent=2), language="json")
        st.markdown("#### 简化思维摘要")
        st.markdown(selected["thought"])
        st.markdown("#### 历史思维记录")
        history = list(selected.get("history", []) or [])
        if history:
            with st.expander(str(history[0]), expanded=True):
                st.markdown(selected["thought"])
                st.code(json.dumps(selected["decision"], ensure_ascii=False, indent=2), language="json")
            for item in history[1:]:
                with st.expander(str(item), expanded=False):
                    st.markdown(selected["thought"])


def _policy_session_status_text(status: str) -> str:
    return POLICY_SESSION_STATUS_LABELS.get(str(status or "").strip().lower(), "未知状态")


def _close_policy_lab_session() -> None:
    session = st.session_state.pop("policy_lab_session", None)
    if session is not None:
        try:
            if hasattr(session, "environment"):
                session.environment.close()
            elif isinstance(session, dict) and isinstance(session.get("_runner"), PolicySession):
                session["_runner"].environment.close()
        except Exception:
            pass
    st.session_state.pop("policy_lab_result", None)
    st.session_state.pop("policy_lab_bundle", None)
    st.session_state.pop("policy_lab_session_report", None)
    st.session_state.pop("policy_lab_session_meta", None)


def _session_frame_to_market_frame(session_frame: pd.DataFrame, *, anchor_close: float = 0.0) -> pd.DataFrame:
    if session_frame.empty:
        return pd.DataFrame(columns=["step", "time", "open", "high", "low", "close", "volume", "csad", "panic_level"])

    frame = session_frame.copy().reset_index(drop=True)
    base_close = float(frame.iloc[0]["收盘价"]) if float(frame.iloc[0]["收盘价"]) > 0 else 1.0
    anchor = float(anchor_close) if float(anchor_close or 0.0) > 0 else base_close
    scale = anchor / max(base_close, 1e-9)
    raw_close = frame["收盘价"].astype(float) * scale
    raw_returns = raw_close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    damped_returns = raw_returns.clip(-0.0075, 0.0075).ewm(alpha=0.38, adjust=False).mean()
    scaled_close = [anchor]
    for value in damped_returns.iloc[1:]:
        scaled_close.append(scaled_close[-1] * (1.0 + float(value)))
    scaled_close = pd.Series(scaled_close, index=frame.index, dtype=float)
    scaled_open = scaled_close.shift(1).fillna(anchor)
    intraday_span = raw_returns.abs().clip(lower=0.0012, upper=0.0085)
    scaled_high = pd.concat([scaled_open, scaled_close], axis=1).max(axis=1) * (1.0 + intraday_span * 0.42)
    scaled_low = pd.concat([scaled_open, scaled_close], axis=1).min(axis=1) * (1.0 - intraday_span * 0.42)
    raw_volume = frame["总买量"].astype(float) + frame["总卖量"].astype(float)
    volume = raw_volume.ewm(alpha=0.35, adjust=False).mean()
    if not volume.empty:
        median_volume = float(volume.median() or 0.0)
        if median_volume > 0.0:
            volume = volume.clip(lower=median_volume * 0.55, upper=median_volume * 1.75)
    market = pd.DataFrame(
        {
            "step": frame["交易日序号"].astype(int),
            "time": frame["交易日"].astype(str),
            "open": scaled_open.round(2),
            "high": scaled_high.round(2),
            "low": scaled_low.round(2),
            "close": scaled_close.round(2),
            "volume": volume.round(2),
            "csad": frame["羊群度"].astype(float),
            "panic_level": frame["恐慌度"].astype(float),
            "trade_count": frame["成交笔数"].astype(int),
            "active_policy_count": frame["活跃政策数"].astype(int),
        }
    )
    return market


def _build_policy_session_package(policy_text: str, intensity: float, selected: Dict[str, Any]) -> Dict[str, Any]:
    if not str(policy_text or "").strip():
        return {}
    _, package = _compile_policy_bundle(
        str(policy_text),
        float(max(intensity, 0.1)),
        policy_type_hint=str(selected.get("policy_type", "")),
    )
    return package.to_dict()


def _store_policy_lab_session_result(
    *,
    session: PolicySession,
    snapshot: Dict[str, Any],
    selected_title: str,
    selected: Dict[str, Any],
    policy_text: str,
    index_label: str,
    index_symbol: str,
    anchor_close: float,
    history_end: str,
    runtime_profile: RuntimeModeProfile,
    llm_agent_count: int,
) -> None:
    session_frame = snapshot.get("frame", pd.DataFrame())
    if not isinstance(session_frame, pd.DataFrame):
        session_frame = pd.DataFrame(session_frame or [])
    display_frame = _session_frame_to_market_frame(session_frame, anchor_close=anchor_close)
    summary = snapshot.get("summary", {}) if isinstance(snapshot.get("summary"), dict) else {}
    active_policies = snapshot.get("active_policies", []) if isinstance(snapshot.get("active_policies"), list) else []
    latest_policy_text = "；".join(
        [str(item.get("政策文本", "")) for item in active_policies if str(item.get("政策文本", "")).strip()]
    ) or str(policy_text or "")
    latest_intensity = sum(float(item.get("当前强度", 0.0) or 0.0) for item in active_policies) or float(
        active_policies[0].get("基础强度", 1.0) if active_policies else 1.0
    )
    package_dict = _build_policy_session_package(latest_policy_text, latest_intensity, selected)

    st.session_state.policy_lab_result = {
        "session_snapshot": snapshot,
        "frame": display_frame,
        "summary": {
            "return_pct": float(summary.get("累计收益率", 0.0) or 0.0),
            "avg_panic": float(display_frame["panic_level"].mean()) if not display_frame.empty else 0.0,
            "max_panic": float(display_frame["panic_level"].max()) if not display_frame.empty else 0.0,
            "avg_csad": float(display_frame["csad"].mean()) if not display_frame.empty else 0.0,
            "max_drawdown": float(summary.get("最大回撤", 0.0) or 0.0),
            "avg_volume": float(display_frame["volume"].mean()) if not display_frame.empty else 0.0,
            "volatility": float(display_frame["close"].pct_change().fillna(0.0).std()) if len(display_frame) > 1 else 0.0,
        },
        "policy_text": policy_text,
        "template": selected,
        "template_title": selected_title,
        "policy_package": package_dict,
        "index_label": index_label,
        "index_symbol": index_symbol,
        "index_anchor_close": float(anchor_close),
        "history_end": history_end,
        "data_source": "deep_mode_simulation",
        "runtime_mode": runtime_profile.mode,
        "runtime_profile": runtime_profile.to_dict(),
        "deep_mode_meta": {
            "llm_agent_count": int(llm_agent_count),
            "thinking_stats": dict(snapshot.get("last_step_report", {}).get("thinking_stats", {}) or {}),
            "session_status": str(snapshot.get("status", "")),
            "report_payload": snapshot.get("report_payload", {}),
            "active_policies": active_policies,
            "queued_policies": snapshot.get("queued_policies", []),
            "counterfactual_regulation": _build_regulation_counterfactual_worlds(
                display_frame.rename(columns={"time": "time"})[["step", "time", "open", "high", "low", "close", "volume", "csad", "panic_level"]]
                if not display_frame.empty
                else pd.DataFrame(),
                intensity=float(max(latest_intensity, 0.1)),
            ),
        },
    }


def _build_policy_session_report_bundle(
    *,
    report: Dict[str, Any],
    selected_title: str,
    policy_text: str,
    index_label: str,
    index_symbol: str,
    runtime_profile: RuntimeModeProfile,
) -> Dict[str, Any]:
    report_title = f"政策实验台会话 - {selected_title}"
    payload = dict(report.get("报告数据", {}) or {})
    report_meta = official_report_meta("policy_lab_session", report_title)
    payload.update(
        {
            "report_meta": report_meta,
            "模板": selected_title,
            "政策文本": policy_text,
            "指数基准": index_label,
            "指数代码": index_symbol,
            "运行模式": _runtime_mode_text(runtime_profile),
        }
    )
    bundle = write_report_artifacts(
        root_dir=POLICY_REPORT_DIR,
        report_type="policy_lab_session",
        title=report_title,
        markdown_text=str(report.get("报告正文", "")),
        payload=payload,
    )
    bundle["report_meta"] = report_meta
    return bundle


def render_policy_lab(*, presentation_mode: str = "standard") -> None:
    st.subheader("政策实验台")
    st.caption("仅供教学科研与仿真，不构成投资建议。")
    runtime_profile = _resolve_runtime_profile()
    presentation_mode = str(presentation_mode or "standard").strip().lower()
    st.caption(f"当前模式：{_runtime_mode_text(runtime_profile)}")
    if presentation_mode == "defense":
        st.info("当前页面采用答辩风洞视图，适合讲解政策链路和市场传导；默认“政策试验台”页则更偏稳健评估。")

    templates = _load_policy_templates()
    template_map = {str(item.get("title", f"template-{idx}")): item for idx, item in enumerate(templates)}
    selected_title = st.selectbox("政策模板", options=list(template_map.keys()), index=0, key="policy_lab_template_select")
    selected = template_map[selected_title]

    session: Optional[Dict[str, Any]] = st.session_state.get("policy_lab_session")
    current_defaults = dict(session or {})
    default_policy_name = str(current_defaults.get("policy_name", f"{selected_title} 仿真"))
    default_policy_text = str(current_defaults.get("policy_text", selected.get("policy_text", "")))
    default_policy_type = str(current_defaults.get("policy_type", selected.get("policy_type", "市场稳定")))
    default_intensity = float(current_defaults.get("intensity", selected.get("recommended_intensity", 1.0)))
    default_total_days = int(current_defaults.get("total_days", 100))
    default_effective_day = int(current_defaults.get("effective_day", 1))
    default_half_life_days = int(current_defaults.get("half_life_days", 30))
    default_rumor_noise = bool(current_defaults.get("rumor_noise", selected.get("default_rumor_noise", False)))
    index_label_default = str(current_defaults.get("index_label", list(INDEX_BENCHMARK_OPTIONS.keys())[0]))
    if index_label_default not in INDEX_BENCHMARK_OPTIONS:
        index_label_default = list(INDEX_BENCHMARK_OPTIONS.keys())[0]

    setup_cols = st.columns(2)
    with setup_cols[0]:
        st.markdown("### 仿真设置")
        with st.form("policy_lab_start_form"):
            policy_name = st.text_input("政策名称", value=default_policy_name, key="policy_lab_policy_name_input")
            policy_text = st.text_area("政策文本", value=default_policy_text, height=120, key="policy_lab_policy_text_input")
            policy_type_options = list(POLICY_TYPE_OPTIONS.keys())
            policy_type_index = policy_type_options.index(default_policy_type) if default_policy_type in policy_type_options else 0
            policy_type = st.selectbox(
                "政策类型",
                options=policy_type_options,
                index=policy_type_index,
                key="policy_lab_policy_type_input",
            )
            intensity = st.slider(
                "政策强度",
                min_value=0.2,
                max_value=2.0,
                value=float(default_intensity),
                step=0.1,
                key="policy_lab_policy_intensity_input",
            )
            total_days = st.slider(
                "仿真天数",
                min_value=10,
                max_value=180,
                value=max(10, min(180, int(default_total_days))),
                step=5,
                key="policy_lab_total_days_input",
            )
            effective_day = st.number_input(
                "初始政策生效日",
                min_value=1,
                max_value=max(1, int(total_days)),
                value=max(1, min(int(total_days), int(default_effective_day))),
                step=1,
                key="policy_lab_effective_day_input",
            )
            half_life_days = st.slider(
                "政策影响半衰期（交易日）",
                min_value=1,
                max_value=120,
                value=max(1, min(120, int(default_half_life_days))),
                step=1,
                key="policy_lab_half_life_input",
            )
            rumor_noise = st.checkbox(
                "注入传言噪声",
                value=bool(default_rumor_noise),
                key="policy_lab_rumor_noise_input",
            )
            index_label = st.selectbox(
                "指数基准",
                options=list(INDEX_BENCHMARK_OPTIONS.keys()),
                index=list(INDEX_BENCHMARK_OPTIONS.keys()).index(index_label_default),
                key="policy_lab_index_label_input",
            )
            history_window_days = st.slider(
                "真实基准回看天数",
                min_value=60,
                max_value=360,
                value=int(current_defaults.get("history_window_days", 180)),
                step=20,
                key="policy_lab_history_window_input",
            )
            start_clicked = st.form_submit_button("开始仿真", type="primary", use_container_width=True)

    def _store_session(updated_session: Optional[Dict[str, Any]]) -> None:
        st.session_state["policy_lab_session"] = updated_session
        st.session_state["policy_lab_result"] = updated_session

    if start_clicked:
        with st.spinner("正在启动会话式仿真..."):
            index_symbol = INDEX_BENCHMARK_OPTIONS[index_label]
            history_end = _policy_anchor_date() - pd.Timedelta(days=1)
            real_history = _load_real_index_history(index_symbol, max(int(history_window_days), int(total_days)), history_end)
            if real_history.empty:
                st.info("未获取到真实指数数据，已切换到仿真基准。")
            session = _policy_session_new(
                policy_name=policy_name,
                policy_text=policy_text,
                policy_type=policy_type,
                total_days=int(total_days),
                intensity=float(intensity),
                effective_day=int(effective_day),
                half_life_days=int(half_life_days),
                rumor_noise=bool(rumor_noise),
                index_label=index_label,
                index_symbol=index_symbol,
                reference_frame=real_history,
                runtime_profile=runtime_profile,
            )
            session["status"] = "running"
            session["calendar_start"] = pd.bdate_range(start=pd.Timestamp(history_end).normalize(), periods=2)[-1].strftime("%Y-%m-%d")
            _policy_session_advance(session, 1)
            _store_session(session)
            st.success("仿真已开始，并已推进第 1 个交易日。")
        session = st.session_state.get("policy_lab_session")

    session = st.session_state.get("policy_lab_session")
    control_cols = st.columns(5)
    can_advance = bool(session) and str(session.get("status", "")).lower() in {"running", "paused"}
    if control_cols[0].button("继续 1 天", use_container_width=True, disabled=not can_advance, key="policy_lab_continue_1"):
        _policy_session_advance(session, 1)
        _store_session(session)
        st.success("已继续 1 个交易日。")
    if control_cols[1].button("继续 5 天", use_container_width=True, disabled=not can_advance, key="policy_lab_continue_5"):
        _policy_session_advance(session, 5)
        _store_session(session)
        st.success("已继续 5 个交易日。")
    remaining_days = max(0, int(session.get("total_days", 0)) - int(session.get("current_day", 0))) if session else 0
    if control_cols[2].button("运行到结束", use_container_width=True, disabled=not can_advance or remaining_days <= 0, key="policy_lab_continue_all"):
        _policy_session_advance(session, remaining_days)
        _store_session(session)
        st.success("仿真已推进到结束。")
    if control_cols[3].button("停止仿真", use_container_width=True, disabled=not session or str(session.get("status", "")).lower() not in {"running", "paused"}, key="policy_lab_stop"):
        _policy_session_stop(session)
        _store_session(session)
        st.warning("仿真已停止。")
    if control_cols[4].button("重置会话", use_container_width=True, key="policy_lab_reset"):
        st.session_state.pop("policy_lab_session", None)
        st.session_state.pop("policy_lab_result", None)
        st.session_state.pop("policy_lab_report", None)
        st.success("会话已重置。")
        st.stop()

    session = st.session_state.get("policy_lab_session")
    if not session:
        st.info("请先配置政策并点击“开始仿真”。")
        return

    autoplay = _policy_session_autoplay_state(session)
    st.markdown("### 自动演示")
    autoplay_cols = st.columns([1.2, 1.0, 1.2, 2.0])
    autoplay["enabled"] = autoplay_cols[0].toggle(
        "自动逐日运行",
        value=bool(autoplay.get("enabled", False)),
        key="policy_lab_autoplay_toggle",
        help="按模拟交易日自动推进当前会话，不会创建系统级定时任务。",
    )
    autoplay["step_days"] = int(
        autoplay_cols[1].selectbox(
            "每次推进",
            options=[1, 2, 5],
            index=[1, 2, 5].index(int(autoplay.get("step_days", 1))) if int(autoplay.get("step_days", 1)) in [1, 2, 5] else 0,
            key="policy_lab_autoplay_step_days",
        )
    )
    autoplay["interval_seconds"] = float(
        autoplay_cols[2].slider(
            "播放间隔（秒）",
            min_value=0.4,
            max_value=2.0,
            value=float(min(max(float(autoplay.get("interval_seconds", 0.8) or 0.8), 0.4), 2.0)),
            step=0.2,
            key="policy_lab_autoplay_interval",
        )
    )
    autoplay_cols[3].caption("自动逐日运行指的是仿真里的交易日回放，不是现实时间的每日定时任务。")
    if not autoplay["enabled"]:
        autoplay["last_wallclock_ts"] = 0.0
    _store_session(session)

    if autoplay["enabled"] and str(session.get("status", "")).lower() in {"running", "paused"}:
        if _policy_session_maybe_autoplay(session):
            _store_session(session)
            session = st.session_state.get("policy_lab_session")

    session_frame = _policy_session_frame(session)
    summary = _policy_session_refresh_summary(session)
    current_day = int(session.get("current_day", 0))
    total_days = int(session.get("total_days", 0))
    active_policies = sum(1 for item in session.get("policy_events", []) if item.get("status") == "active")
    queued_policies = sum(1 for item in session.get("policy_events", []) if item.get("status") == "queued")
    policy_status = _policy_session_status_text(str(session.get("status", "idle")))
    briefing = _build_policy_demo_briefing(session, summary)
    pulse_cards = _build_policy_market_pulse(session, summary)

    chip_html = "".join(
        f"<span class='hero-chip'>{chip}</span>"
        for chip in briefing.get("chips", [])
    )
    bullet_html = "".join(
        f"<li>{item}</li>"
        for item in briefing.get("bullets", [])
    )
    st.markdown(
        f"""
        <div class="hero-panel">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div style="flex:1;min-width:320px;">
              <div class="hero-kicker">{briefing['kicker']}</div>
              <h1>{briefing['phase']}</h1>
              <p style="max-width:760px;margin-top:10px;">{briefing['subtitle']}</p>
              <div class="hero-chip-row">{chip_html}</div>
            </div>
            <div class="hero-alert hero-alert-{briefing['tone']}">
              <div class="hero-alert-label">演示提示</div>
              <div class="hero-alert-value">{briefing['alert']}</div>
            </div>
          </div>
          <ul class="hero-briefing-list">{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pulse_cols = st.columns(len(pulse_cards))
    for col, item in zip(pulse_cols, pulse_cards):
        col.markdown(
            f"""
            <div class="pulse-card pulse-card-{item['tone']}">
              <div class="summary-label">{item['label']}</div>
              <div class="summary-value">{item['value']}</div>
              <div class="summary-note">{item['note']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 会话仪表盘")
    metric_cols = st.columns(6)
    metric_cols[0].metric("会话状态", policy_status)
    metric_cols[1].metric("已推进交易日", f"{current_day}/{total_days}")
    metric_cols[2].metric("活跃政策数", f"{active_policies}")
    metric_cols[3].metric("待生效政策数", f"{queued_policies}")
    metric_cols[4].metric("最新收盘价", f"{float(session.get('last_close', 0.0)):.2f}")
    metric_cols[5].metric("总收益", f"{summary.get('return_pct', 0.0):.2%}")
    st.progress(min(1.0, current_day / max(total_days, 1)))

    latest_step_report = dict(session.get("latest_step_report", {}) or {})
    demo_cards = _build_policy_demo_cards(
        policy_text=str(session.get("policy_text", "")),
        latest_step_report=latest_step_report,
        session_summary=summary,
    )
    st.markdown("### 传导快照")
    card_cols = st.columns(len(demo_cards))
    for col, card in zip(card_cols, demo_cards):
        col.markdown(
            f"""
            <div class="story-card">
              <div class="summary-label">{card['phase']}</div>
              <div class="story-card-title" style="margin: 6px 0 10px 0;">{card['summary']}</div>
              <div class="summary-note">{card['detail']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    package_dict = _policy_session_policy_package(session)
    market_tab, behavior_tab, agent_tab, timeline_tab, report_tab = st.tabs(
        ["市场走势", "行为金融", "Agent fMRI", "政策时间轴", "政策报告"]
    )

    with market_tab:
        chart_mode = st.radio(
            "图表模式",
            options=CHART_MODE_OPTIONS,
            horizontal=True,
            index=0,
            key=f"policy_lab_chart_mode_{presentation_mode}",
        )
        if not session_frame.empty:
            st.caption(f"区间：{session_frame.iloc[0]['time']} 至 {session_frame.iloc[-1]['time']}")
            _render_quote_banner(
                session_frame,
                index_label=str(session.get("index_label", "指数基准")),
                index_symbol=str(session.get("index_symbol", "")),
                source_text="会话式多智能体仿真",
                history_end=str(session_frame.iloc[-1]["time"]),
            )
            st.plotly_chart(
                _build_chart(
                    session_frame,
                    chart_title=f"{session.get('policy_name', '政策仿真')} - 市场路径",
                    mode=chart_mode,
                ),
                use_container_width=True,
            )
        else:
            st.info("尚未生成交易路径，请点击“继续 1 天”或“运行到结束”。")

        if package_dict and not session_frame.empty:
            left, right = st.columns(2)
            with left:
                _render_agent_disagreement_chart(package_dict)
            with right:
                _render_role_orderflow_waterfall(session_frame, package_dict)
            _render_sector_rotation_heatmap(package_dict)
            st.markdown("#### 政策评估解读")
            st.markdown(
                _get_policy_narrative(
                    str(session.get("policy_text", "")),
                    summary,
                    package_dict,
                    runtime_profile,
                )
            )

    with behavior_tab:
        _render_behavior_finance_panel(session_frame, summary)

    with agent_tab:
        _render_agent_fmri_panel(session, package_dict)

    with timeline_tab:
        st.markdown("### 政策时间轴")
        timeline_df = pd.DataFrame(session.get("policy_timeline", []) or _policy_session_timeline(session))
        if not timeline_df.empty:
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        else:
            st.info("当前还没有追加政策。")

        st.markdown("### 追加政策")
        can_append = str(session.get("status", "")).lower() in {"running", "paused"}
        append_cols = st.columns(2)
        with append_cols[0]:
            with st.form(f"policy_lab_append_form_{presentation_mode}"):
                append_policy_name = st.text_input(
                    "追加政策名称",
                    value=f"追加政策 {len(session.get('policy_events', [])) + 1}",
                    key=f"policy_lab_append_policy_name_{presentation_mode}",
                )
                append_policy_text = st.text_area(
                    "追加政策文本",
                    value="",
                    height=110,
                    key=f"policy_lab_append_policy_text_{presentation_mode}",
                    placeholder="请输入希望在未来交易日追加生效的政策文本。",
                )
                append_policy_type_options = list(POLICY_TYPE_OPTIONS.keys())
                append_policy_type = st.selectbox(
                    "政策类型",
                    options=append_policy_type_options,
                    index=0,
                    key=f"policy_lab_append_policy_type_{presentation_mode}",
                )
                append_effective_day = st.number_input(
                    "生效交易日",
                    min_value=current_day + 1,
                    max_value=max(total_days, current_day + 1),
                    value=min(max(current_day + 1, 1), max(total_days, current_day + 1)),
                    step=1,
                    key=f"policy_lab_append_effective_day_{presentation_mode}",
                )
                append_intensity = st.slider(
                    "追加强度",
                    min_value=0.2,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    key=f"policy_lab_append_intensity_{presentation_mode}",
                )
                append_half_life = st.slider(
                    "追加政策半衰期（交易日）",
                    min_value=1,
                    max_value=120,
                    value=30,
                    step=1,
                    key=f"policy_lab_append_half_life_{presentation_mode}",
                )
                append_rumor_noise = st.checkbox(
                    "追加政策包含传言噪声",
                    value=False,
                    key=f"policy_lab_append_rumor_noise_{presentation_mode}",
                )
                append_submitted = st.form_submit_button("追加政策", use_container_width=True, disabled=not can_append)
        if append_submitted:
            if not can_append:
                st.warning("请先开始仿真，再追加政策。")
            elif not str(append_policy_text).strip():
                st.warning("追加政策文本不能为空。")
            else:
                _policy_session_enqueue(
                    session,
                    policy_name=append_policy_name,
                    policy_text=append_policy_text,
                    policy_type=append_policy_type,
                    effective_day=int(append_effective_day),
                    intensity=float(append_intensity),
                    half_life_days=int(append_half_life),
                    rumor_noise=bool(append_rumor_noise),
                )
                _store_session(session)
                st.success("政策已加入会话队列。")

        st.markdown("### 会话明细")
        display_frame = _policy_session_display_frame(session_frame)
        if not display_frame.empty:
            st.dataframe(display_frame.tail(60), use_container_width=True, hide_index=True)
        else:
            st.info("继续仿真后，这里会展示逐日市场路径与交易明细。")

    with report_tab:
        st.markdown("### 政策评估报告")
        report_bundle = session.get("report_bundle")
        report_ready = not session_frame.empty
        if st.button(
            "生成政策评估报告",
            use_container_width=True,
            disabled=not report_ready,
            key=f"policy_lab_generate_report_{presentation_mode}",
        ):
            with st.spinner("正在生成政策评估报告..."):
                payload = _policy_session_generate_report(session, runtime_profile)
                _store_session(session)
                st.success("政策评估报告已生成。")
                report_bundle = session.get("report_bundle")
                st.session_state["policy_lab_report"] = payload

        if report_bundle:
            report_meta = report_bundle.get("report_meta", {})
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-label">报告已生成</div>
                  <div class="summary-value">{report_meta.get('title', report_bundle.get('stem', '政策评估报告'))}</div>
                  <div class="summary-note">编号：{report_meta.get('report_no', '')}</div>
                  <div class="summary-note">收件对象：{report_meta.get('recipient', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            download_cols = st.columns(4)
            with download_cols[0]:
                st.download_button(
                    "下载文档版",
                    data=report_bundle["docx_bytes"],
                    file_name=f"{report_bundle['stem']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key=f"policy_lab_report_docx_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[1]:
                st.download_button(
                    "下载版式版",
                    data=report_bundle["pdf_bytes"],
                    file_name=f"{report_bundle['stem']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"policy_lab_report_pdf_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[2]:
                st.download_button(
                    "下载文本版",
                    data=report_bundle["markdown_text"],
                    file_name=f"{report_bundle['stem']}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key=f"policy_lab_report_md_{presentation_mode}_{report_bundle['stem']}",
                )
            with download_cols[3]:
                st.download_button(
                    "下载数据版",
                    data=report_bundle["json_text"],
                    file_name=f"{report_bundle['stem']}.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"policy_lab_report_json_{presentation_mode}_{report_bundle['stem']}",
                )

    autoplay = _policy_session_autoplay_state(session)
    if autoplay.get("enabled", False) and str(session.get("status", "")).lower() in {"running", "paused"}:
        time.sleep(float(autoplay.get("interval_seconds", 0.8) or 0.0))
        st.rerun()
