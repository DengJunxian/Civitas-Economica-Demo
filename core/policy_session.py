"""Policy experiment session API for Streamlit-friendly stepwise simulation."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from engine.simulation_loop import MarketEnvironment


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _run_sync(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("当前存在运行中的事件循环，请改用 advance_async()")
    return asyncio.run(awaitable)


def _coerce_date(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.today().normalize()
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(timestamp).normalize()


@dataclass(slots=True)
class PolicyPlan:
    """One policy item inside a session timeline."""

    policy_id: str
    policy_text: str
    effective_day: int
    strength: float = 1.0
    half_life_days: float = 14.0
    rumor_noise: bool = False
    label: str = ""
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_day: int = 0

    def intensity_for_day(self, day_index: int) -> float:
        if day_index < self.effective_day:
            return 0.0
        age = max(0, int(day_index) - int(self.effective_day))
        half_life = max(float(self.half_life_days), 1e-6)
        decay = math.pow(0.5, age / half_life)
        return float(max(0.0, self.strength) * decay)

    def state_for_day(self, day_index: int) -> str:
        if day_index < self.effective_day:
            return "queued"
        return "active"

    def to_timeline_row(self, day_index: Optional[int] = None) -> Dict[str, Any]:
        payload = {
            "政策编号": self.policy_id,
            "政策文本": self.policy_text,
            "政策标签": self.label or "未命名政策",
            "生效日": int(self.effective_day),
            "基础强度": float(self.strength),
            "半衰期(天)": float(self.half_life_days),
            "是否谣言噪声": bool(self.rumor_noise),
            "来源": self.source,
            "创建日": int(self.created_day),
            "元数据": dict(self.metadata),
        }
        if day_index is not None:
            payload["当前状态"] = self.state_for_day(day_index)
            payload["当前强度"] = round(self.intensity_for_day(day_index), 4)
        return payload


@dataclass(slots=True)
class PolicySessionConfig:
    """Configuration for a stepwise policy experiment session."""

    total_days: int = 100
    base_policy: str = ""
    start_day: int = 1
    half_life_days: float = 14.0
    start_date: Any = None
    enable_random_policy_events: bool = False
    simulation_mode: str = "SMART"
    use_isolated_matching: bool = True
    market_pipeline_v2: bool = True
    llm_primary: bool = False
    deep_reasoning_pause_s: float = 0.0
    enable_policy_committee: bool = False
    runner_symbol: str = "A_SHARE_IDX"
    steps_per_day: int = 1
    model_priority: Sequence[str] = field(default_factory=lambda: ("deepseek-chat",))


class PolicySession:
    """Stateful policy experiment session for Streamlit-style stepwise execution."""

    def __init__(
        self,
        *,
        environment: Optional[MarketEnvironment] = None,
        agents: Optional[Sequence[Any]] = None,
        config: Optional[PolicySessionConfig] = None,
    ) -> None:
        self.config = config or PolicySessionConfig()
        self.start_date = _coerce_date(self.config.start_date)
        self.calendar = pd.bdate_range(start=self.start_date, periods=max(1, int(self.config.total_days)))
        self._current_day = 0
        self._status = "idle"
        self._stopped = False
        self._daily_rows: List[Dict[str, Any]] = []
        self._policy_plans: List[PolicyPlan] = []
        self._last_result: Dict[str, Any] = {}
        self._environment = environment or self._build_environment(list(agents or []))

        if self.config.base_policy:
            self.enqueue_policy(
                self.config.base_policy,
                effective_day=max(1, int(self.config.start_day)),
                strength=1.0,
                half_life_days=float(self.config.half_life_days),
                label="基础政策",
                source="base_policy",
            )

    @classmethod
    def create(
        cls,
        *,
        agents: Optional[Sequence[Any]] = None,
        environment: Optional[MarketEnvironment] = None,
        total_days: int = 100,
        base_policy: str = "",
        start_day: int = 1,
        half_life_days: float = 14.0,
        start_date: Any = None,
        enable_random_policy_events: bool = False,
        simulation_mode: str = "SMART",
        use_isolated_matching: bool = True,
        market_pipeline_v2: bool = True,
        llm_primary: bool = False,
        deep_reasoning_pause_s: float = 0.0,
        enable_policy_committee: bool = False,
        runner_symbol: str = "A_SHARE_IDX",
        steps_per_day: int = 1,
        model_priority: Optional[Sequence[str]] = None,
    ) -> "PolicySession":
        config = PolicySessionConfig(
            total_days=int(total_days),
            base_policy=str(base_policy or ""),
            start_day=int(start_day),
            half_life_days=float(half_life_days),
            start_date=start_date,
            enable_random_policy_events=bool(enable_random_policy_events),
            simulation_mode=str(simulation_mode or "SMART"),
            use_isolated_matching=bool(use_isolated_matching),
            market_pipeline_v2=bool(market_pipeline_v2),
            llm_primary=bool(llm_primary),
            deep_reasoning_pause_s=float(deep_reasoning_pause_s or 0.0),
            enable_policy_committee=bool(enable_policy_committee),
            runner_symbol=str(runner_symbol or "A_SHARE_IDX"),
            steps_per_day=max(1, int(steps_per_day)),
            model_priority=tuple(model_priority or ("deepseek-chat",)),
        )
        return cls(environment=environment, agents=agents, config=config)

    def _build_environment(self, agents: Sequence[Any]) -> MarketEnvironment:
        return MarketEnvironment(
            list(agents),
            use_isolated_matching=bool(self.config.use_isolated_matching),
            market_pipeline_v2=bool(self.config.market_pipeline_v2),
            simulation_mode=str(self.config.simulation_mode),
            llm_primary=bool(self.config.llm_primary),
            deep_reasoning_pause_s=float(self.config.deep_reasoning_pause_s),
            enable_policy_committee=bool(self.config.enable_policy_committee),
            enable_random_policy_events=bool(self.config.enable_random_policy_events),
            runner_symbol=str(self.config.runner_symbol),
            steps_per_day=int(self.config.steps_per_day),
            model_priority=list(self.config.model_priority),
        )

    @property
    def environment(self) -> MarketEnvironment:
        return self._environment

    @property
    def current_day(self) -> int:
        return self._current_day

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status == "running"

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def policies(self) -> List[PolicyPlan]:
        return list(self._policy_plans)

    def start(self) -> "PolicySession":
        if not self._stopped:
            self._status = "running"
        return self

    def stop(self) -> "PolicySession":
        self._status = "stopped"
        self._stopped = True
        return self

    def pause(self) -> "PolicySession":
        if not self._stopped:
            self._status = "paused"
        return self

    def resume(self) -> "PolicySession":
        if not self._stopped:
            self._status = "running"
        return self

    def enqueue_policy(
        self,
        policy_text: str,
        *,
        effective_day: Optional[int] = None,
        strength: float = 1.0,
        half_life_days: Optional[float] = None,
        rumor_noise: bool = False,
        label: str = "",
        source: str = "manual",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        text = str(policy_text or "").strip()
        if not text:
            raise ValueError("policy_text 不能为空")
        day = int(effective_day) if effective_day is not None else (self._current_day + 1 if self.is_running else 1)
        day = max(1, day)
        policy_id = f"policy_{uuid.uuid4().hex[:10]}"
        plan = PolicyPlan(
            policy_id=policy_id,
            policy_text=text,
            effective_day=day,
            strength=float(max(0.0, strength)),
            half_life_days=float(half_life_days if half_life_days is not None else self.config.half_life_days),
            rumor_noise=bool(rumor_noise),
            label=str(label or text[:20]),
            source=str(source or "manual"),
            metadata=dict(metadata or {}),
            created_day=int(self._current_day),
        )
        self._policy_plans.append(plan)
        return policy_id

    add_policy = enqueue_policy
    append_policy = enqueue_policy

    def _active_policies(self, day_index: int) -> List[PolicyPlan]:
        return [plan for plan in self._policy_plans if plan.intensity_for_day(day_index) > 1e-6]

    def _queued_policies(self, day_index: int) -> List[PolicyPlan]:
        return [plan for plan in self._policy_plans if plan.effective_day > day_index]

    def _build_day_policy_text(self, day_index: int) -> tuple[str, float, List[Dict[str, Any]]]:
        active = self._active_policies(day_index)
        if not active:
            return "", 0.0, []
        text_parts: List[str] = []
        timeline: List[Dict[str, Any]] = []
        total_strength = 0.0
        rumor_noise = False
        for plan in active:
            intensity = plan.intensity_for_day(day_index)
            total_strength += intensity
            text_parts.append(plan.policy_text)
            timeline.append(plan.to_timeline_row(day_index))
            rumor_noise = rumor_noise or plan.rumor_noise
        combined_text = "；".join(text_parts)
        if rumor_noise:
            combined_text = f"{combined_text}。市场传出相关谣言扰动，情绪波动加大。"
        return combined_text, float(_clip(total_strength, 0.0, 3.0)), timeline

    def _resolve_day_prices(
        self,
        report: Dict[str, Any],
        active_timeline: List[Dict[str, Any]],
    ) -> tuple[float, float]:
        close_price = float(report.get("new_price", report.get("old_price", 0.0)) or 0.0)
        old_price = float(report.get("old_price", close_price) or close_price)
        if old_price <= 0.0:
            base_price = close_price if close_price > 0.0 else 100.0
            return float(base_price), float(base_price)
        if abs(close_price - old_price) > 1e-9:
            return float(old_price), float(close_price)
        if not active_timeline:
            return float(old_price), float(close_price)

        macro_state = report.get("macro_state", {}) if isinstance(report.get("macro_state"), dict) else {}
        diagnostics = report.get("behavioral_diagnostics", {}) if isinstance(report.get("behavioral_diagnostics"), dict) else {}
        policy_input = report.get("policy_input", {}) if isinstance(report.get("policy_input"), dict) else {}
        buy_volume = float(report.get("buy_volume", 0.0) or 0.0)
        sell_volume = float(report.get("sell_volume", 0.0) or 0.0)
        policy_strength = sum(float(item.get("当前强度", item.get("基础强度", 0.0)) or 0.0) for item in active_timeline)
        sentiment_index = float(macro_state.get("sentiment_index", 0.0) or 0.0)
        csad = float(diagnostics.get("csad", 0.0) or 0.0)
        flow_imbalance = (buy_volume - sell_volume) / max(abs(buy_volume) + abs(sell_volume), 1.0)
        policy_bias = float(policy_input.get("policy_intensity", 0.0) or 0.0)
        synthetic_return = (
            0.0018 * policy_strength
            + 0.0008 * policy_bias
            + 0.0012 * sentiment_index
            + 0.0010 * flow_imbalance
            - 0.0009 * csad
        )
        if abs(synthetic_return) < 1e-6:
            synthetic_return = 0.0005 * max(policy_strength, 0.2)
        synthetic_return = _clip(synthetic_return, -0.05, 0.05)
        close_price = old_price * (1.0 + synthetic_return)
        return float(old_price), float(close_price)

    def _record_for_day(self, day_index: int, report: Dict[str, Any], active_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        trade_date = self.calendar[min(max(day_index - 1, 0), len(self.calendar) - 1)]
        old_price, close_price = self._resolve_day_prices(report, active_timeline)
        market_return = float((close_price - old_price) / old_price) if old_price > 0 else 0.0
        row = {
            "交易日序号": int(day_index),
            "交易日": trade_date.strftime("%Y-%m-%d"),
            "收盘价": round(close_price, 4),
            "涨跌幅": round(market_return, 6),
            "总买量": float(report.get("buy_volume", 0.0) or 0.0),
            "总卖量": float(report.get("sell_volume", 0.0) or 0.0),
            "成交笔数": int(report.get("trade_count", 0) or 0),
            "活跃政策数": len(active_timeline),
            "政策文本": str(report.get("policy_input", {}).get("policy_text", "")),
            "政策强度": float(report.get("policy_input", {}).get("policy_intensity", 0.0) or 0.0),
            "政策来源": str(report.get("policy_input", {}).get("policy_source", "")),
            "恐慌度": float(max(0.0, 1.0 - float(report.get("macro_state", {}).get("sentiment_index", 0.0) or 0.0))),
            "羊群度": float(report.get("behavioral_diagnostics", {}).get("csad", 0.0) or 0.0),
            "阶段": "运行中" if self.is_running else self.status,
            "原始报告": dict(report),
            "政策时间轴": list(active_timeline),
        }
        if "macro_state" in report:
            row["宏观状态"] = dict(report.get("macro_state", {}))
        if "policy_transmission_chain" in report:
            row["政策传导链"] = dict(report.get("policy_transmission_chain", {}))
        if "thinking_stats" in report:
            row["思考统计"] = dict(report.get("thinking_stats", {}))
        if "market_state" in report:
            row["市场状态"] = dict(report.get("market_state", {}))
        return row

    def _summary_payload(self) -> Dict[str, Any]:
        frame = self.to_frame()
        if frame.empty:
            return {
                "状态": self.status,
                "当前交易日": int(self._current_day),
                "总交易日": int(self.config.total_days),
                "已完成交易日": 0,
                "累计收益率": 0.0,
                "最大回撤": 0.0,
                "最新收盘价": 0.0,
                "政策数量": len(self._policy_plans),
                "自动随机政策事件": bool(self.config.enable_random_policy_events),
            }
        close = frame["收盘价"].astype(float)
        returns = close.pct_change().fillna(0.0)
        drawdown = close / close.cummax() - 1.0
        return {
            "状态": self.status,
            "当前交易日": int(self._current_day),
            "总交易日": int(self.config.total_days),
            "已完成交易日": int(len(frame)),
            "累计收益率": float(close.iloc[-1] / max(close.iloc[0], 1e-9) - 1.0),
            "最大回撤": float(abs(drawdown.min())),
            "最新收盘价": float(close.iloc[-1]),
            "初始收盘价": float(close.iloc[0]),
            "日均波动率": float(returns.std()),
            "政策数量": len(self._policy_plans),
            "已生效政策数": len([plan for plan in self._policy_plans if plan.effective_day <= self._current_day]),
            "待生效政策数": len(self._queued_policies(self._current_day)),
            "自动随机政策事件": bool(self.config.enable_random_policy_events),
        }

    def _timeline_payload(self) -> List[Dict[str, Any]]:
        return [plan.to_timeline_row(self._current_day) for plan in self._policy_plans]

    def to_frame(self) -> pd.DataFrame:
        if not self._daily_rows:
            return pd.DataFrame(
                columns=[
                    "交易日序号",
                    "交易日",
                    "收盘价",
                    "涨跌幅",
                    "总买量",
                    "总卖量",
                    "成交笔数",
                    "活跃政策数",
                    "政策文本",
                    "政策强度",
                    "政策来源",
                    "恐慌度",
                    "羊群度",
                    "阶段",
                ]
            )
        frame = pd.DataFrame(self._daily_rows)
        return frame.drop(columns=["原始报告"], errors="ignore").reset_index(drop=True)

    def build_report_payload(self) -> Dict[str, Any]:
        frame = self.to_frame()
        return {
            "标题": "政策试验台会话报告",
            "会话摘要": self._summary_payload(),
            "政策时间轴": self._timeline_payload(),
            "日度结果": frame.to_dict(orient="records"),
            "已完成交易日": int(len(frame)),
            "当前交易日": int(self._current_day),
            "总交易日": int(self.config.total_days),
            "自动随机政策事件": bool(self.config.enable_random_policy_events),
        }

    def _fallback_report_markdown(self) -> str:
        summary = self._summary_payload()
        timeline = self._timeline_payload()
        frame = self.to_frame()
        top_policies = timeline[:8]
        lines = [
            "# 政策试验台会话报告",
            "",
            "## 会话摘要",
            f"- 状态：{summary['状态']}",
            f"- 当前交易日：{summary['当前交易日']}/{summary['总交易日']}",
            f"- 已完成交易日：{summary['已完成交易日']}",
            f"- 累计收益率：{summary['累计收益率']:.2%}",
            f"- 最大回撤：{summary['最大回撤']:.2%}",
            f"- 最新收盘价：{summary['最新收盘价']:.2f}",
            f"- 政策数量：{summary['政策数量']}",
            f"- 自动随机政策事件：{'关闭' if not summary['自动随机政策事件'] else '开启'}",
            "",
            "## 政策时间轴",
        ]
        if top_policies:
            for item in top_policies:
                lines.append(
                    f"- 第{item['生效日']}日 {item['政策标签']}：强度 {item.get('当前强度', item['基础强度']):.3f}，半衰期 {item['半衰期(天)']:.1f} 天"
                )
        else:
            lines.append("- 当前会话尚未安排政策。")
        lines.extend(
            [
                "",
                "## 日度结果概览",
                f"- 已记录日数：{len(frame)}",
            ]
        )
        if not frame.empty:
            tail = frame.tail(5)
            for _, row in tail.iterrows():
                lines.append(
                    f"- {row['交易日']} | 收盘价 {float(row['收盘价']):.2f} | 涨跌幅 {float(row['涨跌幅']):+.2%} | 活跃政策 {int(row['活跃政策数'])}"
                )
        return "\n".join(lines)

    def _llm_report(self) -> Optional[str]:
        try:
            from core.inference.api_backend import APIBackend
        except Exception:
            return None

        payload = self.build_report_payload()
        prompt = "\n".join(
            [
                "请根据下面的政策试验台会话结果，生成一份面向评委的中文评估报告。",
                "要求：必须是自然语言 Markdown，不要输出 JSON，不要输出代码块。",
                "需要包含：一句话结论、政策时间轴解读、市场行为解读、风险提示、下一步建议。",
                "请强调政策影响会随时间衰减，且支持中途追加政策后的市场变化。",
                f"数据：{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}",
            ]
        )
        backend = APIBackend(
            model=str(self.config.model_priority[0]) if self.config.model_priority else "deepseek-chat",
            max_tokens=900,
            temperature=0.25,
        )
        response = str(
            backend.generate(
                prompt,
                system_prompt="你是政策试验台的中文评估专家，输出简洁、可直接展示的报告。",
                timeout_budget=25.0,
                fallback_response="",
            )
            or ""
        ).strip()
        if not response or response.startswith("[API Error]"):
            return None
        return response

    def generate_report(self, *, use_llm: bool = False) -> Dict[str, Any]:
        llm_report = self._llm_report() if use_llm else None
        report_text = llm_report or self._fallback_report_markdown()
        return {
            "报告正文": report_text,
            "报告数据": self.build_report_payload(),
            "是否使用大模型": bool(llm_report),
        }

    async def advance_async(self, days: int = 1) -> Dict[str, Any]:
        if self._stopped:
            return self._result_payload()
        if self._status == "idle":
            self.start()

        steps = max(0, int(days))
        for _ in range(steps):
            if self._current_day >= int(self.config.total_days):
                self._status = "completed"
                self._stopped = True
                break

            self._current_day += 1
            day_text, day_strength, active_timeline = self._build_day_policy_text(self._current_day)
            if day_text:
                self._environment.schedule_policy_shock(
                    day_text,
                    intensity=float(day_strength),
                    policy_id=f"session_day_{self._current_day}",
                    source="policy_session",
                    metadata={
                        "day_index": int(self._current_day),
                        "active_policies": list(active_timeline),
                    },
                )
            report = dict(await self._environment.simulation_step())
            daily_row = self._record_for_day(self._current_day, report, active_timeline)
            self._daily_rows.append(daily_row)
            self._last_result = report

        if self._current_day >= int(self.config.total_days):
            self._status = "completed"
            self._stopped = True
        return self._result_payload()

    def advance(self, days: int = 1) -> Dict[str, Any]:
        return _run_sync(self.advance_async(days=days))

    def run(self) -> Dict[str, Any]:
        remaining = max(0, int(self.config.total_days) - int(self._current_day))
        return self.advance(days=remaining)

    def _result_payload(self) -> Dict[str, Any]:
        frame = self.to_frame()
        summary = self._summary_payload()
        active_policies = [plan.to_timeline_row(self._current_day) for plan in self._active_policies(self._current_day)]
        queued_policies = [plan.to_timeline_row(self._current_day) for plan in self._queued_policies(self._current_day)]
        report_payload = self.build_report_payload()
        result = {
            "frame": frame,
            "current_day": int(self._current_day),
            "active_policies": active_policies,
            "queued_policies": queued_policies,
            "summary": summary,
            "report_payload": report_payload,
            "status": self.status,
        }
        if self._last_result:
            result["last_step_report"] = dict(self._last_result)
        return result


__all__ = ["PolicyPlan", "PolicySession", "PolicySessionConfig"]
