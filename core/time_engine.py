# file: core/time_engine.py
"""
离散事件驱动 Time Engine。

设计思想：
- 摒弃固定步长，改为事件驱动调度。
- 系统1（快思考）群体：低精度外推，合并时间片。
- 系统2（慢思考）群体：高精度切片，细粒度演算。
- 推理算力路由：常态事件走本地轻量模型，极端阈值突破才走云端高阶模型。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List


class CognitiveMode(str, Enum):
    SYSTEM1_FAST = "system1_fast"
    SYSTEM2_SLOW = "system2_slow"


class InferenceTier(str, Enum):
    LOCAL_VLLM = "local_vllm_qwen7b"
    CLOUD_API = "cloud_high_iq_api"


@dataclass(order=True)
class ScheduledEvent:
    """离散事件。按照 event_time 升序弹出。"""

    event_time: float
    priority: int
    agent_id: str = field(compare=False)
    event_type: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class AgentRuntimeState:
    agent_id: str
    panic_level: float
    shock_score: float
    mode: CognitiveMode


class TimeEngine:
    """离散事件时间引擎。"""

    def __init__(
        self,
        panic_threshold: float = 0.75,
        shock_threshold: float = 0.8,
        fast_merge_dt: float = 1.0,
        slow_slice_dt: float = 0.1,
        cloud_threshold: float = 0.9,
    ):
        self.panic_threshold = panic_threshold
        self.shock_threshold = shock_threshold
        self.fast_merge_dt = fast_merge_dt
        self.slow_slice_dt = slow_slice_dt
        self.cloud_threshold = cloud_threshold

        self.current_time: float = 0.0
        self._queue: List[ScheduledEvent] = []

    def classify_mode(self, panic_level: float, shock_score: float) -> CognitiveMode:
        """判断 Agent 属于系统1还是系统2。"""
        if panic_level >= self.panic_threshold or shock_score >= self.shock_threshold:
            return CognitiveMode.SYSTEM2_SLOW
        return CognitiveMode.SYSTEM1_FAST

    def route_inference_tier(self, panic_level: float, shock_score: float) -> InferenceTier:
        """算力路由：极端阈值才调用云端高阶模型。"""
        pressure = max(panic_level, shock_score)
        if pressure >= self.cloud_threshold:
            return InferenceTier.CLOUD_API
        return InferenceTier.LOCAL_VLLM

    def next_dt(self, mode: CognitiveMode) -> float:
        """根据认知模式确定下一次调度粒度。"""
        return self.fast_merge_dt if mode == CognitiveMode.SYSTEM1_FAST else self.slow_slice_dt

    def schedule_agent_tick(self, state: AgentRuntimeState, now: float | None = None) -> ScheduledEvent:
        """调度单个 Agent 的下一次事件。"""
        base_time = self.current_time if now is None else now
        dt = self.next_dt(state.mode)
        event = ScheduledEvent(
            event_time=base_time + dt,
            priority=0 if state.mode == CognitiveMode.SYSTEM2_SLOW else 1,
            agent_id=state.agent_id,
            event_type="agent_tick",
            payload={
                "mode": state.mode.value,
                "panic_level": state.panic_level,
                "shock_score": state.shock_score,
                "inference_tier": self.route_inference_tier(state.panic_level, state.shock_score).value,
            },
        )
        heapq.heappush(self._queue, event)
        return event

    def run_until(self, end_time: float, handler: Callable[[ScheduledEvent], None]) -> int:
        """运行到指定时间，逐个消费事件。"""
        processed = 0
        while self._queue and self._queue[0].event_time <= end_time:
            event = heapq.heappop(self._queue)
            self.current_time = event.event_time
            handler(event)
            processed += 1
        self.current_time = max(self.current_time, end_time)
        return processed

    def queue_size(self) -> int:
        return len(self._queue)
