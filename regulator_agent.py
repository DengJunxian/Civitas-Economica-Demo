"""
Regulator Agent 顶层框架。

目标：
1. 将“海量交易 Agent + 市场波动”抽象为统一 RL 环境；
2. 用“宏观平稳度 + 崩盘损失率”构建奖励函数；
3. 支持动作空间：印花税、降准、降息、定向辟谣；
4. 提供基础训练循环，输出最优组合调控方案。
"""

from __future__ import annotations

import asyncio
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np


def _run_coro_blocking(coro: Any, *, timeout: float) -> Any:
    """
    同步上下文下安全执行协程，避免“慢速 I/O + 快速状态机”耦合导致的阻塞死锁。

    设计要点：
    1. 当前线程没有事件循环时，直接 `asyncio.run`，路径最短、开销最低。
    2. 当前线程已存在运行中的事件循环时，不能再在同线程上 `future.result()` 阻塞等待，
       否则会卡住事件循环自身。这里改为在独立线程中新建事件循环执行协程，再同步回收结果。
    3. 通过 timeout 对桥接线程做硬约束，避免策略训练在异常协程上无限等待。
    """
    try:
        # 仅用于探测“当前线程是否已有运行中的 loop”
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_box["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - 由主线程统一抛出
            error_box["error"] = exc

    worker = threading.Thread(target=_runner, daemon=True, name="regulator-coro-bridge")
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TimeoutError(f"Coroutine bridge timed out after {timeout:.1f}s")
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


# ==========================================================
# 1) 统一环境接口
# ==========================================================


@dataclass
class EnvObservation:
    """
    监管智能体可观测状态。

    字段说明：
    - macro_stability: 宏观平稳度，越大越稳定 (0~1)
    - crash_loss_rate: 崩盘损失率，越大越糟糕 (0~1)
    - volatility: 市场波动率，越大风险越高
    - panic_level: 市场恐慌因子 (0~1)
    """

    step: int
    macro_stability: float
    crash_loss_rate: float
    volatility: float
    panic_level: float
    agent_count: int
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegulatoryAction:
    """
    监管动作（组合调控）定义。
    """

    stamp_tax_rate: float
    reserve_cut_bps: int
    policy_rate_cut_bps: int
    rumor_refute_strength: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stamp_tax_rate": self.stamp_tax_rate,
            "reserve_cut_bps": self.reserve_cut_bps,
            "policy_rate_cut_bps": self.policy_rate_cut_bps,
            "rumor_refute_strength": self.rumor_refute_strength,
        }

    def describe(self) -> str:
        return (
            f"印花税={self.stamp_tax_rate:.4%}, "
            f"降准={self.reserve_cut_bps}bps, "
            f"降息={self.policy_rate_cut_bps}bps, "
            f"定向辟谣强度={self.rumor_refute_strength:.2f}"
        )


class RegulatoryEnvironment(Protocol):
    """统一环境协议：可替换为真实沙盘或轻量仿真。"""

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        ...

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        ...


# ==========================================================
# 2) 奖励函数
# ==========================================================


@dataclass
class RewardFunction:
    """
    奖励函数：
    - 奖励宏观平稳度；
    - 惩罚崩盘损失率；
    - 额外轻惩罚高波动（防止“表面稳定、底层剧烈震荡”）。
    """

    w_stability: float = 1.8
    w_crash_loss: float = 2.4
    w_volatility: float = 0.4
    w_stability_improve: float = 0.6

    def __call__(self, obs: EnvObservation, prev_obs: Optional[EnvObservation] = None) -> float:
        reward = (
            self.w_stability * float(obs.macro_stability)
            - self.w_crash_loss * float(obs.crash_loss_rate)
            - self.w_volatility * float(obs.volatility)
        )
        if prev_obs is not None:
            reward += self.w_stability_improve * (float(obs.macro_stability) - float(prev_obs.macro_stability))
        return float(reward)


# ==========================================================
# 3) 轻量沙盘环境（默认）
# ==========================================================


class ToyRegulatoryMarketEnv(RegulatoryEnvironment):
    """
    轻量可训练环境。

    说明：
    - 该环境用于快速训练与框架验证；
    - 可在后续替换为 Civitas 真正的多智能体仿真环境适配器。
    """

    def __init__(
        self,
        agent_count: int = 10_000,
        episode_steps: int = 48,
        reward_fn: Optional[RewardFunction] = None,
    ):
        self.agent_count = agent_count
        self.episode_steps = episode_steps
        self.reward_fn = reward_fn or RewardFunction()
        self._rng = random.Random(42)
        self._step = 0
        self._obs = EnvObservation(
            step=0,
            macro_stability=0.5,
            crash_loss_rate=0.2,
            volatility=0.06,
            panic_level=0.4,
            agent_count=self.agent_count,
        )
        self._target_action = RegulatoryAction(
            stamp_tax_rate=0.0005,
            reserve_cut_bps=25,
            policy_rate_cut_bps=10,
            rumor_refute_strength=0.6,
        )

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        if seed is not None:
            self._rng.seed(seed)
        self._step = 0

        # 每个 episode 随机化“外部冲击强度”，模拟不同宏观周期。
        shock = self._rng.uniform(0.0, 0.35)
        self._obs = EnvObservation(
            step=0,
            macro_stability=float(np.clip(0.68 - 0.5 * shock, 0.0, 1.0)),
            crash_loss_rate=float(np.clip(0.08 + 0.6 * shock, 0.0, 1.0)),
            volatility=float(np.clip(0.02 + 0.20 * shock, 0.0, 1.0)),
            panic_level=float(np.clip(0.15 + 0.70 * shock, 0.0, 1.0)),
            agent_count=self.agent_count,
            extras={"shock": shock},
        )
        return self._obs

    def _action_distance(self, action: RegulatoryAction) -> float:
        # 将不同量纲归一化到 [0,1] 后计算距离，距离越小说明越接近“有效组合”。
        tax_diff = abs(action.stamp_tax_rate - self._target_action.stamp_tax_rate) / 0.0015
        rrr_diff = abs(action.reserve_cut_bps - self._target_action.reserve_cut_bps) / 100.0
        rate_diff = abs(action.policy_rate_cut_bps - self._target_action.policy_rate_cut_bps) / 50.0
        rumor_diff = abs(action.rumor_refute_strength - self._target_action.rumor_refute_strength)
        return float(np.clip(0.35 * tax_diff + 0.25 * rrr_diff + 0.20 * rate_diff + 0.20 * rumor_diff, 0.0, 1.0))

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        prev_obs = self._obs
        self._step += 1

        # 控制增益：动作越接近有效组合，越能压制波动和崩盘损失。
        distance = self._action_distance(action)
        control_gain = 1.0 - distance

        noise = self._rng.uniform(-0.015, 0.015)
        next_vol = float(np.clip(0.78 * prev_obs.volatility + 0.14 * (1.0 - control_gain) + 0.10 * prev_obs.panic_level + noise, 0.0, 1.0))
        next_panic = float(
            np.clip(
                0.75 * prev_obs.panic_level
                + 0.16 * (1.0 - control_gain)
                - 0.18 * action.rumor_refute_strength
                + noise,
                0.0,
                1.0,
            )
        )
        next_crash = float(np.clip(0.80 * prev_obs.crash_loss_rate + 0.22 * max(0.0, next_vol + next_panic - 0.85) + noise, 0.0, 1.0))
        next_stability = float(np.clip(1.0 - (0.55 * next_vol + 0.35 * next_panic + 0.45 * next_crash), 0.0, 1.0))

        self._obs = EnvObservation(
            step=self._step,
            macro_stability=next_stability,
            crash_loss_rate=next_crash,
            volatility=next_vol,
            panic_level=next_panic,
            agent_count=self.agent_count,
            extras={"control_gain": control_gain, "distance": distance},
        )

        reward = self.reward_fn(self._obs, prev_obs=prev_obs)
        done = bool(self._step >= self.episode_steps or next_crash >= 0.95)
        info = {"distance": distance, "control_gain": control_gain}
        return self._obs, reward, done, info


# ==========================================================
# 4) 真实 Civitas 环境适配器（可选接入）
# ==========================================================


class CivitasMarketEnvAdapter(RegulatoryEnvironment):
    """
    真实沙盘适配器框架。

    通过 model_factory 注入具体的 CivitasModel/Controller 实例，
    使监管智能体把“成千上万交易 Agent + 市场波动”作为统一环境学习。
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        episode_steps: int = 24,
        reward_fn: Optional[RewardFunction] = None,
    ):
        self.model_factory = model_factory
        self.episode_steps = episode_steps
        self.reward_fn = reward_fn or RewardFunction()
        self.model: Any = None
        self._step = 0
        self._last_obs: Optional[EnvObservation] = None

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        self.model = self.model_factory()
        self._step = 0
        self._last_obs = self._build_observation(step=0)
        return self._last_obs

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        self._apply_action(action)
        self._advance_model_one_step()
        self._step += 1

        obs = self._build_observation(step=self._step)
        reward = self.reward_fn(obs, prev_obs=self._last_obs)
        self._last_obs = obs
        done = bool(self._step >= self.episode_steps or obs.crash_loss_rate > 0.45)
        return obs, reward, done, {"action": action.to_dict()}

    def _apply_action(self, action: RegulatoryAction) -> None:
        """
        将 RL 动作映射到现有监管参数。

        映射策略（可按业务进一步精化）：
        - 印花税 -> policy_manager.tax.rate
        - 降准 -> liquidity_injection 增量
        - 降息 -> risk_free_rate 下调
        - 定向辟谣 -> panic_level 下降
        """
        if self.model is None:
            return

        try:
            market = getattr(self.model, "market_manager", None)
            if market is None:
                return

            pm = getattr(market, "policy_manager", None)
            if pm is not None and hasattr(pm, "set_policy_param"):
                pm.set_policy_param("tax", "rate", float(action.stamp_tax_rate))

            policy = getattr(market, "policy", None)
            if policy is not None:
                if hasattr(policy, "liquidity_injection"):
                    policy.liquidity_injection = float(getattr(policy, "liquidity_injection", 0.0) + action.reserve_cut_bps * 1_000_000.0)
                if hasattr(policy, "risk_free_rate"):
                    policy.risk_free_rate = float(max(0.0, getattr(policy, "risk_free_rate", 0.02) - action.policy_rate_cut_bps / 10_000.0))

            if hasattr(self.model, "panic_level"):
                self.model.panic_level = float(np.clip(self.model.panic_level - 0.25 * action.rumor_refute_strength, 0.0, 1.0))
        except Exception:
            # 顶层框架保持容错，避免某个字段不存在导致训练中断。
            return

    def _advance_model_one_step(self) -> None:
        if self.model is None:
            return
        if hasattr(self.model, "async_step"):
            coro = self.model.async_step()
            self._run_coro(coro)
        elif hasattr(self.model, "step"):
            self.model.step()

    @staticmethod
    def _run_coro(coro: Any) -> Any:
        return _run_coro_blocking(coro, timeout=30.0)

    def _build_observation(self, step: int) -> EnvObservation:
        if self.model is None:
            return EnvObservation(
                step=step,
                macro_stability=0.5,
                crash_loss_rate=0.2,
                volatility=0.1,
                panic_level=0.5,
                agent_count=0,
            )

        price_history = list(getattr(self.model, "price_history", []))
        if len(price_history) >= 3:
            prices = np.array(price_history[-64:], dtype=float)
            returns = np.diff(prices) / np.clip(prices[:-1], 1e-8, None)
            volatility = float(np.std(returns))
            peak = np.maximum.accumulate(prices)
            drawdown = (peak - prices) / np.clip(peak, 1e-8, None)
            crash_loss_rate = float(np.max(drawdown))
        else:
            volatility = 0.0
            crash_loss_rate = 0.0

        panic_level = float(getattr(self.model, "panic_level", 0.0))
        macro_stability = float(np.clip(1.0 - (12.0 * volatility + 0.6 * crash_loss_rate + 0.35 * panic_level), 0.0, 1.0))

        return EnvObservation(
            step=step,
            macro_stability=macro_stability,
            crash_loss_rate=float(np.clip(crash_loss_rate, 0.0, 1.0)),
            volatility=float(np.clip(volatility, 0.0, 1.0)),
            panic_level=float(np.clip(panic_level, 0.0, 1.0)),
            agent_count=int(len(getattr(self.model, "agents", []))),
            extras={"price_len": len(price_history)},
        )


class SimulationControllerEnvAdapter(RegulatoryEnvironment):
    """
    基于 SimulationController 的训练环境适配器。

    适配思路：
    - reset: 创建一个新的 SimulationController（或兼容对象）作为 episode 环境；
    - step: 将监管动作映射到控制器的市场/政策参数，然后执行 run_tick；
    - observation: 从 controller.model 与 controller.market 聚合状态。
    """

    def __init__(
        self,
        controller_factory: Callable[[], Any],
        episode_steps: int = 24,
        reward_fn: Optional[RewardFunction] = None,
    ):
        self.controller_factory = controller_factory
        self.episode_steps = episode_steps
        self.reward_fn = reward_fn or RewardFunction()
        self.controller: Any = None
        self._step = 0
        self._last_obs: Optional[EnvObservation] = None

    def reset(self, seed: Optional[int] = None) -> EnvObservation:
        self.controller = self.controller_factory()
        self._step = 0
        self._last_obs = self._build_observation(step=0)
        return self._last_obs

    def step(self, action: RegulatoryAction) -> Tuple[EnvObservation, float, bool, Dict[str, Any]]:
        self._apply_action(action)
        self._run_controller_tick()
        self._step += 1

        obs = self._build_observation(step=self._step)
        reward = self.reward_fn(obs, prev_obs=self._last_obs)
        self._last_obs = obs
        done = bool(self._step >= self.episode_steps or obs.crash_loss_rate > 0.45)
        return obs, reward, done, {"action": action.to_dict()}

    def _apply_action(self, action: RegulatoryAction) -> None:
        if self.controller is None:
            return
        try:
            market = getattr(self.controller, "market", None)
            if market is None:
                return

            pm = getattr(market, "policy_manager", None)
            if pm is not None and hasattr(pm, "set_policy_param"):
                pm.set_policy_param("tax", "rate", float(action.stamp_tax_rate))

            policy = getattr(market, "policy", None)
            if policy is not None:
                if hasattr(policy, "liquidity_injection"):
                    policy.liquidity_injection = float(getattr(policy, "liquidity_injection", 0.0) + action.reserve_cut_bps * 1_000_000.0)
                if hasattr(policy, "risk_free_rate"):
                    policy.risk_free_rate = float(max(0.0, getattr(policy, "risk_free_rate", 0.02) - action.policy_rate_cut_bps / 10_000.0))

            if hasattr(market, "panic_level"):
                market.panic_level = float(np.clip(market.panic_level - 0.25 * action.rumor_refute_strength, 0.0, 1.0))
            if hasattr(market, "current_news"):
                market.current_news = (
                    f"[RegulatorRefute] 辟谣强度={action.rumor_refute_strength:.2f}，"
                    f"税率={action.stamp_tax_rate:.4%}，降准={action.reserve_cut_bps}bps，降息={action.policy_rate_cut_bps}bps。"
                )
        except Exception:
            return

    def _run_controller_tick(self) -> None:
        if self.controller is None:
            return
        run_tick = getattr(self.controller, "run_tick", None)
        if run_tick is None:
            return
        result = run_tick()
        if asyncio.iscoroutine(result):
            self._run_coro(result)

    @staticmethod
    def _run_coro(coro: Any) -> Any:
        return _run_coro_blocking(coro, timeout=60.0)

    def _build_observation(self, step: int) -> EnvObservation:
        if self.controller is None:
            return EnvObservation(
                step=step,
                macro_stability=0.5,
                crash_loss_rate=0.2,
                volatility=0.1,
                panic_level=0.5,
                agent_count=0,
            )

        model = getattr(self.controller, "model", None)
        market = getattr(self.controller, "market", None)
        price_history = list(getattr(model, "price_history", [])) if model is not None else []

        if len(price_history) >= 3:
            prices = np.array(price_history[-64:], dtype=float)
            returns = np.diff(prices) / np.clip(prices[:-1], 1e-8, None)
            volatility = float(np.std(returns))
            peak = np.maximum.accumulate(prices)
            drawdown = (peak - prices) / np.clip(peak, 1e-8, None)
            crash_loss_rate = float(np.max(drawdown))
        else:
            volatility = 0.0
            crash_loss_rate = 0.0

        panic_level = float(getattr(market, "panic_level", 0.0))
        macro_stability = float(np.clip(1.0 - (12.0 * volatility + 0.6 * crash_loss_rate + 0.35 * panic_level), 0.0, 1.0))

        return EnvObservation(
            step=step,
            macro_stability=macro_stability,
            crash_loss_rate=float(np.clip(crash_loss_rate, 0.0, 1.0)),
            volatility=float(np.clip(volatility, 0.0, 1.0)),
            panic_level=float(np.clip(panic_level, 0.0, 1.0)),
            agent_count=int(len(getattr(model, "agents", []))) if model is not None else 0,
            extras={"price_len": len(price_history), "controller_mode": str(getattr(self.controller, "mode", "UNKNOWN"))},
        )


# ==========================================================
# 5) Regulator RL Agent（Q-Learning 基础版）
# ==========================================================


@dataclass
class TrainingSummary:
    episodes: int
    avg_episode_reward: float
    best_action: RegulatoryAction
    best_action_score: float
    top_actions: List[Tuple[RegulatoryAction, float]]
    q_states: int


class RegulatorAgent:
    """
    监管强化学习智能体（基础实现）。

    算法：
    - 离散动作空间 + 状态分桶 + Q-Learning
    - epsilon-greedy 探索
    """

    def __init__(
        self,
        action_space: Optional[List[RegulatoryAction]] = None,
        alpha: float = 0.15,
        gamma: float = 0.96,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.03,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_space = action_space or self.build_default_action_space()
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = defaultdict(self._q_init)
        self.action_reward_sum = np.zeros(len(self.action_space), dtype=float)
        self.action_count = np.zeros(len(self.action_space), dtype=float)

    def _q_init(self) -> np.ndarray:
        return np.zeros(len(self.action_space), dtype=float)

    @staticmethod
    def build_default_action_space() -> List[RegulatoryAction]:
        """
        构建组合调控动作空间。
        """
        stamp_tax_candidates = [0.0003, 0.0005, 0.0008, 0.0010]
        reserve_cut_candidates = [0, 25, 50]
        rate_cut_candidates = [0, 10, 25]
        rumor_refute_candidates = [0.0, 0.4, 0.7, 1.0]

        actions: List[RegulatoryAction] = []
        for tax in stamp_tax_candidates:
            for rrr in reserve_cut_candidates:
                for rate in rate_cut_candidates:
                    for rumor in rumor_refute_candidates:
                        actions.append(
                            RegulatoryAction(
                                stamp_tax_rate=float(tax),
                                reserve_cut_bps=int(rrr),
                                policy_rate_cut_bps=int(rate),
                                rumor_refute_strength=float(rumor),
                            )
                        )
        return actions

    @staticmethod
    def discretize_state(obs: EnvObservation) -> Tuple[int, int, int, int]:
        """
        连续状态离散化。

        说明：基础版用分桶降低状态维度，便于稳定训练和可解释输出。
        """
        vol_bucket = int(np.digitize(obs.volatility, bins=[0.01, 0.02, 0.04, 0.08, 0.15]))
        crash_bucket = int(np.digitize(obs.crash_loss_rate, bins=[0.02, 0.05, 0.10, 0.20, 0.35]))
        stability_bucket = int(np.digitize(obs.macro_stability, bins=[0.2, 0.4, 0.6, 0.8]))
        panic_bucket = int(np.digitize(obs.panic_level, bins=[0.1, 0.3, 0.5, 0.7, 0.9]))
        return vol_bucket, crash_bucket, stability_bucket, panic_bucket

    def _select_action_index(self, state: Tuple[int, int, int, int]) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.action_space))
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def _update_q(
        self,
        state: Tuple[int, int, int, int],
        action_idx: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
    ) -> None:
        q = self.q_table[state][action_idx]
        q_next_max = float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * q_next_max
        self.q_table[state][action_idx] = q + self.alpha * (td_target - q)

    def train(
        self,
        env: RegulatoryEnvironment,
        episodes: int = 500,
        max_steps_per_episode: int = 96,
    ) -> TrainingSummary:
        episode_rewards: List[float] = []

        for ep in range(episodes):
            obs = env.reset(seed=ep)
            state = self.discretize_state(obs)
            total_reward = 0.0

            for _ in range(max_steps_per_episode):
                action_idx = self._select_action_index(state)
                action = self.action_space[action_idx]

                next_obs, reward, done, _info = env.step(action)
                next_state = self.discretize_state(next_obs)
                self._update_q(state, action_idx, reward, next_state)

                self.action_reward_sum[action_idx] += reward
                self.action_count[action_idx] += 1.0
                total_reward += reward
                state = next_state
                if done:
                    break

            episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        mean_scores = self.action_reward_sum / np.clip(self.action_count, 1.0, None)
        best_idx = int(np.argmax(mean_scores))
        ranked_idx = np.argsort(mean_scores)[::-1][:5]
        top_actions = [(self.action_space[int(i)], float(mean_scores[int(i)])) for i in ranked_idx]

        return TrainingSummary(
            episodes=episodes,
            avg_episode_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            best_action=self.action_space[best_idx],
            best_action_score=float(mean_scores[best_idx]),
            top_actions=top_actions,
            q_states=len(self.q_table),
        )


# ==========================================================
# 6) 一键运行入口
# ==========================================================


def run_regulatory_optimization(
    episodes: int = 1200,
    max_steps_per_episode: int = 72,
    use_toy_env: bool = True,
    env: Optional[RegulatoryEnvironment] = None,
) -> TrainingSummary:
    """
    一键训练入口。

    默认使用 Toy 环境做框架验证；接入真实沙盘时传入 env 参数即可。
    """
    if env is None:
        if not use_toy_env:
            raise ValueError("use_toy_env=False 时必须显式传入真实环境实例 env。")
        env = ToyRegulatoryMarketEnv()

    agent = RegulatorAgent()
    return agent.train(env=env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)


def build_simulation_controller_factory(
    deepseek_key: str,
    zhipu_key: Optional[str] = None,
    mode: str = "FAST",
    quant_manager: Optional[Any] = None,
    regulatory_module: Optional[Any] = None,
) -> Callable[[], Any]:
    """
    构建 SimulationController 工厂。

    说明：
    - 延迟导入，避免在未安装完整依赖或仅跑单测时触发重型初始化；
    - 供 SimulationControllerEnvAdapter 使用。
    """

    def _factory() -> Any:
        from core.scheduler import SimulationController

        return SimulationController(
            deepseek_key=deepseek_key,
            zhipu_key=zhipu_key,
            mode=mode,
            quant_manager=quant_manager,
            regulatory_module=regulatory_module,
        )

    return _factory


def run_controller_regulatory_optimization(
    controller_factory: Callable[[], Any],
    episodes: int = 120,
    max_steps_per_episode: int = 24,
) -> TrainingSummary:
    """
    在真实 SimulationController 环境上训练监管智能体。
    """
    env = SimulationControllerEnvAdapter(
        controller_factory=controller_factory,
        episode_steps=max_steps_per_episode,
    )
    agent = RegulatorAgent()
    return agent.train(env=env, episodes=episodes, max_steps_per_episode=max_steps_per_episode)


def training_summary_to_dict(summary: TrainingSummary) -> Dict[str, Any]:
    """
    将训练结果转换为可 JSON 序列化结构。
    """
    return {
        "episodes": int(summary.episodes),
        "avg_episode_reward": float(summary.avg_episode_reward),
        "best_action_score": float(summary.best_action_score),
        "best_action": summary.best_action.to_dict(),
        "best_action_description": summary.best_action.describe(),
        "top_actions": [
            {
                "score": float(score),
                "action": action.to_dict(),
                "description": action.describe(),
            }
            for action, score in summary.top_actions
        ],
        "q_states": int(summary.q_states),
    }


if __name__ == "__main__":
    summary = run_regulatory_optimization(episodes=600, max_steps_per_episode=64, use_toy_env=True)
    print("=== Regulator Agent Training Summary ===")
    print(f"Episodes: {summary.episodes}")
    print(f"Avg Episode Reward: {summary.avg_episode_reward:.4f}")
    print(f"Q States: {summary.q_states}")
    print(f"Best Action Score: {summary.best_action_score:.4f}")
    print(f"Best Action: {summary.best_action.describe()}")
    print("Top Action Bundles:")
    for idx, (action, score) in enumerate(summary.top_actions, start=1):
        print(f"{idx}. score={score:.4f} -> {action.describe()}")
