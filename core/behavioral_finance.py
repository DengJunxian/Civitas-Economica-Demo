# file: core/behavioral_finance.py
"""
行为金融量化库

将人性代码化 - 实现：
1. 前景理论（Prospect Theory）效用计算
2. 羊群效应（Herding Effect）增强检测
3. 心理账户（Mental Accounting）
4. 过度自信（Overconfidence）偏差
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np


# ========== 前景理论 (Prospect Theory) ==========

@dataclass
class ProspectTheoryParams:
    """前景理论参数"""
    alpha: float = 0.88       # 收益区域曲率
    beta: float = 0.88        # 损失区域曲率
    lambda_: float = 2.25     # 损失厌恶系数 (Kahneman-Tversky 经典值)
    gamma: float = 0.61       # 权重函数参数（正面）
    delta: float = 0.69       # 权重函数参数（负面）


def prospect_value(
    outcome: float, 
    reference: float = 0,
    params: Optional[ProspectTheoryParams] = None
) -> float:
    """
    前景理论价值函数 V(x)
    
    Kahneman-Tversky (1992) 累积前景理论价值函数
    
    Args:
        outcome: 实际结果（如盈利金额或收益率）
        reference: 参考点（通常为0或成本价）
        params: 前景理论参数
        
    Returns:
        心理效用值
    """
    if params is None:
        params = ProspectTheoryParams()
    
    x = outcome - reference
    
    if x >= 0:
        # 收益区域：V(x) = x^α
        return np.power(x, params.alpha)
    else:
        # 损失区域：V(x) = -λ * (-x)^β
        return -params.lambda_ * np.power(-x, params.beta)


def prospect_utility(
    gain_pct: float,  # 盈亏百分比，如 0.05 = 5%
    reference: float = 0,
    loss_aversion: float = 2.25
) -> float:
    """
    简化版前景理论效用计算
    
    Args:
        gain_pct: 盈亏百分比
        reference: 参考点
        loss_aversion: 损失厌恶系数
        
    Returns:
        心理效用值（-100 ~ +100 范围）
    """
    x = gain_pct - reference
    
    if x >= 0:
        # 收益带来的快乐
        utility = 100 * np.power(x, 0.88)
    else:
        # 损失带来的痛苦（放大 λ 倍）
        utility = -100 * loss_aversion * np.power(-x, 0.88)
    
    return np.clip(utility, -300, 100)


def probability_weight(p: float, is_gain: bool = True) -> float:
    """
    前景理论概率权重函数 w(p)
    
    人们倾向于高估小概率事件，低估大概率事件
    
    Args:
        p: 客观概率 (0-1)
        is_gain: 是否为收益情景
        
    Returns:
        主观决策权重
    """
    if is_gain:
        gamma = 0.61
        return np.power(p, gamma) / np.power(
            np.power(p, gamma) + np.power(1-p, gamma), 
            1/gamma
        )
    else:
        delta = 0.69
        return np.power(p, delta) / np.power(
            np.power(p, delta) + np.power(1-p, delta), 
            1/delta
        )


def disposition_effect_score(
    current_pnl: float,
    action: str,
    holding_days: int = 1
) -> float:
    """
    处置效应评分
    
    处置效应：投资者倾向于过早卖出盈利股票，过久持有亏损股票
    
    Args:
        current_pnl: 当前盈亏百分比
        action: 采取的动作 (BUY/SELL/HOLD)
        holding_days: 持有天数
        
    Returns:
        处置效应得分 (0-1，越高越受处置效应影响)
    """
    if current_pnl > 0 and action == 'SELL':
        # 盈利就卖 - 典型处置效应
        return min(1.0, 0.5 + current_pnl * 2)
    elif current_pnl < 0 and action == 'HOLD':
        # 亏损死扛 - 典型处置效应
        return min(1.0, 0.5 + abs(current_pnl) * 2 + holding_days * 0.01)
    elif current_pnl < 0 and action == 'BUY':
        # 亏损加仓 - 极端处置效应
        return min(1.0, 0.7 + abs(current_pnl) * 3)
    else:
        return 0.0


# ========== 羊群效应 (Herding Effect) ==========

def calculate_csad(returns: np.ndarray, market_return: float) -> float:
    """
    横截面绝对偏差 (Cross-Sectional Absolute Deviation)
    
    CSAD 衡量个股收益率与市场收益率的分散程度
    CSAD 下降 + 市场大涨/大跌 = 羊群效应
    
    Args:
        returns: 个股收益率数组
        market_return: 市场收益率
        
    Returns:
        CSAD 值
    """
    if len(returns) == 0:
        return 0.0
    
    return np.mean(np.abs(returns - market_return))


def calculate_lsv_herding(
    buy_counts: np.ndarray,
    sell_counts: np.ndarray
) -> Tuple[float, str]:
    """
    Lakonishok-Shleifer-Vishny (LSV) 羊群指标
    
    衡量机构投资者在同一方向交易的程度
    
    Args:
        buy_counts: 各股票的买入人数
        sell_counts: 各股票的卖出人数
        
    Returns:
        (LSV 羊群指标, 羊群方向 'buy'/'sell'/'none')
    """
    total = buy_counts + sell_counts
    valid_mask = total > 0
    
    if not np.any(valid_mask):
        return 0.0, 'none'
    
    buy_ratio = buy_counts[valid_mask] / total[valid_mask]
    expected_ratio = np.mean(buy_ratio)
    
    # LSV = |p(i) - E[p]| - AF
    # AF = Adjustment Factor for finite sample
    af = np.mean(np.abs(buy_ratio - expected_ratio))
    
    # 简化：直接返回买入比例的偏差
    herding_score = np.mean(np.abs(buy_ratio - 0.5))
    
    direction = 'buy' if np.mean(buy_ratio) > 0.5 else 'sell'
    
    return herding_score, direction


def csad_regression(
    returns_history: np.ndarray,  # Shape: (T, N) T期，N只股票
    market_returns: np.ndarray,   # Shape: (T,)
    window: int = 20
) -> Dict[str, float]:
    """
    CSAD 非线性回归检测羊群效应
    
    模型: CSAD_t = α + γ1 * |R_m,t| + γ2 * R_m,t^2 + ε
    
    如果 γ2 显著为负，说明存在羊群效应
    
    Args:
        returns_history: 历史收益率矩阵
        market_returns: 市场收益率序列
        window: 滚动窗口大小
        
    Returns:
        回归系数字典
    """
    T = len(market_returns)
    
    if T < window:
        return {'gamma1': 0.0, 'gamma2': 0.0, 'herding_detected': False}
    
    # 计算每期的 CSAD
    csad_series = np.array([
        calculate_csad(returns_history[t], market_returns[t])
        for t in range(T)
    ])
    
    # 取最近 window 期数据
    csad_recent = csad_series[-window:]
    rm_recent = market_returns[-window:]
    rm_abs = np.abs(rm_recent)
    rm_sq = rm_recent ** 2
    
    # 简化 OLS 回归
    X = np.column_stack([np.ones(window), rm_abs, rm_sq])
    y = csad_recent
    
    try:
        # β = (X'X)^(-1) X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        gamma1 = beta[1]
        gamma2 = beta[2]
        
        # γ2 < -0.01 视为存在羊群效应
        herding_detected = gamma2 < -0.01
        
        return {
            'gamma1': float(gamma1),
            'gamma2': float(gamma2),
            'herding_detected': bool(herding_detected),
            'strength': float(abs(gamma2) if herding_detected else 0.0)
        }
        
    except np.linalg.LinAlgError:
        return {'gamma1': 0.0, 'gamma2': 0.0, 'herding_detected': False}


def herding_intensity(
    csad: float,
    market_return: float,
    baseline_csad: float = 0.02
) -> float:
    """
    羊群效应强度指标
    
    当市场大涨/大跌时，CSAD 应该上升
    如果 CSAD 反而下降，说明存在羊群效应
    
    Args:
        csad: 当前 CSAD 值
        market_return: 市场收益率
        baseline_csad: 基准 CSAD (平静期)
        
    Returns:
        羊群强度 (0-1)
    """
    # 预期 CSAD（市场波动越大，预期 CSAD 越高）
    expected_csad = baseline_csad * (1 + 3 * abs(market_return))
    
    if expected_csad == 0:
        return 0.0
    
    # 实际 CSAD 低于预期 → 羊群效应
    deviation_ratio = (expected_csad - csad) / expected_csad
    
    return np.clip(deviation_ratio, 0, 1)


# ========== 过度自信 (Overconfidence) ==========

def overconfidence_score(
    predicted_returns: List[float],
    actual_returns: List[float]
) -> float:
    """
    过度自信评分
    
    基于预测精度衡量过度自信程度
    
    Args:
        predicted_returns: 预测收益率列表
        actual_returns: 实际收益率列表
        
    Returns:
        过度自信得分 (0-1)
    """
    if len(predicted_returns) < 2:
        return 0.5  # 数据不足，返回中性
    
    predicted = np.array(predicted_returns)
    actual = np.array(actual_returns)
    
    # 预测误差
    errors = np.abs(predicted - actual)
    mean_error = np.mean(errors)
    
    # 预测的绝对值（代表信心程度）
    confidence = np.mean(np.abs(predicted))
    
    if confidence < 0.001:
        return 0.5
    
    # 过度自信 = 高信心 + 高误差
    overconfidence = confidence * mean_error
    
    return np.clip(overconfidence * 10, 0, 1)


def predict_next_return(
    prices: List[float],
    short_window: int = 5,
    long_window: int = 20
) -> Optional[float]:
    """
    行为金融风洞预测：基于历史价格序列预测次日收益。
    结合动量与均值回归的简化模型，用于“内部模拟风洞”拦截。
    """
    if prices is None or len(prices) < max(short_window, long_window) + 1:
        return None
    series = np.array([float(x) for x in prices if x is not None])
    if len(series) < max(short_window, long_window) + 1:
        return None

    short_ma = np.mean(series[-short_window:])
    long_ma = np.mean(series[-long_window:])
    momentum = (short_ma - long_ma) / long_ma if long_ma != 0 else 0.0
    reversion = (series[-1] - long_ma) / long_ma if long_ma != 0 else 0.0
    predicted = 0.6 * momentum - 0.4 * reversion
    return float(np.clip(predicted, -0.05, 0.05))


# ========== 心理账户 (Mental Accounting) ==========

@dataclass
class MentalAccount:
    """心理账户"""
    name: str
    balance: float
    pnl: float
    importance: float  # 0-1 重要性权重


def mental_accounting_bias(
    accounts: List[MentalAccount]
) -> float:
    """
    心理账户偏差评分
    
    投资者倾向于将资金分割到不同"心理账户"
    并对不同账户采取不同的风险态度
    
    Args:
        accounts: 心理账户列表
        
    Returns:
        心理账户偏差得分
    """
    if len(accounts) < 2:
        return 0.0
    
    # 计算各账户盈亏的方差
    pnl_values = [a.pnl for a in accounts]
    pnl_variance = np.var(pnl_values)
    
    # 方差越大 → 心理账户偏差越明显
    return np.clip(pnl_variance * 100, 0, 1)


# ========== 综合行为评估 ==========

@dataclass
class BehavioralProfile:
    """行为画像"""
    loss_aversion: float      # 损失厌恶程度
    herding_tendency: float   # 羊群倾向
    overconfidence: float     # 过度自信
    disposition_effect: float  # 处置效应
    mental_utility: float     # 心理效用
    
    def get_bias_summary(self) -> str:
        """获取偏差摘要"""
        biases = []
        
        if self.loss_aversion > 2.5:
            biases.append("🔴 高度损失厌恶")
        if self.herding_tendency > 0.6:
            biases.append("🐑 易受羊群效应")
        if self.overconfidence > 0.7:
            biases.append("😎 过度自信")
        if self.disposition_effect > 0.5:
            biases.append("💰 处置效应明显")
        
        return " | ".join(biases) if biases else "✅ 行为相对理性"


def create_behavioral_profile(
    pnl_pct: float,
    action: str,
    market_csad: float,
    market_return: float,
    predictions: List[float] = None,
    actuals: List[float] = None,
    loss_aversion: float = 2.25
) -> BehavioralProfile:
    """
    创建行为画像
    
    综合评估投资者的行为偏差
    """
    # 计算各项指标
    mental_utility = prospect_utility(pnl_pct, loss_aversion=loss_aversion)
    herding = herding_intensity(market_csad, market_return)
    disposition = disposition_effect_score(pnl_pct, action)
    
    if predictions and actuals:
        overconf = overconfidence_score(predictions, actuals)
    else:
        overconf = 0.5
    
    return BehavioralProfile(
        loss_aversion=loss_aversion,
        herding_tendency=herding,
        overconfidence=overconf,
        disposition_effect=disposition,
        mental_utility=mental_utility
    )


# ========== Cumulative Prospect Theory (CPT) Profiles ==========

@dataclass
class CPTAgentProfile:
    """Standardized CPT profile for different agent archetypes."""

    name: str
    params: ProspectTheoryParams
    reference_return: float = 0.0


_CPT_PROFILE_MAP: Dict[str, ProspectTheoryParams] = {
    "retail": ProspectTheoryParams(alpha=0.86, beta=0.90, lambda_=2.8, gamma=0.60, delta=0.70),
    "institution": ProspectTheoryParams(alpha=0.92, beta=0.92, lambda_=1.6, gamma=0.72, delta=0.78),
    "quant": ProspectTheoryParams(alpha=0.95, beta=0.95, lambda_=1.2, gamma=0.80, delta=0.82),
}


def get_cpt_profile(agent_type: Literal["retail", "institution", "quant"] | str) -> CPTAgentProfile:
    key = str(agent_type).strip().lower()
    params = _CPT_PROFILE_MAP.get(key, ProspectTheoryParams())
    return CPTAgentProfile(name=key or "custom", params=params)


def cpt_decision_utility(
    outcome: float,
    probability: float,
    *,
    profile: Optional[CPTAgentProfile] = None,
    reference: Optional[float] = None,
) -> float:
    """CPT subjective utility for a single outcome-probability pair."""
    prof = profile or CPTAgentProfile(name="default", params=ProspectTheoryParams())
    ref = prof.reference_return if reference is None else float(reference)
    subjective_value = prospect_value(outcome=outcome, reference=ref, params=prof.params)
    weighted_prob = probability_weight(
        p=float(np.clip(probability, 1e-8, 1 - 1e-8)),
        is_gain=(outcome - ref) >= 0,
    )
    return float(subjective_value * weighted_prob)


def cpt_expected_utility(
    outcomes: List[float],
    probabilities: List[float],
    *,
    profile: Optional[CPTAgentProfile] = None,
    reference: Optional[float] = None,
) -> float:
    """Aggregate CPT utility over a distribution of outcomes."""
    if not outcomes or not probabilities:
        return 0.0
    n = min(len(outcomes), len(probabilities))
    if n == 0:
        return 0.0
    utils = [
        cpt_decision_utility(
            outcome=float(outcomes[i]),
            probability=float(probabilities[i]),
            profile=profile,
            reference=reference,
        )
        for i in range(n)
    ]
    return float(np.sum(utils))


# ========== Reference-Point-Driven Behavioral Layer ==========


def _safe_price(value: float, fallback: float = 1.0) -> float:
    price = float(value)
    if price <= 0:
        return float(max(fallback, 1e-6))
    return price


@dataclass
class ReferencePoints:
    """
    Agent-level reference points used by behavioral finance dynamics.

    All anchors are price-like levels (must be > 0).
    """

    purchase_anchor: float
    recent_high_anchor: float
    peer_anchor: float
    policy_anchor: float

    def normalized_returns(self, current_price: float) -> Dict[str, float]:
        """Return anchor-relative returns at current price."""
        price = _safe_price(current_price)
        return {
            "purchase_anchor": (price - _safe_price(self.purchase_anchor, price)) / _safe_price(self.purchase_anchor, price),
            "recent_high_anchor": (price - _safe_price(self.recent_high_anchor, price)) / _safe_price(self.recent_high_anchor, price),
            "peer_anchor": (price - _safe_price(self.peer_anchor, price)) / _safe_price(self.peer_anchor, price),
            "policy_anchor": (price - _safe_price(self.policy_anchor, price)) / _safe_price(self.policy_anchor, price),
        }


@dataclass
class ReferenceShiftConfig:
    """Smoothing parameters for reference-point shift."""

    purchase_decay: float = 0.03
    recent_high_decay: float = 0.12
    peer_decay: float = 0.18
    policy_decay: float = 0.20
    policy_sensitivity: float = 0.20


@dataclass
class BehavioralStepState:
    """Output state from one behavioral update step."""

    sentiment: float
    reference_points: ReferencePoints
    prospect_direction: float
    risk_appetite: float
    trading_intent: float
    loss_aversion_intensity: float
    weighted_reference_return: float


def initialize_reference_points(
    current_price: float,
    *,
    peer_anchor: Optional[float] = None,
    policy_anchor: Optional[float] = None,
) -> ReferencePoints:
    """Initialize four anchors with sensible defaults."""
    price = _safe_price(current_price)
    return ReferencePoints(
        purchase_anchor=price,
        recent_high_anchor=price,
        peer_anchor=_safe_price(peer_anchor if peer_anchor is not None else price, price),
        policy_anchor=_safe_price(policy_anchor if policy_anchor is not None else price, price),
    )


def update_reference_points(
    reference_points: ReferencePoints,
    *,
    current_price: float,
    peer_anchor: float,
    policy_anchor: float,
    policy_shock: float = 0.0,
    config: Optional[ReferenceShiftConfig] = None,
) -> ReferencePoints:
    """
    Update reference points every step.

    Order requirement mapping:
    sentiment -> reference point shift -> risk appetite -> trading intent
    """
    cfg = config or ReferenceShiftConfig()
    price = _safe_price(current_price)
    old = reference_points

    purchase_target = old.purchase_anchor * (1.0 - cfg.purchase_decay) + price * cfg.purchase_decay
    recent_high_target = max(old.recent_high_anchor, price)
    recent_high = old.recent_high_anchor * (1.0 - cfg.recent_high_decay) + recent_high_target * cfg.recent_high_decay

    peer = old.peer_anchor * (1.0 - cfg.peer_decay) + _safe_price(peer_anchor, price) * cfg.peer_decay
    policy_target = _safe_price(policy_anchor, price) * (1.0 + cfg.policy_sensitivity * float(policy_shock))
    policy = old.policy_anchor * (1.0 - cfg.policy_decay) + policy_target * cfg.policy_decay

    return ReferencePoints(
        purchase_anchor=_safe_price(purchase_target, price),
        recent_high_anchor=_safe_price(recent_high, price),
        peer_anchor=_safe_price(peer, price),
        policy_anchor=_safe_price(policy, price),
    )


def prospect_direction_preference(
    current_price: float,
    reference_points: ReferencePoints,
    *,
    loss_aversion: float = 2.25,
    reference_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Prospect-theory style direction preference from four anchors."""
    returns = reference_points.normalized_returns(current_price)
    weights = {
        "purchase_anchor": 0.34,
        "recent_high_anchor": 0.26,
        "peer_anchor": 0.20,
        "policy_anchor": 0.20,
    }
    if reference_weights:
        weights.update(reference_weights)

    weighted_return = 0.0
    weighted_utility = 0.0
    underwater = 0.0
    total_w = 0.0

    for key, value in returns.items():
        w = float(max(0.0, weights.get(key, 0.0)))
        total_w += w
        weighted_return += w * value
        utility = prospect_utility(value, reference=0.0, loss_aversion=loss_aversion)
        weighted_utility += w * utility
        if value < 0:
            underwater += w

    if total_w <= 0:
        total_w = 1.0
    weighted_return /= total_w
    weighted_utility /= total_w
    underwater_ratio = underwater / total_w

    # Direction in [-1, 1], positive = buy bias, negative = sell bias
    direction = float(np.tanh(weighted_utility / 55.0))
    # Loss-aversion intensity (dynamic) increases when more anchors are underwater
    loss_intensity = float(np.clip(loss_aversion * (1.0 + 0.8 * underwater_ratio), 0.5, 6.0))

    return {
        "direction": direction,
        "weighted_return": float(weighted_return),
        "weighted_utility": float(weighted_utility),
        "underwater_ratio": float(underwater_ratio),
        "loss_aversion_intensity": loss_intensity,
    }


def update_risk_appetite(
    base_risk_appetite: float,
    sentiment: float,
    direction_preference: float,
    loss_aversion_intensity: float,
) -> float:
    """Update risk appetite in [0, 1] using sentiment + prospect signals."""
    base = float(np.clip(base_risk_appetite, 0.0, 1.0))
    senti = float(np.clip(sentiment, -1.0, 1.0))
    direction = float(np.clip(direction_preference, -1.0, 1.0))
    loss_penalty = (float(loss_aversion_intensity) - 1.0) / 5.0
    updated = base + 0.22 * senti + 0.30 * direction - 0.18 * max(0.0, loss_penalty)
    return float(np.clip(updated, 0.0, 1.0))


def build_trading_intent(
    risk_appetite: float,
    direction_preference: float,
    sentiment: float,
) -> float:
    """
    Trading intent score in [-1, 1].
    Positive means buy-side intent, negative means sell-side intent.
    """
    risk = float(np.clip(risk_appetite, 0.0, 1.0))
    direction = float(np.clip(direction_preference, -1.0, 1.0))
    senti = float(np.clip(sentiment, -1.0, 1.0))
    intent = 0.65 * direction + 0.25 * senti + 0.10 * (risk - 0.5) * 2.0
    return float(np.clip(intent, -1.0, 1.0))


def behavioral_update_step(
    *,
    sentiment: float,
    current_price: float,
    reference_points: ReferencePoints,
    base_risk_appetite: float,
    peer_anchor: float,
    policy_anchor: float,
    policy_shock: float,
    loss_aversion: float = 2.25,
    shift_config: Optional[ReferenceShiftConfig] = None,
    reference_weights: Optional[Dict[str, float]] = None,
) -> BehavioralStepState:
    """
    One-step behavioral pipeline:
    sentiment -> reference point shift -> risk appetite -> trading intent
    """
    shifted = update_reference_points(
        reference_points,
        current_price=current_price,
        peer_anchor=peer_anchor,
        policy_anchor=policy_anchor,
        policy_shock=policy_shock,
        config=shift_config,
    )
    prospect = prospect_direction_preference(
        current_price=current_price,
        reference_points=shifted,
        loss_aversion=loss_aversion,
        reference_weights=reference_weights,
    )
    risk = update_risk_appetite(
        base_risk_appetite=base_risk_appetite,
        sentiment=sentiment,
        direction_preference=prospect["direction"],
        loss_aversion_intensity=prospect["loss_aversion_intensity"],
    )
    intent = build_trading_intent(risk, prospect["direction"], sentiment)
    return BehavioralStepState(
        sentiment=float(np.clip(sentiment, -1.0, 1.0)),
        reference_points=shifted,
        prospect_direction=prospect["direction"],
        risk_appetite=risk,
        trading_intent=intent,
        loss_aversion_intensity=prospect["loss_aversion_intensity"],
        weighted_reference_return=prospect["weighted_return"],
    )


# ========== Disposition Effect (PGR / PLR) ==========


@dataclass
class DispositionCounter:
    """Counters for computing PGR / PLR."""

    realized_gains: int = 0
    realized_losses: int = 0
    paper_gains: int = 0
    paper_losses: int = 0

    def record_realized(self, pnl: float) -> None:
        if pnl >= 0:
            self.realized_gains += 1
        else:
            self.realized_losses += 1

    def record_paper(self, pnl: float) -> None:
        if pnl >= 0:
            self.paper_gains += 1
        else:
            self.paper_losses += 1


def calculate_pgr_plr(counter: DispositionCounter) -> Dict[str, float]:
    """Compute realization rates for gains and losses."""
    pgr_den = counter.realized_gains + counter.paper_gains
    plr_den = counter.realized_losses + counter.paper_losses
    pgr = counter.realized_gains / pgr_den if pgr_den > 0 else 0.0
    plr = counter.realized_losses / plr_den if plr_den > 0 else 0.0
    return {
        "pgr": float(pgr),
        "plr": float(plr),
        "disposition_gap": float(pgr - plr),
        "realized_gains": float(counter.realized_gains),
        "realized_losses": float(counter.realized_losses),
        "paper_gains": float(counter.paper_gains),
        "paper_losses": float(counter.paper_losses),
    }


# ========== Stylized Facts Helpers ==========


def all_time_high_effect(returns: Sequence[float], ath_flags: Sequence[bool]) -> Dict[str, float]:
    """Measure return anomaly around all-time-high (ATH) states."""
    if not returns:
        return {
            "ath_hit_rate": 0.0,
            "mean_return_on_ath": 0.0,
            "mean_return_off_ath": 0.0,
            "ath_outperformance": 0.0,
        }
    n = min(len(returns), len(ath_flags))
    if n <= 0:
        return {
            "ath_hit_rate": 0.0,
            "mean_return_on_ath": 0.0,
            "mean_return_off_ath": 0.0,
            "ath_outperformance": 0.0,
        }

    arr = np.asarray([float(x) for x in returns[:n]], dtype=float)
    flags = np.asarray([bool(x) for x in ath_flags[:n]], dtype=bool)
    on = arr[flags]
    off = arr[~flags]
    mean_on = float(np.mean(on)) if on.size else 0.0
    mean_off = float(np.mean(off)) if off.size else 0.0
    return {
        "ath_hit_rate": float(np.mean(flags)),
        "mean_return_on_ath": mean_on,
        "mean_return_off_ath": mean_off,
        "ath_outperformance": float(mean_on - mean_off),
    }


def volatility_clustering_metrics(returns: Sequence[float]) -> Dict[str, float]:
    """Estimate volatility clustering via lag autocorrelation of |r| and r^2."""
    if len(returns) < 3:
        return {
            "abs_return_lag1_autocorr": 0.0,
            "squared_return_lag1_autocorr": 0.0,
            "volatility_level": 0.0,
        }

    arr = np.asarray([float(x) for x in returns], dtype=float)
    abs_r = np.abs(arr)
    sq_r = arr ** 2

    def _lag1_autocorr(series: np.ndarray) -> float:
        x0 = series[:-1]
        x1 = series[1:]
        if x0.size < 2 or float(np.std(x0)) < 1e-12 or float(np.std(x1)) < 1e-12:
            return 0.0
        return float(np.corrcoef(x0, x1)[0, 1])

    return {
        "abs_return_lag1_autocorr": _lag1_autocorr(abs_r),
        "squared_return_lag1_autocorr": _lag1_autocorr(sq_r),
        "volatility_level": float(np.std(arr)),
    }


def drawdown_distribution(prices: Sequence[float]) -> Dict[str, float]:
    """Build drawdown distribution summary."""
    if len(prices) < 2:
        return {
            "mean_drawdown": 0.0,
            "max_drawdown": 0.0,
            "median_drawdown": 0.0,
            "q90_drawdown": 0.0,
            "count": 0.0,
        }
    arr = np.asarray([_safe_price(x) for x in prices], dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = arr / running_max - 1.0
    dd = drawdowns[drawdowns < 0]
    if dd.size == 0:
        return {
            "mean_drawdown": 0.0,
            "max_drawdown": 0.0,
            "median_drawdown": 0.0,
            "q90_drawdown": 0.0,
            "count": 0.0,
        }
    return {
        "mean_drawdown": float(np.mean(dd)),
        "max_drawdown": float(np.min(dd)),
        "median_drawdown": float(np.median(dd)),
        "q90_drawdown": float(np.quantile(dd, 0.9)),
        "count": float(dd.size),
    }


@dataclass
class StylizedFactsTracker:
    """
    Collect per-step market/behavior traces and export stylized facts.
    """

    prices: List[float] = field(default_factory=list)
    market_returns: List[float] = field(default_factory=list)
    csad_series: List[float] = field(default_factory=list)
    cross_sectional_returns: List[List[float]] = field(default_factory=list)
    ath_flags: List[bool] = field(default_factory=list)
    loss_aversion_intensity: List[float] = field(default_factory=list)
    disposition: DispositionCounter = field(default_factory=DispositionCounter)

    def record_step(
        self,
        *,
        price: float,
        market_return: float,
        csad: float,
        cross_returns: Sequence[float],
        is_all_time_high: bool,
        loss_aversion_intensity: float,
    ) -> None:
        self.prices.append(float(price))
        self.market_returns.append(float(market_return))
        self.csad_series.append(float(csad))
        self.cross_sectional_returns.append([float(x) for x in cross_returns])
        self.ath_flags.append(bool(is_all_time_high))
        self.loss_aversion_intensity.append(float(loss_aversion_intensity))

    def report(self) -> Dict[str, object]:
        csad_mean = float(np.mean(self.csad_series)) if self.csad_series else 0.0
        csad_std = float(np.std(self.csad_series)) if self.csad_series else 0.0
        matrix = np.zeros((0, 0), dtype=float)
        if self.cross_sectional_returns:
            valid_rows = [row for row in self.cross_sectional_returns if row]
            if valid_rows:
                width = min(len(row) for row in valid_rows)
                if width > 0:
                    matrix = np.asarray([row[:width] for row in valid_rows], dtype=float)
        aligned_market_returns = np.asarray(self.market_returns, dtype=float) if self.market_returns else np.zeros(0, dtype=float)
        if matrix.shape[0] > 0 and aligned_market_returns.shape[0] != matrix.shape[0]:
            aligned_market_returns = aligned_market_returns[-matrix.shape[0]:]
        herding = csad_regression(
            returns_history=matrix,
            market_returns=aligned_market_returns,
            window=min(20, int(aligned_market_returns.shape[0])) if aligned_market_returns.size else 20,
        )
        ath = all_time_high_effect(self.market_returns, self.ath_flags)
        vol_cluster = volatility_clustering_metrics(self.market_returns)
        drawdowns = drawdown_distribution(self.prices)
        pgr_plr = calculate_pgr_plr(self.disposition)
        loss_avg = float(np.mean(self.loss_aversion_intensity)) if self.loss_aversion_intensity else 0.0
        loss_p90 = float(np.quantile(self.loss_aversion_intensity, 0.9)) if self.loss_aversion_intensity else 0.0
        payload = {
            "csad": {
                "mean": csad_mean,
                "std": csad_std,
                "herding_regression": herding,
            },
            "pgr_plr": pgr_plr,
            "volatility_clustering": vol_cluster,
            "drawdown_distribution": drawdowns,
            "all_time_high_effect": ath,
            "loss_aversion_intensity": {
                "mean": loss_avg,
                "p90": loss_p90,
                "n_obs": len(self.loss_aversion_intensity),
            },
        }
        # Alias keys kept for compatibility with external evaluators.
        payload["volatility clustering"] = payload["volatility_clustering"]
        payload["drawdown distribution"] = payload["drawdown_distribution"]
        payload["all_time_high effect"] = payload["all_time_high_effect"]
        return payload

    def save_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.report()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
