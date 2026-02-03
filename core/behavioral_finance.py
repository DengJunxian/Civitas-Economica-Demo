# file: core/behavioral_finance.py
"""
行为金融量化库

将人性代码化 - 实现：
1. 前景理论（Prospect Theory）效用计算
2. 羊群效应（Herding Effect）增强检测
3. 心理账户（Mental Accounting）
4. 过度自信（Overconfidence）偏差
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


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
            'gamma1': gamma1,
            'gamma2': gamma2,
            'herding_detected': herding_detected,
            'strength': abs(gamma2) if herding_detected else 0.0
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
