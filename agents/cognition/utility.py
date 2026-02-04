# file: agents/cognition/utility.py
"""
前景理论效用模块

实现 Kahneman & Tversky (1979) 的前景理论 (Prospect Theory)：
- 价值函数：收益呈凹形，损失呈凸形且更陡峭
- 概率权重函数：小概率高估，大概率低估
- 损失厌恶：损失的心理痛苦约为等量收益快乐的 2.25 倍

参考文献:
- Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty.

作者: Civitas Economica Team
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum


# ==========================================
# 投资者类型枚举
# ==========================================

class InvestorType(str, Enum):
    """投资者类型"""
    PANIC_RETAIL = "panic_retail"       # 恐慌型散户
    NORMAL = "normal"                   # 普通投资者
    DISCIPLINED_QUANT = "disciplined_quant"  # 纪律型量化


# ==========================================
# 前景理论参数预设
# ==========================================

INVESTOR_PRESETS: Dict[InvestorType, Dict[str, float]] = {
    InvestorType.PANIC_RETAIL: {
        "alpha": 0.88,
        "beta": 0.88,
        "lambda_coeff": 3.0,  # 损失时极度痛苦
        "gamma": 0.61,
        "delta": 0.69,
        "description": "恐慌型散户：损失厌恶系数极高，容易在亏损时做出非理性决策"
    },
    InvestorType.NORMAL: {
        "alpha": 0.88,
        "beta": 0.88,
        "lambda_coeff": 2.25,  # 标准前景理论参数
        "gamma": 0.61,
        "delta": 0.69,
        "description": "普通投资者：符合标准前景理论预测的行为模式"
    },
    InvestorType.DISCIPLINED_QUANT: {
        "alpha": 0.95,
        "beta": 0.95,
        "lambda_coeff": 1.0,  # 理性决策者，损失收益对称
        "gamma": 0.80,
        "delta": 0.80,
        "description": "纪律型量化：接近理性人假设，损失厌恶较低"
    },
}


# ==========================================
# 核心效用函数
# ==========================================

def calculate_prospect_value(
    pnl: float,
    alpha: float = 0.88,
    beta: float = 0.88,
    lambda_coeff: float = 2.25
) -> float:
    """
    计算前景理论价值函数
    
    公式:
    V(x) = x^α           if x >= 0 (收益)
    V(x) = -λ * (-x)^β   if x < 0  (损失)
    
    Args:
        pnl: 盈亏比例 (如 0.05 表示 +5%, -0.03 表示 -3%)
        alpha: 收益敏感度参数 (0 < α < 1)，控制收益区域的凹度
        beta: 损失敏感度参数 (0 < β < 1)，控制损失区域的凸度
        lambda_coeff: 损失厌恶系数 (λ > 1)，典型值 2.25
        
    Returns:
        效用值 (主观价值)
        
    Examples:
        >>> calculate_prospect_value(0.05)  # 5% 收益
        0.0676...
        >>> calculate_prospect_value(-0.05)  # 5% 损失
        -0.152...  # 损失的痛苦约为收益快乐的 2.25 倍
    """
    if pnl >= 0:
        # 收益区域：凹形曲线 (边际效用递减)
        return pow(pnl, alpha)
    else:
        # 损失区域：凸形曲线且更陡峭 (损失厌恶)
        return -lambda_coeff * pow(-pnl, beta)


def weight_probability(
    p: float,
    gamma: float = 0.61,
    delta: float = 0.69
) -> Tuple[float, float]:
    """
    计算概率权重函数 (Probability Weighting Function)
    
    Prelec (1998) 形式:
    w+(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ)   收益权重
    w-(p) = p^δ / (p^δ + (1-p)^δ)^(1/δ)   损失权重
    
    特点:
    - 小概率事件被高估 (彩票效应)
    - 高概率事件被低估 (保险效应)
    
    Args:
        p: 客观概率 [0, 1]
        gamma: 收益侧曲率参数
        delta: 损失侧曲率参数
        
    Returns:
        (w_plus, w_minus): 收益权重和损失权重
    """
    if p <= 0:
        return 0.0, 0.0
    if p >= 1:
        return 1.0, 1.0
    
    # 收益侧权重
    p_gamma = pow(p, gamma)
    one_minus_p_gamma = pow(1 - p, gamma)
    w_plus = p_gamma / pow(p_gamma + one_minus_p_gamma, 1 / gamma)
    
    # 损失侧权重
    p_delta = pow(p, delta)
    one_minus_p_delta = pow(1 - p, delta)
    w_minus = p_delta / pow(p_delta + one_minus_p_delta, 1 / delta)
    
    return w_plus, w_minus


# ==========================================
# 前景理论类
# ==========================================

@dataclass
class ProspectValue:
    """前景理论计算结果"""
    raw_pnl: float              # 原始盈亏
    subjective_value: float     # 主观价值
    pain_gain_ratio: float      # 痛苦/快乐比率
    emotional_intensity: float  # 情绪强度 [0, 1]
    decision_bias: str          # 决策偏差描述


class ProspectTheory:
    """
    前景理论效用计算器
    
    模拟人类投资者的非理性决策行为：
    1. 损失厌恶：对损失的敏感度高于收益
    2. 参照点依赖：以初始成本作为参照点
    3. 边际效用递减：收益越大，边际快乐越小
    4. 风险偏好反转：盈利时保守，亏损时冒险
    
    Attributes:
        alpha: 收益敏感度
        beta: 损失敏感度
        lambda_coeff: 损失厌恶系数
        gamma: 收益概率权重参数
        delta: 损失概率权重参数
        investor_type: 投资者类型
    """
    
    def __init__(
        self,
        alpha: float = 0.88,
        beta: float = 0.88,
        lambda_coeff: float = 2.25,
        gamma: float = 0.61,
        delta: float = 0.69,
        investor_type: Optional[InvestorType] = None
    ):
        """
        初始化前景理论计算器
        
        Args:
            alpha: 收益敏感度参数 (0 < α < 1)
            beta: 损失敏感度参数 (0 < β < 1)
            lambda_coeff: 损失厌恶系数 (λ > 1)
            gamma: 收益概率权重参数
            delta: 损失概率权重参数
            investor_type: 使用预设的投资者类型 (覆盖其他参数)
        """
        if investor_type is not None:
            preset = INVESTOR_PRESETS[investor_type]
            self.alpha = preset["alpha"]
            self.beta = preset["beta"]
            self.lambda_coeff = preset["lambda_coeff"]
            self.gamma = preset["gamma"]
            self.delta = preset["delta"]
            self.investor_type = investor_type
        else:
            self.alpha = alpha
            self.beta = beta
            self.lambda_coeff = lambda_coeff
            self.gamma = gamma
            self.delta = delta
            self.investor_type = InvestorType.NORMAL
        
        # 参数校验
        assert 0 < self.alpha <= 1, f"alpha must be in (0, 1], got {self.alpha}"
        assert 0 < self.beta <= 1, f"beta must be in (0, 1], got {self.beta}"
        assert self.lambda_coeff > 0, f"lambda_coeff must be > 0, got {self.lambda_coeff}"
    
    def calculate_value(self, pnl: float) -> float:
        """
        计算前景理论效用值
        
        Args:
            pnl: 盈亏比例 (如 0.05 表示 +5%)
            
        Returns:
            主观效用值
        """
        return calculate_prospect_value(
            pnl, self.alpha, self.beta, self.lambda_coeff
        )
    
    def calculate_full(self, pnl: float) -> ProspectValue:
        """
        计算完整的前景理论结果
        
        Args:
            pnl: 盈亏比例
            
        Returns:
            ProspectValue 对象，包含详细分析
        """
        value = self.calculate_value(pnl)
        
        # 计算痛苦/快乐比率
        if pnl >= 0 and pnl > 0:
            # 等量损失的痛苦 / 当前收益的快乐
            loss_pain = -self.calculate_value(-pnl)
            pain_gain_ratio = loss_pain / value if value > 0 else self.lambda_coeff
        elif pnl < 0:
            # 当前损失的痛苦 / 等量收益的快乐
            gain_joy = self.calculate_value(-pnl)
            pain_gain_ratio = -value / gain_joy if gain_joy > 0 else self.lambda_coeff
        else:
            pain_gain_ratio = 1.0
        
        # 情绪强度 (归一化到 [0, 1])
        # 使用 sigmoid 函数映射
        emotional_intensity = 1.0 / (1.0 + math.exp(-10 * abs(value)))
        
        # 决策偏差描述
        if pnl < -0.10:
            decision_bias = "极度恐惧：可能非理性死扛或恐慌抛售"
        elif pnl < -0.03:
            decision_bias = "损失厌恶：倾向冒险以期回本"
        elif pnl > 0.10:
            decision_bias = "过度自信：可能忽视回撤风险"
        elif pnl > 0.03:
            decision_bias = "处置效应：倾向过早获利了结"
        else:
            decision_bias = "相对理性：情绪影响较小"
        
        return ProspectValue(
            raw_pnl=pnl,
            subjective_value=value,
            pain_gain_ratio=pain_gain_ratio,
            emotional_intensity=emotional_intensity,
            decision_bias=decision_bias
        )
    
    def should_override_decision(
        self,
        llm_action: str,
        pnl: float,
        fear_threshold: float = -0.5,
        greed_threshold: float = 0.3
    ) -> Tuple[str, Optional[str]]:
        """
        判断是否应该覆盖 LLM 的决策
        
        基于前景理论效用值，在极端情绪下覆盖 LLM 决策。
        
        Args:
            llm_action: LLM 建议的动作 ("BUY", "SELL", "HOLD")
            pnl: 当前盈亏比例
            fear_threshold: 恐惧覆盖阈值 (效用值低于此值触发)
            greed_threshold: 贪婪覆盖阈值 (效用值高于此值触发)
            
        Returns:
            (final_action, override_reason): 最终动作和覆盖原因
        """
        value = self.calculate_value(pnl)
        
        # 规则 1: 极度恐惧时，覆盖买入为观望
        if llm_action == "BUY" and value < fear_threshold:
            return "HOLD", f"前景值 {value:.3f} < {fear_threshold}，恐惧情绪覆盖买入决策"
        
        # 规则 2: 极度恐惧时，可能加速卖出
        if llm_action == "HOLD" and value < fear_threshold * 1.5:
            return "SELL", f"前景值 {value:.3f} 极低，恐慌情绪触发卖出"
        
        # 规则 3: 贪婪时，覆盖观望为买入 (谨慎使用)
        # 此规则可能导致过度交易，默认不启用
        # if llm_action == "HOLD" and value > greed_threshold:
        #     return "BUY", f"前景值 {value:.3f} > {greed_threshold}，贪婪情绪驱动买入"
        
        # 规则 4: 盈利时的处置效应
        if llm_action == "BUY" and pnl > 0.05 and value > 0:
            # 已盈利时不应追加，应考虑获利了结
            return "HOLD", f"已盈利 {pnl:.2%}，处置效应建议观望而非追加"
        
        return llm_action, None
    
    def get_risk_preference(self, pnl: float) -> str:
        """
        获取当前盈亏状态下的风险偏好
        
        前景理论的核心发现之一：
        - 收益域：风险厌恶 (Risk Averse)
        - 损失域：风险寻求 (Risk Seeking)
        
        Args:
            pnl: 盈亏比例
            
        Returns:
            风险偏好描述
        """
        if pnl > 0.02:
            return "风险厌恶：盈利时倾向保守，落袋为安"
        elif pnl < -0.02:
            return "风险寻求：亏损时倾向冒险，试图回本"
        else:
            return "风险中性：盈亏较小时较为理性"
    
    def __repr__(self) -> str:
        return (
            f"ProspectTheory(type={self.investor_type.value}, "
            f"α={self.alpha}, β={self.beta}, λ={self.lambda_coeff})"
        )


# ==========================================
# 便捷工厂函数
# ==========================================

def create_panic_retail() -> ProspectTheory:
    """创建恐慌型散户"""
    return ProspectTheory(investor_type=InvestorType.PANIC_RETAIL)


def create_normal_investor() -> ProspectTheory:
    """创建普通投资者"""
    return ProspectTheory(investor_type=InvestorType.NORMAL)


def create_disciplined_quant() -> ProspectTheory:
    """创建纪律型量化"""
    return ProspectTheory(investor_type=InvestorType.DISCIPLINED_QUANT)


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("前景理论效用函数测试")
    print("=" * 60)
    
    # 创建不同类型的投资者
    panic = create_panic_retail()
    normal = create_normal_investor()
    quant = create_disciplined_quant()
    
    # 测试不同盈亏水平
    pnl_levels = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]
    
    print(f"\n{'PnL':>8} | {'恐慌散户':>10} | {'普通投资者':>10} | {'量化':>10}")
    print("-" * 50)
    
    for pnl in pnl_levels:
        v_panic = panic.calculate_value(pnl)
        v_normal = normal.calculate_value(pnl)
        v_quant = quant.calculate_value(pnl)
        print(f"{pnl:>8.2%} | {v_panic:>10.4f} | {v_normal:>10.4f} | {v_quant:>10.4f}")
    
    print("\n" + "-" * 50)
    
    # 测试决策覆盖
    print("\n决策覆盖测试:")
    test_cases = [
        ("BUY", -0.08),   # 大亏时买入
        ("HOLD", -0.15),  # 巨亏时观望
        ("BUY", 0.06),    # 盈利时追加
        ("SELL", 0.03),   # 小盈时卖出
    ]
    
    for action, pnl in test_cases:
        final, reason = panic.should_override_decision(action, pnl)
        override = "→ " + final if reason else "(保持)"
        print(f"  LLM={action}, PnL={pnl:>+.2%}: {override}")
        if reason:
            print(f"    原因: {reason}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
