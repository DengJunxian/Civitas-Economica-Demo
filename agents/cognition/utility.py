# file: agents/cognition/utility.py
"""
认知 Agent 工具库

包含:
1. 投资者类型定义
2. 前景理论参数包装器
3. 心理状态跟踪器
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

from core.behavioral_finance import ProspectTheoryParams, prospect_value, prospect_utility

class InvestorType(Enum):
    """投资者类型"""
    NORMAL = "normal"             # 普通投资者 (基准)
    PANIC_RETAIL = "panic"        # 恐慌型散户 (高羊群, 高损失厌恶)
    DISCIPLINED_QUANT = "quant"   # 纪律型量化 (理性, 低偏差)

@dataclass
class ProspectValue:
    """前景理论估值结果"""
    utility: float
    reference_point: float

class ProspectTheory:
    """前景理论计算器包装器"""
    
    def __init__(self, investor_type: InvestorType = InvestorType.NORMAL, lambda_coeff: Optional[float] = None):
        self.params = ProspectTheoryParams()
        
        # 根据类型调整参数
        if lambda_coeff:
            self.params.lambda_ = lambda_coeff
        elif investor_type == InvestorType.PANIC_RETAIL:
            self.params.lambda_ = 3.5  # 更高的损失厌恶
        elif investor_type == InvestorType.DISCIPLINED_QUANT:
            self.params.lambda_ = 1.0  # 风险中性/理性
            self.params.alpha = 1.0    # 线性效用
            self.params.beta = 1.0
            
    def calculate_utility(self, pnl_pct: float) -> float:
        """计算效用值"""
        # 使用 core.behavioral_finance 中的函数
        # 注意: prospect_utility 返回 -300~100，这里可能需要归一化或直接使用
        return prospect_utility(pnl_pct, loss_aversion=self.params.lambda_)

def calculate_prospect_value(amount: float, ref: float, params: ProspectTheoryParams) -> float:
    return prospect_value(amount, ref, params)

class ConfidenceTracker:
    """信心跟踪器"""
    def __init__(self, initial: float = 0.5, decay: float = 0.95):
        self.confidence = initial
        self.decay = decay
        
    def update(self, success: bool):
        if success:
            self.confidence = 0.1 + 0.9 * (self.confidence + 0.1) # Boost
        else:
            self.confidence = max(0.0, self.confidence * self.decay)

class AnchorTracker:
    """锚定效应跟踪器"""
    def __init__(self, initial_price: float):
        self.anchor_price = initial_price
        
    def update(self, current_price: float, weight: float = 0.1):
        # 缓慢移动锚点
        self.anchor_price = self.anchor_price * (1 - weight) + current_price * weight

# 工厂函数
def create_panic_retail():
    return InvestorType.PANIC_RETAIL

def create_normal_investor():
    return InvestorType.NORMAL

def create_disciplined_quant():
    return InvestorType.DISCIPLINED_QUANT
