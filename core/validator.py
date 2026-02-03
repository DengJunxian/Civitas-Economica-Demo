# file: core/validator.py
"""
典型事实验证器 (Stylized Facts Validator)

用于验证仿真市场是否表现出真实市场的统计特征，
包括尖峰厚尾、波动率聚集、量价相关性等。
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """单项验证结果"""
    name: str
    passed: bool
    actual_value: float
    threshold: float
    description: str


class StylizedFactsValidator:
    """
    典型事实验证器
    
    验证仿真数据是否符合真实金融市场的统计特征：
    1. 尖峰厚尾 (Fat Tails) - 收益率分布峰度 > 3
    2. 波动率聚集 (Volatility Clustering) - 收益率平方的自相关性
    3. 量价相关性 - 价格波动与成交量的相关系数
    4. 收益率均值接近零 - 日收益率均值接近0
    5. 负偏度 - 市场下跌时波动更大
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def validate_fat_tails(self, returns: np.ndarray, threshold: float = 3.0) -> ValidationResult:
        """
        验证尖峰厚尾特征
        
        真实市场的收益率分布具有"尖峰厚尾"特征，即峰度(Kurtosis)显著大于正态分布的3。
        这意味着极端事件（大涨大跌）发生的频率比正态分布预期的要高。
        
        Args:
            returns: 收益率序列
            threshold: 峰度阈值，默认3（正态分布的峰度）
        """
        if len(returns) < 30:
            return ValidationResult(
                name="尖峰厚尾",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足（需要至少30个数据点）"
            )
        
        kurtosis = stats.kurtosis(returns, fisher=False)  # Fisher=False 返回超额峰度+3
        passed = kurtosis > threshold
        
        return ValidationResult(
            name="尖峰厚尾",
            passed=passed,
            actual_value=round(kurtosis, 3),
            threshold=threshold,
            description=f"峰度={kurtosis:.3f}，{'符合' if passed else '不符合'}真实市场特征（阈值>{threshold}）"
        )
    
    def validate_volatility_clustering(self, returns: np.ndarray, lag: int = 1, threshold: float = 0.1) -> ValidationResult:
        """
        验证波动率聚集特征
        
        真实市场中，大波动后往往跟着大波动，小波动后往往跟着小波动。
        通过检验收益率平方的自相关性来验证。
        
        Args:
            returns: 收益率序列
            lag: 滞后期数
            threshold: 自相关系数阈值
        """
        if len(returns) < 50:
            return ValidationResult(
                name="波动率聚集",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足（需要至少50个数据点）"
            )
        
        squared_returns = returns ** 2
        # 计算滞后自相关
        autocorr = np.corrcoef(squared_returns[:-lag], squared_returns[lag:])[0, 1]
        passed = autocorr > threshold
        
        return ValidationResult(
            name="波动率聚集",
            passed=passed,
            actual_value=round(autocorr, 3),
            threshold=threshold,
            description=f"收益率平方自相关={autocorr:.3f}，{'符合' if passed else '不符合'}聚集特征（阈值>{threshold}）"
        )
    
    def validate_volume_price_correlation(
        self, 
        price_changes: np.ndarray, 
        volumes: np.ndarray, 
        threshold: float = 0.2
    ) -> ValidationResult:
        """
        验证量价相关性
        
        真实市场中，价格剧烈波动时往往伴随着成交量放大。
        
        Args:
            price_changes: 价格变化率的绝对值
            volumes: 成交量序列
            threshold: 相关系数阈值
        """
        if len(price_changes) != len(volumes) or len(price_changes) < 30:
            return ValidationResult(
                name="量价相关性",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="数据不足或长度不匹配"
            )
        
        # 使用价格变化绝对值与成交量的相关性
        abs_changes = np.abs(price_changes)
        correlation = np.corrcoef(abs_changes, volumes)[0, 1]
        
        # 处理 NaN（成交量全为0时可能出现）
        if np.isnan(correlation):
            correlation = 0.0
            
        passed = correlation > threshold
        
        return ValidationResult(
            name="量价相关性",
            passed=passed,
            actual_value=round(correlation, 3),
            threshold=threshold,
            description=f"|价格变化|与成交量相关系数={correlation:.3f}，{'符合' if passed else '不符合'}量价关系（阈值>{threshold}）"
        )
    
    def validate_negative_skewness(self, returns: np.ndarray, threshold: float = 0.0) -> ValidationResult:
        """
        验证负偏度特征
        
        真实市场通常呈现负偏度，即下跌时的波动往往比上涨时更剧烈。
        
        Args:
            returns: 收益率序列
            threshold: 偏度阈值（负偏度应小于此值）
        """
        if len(returns) < 30:
            return ValidationResult(
                name="负偏度",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足"
            )
        
        skewness = stats.skew(returns)
        passed = skewness < threshold
        
        return ValidationResult(
            name="负偏度",
            passed=passed,
            actual_value=round(skewness, 3),
            threshold=threshold,
            description=f"偏度={skewness:.3f}，{'符合' if passed else '不符合'}负偏度特征（阈值<{threshold}）"
        )
    
    def validate_mean_reversion(self, returns: np.ndarray, threshold: float = 0.001) -> ValidationResult:
        """
        验证收益率均值接近零
        
        有效市场假说下，日收益率均值应接近零（扣除无风险利率后）。
        
        Args:
            returns: 收益率序列
            threshold: 均值绝对值阈值
        """
        if len(returns) < 30:
            return ValidationResult(
                name="均值回归",
                passed=False,
                actual_value=0,
                threshold=threshold,
                description="样本量不足"
            )
        
        mean_return = np.mean(returns)
        passed = abs(mean_return) < threshold
        
        return ValidationResult(
            name="均值回归",
            passed=passed,
            actual_value=round(mean_return, 5),
            threshold=threshold,
            description=f"日均收益率={mean_return:.5f}，{'符合' if passed else '不符合'}均值回归（阈值<{threshold}）"
        )
    
    def run_full_validation(
        self, 
        prices: List[float], 
        volumes: Optional[List[int]] = None
    ) -> Dict[str, any]:
        """
        执行完整验证
        
        Args:
            prices: 价格序列（收盘价）
            volumes: 成交量序列（可选）
            
        Returns:
            包含所有验证结果的字典
        """
        self.results = []
        
        # 计算收益率
        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        
        # 1. 尖峰厚尾
        self.results.append(self.validate_fat_tails(returns))
        
        # 2. 波动率聚集
        self.results.append(self.validate_volatility_clustering(returns))
        
        # 3. 负偏度
        self.results.append(self.validate_negative_skewness(returns))
        
        # 4. 均值回归
        self.results.append(self.validate_mean_reversion(returns))
        
        # 5. 量价相关性（如果有成交量数据）
        if volumes is not None and len(volumes) == len(prices):
            volumes_arr = np.array(volumes[1:])  # 与收益率对齐
            self.results.append(self.validate_volume_price_correlation(returns, volumes_arr))
        
        # 统计结果
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        return {
            "passed": passed_count,
            "total": total_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0,
            "results": self.results,
            "summary": self.generate_summary()
        }
    
    def generate_summary(self) -> str:
        """生成验证报告摘要"""
        lines = ["=" * 50, "📊 典型事实验证报告", "=" * 50, ""]
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            lines.append(f"{status} {result.name}")
            lines.append(f"   {result.description}")
            lines.append("")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines.append("-" * 50)
        lines.append(f"通过率: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            lines.append("🎉 仿真市场表现出所有典型事实特征！")
        elif passed >= total * 0.6:
            lines.append("⚠️ 仿真市场基本符合真实市场特征。")
        else:
            lines.append("❗ 仿真市场与真实市场存在较大差异，需调整参数。")
        
        return "\n".join(lines)


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    
    # 模拟具有厚尾特征的收益率（t分布）
    n_days = 200
    returns = np.random.standard_t(df=3, size=n_days) * 0.02
    prices = [3000.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    volumes = [int(1e8 * (1 + abs(r) * 10)) for r in returns]  # 量价正相关
    volumes.insert(0, int(1e8))
    
    # 验证
    validator = StylizedFactsValidator()
    result = validator.run_full_validation(prices, volumes)
    
    print(result["summary"])
