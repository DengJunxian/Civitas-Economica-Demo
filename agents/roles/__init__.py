# agents/roles 模块
"""
分析师角色模块

包含:
- Analyst 基类
- FundamentalAnalyst 基本面分析
- TechnicalAnalyst 技术分析
- SentimentAnalyst 情绪分析
- Signal 信号数据类
"""

from agents.roles.analyst import (
    Analyst,
    FundamentalAnalyst,
    TechnicalAnalyst,
    SentimentAnalyst,
    Signal,
    SignalType,
    tool,
)

__all__ = [
    "Analyst",
    "FundamentalAnalyst",
    "TechnicalAnalyst",
    "SentimentAnalyst",
    "Signal",
    "SignalType",
    "tool",
]
