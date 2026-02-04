# core/society 模块
"""
社会网络模块

包含:
- SocialGraph 社会网络图
- InformationDiffusion SIR 情绪传播
- 羊群效应动态
"""

from core.society.network import (
    SocialGraph,
    InformationDiffusion,
    AgentNode,
    SentimentState,
)

__all__ = [
    "SocialGraph",
    "InformationDiffusion",
    "AgentNode",
    "SentimentState",
]
