# file: agents/manager_agent.py
"""
ManagerAgent 兼容入口。

将 TraderAgent 的能力以 ManagerAgent 名称对外暴露，满足角色解耦后的主节点语义，
并保持旧接口的稳定性。
"""

from agents.trader_agent import TraderAgent


ManagerAgent = TraderAgent

__all__ = ["ManagerAgent"]
