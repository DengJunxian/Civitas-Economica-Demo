# file: tests/test_ipc.py
import asyncio
import pytest
import json

from core.ipc.engine_server import IPCEngineServer
from core.ipc.agent_node import IPCAgentNode
from core.ipc.message_types import MarketStatePacket
from agents.persona import Persona
from core.market_engine import Order, OrderType

@pytest.mark.asyncio
async def test_ipc_communication_loop():
    """验证 EngineServer 和 AgentNode 之间的通信回路"""
    pub_port = 15555
    pull_port = 15556
    
    server = IPCEngineServer(pub_port=pub_port, pull_port=pull_port)
    
    # 构建 Mock Model Router 避免真实调用
    class MockRouter:
        async def call_with_fallback(self, *args, **kwargs):
            return '{"action": "BUY", "qty": 100, "reasoning": "mock"}', "MockReasoning", "mock-model"
    
    # 启动 Node
    persona = Persona(name="TestNode")
    node = IPCAgentNode(
        agent_id="Test_1", 
        persona=persona, 
        pub_port=pub_port, 
        pull_port=pull_port,
        model_router=MockRouter()
    )
    
    # 建立一条独立的 Node 监听任务
    node_task = asyncio.create_task(node.run_loop())
    
    # 等待 ZMQ 底层套接字建立连接
    await asyncio.sleep(0.5)
    
    # 1. 服务端发送广播
    state_pkt = MarketStatePacket(
        step=1,
        timestamp="2026-03-10",
        price=3200.0,
        trend="上涨",
        panic_level=0.1,
        csad=0.05,
        volatility=0.01,
        recent_news=["A股大涨"]
    )
    await server.broadcast_market_state(state_pkt)
    
    # 2. 服务端收集结果 (期待 1 个)
    actions = await server.collect_agent_actions(collect_window=2.0, expected_count=1)
    
    # 3. 验证结果
    assert len(actions) == 1
    action = actions[0]
    assert action.agent_id == "Test_1"
    assert action.step == 1
    assert action.has_order is True
    assert action.order.side.upper() == "BUY"
    assert action.order.quantity == 100
    
    # 清理
    node_task.cancel()
    server.close()
