# file: tests/test_graph_memory.py
import pytest
import os
import shutil
import time
import asyncio
from unittest.mock import MagicMock, AsyncMock

from agents.cognition.graph_storage import GraphMemoryBank, KnowledgeCapsule
from agents.cognition.graph_builder import GraphExtractor
from agents.trader_agent import TraderAgent
from agents.persona import Persona

@pytest.fixture
def temp_graph_dir():
    path = "tests/temp_graph_data"
    yield path
    # Cleanup
    if os.path.exists(path):
         shutil.rmtree(path)

def test_graph_storage(temp_graph_dir):
    """测试 GraphMemoryBank 的存储和胶囊机制"""
    bank = GraphMemoryBank(agent_id="test_agent", storage_dir=temp_graph_dir)
    
    # 1. 测试三元组插入与检索
    bank.add_triplet("央行", "发布", "降准政策", 1.0)
    bank.add_triplet("降准政策", "导致", "流动性增加", 0.9)
    bank.add_triplet("流动性增加", "利好", "股市", 0.8)
    
    subgraph = bank.retrieve_subgraph(["流动性"])
    assert "降准政策 -> 导致 -> 流动性增加" in subgraph or "流动性增加 -> 利好 -> 股市" in subgraph
    
    # 2. 测试知识胶囊
    current_time = 1000.0
    bank.add_capsule("降准", "央行降准0.5个百分点", current_time, ttl_seconds=3600)
    
    valid_capsules = bank.get_valid_capsules(current_time + 100)
    assert len(valid_capsules) == 1
    assert "央行降准0.5个百分点" in valid_capsules[0]
    
    invalid_capsules = bank.get_valid_capsules(current_time + 4000)
    assert len(invalid_capsules) == 0

@pytest.mark.asyncio
async def test_graph_extractor():
    """测试 图谱抽取器"""
    mock_router = AsyncMock()
    # Mock LLM 返回标准的 JSON 三元组数组
    mock_response = '''```json
    [
        {"subject": "科技股", "predicate": "暴跌", "target": "情绪恐慌", "weight": 0.8},
        {"subject": "外资", "predicate": "流出", "target": "科技股", "weight": 0.9}
    ]
    ```'''
    mock_router.call_with_fallback.return_value = (mock_response, "", "mock_model")
    
    extractor = GraphExtractor(model_router=mock_router)
    triplets = await extractor.extract_graph("科技股暴跌，外资大量流出导致情绪恐慌。")
    
    assert len(triplets) == 2
    assert triplets[0]["subject"] == "科技股"
    assert triplets[1]["predicate"] == "流出"

@pytest.mark.asyncio
async def test_trader_agent_graph_integration(temp_graph_dir):
    """测试 TraderAgent 集成了新记忆模块"""
    agent = TraderAgent(agent_id="test_trader", persona=Persona())
    agent.graph_memory.storage_dir = temp_graph_dir
    
    # 劫持 _async_extract_and_store 避免真的调用大模型
    agent._async_extract_and_store = AsyncMock()
    
    # 测试 update_memory 的触发
    decision = {"action": "BUY", "qty": 100}
    outcome = {"pnl": -1500.0, "status": "FILLED"} # pnl absolute > 1000
    
    await agent.update_memory(decision, outcome)
    
    # Assert _async_extract_and_store was called
    agent._async_extract_and_store.assert_called_once()
    args, kwargs = agent._async_extract_and_store.call_args
    assert "交易复盘" in args[0]
