# file: tests/test_integration.py
"""
集成测试: Model <-> Agent 交互验证

验证新的 Agentic Architecture 是否在仿真循环中正常工作:
1. Model 初始化与 Agent 创建
2. MarketSnapshot 正确生成
3. Agent 并发 act() 调用
4. 订单提交与撮合
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, patch
from core.mesa.civitas_model import CivitasModel
from core.mesa.civitas_agent import CivitasAgent
from core.types import OrderSide

class TestAgenticIntegration:
    
    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """测试模型和 Agent 初始化"""
        model = CivitasModel(n_agents=10, panic_ratio=0.2, quant_ratio=0.1)
        
        assert len(model.agents) == 10
        assert isinstance(model.agents[0], CivitasAgent)
        # 验证底层 TraderAgent 初始化
        assert hasattr(model.agents[0], "core")
        assert model.agents[0].core.cash_balance > 0

    @pytest.mark.asyncio
    async def test_snapshot_generation_and_step(self):
        """测试市场快照生成逻辑 (隐式在 async_step 中) 和 并行 Act"""
        model = CivitasModel(n_agents=5)
        
        # Mock Brain.think_async globally to avoid API calls
        with patch("agents.brain.DeepSeekBrain.think_async", new_callable=AsyncMock) as mock_think:
            # Setup mock return
            mock_think.return_value = {
                "decision": {"action": "HOLD"},
                "reasoning": "Mocked Reasoning: Just holding",
                "emotion_score": 0.0
            }
            
            # 运行一步
            await model.async_step()
            
            # 验证 mock 被调用
            assert mock_think.called
            assert mock_think.call_count == 5 # 5 agents

        # 检查是否有时钟推进
        assert model.clock.timestamp > 0
        
        # 检查是否有价格历史
        assert len(model.price_history) >= 1

    @pytest.mark.asyncio
    async def test_agent_trade_flow(self):
        """测试 Agent 决策流和交易执行"""
        model = CivitasModel(n_agents=5)
        
        # Mock Brain to return BUY
        with patch("agents.brain.DeepSeekBrain.think_async", new_callable=AsyncMock) as mock_think:
            mock_think.return_value = {
                "decision": {
                    "action": "BUY", 
                    "qty": 100, 
                    "price": 3000.0
                },
                "reasoning": "Mocked Reasoning: Aggressive Buy",
                "emotion_score": 0.8
            }
            
            # 运行一步
            await model.async_step()
            
            # 检查是否有成交量
            # Model 应该提交了 agent orders，但是如果没有卖单，可能不成交 (除非 OrderBook 有做市商或者初始卖单)
            # Market engine 默认会有一些初始挂单吗？
            # 即使不成交，order book 应该有挂单。
            
            # 检查是否有挂单在 order book
            bids, asks = model.market_manager.get_order_book_depth()
            assert len(bids) > 0 # Agents placed buy orders
            
            # 检查 DataCollector 是否收集了数据
            df = model.datacollector.get_model_vars_dataframe()
            assert not df.empty
            assert "Price" in df.columns
            # Assert smart sentiment updated
            assert model.last_smart_sentiment > 0 # Should be 1.0 (all buy)

    @pytest.mark.asyncio
    async def test_full_simulation_run(self):
        """测试完整运行"""
        model = CivitasModel(n_agents=10)
        
        with patch("agents.brain.DeepSeekBrain.think_async", new_callable=AsyncMock) as mock_think:
            mock_think.return_value = {"decision": {"action": "HOLD"}}
            
            await model.run_simulation(n_steps=5)
            
            results = model.get_results()
            assert results["n_steps"] == 5
            assert len(results["price_history"]) >= 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
