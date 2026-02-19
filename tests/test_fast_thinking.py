# file: tests/test_fast_thinking.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from agents.trader_agent import TraderAgent
from agents.base_agent import MarketSnapshot

def create_snapshot(price=10.0, trend=0.0):
    return MarketSnapshot(
        symbol="TEST",
        last_price=price,
        best_bid=price-0.1,
        best_ask=price+0.1,
        market_trend=trend,
        panic_level=0.0,
        timestamp=1000.0,
        mid_price=price,
        bid_ask_spread=0.2
    )

class TestFastThinking:
    
    @pytest.mark.asyncio
    async def test_fast_think_triggered(self):
        """验证在无新闻、无波动时触发快思考"""
        agent = TraderAgent("fast_agent")
        snapshot = create_snapshot(trend=0.0)
        
        # Mock think_async to fail if called (ensure it's NOT called)
        agent.brain.think_async = AsyncMock(side_effect=Exception("Should NOT be called!"))
        
        # Act
        # perceived_data includes news=[], pnl_pct=0
        # social_signal = "Market is quiet."
        # needs_deep_thinking should be False
        
        order = await agent.act(snapshot, [])
        
        # Fast Agent logic for trend=0 is HOLD -> order is None
        assert order is None
        
        # Verify internal state
        assert agent._fast_mode_consecutive_steps == 1
        
    @pytest.mark.asyncio
    async def test_slow_think_trigger_news(self):
        """验证有新闻时触发慢思考"""
        agent = TraderAgent("slow_agent_news")
        snapshot = create_snapshot()
        
        # Mock think_async to succeed
        agent.brain.think_async = AsyncMock(return_value={
            "decision": {"action": "BUY", "qty": 100, "price": 10.0},
            "reasoning": "News is good"
        })
        
        # Act with news
        order = await agent.act(snapshot, ["Big News!"])
        
        assert order is not None
        assert agent._last_news_count == 1
        assert agent._fast_mode_consecutive_steps == 0
        
    @pytest.mark.asyncio
    async def test_slow_think_trigger_social(self):
        """验证社交信号变化触发慢思考"""
        # Give agent some stock to sell
        agent = TraderAgent("slow_agent_social", portfolio={"TEST": 1000})
        snapshot = create_snapshot()
        
        # Initial: Neutral
        agent.brain.think_async = AsyncMock(return_value={"decision": {"action": "HOLD"}})
        await agent.act(snapshot, [])
        
        # Mock social signal change (e.g. by binding to a graph or mocking perceive_social_signal)
        # Here we mock perceive_social_signal method
        agent.perceive_social_signal = MagicMock(return_value="Panic Alert! Everyone selling!")
        
        # Expect slow think
        agent.brain.think_async = AsyncMock(return_value={
            "decision": {"action": "SELL", "qty": 100, "price": 9.0},
            "reasoning": "Panic sell"
        })
        
        order = await agent.act(snapshot, [])
        
        assert order is not None
        assert agent._last_social_sentiment == "bearish"

