# file: tests/test_trader_agent.py
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from typing import List, Dict, Any
from agents.base_agent import MarketSnapshot
from agents.trader_agent import TraderAgent
from core.types import Order, OrderSide, OrderType

# Helper to create a dummy snapshot
def create_snapshot(price=10.0, trend=0.0, panic=0.0) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="TEST",
        last_price=price,
        best_bid=price-0.1,
        best_ask=price+0.1,
        market_trend=trend,
        panic_level=panic,
        timestamp=1000.0,
        mid_price=price,
        bid_ask_spread=0.2
    )

class TestTraderAgent:
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        agent = TraderAgent("agent_001", cash_balance=50000)
        assert agent.agent_id == "agent_001"
        assert agent.cash_balance == 50000 
        assert agent.portfolio == {}
        # Check default profile
        assert agent.profile["risk_aversion"] == 0.5
        # Check Brain initialization
        assert agent.brain is not None
        assert agent.brain.agent_id == "agent_001"

    @pytest.mark.asyncio
    async def test_perceive(self):
        agent = TraderAgent("agent_002")
        snapshot = create_snapshot()
        news = ["News 1", "News 2", "News 3", "News 4"]
        
        # Set attention span to 2
        agent.profile["attention_span"] = 2
        
        perceived = await agent.perceive(snapshot, news)
        
        assert perceived["snapshot"] == snapshot
        assert len(perceived["news"]) == 2
        assert perceived["news"] == ["News 1", "News 2"]
        assert "portfolio_value" in perceived
        assert "cash" in perceived
        assert "pnl_pct" in perceived

    @pytest.mark.asyncio
    async def test_act_buy_signal(self):
        agent = TraderAgent("agent_buy", cash_balance=10000)
        snapshot = create_snapshot(price=10.0)
        
        # Mock Brain.think_async to return BUY decision
        agent.brain.think_async = AsyncMock(return_value={
            "decision": {
                "action": "BUY",
                "qty": 100,
                "price": 10.0
            },
            "reasoning": "Mocked Reasoning: Buy Signal",
            "emotion_score": 0.8
        })
        
        order = await agent.act(snapshot, [])
        
        assert order is not None
        assert isinstance(order, Order)
        assert order.side == OrderSide.BUY
        assert order.symbol == "TEST"
        assert order.quantity == 100
        assert order.price == 10.0
        
    @pytest.mark.asyncio
    async def test_act_sell_signal(self):
        # Setup agent with holdings
        portfolio = {"TEST": 1000}
        agent = TraderAgent("agent_sell", cash_balance=10000, portfolio=portfolio)
        snapshot = create_snapshot(price=10.0)
        
        # Mock Brain.think_async to return SELL decision
        agent.brain.think_async = AsyncMock(return_value={
            "decision": {
                "action": "SELL",
                "qty": 500,
                "price": 9.9
            },
            "reasoning": "Mocked Reasoning: Sell Signal",
            "emotion_score": -0.8
        })
        
        order = await agent.act(snapshot, [])
        
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == 500
        assert order.price == 9.9
        
    @pytest.mark.asyncio
    async def test_fund_check(self):
        # Agent with little cash
        agent = TraderAgent("agent_poor", cash_balance=10) 
        snapshot = create_snapshot(price=10.0)
        
        # Mock decision to buy 100 shares @ 10.0 (Cost 1000 > Cash 10)
        agent.brain.think_async = AsyncMock(return_value={
            "decision": {
                "action": "BUY",
                "qty": 100,
                "price": 10.0
            },
            "reasoning": "I want to buy but I am poor"
        })
        
        # Should return None because logic in `decide` checks funds
        # and if it adjusts to 0 quantity, it returns None.
        order = await agent.act(snapshot, [])
        assert order is None

    @pytest.mark.asyncio
    async def test_memory_update(self):
        agent = TraderAgent("agent_mem", cash_balance=100000)
        decision = {"action": "BUY"}
        outcome = {"pnl": 100.0}
        
        # Mock memory.add_memory to verify it's called
        agent.brain.memory.add_memory = MagicMock()
        
        await agent.update_memory(decision, outcome)
        
        # Check Brain State update (Mocked or check attribute change)
        assert agent.brain.state.total_pnl == 100.0

    @pytest.mark.asyncio
    async def test_reason_and_act_flow(self):
        """Test reason_and_act flow and emotional state update (Integrated via act)"""
        snapshot = create_snapshot(price=3000.0, trend=-0.06, panic=0.8)
        # Setup for LOSS: Cash 10k + Stock 30k = 40k. Base 100k -> -60% PnL -> Regretful
        agent = TraderAgent("agent_panic", cash_balance=10000, portfolio={"TEST": 10})
        
        # Mock Brain.think_async
        agent.brain.think_async = AsyncMock(return_value={
            "decision": {"action": "HOLD"},
            "reasoning": "Too scared to move",
            "emotion_score": -0.9
        })
        
        # 1. Verify Emotion Logic (Manual Trigger for granularity)
        perceived_bad = {"pnl_pct": -0.15, "snapshot": snapshot}
        agent.update_emotional_state(perceived_bad)
        assert agent.emotional_state == "Regretful"
        
        # 2. Call Act (Full loop)
        await agent.act(snapshot, ["Crash!"])
        
        # 3. Verify Brain Call
        agent.brain.think_async.assert_called_once()
        call_kwargs = agent.brain.think_async.call_args.kwargs
        assert call_kwargs["emotional_state"] == "Regretful"
        # Verify social signal (Trend -0.05 -> Panic)
        assert "panic selling" in call_kwargs["social_signal"]

    @pytest.mark.asyncio
    async def test_emotion_logic_coverage(self):
        """Test all emotion branches"""
        agent = TraderAgent("test_emotion")
        
        # Regretful
        agent.update_emotional_state({"pnl_pct": -0.15, "snapshot": MagicMock()})
        assert agent.emotional_state == "Regretful"
        
        # Anxious
        agent.update_emotional_state({"pnl_pct": -0.06, "snapshot": MagicMock()})
        assert agent.emotional_state == "Anxious"
        
        # Greedy
        agent.update_emotional_state({"pnl_pct": 0.15, "snapshot": MagicMock()})
        assert agent.emotional_state == "Greedy"
            
        # Confident
        agent.update_emotional_state({"pnl_pct": 0.06, "snapshot": MagicMock()})
        assert agent.emotional_state == "Confident"
            
        # Fearful (Panic Level High)
        s = MagicMock(); s.panic_level = 0.8
        agent.update_emotional_state({"pnl_pct": 0.0, "snapshot": s})
        assert agent.emotional_state == "Fearful"
