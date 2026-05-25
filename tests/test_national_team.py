# file: tests/test_national_team.py
import pytest
import asyncio
from unittest.mock import MagicMock
from agents.roles.national_team import NationalTeamAgent
from agents.base_agent import MarketSnapshot
from core.society.network import SocialGraph, SentimentState
from core.types import Order, OrderSide

def create_snapshot(price=3000.0, panic=0.0):
    return MarketSnapshot(
        symbol="SH000001",
        last_price=price,
        best_bid=price-1,
        best_ask=price+1,
        market_trend=-0.06 if panic > 0.5 else 0.0,
        panic_level=panic,
        timestamp=1000.0,
        mid_price=price,
        bid_ask_spread=2.0
    )

class TestNationalTeam:
    
    def test_init(self):
        nt = NationalTeamAgent("nt_001")
        assert nt.cash_balance == 10_000_000_000.0
        assert nt.psychology_profile["confidence_level"] == 1.0
        assert nt.psychology_profile["risk_aversion"] == 0.0

    @pytest.mark.asyncio
    async def test_intervention_trigger_panic(self):
        """测试恐慌阈值触发干预"""
        nt = NationalTeamAgent("nt_panic")
        # Panic level 0.9 > 0.8
        snapshot = create_snapshot(panic=0.9)
        
        order = await nt.act(snapshot, [])
        
        assert nt.is_intervening is True
        assert order is not None
        assert order.side == OrderSide.BUY
        # Price should be 1.005 * 3000 = 3015
        assert order.price == pytest.approx(3015.0)
        assert order.quantity > 0

    @pytest.mark.asyncio
    async def test_intervention_trigger_drop(self):
        """测试连续下跌触发干预"""
        nt = NationalTeamAgent("nt_drop")
        
        # Feed price history: 3000, 2900, 2800 (Drop > 5%)
        snap1 = create_snapshot(price=3000.0)
        snap2 = create_snapshot(price=2900.0)
        snap3 = create_snapshot(price=2800.0) # Drop (2800-3000)/3000 = -6.6%
        
        await nt.act(snap1, [])
        await nt.act(snap2, [])
        order = await nt.act(snap3, [])
        
        assert nt.is_intervening is True
        assert order is not None
        assert order.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_social_broadcast(self):
        """测试信心广播"""
        nt = NationalTeamAgent("nt_social")
        graph = SocialGraph(n_agents=10, k=4, p=0, seed=42)
        nt.bind_social_node(0, graph)
        
        # Infect neighbors
        neighbors = graph.get_neighbors(0)
        for nid in neighbors:
            graph.agents[nid].sentiment_state = SentimentState.INFECTED
            
        # Trigger intervention
        snapshot = create_snapshot(panic=0.9)
        await nt.act(snapshot, [])
        
        # Verify neighbors are recovered
        for nid in neighbors:
             assert graph.agents[nid].sentiment_state == SentimentState.RECOVERED
             
        # Verify self is bullish
        assert graph.agents[0].sentiment_state == SentimentState.BULLISH
