
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from core.mesa.civitas_agent import CivitasAgent
from agents.cognition.cognitive_agent import CognitiveAgent
from agents.cognition.llm_brain import DeepSeekReasoner
from agents.cognition.memory import TraumaMemory
from core.time_manager import SimulationClock
from core.account import Portfolio

class TestAgentOptimization(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = MagicMock()
        self.model.symbol = "SH000001"
        self.model.clock = SimulationClock()
        
        self.agent = CivitasAgent(
            model=self.model,
            investor_type="MM", # Assuming string or Enum is handled, checking type hint says InvestorType but let's see
            initial_cash=100000
        )
        
        # Mock portfolio and positions
        self.agent.portfolio = MagicMock(spec=Portfolio)
        self.agent.portfolio.available_cash = 100000
        self.agent.portfolio.positions = {}
        self.agent.portfolio.get_sellable_qty.return_value = 0
        
        # Mock CognitiveAgent and its components
        self.agent.cognitive_agent = MagicMock(spec=CognitiveAgent)
        
        # Real logic test for CognitiveAgent
        self.real_cognitive = CognitiveAgent(agent_id="test_cog", use_local_reasoner=False)
        self.real_cognitive.reasoner = MagicMock(spec=DeepSeekReasoner)
        self.real_cognitive.reasoner.build_messages = MagicMock(return_value=[{"role":"user", "content":"prompt"}])
        self.real_cognitive.memory = MagicMock(spec=TraumaMemory)
        self.real_cognitive.memory.get_context_for_decision_async = AsyncMock(return_value="Mock Memory Context")
        
        # Re-bind Async method
        self.real_cognitive.prepare_decision_prompt_async = CognitiveAgent.prepare_decision_prompt_async.__get__(self.real_cognitive, CognitiveAgent)

    async def test_civitas_agent_prepare_decision_async(self):
        print("\n[Test] CivitasAgent.prepare_decision Async Flow")
        
        # Setup specific mock for this test
        self.agent.cognitive_agent.prepare_decision_prompt_async = AsyncMock(return_value=[{"role": "user", "content": "mock_prompt"}])
        
        market_state = {"price": 100, "news": "Good news", "history": "Up"}
        
        prompt, account_state = await self.agent.prepare_decision(market_state)
        
        print(f"Prompt: {prompt}")
        print(f"Account State: {account_state}")
        
        self.assertIsNotNone(prompt)
        self.assertEqual(account_state["cash"], 100000)
        self.agent.cognitive_agent.prepare_decision_prompt_async.assert_awaited_once()
        
    async def test_cognitive_agent_prepare_async(self):
        print("\n[Test] CognitiveAgent.prepare_decision_prompt_async")
        
        market_state = {"price": 100}
        account_state = {"cash": 100000}
        
        prompt = await self.real_cognitive.prepare_decision_prompt_async(market_state, account_state)
        
        self.real_cognitive.memory.get_context_for_decision_async.assert_awaited_once()
        self.real_cognitive.reasoner.build_messages.assert_called_once()
        print("Memory retrieval and prompt build called.")

    def test_llm_chain_truncation(self):
        print("\n[Test] DeepSeekReasoner Prompt Truncation")
        from agents.cognition.llm_brain import DeepSeekReasoner
        reasoner = DeepSeekReasoner()
        
        long_news = "A" * 2000
        long_memory = "B" * 2000
        
        market = {"price": 100, "news": long_news}
        account = {"cash": 10000, "position": 0, "avg_cost": 0, "pnl_pct": 0}
        
        prompt_str = reasoner._build_user_prompt(market, account, memory_context=long_memory)
        
        self.assertLess(len(prompt_str), 3000)
        self.assertIn("AAAA", prompt_str)
        self.assertNotIn("A"*1000, prompt_str) # Should be truncated
        print("Prompt truncation verified.")

if __name__ == "__main__":
    unittest.main()
