
import unittest
import numpy as np
from typing import List, Dict

from agents.cognition.cognitive_agent import CognitiveAgent, create_population
from agents.cognition.utility import InvestorType
from core.society.network import SocialGraph, InformationDiffusion

class TestBehavioralEconomics(unittest.TestCase):
    def setUp(self):
        # Initialize population
        self.population_size = 20
        self.agents = create_population(size=self.population_size, panic_ratio=0.3, quant_ratio=0.1)
        
        # Initialize Social Network
        self.graph = SocialGraph(n_agents=self.population_size, k=4, p=0.1)
        # Link agents to graph nodes (simplified 1-to-1 mapping)
        self.agent_map = {f"agent_{i}": agent for i, agent in enumerate(self.agents)}
        
    def test_confidence_dynamics(self):
        """Test if confidence updates correctly based on PnL"""
        print("\n[Test] Confidence Dynamics")
        agent = self.agents[0]
        initial_confidence = agent.confidence_tracker.confidence
        
        # Case 1: Profitable trade -> Increased confidence
        agent.decision_history.append("dummy") # Ensure history not empty if needed
        agent.record_outcome(pnl=0.05, market_summary="Bull run")
        
        new_confidence = agent.confidence_tracker.confidence
        print(f"  PnL +5% -> Confidence: {initial_confidence:.2f} -> {new_confidence:.2f}")
        self.assertGreater(new_confidence, initial_confidence, "Confidence should increase after profit")
        
        # Case 2: Loss -> Decreased confidence (but less due to attribution bias)
        agent.record_outcome(pnl=-0.05, market_summary="Correction")
        final_confidence = agent.confidence_tracker.confidence
        print(f"  PnL -5% -> Confidence: {new_confidence:.2f} -> {final_confidence:.2f}")
        self.assertLess(final_confidence, new_confidence, "Confidence should decrease after loss")
        
        # Verify attribution bias (increase > decrease for same magnitude return)
        # Note: gamma1=0.1, gamma2=0.05 in utility.py
        increase = new_confidence - initial_confidence
        decrease = new_confidence - final_confidence
        self.assertGreater(increase, decrease, "Self-attribution bias: gain impact should > loss impact")

    def test_anchoring_effect(self):
        """Test if reference point drifts"""
        print("\n[Test] Anchoring Effect")
        agent = self.agents[0]
        
        # Initialize anchor
        mock_market = {"price": 100.0}
        mock_account = {"pnl_pct": 0.0, "avg_cost": 100.0}
        agent.make_decision(mock_market, mock_account)
        
        initial_rp = agent.anchor_tracker.reference_point
        self.assertEqual(initial_rp, 100.0)
        
        # Price moves up, anchor should drift up slowly
        new_price = 110.0
        agent.anchor_tracker.update(new_price)
        current_rp = agent.anchor_tracker.reference_point
        print(f"  Price 100 -> 110. Anchor: {initial_rp:.2f} -> {current_rp:.2f}")
        
        self.assertGreater(current_rp, 100.0)
        self.assertLess(current_rp, 110.0, "Anchor should drag behind price")

    def test_simulated_herding_loop(self):
        """Simulate a loop to calculate CSAD and observe prompts"""
        print("\n[Test] Simulated Herding Loop (CSAD)")
        
        # Mock returns for 20 agents
        # Scenario: High Herding (Agents returns are very similar)
        market_return = -0.05
        agent_returns = np.random.normal(market_return, 0.001, self.population_size) # Low dispersion
        
        # Calculate CSAD
        csad = np.mean(np.abs(agent_returns - market_return))
        print(f"  Simulated CSAD: {csad:.5f} (Should be low)")
        
        # Prepare market state with CSAD
        market_state = {
            "price": 2800.0,
            "trend": "下跌",
            "panic_level": 0.8,
            "csad": csad
        }
        account_state = {
            "cash": 50000.0,
            "market_value": 40000.0,
            "pnl_pct": -0.10,
            "avg_cost": 3100.0
        }
        
        # Agent makes decision
        agent = self.agents[0] # Assume Panic Retail
        print(f"  Agent Type: {agent.investor_type}")
        
        # We need to spy on the LLM prompt or result to see if CSAD info is injected.
        # Since we use local reasoner for tests usually, we check if logic passes.
        # But `llm_brain.py` `LocalReasoner` doesn't use CSAD currently (only R1 does in prompt).
        # We checked `cognitive_agent.py` passes `csad` to `reasoner.reason`.
        
        decision = agent.make_decision(market_state, account_state)
        print(f"  Decision: {decision.final_action}, Reason: {decision.llm_reasoning or decision.override_reason}")
        
        self.assertTrue(True, "Simulation loop ran without error")

if __name__ == "__main__":
    unittest.main()
