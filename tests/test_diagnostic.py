# file: tests/test_diagnostic.py
import pytest
import json

from agents.diagnostic.tools import ProbeExecutor, get_diagnostic_tools

def test_diagnostic_tools_schema():
    tools = get_diagnostic_tools()
    assert len(tools) == 7
    names = [t["function"]["name"] for t in tools]
    assert "get_agent_state" in names
    assert "query_agent_memory" in names
    assert "search_agent_memory" in names
    assert "query_lob_log" in names
    assert "send_interview" in names

def test_probe_executor_execution():
    # Mock some basic agents to test ProbeExecutor locally
    class MockBrain:
        class State:
            confidence = 60.0
            evolved_risk_preference = "稳健"
        state = State()
        
    class MockAgent:
        def __init__(self, id):
            self.agent_id = id
            self.cash_balance = 1000
            self.total_pnl = -500
            self.portfolio = {"000001": 100}
            self.brain = MockBrain()
            self.persona = {"risk_preference": "保守"}
            
        class MockGraphMemory:
            def retrieve_subgraph(self, keywords, depth):
                if "利率" in keywords:
                    return "利率 -> 导致 -> 下跌"
                return ""
        
        graph_memory = MockGraphMemory()

    agents_map = {"AgentA": MockAgent("AgentA")}
    class MockEnv:
        last_price = 3000
        trend = "震荡"
        panic_level = 0.5
        current_time = 1000.0
        
    executor = ProbeExecutor(agents_map, MockEnv())
    
    # 1. Test state
    state_res = json.loads(executor.execute_tool("get_agent_state", {"agent_id": "AgentA"}))
    assert state_res["cash"] == 1000
    assert state_res["total_pnl"] == -500
    assert state_res["confidence"] == 60.0
    
    # 2. Test memory
    mem_res = json.loads(executor.execute_tool("query_agent_memory", {"agent_id": "AgentA", "topic": "利率"}))
    assert "graph_subgraph" in mem_res
    assert "导致" in mem_res["graph_subgraph"]
    
    # 3. Test unknown tool
    unknown_res = json.loads(executor.execute_tool("unknown_tool", {}))
    assert "error" in unknown_res
