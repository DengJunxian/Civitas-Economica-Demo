import json
import time

from agents.diagnostic.tools import ProbeExecutor
from agents.brain import DeepSeekBrain, ThoughtRecord
from core.types import Trade


def test_query_lob_log_summary():
    class MockEngine:
        def __init__(self):
            self.trades_history = []
            self.step_trades_buffer = []

    engine = MockEngine()
    engine.trades_history.append(
        Trade(
            trade_id="T100",
            price=10.5,
            quantity=200,
            maker_id="m1",
            taker_id="t1",
            maker_agent_id="Inst_A",
            taker_agent_id="Retail_1",
            buyer_agent_id="Inst_A",
            seller_agent_id="Retail_1",
            timestamp=time.time()
        )
    )

    class MockEnv:
        def __init__(self, engine):
            self.engine = engine

    executor = ProbeExecutor(agents_map={}, market_env=MockEnv(engine))
    res = json.loads(executor.execute_tool("query_lob_log", {}))
    assert res["summary"]["trade_count"] == 1
    assert res["summary"]["total_qty"] == 200


def test_send_interview_synthesized_reply():
    class MockGraphMemory:
        def retrieve_subgraph(self, keywords, depth):
            return "风险 -> 保证金 -> 强平"

    class MockAgent:
        def __init__(self):
            self.persona = {"risk_preference": "谨慎"}
            self.graph_memory = MockGraphMemory()

    agent = MockAgent()
    agents_map = {"AgentA": agent}

    # 注入思维链
    DeepSeekBrain.thought_history["AgentA"] = [
        ThoughtRecord(
            agent_id="AgentA",
            timestamp=123.0,
            reasoning_content="近期杠杆过高，需收缩仓位",
            emotion_score=-0.2,
            decision={"action": "SELL"}
        )
    ]

    executor = ProbeExecutor(agents_map=agents_map, market_env=None)
    res = json.loads(
        executor.execute_tool(
            "send_interview",
            {"agent_id": "AgentA", "question": "为何先行减仓？", "topic": "风险"}
        )
    )

    assert res["interviewee"] == "AgentA"
    assert "response" in res
    assert "谨慎" in res["response"] or res["persona"]["risk_preference"] == "谨慎"
