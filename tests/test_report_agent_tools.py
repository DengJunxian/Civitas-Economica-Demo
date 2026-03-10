import json

from agents.diagnostic.report_agent import ReportAgent
from agents.diagnostic.tools import ReportToolExecutor


class DummyBrainState:
    confidence = 42


class DummyBrain:
    state = DummyBrainState()


class DummyAgent:
    def __init__(self):
        self.cash_balance = 1000
        self.portfolio = {"000001": 200}
        self.total_pnl = -123
        self.brain = DummyBrain()


class DummyEnv:
    lob_logs = [
        {"step": 15, "event": "liquidation", "agent_id": "A"},
        {"step": 16, "event": "trade", "agent_id": "B"},
    ]


def test_query_lob_log_filters_step_and_keyword():
    executor = ReportToolExecutor(agents_map={}, market_env=DummyEnv())
    result = json.loads(executor.query_lob_log(step=15, keyword="liquidation"))
    assert result["count"] == 1
    assert result["records"][0]["agent_id"] == "A"


def test_send_interview_fallback_contains_evidence():
    executor = ReportToolExecutor(agents_map={"A": DummyAgent()}, market_env=DummyEnv())
    result = json.loads(executor.send_interview(agent_id="A", question="为什么先卖出？"))
    assert result["agent_id"] == "A"
    assert "evidence" in result
    assert result["evidence"]["cash_balance"] == 1000


def test_report_agent_parse_tool_args_with_think_tags():
    agent = ReportAgent(agents_map={}, market_env=None, api_key="x", base_url="http://localhost")
    parsed = agent._parse_tool_args('<think>hidden</think>{"step":15}')
    assert parsed["step"] == 15
