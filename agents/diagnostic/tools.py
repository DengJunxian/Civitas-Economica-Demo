# file: agents/diagnostic/tools.py
import json
from typing import Dict, Any, List, Optional


def get_diagnostic_tools() -> List[Dict[str, Any]]:
    """返回供大模型调用的 Tools JSON Schema 列表"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_agent_state",
                "description": "获取指定交易智能体当前的资金、持仓、浮动盈亏、信心指数及风险偏好状态",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "智能体的纯文本 ID，如 'Retail_1' 或 'Inst_A'"
                        }
                    },
                    "required": ["agent_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_agent_thought_history",
                "description": "获取指定智能体在最近几轮的交易思维链路记录（包含内部反思、情绪和生成决策）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "智能体 ID"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "想要拉取的历史条数",
                            "default": 3
                        }
                    },
                    "required": ["agent_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_agent_memory",
                "description": "查询指定智能体的 GraphRAG 私有图谱记忆（如他对’降准‘的认知关联）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "智能体 ID"
                        },
                        "topic": {
                            "type": "string",
                            "description": "要查询的概念关键词，如'利率'、'芯片'、'风控'"
                        }
                    },
                    "required": ["agent_id", "topic"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_snapshot",
                "description": "获取沙箱当前全局的市场快照状态，包括价格、恐慌指数、微观趋势等环境参数",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]


def get_report_tools() -> List[Dict[str, Any]]:
    """
    ReportAgent 专用工具定义（ReACT + Tool Calling）。

    说明：
    - query_lob_log: 查询底层撮合/交易日志
    - search_agent_memory: 检索个体图谱/思维记忆
    - send_interview: 对指定 Agent 发起微观质询
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "query_lob_log",
                "description": "查询 LOB 撮合日志，可按时间步、关键字筛选，定位先爆仓/先触发连锁反应的源头事件",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "description": "可选，离散时间步"},
                        "keyword": {"type": "string", "description": "可选，日志关键字，如 '爆仓'、'强平'、'liquidation'"},
                        "limit": {"type": "integer", "description": "返回条数上限", "default": 20}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_agent_memory",
                "description": "检索指定微观个体（大V/散户）的历史图谱、近期思维链与关键决策",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "个体 Agent ID"},
                        "topic": {"type": "string", "description": "可选关键词，如 '杠杆'、'止损'、'恐慌'"},
                        "limit": {"type": "integer", "description": "最近思维链条数", "default": 3}
                    },
                    "required": ["agent_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_interview",
                "description": "向特定大V/散户发起一对一微观原因质询（微观探针），返回其解释与证据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string", "description": "被访谈 Agent ID"},
                        "question": {"type": "string", "description": "访谈问题"}
                    },
                    "required": ["agent_id", "question"]
                }
            }
        }
    ]


# 本地执行函数的占位接口，由 DiagnosticAgent 初始化时绑定实际沙箱上下文
class ProbeExecutor:
    def __init__(self, agents_map: Dict[str, Any], market_env: Any):
        self.agents = agents_map
        self.market_env = market_env

    def execute_tool(self, name: str, kwargs: Dict[str, Any]) -> str:
        """根据大模型请求派发到具体的方法，返回 JSON 字符串结果"""
        try:
            if name == "get_agent_state":
                return self.get_agent_state(**kwargs)
            elif name == "get_agent_thought_history":
                return self.get_agent_thought_history(**kwargs)
            elif name == "query_agent_memory":
                return self.query_agent_memory(**kwargs)
            elif name == "get_market_snapshot":
                return self.get_market_snapshot()
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return json.dumps({"error": f"Error executing {name}: {str(e)}"})

    def get_agent_state(self, agent_id: str) -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."})

        state_info = {
            "cash": getattr(agent, "cash_balance", 0.0),
            "portfolio": getattr(agent, "portfolio", {}),
            "total_pnl": getattr(agent, "total_pnl", 0.0) if hasattr(agent, "total_pnl") else 0.0,
        }

        if hasattr(agent, "brain") and hasattr(agent.brain, "state"):
            state_info["confidence"] = agent.brain.state.confidence
            state_info["risk_preference"] = getattr(agent.brain.state, "evolved_risk_preference", None) or agent.persona.get("risk_preference", "未知")

        return json.dumps(state_info, ensure_ascii=False)

    def get_agent_thought_history(self, agent_id: str, limit: int = 3) -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."})

        if not hasattr(agent, "brain"):
            return json.dumps({"error": f"Agent {agent_id} has no brain component."})

        # Access class-level thought memory directly
        from agents.brain import DeepSeekBrain
        history = DeepSeekBrain.thought_history.get(agent_id, [])
        if not history:
            return json.dumps({"message": "No thought history available."})

        recent = history[-limit:]
        records = []
        for r in recent:
            records.append({
                "timestamp": r.timestamp,
                "emotion_score": r.emotion_score,
                "reasoning": r.reasoning_content,
                "decision": r.decision
            })
        return json.dumps(records, ensure_ascii=False)

    def query_agent_memory(self, agent_id: str, topic: str) -> str:
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."})

        if not hasattr(agent, "graph_memory"):
            return json.dumps({"error": f"Agent {agent_id} doesn't have GraphMemoryBank enabled."})

        subgraph = agent.graph_memory.retrieve_subgraph([topic], depth=2)
        if not subgraph:
            return json.dumps({"message": f"No graph memory found for topic '{topic}'."})

        return json.dumps({"graph_subgraph": subgraph}, ensure_ascii=False)

    def get_market_snapshot(self) -> str:
        if not self.market_env:
            return json.dumps({"error": "Market environment not bound to ProbeExecutor."})

        # Mock mapping, real implementation depends on market_env structure
        try:
            state = {
                "price": getattr(self.market_env, "last_price", getattr(self.market_env, "current_price", 0.0)),
                "trend": getattr(self.market_env, "trend", "未知"),
                "panic_level": getattr(self.market_env, "panic_level", 0.0),
                "news_count": len(getattr(self.market_env, "recent_news", [])),
                "timestamp": getattr(self.market_env, "current_time", getattr(self.market_env, "timestamp", 0))
            }
            return json.dumps(state, ensure_ascii=False)
        except Exception:
            return json.dumps({"error": "Could not extract standard market state."})


class ReportToolExecutor:
    """ReportAgent 的工具执行器：聚焦复盘与溯源。"""

    def __init__(self, agents_map: Dict[str, Any], market_env: Any):
        self.agents = agents_map
        self.market_env = market_env

    def execute_tool(self, name: str, kwargs: Dict[str, Any]) -> str:
        try:
            if name == "query_lob_log":
                return self.query_lob_log(**kwargs)
            if name == "search_agent_memory":
                return self.search_agent_memory(**kwargs)
            if name == "send_interview":
                return self.send_interview(**kwargs)
            return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"error": f"Error executing {name}: {exc}"}, ensure_ascii=False)

    def _extract_lob_logs(self) -> List[Dict[str, Any]]:
        """兼容不同 market_env 字段，统一提取日志列表。"""
        if not self.market_env:
            return []

        raw_logs = (
            getattr(self.market_env, "lob_logs", None)
            or getattr(self.market_env, "match_logs", None)
            or getattr(self.market_env, "trade_log", None)
            or getattr(self.market_env, "trade_logs", None)
            or []
        )

        normalized: List[Dict[str, Any]] = []
        for item in raw_logs:
            if isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"message": str(item)})
        return normalized

    def query_lob_log(self, step: Optional[int] = None, keyword: Optional[str] = None, limit: int = 20) -> str:
        """
        查询撮合引擎日志。

        该工具是复盘链路中的“事实底座”：
        - step 用于离散时间定位
        - keyword 用于定位爆仓、强平、穿仓等关键事件
        """
        logs = self._extract_lob_logs()
        if not logs:
            return json.dumps({"message": "LOB 日志为空或未挂载。"}, ensure_ascii=False)

        filtered = []
        for entry in logs:
            msg = json.dumps(entry, ensure_ascii=False)
            if step is not None:
                entry_step = entry.get("step", entry.get("timestep", entry.get("time_step")))
                if entry_step != step:
                    continue
            if keyword and keyword.lower() not in msg.lower():
                continue
            filtered.append(entry)

        return json.dumps({"count": len(filtered), "records": filtered[: max(1, limit)]}, ensure_ascii=False)

    def search_agent_memory(self, agent_id: str, topic: str = "", limit: int = 3) -> str:
        """检索指定 Agent 的图谱记忆 + 最近思维链。"""
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."}, ensure_ascii=False)

        result: Dict[str, Any] = {"agent_id": agent_id}

        # 图谱记忆：优先 topic 检索，未提供 topic 时给出默认提示
        if hasattr(agent, "graph_memory"):
            if topic:
                result["graph_memory"] = agent.graph_memory.retrieve_subgraph([topic], depth=2)
            else:
                result["graph_memory"] = "未提供 topic，建议指定关键词提高命中率。"
        else:
            result["graph_memory"] = "该 Agent 未启用 GraphMemoryBank。"

        # 思维链历史
        from agents.brain import DeepSeekBrain
        history = DeepSeekBrain.thought_history.get(agent_id, [])
        recent_records = []
        for r in history[-max(1, limit):]:
            recent_records.append(
                {
                    "timestamp": r.timestamp,
                    "emotion_score": r.emotion_score,
                    "decision": r.decision,
                    "reasoning": r.reasoning_content,
                }
            )
        result["recent_thoughts"] = recent_records

        return json.dumps(result, ensure_ascii=False)

    def send_interview(self, agent_id: str, question: str) -> str:
        """
        发送一对一微观质询（微观探针）。

        由于当前沙箱 Agent 多数未暴露统一“访谈接口”，这里提供两级机制：
        1) 若 agent 实现 `interview(question)`，直接调用。
        2) 否则基于最新思维链/仓位/盈亏生成结构化“访谈纪要”。
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."}, ensure_ascii=False)

        if hasattr(agent, "interview") and callable(agent.interview):
            answer = agent.interview(question)
            return json.dumps({"agent_id": agent_id, "question": question, "answer": answer}, ensure_ascii=False)

        # 回退：抽取近期“可解释证据”组装回答
        from agents.brain import DeepSeekBrain

        history = DeepSeekBrain.thought_history.get(agent_id, [])
        latest_reasoning = history[-1].reasoning_content if history else "无近期思维链记录"
        latest_decision = history[-1].decision if history else {"action": "UNKNOWN"}

        mock_answer = {
            "agent_id": agent_id,
            "question": question,
            "answer": (
                f"作为 {agent_id}，我近期决策主要基于市场趋势与风险偏好。"
                "在你提问的事件点，我优先考虑了仓位风险与情绪波动。"
            ),
            "evidence": {
                "cash_balance": getattr(agent, "cash_balance", None),
                "portfolio": getattr(agent, "portfolio", {}),
                "total_pnl": getattr(agent, "total_pnl", None),
                "latest_decision": latest_decision,
                "latest_reasoning": latest_reasoning,
            },
        }
        return json.dumps(mock_answer, ensure_ascii=False)
