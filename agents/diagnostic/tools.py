# file: agents/diagnostic/tools.py
import json
from typing import Dict, Any, List

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
