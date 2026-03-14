# file: agents/diagnostic/tools.py
import json
from typing import Dict, Any, List


def get_diagnostic_tools() -> List[Dict[str, Any]]:
    """返回给大模型的 Tools JSON Schema 列表"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_agent_state",
                "description": "获取指定智能体当前的资金、持仓、浮动盈亏、自信指数与风险偏好",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "智能体 ID，如 'Retail_1' 或 'Inst_A'"
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
                "description": "获取指定智能体最近几轮交易思维链路（含反思、情绪与生成决策）",
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
                "description": "查询指定智能体的 GraphRAG 私有图谱记忆（如对‘降准’的认知关联）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "智能体 ID"
                        },
                        "topic": {
                            "type": "string",
                            "description": "查询概念关键词，如 '利率'、'芯片'、'风控'"
                        }
                    },
                    "required": ["agent_id", "topic"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_agent_memory",
                "description": "检索特定微观个体的历史图谱（query_agent_memory 别名）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "智能体 ID"
                        },
                        "topic": {
                            "type": "string",
                            "description": "检索主题关键词"
                        }
                    },
                    "required": ["agent_id", "topic"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_lob_log",
                "description": "查询撮合引擎日志（成交记录），默认返回摘要统计，可选明细",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "明细返回条数上限（默认 50）"
                        },
                        "detail": {
                            "type": "boolean",
                            "description": "是否返回明细列表（默认 false）"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "按智能体 ID 过滤成交记录（可选）"
                        },
                        "current_step": {
                            "type": "boolean",
                            "description": "仅查询当前时间步缓冲区（默认 false）"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_interview",
                "description": "向沙箱中特定的“大V”或“散户”发起一对一质询（返回合成回应）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "受访智能体 ID"
                        },
                        "question": {
                            "type": "string",
                            "description": "访谈问题"
                        },
                        "topic": {
                            "type": "string",
                            "description": "可选主题关键词（用于检索图谱）"
                        }
                    },
                    "required": ["agent_id", "question"]
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
        """根据大模型请求派发到具体方法，返回 JSON 字符串结果"""
        try:
            if name == "get_agent_state":
                return self.get_agent_state(**kwargs)
            if name == "get_agent_thought_history":
                return self.get_agent_thought_history(**kwargs)
            if name == "query_agent_memory":
                return self.query_agent_memory(**kwargs)
            if name == "search_agent_memory":
                return self.query_agent_memory(**kwargs)
            if name == "query_lob_log":
                return self.query_lob_log(**kwargs)
            if name == "send_interview":
                return self.send_interview(**kwargs)
            if name == "get_market_snapshot":
                return self.get_market_snapshot()
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
            state_info["risk_preference"] = (
                getattr(agent.brain.state, "evolved_risk_preference", None)
                or getattr(agent.persona, "risk_preference", "未知") if hasattr(agent, "persona") else "未知"
            )

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

    def query_lob_log(
        self,
        limit: int = 50,
        detail: bool = False,
        agent_id: str = "",
        current_step: bool = False
    ) -> str:
        """
        查询撮合引擎成交日志。
        复杂逻辑说明：
        1) 仅依赖 MatchingEngine 的 trades_history/step_trades_buffer
        2) 默认返回摘要，避免明细过大影响上下文
        """
        if not self.market_env or not hasattr(self.market_env, "engine"):
            return json.dumps({"error": "Market environment or engine not bound."})

        engine = getattr(self.market_env, "engine", None)
        if not engine:
            return json.dumps({"error": "Matching engine not available in market_env."})

        trades = engine.step_trades_buffer if current_step else engine.trades_history
        trades = trades or []

        # 过滤指定 agent 的成交
        if agent_id:
            trades = [
                t for t in trades
                if agent_id in {
                    getattr(t, "buyer_agent_id", ""),
                    getattr(t, "seller_agent_id", ""),
                    getattr(t, "maker_agent_id", ""),
                    getattr(t, "taker_agent_id", "")
                }
            ]

        # 摘要统计
        total_qty = sum(getattr(t, "quantity", 0) for t in trades)
        total_notional = sum(getattr(t, "price", 0.0) * getattr(t, "quantity", 0) for t in trades)
        timestamps = [getattr(t, "timestamp", 0) for t in trades]

        buyer_stats = {}
        seller_stats = {}
        for t in trades:
            buyer = getattr(t, "buyer_agent_id", "")
            seller = getattr(t, "seller_agent_id", "")
            qty = getattr(t, "quantity", 0)
            if buyer:
                buyer_stats[buyer] = buyer_stats.get(buyer, 0) + qty
            if seller:
                seller_stats[seller] = seller_stats.get(seller, 0) + qty

        top_buyers = sorted(buyer_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        top_sellers = sorted(seller_stats.items(), key=lambda x: x[1], reverse=True)[:5]

        result = {
            "summary": {
                "trade_count": len(trades),
                "total_qty": total_qty,
                "total_notional": total_notional,
                "time_range": [min(timestamps), max(timestamps)] if timestamps else None,
                "top_buyers": top_buyers,
                "top_sellers": top_sellers
            },
            "source": "step_buffer" if current_step else "trades_history",
            "filtered_agent": agent_id or None
        }

        # 明细列表（可选）
        if detail:
            recent = trades[-limit:] if limit and limit > 0 else trades
            detail_list = []
            for t in recent:
                detail_list.append({
                    "trade_id": getattr(t, "trade_id", ""),
                    "price": getattr(t, "price", 0.0),
                    "quantity": getattr(t, "quantity", 0),
                    "buyer_agent_id": getattr(t, "buyer_agent_id", ""),
                    "seller_agent_id": getattr(t, "seller_agent_id", ""),
                    "maker_agent_id": getattr(t, "maker_agent_id", ""),
                    "taker_agent_id": getattr(t, "taker_agent_id", ""),
                    "timestamp": getattr(t, "timestamp", 0)
                })
            result["detail"] = detail_list

        return json.dumps(result, ensure_ascii=False)

    def send_interview(self, agent_id: str, question: str, topic: str = "") -> str:
        """
        基于智能体画像 + 思维链 + 图谱记忆合成访谈回应。
        复杂逻辑说明：
        1) 不额外调用 LLM，避免延迟与黑盒
        2) 优先引用最近思维链与记忆片段作为证据
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return json.dumps({"error": f"Agent {agent_id} not found."})

        persona = getattr(agent, "persona", None)
        risk_pref = getattr(persona, "risk_preference", "未知") if persona else "未知"

        # 思维链摘要（取最近 2 条）
        from agents.brain import DeepSeekBrain
        history = DeepSeekBrain.thought_history.get(agent_id, [])
        recent = history[-2:] if history else []
        thought_summaries = [
            {
                "timestamp": r.timestamp,
                "emotion_score": r.emotion_score,
                "reasoning": r.reasoning_content,
                "decision": r.decision
            }
            for r in recent
        ]

        # 图谱记忆片段
        memory_snippet = None
        if topic and hasattr(agent, "graph_memory"):
            try:
                memory_snippet = agent.graph_memory.retrieve_subgraph([topic], depth=2)
            except Exception:
                memory_snippet = None

        # 合成回应
        response_text = (
            f"我倾向于 {risk_pref} 风格。在当前信息下，"
            f"我对问题“{question}”的判断主要基于近期决策线索与记忆关联。"
        )
        if thought_summaries:
            response_text += "近期思维显示我关注价格波动与仓位风险。"
        if memory_snippet:
            response_text += f"图谱记忆提示与“{topic}”相关的因果链路需优先关注。"

        payload = {
            "interviewee": agent_id,
            "question": question,
            "persona": {
                "risk_preference": risk_pref
            },
            "recent_thoughts": thought_summaries,
            "memory_snippet": memory_snippet,
            "response": response_text
        }
        return json.dumps(payload, ensure_ascii=False)

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
