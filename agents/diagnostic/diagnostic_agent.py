# file: agents/diagnostic/diagnostic_agent.py
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
import asyncio

from agents.diagnostic.tools import get_diagnostic_tools, ProbeExecutor
from config import GLOBAL_CONFIG

logger = logging.getLogger(__name__)

class DiagnosticAgent:
    """
    负责进行微观态探针分析的诊断 Agent。
    利用 OpenAI 兼容的 Function Calling 机制调用本地沙箱探针。
    """
    def __init__(
        self,
        agents_map: Dict[str, Any],
        market_env: Any = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "deepseek-chat"
    ):
        self.api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        self.base_url = base_url or GLOBAL_CONFIG.API_BASE_URL
        self.model_name = model_name
        
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.executor = ProbeExecutor(agents_map, market_env)
        
        self.system_prompt = """你是一个专业的金融多智能体沙箱诊断助手。
你的目标是通过调用提供的内部探针工具（Tools），分析系统内交易智能体的内部状态、深层记忆及当前宏观市场共识，解答用户对于“某个 Agent 为什么发生特定操作”或“当前市场群体行为归因”的提问。
你可以多次连续使用工具直到你找到想要的答案。回答应当清晰、逻辑严密、并指明数据来源（如：根据其最新的思维链...）。"""
        
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    async def chat(self, user_message: str) -> str:
        """接收用户文本输入，进行 Function Calling 路由循环并返回最终文本"""
        self.history.append({"role": "user", "content": user_message})
        tools = get_diagnostic_tools()
        
        # 防止无限循环调用
        max_tool_iterations = 5
        
        for i in range(max_tool_iterations):
            try:
                # 过滤掉不合规的消息参数
                clean_history = []
                for msg in self.history:
                    clean_msg = {"role": msg["role"]}
                    if "content" in msg:
                        clean_msg["content"] = msg["content"]
                    if "tool_calls" in msg:
                        clean_msg["tool_calls"] = msg["tool_calls"]
                    if "tool_call_id" in msg:
                        clean_msg["tool_call_id"] = msg["tool_call_id"]
                    if "name" in msg:
                        clean_msg["name"] = msg["name"]
                    clean_history.append(clean_msg)

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=clean_history,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.3
                )
                
                message = response.choices[0].message
                
                if message.tool_calls:
                    # 记录助手的 tool call
                    tool_calls_record = []
                    for t in message.tool_calls:
                        tool_calls_record.append({
                            "id": t.id,
                            "type": t.type,
                            "function": {
                                "name": t.function.name,
                                "arguments": t.function.arguments
                            }
                        })
                    self.history.append({
                        "role": "assistant",
                        "content": message.content, # could be None
                        "tool_calls": tool_calls_record
                    })
                    
                    # 执行每个工具
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        try:
                            # 尝试解析参数
                            func_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            func_args = {}
                            
                        print(f"\n[Diagnostic Probe] Executing Tool: {func_name} with {func_args}")
                        
                        tool_result_str = self.executor.execute_tool(func_name, func_args)
                        
                        # 压入工具执行结果
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": tool_result_str
                        })
                else:
                    # 没有需要调用的工具，正常回复文本
                    response_content = message.content or ""
                    self.history.append({"role": "assistant", "content": response_content})
                    return response_content
                    
            except Exception as e:
                logger.error(f"DiagnosticAgent Chat Error: {e}")
                err_msg = f"与大模型通信或执行探针时发生错误: {e}"
                self.history.append({"role": "assistant", "content": err_msg})
                return err_msg
                
        # 超过最大轮数
        return "诊断助手进行了过多内部探针调用，已强行中断防止阻挠。请重试或简化你的问题。"

    def clear_history(self):
        """清空对话历史"""
        self.history = [{"role": "system", "content": self.system_prompt}]
