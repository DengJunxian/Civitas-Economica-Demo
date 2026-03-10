# file: agents/diagnostic/report_agent.py
import json
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from agents.diagnostic.tools import ReportToolExecutor, get_report_tools
from config import GLOBAL_CONFIG
from core.llm_client import _try_fix_json

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    自治复盘诊断 Agent（金融风控专家人设，ReACT 架构）。

    设计目标：
    1) 摒弃静态模板报告，支持“问题驱动”的动态溯源。
    2) 通过 Tool-Calling 连续调用底层探针，形成可追溯证据链。
    3) 用户可在控制台直接自然语言提问。
    """

    def __init__(
        self,
        agents_map: Dict[str, Any],
        market_env: Any = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "deepseek-chat",
    ):
        self.api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        self.base_url = base_url or GLOBAL_CONFIG.API_BASE_URL
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        self.executor = ReportToolExecutor(agents_map=agents_map, market_env=market_env)

        # ReACT 约束：每轮要么给出 tool call，要么输出 final answer。
        self.system_prompt = """你是“数治观澜”沙箱的首席复盘官，具备资深金融风控专家背景。
你必须遵循 ReACT 工作流：
- Reason: 先明确你要验证的假设
- Act: 调用最小必要工具(query_lob_log / search_agent_memory / send_interview)
- Observe: 读取工具返回结果，更新判断
- Final: 仅在证据充分时给出结论

回答要求：
1) 结论必须引用工具证据；
2) 若证据不足，明确列出还需补采的数据；
3) 尽量给出“时间步 -> 行为主体 -> 触发机制 -> 风险后果”的因果链。
"""
        self.history: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    async def chat(self, user_message: str, max_iterations: int = 8) -> str:
        """处理自然语言提问，并通过 ReACT + Tool Calling 给出复盘诊断结论。"""
        self.history.append({"role": "user", "content": user_message})
        tools = get_report_tools()

        for _ in range(max_iterations):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self._clean_history(),
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                )
                message = response.choices[0].message

                if message.tool_calls:
                    tool_calls_record = []
                    for tool_call in message.tool_calls:
                        tool_calls_record.append(
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )
                    self.history.append(
                        {"role": "assistant", "content": message.content, "tool_calls": tool_calls_record}
                    )

                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = self._parse_tool_args(tool_call.function.arguments)
                        tool_result_str = self.executor.execute_tool(func_name, func_args)

                        self.history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": func_name,
                                "content": tool_result_str,
                            }
                        )
                else:
                    final_text = message.content or ""
                    self.history.append({"role": "assistant", "content": final_text})
                    return final_text
            except Exception as exc:
                logger.error("ReportAgent chat error: %s", exc)
                return f"复盘诊断过程中发生异常: {exc}"

        return "诊断回路超过最大迭代次数，请缩小问题范围后重试。"

    def _parse_tool_args(self, raw_args: str) -> Dict[str, Any]:
        """解析函数参数，兼容模型输出 JSON 污染或截断。"""
        if not raw_args:
            return {}

        fixed = _try_fix_json(raw_args)
        if not fixed:
            return {}

        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}

    def _clean_history(self) -> List[Dict[str, Any]]:
        """过滤消息字段，避免非标准字段污染 API 请求。"""
        clean_history: List[Dict[str, Any]] = []
        for msg in self.history:
            clean_msg: Dict[str, Any] = {"role": msg["role"]}
            if "content" in msg:
                clean_msg["content"] = msg["content"]
            if "tool_calls" in msg:
                clean_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                clean_msg["name"] = msg["name"]
            clean_history.append(clean_msg)
        return clean_history

    def clear_history(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]
