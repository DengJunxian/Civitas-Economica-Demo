"""LLM 调用客户端（带容错 JSON 修复与抖动重试）。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_random


# 预编译正则，减少高并发时重复编译开销
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_non_standard_output(raw_text: str) -> str:
    """移除 `<think>` 标签、Markdown 围栏和不可见控制字符。"""
    if not raw_text:
        return ""

    text = _THINK_TAG_RE.sub("", raw_text)
    text = _CONTROL_CHAR_RE.sub("", text)

    # 优先提取 markdown code block 里的主体
    fence_match = _CODE_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    return text.strip()


def _auto_close_brackets(candidate: str) -> str:
    """自动闭合被截断的 JSON 花括号和方括号。"""
    if not candidate:
        return candidate

    stack: List[str] = []
    in_string = False
    escape = False

    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    if in_string:
        candidate += '"'

    while stack:
        opener = stack.pop()
        candidate += "}" if opener == "{" else "]"

    return candidate


def _try_fix_json(raw_text: str) -> str:
    """
    尝试修复大模型输出中的“伪 JSON/截断 JSON”。

    修复策略：
    1) 清洗 `<think>`、代码围栏、控制字符。
    2) 从文本中提取最可能的 JSON 对象或数组。
    3) 自动闭合 `{`/`[`，再进行一次 JSON 有效性校验。
    """
    cleaned = _strip_non_standard_output(raw_text)
    if not cleaned:
        return ""

    # 先尝试整体解析
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(cleaned) or _JSON_ARRAY_RE.search(cleaned)
    candidate = match.group(0).strip() if match else cleaned
    candidate = _auto_close_brackets(candidate)

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return cleaned


class RobustLLMClient:
    """对 AsyncOpenAI 的轻量封装：提供抖动退避重试 + 输出容错修复。"""

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        # 指数退避 + 微抖动，避免并发失败时同时重试导致惊群
        wait=wait_exponential(multiplier=0.25, min=0.25, max=2.0) + wait_random(0, 0.2),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
    )
    async def chat_completion(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        timeout: float,
    ) -> Tuple[str, Optional[str]]:
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )

        message = response.choices[0].message
        content = _try_fix_json(message.content or "")
        reasoning = _strip_non_standard_output(getattr(message, "reasoning_content", None) or "")
        return content, reasoning
