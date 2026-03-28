"""Natural-language rendering helpers for structured UI payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

import streamlit as st


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _compact_payload(value: Any, *, depth: int = 0, max_depth: int = 3, max_items: int = 8) -> Any:
    if depth >= max_depth:
        if _is_scalar(value):
            return value
        return str(value)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["..."] = f"trimmed_{len(value) - max_items}_items"
                break
            out[str(key)] = _compact_payload(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return out
    if isinstance(value, list):
        head = value[:max_items]
        out = [_compact_payload(item, depth=depth + 1, max_depth=max_depth, max_items=max_items) for item in head]
        if len(value) > max_items:
            out.append(f"trimmed_{len(value) - max_items}_items")
        return out
    if _is_scalar(value):
        return value
    return str(value)


def _flatten_payload(
    value: Any,
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 3,
    max_items: int = 16,
    out: List[Tuple[str, str]] | None = None,
) -> List[Tuple[str, str]]:
    if out is None:
        out = []
    if len(out) >= max_items:
        return out
    if depth >= max_depth or _is_scalar(value):
        key = prefix or "value"
        out.append((key, str(value)))
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            if len(out) >= max_items:
                break
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_payload(
                item,
                prefix=next_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                out=out,
            )
        return out
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if len(out) >= max_items:
                break
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten_payload(
                item,
                prefix=next_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                out=out,
            )
        return out
    key = prefix or "value"
    out.append((key, str(value)))
    return out


def _fallback_narrative(title: str, payload: Any, context: str) -> str:
    pairs = _flatten_payload(payload)
    if not pairs:
        return f"**一句话结论**：{title}暂无可解读内容。"
    lines = [
        f"**一句话结论**：{title}已生成，核心变化集中在关键指标与风险信号。",
        "**关键观察**：",
    ]
    if context:
        lines.append(f"- 解读范围：{context}")
    for key, value in pairs[:6]:
        lines.append(f"- {key}: {value}")
    lines.append("**风险与建议**：")
    lines.append("- 建议先看趋势方向，再结合波动和风险项判断执行节奏。")
    if len(pairs) > 6:
        lines.append("- 其余细节已省略，可结合表格进一步查看。")
    return "\n".join(lines)


def _llm_narrative(title: str, payload: Any, context: str) -> str:
    try:
        from core.inference.api_backend import APIBackend
    except Exception:
        return ""

    compact = _compact_payload(payload)
    serialized = json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str)
    if len(serialized) > 5000:
        serialized = f"{serialized[:5000]}...(truncated)"
    prompt = "\n".join(
        [
            "请把下面结构化信息写成给评委看的中文自然语言解读。",
            "要求：不要输出 JSON、代码块、键名清单。",
            "请严格按这个格式输出：",
            "【一句话结论】",
            "【关键观察】(4-6条)",
            "【风险提示】(1条)",
            "【建议关注指标】(2条)",
            f"标题：{title}",
            f"上下文：{context or '无'}",
            f"数据：{serialized}",
        ]
    )

    try:
        backend = APIBackend(model="deepseek-chat", max_tokens=420, temperature=0.25)
        content = str(
            backend.generate(
                prompt,
                system_prompt="你是比赛答辩讲解助手，擅长把结构化数据翻译成清晰自然语言。",
                timeout_budget=20.0,
            )
            or ""
        ).strip()
    except Exception:
        return ""
    if not content or content.startswith("[API Error]"):
        return ""
    stripped = content.lstrip()
    if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```"):
        return ""
    return content


def narrate_payload(title: str, payload: Any, *, context: str = "", cache_namespace: str = "ui_narrative_cache") -> str:
    compact = _compact_payload(payload)
    cache_key = hashlib.sha256(
        json.dumps(
            {"title": title, "context": context, "payload": compact},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    cache: Dict[str, str] = st.session_state.setdefault(cache_namespace, {})
    if cache_key in cache:
        return cache[cache_key]

    text = _llm_narrative(title, compact, context)
    if not text:
        text = _fallback_narrative(title, compact, context)
    cache[cache_key] = text
    return text
