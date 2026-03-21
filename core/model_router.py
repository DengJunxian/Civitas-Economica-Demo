# file: core/model_router.py
"""Unified model router with fallback, local cache, and deterministic stub."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from config import GLOBAL_CONFIG


class ModelType(Enum):
    """Model class used for routing heuristics."""

    REASONER = "reasoner"
    CHAT = "chat"
    FLASH = "flash"


@dataclass
class ModelInfo:
    name: str
    provider: str
    model_type: ModelType
    base_url: str
    timeout: float


@dataclass
class ModelStats:
    call_count: int = 0
    total_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

    @property
    def avg_time(self) -> float:
        return 0.0 if self.call_count == 0 else self.total_time / self.call_count

    @property
    def success_rate(self) -> float:
        return 1.0 if self.call_count == 0 else 1.0 - (self.error_count / self.call_count)


class ModelRouter:
    """Multi-provider router for all LLM calls in Civitas."""

    MODEL_REGISTRY: Dict[str, ModelInfo] = {
        "deepseek-reasoner": ModelInfo(
            name="deepseek-reasoner",
            provider="deepseek",
            model_type=ModelType.REASONER,
            base_url=GLOBAL_CONFIG.API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER,
        ),
        "deepseek-chat": ModelInfo(
            name="deepseek-chat",
            provider="deepseek",
            model_type=ModelType.CHAT,
            base_url=GLOBAL_CONFIG.API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_CHAT,
        ),
        "glm-4-flashx": ModelInfo(
            name="glm-4-flashx",
            provider="zhipu",
            model_type=ModelType.FLASH,
            base_url=GLOBAL_CONFIG.ZHIPU_API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_FLASH,
        ),
        "glm-4-flashx-250414": ModelInfo(
            name="glm-4-flashx-250414",
            provider="zhipu",
            model_type=ModelType.FLASH,
            base_url=GLOBAL_CONFIG.ZHIPU_API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_FLASH,
        ),
    }

    def __init__(
        self,
        deepseek_key: str,
        zhipu_key: Optional[str] = None,
        local_cache_path: Optional[str] = None,
    ):
        self.deepseek_key = deepseek_key
        self.zhipu_key = zhipu_key

        self.clients: Dict[str, AsyncOpenAI] = {}
        self._init_clients()

        self.stats: Dict[str, ModelStats] = {model: ModelStats() for model in self.MODEL_REGISTRY}
        self.available_models = self._get_available_models()

        self.fallback_events: List[Dict[str, Any]] = []
        self.has_deepseek = bool(deepseek_key)
        self.has_zhipu = bool(zhipu_key)

        default_cache_path = Path("data") / "cache" / "model_router_cache.json"
        self.local_cache_path = Path(local_cache_path) if local_cache_path else default_cache_path
        self.local_cache: Dict[str, Dict[str, Any]] = {}
        self._load_local_cache()

    def _init_clients(self):
        if self.deepseek_key:
            self.clients["deepseek"] = AsyncOpenAI(
                api_key=self.deepseek_key,
                base_url=GLOBAL_CONFIG.API_BASE_URL,
                timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER,
            )
        if self.zhipu_key:
            self.clients["zhipu"] = AsyncOpenAI(
                api_key=self.zhipu_key,
                base_url=GLOBAL_CONFIG.ZHIPU_API_BASE_URL,
                timeout=GLOBAL_CONFIG.API_TIMEOUT_FLASH,
            )

    def _get_available_models(self) -> List[str]:
        available: List[str] = []
        if self.deepseek_key:
            available.extend(["deepseek-reasoner", "deepseek-chat"])
        if self.zhipu_key:
            available.extend(["glm-4-flashx", "glm-4-flashx-250414"])
        return available

    def _build_cache_key(self, messages: List[Dict], priority_models: List[str]) -> str:
        payload = {"messages": messages, "priority_models": priority_models}
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _load_local_cache(self):
        try:
            if not self.local_cache_path.exists():
                return
            raw = json.loads(self.local_cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.local_cache = raw
        except Exception:
            self.local_cache = {}

    def _persist_local_cache(self):
        try:
            self.local_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.local_cache_path.write_text(
                json.dumps(self.local_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _read_local_cache(self, cache_key: Optional[str]) -> Optional[Tuple[str, Optional[str], str]]:
        if not cache_key:
            return None
        item = self.local_cache.get(cache_key)
        if not item:
            return None
        return (
            str(item.get("content", "")),
            item.get("reasoning"),
            str(item.get("model_used", "local_cache")),
        )

    def _write_local_cache(
        self,
        cache_key: Optional[str],
        content: str,
        reasoning: Optional[str],
        model_used: str,
    ):
        if not cache_key:
            return
        self.local_cache[cache_key] = {
            "content": content,
            "reasoning": reasoning,
            "model_used": model_used,
            "timestamp": time.time(),
        }
        self._persist_local_cache()

    def register_local_response(
        self,
        cache_key: str,
        content: str,
        reasoning: Optional[str] = None,
        model_used: str = "local_cache",
    ):
        self._write_local_cache(cache_key, content, reasoning, model_used)

    def _build_deterministic_stub(self, messages: List[Dict], error: Optional[str]) -> str:
        seed_text = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        score = int(digest[:2], 16) / 255.0
        if score > 0.66:
            action, qty = "BUY", 120
        elif score < 0.33:
            action, qty = "SELL", 80
        else:
            action, qty = "HOLD", 0
        payload = {
            "action": action,
            "qty": qty,
            "confidence": round(abs(score - 0.5) * 2, 3),
            "error": error or "MODEL_UNAVAILABLE",
            "mode": "deterministic_stub",
        }
        return json.dumps(payload, ensure_ascii=False)

    def get_model_priority(self, mode: str) -> List[str]:
        """
        根据仿真模式获取模型优先级。
        
        SMART (智能模式): GLM (flashx) -> DeepSeek Chat (回退)
        DEEP (深度模式): DeepSeek Reasoner -> DeepSeek Chat (超时/失败回退)
        FAST (快速模式): 同智能模式
        """
        if mode == "DEEP":
            base_priority = ["deepseek-reasoner", "deepseek-chat"]
        elif mode == "SMART" or mode == "FAST":
            base_priority = ["glm-4-flashx", "deepseek-chat"]
        else:
            base_priority = ["deepseek-chat", "glm-4-flashx"]
            
        return [m for m in base_priority if m in self.available_models]

    async def call_with_fallback(
        self,
        messages: List[Dict],
        priority_models: List[str],
        timeout_budget: float = 15.0,
        fallback_response: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[str, Optional[str], str]:
        if not priority_models:
            priority_models = self.get_model_priority("SMART") or ["deepseek-chat", "glm-4-flashx"]

        resolved_cache_key = cache_key or self._build_cache_key(messages, priority_models)
        cached = self._read_local_cache(resolved_cache_key)
        if cached:
            self.fallback_events.append(
                {"type": "cache_hit", "timestamp": time.time(), "message": "local cache hit"}
            )
            return cached

        start_time = time.time()
        last_error = None

        for model_name in priority_models:
            elapsed = time.time() - start_time
            remaining = timeout_budget - elapsed
            if remaining <= 1.0:
                break

            model_info = self.MODEL_REGISTRY.get(model_name)
            if not model_info:
                continue
            client = self.clients.get(model_info.provider)
            if not client:
                continue

            call_timeout = remaining if model_name == priority_models[-1] else min(model_info.timeout * 2.5, remaining)

            try:
                content, reasoning = await self._call_model(client, model_name, messages, call_timeout)
                self._update_stats(model_name, time.time() - start_time, success=True)
                self._write_local_cache(resolved_cache_key, content, reasoning, model_name)
                return content, reasoning, model_name
            except asyncio.TimeoutError:
                last_error = f"{model_name}: timeout"
                self._update_stats(model_name, 0, success=False, error="timeout")
                continue
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = f"{model_name}: {type(e).__name__}"
                self._update_stats(model_name, 0, success=False, error=str(e))
                continue
            except Exception as e:
                last_error = f"{model_name}: {str(e)}"
                self._update_stats(model_name, 0, success=False, error=str(e))
                continue

        return self._fallback_response(
            error=last_error,
            fallback_content=fallback_response,
            messages=messages,
            cache_key=resolved_cache_key,
        )

    def sync_call_with_fallback(
        self,
        messages: List[Dict],
        priority_models: Optional[List[str]] = None,
        timeout_budget: float = 30.0,
        fallback_response: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[str, Optional[str], str]:
        if not priority_models:
            priority_models = ["deepseek-chat", "glm-4-flashx"]
        return self._run_coro_sync(
            self.call_with_fallback(
                messages=messages,
                priority_models=priority_models,
                timeout_budget=timeout_budget,
                fallback_response=fallback_response,
                cache_key=cache_key,
            )
        )

    def _extract_json_object(self, raw: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", raw or "")
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _build_schema_fallback(self, json_schema: Dict[str, Any], fallback_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(fallback_obj, dict):
            return dict(fallback_obj)
        properties = json_schema.get("properties", {}) if isinstance(json_schema, dict) else {}
        output: Dict[str, Any] = {}
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                output[key] = None
                continue
            if "default" in spec:
                output[key] = spec["default"]
                continue
            typ = spec.get("type")
            if typ == "string":
                output[key] = ""
            elif typ == "number":
                output[key] = 0.0
            elif typ == "integer":
                output[key] = 0
            elif typ == "boolean":
                output[key] = False
            elif typ == "array":
                output[key] = []
            elif typ == "object":
                output[key] = {}
            else:
                output[key] = None
        return output

    async def call_with_schema(
        self,
        *,
        messages: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        priority_models: Optional[List[str]] = None,
        timeout_budget: float = 15.0,
        fallback_obj: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        """
        Call LLM with fallback and coerce output into a JSON object schema.

        Returns (parsed_object, model_used, raw_content).
        """
        selected_models = priority_models or self.get_model_priority("SMART")
        schema_hint = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        schema_messages = list(messages) + [
            {
                "role": "system",
                "content": f"Return strictly valid JSON matching this schema hint: {json.dumps(schema_hint, ensure_ascii=False)}",
            }
        ]
        raw_content, _, model_used = await self.call_with_fallback(
            messages=schema_messages,
            priority_models=selected_models,
            timeout_budget=timeout_budget,
            cache_key=cache_key,
        )
        parsed = self._extract_json_object(raw_content)
        if parsed is None:
            parsed = self._build_schema_fallback(json_schema, fallback_obj=fallback_obj)
            model_used = f"{model_used}+schema_fallback"
        return parsed, model_used, raw_content

    def sync_call_with_schema(
        self,
        *,
        messages: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        priority_models: Optional[List[str]] = None,
        timeout_budget: float = 15.0,
        fallback_obj: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str, str]:
        return self._run_coro_sync(
            self.call_with_schema(
                messages=messages,
                json_schema=json_schema,
                priority_models=priority_models,
                timeout_budget=timeout_budget,
                fallback_obj=fallback_obj,
                cache_key=cache_key,
            )
        )

    def _run_coro_sync(self, coro: Any) -> Any:
        """同步调用异步协程，兼容“当前线程事件循环已在运行”的场景。"""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        # 若当前线程已有运行中的事件循环，改用子线程新事件循环执行，避免重入冲突。
        if running_loop and running_loop.is_running():
            result_box: Dict[str, Any] = {}
            error_box: Dict[str, BaseException] = {}

            def _runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_box["value"] = loop.run_until_complete(coro)
                except BaseException as exc:  # noqa: BLE001
                    error_box["error"] = exc
                finally:
                    loop.close()

            t = threading.Thread(target=_runner, daemon=True, name="ModelRouterSyncBridge")
            t.start()
            t.join()
            if "error" in error_box:
                raise error_box["error"]
            return result_box.get("value")

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    async def _call_model(
        self,
        client: AsyncOpenAI,
        model_name: str,
        messages: List[Dict],
        timeout: float,
    ) -> Tuple[str, Optional[str]]:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=cast(Any, messages),
                temperature=0.6,
            ),
            timeout=timeout,
        )
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        return content, reasoning

    def _update_stats(self, model_name: str, call_time: float, success: bool, error: Optional[str] = None):
        stats = self.stats.get(model_name)
        if not stats:
            return
        stats.call_count += 1
        if success:
            stats.total_time += call_time
        else:
            stats.error_count += 1
            stats.last_error = error

    def _fallback_response(
        self,
        error: Optional[str] = None,
        fallback_content: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        cache_key: Optional[str] = None,
    ) -> Tuple[str, Optional[str], str]:
        self.fallback_events.append(
            {
                "type": "fallback",
                "timestamp": time.time(),
                "message": f"model fallback engaged: {error or 'all models unavailable'}",
            }
        )
        if fallback_content is not None:
            content = fallback_content
            model_used = "fallback"
        else:
            content = self._build_deterministic_stub(messages or [], error)
            model_used = "deterministic_stub"
        self._write_local_cache(cache_key, content, f"fallback: {error}" if error else "fallback", model_used)
        return (
            content,
            f"all model calls failed: {error}" if error else "all models unavailable",
            model_used,
        )

    def get_stats_summary(self) -> Dict[str, Any]:
        return {
            model: {
                "calls": stats.call_count,
                "avg_time": f"{stats.avg_time:.2f}s",
                "success_rate": f"{stats.success_rate:.1%}",
                "last_error": stats.last_error,
            }
            for model, stats in self.stats.items()
            if stats.call_count > 0
        }

    def get_recommended_model(self, time_budget: float) -> str:
        candidates: List[Tuple[str, float]] = []
        for model_name in self.available_models:
            stats = self.stats.get(model_name)
            if stats and stats.call_count > 0:
                estimated_time = stats.avg_time * 1.2
            else:
                info = self.MODEL_REGISTRY.get(model_name)
                if not info:
                    continue
                estimated_time = 15.0 if info.model_type == ModelType.REASONER else 3.0
            if estimated_time <= time_budget:
                candidates.append((model_name, estimated_time))

        if not candidates:
            return "deepseek-chat" if "deepseek-chat" in self.available_models else (self.available_models[0] if self.available_models else "fallback")

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
