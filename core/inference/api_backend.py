# file: core/inference/api_backend.py
"""
API 推理后端

使用 DeepSeek API 进行推理。
"""

import os
from types import ModuleType
from typing import Optional, List, Any, cast

from config import DEFAULT_DEEPSEEK_API_KEY
from core.llm.router import sync_llm_complete

_openai_module: Optional[ModuleType]
try:
    import openai as _openai_module
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    _openai_module = None


class APIBackend:
    """
    DeepSeek API 推理后端
    
    使用 OpenAI 兼容的 API 进行推理。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-reasoner",
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or DEFAULT_DEEPSEEK_API_KEY
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._client: Optional[Any] = None

    def _mode_for_model(self) -> str:
        name = str(self.model or "").lower()
        if "v4-pro" in name or "reasoner" in name:
            return "slow"
        return "fast"
        
    def _get_client(self) -> Any:
        if self._client is None:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai 库未安装，请运行: pip install openai")
            if not self.api_key:
                raise ValueError("DEEPSEEK_API_KEY 未设置")
            module: Optional[ModuleType] = cast(Optional[ModuleType], _openai_module)
            if module is None:
                raise ImportError("openai 客户端不可用")
            self._client = module.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 额外参数
            
        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            routed = sync_llm_complete(
                messages,
                mode=kwargs.get("mode", self._mode_for_model()),
                task_type=kwargs.get("task_type"),
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                timeout=kwargs.get("timeout_budget", kwargs.get("timeout")),
                fallback_response=kwargs.get("fallback_response"),
            )
            if routed.ok or kwargs.get("fallback_response") is not None:
                return routed.text
        except Exception:
            # Fallback to legacy direct client path if router initialization/call fails.
            pass
        
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            fallback = kwargs.get("fallback_response")
            if fallback is not None:
                return str(fallback)
            return f"[API Error] {e}"
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """批量生成 (串行，未来可优化为并行)"""
        return [self.generate(p, system_prompt) for p in prompts]
