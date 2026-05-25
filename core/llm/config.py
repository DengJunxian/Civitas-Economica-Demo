"""Environment-backed LLM configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True, slots=True)
class LLMSettings:
    deepseek_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", "").strip())
    zhipu_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("ZHIPUAI_API_KEY", "").strip()
            or os.environ.get("ZHIPU_API_KEY", "").strip()
        )
    )
    default_provider: str = field(default_factory=lambda: os.environ.get("LLM_DEFAULT_PROVIDER", "auto").strip() or "auto")
    timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 20.0))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))
    deepseek_base_url: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com").strip())
    zhipu_base_url: str = field(default_factory=lambda: os.environ.get("ZHIPU_API_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip())


def load_llm_settings() -> LLMSettings:
    return LLMSettings()

