import logging

import pytest

from core.llm import LLMResponse, LLMRouter, LLMSettings, load_llm_settings


class FakeClient:
    def __init__(self, provider: str, outcomes):
        self.provider = provider
        self.outcomes = list(outcomes)
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is False:
            return LLMResponse(
                text="",
                provider=self.provider,
                model=str(kwargs.get("model")),
                latency_ms=1.0,
                raw={"error": "timeout"},
                ok=False,
            )
        return LLMResponse(
            text=str(outcome),
            provider=self.provider,
            model=str(kwargs.get("model")),
            latency_ms=1.0,
            ok=True,
        )


@pytest.mark.asyncio
async def test_deepseek_success_uses_slow_primary_model():
    deepseek = FakeClient("deepseek", ["ok"])
    router = LLMRouter(deepseek_client=deepseek, zhipu_client=FakeClient("zhipu", []))

    response = await router.complete([{"role": "user", "content": "policy"}], mode="slow")

    assert response.ok is True
    assert response.text == "ok"
    assert response.model == "deepseek-v4-pro"
    assert response.fallback_chain == ["deepseek:deepseek-v4-pro:thinking=true"]
    assert deepseek.calls[0]["thinking"] is True


@pytest.mark.asyncio
async def test_deepseek_timeout_falls_back_to_flash_non_thinking():
    deepseek = FakeClient("deepseek", [False, "flash-ok"])
    router = LLMRouter(deepseek_client=deepseek, zhipu_client=FakeClient("zhipu", []))

    response = await router.complete([{"role": "user", "content": "policy"}], mode="slow")

    assert response.ok is True
    assert response.model == "deepseek-v4-flash"
    assert response.fallback_chain == [
        "deepseek:deepseek-v4-pro:thinking=true",
        "deepseek:deepseek-v4-flash:thinking=false",
    ]
    assert deepseek.calls[1]["thinking"] is False


@pytest.mark.asyncio
async def test_deepseek_all_failed_falls_back_to_glm():
    deepseek = FakeClient("deepseek", [False, False])
    zhipu = FakeClient("zhipu", ["glm-ok"])
    router = LLMRouter(deepseek_client=deepseek, zhipu_client=zhipu)

    response = await router.complete([{"role": "user", "content": "policy"}], mode="slow")

    assert response.ok is True
    assert response.provider == "zhipu"
    assert response.model == "glm-4-flashx"
    assert response.fallback_chain[-1] == "zhipu:glm-4-flashx"


@pytest.mark.asyncio
async def test_fast_low_latency_tasks_do_not_call_slow_model():
    deepseek = FakeClient("deepseek", ["fast-ok"])
    router = LLMRouter(deepseek_client=deepseek, zhipu_client=FakeClient("zhipu", []))

    response = await router.complete(
        [{"role": "user", "content": "tick"}],
        mode="fast",
        task_type="belief_update",
    )

    assert response.ok is True
    assert response.model == "deepseek-v4-flash"
    assert response.fallback_chain == ["deepseek:deepseek-v4-flash:thinking=false"]


@pytest.mark.asyncio
async def test_smart_mode_uses_zhipu_only():
    deepseek = FakeClient("deepseek", ["should-not-run"])
    zhipu = FakeClient("zhipu", ["glm-ok"])
    router = LLMRouter(deepseek_client=deepseek, zhipu_client=zhipu)

    response = await router.complete([{"role": "user", "content": "policy"}], mode="smart")

    assert response.ok is True
    assert response.provider == "zhipu"
    assert response.model == "glm-4-flashx"
    assert response.fallback_chain == ["zhipu:glm-4-flashx"]
    assert deepseek.calls == []


@pytest.mark.asyncio
async def test_smart_mode_ignores_deepseek_model_override():
    deepseek = FakeClient("deepseek", ["should-not-run"])
    zhipu = FakeClient("zhipu", ["glm-ok"])
    router = LLMRouter(deepseek_client=deepseek, zhipu_client=zhipu)

    response = await router.complete(
        [{"role": "user", "content": "policy"}],
        mode="smart",
        model="deepseek-chat",
    )

    assert response.ok is True
    assert response.provider == "zhipu"
    assert response.model == "glm-4-flashx"
    assert deepseek.calls == []


@pytest.mark.asyncio
async def test_router_logs_do_not_expose_authorization_header(caplog):
    caplog.set_level(logging.WARNING, logger="civitas.llm.router")
    leaked_header = "Authorization" + ": Bearer should_not_leak"
    deepseek = FakeClient("deepseek", [RuntimeError(f"upstream failed {leaked_header}")])
    router = LLMRouter(deepseek_client=deepseek, zhipu_client=FakeClient("zhipu", ["glm-ok"]))

    response = await router.complete([{"role": "user", "content": "policy"}], mode="auto")

    assert response.ok is True
    assert "should_not_leak" not in caplog.text
    assert "Authorization:" not in caplog.text


def test_zhipu_old_env_alias_is_supported(monkeypatch):
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.setenv("ZHIPU_API_KEY", "legacy-zhipu-value")

    settings = load_llm_settings()

    assert settings.zhipu_api_key == "legacy-zhipu-value"


@pytest.mark.asyncio
async def test_no_api_key_path_returns_offline_response_without_crashing():
    settings = LLMSettings(deepseek_api_key="", zhipu_api_key="")
    router = LLMRouter(settings=settings)

    response = await router.complete([{"role": "user", "content": "policy"}], mode="fast")

    assert response.ok is False
    assert response.provider == "offline"
    assert response.fallback_chain
