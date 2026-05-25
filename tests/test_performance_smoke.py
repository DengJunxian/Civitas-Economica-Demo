from __future__ import annotations

import pytest

from core.data.market_data_provider import MarketDataProvider, MarketDataQuery
from core.llm import LLMSettings, LLMRouter
from scripts.benchmark_simulation import run_benchmark


def test_market_data_provider_uses_synthetic_fallback_when_network_unavailable(tmp_path) -> None:
    provider = MarketDataProvider(
        cache_dir=str(tmp_path / "cache"),
        snapshot_dir=str(tmp_path / "snapshots"),
        provider_priority=("akshare", "yfinance", "ashare"),
    )
    provider._fetch_from_akshare = lambda _query: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[method-assign]
    provider._fetch_from_yfinance = lambda _query: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[method-assign]
    provider._fetch_from_ashare = lambda _query: (_ for _ in ()).throw(RuntimeError("offline"))  # type: ignore[method-assign]

    frame = provider.get_ohlcv(MarketDataQuery(symbol="sh000001", interval="1d", period_days=20, adjust="", market="CN"), use_cache=False)

    assert not frame.empty
    assert frame["provider"].eq("synthetic").all()
    assert {"open", "high", "low", "close", "volume"}.issubset(frame.columns)


@pytest.mark.asyncio
async def test_llm_router_no_key_smoke_uses_mockable_offline_path(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    router = LLMRouter(settings=LLMSettings(deepseek_api_key="", zhipu_api_key=""))

    response = await router.complete([{"role": "user", "content": "policy"}], mode="slow")

    assert response.provider == "offline"
    assert response.model == "deterministic_stub"
    assert router.get_call_records()
    assert "content" not in router.get_call_records()[-1]


@pytest.mark.asyncio
async def test_performance_benchmark_smoke_runs_quickly() -> None:
    result = await run_benchmark(agents=4, ticks=1)

    assert result["ticks"] == 1
    assert result["agent_count"] == 4
    assert result["decisions_per_second"] > 0
    assert result["tick_latency_ms"] >= 0
