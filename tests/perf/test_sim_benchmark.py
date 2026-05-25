from __future__ import annotations

import pytest

from scripts.benchmark_simulation import run_benchmark


@pytest.mark.asyncio
async def test_sim_benchmark_reports_trend_metrics():
    result = await run_benchmark(agents=8, ticks=2)

    assert result["ticks"] == 2
    assert result["agent_count"] == 8
    assert result["tick_latency_ms"] >= 0.0
    assert result["decisions_per_second"] > 0.0
    assert "llm_calls_per_tick" in result
    assert "cache_hit_rate" in result
    assert result["fast_agent_count"] >= 0
    assert result["slow_agent_count"] >= 0

