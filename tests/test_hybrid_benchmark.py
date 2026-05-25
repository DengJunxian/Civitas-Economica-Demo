from __future__ import annotations

from core.hybrid_simulation_benchmark import HybridBenchmarkConfig, HybridSimulationBenchmark


def test_hybrid_benchmark_has_expected_rows():
    config = HybridBenchmarkConfig(agent_counts=(1000, 5000), steps=3, seed=5)
    bench = HybridSimulationBenchmark(config)
    result = bench.run()
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        assert row["agent_count"] in (1000, 5000)
        assert row["avg_step_latency_ms"] >= 0.0
        assert row["memory_peak_mb"] >= 0.0
        assert row["determinism_hash"]


def test_hybrid_benchmark_determinism_hash_stable():
    cfg = HybridBenchmarkConfig(agent_counts=(1000,), steps=2, seed=13)
    a = HybridSimulationBenchmark(cfg).run()
    b = HybridSimulationBenchmark(cfg).run()
    assert a["rows"][0]["determinism_hash"] == b["rows"][0]["determinism_hash"]
