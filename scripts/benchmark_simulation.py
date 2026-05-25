"""Lightweight local simulation benchmark.

Runs the default CPU fallback path and prints trend-friendly JSON metrics. This
is intentionally relative, not a hard CI threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.simulation_loop import MarketEnvironment


@dataclass
class _Action:
    action: str = "HOLD"
    amount: float = 0.0
    target_price: float = 100.0


class _Persona:
    def __init__(self, archetype_key: str, risk_tolerance: float = 0.5) -> None:
        self.archetype_key = archetype_key
        self.risk_tolerance = risk_tolerance


class _Agent:
    def __init__(self, agent_id: str, archetype_key: str, *, use_llm: bool = False) -> None:
        self.agent_id = agent_id
        self.persona = _Persona(archetype_key)
        self.psychology_profile = {"institution_type": archetype_key}
        self.use_llm = use_llm
        self.portfolio = {"A_SHARE_IDX": 100}

    async def generate_trading_decision(self, market_data, retrieved_context):
        price = float(market_data.get("current_price", market_data.get("last_price", 100.0)) or 100.0)
        return _Action(action="BUY" if "momentum" in self.persona.archetype_key else "SELL", amount=25.0, target_price=price)


def _build_agents(count: int) -> List[_Agent]:
    agents: List[_Agent] = []
    for idx in range(max(1, int(count))):
        if idx == 0:
            agents.append(_Agent(f"strategic_{idx}", "mutual_fund", use_llm=False))
        else:
            archetype = "retail_momentum_chaser" if idx % 2 else "retail_mean_reverter"
            agents.append(_Agent(f"fast_{idx}", archetype, use_llm=False))
    return agents


async def run_benchmark(*, agents: int = 32, ticks: int = 12) -> Dict[str, Any]:
    env = MarketEnvironment(
        _build_agents(agents),
        runner_symbol="A_SHARE_IDX",
        llm_primary=False,
        use_isolated_matching=False,
    )
    reports: List[Dict[str, Any]] = []
    started = time.perf_counter()
    try:
        env.schedule_experiment_event(
            "benchmark liquidity and expectation shock",
            event_type="major_news",
            title="benchmark_news",
            strength=1.0,
        )
        for _ in range(max(1, int(ticks))):
            reports.append(dict(await env.simulation_step()))
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    latest = reports[-1] if reports else {}
    thinking = dict(latest.get("thinking_stats", {}) or {})
    return {
        "ticks": int(len(reports)),
        "agent_count": int(agents),
        "tick_latency_ms": float(elapsed * 1000.0 / max(len(reports), 1)),
        "decisions_per_second": float((len(reports) * max(1, int(agents))) / max(elapsed, 1e-9)),
        "llm_calls_per_tick": float(thinking.get("llm_call_count", 0.0) or 0.0) / max(len(reports), 1),
        "cache_hit_rate": float(thinking.get("cache_hit_rate", 0.0) or 0.0),
        "fast_agent_count": int(thinking.get("fast_agent_count", 0) or 0),
        "slow_agent_count": int(thinking.get("slow_agent_count", 0) or 0),
        "memory_footprint_mb_approx": 0.0,
        "batch_scenario_throughput": float(len(reports) / max(elapsed, 1e-9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_benchmark(agents=args.agents, ticks=args.ticks)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
