"""Benchmark one online DeepSeek-backed simulation round.

This script intentionally reads DEEPSEEK_API_KEY from the local environment or
the git-ignored `.env` file. It does not contain or print API keys.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL = "deepseek-chat"
SYMBOL = "A_SHARE_IDX"


def _load_local_env_file(path: Path = REPO_ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class _Action:
    action: str
    amount: float
    target_price: float


class _Persona:
    def __init__(self, archetype_key: str, risk_tolerance: float = 0.5) -> None:
        self.archetype_key = archetype_key
        self.risk_tolerance = float(risk_tolerance)


class StrictDeepSeekAgent:
    client = None
    latencies_ms: List[float] = []
    raw_models: List[str] = []

    def __init__(self, agent_id: str, archetype_key: str, *, use_llm: bool = True) -> None:
        self.agent_id = agent_id
        self.persona = _Persona(archetype_key, risk_tolerance=0.65 if use_llm else 0.45)
        self.psychology_profile = {"institution_type": archetype_key}
        self.use_llm = bool(use_llm)
        self.portfolio = {SYMBOL: 300}

    async def generate_trading_decision(self, market_data: Dict[str, Any], retrieved_context: str) -> _Action:
        price = float(market_data.get("current_price", market_data.get("last_price", 100.0)) or 100.0)
        if not self.use_llm:
            return _Action("BUY" if "momentum" in self.persona.archetype_key else "SELL", 25.0, price)

        prompt = {
            "agent_id": self.agent_id,
            "archetype": self.persona.archetype_key,
            "price": price,
            "tick": market_data.get("tick"),
            "policy_or_news": market_data.get("latest_broadcast", ""),
            "macro_context": market_data.get("macro_context", {}),
            "instruction": "Return JSON only: action BUY/SELL/HOLD, amount numeric, target_price numeric.",
        }
        started = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an A-share trading agent. Return strict JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=float(os.environ.get("CIVITAS_DEEPSEEK_DIRECT_TIMEOUT_SECONDS", "20") or 20.0),
        )
        StrictDeepSeekAgent.latencies_ms.append((time.perf_counter() - started) * 1000.0)
        StrictDeepSeekAgent.raw_models.append(str(getattr(response, "model", MODEL) or MODEL))
        parsed = json.loads(response.choices[0].message.content or "{}")
        action = str(parsed.get("action", "HOLD")).upper()
        if action not in {"BUY", "SELL", "HOLD"}:
            action = "HOLD"
        return _Action(
            action=action,
            amount=max(0.0, float(parsed.get("amount", 0.0) or 0.0)),
            target_price=max(0.01, float(parsed.get("target_price", price) or price)),
        )


def _build_agents(llm_agents: int) -> List[StrictDeepSeekAgent]:
    archetypes = [
        "mutual_fund",
        "market_maker",
        "quant_arbitrage",
        "policy_capital",
        "retail_momentum_chaser",
        "retail_swing",
    ]
    agents: List[StrictDeepSeekAgent] = []
    for idx, archetype in enumerate(archetypes):
        agents.append(StrictDeepSeekAgent(f"strict_{idx:02d}", archetype, use_llm=idx < int(llm_agents)))
    return agents


async def run_once(*, llm_agents: int = 4) -> Dict[str, Any]:
    _load_local_env_file()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured in environment or .env")

    from openai import AsyncOpenAI

    from core.reproducibility import seed_everything
    from engine.simulation_loop import MarketEnvironment

    seed_everything(42)
    StrictDeepSeekAgent.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=20.0)
    StrictDeepSeekAgent.latencies_ms.clear()
    StrictDeepSeekAgent.raw_models.clear()

    agents = _build_agents(llm_agents=llm_agents)
    env = MarketEnvironment(
        agents,
        runner_symbol=SYMBOL,
        simulation_mode="DEEP",
        llm_primary=True,
        model_priority=[MODEL],
        use_isolated_matching=True,
        market_pipeline_v2=True,
        enable_random_policy_events=False,
        deep_reasoning_pause_s=0.0,
    )
    started = time.perf_counter()
    try:
        env.schedule_experiment_event(
            "央行释放流动性并稳定资本市场预期，监管同步强化异常交易监测。",
            event_type="policy",
            title="strict_deepseek_benchmark_policy",
            strength=1.0,
        )
        report = await env.simulation_step()
    finally:
        env.close()
        await StrictDeepSeekAgent.client.close()

    elapsed = time.perf_counter() - started
    thinking = dict(report.get("thinking_stats", {}) or {})
    matching_mode = str(report.get("matching_mode", ""))
    llm_latencies = list(StrictDeepSeekAgent.latencies_ms)
    no_fallback = len(llm_latencies) == int(llm_agents) and matching_mode != "fallback_impact_model"
    return {
        "benchmark_name": "strict_deepseek_direct_one_round",
        "model_requested": MODEL,
        "models_returned": StrictDeepSeekAgent.raw_models,
        "simulation_rounds": 1,
        "agent_count": len(agents),
        "llm_agent_count": int(llm_agents),
        "deepseek_call_count": len(llm_latencies),
        "deepseek_latency_ms": [round(x, 1) for x in llm_latencies],
        "deepseek_latency_total_ms": round(sum(llm_latencies), 1),
        "deepseek_latency_avg_ms": round(sum(llm_latencies) / max(len(llm_latencies), 1), 1),
        "wall_time_seconds": round(elapsed, 3),
        "wall_time_ms": round(elapsed * 1000.0, 1),
        "no_fallback": bool(no_fallback),
        "simulation": {
            "tick": report.get("tick"),
            "matching_mode": matching_mode,
            "trade_count": report.get("trade_count"),
            "old_price": report.get("old_price"),
            "new_price": report.get("new_price"),
            "buy_volume": report.get("buy_volume"),
            "sell_volume": report.get("sell_volume"),
        },
        "thinking_stats": {
            "slow_agent_count": thinking.get("slow_agent_count"),
            "fast_agent_count": thinking.get("fast_agent_count"),
            "llm_call_count_scheduler_estimate": thinking.get("llm_call_count"),
            "avg_slow_agent_latency_ms": thinking.get("avg_slow_agent_latency_ms"),
            "cache_hit_rate": thinking.get("cache_hit_rate"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one DeepSeek online simulation round.")
    parser.add_argument("--llm-agents", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run_once(llm_agents=args.llm_agents)), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
