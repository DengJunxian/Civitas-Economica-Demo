"""Run parameter sensitivity scan for behavioral-finance layer.

Usage:
  python scripts/run_parameter_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.simulation_loop import MarketEnvironment


class _DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.persona = type("PersonaStub", (), {"risk_tolerance": 0.5})()
        self.memory_bank = None

    async def generate_trading_decision(self, market_data, retrieved_context):
        _ = market_data, retrieved_context
        return type("Action", (), {"action": "HOLD", "amount": 0.0, "target_price": None})()


def main() -> None:
    agents = [_DummyAgent("scan_agent_1"), _DummyAgent("scan_agent_2")]
    env = MarketEnvironment(agents, use_isolated_matching=False, runner_symbol="SCAN")
    try:
        out = env.run_parameter_sensitivity_scan(
            output_csv=Path("outputs") / "parameter_sensitivity.csv",
        )
        print(f"Saved sensitivity CSV: {out}")
        print(f"Saved summary JSON: {out.with_name('parameter_sensitivity_summary.json')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
