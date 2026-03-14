"""
Regulator Agent 训练脚本。

支持两种模式：
1) toy: 轻量环境快速训练；
2) controller: 使用 SimulationController 进行真实沙盘训练。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 确保从 scripts 目录执行时也能导入项目根目录模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regulator_agent import (
    build_simulation_controller_factory,
    run_controller_regulatory_optimization,
    run_regulatory_optimization,
    training_summary_to_dict,
)


def _resolve_key(cli_value: str | None, env_name: str) -> str | None:
    if cli_value:
        return cli_value
    val = os.getenv(env_name)
    if val:
        return val
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train regulator RL agent and export best policy bundle."
    )
    parser.add_argument(
        "--env",
        choices=["toy", "controller"],
        default="toy",
        help="Training environment type.",
    )
    parser.add_argument("--episodes", type=int, default=120, help="Number of training episodes.")
    parser.add_argument("--steps", type=int, default=24, help="Max steps per episode.")
    parser.add_argument("--mode", type=str, default="FAST", help="Simulation mode for controller env.")
    parser.add_argument("--deepseek-key", type=str, default=None, help="DeepSeek API key.")
    parser.add_argument("--zhipu-key", type=str, default=None, help="Zhipu API key (optional).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path. Default: outputs/regulator/<timestamp>.json",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.env == "toy":
        summary = run_regulatory_optimization(
            episodes=args.episodes,
            max_steps_per_episode=args.steps,
            use_toy_env=True,
        )
    else:
        deepseek_key = _resolve_key(args.deepseek_key, "DEEPSEEK_API_KEY")
        zhipu_key = _resolve_key(args.zhipu_key, "ZHIPU_API_KEY")
        if not deepseek_key:
            raise SystemExit(
                "Controller mode requires DeepSeek key. Provide --deepseek-key or set DEEPSEEK_API_KEY."
            )

        controller_factory = build_simulation_controller_factory(
            deepseek_key=deepseek_key,
            zhipu_key=zhipu_key,
            mode=args.mode,
        )
        summary = run_controller_regulatory_optimization(
            controller_factory=controller_factory,
            episodes=args.episodes,
            max_steps_per_episode=args.steps,
        )

    payload: dict[str, Any] = training_summary_to_dict(summary)
    payload["meta"] = {
        "env": args.env,
        "mode": args.mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path("outputs") / "regulator" / f"regulator_summary_{args.env}_{stamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== Regulator Optimization Completed ===")
    print(f"Output: {output_path.resolve()}")
    print(f"Episodes: {payload['episodes']}")
    print(f"Avg Episode Reward: {payload['avg_episode_reward']:.4f}")
    print(f"Best Action Score: {payload['best_action_score']:.4f}")
    print(f"Best Action: {payload['best_action_description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
