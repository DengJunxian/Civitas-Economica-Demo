# file: core/simulation_runner.py
"""
仿真运行编排器：将 LOB 作为独立子进程运行（OASIS Runner 风格）。

核心目标：
1) 物理隔离：LLM 决策和 LOB 撮合彻底分离。
2) 异步缓冲：交易意图先入 IPC 队列，再由 LOB 进程高速消费。
3) 防阻塞：LLM 慢 I/O 不影响撮合主循环。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.simulation_ipc import FileSystemIPC


class OASISRunner:
    """管理独立 LOB 子进程。"""

    def __init__(self, ipc_dir: str = "data/ipc", worker_poll_interval: float = 0.01):
        self.ipc_dir = ipc_dir
        self.worker_poll_interval = worker_poll_interval
        self._process: Optional[subprocess.Popen] = None
        self.ipc = FileSystemIPC(base_dir=ipc_dir)

    def start(self) -> None:
        if self._process and self._process.poll() is None:
            return

        cmd = [
            sys.executable,
            "-m",
            "core.simulation_runner",
            "--lob-worker",
            "--ipc-dir",
            self.ipc_dir,
            "--poll-interval",
            str(self.worker_poll_interval),
        ]
        self._process = subprocess.Popen(cmd)

    def stop(self, timeout_s: float = 2.0) -> None:
        if not self._process:
            return

        # 发退出命令，优雅停机。
        cmd_id = self.ipc.push_command("shutdown", {})
        self.ipc.wait_response(cmd_id, timeout_s=timeout_s)

        try:
            self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._process.kill()
        finally:
            self._process = None

    def submit_intentions(self, intentions: List[Dict[str, Any]]) -> List[str]:
        """
        异步提交交易意图到 IPC。

        intentions 示例:
        [{"agent_id":"A1","symbol":"000001","side":"buy","price":10.2,"quantity":100}]
        """
        ids: List[str] = []
        for intent in intentions:
            cmd_id = self.ipc.push_command("trade_intent", intent)
            ids.append(cmd_id)
        return ids

    def flush_responses(self, command_ids: List[str], timeout_s: float = 1.0) -> List[Dict[str, Any]]:
        """批量收集响应。"""
        results: List[Dict[str, Any]] = []
        for command_id in command_ids:
            resp = self.ipc.wait_response(command_id, timeout_s=timeout_s)
            if resp:
                results.append({"command_id": command_id, "status": resp.status, "payload": resp.payload})
            else:
                results.append({"command_id": command_id, "status": "timeout", "payload": {}})
        return results


class _DummyLOBEngine:
    """
    轻量 LOB 演示引擎。

    真实项目可在这里接入 `core.market_engine` 或 C++ LOB 扩展。
    该类只负责演示“异步意图入队 -> 独立进程撮合 -> 回包”的完整链路。
    """

    def __init__(self):
        self.last_trade_price = 0.0

    def process_intent(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        quantity = int(payload.get("quantity", 0))
        price = float(payload.get("price", 0.0))
        side = str(payload.get("side", "hold")).lower()

        # 简化版撮合回执
        filled = quantity if side in {"buy", "sell"} and quantity > 0 else 0
        self.last_trade_price = price if filled else self.last_trade_price
        return {
            "filled_qty": filled,
            "avg_price": price if filled else None,
            "last_trade_price": self.last_trade_price,
            "side": side,
        }


def lob_worker_main(ipc_dir: str, poll_interval: float = 0.01) -> None:
    """LOB 子进程入口。"""
    ipc = FileSystemIPC(base_dir=ipc_dir)
    engine = _DummyLOBEngine()

    running = True
    while running:
        # 回收僵死命令，防止异常中断导致处理队列卡死。
        ipc.collect_stale_processing(stale_after_s=5.0)

        cmd = ipc.pop_next_command()
        if not cmd:
            time.sleep(poll_interval)
            continue

        if cmd.command_type == "shutdown":
            ipc.ack_command(cmd.command_id, "ok", {"message": "lob worker stopped"})
            running = False
            continue

        if cmd.command_type == "trade_intent":
            result = engine.process_intent(cmd.payload)
            ipc.ack_command(cmd.command_id, "ok", result)
            continue

        ipc.ack_command(cmd.command_id, "error", {"message": f"unknown command: {cmd.command_type}"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulation runner with OASIS-style IPC isolation")
    parser.add_argument("--lob-worker", action="store_true", help="start as LOB worker process")
    parser.add_argument("--ipc-dir", default="data/ipc", help="ipc base directory")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="worker polling interval")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.lob_worker:
        lob_worker_main(ipc_dir=args.ipc_dir, poll_interval=args.poll_interval)
        return

    # 演示模式（手动运行 python -m core.simulation_runner 可快速验证）
    runner = OASISRunner(ipc_dir=args.ipc_dir)
    runner.start()
    cmd_ids = runner.submit_intentions(
        [
            {"agent_id": "Retail_1", "symbol": "000001", "side": "buy", "price": 10.5, "quantity": 100},
            {"agent_id": "KOL_1", "symbol": "000001", "side": "sell", "price": 10.6, "quantity": 80},
        ]
    )
    print(json.dumps(runner.flush_responses(cmd_ids, timeout_s=2.0), ensure_ascii=False, indent=2))
    runner.stop()


if __name__ == "__main__":
    main()
