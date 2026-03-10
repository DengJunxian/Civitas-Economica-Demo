# file: core/simulation_ipc.py
"""
基于文件系统的异步 Command-Response IPC。

目标：
1) 将慢速 LLM 推理进程与极速 LOB 撮合进程解耦。
2) 通过命令队列 + 响应队列的双向通道，实现物理隔离与异步削峰。
3) 即使推理侧暂时阻塞，撮合侧依旧可独立消费“交易意图缓存”。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class IPCCommand:
    """发送给撮合子进程的命令。"""

    command_id: str
    command_type: str
    payload: Dict[str, Any]
    created_at: float


@dataclass
class IPCResponse:
    """撮合子进程返回的响应。"""

    command_id: str
    status: str
    payload: Dict[str, Any]
    created_at: float


class FileSystemIPC:
    """
    文件系统 IPC 通道。

    目录结构：
    - <base>/commands/pending
    - <base>/commands/processing
    - <base>/responses

    这样做的原因：
    - pending -> processing 的原子 rename 可减少并发抢占冲突。
    - commands / responses 解耦，天然支持 Command-Response 异步双向机制。
    """

    def __init__(self, base_dir: str = "data/ipc"):
        self.base_dir = Path(base_dir)
        self.pending_dir = self.base_dir / "commands" / "pending"
        self.processing_dir = self.base_dir / "commands" / "processing"
        self.responses_dir = self.base_dir / "responses"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def push_command(self, command_type: str, payload: Dict[str, Any]) -> str:
        """向命令队列写入一条命令，返回 command_id。"""
        command_id = str(uuid.uuid4())
        cmd = IPCCommand(
            command_id=command_id,
            command_type=command_type,
            payload=payload,
            created_at=time.time(),
        )
        path = self.pending_dir / f"{command_id}.json"
        path.write_text(json.dumps(asdict(cmd), ensure_ascii=False), encoding="utf-8")
        return command_id

    def pop_next_command(self) -> Optional[IPCCommand]:
        """
        由撮合进程调用：尝试获取下一条待处理命令。

        关键细节：
        - 采用 rename 抢占，避免多个 worker 同时拿到同一条命令。
        """
        for path in sorted(self.pending_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            target = self.processing_dir / path.name
            try:
                os.rename(path, target)
            except FileNotFoundError:
                continue
            except OSError:
                continue

            try:
                raw = target.read_text(encoding="utf-8")
                data = json.loads(raw)
                return IPCCommand(**data)
            except Exception:
                # 非法文件直接丢弃，防止阻塞队列。
                target.unlink(missing_ok=True)
                continue
        return None

    def ack_command(self, command_id: str, status: str, payload: Dict[str, Any]) -> None:
        """写入响应，并清理 processing 中对应命令。"""
        resp = IPCResponse(
            command_id=command_id,
            status=status,
            payload=payload,
            created_at=time.time(),
        )
        response_path = self.responses_dir / f"{command_id}.json"
        response_path.write_text(json.dumps(asdict(resp), ensure_ascii=False), encoding="utf-8")

        processing_path = self.processing_dir / f"{command_id}.json"
        processing_path.unlink(missing_ok=True)

    def wait_response(self, command_id: str, timeout_s: float = 3.0, poll_interval: float = 0.05) -> Optional[IPCResponse]:
        """调用侧等待指定命令响应（可超时）。"""
        deadline = time.time() + timeout_s
        response_path = self.responses_dir / f"{command_id}.json"
        while time.time() < deadline:
            if response_path.exists():
                try:
                    data = json.loads(response_path.read_text(encoding="utf-8"))
                    return IPCResponse(**data)
                finally:
                    response_path.unlink(missing_ok=True)
            time.sleep(poll_interval)
        return None

    def collect_stale_processing(self, stale_after_s: float = 10.0) -> List[str]:
        """将长时间未完成的 processing 命令回收至 pending，防止僵死。"""
        recycled: List[str] = []
        now = time.time()
        for path in self.processing_dir.glob("*.json"):
            if now - path.stat().st_mtime > stale_after_s:
                target = self.pending_dir / path.name
                try:
                    os.rename(path, target)
                    recycled.append(path.stem)
                except OSError:
                    continue
        return recycled
