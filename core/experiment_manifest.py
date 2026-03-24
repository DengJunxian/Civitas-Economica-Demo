"""Unified experiment manifest for reproducible Civitas runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import hashlib
import json
import subprocess


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class ExperimentManifest:
    """Run-level metadata used by backtest/replay/policy/regulator/report pipelines."""

    run_id: str
    timestamp: str
    git_commit: str
    config_hash: str
    seed: int
    dataset_snapshot_id: str
    module: str
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    parent_run_id: str = ""
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @staticmethod
    def build(
        *,
        module: str,
        seed: int,
        dataset_snapshot_id: str,
        config: Optional[Mapping[str, Any]] = None,
        feature_flags: Optional[Mapping[str, Any]] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
        parent_run_id: str = "",
        notes: Optional[Mapping[str, Any]] = None,
        git_commit: Optional[str] = None,
        config_hash: Optional[str] = None,
        timestamp: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> "ExperimentManifest":
        clean_config = dict(config or {})
        clean_flags = {str(k): bool(v) for k, v in dict(feature_flags or {}).items()}
        clean_artifacts = dict(artifacts or {})
        clean_notes = dict(notes or {})
        manifest_timestamp = str(timestamp or _utc_now_iso())
        manifest_config_hash = str(
            config_hash
            or _stable_hash(
                {
                    "module": module,
                    "seed": int(seed),
                    "dataset_snapshot_id": str(dataset_snapshot_id or ""),
                    "config": clean_config,
                    "feature_flags": clean_flags,
                }
            )
        )
        manifest_run_id = str(
            run_id
            or f"{module}-{manifest_timestamp.replace(':', '').replace('-', '').replace('+00:00', 'Z')}-{manifest_config_hash[:12]}"
        )
        return ExperimentManifest(
            run_id=manifest_run_id,
            timestamp=manifest_timestamp,
            git_commit=str(git_commit or _safe_git_commit()),
            config_hash=manifest_config_hash,
            seed=int(seed),
            dataset_snapshot_id=str(dataset_snapshot_id or ""),
            module=str(module),
            feature_flags=clean_flags,
            config=clean_config,
            artifacts=clean_artifacts,
            parent_run_id=str(parent_run_id or ""),
            notes=clean_notes,
        )


def write_experiment_manifest(manifest: ExperimentManifest, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{manifest.run_id}.manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentManifest(**payload)


def attach_manifest_to_metadata(
    metadata: Optional[Mapping[str, Any]],
    *,
    module: str,
    seed: int,
    dataset_snapshot_id: str,
    feature_flags: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    meta = dict(metadata or {})
    manifest = ExperimentManifest.build(
        module=module,
        seed=int(seed),
        dataset_snapshot_id=str(dataset_snapshot_id),
        feature_flags=feature_flags,
        config=config,
        artifacts=artifacts,
    )
    meta["experiment_manifest"] = manifest.to_dict()
    meta["experiment_manifest_id"] = manifest.run_id
    meta["experiment_config_hash"] = manifest.config_hash
    meta["seed"] = int(seed)
    meta["dataset_snapshot_id"] = str(dataset_snapshot_id)
    return meta

