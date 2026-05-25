"""Replay scorecard builder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from core.calibration.losses import direction_hit_rate, ks_distance, max_drawdown, realized_volatility, rmse, returns
from core.exchange.trade_tape import stable_hash


@dataclass
class ReplayScorecard:
    experiment_id: str
    config_hash: str
    data_snapshot_hash: str
    seed: int
    benchmark_symbol: str
    replay_window: Dict[str, Any]
    path_fit_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    microstructure_metrics: Dict[str, float] = field(default_factory=dict)
    behavioral_metrics: Dict[str, float] = field(default_factory=dict)
    contagion_metrics: Dict[str, float] = field(default_factory=dict)
    regulatory_metrics: Dict[str, float] = field(default_factory=dict)
    pass_fail_flags: Dict[str, bool] = field(default_factory=dict)
    narrative_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _corr(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    n = min(a.size, b.size)
    if n < 2 or np.std(a[:n]) < 1e-12 or np.std(b[:n]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def build_replay_scorecard(
    *,
    sim_close: Sequence[float],
    real_close: Sequence[float],
    benchmark_symbol: str,
    replay_window: Mapping[str, Any],
    seed: int,
    config_hash: str,
    data_snapshot_hash: str,
    microstructure_metrics: Mapping[str, Any] | None = None,
    behavioral_metrics: Mapping[str, Any] | None = None,
    contagion_metrics: Mapping[str, Any] | None = None,
    regulatory_metrics: Mapping[str, Any] | None = None,
) -> ReplayScorecard:
    sim = np.asarray(list(sim_close), dtype=float)
    real = np.asarray(list(real_close), dtype=float)
    n = min(sim.size, real.size)
    sim = sim[:n]
    real = real[:n]
    sim_ret = returns(sim)
    real_ret = returns(real)
    path = {
        "normalized_rmse": rmse(sim / max(sim[0], 1e-12), real / max(real[0], 1e-12)) if n else 0.0,
        "price_correlation": _corr(sim, real),
        "return_correlation": _corr(sim_ret, real_ret),
        "direction_hit_rate": direction_hit_rate(sim, real),
        "ks_distance": ks_distance(sim_ret, real_ret),
    }
    risk = {
        "sim_volatility": realized_volatility(sim),
        "real_volatility": realized_volatility(real),
        "volatility_gap": abs(realized_volatility(sim) - realized_volatility(real)),
        "sim_max_drawdown": max_drawdown(sim),
        "real_max_drawdown": max_drawdown(real),
        "max_drawdown_gap": abs(max_drawdown(sim) - max_drawdown(real)),
    }
    flags = {
        "direction_beats_random": path["direction_hit_rate"] >= 0.50,
        "volatility_not_catastrophic": risk["sim_volatility"] <= max(risk["real_volatility"] * 4.0, 0.08),
        "drawdown_not_catastrophic": risk["sim_max_drawdown"] <= max(risk["real_max_drawdown"] * 3.0, 0.35),
        "has_hashes": bool(config_hash and data_snapshot_hash),
    }
    experiment_id = "replay_" + stable_hash(
        {
            "benchmark_symbol": benchmark_symbol,
            "window": dict(replay_window),
            "seed": int(seed),
            "config_hash": config_hash,
            "data_snapshot_hash": data_snapshot_hash,
        }
    )[:16]
    narrative = (
        f"{benchmark_symbol} replay {dict(replay_window).get('start', '')} to {dict(replay_window).get('end', '')}: "
        f"direction hit {path['direction_hit_rate']:.2%}, normalized RMSE {path['normalized_rmse']:.4f}, "
        f"vol gap {risk['volatility_gap']:.4f}."
    )
    return ReplayScorecard(
        experiment_id=experiment_id,
        config_hash=str(config_hash),
        data_snapshot_hash=str(data_snapshot_hash),
        seed=int(seed),
        benchmark_symbol=str(benchmark_symbol),
        replay_window=dict(replay_window),
        path_fit_metrics=path,
        risk_metrics=risk,
        microstructure_metrics={str(k): float(v) for k, v in dict(microstructure_metrics or {}).items() if isinstance(v, (int, float))},
        behavioral_metrics={str(k): float(v) for k, v in dict(behavioral_metrics or {}).items() if isinstance(v, (int, float))},
        contagion_metrics={str(k): float(v) for k, v in dict(contagion_metrics or {}).items() if isinstance(v, (int, float))},
        regulatory_metrics={str(k): float(v) for k, v in dict(regulatory_metrics or {}).items() if isinstance(v, (int, float))},
        pass_fail_flags=flags,
        narrative_summary=narrative,
    )


def scorecard_to_json(scorecard: ReplayScorecard) -> str:
    return json.dumps(scorecard.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


__all__ = ["ReplayScorecard", "build_replay_scorecard", "scorecard_to_json"]
