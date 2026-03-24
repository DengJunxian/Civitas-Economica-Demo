"""Calibration pipeline with GP-Bayesian and evolutionary search backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import math
import random

import numpy as np
import pandas as pd

from core.backtester import BacktestConfig, BacktestResult, FactorBacktestEngine
from core.behavioral_finance import StylizedFactsEvaluator
from core.experiment_manifest import ExperimentManifest, write_experiment_manifest


@dataclass
class CalibrationSpec:
    """Search specification for automatic parameter calibration."""

    parameter_space: Dict[str, Any]
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "path_fit": 0.35,
            "volatility_fit": 0.20,
            "turnover_fit": 0.15,
            "stylized_facts_fit": 0.30,
        }
    )
    method: str = "bayesian"
    bayes_init: int = 24
    bayes_iterations: int = 80
    bayes_patience: int = 12
    gp_candidate_pool: int = 256
    gp_length_scale: float = 0.35
    gp_exploration: float = 0.01
    evo_population: int = 24
    evo_elite: int = 6
    evo_generations: int = 40
    evo_sigma0: float = 0.18
    evo_restarts: int = 2
    split_train: float = 0.50
    split_validation: float = 0.25
    split_holdout: float = 0.25
    rolling_window: bool = False
    random_seed: int = 42
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    dataset_snapshot_id: str = ""
    output_dir: str = "outputs/calibration"

    def normalized_weights(self) -> Dict[str, float]:
        base = dict(self.objective_weights)
        total = float(sum(max(0.0, float(v)) for v in base.values()))
        if total <= 1e-12:
            return {"path_fit": 0.25, "volatility_fit": 0.25, "turnover_fit": 0.25, "stylized_facts_fit": 0.25}
        return {k: float(max(0.0, float(v)) / total) for k, v in base.items()}


@dataclass
class CalibrationResult:
    best_config: Dict[str, Any]
    best_score: float
    method: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity: List[Dict[str, Any]] = field(default_factory=list)
    holdout_report: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)


class CalibrationPipeline:
    """Reproducible calibration pipeline with two search methods."""

    def __init__(
        self,
        *,
        base_config: BacktestConfig,
        spec: CalibrationSpec,
        historical_data: Optional[pd.DataFrame] = None,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.base_config = base_config
        self.spec = spec
        self.historical_data = historical_data.copy() if historical_data is not None else pd.DataFrame()
        self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else pd.DataFrame()
        self.rng = random.Random(int(spec.random_seed))
        self.np_rng = np.random.default_rng(int(spec.random_seed))
        self._space_meta = self._build_space_meta()

    def _prepare_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.historical_data.empty:
            loader = FactorBacktestEngine(self.base_config)
            ok = loader.load_data()
            if not ok:
                return pd.DataFrame(), pd.DataFrame()
            self.historical_data = loader.historical_data.copy()
            self.benchmark_data = loader.benchmark_data.copy()
        if self.benchmark_data.empty and not self.historical_data.empty:
            bm = self.historical_data[["date", "close"]].copy()
            bm.rename(columns={"close": "benchmark_close"}, inplace=True)
            self.benchmark_data = bm
        return self.historical_data.copy(), self.benchmark_data.copy()

    def _split_frame(self, frame: pd.DataFrame, benchmark: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        if frame.empty:
            return {"train": (pd.DataFrame(), pd.DataFrame()), "validation": (pd.DataFrame(), pd.DataFrame()), "holdout": (pd.DataFrame(), pd.DataFrame())}

        ordered = frame.sort_values("date").reset_index(drop=True)
        total = len(ordered)
        train_n = max(1, int(total * float(self.spec.split_train)))
        val_n = max(1, int(total * float(self.spec.split_validation)))
        holdout_n = max(1, total - train_n - val_n)
        if train_n + val_n + holdout_n > total:
            holdout_n = max(1, total - train_n - val_n)
        if holdout_n <= 0:
            holdout_n = 1
            val_n = max(1, total - train_n - holdout_n)
        if train_n + val_n + holdout_n > total:
            train_n = max(1, total - val_n - holdout_n)

        train_df = ordered.iloc[:train_n].reset_index(drop=True)
        val_df = ordered.iloc[train_n : train_n + val_n].reset_index(drop=True)
        hold_df = ordered.iloc[train_n + val_n : train_n + val_n + holdout_n].reset_index(drop=True)

        def _slice_benchmark(left: pd.DataFrame) -> pd.DataFrame:
            if benchmark.empty or left.empty:
                return pd.DataFrame()
            dates = set(left["date"].astype(str).tolist())
            return benchmark[benchmark["date"].astype(str).isin(dates)].reset_index(drop=True)

        return {
            "train": (train_df, _slice_benchmark(train_df)),
            "validation": (val_df, _slice_benchmark(val_df)),
            "holdout": (hold_df, _slice_benchmark(hold_df)),
        }

    def _clone_config(self, params: Mapping[str, Any]) -> BacktestConfig:
        payload = asdict(self.base_config)
        payload.update(dict(params))
        payload["feature_flags"] = {**dict(self.base_config.feature_flags or {}), **dict(self.spec.feature_flags or {})}
        payload["random_seed"] = int(self.spec.random_seed)
        return BacktestConfig(**payload)

    def _run_backtest(self, cfg: BacktestConfig, frame: pd.DataFrame, benchmark: pd.DataFrame) -> BacktestResult:
        engine = FactorBacktestEngine(cfg)
        engine.historical_data = frame.copy()
        engine.benchmark_data = benchmark.copy() if not benchmark.empty else pd.DataFrame()
        return engine.run_backtest()

    @staticmethod
    def _turnover_target_proxy(frame: pd.DataFrame) -> float:
        if frame.empty or "volume" not in frame.columns:
            return 0.15
        series = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        if series.empty:
            return 0.15
        vol_ratio = float(series.std() / max(series.mean(), 1e-12))
        return float(np.clip(0.08 + 0.15 * vol_ratio, 0.05, 0.40))

    def _stylized_score(self, result: BacktestResult, frame: pd.DataFrame) -> float:
        evaluator = StylizedFactsEvaluator(
            feature_flag=True,
            seed=int(self.spec.random_seed),
            config={
                "method": self.spec.method,
                "feature_flags": self.spec.feature_flags,
            },
        )
        report = evaluator.evaluate(
            real_prices=result.real_prices,
            simulated_prices=result.simulated_prices,
            real_volumes=frame.get("volume", pd.Series(dtype=float)).tolist() if not frame.empty else [],
            simulated_volumes=[x.get("volume", 0.0) for x in (result.simulated_bars or [])],
            timestamps=result.dates,
            legacy_metrics={
                "price_correlation": result.price_correlation,
                "volatility_correlation": result.volatility_correlation,
                "price_rmse": result.price_rmse,
                "price_mae": result.price_mae,
                "credibility_score": result.credibility_score,
            },
            snapshot_info={
                "dataset_snapshot_id": self.spec.dataset_snapshot_id,
            },
        )
        micro = float((report.microstructure_fit or {}).get("score", 0.0))
        behavior = float((report.behavioral_fit or {}).get("score", 0.0))
        path = float((report.path_fit or {}).get("score", 0.0))
        return float(np.clip(np.mean([path, micro, behavior]), 0.0, 1.0))

    def _evaluate_params(
        self,
        params: Mapping[str, Any],
        split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> Dict[str, Any]:
        cfg = self._clone_config(params)
        train_df, train_bm = split["train"]
        val_df, val_bm = split["validation"]
        hold_df, hold_bm = split["holdout"]

        train_result = self._run_backtest(cfg, train_df, train_bm)
        val_result = self._run_backtest(cfg, val_df, val_bm) if not val_df.empty else train_result

        path_fit = float(np.clip((val_result.price_correlation + 1.0) / 2.0, 0.0, 1.0))
        volatility_fit = float(np.clip(val_result.volatility_correlation, 0.0, 1.0))
        turnover_target = self._turnover_target_proxy(train_df)
        turnover_fit = float(np.clip(1.0 - abs(val_result.agent_turnover_rate - turnover_target), 0.0, 1.0))
        stylized_fit = self._stylized_score(val_result, val_df if not val_df.empty else train_df)

        weights = self.spec.normalized_weights()
        score = (
            weights["path_fit"] * path_fit
            + weights["volatility_fit"] * volatility_fit
            + weights["turnover_fit"] * turnover_fit
            + weights["stylized_facts_fit"] * stylized_fit
        )

        holdout_result = self._run_backtest(cfg, hold_df, hold_bm) if not hold_df.empty else val_result
        holdout_score = float(
            weights["path_fit"] * np.clip((holdout_result.price_correlation + 1.0) / 2.0, 0.0, 1.0)
            + weights["volatility_fit"] * np.clip(holdout_result.volatility_correlation, 0.0, 1.0)
            + weights["turnover_fit"] * np.clip(1.0 - abs(holdout_result.agent_turnover_rate - turnover_target), 0.0, 1.0)
            + weights["stylized_facts_fit"] * self._stylized_score(holdout_result, hold_df if not hold_df.empty else train_df)
        )
        return {
            "params": dict(params),
            "score": float(score),
            "path_fit": path_fit,
            "volatility_fit": volatility_fit,
            "turnover_fit": turnover_fit,
            "stylized_facts_fit": stylized_fit,
            "holdout_score": holdout_score,
            "holdout_return": float(holdout_result.total_return),
            "holdout_sharpe": float(holdout_result.sharpe_ratio),
        }

    def _sample_value(self, space: Any) -> Any:
        if isinstance(space, list):
            return self.rng.choice(space)
        if isinstance(space, tuple) and len(space) == 2:
            low, high = float(space[0]), float(space[1])
            return self.rng.uniform(low, high)
        if isinstance(space, dict):
            if "values" in space:
                return self.rng.choice(list(space["values"]))
            if "min" in space and "max" in space:
                low, high = float(space["min"]), float(space["max"])
                if str(space.get("type", "float")).lower() in {"int", "integer"}:
                    return int(self.rng.randint(int(low), int(high)))
                return self.rng.uniform(low, high)
        return space

    def _sample_params(self) -> Dict[str, Any]:
        return {name: self._sample_value(space) for name, space in self.spec.parameter_space.items()}

    def _mutate_params(self, base: Mapping[str, Any], scale: float = 0.15) -> Dict[str, Any]:
        out = dict(base)
        for name, space in self.spec.parameter_space.items():
            if isinstance(space, list):
                if self.rng.random() < 0.25:
                    out[name] = self.rng.choice(space)
                continue
            if isinstance(space, tuple) and len(space) == 2:
                low, high = float(space[0]), float(space[1])
                span = high - low
                center = float(out.get(name, (low + high) / 2.0))
                value = center + self.np_rng.normal(0.0, span * scale)
                out[name] = float(np.clip(value, low, high))
                continue
            if isinstance(space, dict) and "min" in space and "max" in space:
                low, high = float(space["min"]), float(space["max"])
                span = high - low
                center = float(out.get(name, (low + high) / 2.0))
                value = center + self.np_rng.normal(0.0, span * scale)
                if str(space.get("type", "float")).lower() in {"int", "integer"}:
                    out[name] = int(np.clip(round(value), low, high))
                else:
                    out[name] = float(np.clip(value, low, high))
        return out

    def _build_space_meta(self) -> List[Dict[str, Any]]:
        meta: List[Dict[str, Any]] = []
        for name, space in self.spec.parameter_space.items():
            if isinstance(space, list):
                choices = list(space) if list(space) else [None]
                meta.append({"name": name, "kind": "categorical", "choices": choices})
                continue
            if isinstance(space, tuple) and len(space) == 2:
                low, high = float(space[0]), float(space[1])
                if high < low:
                    low, high = high, low
                meta.append({"name": name, "kind": "float", "low": low, "high": high})
                continue
            if isinstance(space, dict):
                if "values" in space:
                    values = list(space["values"])
                    choices = values if values else [None]
                    meta.append({"name": name, "kind": "categorical", "choices": choices})
                    continue
                if "min" in space and "max" in space:
                    low, high = float(space["min"]), float(space["max"])
                    if high < low:
                        low, high = high, low
                    is_int = str(space.get("type", "float")).lower() in {"int", "integer"}
                    meta.append({"name": name, "kind": "int" if is_int else "float", "low": low, "high": high})
                    continue
            meta.append({"name": name, "kind": "fixed", "value": space})
        return meta

    @staticmethod
    def _params_signature(params: Mapping[str, Any]) -> str:
        return json.dumps(dict(params), sort_keys=True, default=str)

    def _vectorize_params(self, params: Mapping[str, Any]) -> np.ndarray:
        if not self._space_meta:
            return np.zeros((0,), dtype=float)
        values: List[float] = []
        for item in self._space_meta:
            name = str(item["name"])
            kind = str(item["kind"])
            value = params.get(name)
            if kind == "categorical":
                choices = list(item["choices"])
                try:
                    idx = choices.index(value)
                except ValueError:
                    idx = 0
                denom = max(1, len(choices) - 1)
                values.append(float(idx) / float(denom))
            elif kind in {"float", "int"}:
                low, high = float(item["low"]), float(item["high"])
                if high - low <= 1e-12:
                    values.append(0.0)
                else:
                    values.append(float(np.clip((float(value) - low) / (high - low), 0.0, 1.0)))
            else:
                values.append(0.0)
        return np.array(values, dtype=float)

    def _decode_vector(self, vec: Sequence[float]) -> Dict[str, Any]:
        if not self._space_meta:
            return {}
        arr = np.asarray(vec, dtype=float)
        out: Dict[str, Any] = {}
        for idx, item in enumerate(self._space_meta):
            name = str(item["name"])
            kind = str(item["kind"])
            value = float(np.clip(arr[idx] if idx < len(arr) else 0.0, 0.0, 1.0))
            if kind == "categorical":
                choices = list(item["choices"])
                denom = max(1, len(choices) - 1)
                choice_idx = int(np.clip(round(value * denom), 0, len(choices) - 1))
                out[name] = choices[choice_idx]
            elif kind == "int":
                low, high = float(item["low"]), float(item["high"])
                if high - low <= 1e-12:
                    out[name] = int(round(low))
                else:
                    mapped = low + value * (high - low)
                    out[name] = int(np.clip(round(mapped), low, high))
            elif kind == "float":
                low, high = float(item["low"]), float(item["high"])
                if high - low <= 1e-12:
                    out[name] = float(low)
                else:
                    out[name] = float(low + value * (high - low))
            else:
                out[name] = item["value"]
        return out

    @staticmethod
    def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, *, length_scale: float) -> np.ndarray:
        if x1.size == 0 or x2.size == 0:
            return np.zeros((x1.shape[0], x2.shape[0]), dtype=float)
        ls = float(max(1e-6, length_scale))
        x1_sq = np.sum(x1 * x1, axis=1, keepdims=True)
        x2_sq = np.sum(x2 * x2, axis=1, keepdims=True).T
        sq_dist = np.clip(x1_sq + x2_sq - 2.0 * (x1 @ x2.T), 0.0, None)
        return np.exp(-0.5 * sq_dist / (ls * ls))

    @staticmethod
    def _normal_cdf(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + np.vectorize(math.erf)(arr / np.sqrt(2.0)))

    @staticmethod
    def _normal_pdf(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return np.exp(-0.5 * arr * arr) / np.sqrt(2.0 * np.pi)

    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray, *, best: float, xi: float) -> np.ndarray:
        mu_arr = np.asarray(mu, dtype=float)
        sigma_arr = np.asarray(sigma, dtype=float)
        safe_sigma = np.clip(sigma_arr, 1e-12, None)
        improvement = mu_arr - float(best) - float(xi)
        z = improvement / safe_sigma
        ei = improvement * self._normal_cdf(z) + safe_sigma * self._normal_pdf(z)
        ei[sigma_arr <= 1e-12] = 0.0
        return ei

    def _gp_predict(self, x_train: np.ndarray, y_train: np.ndarray, x_candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = int(x_train.shape[0])
        n_candidates = int(x_candidates.shape[0])
        if n_samples < 2 or n_candidates <= 0:
            mean = float(np.mean(y_train)) if y_train.size else 0.0
            return np.full((n_candidates,), mean, dtype=float), np.full((n_candidates,), 1.0, dtype=float)

        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std <= 1e-12:
            y_std = 1.0
        y_norm = (y_train - y_mean) / y_std

        dim = max(1, int(x_train.shape[1]))
        length_scale = float(max(1e-6, self.spec.gp_length_scale)) * np.sqrt(float(dim))
        k_train = self._rbf_kernel(x_train, x_train, length_scale=length_scale)
        jitter = 1e-8
        for _ in range(6):
            try:
                chol = np.linalg.cholesky(k_train + np.eye(n_samples, dtype=float) * jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            mean = np.full((n_candidates,), y_mean, dtype=float)
            std = np.full((n_candidates,), y_std, dtype=float)
            return mean, std

        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
        k_star = self._rbf_kernel(x_train, x_candidates, length_scale=length_scale)
        mu_norm = k_star.T @ alpha
        v = np.linalg.solve(chol, k_star)
        var_norm = np.clip(1.0 - np.sum(v * v, axis=0), 1e-12, None)

        mu = mu_norm * y_std + y_mean
        sigma = np.sqrt(var_norm) * y_std
        return mu, np.clip(sigma, 1e-12, None)

    def _propose_gp_candidate(self, records: Sequence[Mapping[str, Any]], best_params: Mapping[str, Any]) -> Dict[str, Any]:
        if not records or not self._space_meta:
            return self._sample_params()
        if len(records) < 2:
            return self._mutate_params(best_params, scale=0.10) if best_params else self._sample_params()

        x_train = np.vstack([self._vectorize_params(row["params"]) for row in records]).astype(float)
        y_train = np.array([float(row["score"]) for row in records], dtype=float)
        if x_train.shape[0] < 2 or np.std(y_train) <= 1e-12:
            return self._mutate_params(best_params, scale=0.10) if best_params else self._sample_params()

        dim = x_train.shape[1]
        pool = max(64, int(self.spec.gp_candidate_pool))
        random_pool = self.np_rng.random((pool, dim), dtype=float)
        best_vec = self._vectorize_params(best_params) if best_params else x_train[int(np.argmax(y_train))]
        local_count = max(24, pool // 3)
        local_pool = np.clip(best_vec + self.np_rng.normal(0.0, 0.10, size=(local_count, dim)), 0.0, 1.0)
        candidates = np.vstack([random_pool, local_pool])

        mu, sigma = self._gp_predict(x_train, y_train, candidates)
        ei = self._expected_improvement(
            mu,
            sigma,
            best=float(np.max(y_train)),
            xi=float(max(0.0, self.spec.gp_exploration)),
        )
        if ei.size <= 0:
            return self._sample_params()
        best_idx = int(np.argmax(ei))
        return self._decode_vector(candidates[best_idx])

    def _bayesian_like_search(self, split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        best_score = -1e18
        best_params: Dict[str, Any] = {}
        no_improve = 0

        for _ in range(max(1, int(self.spec.bayes_init))):
            params = self._sample_params()
            evaluated = self._evaluate_params(params, split)
            records.append(evaluated)
            if evaluated["score"] > best_score:
                best_score = float(evaluated["score"])
                best_params = dict(evaluated["params"])
                no_improve = 0
            else:
                no_improve += 1

        for _ in range(max(1, int(self.spec.bayes_iterations))):
            if no_improve >= max(1, int(self.spec.bayes_patience)):
                break
            exploration = self.rng.random() < 0.25 or not best_params
            params = self._sample_params() if exploration else self._mutate_params(best_params, scale=0.12)
            evaluated = self._evaluate_params(params, split)
            records.append(evaluated)
            if evaluated["score"] > best_score:
                best_score = float(evaluated["score"])
                best_params = dict(evaluated["params"])
                no_improve = 0
            else:
                no_improve += 1
        return records

    def _bayesian_search(self, split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        best_score = -1e18
        best_params: Dict[str, Any] = {}
        no_improve = 0
        seen: set[str] = set()

        init_rounds = max(2, int(self.spec.bayes_init))
        iter_rounds = max(1, int(self.spec.bayes_iterations))
        patience = max(1, int(self.spec.bayes_patience))

        for _ in range(init_rounds):
            params = self._sample_params()
            sig = self._params_signature(params)
            if sig in seen:
                continue
            seen.add(sig)
            evaluated = self._evaluate_params(params, split)
            records.append(evaluated)
            if evaluated["score"] > best_score:
                best_score = float(evaluated["score"])
                best_params = dict(evaluated["params"])
                no_improve = 0
            else:
                no_improve += 1

        for _ in range(iter_rounds):
            if no_improve >= patience:
                break
            if self.rng.random() < 0.10 or len(records) < 2:
                params = self._sample_params()
            else:
                params = self._propose_gp_candidate(records, best_params)

            for _retry in range(8):
                sig = self._params_signature(params)
                if sig not in seen:
                    break
                params = self._mutate_params(best_params, scale=0.08) if best_params else self._sample_params()
            sig = self._params_signature(params)
            if sig in seen:
                continue
            seen.add(sig)

            evaluated = self._evaluate_params(params, split)
            records.append(evaluated)
            if evaluated["score"] > best_score:
                best_score = float(evaluated["score"])
                best_params = dict(evaluated["params"])
                no_improve = 0
            else:
                no_improve += 1
        return records

    def _evolutionary_search(self, split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        pop_size = max(4, int(self.spec.evo_population))
        elite_n = max(1, min(pop_size, int(self.spec.evo_elite)))
        sigma = float(max(1e-6, self.spec.evo_sigma0))
        restarts = max(1, int(self.spec.evo_restarts))

        for _restart in range(restarts):
            population = [self._sample_params() for _ in range(pop_size)]
            for _gen in range(max(1, int(self.spec.evo_generations))):
                scored = [self._evaluate_params(params, split) for params in population]
                records.extend(scored)
                scored_sorted = sorted(scored, key=lambda x: float(x["score"]), reverse=True)
                elites = [dict(item["params"]) for item in scored_sorted[:elite_n]]
                next_pop = list(elites)
                while len(next_pop) < pop_size:
                    parent = self.rng.choice(elites)
                    child = self._mutate_params(parent, scale=sigma)
                    next_pop.append(child)
                population = next_pop
                sigma = max(0.02, sigma * 0.98)
        return records

    def _build_sensitivity(self, best_params: Mapping[str, Any], split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for name, space in self.spec.parameter_space.items():
            base_value = best_params.get(name)
            candidates: List[Any] = []
            if isinstance(space, list):
                candidates = [x for x in space if x != base_value][:4]
            elif isinstance(space, tuple) and len(space) == 2 and base_value is not None:
                low, high = float(space[0]), float(space[1])
                delta = (high - low) * 0.10
                candidates = [float(np.clip(float(base_value) - delta, low, high)), float(np.clip(float(base_value) + delta, low, high))]
            elif isinstance(space, dict) and "min" in space and "max" in space and base_value is not None:
                low, high = float(space["min"]), float(space["max"])
                delta = (high - low) * 0.10
                if str(space.get("type", "float")).lower() in {"int", "integer"}:
                    candidates = [int(np.clip(int(base_value) - max(1, int(round(delta))), low, high)), int(np.clip(int(base_value) + max(1, int(round(delta))), low, high))]
                else:
                    candidates = [float(np.clip(float(base_value) - delta, low, high)), float(np.clip(float(base_value) + delta, low, high))]

            for value in candidates:
                params = dict(best_params)
                params[name] = value
                evaluated = self._evaluate_params(params, split)
                rows.append(
                    {
                        "parameter": name,
                        "candidate": value,
                        "score": float(evaluated["score"]),
                        "holdout_score": float(evaluated["holdout_score"]),
                    }
                )
        return rows

    def _write_outputs(
        self,
        *,
        result: CalibrationResult,
        split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ) -> CalibrationResult:
        root = Path(self.spec.output_dir)
        root.mkdir(parents=True, exist_ok=True)

        best_path = root / "best_config.json"
        sensitivity_path = root / "sensitivity.csv"
        holdout_path = root / "holdout_report.json"
        manifest_path = root / "calibration_manifest.json"

        best_payload = {
            "best_config": result.best_config,
            "best_score": float(result.best_score),
            "method": result.method,
            "seed": int(self.spec.random_seed),
            "dataset_snapshot_id": str(self.spec.dataset_snapshot_id),
        }
        best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        pd.DataFrame(result.sensitivity).to_csv(sensitivity_path, index=False, encoding="utf-8-sig")
        holdout_path.write_text(json.dumps(result.holdout_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        manifest = ExperimentManifest.build(
            module="calibration_pipeline",
            seed=int(self.spec.random_seed),
            dataset_snapshot_id=str(self.spec.dataset_snapshot_id),
            feature_flags=self.spec.feature_flags,
            config={
                "base_config": asdict(self.base_config),
                "spec": asdict(self.spec),
            },
            artifacts={
                "best_config": str(best_path),
                "sensitivity": str(sensitivity_path),
                "holdout_report": str(holdout_path),
            },
            notes={
                "split_sizes": {
                    "train": int(len(split["train"][0])),
                    "validation": int(len(split["validation"][0])),
                    "holdout": int(len(split["holdout"][0])),
                }
            },
        )
        write_experiment_manifest(manifest, root)
        manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

        result.artifacts = {
            "best_config": str(best_path),
            "sensitivity": str(sensitivity_path),
            "holdout_report": str(holdout_path),
            "calibration_manifest": str(manifest_path),
        }
        result.manifest = manifest.to_dict()
        return result

    def run(self) -> CalibrationResult:
        frame, benchmark = self._prepare_frames()
        split = self._split_frame(frame, benchmark)
        if split["train"][0].empty:
            empty = CalibrationResult(best_config={}, best_score=0.0, method=self.spec.method)
            return self._write_outputs(result=empty, split=split)

        method = str(self.spec.method or "bayesian").strip().lower()
        if method in {"evo", "evolution", "cma-es", "cmaes"}:
            records = self._evolutionary_search(split)
            method_name = "evolutionary"
        elif method in {"bayesian_like", "bayes_like"}:
            records = self._bayesian_like_search(split)
            method_name = "bayesian_like"
        else:
            use_gp_bo = bool((self.spec.feature_flags or {}).get("calibration_gp_bo_v1", True))
            if use_gp_bo:
                records = self._bayesian_search(split)
                method_name = "bayesian_gp"
            else:
                records = self._bayesian_like_search(split)
                method_name = "bayesian_like"

        if not records:
            empty = CalibrationResult(best_config={}, best_score=0.0, method=method_name)
            return self._write_outputs(result=empty, split=split)

        records_sorted = sorted(records, key=lambda x: float(x["score"]), reverse=True)
        best = records_sorted[0]
        best_config = dict(best["params"])
        sensitivity = self._build_sensitivity(best_config, split)

        holdout_report = {
            "score": float(best.get("holdout_score", 0.0)),
            "return": float(best.get("holdout_return", 0.0)),
            "sharpe": float(best.get("holdout_sharpe", 0.0)),
            "path_fit": float(best.get("path_fit", 0.0)),
            "volatility_fit": float(best.get("volatility_fit", 0.0)),
            "turnover_fit": float(best.get("turnover_fit", 0.0)),
            "stylized_facts_fit": float(best.get("stylized_facts_fit", 0.0)),
        }

        result = CalibrationResult(
            best_config=best_config,
            best_score=float(best["score"]),
            method=method_name,
            records=records_sorted,
            sensitivity=sensitivity,
            holdout_report=holdout_report,
        )
        return self._write_outputs(result=result, split=split)


__all__ = ["CalibrationPipeline", "CalibrationResult", "CalibrationSpec"]
