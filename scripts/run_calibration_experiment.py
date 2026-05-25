from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtester import BacktestConfig
from core.calibration_pipeline import CalibrationPipeline, CalibrationSpec


def _mock_price_frame(n: int = 160, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.011, size=n)
    close = 3000.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 - rng.normal(0.0, 0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.002, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.002, size=n)))
    volume = np.clip(rng.normal(950_000, 90_000, size=n), 120_000, None)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal reproducible calibration experiment.")
    parser.add_argument("--output-dir", default="outputs/calibration_demo", help="Root directory for calibration artifacts.")
    parser.add_argument("--seed", type=int, default=21, help="Random seed for both data generation and calibration.")
    parser.add_argument("--rolling-window", action="store_true", help="Enable minimal rolling-window evaluation.")
    args = parser.parse_args()

    frame = _mock_price_frame(seed=args.seed)
    benchmark = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    cfg = BacktestConfig(
        symbol="sh000001",
        benchmark_symbol="sh000001",
        period_days=0,
        random_seed=args.seed,
        strategy_name="portfolio_system",
    )
    spec = CalibrationSpec(
        parameter_space={
            "policy_shock": (-0.2, 0.2),
            "rebalance_frequency": [3, 5, 10],
            "max_position": [0.8, 1.0],
            "signal_threshold": {"min": 0.02, "max": 0.12, "type": "float"},
        },
        method="bayesian",
        bayes_init=3,
        bayes_iterations=5,
        bayes_patience=4,
        random_seed=args.seed,
        dataset_snapshot_id="demo-slice-001",
        rolling_window=bool(args.rolling_window),
        output_dir=str(Path(args.output_dir)),
    )
    pipeline = CalibrationPipeline(
        base_config=cfg,
        spec=spec,
        historical_data=frame,
        benchmark_data=benchmark,
    )
    result = pipeline.run()
    summary = {
        "run_id": result.manifest.get("run_id", ""),
        "best_score": result.best_score,
        "best_config": result.best_config,
        "artifacts": result.artifacts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
