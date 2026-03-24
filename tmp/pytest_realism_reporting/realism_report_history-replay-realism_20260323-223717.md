# History Replay Realism

- Report No: CIVITAS-REAL-20260323-223717
- Generated At: 2026-03-23T22:37:17
- Seed: 99
- Config Hash: ea889afd3d204eb4a54a541b6325d53a4a6142c0f920378e57da1b8ea256ca87
- Feature Flag: True

## Snapshot
```json
{
  "real_points": 8,
  "real_price_max": 104.9,
  "real_price_min": 100.0,
  "real_volume_sum": 8650.0,
  "sim_price_max": 104.5,
  "sim_price_min": 100.0,
  "sim_volume_sum": 8650.0,
  "simulated_points": 8,
  "timestamp_count": 8,
  "window_end": "2024-02-08",
  "window_start": "2024-02-01"
}
```

## Path Fit
| Metric | Value |
| --- | --- |
| score | 0.992883 |
| price_correlation | 0.998966 |
| volatility_correlation | 0.972932 |
| price_rmse | 0.002739 |
| price_mae | 0.002500 |
| drawdown_shape | {"real": {"average_drawdown": 0.004360793316272815, "drawdown_duration": 1.0, "max_drawdown": 0.004807692307692291, "recovery_speed": 0.5, "shape_score": 0.9772975312358874}, "simulated": {"average_drawdown": 0.004373508923510239, "drawdown_duration": 1.0, "max_drawdown": 0.004821600771456103, "recovery_speed": 0.5, "shape_score": 0.977286066898646}} |

## Microstructure Fit
| Metric | Value |
| --- | --- |
| volume_volatility_correlation | {"real": 0.2338932949784659, "simulated": 0.3465855705861067} |
| order_sign_autocorrelation | -0.400000 |
| price_impact_curve | {"bucketed_curve": [{"bucket": 1.0, "future_return_mean": 0.013148353786433581, "signed_volume_mean": -12.0}, {"bucket": 2.0, "future_return_mean": 0.010940594059405954, "signed_volume_mean": 10.5}, {"bucket": 3.0, "future_return_mean": 0.0024077473089603885, "signed_volume_mean": 12.0}, {"bucket": 4.0, "future_return_mean": -0.004807692307692308, "signed_volume_mean": 14.0}], "correlation": -0.6072693652987343, "slope": -0.00047600175845961996} |
| turning_point_f1 | 1.000000 |
| score | 0.542415 |

## Behavioral Fit
| Metric | Value |
| --- | --- |
| return_autocorrelation | {"real": -0.5730540190713918, "simulated": -0.5929513582130299} |
| abs_return_autocorrelation | {"real": -0.6518793574770432, "simulated": -0.6823634625601316} |
| volatility_clustering | {"real": -0.6796437620146856, "simulated": -0.6549478321167984} |
| tail_heaviness_kurtosis | {"real": -1.1528449098121802, "simulated": -1.195696069806602} |
| csad_herding | {"gamma1": 0.11666666666666634, "gamma2": -8.881784197001252e-15, "herding_detected": false, "mean_csad": 0.0010938145211166455, "strength": 0.0} |
| direction_accuracy | 1.000000 |
| score | 0.498384 |

## Charts
- price_overlay: line
- return_autocorrelation: paired
- price_impact_curve: structured

## Reproducibility
```json
{
  "config_hash": "ea889afd3d204eb4a54a541b6325d53a4a6142c0f920378e57da1b8ea256ca87",
  "feature_flag": true,
  "seed": 99,
  "snapshot_info": {
    "real_points": 8,
    "real_price_max": 104.9,
    "real_price_min": 100.0,
    "real_volume_sum": 8650.0,
    "sim_price_max": 104.5,
    "sim_price_min": 100.0,
    "sim_volume_sum": 8650.0,
    "simulated_points": 8,
    "timestamp_count": 8,
    "window_end": "2024-02-08",
    "window_start": "2024-02-01"
  }
}
```