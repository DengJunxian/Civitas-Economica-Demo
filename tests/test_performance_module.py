import numpy as np

from core.performance import compute_backtest_credibility, compute_performance_metrics


def test_compute_performance_metrics_has_core_fields():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0008, 0.01, size=260)
    benchmark = rng.normal(0.0005, 0.009, size=260)

    metrics = compute_performance_metrics(returns, benchmark)

    required = {
        "total_return",
        "annual_return",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "annual_volatility",
        "alpha",
        "beta",
        "information_ratio",
        "var_95",
        "cvar_95",
        "omega_ratio",
        "tail_ratio",
        "stability",
    }
    assert required.issubset(set(metrics.keys()))
    assert metrics["n_obs"] == 260


def test_backtest_credibility_score_is_bounded():
    rng = np.random.default_rng(11)
    returns = rng.normal(0.001, 0.012, size=300)
    metrics = compute_performance_metrics(returns)

    score = compute_backtest_credibility(metrics, sample_count=300, avg_turnover=0.2)
    assert 0.0 <= score <= 1.0

