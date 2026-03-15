import numpy as np
import pandas as pd

from core.portfolio import PortfolioConstructionLayer, PortfolioConstraints, PortfolioInput


def _mock_input() -> PortfolioInput:
    assets = ["A", "B", "C"]
    mu = pd.Series([0.12, 0.08, 0.06], index=assets)
    cov = pd.DataFrame(
        [
            [0.08, 0.02, 0.01],
            [0.02, 0.05, 0.015],
            [0.01, 0.015, 0.03],
        ],
        index=assets,
        columns=assets,
    )
    current = pd.Series([0.4, 0.4, 0.2], index=assets)
    sentiment = pd.Series([0.8, 0.1, 0.2], index=assets)
    policy = pd.Series([0.5, 0.1, 0.1], index=assets)
    return PortfolioInput(
        expected_returns=mu,
        cov_matrix=cov,
        current_weights=current,
        sentiment_risk=sentiment,
        policy_risk=policy,
        asset_to_industry={"A": "tech", "B": "finance", "C": "finance"},
    )


def test_inverse_vol_weights_sum_to_one():
    data = _mock_input()
    layer = PortfolioConstructionLayer(method="inverse_vol")
    weights = layer.optimize(data)
    assert set(weights.index) == {"A", "B", "C"}
    assert np.isclose(float(weights.sum()), 1.0, atol=1e-6)
    assert (weights >= 0).all()


def test_turnover_and_industry_caps_apply():
    data = _mock_input()
    constraints = PortfolioConstraints(
        long_only=True,
        fully_invested=True,
        max_turnover=0.10,
        industry_caps={"finance": 0.6},
    )
    layer = PortfolioConstructionLayer(method="equal_weight", constraints=constraints)
    weights = layer.optimize(data)

    turnover = float(np.abs(weights - data.current_weights).sum())
    finance_weight = float(weights.loc[["B", "C"]].sum())
    assert turnover <= 0.100001
    assert finance_weight <= 0.600001


def test_sentiment_policy_penalty_reduces_risky_weight():
    data = _mock_input()
    base_layer = PortfolioConstructionLayer(method="mean_variance")
    penalized_layer = PortfolioConstructionLayer(
        method="mean_variance",
        constraints=PortfolioConstraints(sentiment_penalty=0.2, policy_penalty=0.2),
    )

    base_w = base_layer.optimize(data)
    penalized_w = penalized_layer.optimize(data)
    # Asset A has highest sentiment/policy risk, should be down-weighted.
    assert penalized_w["A"] <= base_w["A"] + 1e-8

