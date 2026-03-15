from core.market_engine import MarketDataManager


def test_market_manager_ingest_text_factors_updates_snapshot():
    manager = MarketDataManager(api_key_or_router="dummy", load_real_data=False, clock=None)

    factors = {
        "dominant_topic": "regulation",
        "sentiment_score": -0.6,
        "financial_factors": {
            "panic_index": 0.8,
            "greed_index": 0.1,
            "policy_shock": 0.7,
            "regime_bias": "risk_off",
        },
        "impact_paths": [{"source": "regulation", "relation": "impacts_sector", "target": "bank", "weight": 0.8}],
    }

    base_panic = manager.panic_level
    manager.ingest_text_factors(factors, headline="监管收紧引发担忧")
    snap = manager.get_market_snapshot()

    assert manager.panic_level >= base_panic
    assert snap.text_dominant_topic == "regulation"
    assert snap.text_regime_bias == "risk_off"
    assert snap.text_policy_shock > 0.5
    assert isinstance(snap.text_impact_paths, list)

