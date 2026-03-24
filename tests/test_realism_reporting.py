import json
from pathlib import Path

from core.backtester import BacktestResult
from core.behavioral_finance import StylizedFactsEvaluator
from ui.history_replay import _build_authenticity_layers, _build_history_report
from ui.reporting import stable_payload_hash, write_realism_report_artifacts


def _build_report_payload():
    real_prices = [100.0, 101.0, 102.2, 101.8, 103.1, 104.0, 103.5, 104.9]
    simulated_prices = [100.0, 100.8, 101.9, 101.5, 102.9, 103.7, 103.2, 104.5]
    real_volumes = [1000, 1025, 990, 1115, 1095, 1135, 1120, 1170]
    order_signs = [1, 1, 1, -1, 1, 1, -1, 1]
    trade_sizes = [10, 11, 12, 13, 12, 14, 11, 15]
    market_returns = [(real_prices[i + 1] - real_prices[i]) / real_prices[i] for i in range(len(real_prices) - 1)]
    cross_sectional_returns = [[r * 0.8, r * 1.1, r * 1.05] for r in market_returns]
    evaluator = StylizedFactsEvaluator(feature_flag=True, seed=99, config={"label": "report"})
    report = evaluator.evaluate(
        real_prices=real_prices,
        simulated_prices=simulated_prices,
        real_volumes=real_volumes,
        simulated_volumes=real_volumes,
        order_signs=order_signs,
        trade_sizes=trade_sizes,
        market_returns=market_returns,
        cross_sectional_returns=cross_sectional_returns,
        timestamps=[f"2024-02-0{i+1}" for i in range(len(real_prices))],
    )
    return report.to_dict()


def test_write_realism_report_artifacts_creates_markdown_json_and_charts() -> None:
    payload = _build_report_payload()
    first_hash = stable_payload_hash(payload)
    second_hash = stable_payload_hash(payload)

    assert first_hash == second_hash

    output_dir = Path.cwd() / "tmp" / "pytest_realism_reporting"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = write_realism_report_artifacts(
        root_dir=output_dir,
        title="History Replay Realism",
        payload=payload,
        feature_flag=True,
    )

    assert bundle["markdown_path"].exists()
    assert bundle["json_path"].exists()
    assert bundle["charts_path"].exists()
    assert bundle["stem"].startswith("realism_report_")

    saved = json.loads(bundle["json_path"].read_text(encoding="utf-8"))
    charts = json.loads(bundle["charts_path"].read_text(encoding="utf-8"))

    assert saved["feature_flag"] is True
    assert saved["feature_flags"]["stylized_facts_v2"] is True
    assert saved["reproducibility"]["seed"] == 99
    assert saved["reproducibility"]["config_hash"] == payload["config_hash"]
    assert "Path Fit" in bundle["markdown_text"]
    assert "Behavioral Fit" in bundle["markdown_text"]
    assert isinstance(charts, list) and charts
    assert saved["snapshot_info"]["real_points"] == payload["snapshot_info"]["real_points"]


def test_history_replay_report_exports_three_authenticity_layers() -> None:
    result = BacktestResult(
        strategy_name="history_replay",
        total_days=5,
        total_trades=6,
        total_volume=120,
        total_return=0.03,
        max_drawdown=-0.02,
        excess_return=0.01,
        price_correlation=0.9,
        volatility_correlation=0.8,
        price_rmse=0.04,
        simulated_prices=[100.0, 100.8, 101.5, 101.1, 102.0],
        real_prices=[100.0, 101.0, 101.6, 101.0, 102.2],
        dates=["2024-02-01", "2024-02-02", "2024-02-05", "2024-02-06", "2024-02-07"],
        simulated_bars=[
            {"date": "2024-02-01", "open": 100.0, "high": 100.9, "low": 99.8, "close": 100.8, "volume": 20},
            {"date": "2024-02-02", "open": 100.8, "high": 101.6, "low": 100.6, "close": 101.5, "volume": 25},
            {"date": "2024-02-05", "open": 101.5, "high": 101.7, "low": 100.9, "close": 101.1, "volume": 18},
            {"date": "2024-02-06", "open": 101.1, "high": 102.1, "low": 101.0, "close": 102.0, "volume": 30},
        ],
        trade_log=[
            {"side": "buy", "event": "opening_auction", "timestamp": "2024-02-01T09:25:00"},
            {"side": "sell", "event": "midday_liquidity", "timestamp": "2024-02-02T13:15:00"},
            {"side": "buy", "event": "closing_rebalance", "timestamp": "2024-02-06T14:55:00"},
        ],
        metadata={
            "event_schedule": {"total_events": 8},
            "reference_bars": [
                {"date": "2024-02-01", "open": 100.0, "high": 101.0, "low": 99.7, "close": 101.0, "volume": 1000},
                {"date": "2024-02-02", "open": 101.0, "high": 101.8, "low": 100.8, "close": 101.6, "volume": 1025},
                {"date": "2024-02-05", "open": 101.6, "high": 101.9, "low": 100.7, "close": 101.0, "volume": 980},
                {"date": "2024-02-06", "open": 101.0, "high": 102.4, "low": 100.9, "close": 102.2, "volume": 1110},
            ],
        },
    )
    layers = _build_authenticity_layers(result)
    assert set(layers.keys()) == {"path", "microstructure", "stylized_facts"}

    bundle = _build_history_report(
        {
            "policy_name": "Case replay",
            "policy_text": "Liquidity support replay",
            "background": "test",
            "strength": 1.0,
            "symbol_label": "CSI 300",
            "start_date": "2024-02-01",
            "end_date": "2024-02-07",
            "engine_mode": "agent",
            "feature_flags": {
                "agent_replay": True,
                "history_replay_event_driven_v2": True,
                "history_replay_rolling_calibration_v1": True,
            },
            "result": result,
            "baseline_result": None,
            "replay_cards": [],
            "authenticity_layers": layers,
        },
        metrics={
            "trend_alignment": 0.75,
            "turning_point_match": 0.5,
            "drawdown_gap": 0.02,
            "vol_similarity": 0.8,
            "response_gap": 1.0,
        },
    )

    assert bundle["json_path"].exists()
    assert bundle["csv_path"].exists()
    assert bundle["charts_path"].exists()
    saved = json.loads(bundle["json_path"].read_text(encoding="utf-8"))
    charts = json.loads(bundle["charts_path"].read_text(encoding="utf-8"))

    assert set(saved["authenticity_layers"].keys()) == {"path", "microstructure", "stylized_facts"}
    assert saved["authenticity_metrics_flat"]
    assert isinstance(charts, list) and charts
