import json
from pathlib import Path

from core.behavioral_finance import StylizedFactsEvaluator
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
