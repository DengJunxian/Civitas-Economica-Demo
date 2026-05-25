from __future__ import annotations

from pathlib import Path

from ui.reporting import export_defense_bundle


def test_export_defense_bundle(tmp_path):
    compliance_manifest = Path(tmp_path) / "ai_tool_usage_manifest.json"
    compliance_route = Path(tmp_path) / "technical_route_template.md"
    compliance_manifest.write_text("{}", encoding="utf-8")
    compliance_route.write_text("# route", encoding="utf-8")

    bundle = export_defense_bundle(
        root_dir=Path(tmp_path),
        bundle_name="contest_bundle",
        design_chapter_markdown="# 设计说明\n\ncontent",
        realism_payload={
            "title": "真实性评估报告",
            "path_fit": {"enabled": True, "score": 0.5, "price_correlation": 0.5, "volatility_correlation": 0.5, "price_rmse": 0.2, "price_mae": 0.1},
            "microstructure_fit": {"enabled": True, "score": 0.4},
            "behavioral_fit": {"enabled": True, "score": 0.45},
            "reproducibility": {"seed": 42, "config_hash": "abc"},
            "snapshot_info": {"snapshot_id": "snap1"},
            "charts": [],
        },
        policy_ab_markdown="# 政策A/B\n\ncontent",
        architecture_graph={"nodes": [], "edges": []},
        causal_chain_graph={"nodes": [], "edges": []},
        defense_outline_markdown="# 答辩提纲\n\ncontent",
        feature_flags={"stylized_facts_v2": True, "competition_mode_v1": True},
        compliance_artifacts={
            "manifest": {"ok": True},
            "files": {
                "ai_tool_usage_manifest": compliance_manifest,
                "technical_route_template": compliance_route,
            },
        },
    )
    assert bundle["bundle_root"].exists()
    assert bundle["manifest_path"].exists()
    files = bundle["manifest"]["files"]
    assert Path(files["design_chapter_draft"]).exists()
    assert Path(files["realism_json"]).exists()
    assert Path(files["policy_ab_report"]).exists()
    assert Path(files["ai_tool_usage_manifest"]).exists()
    assert Path(files["technical_route_template"]).exists()
