from __future__ import annotations

from core.experiment_manifest import (
    ExperimentManifest,
    attach_manifest_to_metadata,
    load_experiment_manifest,
    write_experiment_manifest,
)


def test_experiment_manifest_build_write_load(tmp_path):
    manifest = ExperimentManifest.build(
        module="unit_test",
        seed=123,
        dataset_snapshot_id="snap-001",
        model_version="portfolio_system@v2",
        data_slice={"date_start": "2024-01-01", "date_end": "2024-03-31"},
        config={"alpha": 1, "beta": 2},
        feature_flags={"x": True},
    )
    path = write_experiment_manifest(manifest, tmp_path)
    assert path.exists()

    loaded = load_experiment_manifest(path)
    assert loaded.run_id == manifest.run_id
    assert loaded.seed == 123
    assert loaded.dataset_snapshot_id == "snap-001"
    assert loaded.code_version == loaded.git_commit
    assert loaded.model_version == "portfolio_system@v2"
    assert loaded.data_slice["date_start"] == "2024-01-01"


def test_attach_manifest_to_metadata():
    metadata = attach_manifest_to_metadata(
        {"a": 1},
        module="attach_test",
        seed=9,
        dataset_snapshot_id="snap-x",
        feature_flags={"f": True},
        config={"k": "v"},
        model_version="demo-model",
        data_slice={"fold": "holdout"},
    )
    assert "experiment_manifest" in metadata
    assert metadata["dataset_snapshot_id"] == "snap-x"
    assert metadata["seed"] == 9
    assert metadata["model_version"] == "demo-model"
    assert metadata["data_slice"]["fold"] == "holdout"
