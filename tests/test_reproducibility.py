from __future__ import annotations

import importlib
import random

import numpy as np
import pandas as pd

from core.calibration.parameter_store import ParameterStore
from core.llm import LLMResponse
from core.reproducibility import (
    build_reproducibility_envelope,
    config_hash,
    dataset_snapshot_hash,
    llm_call_record_from_response,
    parameter_set_id,
    seed_everything,
    stable_json_dumps,
)


def test_seed_everything_replays_python_and_numpy_rng() -> None:
    seed_everything(2026)
    first = (random.random(), np.random.random())

    seed_everything(2026)
    second = (random.random(), np.random.random())

    assert first == second


def test_hashes_are_stable_and_redact_sensitive_config() -> None:
    safe = {"model": "deepseek-v4-pro", "temperature": 0.2, "api_key": "secret-a"}
    same_except_secret = {"temperature": 0.2, "model": "deepseek-v4-pro", "api_key": "secret-b"}

    assert config_hash(safe) == config_hash(same_except_secret)
    dumped = stable_json_dumps(safe)
    assert "secret-a" not in dumped
    assert "[REDACTED]" in dumped


def test_dataset_snapshot_hash_and_parameter_set_id_are_stable() -> None:
    frame = pd.DataFrame({"close": [1.0, 2.0], "date": ["2026-01-01", "2026-01-02"]})
    reordered = frame[["date", "close"]]

    assert dataset_snapshot_hash(frame) == dataset_snapshot_hash(reordered)
    assert parameter_set_id({"alpha": 0.2, "beta": 0.8}) == parameter_set_id({"beta": 0.8, "alpha": 0.2})


def test_parameter_store_exports_parameter_set_id(tmp_path) -> None:
    store = ParameterStore(root_dir=tmp_path)
    parameter_set = store.from_mapping(
        "unit",
        {"risk_aversion": 0.7},
        seed=42,
        config_hash="cfg",
        data_snapshot_hash="snap",
    )

    path = store.save(parameter_set)
    loaded = store.load(path)

    assert loaded is not None
    assert parameter_set.parameter_set_id.startswith("pset_")
    assert loaded.parameter_set_id == parameter_set.parameter_set_id


def test_reproducibility_envelope_contains_required_ids_without_llm_inputs() -> None:
    response = LLMResponse(
        text="sensitive completion that should not be stored",
        provider="deepseek",
        model="deepseek-v4-pro",
        latency_ms=12.5,
        fallback_chain=["deepseek:deepseek-v4-pro:thinking=true"],
        raw={"Authorization": "Bearer should_not_leak"},
        ok=True,
    )
    llm_record = llm_call_record_from_response(response)
    envelope = build_reproducibility_envelope(
        module="unit_test",
        config={"route": "slow"},
        dataset=pd.DataFrame({"x": [1, 2, 3]}),
        parameters={"alpha": 1.0},
        seed=7,
        llm_calls=[llm_record],
    )
    payload = envelope.to_dict()

    assert payload["experiment_id"].startswith("exp_")
    assert payload["config_hash"]
    assert payload["data_snapshot_hash"]
    assert payload["parameter_set_id"].startswith("pset_")
    assert payload["random_seed"] == 7
    assert payload["llm_calls"][0]["provider"] == "deepseek"
    assert payload["llm_calls"][0]["model"] == "deepseek-v4-pro"
    assert "sensitive completion" not in stable_json_dumps(payload)
    assert "should_not_leak" not in stable_json_dumps(payload)


def test_streamlit_frontend_imports_without_api_keys(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)

    app = importlib.import_module("app")

    assert hasattr(app, "main")
