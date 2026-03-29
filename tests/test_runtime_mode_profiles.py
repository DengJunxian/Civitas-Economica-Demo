from __future__ import annotations

from core.runtime_mode import merge_mode_feature_flags, resolve_runtime_mode_profile


def test_smart_mode_profile_defaults_to_competition_safe() -> None:
    profile = resolve_runtime_mode_profile("SMART")
    assert profile.mode == "SMART"
    assert profile.competition_safe_mode is True
    assert profile.market_pipeline_v2 is True
    assert profile.llm_primary is False
    assert profile.use_live_api is False


def test_deep_mode_profile_enables_api_and_committee() -> None:
    profile = resolve_runtime_mode_profile("DEEP")
    assert profile.mode == "DEEP"
    assert profile.llm_primary is True
    assert profile.use_live_api is True
    assert profile.enable_policy_committee is True
    assert profile.pause_for_llm_seconds > 0.0

    flags = merge_mode_feature_flags("DEEP", {"market_pipeline_v2": False})
    assert flags["runtime_llm_primary"] is True
    assert flags["runtime_use_live_api"] is True
    assert flags["runtime_policy_committee_v1"] is True
    assert flags["market_pipeline_v2"] is True
