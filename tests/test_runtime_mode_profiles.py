from __future__ import annotations

from core.model_router import ModelRouter
from core.runtime_mode import merge_mode_feature_flags, resolve_runtime_mode_profile


def test_smart_mode_profile_defaults_to_competition_safe() -> None:
    profile = resolve_runtime_mode_profile("SMART")
    assert profile.mode == "SMART"
    assert profile.label == "智能模式"
    assert profile.competition_safe_mode is True
    assert profile.market_pipeline_v2 is True
    assert profile.llm_primary is True
    assert profile.use_live_api is True
    assert profile.model_priority
    assert all(model.startswith("glm-") for model in profile.model_priority)


def test_deep_mode_profile_enables_api_and_committee() -> None:
    profile = resolve_runtime_mode_profile("DEEP")
    assert profile.mode == "DEEP"
    assert profile.label == "高级模式"
    assert profile.llm_primary is True
    assert profile.use_live_api is True
    assert profile.enable_policy_committee is True
    assert profile.pause_for_llm_seconds > 0.0
    assert any(model.startswith("deepseek-") for model in profile.model_priority)
    assert any(model.startswith("glm-") for model in profile.model_priority)

    flags = merge_mode_feature_flags("DEEP", {"market_pipeline_v2": False})
    assert flags["runtime_llm_primary"] is True
    assert flags["runtime_use_live_api"] is True
    assert flags["runtime_policy_committee_v1"] is True
    assert flags["market_pipeline_v2"] is True


def test_advanced_alias_resolves_to_deep_profile() -> None:
    profile = resolve_runtime_mode_profile("ADVANCED")

    assert profile.mode == "DEEP"
    assert profile.label == "高级模式"


def test_model_router_smart_priority_is_zhipu_only() -> None:
    router = ModelRouter(deepseek_key="deepseek-test", zhipu_key="zhipu-test")

    smart_priority = router.get_model_priority("SMART")
    advanced_priority = router.get_model_priority("ADVANCED")

    assert smart_priority
    assert all(model.startswith("glm-") for model in smart_priority)
    assert any(model.startswith("deepseek-") for model in advanced_priority)
    assert any(model.startswith("glm-") for model in advanced_priority)
