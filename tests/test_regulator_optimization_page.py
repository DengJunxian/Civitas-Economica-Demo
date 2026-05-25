from __future__ import annotations

from regulator_agent import run_regulatory_closed_loop
from ui.regulator_optimization import _build_regulator_result_frames


def test_regulator_optimization_page_frame_builder_outputs_tables():
    result = run_regulatory_closed_loop(
        episodes=12,
        max_steps_per_episode=6,
        seed=101,
        top_k=3,
        use_toy_env=True,
    )
    frames = _build_regulator_result_frames(result)
    assert set(frames.keys()) == {"baseline", "candidates", "deltas", "pareto", "recommendation", "evidence"}
    assert not frames["baseline"].empty
    assert not frames["pareto"].empty
    assert "macro_stability" in frames["pareto"].columns
    assert "intervention_cost" in frames["pareto"].columns
    assert not frames["recommendation"].empty
    assert not frames["evidence"].empty
