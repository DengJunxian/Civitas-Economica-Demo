import pytest

from core.society.network import InformationDiffusion, SentimentState, SocialGraph


def _pick_target_and_two_neighbors(graph: SocialGraph):
    for node_id in graph.agents.keys():
        neighbors = graph.get_neighbors(node_id)
        if len(neighbors) >= 2:
            return node_id, neighbors[0], neighbors[1]
    raise RuntimeError("Graph topology did not provide enough neighbors for test.")


def test_semantic_similarity_and_hot_score_drive_infection_signal():
    graph = SocialGraph(n_agents=32, k=4, p=0.2, seed=42)
    target, src_sim, src_diff = _pick_target_and_two_neighbors(graph)

    graph.agents[src_sim].sentiment_state = SentimentState.INFECTED
    graph.agents[src_diff].sentiment_state = SentimentState.INFECTED

    graph.update_semantic_profile(
        target,
        dominant_narratives=["流动性", "政策预期"],
        focus_topics=["流动性", "政策预期", "风险偏好"],
        risk_tilt=0.1,
        historical_risk_bias=0.0,
    )
    graph.update_semantic_profile(
        src_sim,
        dominant_narratives=["流动性", "政策预期", "估值修复"],
        focus_topics=["流动性", "估值修复"],
        risk_tilt=0.2,
        historical_risk_bias=0.0,
    )
    graph.update_semantic_profile(
        src_diff,
        dominant_narratives=["地缘风险", "监管趋严", "信用收缩"],
        focus_topics=["地缘风险"],
        risk_tilt=-0.7,
        historical_risk_bias=0.0,
    )

    diffusion = InformationDiffusion(graph, beta=1.0, gamma=0.0, delta=0.0)
    sim_hi = graph.get_narrative_similarity(target, src_sim)
    sim_lo = graph.get_narrative_similarity(target, src_diff)
    hot_hi = diffusion.recsys.hot_score(graph.agents[target], graph.agents[src_sim], sim_hi)
    hot_lo = diffusion.recsys.hot_score(graph.agents[target], graph.agents[src_diff], sim_lo)

    assert sim_hi > sim_lo
    assert hot_hi > hot_lo

    signal = diffusion.compute_infection_signal(target)
    assert signal["pressure"] > 0.0
    assert signal["avg_similarity"] > 0.0
    assert signal["avg_hot_score"] > 0.0


def test_echo_chamber_boosts_repeat_source():
    graph = SocialGraph(n_agents=24, k=4, p=0.15, seed=7)
    target, src_sim, _ = _pick_target_and_two_neighbors(graph)

    graph.update_semantic_profile(
        target,
        dominant_narratives=["AI产业", "科技成长"],
        focus_topics=["AI产业", "科技成长", "高波动"],
        risk_tilt=0.7,
    )
    graph.update_semantic_profile(
        src_sim,
        dominant_narratives=["AI产业", "科技成长"],
        focus_topics=["AI产业"],
        risk_tilt=0.75,
    )

    diffusion = InformationDiffusion(graph)
    sim = graph.get_narrative_similarity(target, src_sim)

    graph.agents[target].source_exposure_count[src_sim] = 0
    hot_before = diffusion.recsys.hot_score(graph.agents[target], graph.agents[src_sim], sim)

    graph.agents[target].source_exposure_count[src_sim] = 8
    hot_after = diffusion.recsys.hot_score(graph.agents[target], graph.agents[src_sim], sim)

    assert hot_after > hot_before


def test_semantic_sirs_stats_expose_semantic_metrics():
    graph = SocialGraph(n_agents=40, k=6, p=0.2, seed=99)
    diffusion = InformationDiffusion(graph, beta=0.6, gamma=0.05, delta=0.02)
    diffusion.inject_panic(n_seeds=6, method="influential")

    stats = diffusion.update_sentiment_propagation()

    assert "avg_semantic_similarity" in stats
    assert "avg_hot_score" in stats
    assert 0.0 <= stats["avg_semantic_similarity"] <= 1.0
    assert 0.0 <= stats["avg_hot_score"] <= 1.0

