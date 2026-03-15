from core.society.network import EoNBackendAdapter, NDlibBackendAdapter, SocialGraph


def test_ndlib_adapter_fallback_or_native_step():
    graph = SocialGraph(n_agents=30, k=4, p=0.2, seed=123)
    adapter = NDlibBackendAdapter(graph, model_name="SIR", beta=0.4, gamma=0.1)
    adapter.seed_infected([0, 1, 2])

    stats = adapter.step()
    assert {"susceptible", "infected", "recovered"}.issubset(set(stats.keys()))


def test_eon_adapter_returns_time_series():
    graph = SocialGraph(n_agents=25, k=4, p=0.2, seed=12)
    adapter = EoNBackendAdapter(graph)
    result = adapter.simulate_sir(beta=0.3, gamma=0.1, tmax=20, initial_infected=[0, 3])

    assert set(result.keys()) == {"time", "S", "I", "R", "backend"}
    assert len(result["time"]) == len(result["S"]) == len(result["I"]) == len(result["R"])
    assert len(result["time"]) >= 2

