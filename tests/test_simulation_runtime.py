import tempfile

from core.simulation_ipc import FileSystemIPC
from core.simulation_runner import OASISRunner
from core.time_engine import AgentRuntimeState, CognitiveMode, InferenceTier, TimeEngine


def test_time_engine_mode_and_route():
    engine = TimeEngine(panic_threshold=0.7, cloud_threshold=0.9)
    assert engine.classify_mode(0.2, 0.1) == CognitiveMode.SYSTEM1_FAST
    assert engine.classify_mode(0.8, 0.1) == CognitiveMode.SYSTEM2_SLOW
    assert engine.route_inference_tier(0.95, 0.2) == InferenceTier.CLOUD_API
    assert engine.route_inference_tier(0.5, 0.2) == InferenceTier.LOCAL_VLLM


def test_time_engine_scheduling_precision_split():
    engine = TimeEngine(fast_merge_dt=1.0, slow_slice_dt=0.1)
    fast = AgentRuntimeState("A", 0.2, 0.2, CognitiveMode.SYSTEM1_FAST)
    slow = AgentRuntimeState("B", 0.9, 0.9, CognitiveMode.SYSTEM2_SLOW)
    e_fast = engine.schedule_agent_tick(fast, now=0.0)
    e_slow = engine.schedule_agent_tick(slow, now=0.0)
    assert e_fast.event_time == 1.0
    assert e_slow.event_time == 0.1


def test_oasis_runner_with_filesystem_ipc_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        runner = OASISRunner(ipc_dir=tmp, worker_poll_interval=0.01)
        runner.start()
        ids = runner.submit_intentions([
            {"agent_id": "Retail_1", "symbol": "000001", "side": "buy", "price": 10.5, "quantity": 100}
        ])
        responses = runner.flush_responses(ids, timeout_s=2.0)
        runner.stop()

        assert responses[0]["status"] == "ok"
        assert responses[0]["payload"]["filled_qty"] == 100


def test_filesystem_ipc_basic_command_response():
    with tempfile.TemporaryDirectory() as tmp:
        ipc = FileSystemIPC(base_dir=tmp)
        cid = ipc.push_command("trade_intent", {"quantity": 1})
        cmd = ipc.pop_next_command()
        assert cmd is not None
        assert cmd.command_id == cid
        ipc.ack_command(cid, "ok", {"done": True})
        resp = ipc.wait_response(cid, timeout_s=1.0)
        assert resp is not None
        assert resp.status == "ok"
