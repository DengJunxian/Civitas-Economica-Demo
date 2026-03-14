import uuid

from simulation_ipc import FileSystemIPC
from simulation_runner import BufferedIntent, SimulationRunner


def test_filesystem_ipc_command_response_roundtrip(tmp_path):
    ipc = FileSystemIPC(tmp_path)

    command = ipc.send_command(
        "submit_intent",
        {"intent_id": "demo", "agent_id": "agent-1", "side": "buy", "quantity": 100, "price": 100.0},
    )
    claimed = ipc.claim_next_command()

    assert claimed is not None
    envelope, processing_path = claimed
    assert envelope.message_id == command.message_id
    assert envelope.message_type == "submit_intent"

    ipc.send_response(
        correlation_id=envelope.message_id,
        message_type="submit_intent_ack",
        payload={"accepted": True},
    )
    ipc.acknowledge_command(processing_path)

    response = ipc.wait_for_response(command.message_id, timeout=0.5)
    assert response is not None
    assert response.payload["accepted"] is True


def test_simulation_runner_buffers_intents_until_time_advance():
    runner = SimulationRunner(response_timeout=2.0, prev_close=100.0, symbol="TEST")
    runner.start()
    try:
        sell_intent = BufferedIntent(
            intent_id=str(uuid.uuid4()),
            agent_id="seller",
            side="sell",
            quantity=100,
            price=100.0,
            symbol="TEST",
            activate_step=1,
        )
        buy_intent = BufferedIntent(
            intent_id=str(uuid.uuid4()),
            agent_id="buyer",
            side="buy",
            quantity=100,
            price=100.0,
            symbol="TEST",
            activate_step=2,
        )

        sell_ack = runner.submit_intent(sell_intent)
        buy_ack = runner.submit_intent(buy_intent)

        assert sell_ack["accepted"] is True
        assert buy_ack["buffer_size"] == 2

        first_step = runner.advance_time(1)
        assert first_step["current_step"] == 1
        assert first_step["trade_count"] == 0
        assert first_step["buffer_size"] == 1
        assert first_step["best_ask"] == 100.0

        second_step = runner.advance_time(1)
        assert second_step["current_step"] == 2
        assert second_step["trade_count"] == 1
        assert second_step["buffer_size"] == 0
        assert second_step["last_price"] == 100.0
        assert second_step["best_bid"] is None
        assert second_step["best_ask"] is None
    finally:
        runner.stop()
