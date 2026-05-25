import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scheduler import SimulationController


def test_controller_init_smoke() -> None:
    """控制器初始化冒烟测试（不触发真实异步推理链路）。"""
    controller = SimulationController(
        deepseek_key="test_key",
        zhipu_key="test_key",
        mode="FAST",
    )

    assert controller.mode.value == "FAST"
    assert controller.model is not None
    assert controller.market is controller.model.market_manager
    assert controller.get_time_budget() >= 0
    assert controller.get_deep_agent_count() >= 0
