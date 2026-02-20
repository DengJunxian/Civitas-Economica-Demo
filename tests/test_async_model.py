import asyncio
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GLOBAL_CONFIG
from core.scheduler import SimulationController

async def test_controller():
    print("[Test] Initializing SimulationController...")
    # Mock keys for testing
    controller = SimulationController(
        deepseek_key="test_key",
        zhipu_key="test_key"
    )
    
    print("[Test] Running 5 ticks...")
    for i in range(5):
        try:
            result = await controller.run_tick()
            print(f"  Tick {i+1}: Price={result['candle'].close:.2f}, "
                  f"Trades={result['trades']}, "
                  f"Sentiment={result['smart_sentiment']:.2f}")
        except Exception as e:
            print(f"[Fail] Tick {i+1} failed: {e}")
            raise
    
    print("[Test] Simulation finished.")

if __name__ == "__main__":
    try:
        asyncio.run(test_controller())
    except Exception as e:
        print(f"[Fail] Exception occurred: {e}")
        # raise # Don't raise to avoid cluttering output context if just checking logs
