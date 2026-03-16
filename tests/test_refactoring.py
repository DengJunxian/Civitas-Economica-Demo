
import unittest
import os
from unittest.mock import patch
from core.time_manager import SimulationClock
from core.market_engine import MarketDataManager, Order
from core.utils import PriceQuantizer
from core.types import OrderSide, OrderType

class TestRefactoring(unittest.TestCase):
    def setUp(self):
        self.clock = SimulationClock()
        # Mock config if needed, or rely on defaults
        # Clear cache for testing
        if os.path.exists("data/cache/sh000001_365.csv"):
            os.remove("data/cache/sh000001_365.csv")
            
    def test_clock_integration(self):
        print("\n[Test] Clock Integration")
        initial_time = self.clock.timestamp
        self.clock.tick()
        self.assertGreater(self.clock.timestamp, initial_time)
        print(f"Clock ticked: {initial_time} -> {self.clock.timestamp}")
        
    def test_market_data_manager_caching(self):
        print("\n[Test] Market Data Caching")
        MarketDataManager(api_key_or_router="dummy", load_real_data=True, clock=self.clock)

        self.assertTrue(os.path.exists("data/cache/sh000001_365.csv"))
        with patch("core.market_engine.RealMarketLoader._provider.get_ohlcv", side_effect=AssertionError("cache miss")):
            # 第二次加载应直接命中本地缓存，不应再触发 provider 请求。
            MarketDataManager(api_key_or_router="dummy", load_real_data=True, clock=self.clock)
        
    def test_order_timestamps(self):
        print("\n[Test] Order Timestamps")
        mdm = MarketDataManager(api_key_or_router="dummy", load_real_data=False, clock=self.clock)
        mdm.engine.prev_close = 3000.0
        
        # Inject order using clock
        # Manually create order with current clock time
        order = Order(
            agent_id="test_agent",
            side=OrderSide.BUY,
            price=3000.0,
            quantity=100,
            timestamp=self.clock.timestamp,
            order_type=OrderType.LIMIT,
            symbol="sh000001"
        )
        
        # Submit
        mdm.submit_agent_order(order)
        # Verify order timestamp preserved
        pass 
        
    def test_price_quantizer(self):
        print("\n[Test] Price Quantizer")
        prev = 3000.0
        limit = 0.10
        lower, upper = PriceQuantizer.get_limit_prices(prev, limit)
        self.assertEqual(upper, 3300.00)
        self.assertEqual(lower, 2700.00)
        
        # Test rounding
        p = 3000.1234
        self.assertEqual(PriceQuantizer.quantize(p), 3000.12)

if __name__ == "__main__":
    unittest.main()
