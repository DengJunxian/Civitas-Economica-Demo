
import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from core.market_engine import RealMarketLoader

class TestMarketLoaderFallback(unittest.TestCase):
    def setUp(self):
        # Ensure dummy default file exists
        os.makedirs("data", exist_ok=True)
        self.default_file = "data/sh000001_default.csv"
        df = pd.DataFrame({
            "date": ["2024-01-01"],
            "open": [3000],
            "high": [3100],
            "low": [2900],
            "close": [3050],
            "volume": [1000000]
        })
        df.to_csv(self.default_file, index=False)
        
        # Clean cache
        if os.path.exists("data/cache/sh000001_365.csv"):
            os.remove("data/cache/sh000001_365.csv")

    def tearDown(self):
        if os.path.exists(self.default_file):
            os.remove(self.default_file)

    @patch("core.market_engine.ak.stock_zh_index_daily")
    def test_fallback_logic(self, mock_ak):
        # Simulate API failure
        mock_ak.side_effect = Exception("API Down")
        
        print("\n[Test] Testing RealMarketLoader Fallback...")
        candles = RealMarketLoader.load_history(symbol="sh000001", period="365")
        
        self.assertIsNotNone(candles)
        self.assertTrue(len(candles) > 0)
        print("Fallback successful, loaded default data.")

if __name__ == "__main__":
    unittest.main()
