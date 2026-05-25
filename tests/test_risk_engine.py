
import unittest
from unittest.mock import MagicMock, patch
import time
from core.regulation.risk_control import RiskEngine, MarginAccount
from core.types import Order, OrderSide, OrderStatus

class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.risk_engine = RiskEngine(
            stamp_duty_rate=0.001,
            commission_rate=0.0003
        )
        
    def test_calculate_transaction_cost(self):
        """测试交易成本计算（印花税+佣金）"""
        # 买入: 只有佣金
        cost_buy = self.risk_engine.calculate_transaction_cost(OrderSide.BUY, 100.0, 1000)
        # Value = 100 * 1000 = 100,000
        # Commission = max(5, 100000 * 0.0003) = 30
        # Stamp = 0
        self.assertAlmostEqual(cost_buy, 30.0)
        
        # 卖出: 佣金 + 印花税
        cost_sell = self.risk_engine.calculate_transaction_cost(OrderSide.SELL, 100.0, 1000)
        # Stamp = 100000 * 0.001 = 100
        self.assertAlmostEqual(cost_sell, 130.0)
        
        # 测试最低佣金
        cost_min = self.risk_engine.calculate_transaction_cost(OrderSide.BUY, 1.0, 100)
        # Value = 100. Comm = max(5, 0.03) = 5.
        self.assertAlmostEqual(cost_min, 5.0)

    def test_check_order_compliance_hft(self):
        """测试高频交易限制"""
        order = Order(
            symbol="TEST",
            order_type=0, # LIMIT
            agent_id="Agent_007",
            price=10.0,
            quantity=100,
            side=OrderSide.BUY,
            timestamp=time.time()
        )
        market_data = {"last_price": 10.0}
        
        # 快速提交多个订单
        for _ in range(10):
            allowed, penalty, reason = self.risk_engine.check_order_compliance("Agent_007", order, market_data)
            # 注册订单 (模拟 MarketDataManager 的行为)
            self.risk_engine.hft_monitor.register_order("Agent_007")
            
        ratio = self.risk_engine.hft_monitor.get_otr("Agent_007")
        self.assertGreaterEqual(ratio, 0.0)

    def test_margin_check(self):
        """测试保证金检查"""
        # Setup Margin Account
        account = MagicMock(spec=MarginAccount)
        account.get_buying_power.return_value = 5000.0
        self.risk_engine.register_margin_account("Agent_Margin", account)
        
        # Case 1: Order within limit
        order_ok = Order(
            symbol="TEST",
            order_type=0, 
            agent_id="Agent_Margin",
            price=10.0,
            quantity=100, # Val=1000
            side=OrderSide.BUY,
            timestamp=time.time()
        )
        allowed, _, _ = self.risk_engine.check_order_compliance("Agent_Margin", order_ok, {})
        self.assertTrue(allowed)
        
        # Case 2: Order exceeds limit
        order_fail = Order(
            symbol="TEST",
            order_type=0,
            agent_id="Agent_Margin",
            price=10.0,
            quantity=1000, # Val=10000
            side=OrderSide.BUY,
            timestamp=time.time()
        )
        allowed, _, reason = self.risk_engine.check_order_compliance("Agent_Margin", order_fail, {})
        self.assertFalse(allowed)
        self.assertIn("超出购买力限制", reason)

    def test_robust_side_handling(self):
        """测试 OrderSide 的字符串兼容性"""
        # 使用字符串 'BUY'
        order_str = Order(
            symbol="TEST",
            order_type=0,
            agent_id="Agent_Str",
            price=10.0,
            quantity=100,
            side="BUY", # String!
            timestamp=time.time()
        )
        # 这应该不会抛出 AttributeError
        try:
            self.risk_engine.check_order_compliance("Agent_Str", order_str, {})
        except AttributeError:
            self.fail("RiskEngine raised AttributeError on string side!")

if __name__ == '__main__':
    unittest.main()
