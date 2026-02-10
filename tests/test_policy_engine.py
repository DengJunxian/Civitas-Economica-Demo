import pytest
import time
from core.policy import CircuitBreaker, TransactionTax, PolicyManager, PolicyResult
from core.types import Order, Trade, OrderSide, OrderType
from core.mesa.civitas_model import CivitasModel


class TestCircuitBreaker:
    def test_no_halt_within_threshold(self):
        """价格在阈值内，市场应保持开放"""
        cb = CircuitBreaker(threshold_pct=0.10)
        cb.update_reference_price(3000.0)
        
        # 5% move - within 10% threshold
        assert cb.check_market_status(3150.0, time.time()) == True
        assert cb.is_halted == False
    
    def test_halt_on_breach(self):
        """价格超过阈值，触发熔断"""
        cb = CircuitBreaker(threshold_pct=0.10, halt_duration_sec=60)
        cb.update_reference_price(3000.0)
        
        now = time.time()
        # 15% move - exceeds 10% threshold
        result = cb.check_market_status(3450.0, now)
        assert result == False
        assert cb.is_halted == True
    
    def test_resume_after_duration(self):
        """熔断后经过冷却期恢复交易"""
        cb = CircuitBreaker(threshold_pct=0.10, halt_duration_sec=10)
        cb.update_reference_price(3000.0)
        
        now = time.time()
        cb.check_market_status(3450.0, now)  # Trigger halt
        assert cb.is_halted == True
        
        # After 15 seconds, should resume
        result = cb.check_market_status(3100.0, now + 15)
        assert result == True
        assert cb.is_halted == False
    
    def test_order_rejected_during_halt(self):
        """熔断期间拒绝订单"""
        cb = CircuitBreaker(threshold_pct=0.05)
        cb.update_reference_price(3000.0)
        cb.is_halted = True
        cb.halt_start_time = time.time()
        
        order = Order(
            price=3100.0, quantity=100, agent_id="test",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            symbol="TEST", timestamp=time.time()
        )
        result = cb.check_order(order, {"last_price": 3100.0})
        assert result.is_allowed == False
        assert "Halted" in result.reason


class TestTransactionTax:
    def test_tax_calculation(self):
        """验证交易税计算"""
        tax = TransactionTax(rate=0.001)
        
        trade = Trade(
            trade_id="test-trade-1",
            price=3000.0, quantity=100,
            maker_id="maker_order", taker_id="taker_order",
            maker_agent_id="maker", taker_agent_id="taker",
            buyer_agent_id="buyer", seller_agent_id="seller",
            timestamp=time.time()
        )
        
        expected_tax = 3000.0 * 100 * 0.001  # 300.0
        assert tax.calculate_tax(trade) == pytest.approx(expected_tax)
    
    def test_tax_disabled(self):
        """禁用税收时返回0"""
        tax = TransactionTax(rate=0.001)
        tax.active = False
        
        trade = Trade(
            trade_id="test-trade-2",
            price=3000.0, quantity=100,
            maker_id="maker_order", taker_id="taker_order",
            maker_agent_id="maker", taker_agent_id="taker",
            buyer_agent_id="buyer", seller_agent_id="seller",
            timestamp=time.time()
        )
        assert tax.calculate_tax(trade) == 0.0


class TestPolicyManager:
    def test_default_policies(self):
        """验证默认策略已加载"""
        pm = PolicyManager()
        assert "circuit_breaker" in pm.policies
        assert "tax" in pm.policies
    
    def test_set_policy_param(self):
        """动态修改策略参数"""
        pm = PolicyManager()
        pm.set_policy_param("tax", "rate", 0.005)
        assert pm.policies["tax"].rate == 0.005
    
    def test_order_passes_when_market_open(self):
        """市场正常时订单通过"""
        pm = PolicyManager()
        pm.policies["circuit_breaker"].update_reference_price(3000.0)
        
        order = Order(
            price=3050.0, quantity=100, agent_id="test",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            symbol="TEST", timestamp=time.time()
        )
        result = pm.check_order(order, {"last_price": 3050.0})
        assert result.is_allowed == True


class TestCivitasModelPolicy:
    def test_model_policy_api(self):
        """验证 CivitasModel 策略 API"""
        model = CivitasModel(n_agents=10)
        
        # Get default status
        status = model.get_policy_status()
        assert status["circuit_breaker"]["active"] == True
        assert status["transaction_tax"]["active"] == True
        assert status["transaction_tax"]["rate"] == 0.001
        
        # Change tax rate
        model.set_policy("tax", "rate", 0.005)
        status = model.get_policy_status()
        assert status["transaction_tax"]["rate"] == 0.005
        
        # Disable circuit breaker
        model.set_policy("circuit_breaker", "active", False)
        status = model.get_policy_status()
        assert status["circuit_breaker"]["active"] == False
