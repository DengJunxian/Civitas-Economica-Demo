# file: tests/test_regulation.py
"""
监管压力测试单元测试

测试覆盖:
1. 保证金账户与强制平仓
2. 高频交易监控与惩罚
3. 风险引擎整合
4. 模型路由器
5. 分布式运行器

作者: Civitas Economica Team
"""

import pytest
import numpy as np
import pandas as pd

from core.regulation.risk_control import (
    HighFrequencyMonitor, MarginAccount, RiskEngine,
    MarginCallEvent, HFTViolation, Order, ViolationType
)
from core.system.runner import (
    ModelRouter, SimulationRunner, AgentActor,
    AgentConfig, AgentType, ModelType
)


# ==========================================
# 保证金账户测试
# ==========================================

class TestMarginAccount:
    """保证金账户测试"""
    
    @pytest.fixture
    def margin_account(self):
        return MarginAccount(
            initial_cash=100000,
            leverage_limit=2.0,
            maintenance_margin=0.25
        )
    
    def test_account_creation(self, margin_account):
        """测试账户创建"""
        assert margin_account.available_cash == 100000
        assert margin_account.leverage_limit == 2.0
        assert margin_account.borrowed == 0
    
    def test_borrow(self, margin_account):
        """测试借款"""
        # 最大借款 = 初始资金 * (杠杆 - 1) = 100000 * 1 = 100000
        success = margin_account.borrow(50000)
        assert success
        assert margin_account.borrowed == 50000
        assert margin_account.available_cash == 150000
        
        # 再借 50000 应成功 (总借款 100000)
        success = margin_account.borrow(50000)
        assert success
        assert margin_account.borrowed == 100000
        
        # 再借 10000 应失败 (超过最大借款限额)
        success = margin_account.borrow(10000)
        assert not success
    
    def test_equity_calculation(self, margin_account):
        """测试权益计算"""
        margin_account.borrow(50000)
        margin_account.buy("000001", 10.0, 10000, pd.Timestamp.now())
        
        prices = {"000001": 10.0}
        equity = margin_account.get_equity(prices)
        
        # 权益 = 现金 + 市值 - 借款
        # 150000 - 100000 (买入) + 100000 (市值) - 50000 (借款) = 100000
        assert equity == 100000
    
    def test_margin_level_safe(self, margin_account):
        """测试安全保证金水平"""
        margin_account.borrow(50000)
        margin_account.buy("000001", 10.0, 10000, pd.Timestamp.now())
        
        prices = {"000001": 10.0}
        is_safe = margin_account.check_margin_level(prices)
        assert is_safe
    
    def test_margin_call_trigger(self, margin_account):
        """
        测试保证金追缴触发
        
        场景: 借款后买入，价格下跌导致保证金不足
        """
        # 借款并买入
        margin_account.borrow(100000)
        margin_account.buy("000001", 10.0, 15000, pd.Timestamp.now())
        
        # 初始状态: 权益 = 50000 + 150000 - 100000 = 100000
        # 保证金率 = 100000 / 150000 = 66.7% > 25%
        
        prices_normal = {"000001": 10.0}
        assert margin_account.check_margin_level(prices_normal)
        
        # 价格下跌到 5.0
        # 市值 = 75000
        # 权益 = 50000 + 75000 - 100000 = 25000
        # 保证金率 = 25000 / 75000 = 33.3% > 25% (仍安全)
        prices_drop = {"000001": 5.0}
        is_safe = margin_account.check_margin_level(prices_drop)
        assert is_safe
        
        # 价格继续下跌到 3.0
        # 市值 = 45000
        # 权益 = 50000 + 45000 - 100000 = -5000 (负数!)
        # 保证金率 = -5000 / 45000 = -11% < 25% (触发平仓)
        prices_crash = {"000001": 3.0}
        is_safe = margin_account.check_margin_level(prices_crash)
        assert not is_safe, "价格暴跌应触发保证金追缴"
    
    def test_force_liquidation(self, margin_account):
        """测试强制平仓"""
        margin_account.borrow(100000)
        margin_account.buy("000001", 10.0, 15000, pd.Timestamp.now())
        
        order = margin_account.force_liquidate("000001", 3.0, "test_agent")
        
        assert order is not None
        assert order.action == "SELL"
        assert order.quantity == 15000
        assert order.reason == "MARGIN_CALL"
        assert len(margin_account.margin_call_history) == 1


# ==========================================
# 高频交易监控测试
# ==========================================

class TestHighFrequencyMonitor:
    """高频交易监控测试"""
    
    @pytest.fixture
    def hft_monitor(self):
        return HighFrequencyMonitor(
            otr_threshold=10.0,
            penalty_base_rate=0.001,
            block_threshold=50.0
        )
    
    def test_normal_trading(self, hft_monitor):
        """测试正常交易"""
        # 10 单 10 成交 = OTR 1.0
        for _ in range(10):
            hft_monitor.register_order("normal_trader")
            hft_monitor.register_trade("normal_trader")
        
        allowed, penalty, reason = hft_monitor.check_order("normal_trader", 100000)
        
        assert allowed
        assert penalty == 0.0
        assert reason is None
    
    def test_high_otr_penalty(self, hft_monitor):
        """测试高 OTR 惩罚"""
        # 100 单 5 成交 = OTR 20
        for _ in range(100):
            hft_monitor.register_order("hft_bot")
        for _ in range(5):
            hft_monitor.register_trade("hft_bot")
        
        otr = hft_monitor.get_otr("hft_bot")
        assert otr == 20.0
        
        allowed, penalty, reason = hft_monitor.check_order("hft_bot", 100000)
        
        assert allowed  # OTR 20 < 50，允许但惩罚
        assert penalty > 0
    
    def test_order_blocked(self, hft_monitor):
        """测试订单阻止"""
        # 500 单 5 成交 = OTR 100
        for _ in range(500):
            hft_monitor.register_order("extreme_hft")
        for _ in range(5):
            hft_monitor.register_trade("extreme_hft")
        
        allowed, penalty, reason = hft_monitor.check_order("extreme_hft", 100000)
        
        assert not allowed
        assert "阻止阈值" in reason
    
    def test_daily_reset(self, hft_monitor):
        """测试每日重置"""
        hft_monitor.register_order("trader")
        hft_monitor.register_trade("trader")
        
        hft_monitor.reset_daily()
        
        assert hft_monitor.agent_orders["trader"] == 0
        assert hft_monitor.agent_trades["trader"] == 0


# ==========================================
# 风险引擎测试
# ==========================================

class TestRiskEngine:
    """风险引擎整合测试"""
    
    @pytest.fixture
    def risk_engine(self):
        return RiskEngine(
            stamp_duty_rate=0.001,
            commission_rate=0.0003,
            otr_threshold=10.0
        )
    
    def test_transaction_cost_buy(self, risk_engine):
        """测试买入交易成本 (无印花税)"""
        cost = risk_engine.calculate_transaction_cost("BUY", 10.0, 1000)
        
        # 金额 10000, 佣金 = max(5, 10000 * 0.0003) = 5
        assert cost == 5.0
    
    def test_transaction_cost_sell(self, risk_engine):
        """测试卖出交易成本 (含印花税)"""
        cost = risk_engine.calculate_transaction_cost("SELL", 10.0, 10000)
        
        # 金额 100000
        # 佣金 = max(5, 100000 * 0.0003) = 30
        # 印花税 = 100000 * 0.001 = 100
        # 总计 = 130
        assert cost == 130.0
    
    def test_margin_account_registration(self, risk_engine):
        """测试保证金账户注册"""
        account = MarginAccount(100000)
        risk_engine.register_margin_account("agent_1", account)
        
        assert "agent_1" in risk_engine.margin_accounts
    
    def test_batch_margin_check(self, risk_engine):
        """测试批量保证金检查"""
        # 注册多个账户
        for i in range(3):
            account = MarginAccount(100000, leverage_limit=2.0)
            account.borrow(100000)
            account.buy("000001", 10.0, 15000, pd.Timestamp.now())
            risk_engine.register_margin_account(f"agent_{i}", account)
        
        # 正常价格 - 无平仓
        orders = risk_engine.check_all_margin_accounts({"000001": 10.0})
        assert len(orders) == 0
        
        # 暴跌价格 - 触发平仓
        orders = risk_engine.check_all_margin_accounts({"000001": 3.0})
        assert len(orders) == 3  # 全部触发


# ==========================================
# 模型路由器测试
# ==========================================

class TestModelRouter:
    """模型路由器测试"""
    
    @pytest.fixture
    def router(self):
        return ModelRouter(r1_ratio=0.05)
    
    def test_institutional_gets_r1(self, router):
        """测试机构使用 R1"""
        model = router.assign_model(AgentType.INSTITUTIONAL)
        assert model == ModelType.DEEPSEEK_R1
    
    def test_retail_gets_7b(self, router):
        """测试散户使用 7B"""
        model = router.assign_model(AgentType.RETAIL)
        assert model == ModelType.DEEPSEEK_7B
    
    def test_quant_gets_rules(self, router):
        """测试量化使用规则引擎"""
        model = router.assign_model(AgentType.QUANT)
        assert model == ModelType.LOCAL_RULES
    
    def test_cost_estimate(self, router):
        """测试成本估算"""
        estimate = router.get_cost_estimate(10000)
        
        assert estimate["total_calls"] == 10000
        assert estimate["r1_calls"] == 500  # 5%
        assert estimate["local_calls"] == 9500
        assert estimate["savings_vs_all_r1"] > 0


# ==========================================
# 分布式运行器测试
# ==========================================

class TestSimulationRunner:
    """分布式运行器测试"""
    
    @pytest.fixture
    def runner(self):
        return SimulationRunner(
            n_agents=100,
            agents_per_actor=20,
            r1_ratio=0.05
        )
    
    def test_initialization(self, runner):
        """测试初始化"""
        runner.initialize()
        
        assert runner.initialized
        assert len(runner.actors) == 5  # 100 / 20
    
    def test_step_execution(self, runner):
        """测试步骤执行"""
        runner.initialize()
        
        market = {
            "price": 3000.0,
            "trend": "上涨",
            "panic_level": 0.3,
            "volatility": 0.02
        }
        
        decisions = runner.step(market)
        
        assert len(decisions) == 100
        assert all(d.action in ["BUY", "SELL", "HOLD"] for d in decisions)
    
    def test_decision_distribution(self, runner):
        """测试决策分布"""
        runner.initialize()
        
        # 上涨市场，低恐慌
        market = {
            "trend": "上涨",
            "panic_level": 0.3
        }
        
        decisions = runner.step(market)
        
        buy_count = sum(1 for d in decisions if d.action == "BUY")
        
        # 大部分应该选择买入
        assert buy_count > len(decisions) * 0.5
    
    def test_stats(self, runner):
        """测试统计"""
        runner.initialize()
        runner.step({"trend": "neutral", "panic_level": 0.5})
        
        stats = runner.get_stats()
        
        assert stats["tick"] == 1
        assert stats["n_agents"] == 100
        assert "cost_estimate" in stats
    
    def test_shutdown(self, runner):
        """测试关闭"""
        runner.initialize()
        runner.shutdown()
        
        assert not runner.initialized
        assert len(runner.actors) == 0


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
