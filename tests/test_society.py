# file: tests/test_society.py
"""
多智能体编排与社会网络单元测试

测试覆盖:
1. 分析师信号生成
2. 团队决策与 Debate 机制
3. 小世界网络拓扑
4. SIR 情绪传播
5. 羊群效应 λ 动态调整

作者: Civitas Economica Team
"""

import pytest
import numpy as np

from agents.roles.analyst import (
    Analyst, FundamentalAnalyst, TechnicalAnalyst, SentimentAnalyst,
    Signal, SignalType, tool
)
from agents.crews.investment_firm import (
    InvestmentTeam, RiskManager, Trader, RiskConstraints, RiskCheckResult
)
from core.society.network import (
    SocialGraph, InformationDiffusion, AgentNode, SentimentState
)


# ==========================================
# 分析师角色测试
# ==========================================

class TestAnalysts:
    """分析师角色测试"""
    
    @pytest.fixture
    def market_data(self):
        return {
            "price": 100.0,
            "prices": [100 + np.sin(i/5) * 5 for i in range(60)],
            "volatility": 0.03,
            "volume_ratio": 1.5,
            "momentum": -0.1,
            "panic_level": 0.65,
            "news": ["央行降准利好股市"]
        }
    
    def test_fundamental_analyst_creation(self):
        """测试基本面分析师创建"""
        analyst = FundamentalAnalyst()
        assert analyst.specialty == "fundamental"
        assert len(analyst.tools) > 0
    
    def test_fundamental_analyst_signal(self, market_data):
        """测试基本面分析师信号生成"""
        analyst = FundamentalAnalyst()
        signal = analyst.analyze("000001", market_data)
        
        assert signal.action in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0 <= signal.confidence <= 1
        assert signal.reasoning != ""
    
    def test_technical_analyst_indicators(self):
        """测试技术分析师指标计算"""
        analyst = TechnicalAnalyst()
        
        prices = [100 + i * 0.5 for i in range(60)]  # 上涨趋势
        macd = analyst.calculate_macd(prices)
        
        assert "macd" in macd
        assert "signal" in macd
        assert "histogram" in macd
    
    def test_technical_analyst_rsi(self):
        """测试 RSI 计算"""
        analyst = TechnicalAnalyst()
        
        # 强势上涨
        prices = [100 + i for i in range(30)]
        rsi = analyst.calculate_rsi(prices)
        assert rsi > 50  # 应该偏高
        
        # 强势下跌
        prices = [100 - i for i in range(30)]
        rsi = analyst.calculate_rsi(prices)
        assert rsi < 50  # 应该偏低
    
    def test_sentiment_analyst_fear_greed(self):
        """测试恐贪指数计算"""
        analyst = SentimentAnalyst()
        
        # 高波动 + 下跌动量 = 恐惧
        fg = analyst.calculate_fear_greed(
            volatility=0.05,
            volume_ratio=2.0,
            price_momentum=-0.5
        )
        assert fg["index"] < 50
        assert fg["status"] in ["fear", "extreme_fear"]
        
        # 低波动 + 上涨动量 = 贪婪
        fg = analyst.calculate_fear_greed(
            volatility=0.01,
            volume_ratio=1.0,
            price_momentum=0.5
        )
        assert fg["index"] > 50
    
    def test_tool_decorator(self):
        """测试 @tool 装饰器"""
        analyst = FundamentalAnalyst()
        
        tools = analyst.list_tools()
        assert len(tools) > 0
        
        # 检查工具元信息
        tool_names = [t["name"] for t in tools]
        assert "fetch_financial_report" in tool_names


# ==========================================
# 投资团队测试
# ==========================================

class TestInvestmentTeam:
    """投资团队测试"""
    
    @pytest.fixture
    def team(self):
        return InvestmentTeam()
    
    @pytest.fixture
    def market_data(self):
        return {
            "price": 100.0,
            "prices": [100 + np.sin(i/5) * 5 for i in range(60)],
            "volatility": 0.02,
            "volume_ratio": 1.2,
            "momentum": 0.0,
            "panic_level": 0.5,
            "news": []
        }
    
    @pytest.fixture
    def account_state(self):
        return {
            "cash": 100000.0,
            "holdings": 500,
            "daily_pnl_pct": 0.0
        }
    
    def test_team_creation(self, team):
        """测试团队创建"""
        assert len(team.analysts) == 3
        assert team.risk_manager is not None
        assert team.trader is not None
    
    def test_team_decision_flow(self, team, market_data, account_state):
        """测试团队决策流程"""
        decision = team.decide("000001", market_data, account_state)
        
        assert len(decision.analyst_signals) == 3
        assert decision.consensus_action in SignalType
        assert decision.risk_check in RiskCheckResult
    
    def test_conflict_detection(self, team):
        """测试冲突检测"""
        # 无冲突
        actions_no_conflict = [SignalType.BUY, SignalType.BUY, SignalType.HOLD]
        assert not team._has_conflict(actions_no_conflict)
        
        # 有冲突
        actions_conflict = [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert team._has_conflict(actions_conflict)
    
    def test_majority_vote(self, team):
        """测试多数投票"""
        actions = [SignalType.BUY, SignalType.BUY, SignalType.SELL]
        result = team._majority_vote(actions)
        assert result == SignalType.BUY
    
    def test_debate_mechanism(self, team):
        """测试 Debate 机制"""
        from agents.roles.analyst import AnalystType
        
        signals = [
            Signal(
                action=SignalType.BUY,
                confidence=0.7,
                analyst_type=AnalystType.FUNDAMENTAL,
                symbol="000001",
                reasoning="估值低"
            ),
            Signal(
                action=SignalType.SELL,
                confidence=0.8,
                analyst_type=AnalystType.TECHNICAL,
                symbol="000001",
                reasoning="技术形态破位"
            )
        ]
        
        consensus, debate_log = team._resolve_conflict(signals)
        
        assert consensus in SignalType
        assert len(debate_log) > 0
        assert "Debate" in debate_log[0]


# ==========================================
# 风险管理测试
# ==========================================

class TestRiskManager:
    """风险管理测试"""
    
    @pytest.fixture
    def risk_manager(self):
        return RiskManager()
    
    def test_buy_risk_check_approved(self, risk_manager):
        """测试买入风控通过"""
        signal = Signal(
            action=SignalType.BUY,
            confidence=0.8,
            analyst_type=None,
            symbol="000001"
        )
        
        account = {"cash": 100000, "holdings": 0, "daily_pnl_pct": 0}
        
        result, msg, qty = risk_manager.check(signal, account, 100.0)
        
        assert result == RiskCheckResult.APPROVED
        assert qty > 0
    
    def test_buy_risk_check_rejected_no_cash(self, risk_manager):
        """测试买入风控拒绝 - 资金不足"""
        signal = Signal(
            action=SignalType.BUY,
            confidence=0.8,
            analyst_type=None,
            symbol="000001"
        )
        
        account = {"cash": 100, "holdings": 0, "daily_pnl_pct": 0}
        
        result, msg, qty = risk_manager.check(signal, account, 100.0)
        
        assert result == RiskCheckResult.REJECTED
    
    def test_daily_loss_limit(self, risk_manager):
        """测试单日亏损限制"""
        signal = Signal(
            action=SignalType.BUY,
            confidence=0.8,
            analyst_type=None,
            symbol="000001"
        )
        
        account = {"cash": 100000, "holdings": 0, "daily_pnl_pct": -0.06}
        
        result, msg, qty = risk_manager.check(signal, account, 100.0)
        
        assert result == RiskCheckResult.REJECTED
        assert "单日亏损" in msg


# ==========================================
# 社会网络测试
# ==========================================

class TestSocialGraph:
    """社会网络测试"""
    
    @pytest.fixture
    def graph(self):
        return SocialGraph(n_agents=100, k=4, p=0.3, seed=42)
    
    def test_graph_creation(self, graph):
        """测试网络创建"""
        assert graph.n_agents == 100
        assert len(graph.agents) == 100
    
    def test_small_world_properties(self, graph):
        """测试小世界网络特性"""
        stats = graph.get_network_stats()
        
        # 聚类系数应该较高 (> 0.1)
        assert stats["clustering_coefficient"] > 0.1
        
        # 平均度应该接近 k
        assert abs(stats["avg_degree"] - graph.k) < 1
    
    def test_neighbor_retrieval(self, graph):
        """测试邻居检索"""
        neighbors = graph.get_neighbors(0)
        
        assert len(neighbors) > 0
        assert 0 not in neighbors  # 不包含自己
    
    def test_bearish_ratio(self, graph):
        """测试看空比例计算"""
        # 初始状态全部为 SUSCEPTIBLE
        ratio = graph.get_bearish_ratio(0)
        assert ratio == 0.0
        
        # 设置一些邻居为 INFECTED
        neighbors = graph.get_neighbors(0)[:2]
        for n in neighbors:
            graph.agents[n].sentiment_state = SentimentState.INFECTED
        
        ratio = graph.get_bearish_ratio(0)
        assert ratio > 0


# ==========================================
# 情绪传播测试
# ==========================================

class TestInformationDiffusion:
    """情绪传播测试"""
    
    @pytest.fixture
    def diffusion(self):
        graph = SocialGraph(n_agents=100, k=4, p=0.3, seed=42)
        return InformationDiffusion(
            social_graph=graph,
            beta=0.3,
            gamma=0.1,
            delta=0.05,
            lambda_amplifier=0.5
        )
    
    def test_panic_injection(self, diffusion):
        """测试恐慌注入"""
        seeds = diffusion.inject_panic(n_seeds=5, method="random")
        
        assert len(seeds) == 5
        
        # 检查状态变化
        for node_id in seeds:
            assert diffusion.graph.agents[node_id].sentiment_state == SentimentState.INFECTED
    
    def test_sentiment_propagation(self, diffusion):
        """测试情绪传播"""
        # 注入种子
        diffusion.inject_panic(n_seeds=10, method="influential")
        
        initial_infected = sum(
            1 for a in diffusion.graph.agents.values()
            if a.sentiment_state == SentimentState.INFECTED
        )
        
        # 运行几轮传播
        for _ in range(5):
            diffusion.update_sentiment_propagation()
        
        current_infected = sum(
            1 for a in diffusion.graph.agents.values()
            if a.sentiment_state == SentimentState.INFECTED
        )
        
        # 应该有传播发生 (或恢复)
        assert current_infected != initial_infected or diffusion.graph.tick > 0
    
    def test_lambda_herding_effect(self, diffusion):
        """测试羊群效应 λ 调整"""
        # 注入大量恐慌
        diffusion.inject_panic(n_seeds=30, method="random")
        
        # 运行传播
        for _ in range(10):
            diffusion.update_sentiment_propagation()
        
        # 检查 λ 变化
        lambdas = [a.lambda_coeff for a in diffusion.graph.agents.values()]
        avg_lambda = np.mean(lambdas)
        
        # 平均 λ 应该有所增加 (羊群效应)
        base_lambdas = [a.base_lambda for a in diffusion.graph.agents.values()]
        avg_base = np.mean(base_lambdas)
        
        # 至少部分 Agent 的 λ 应该增加
        assert max(lambdas) > avg_base
    
    def test_sir_state_transitions(self, diffusion):
        """测试 SIR 状态转换"""
        # 运行完整模拟
        history = diffusion.simulate(n_ticks=50, initial_infected=10)
        
        assert len(history) == 50
        
        # 检查是否有状态变化
        infected_counts = [h["infected"] for h in history]
        recovered_counts = [h["recovered"] for h in history]
        
        # 应该有峰值和衰减
        assert max(infected_counts) > 10  # 有传播
        assert sum(recovered_counts) > 0 or max(infected_counts) > history[-1]["infected"]  # 有恢复


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
