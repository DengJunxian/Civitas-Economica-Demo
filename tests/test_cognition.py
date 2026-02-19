# file: tests/test_cognition.py
"""
认知架构单元测试

测试覆盖:
1. 前景理论效用计算
2. 决策覆盖逻辑
3. 创伤记忆检索
4. CognitiveAgent 整合

作者: Civitas Economica Team
"""

import pytest
import math

from agents.cognition.utility import (
    ProspectTheory, calculate_prospect_value, weight_probability,
    InvestorType, create_panic_retail, create_normal_investor, create_disciplined_quant
)
from agents.cognition.llm_brain import (
    DeepSeekReasoner, LocalReasoner, Decision
)
from agents.cognition.memory import TraumaMemory, TraumaEvent
from agents.cognition.cognitive_agent import (
    CognitiveAgent, create_panic_retail_agent, create_quant_agent,
    create_population
)


# ==========================================
# 前景理论测试
# ==========================================

class TestProspectTheory:
    """前景理论效用函数测试"""
    
    @pytest.fixture
    def normal_prospect(self):
        return ProspectTheory(investor_type=InvestorType.NORMAL)
    
    @pytest.fixture
    def panic_prospect(self):
        return ProspectTheory(investor_type=InvestorType.PANIC_RETAIL)
    
    def test_gain_concavity(self, normal_prospect):
        """
        测试收益区域的凹形曲线
        
        边际效用递减：V(0.10) < 2 * V(0.05)
        """
        v_5pct = normal_prospect.calculate_value(0.05)
        v_10pct = normal_prospect.calculate_value(0.10)
        
        assert v_5pct > 0, "收益应为正效用"
        assert v_10pct > v_5pct, "更大收益应有更高效用"
        assert v_10pct < 2 * v_5pct, "边际效用应递减"
    
    def test_loss_steepness(self, normal_prospect):
        """
        测试损失区域比收益更陡峭
        
        损失厌恶：|V(-0.05)| > V(0.05)
        """
        v_gain = normal_prospect.calculate_value(0.05)
        v_loss = normal_prospect.calculate_value(-0.05)
        
        assert v_loss < 0, "损失应为负效用"
        assert abs(v_loss) > v_gain, "损失的痛苦应大于等量收益的快乐"
    
    def test_lambda_coefficient_effect(self):
        """
        测试 λ 系数对损失敏感度的影响
        
        λ 越大，损失越痛苦
        """
        normal = ProspectTheory(lambda_coeff=2.25)
        panic = ProspectTheory(lambda_coeff=3.0)
        quant = ProspectTheory(lambda_coeff=1.0)
        
        pnl = -0.05
        
        v_normal = normal.calculate_value(pnl)
        v_panic = panic.calculate_value(pnl)
        v_quant = quant.calculate_value(pnl)
        
        # 恐慌者感受到更大的痛苦
        assert abs(v_panic) > abs(v_normal)
        assert abs(v_normal) > abs(v_quant)
    
    def test_zero_pnl(self, normal_prospect):
        """测试零盈亏时的效用"""
        v = normal_prospect.calculate_value(0.0)
        assert v == 0.0, "零盈亏应有零效用"
    
    def test_probability_weighting(self):
        """
        测试概率权重函数
        
        特点：小概率高估，大概率低估
        """
        # 小概率
        w_plus, w_minus = weight_probability(0.05)
        assert w_plus > 0.05, "小概率应被高估"
        
        # 大概率
        w_plus, w_minus = weight_probability(0.95)
        assert w_plus < 0.95, "大概率应被低估"
        
        # 边界
        w_plus, w_minus = weight_probability(0.0)
        assert w_plus == 0.0
        
        w_plus, w_minus = weight_probability(1.0)
        assert w_plus == 1.0


class TestDecisionOverride:
    """决策覆盖逻辑测试"""
    
    @pytest.fixture
    def prospect(self):
        return ProspectTheory(investor_type=InvestorType.PANIC_RETAIL)
    
    def test_override_buy_when_fearful(self, prospect):
        """
        测试恐惧时覆盖买入决策
        
        场景：LLM 建议 BUY，但前景值极负
        预期：覆盖为 HOLD
        
        注意：前景值计算为 -λ * |pnl|^β
        对于 λ=3.0, β=0.88, pnl=-0.30:
        V = -3.0 * 0.30^0.88 ≈ -3.0 * 0.35 ≈ -1.05 < -0.5 (触发阈值)
        """
        pnl = -0.30  # 亏损 30% (足够触发恐惧覆盖)
        action, reason = prospect.should_override_decision("BUY", pnl)
        
        # 恐慌散户 λ=3.0 时，-30% 的前景值约 -1.05
        assert action == "HOLD", f"恐惧时应覆盖买入为观望，实际: {action}"
        assert reason is not None
    
    def test_override_hold_when_panic(self, prospect):
        """
        测试极度恐慌时触发卖出
        
        场景：LLM 建议 HOLD，但前景值极低
        预期：可能触发恐慌性卖出
        """
        pnl = -0.15  # 亏损 15%
        action, reason = prospect.should_override_decision("HOLD", pnl)
        
        # 根据阈值设置，可能触发卖出
        assert action in ["HOLD", "SELL"]
    
    def test_no_override_normal_situation(self, prospect):
        """测试正常情况下不覆盖"""
        pnl = -0.02  # 小幅亏损
        action, reason = prospect.should_override_decision("HOLD", pnl)
        
        assert action == "HOLD"
        assert reason is None
    
    def test_disposition_effect(self, prospect):
        """
        测试处置效应
        
        场景：已盈利时尝试追加买入
        预期：建议观望而非追加
        """
        pnl = 0.08  # 盈利 8%
        action, reason = prospect.should_override_decision("BUY", pnl)
        
        # 处置效应：盈利时不应追加
        assert action == "HOLD", "盈利时应观望而非追加"
        assert "处置效应" in reason or "盈利" in reason


# ==========================================
# 记忆模块测试
# ==========================================

class TestTraumaMemory:
    """创伤记忆测试"""
    
    @pytest.fixture
    def memory(self):
        return TraumaMemory()
    
    def test_add_and_retrieve(self, memory):
        """测试基本的添加和检索"""
        memory.add_memory("大盘暴跌，恐慌抛售", outcome=-0.6)
        memory.add_memory("抄底成功，大幅盈利", outcome=0.8)
        
        results = memory.retrieve("市场下跌", top_k=1)
        assert len(results) == 1
    
    def test_trauma_detection(self, memory):
        """测试创伤记忆识别"""
        memory.add_memory("熔断崩盘，血亏出局", outcome=-0.8)
        
        # 检查是否被标记为创伤
        assert memory.fragments[-1].is_trauma()
    
    def test_volatility_triggered_retrieval(self, memory):
        """
        测试波动率触发的创伤记忆检索
        
        高波动时应自动检索历史崩盘
        """
        # 高波动
        results = memory.retrieve_trauma(volatility=0.05, threshold=0.03)
        assert len(results) > 0, "高波动应触发创伤记忆"
        
        # 低波动
        results = memory.retrieve_trauma(volatility=0.01, threshold=0.03)
        assert len(results) == 0, "低波动不应触发"
    
    def test_emotional_baggage(self, memory):
        """测试情绪包袱计算"""
        # 初始状态
        baggage, desc = memory.get_emotional_baggage()
        assert baggage == 0.0, "初始应无包袱"
        
        # 添加创伤记忆
        memory.add_memory("暴跌亏损", outcome=-0.7)
        memory.add_memory("连续跌停", outcome=-0.9)
        
        baggage, desc = memory.get_emotional_baggage()
        assert baggage > 0, "创伤后应有包袱"
    
    def test_crash_injection(self, memory):
        """测试崩盘记忆注入"""
        initial_traumas = len(memory.trauma_events)
        
        memory.inject_crash_memory(TraumaEvent(
            date="2025-01-01",
            description="测试崩盘事件",
            loss_pct=-0.20,
            market_conditions={},
            emotional_impact=0.9
        ))
        
        assert len(memory.trauma_events) == initial_traumas + 1


# ==========================================
# 本地推理器测试
# ==========================================

class TestLocalReasoner:
    """本地规则推理器测试"""
    
    @pytest.fixture
    def reasoner(self):
        return LocalReasoner()
    
    def test_bullish_market(self, reasoner):
        """测试看涨市场"""
        market = {"trend": "上涨", "panic_level": 0.2}
        account = {"pnl_pct": 0.02}
        
        result = reasoner.reason(market, account)
        assert result.decision.action == "BUY"
    
    def test_bearish_market(self, reasoner):
        """测试看跌市场"""
        market = {"trend": "下跌", "panic_level": 0.3}
        account = {"pnl_pct": 0.0}
        
        result = reasoner.reason(market, account)
        assert result.decision.action == "SELL"
    
    def test_high_panic_overrides(self, reasoner):
        """测试高恐慌指数覆盖"""
        market = {"trend": "上涨", "panic_level": 0.8}
        account = {"pnl_pct": 0.0}
        
        result = reasoner.reason(market, account)
        # 高恐慌应谨慎，不应激进买入
        assert result.decision.action in ["HOLD", "SELL"]


# ==========================================
# CognitiveAgent 整合测试
# ==========================================

class TestCognitiveAgent:
    """认知 Agent 整合测试"""
    
    @pytest.fixture
    def panic_agent(self):
        return create_panic_retail_agent("test_panic", use_local_reasoner=True)
    
    @pytest.fixture
    def quant_agent(self):
        return create_quant_agent("test_quant", use_local_reasoner=True)
    
    def test_agent_creation(self, panic_agent, quant_agent):
        """测试 Agent 创建"""
        assert panic_agent.prospect.lambda_coeff == 3.0
        assert quant_agent.prospect.lambda_coeff == 1.0
    
    def test_decision_making(self, panic_agent):
        """测试决策流程"""
        market = {"price": 3000, "trend": "下跌", "panic_level": 0.6, "volatility": 0.03}
        account = {"cash": 50000, "market_value": 50000, "pnl_pct": -0.05}
        
        decision = panic_agent.make_decision(market, account)
        
        assert decision.final_action in ["BUY", "SELL", "HOLD"]
        assert decision.prospect_value != 0  # 应有前景值计算
    
    def test_heterogeneous_response(self, panic_agent, quant_agent):
        """
        测试异质性响应
        
        相同市场条件下，不同类型 Agent 应有不同反应
        """
        market = {"price": 3000, "trend": "下跌", "panic_level": 0.7, "volatility": 0.04}
        account = {"cash": 50000, "market_value": 50000, "pnl_pct": -0.08}
        
        decision_panic = panic_agent.make_decision(market, account)
        decision_quant = quant_agent.make_decision(market, account)
        
        # 前景值应不同 (恐慌者更痛苦)
        assert abs(decision_panic.prospect_value) > abs(decision_quant.prospect_value)
    
    def test_outcome_recording(self, panic_agent):
        """测试结果记录"""
        initial_memories = len(panic_agent.memory)
        
        panic_agent.record_outcome(
            pnl=-0.06,
            market_summary="追高被套，市场反转",
            volatility=0.03
        )
        
        assert len(panic_agent.memory) == initial_memories + 1
    
    def test_emotional_profile(self, panic_agent):
        """测试情绪画像"""
        market = {"price": 3000, "trend": "震荡", "panic_level": 0.5, "volatility": 0.02}
        account = {"cash": 50000, "market_value": 50000, "pnl_pct": 0.0}
        
        # 做几次决策
        for _ in range(3):
            panic_agent.make_decision(market, account)
        
        profile = panic_agent.get_emotional_profile()
        
        assert "agent_id" in profile
        assert "lambda_coeff" in profile
        assert profile["total_decisions"] == 3
    
    def test_population_creation(self):
        """测试群体创建"""
        population = create_population(size=10, panic_ratio=0.3, quant_ratio=0.2)
        
        assert len(population) == 10
        
        # 检查类型分布
        panic_count = sum(1 for a in population if "panic" in a.agent_id)
        quant_count = sum(1 for a in population if "quant" in a.agent_id)
        
        assert panic_count == 3
        assert quant_count == 2


# ==========================================
# 运行测试
# ==========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
