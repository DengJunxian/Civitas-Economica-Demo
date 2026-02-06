# file: agents/cognition/cognitive_agent.py
"""
认知 Agent 整合模块

将前景理论、DeepSeek 推理和 RAG 记忆三大模块整合为
完整的 CognitiveAgent，实现"硅基投资者"的模拟。

核心特性:
1. 效用函数对 LLM 决策的覆盖
2. 创伤记忆的自动检索与注入
3. 完整的思维链记录

作者: Civitas Economica Team
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from agents.cognition.utility import (
    ProspectTheory, ProspectValue, InvestorType,
    ConfidenceTracker, AnchorTracker,
    create_panic_retail, create_normal_investor, create_disciplined_quant
)
from agents.cognition.llm_brain import (
    DeepSeekReasoner, LocalReasoner, ReasoningResult, Decision
)
from agents.cognition.memory import TraumaMemory


# ==========================================
# 数据结构
# ==========================================

@dataclass
class CognitiveDecision:
    """
    认知决策结果
    
    包含原始 LLM 决策、前景理论分析、覆盖逻辑和最终动作。
    """
    # 最终决策
    final_action: str          # "BUY" | "SELL" | "HOLD"
    final_quantity: int = 0
    final_confidence: float = 0.5
    
    # LLM 原始决策
    llm_action: str = ""
    llm_reasoning: str = ""
    
    # 前景理论分析
    prospect_value: float = 0.0
    pain_gain_ratio: float = 0.0
    decision_bias: str = ""
    
    # 覆盖信息
    was_overridden: bool = False
    override_reason: Optional[str] = None
    
    # 情绪状态
    fear_level: float = 0.0
    greed_level: float = 0.0
    
    # 记忆上下文
    memory_context: str = ""
    
    # 元数据
    inference_time_ms: float = 0.0
    model_used: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "action": self.final_action,
            "qty": self.final_quantity,
            "confidence": self.final_confidence,
            "was_overridden": self.was_overridden,
            "override_reason": self.override_reason,
            "prospect_value": self.prospect_value,
            "fear_level": self.fear_level,
            "greed_level": self.greed_level,
            "model_used": self.model_used
        }


# ==========================================
# 认知 Agent
# ==========================================

class CognitiveAgent:
    """
    认知 Agent - "硅基投资者"
    
    整合三大认知模块:
    1. **前景理论效用**: 模拟损失厌恶、处置效应等非理性行为
    2. **DeepSeek R1 推理**: 获取类人思维链 (CoT)
    3. **创伤记忆 RAG**: 高波动时触发历史教训回忆
    
    核心机制:
    - LLM 提供初始决策建议
    - 前景理论计算效用值
    - 当效用值超过阈值时，覆盖 LLM 决策
    
    Attributes:
        agent_id: Agent 唯一标识
        investor_type: 投资者类型
        prospect: 前景理论计算器
        reasoner: LLM 推理引擎
        memory: RAG 记忆库
    """
    
    def __init__(
        self,
        agent_id: str,
        investor_type: InvestorType = InvestorType.NORMAL,
        api_key: Optional[str] = None,
        use_local_reasoner: bool = False,
        lambda_override: Optional[float] = None
    ):
        """
        初始化认知 Agent
        
        Args:
            agent_id: Agent 唯一标识
            investor_type: 投资者类型预设
            api_key: DeepSeek API Key
            use_local_reasoner: 是否使用本地规则引擎 (无 API 调用)
            lambda_override: 覆盖损失厌恶系数 (可选)
        """
        self.agent_id = agent_id
        self.investor_type = investor_type
        
        # 初始化前景理论计算器
        if lambda_override is not None:
            self.prospect = ProspectTheory(lambda_coeff=lambda_override)
        else:
            self.prospect = ProspectTheory(investor_type=investor_type)
        
        # 初始化推理引擎
        if use_local_reasoner:
            self.reasoner = LocalReasoner()
        else:
            self.reasoner = DeepSeekReasoner(api_key=api_key)
        
        # 初始化记忆库
        self.memory = TraumaMemory()
        
        # 初始化心理跟踪器
        self.confidence_tracker = ConfidenceTracker()
        self.anchor_tracker = None  # 将在第一次 make_decision 或持仓时初始化
        
        # 决策历史
        self.decision_history: List[CognitiveDecision] = []
        
        # 覆盖阈值配置
        self.fear_threshold = -0.5    # 恐惧覆盖阈值
        self.greed_threshold = 0.3    # 贪婪覆盖阈值
    
    def prepare_decision_prompt(self, market_state: Dict, account_state: Dict) -> Optional[List[Dict]]:
        """
        [并行化支持] 阶段1: 准备提示词 (不调用 LLM)
        """
        # 如果是本地推理，直接返回 None，表示不需要 LLM 调用
        if isinstance(self.reasoner, LocalReasoner):
            return None
            
        if hasattr(self.reasoner, "build_messages"):
            return self.reasoner.build_messages(market_state, account_state)
        return None

    def finalize_decision_from_result(
        self, 
        result: ReasoningResult, 
        market_state: Dict,
        account_state: Dict
    ) -> CognitiveDecision:
        """
        [并行化支持] 阶段2: 根据 LLM 结果生成最终决策
        """
        # 1. 获取 LLM 建议
        # result 已经是 ReasoningResult 对象
        raw_decision = result.decision
        
        # 2. 前景理论效用计算 (Cognitive Flow)
        # Calculate utility of current state to adjust fear/greed?
        # Or calculate utility of the PROPOSED action?
        # Usually we evaluate the status quo pnl.
        pnl = account_state.get("pnl_pct", 0)
        utility = self.prospect.calculate_utility(pnl)
        
        # 3. 覆盖逻辑
        final_action = raw_decision.action
        final_qty = raw_decision.quantity
        
        # 简单覆盖示例: 极度恐惧时强制卖出
        # (Simplified logic from original make_decision)
        if utility < -150: # Pain threshold
            # Panic Sell
            if final_action != "SELL":
                final_action = "SELL"
                final_qty = account_state.get("position", 0)
        
        from agents.cognition.llm_brain import Decision as LLMDecision
        # Convert back to internal CognitiveDecision or similar?
        # For now, return a named tuple or object expected by CivitasAgent
        # CivitasAgent expects an object with .final_action, .final_quantity, .greed_level...
        
        # Reuse Decision dataclass from llm_brain? 
        # But step() code expects .final_action etc. 
        # LLMDecision has these properties.
        
        # We need to ensure we return something compatible
        return raw_decision

    def make_decision(
        self,
        market_state: Dict,
        account_state: Dict,
        symbol: str = "000001"
    ) -> ReasoningResult:
        """
        执行认知决策过程 (同步模式 - 兼容旧代码)
        """
        start_time = time.time()
        
        pnl_pct = account_state.get('pnl_pct', 0)
        current_price = market_state.get('price', 0)
        avg_cost = account_state.get('avg_cost', 0)
        
        # 初始化/更新锚点
        if self.anchor_tracker is None and avg_cost > 0:
            self.anchor_tracker = AnchorTracker(initial_cost=avg_cost, reference_point=avg_cost)
        elif self.anchor_tracker:
            self.anchor_tracker.update(current_price)
            
        # 获取心理状态描述
        confidence_desc = self.confidence_tracker.get_description()
        anchor_desc = self.anchor_tracker.get_bias_description(current_price) if self.anchor_tracker else ""
        
        # ========== 阶段 1: 记忆检索 ==========
        memory_context = self.memory.get_context_for_decision(market_state)
        
        # ========== 阶段 2: LLM 推理 ==========
        # 计算前景值用于本地推理器
        prospect_value = self.prospect.calculate_value(pnl_pct)
        
        if isinstance(self.reasoner, LocalReasoner):
            result = self.reasoner.reason(
                market_state, account_state,
                prospect_value=prospect_value
            )
        else:
            result = self.reasoner.reason(
                market_state, account_state,
                symbol=symbol,
                lambda_coeff=self.prospect.lambda_coeff,
                memory_context=memory_context,
                confidence_desc=confidence_desc,
                anchor_desc=anchor_desc,
                csad=market_state.get('csad', None)
            )
        
        llm_action = result.decision.action
        llm_quantity = result.decision.quantity
        
        # ========== 阶段 3: 前景理论分析 ==========
        prospect_result = self.prospect.calculate_full(pnl_pct)
        
        # ========== 阶段 4: 效用覆盖逻辑 ==========
        final_action, override_reason = self.prospect.should_override_decision(
            llm_action=llm_action,
            pnl=pnl_pct,
            fear_threshold=self.fear_threshold,
            greed_threshold=self.greed_threshold
        )
        
        was_overridden = (final_action != llm_action)
        
        # 如果被覆盖为 HOLD，数量设为 0
        if was_overridden and final_action == "HOLD":
            final_quantity = 0
        else:
            final_quantity = llm_quantity
        
        # ========== 构建决策结果 ==========
        decision = CognitiveDecision(
            final_action=final_action,
            final_quantity=final_quantity,
            final_confidence=result.decision.confidence,
            llm_action=llm_action,
            llm_reasoning=result.chain_of_thought,
            prospect_value=prospect_result.subjective_value,
            pain_gain_ratio=prospect_result.pain_gain_ratio,
            decision_bias=prospect_result.decision_bias,
            was_overridden=was_overridden,
            override_reason=override_reason,
            fear_level=result.emotional_state.fear_level,
            greed_level=result.emotional_state.greed_level,
            memory_context=memory_context,
            inference_time_ms=(time.time() - start_time) * 1000,
            model_used=result.model_used
        )
        
        # 记录历史
        self.decision_history.append(decision)
        if len(self.decision_history) > 50:
            self.decision_history.pop(0)
        
        return decision
    
    def record_outcome(
        self,
        pnl: float,
        market_summary: str,
        volatility: float = 0.0
    ) -> None:
        """
        记录交易结果
        
        将结果写入记忆库，供未来决策参考。
        
        Args:
            pnl: 盈亏比例
            market_summary: 市场概况
            volatility: 当时的波动率
        """
        if pnl < -0.05:
            content = f"[亏损 {pnl:.1%}] {market_summary}。教训：{self._generate_lesson(pnl)}"
        elif pnl > 0.05:
            content = f"[盈利 {pnl:.1%}] {market_summary}。成功经验。"
        else:
            content = f"[持平] {market_summary}。"
        
        self.memory.add_memory(content, outcome=pnl, volatility=volatility)
        
        # 更新自信心
        self.confidence_tracker.update(pnl)
    
    def _generate_lesson(self, pnl: float) -> str:
        """生成教训描述"""
        if pnl < -0.10:
            return "重大亏损，需要严格止损纪律"
        elif pnl < -0.05:
            return "中等亏损，应该更谨慎"
        else:
            return "小幅亏损，需要优化入场时机"
    
    def get_emotional_profile(self) -> Dict:
        """
        获取情绪画像
        
        Returns:
            包含情绪包袱、风险偏好等信息的字典
        """
        baggage, baggage_desc = self.memory.get_emotional_baggage()
        
        avg_fear = 0.0
        avg_greed = 0.0
        if self.decision_history:
            avg_fear = sum(d.fear_level for d in self.decision_history) / len(self.decision_history)
            avg_greed = sum(d.greed_level for d in self.decision_history) / len(self.decision_history)
        
        return {
            "agent_id": self.agent_id,
            "investor_type": self.investor_type.value,
            "lambda_coeff": self.prospect.lambda_coeff,
            "emotional_baggage": baggage,
            "baggage_description": baggage_desc,
            "avg_fear_level": avg_fear,
            "avg_greed_level": avg_greed,
            "total_decisions": len(self.decision_history),
            "override_rate": sum(1 for d in self.decision_history if d.was_overridden) / max(1, len(self.decision_history))
        }
    
    def __repr__(self) -> str:
        return (
            f"CognitiveAgent(id={self.agent_id}, "
            f"type={self.investor_type.value}, "
            f"λ={self.prospect.lambda_coeff})"
        )


# ==========================================
# 工厂函数
# ==========================================

def create_panic_retail_agent(agent_id: str, **kwargs) -> CognitiveAgent:
    """创建恐慌型散户 Agent"""
    return CognitiveAgent(agent_id, InvestorType.PANIC_RETAIL, **kwargs)


def create_normal_agent(agent_id: str, **kwargs) -> CognitiveAgent:
    """创建普通投资者 Agent"""
    return CognitiveAgent(agent_id, InvestorType.NORMAL, **kwargs)


def create_quant_agent(agent_id: str, **kwargs) -> CognitiveAgent:
    """创建纪律型量化 Agent"""
    return CognitiveAgent(agent_id, InvestorType.DISCIPLINED_QUANT, **kwargs)


def create_population(
    size: int = 10,
    panic_ratio: float = 0.3,
    quant_ratio: float = 0.1
) -> List[CognitiveAgent]:
    """
    创建异质性 Agent 群体
    
    Args:
        size: 群体规模
        panic_ratio: 恐慌散户比例
        quant_ratio: 量化比例
        
    Returns:
        CognitiveAgent 列表
    """
    agents = []
    
    num_panic = int(size * panic_ratio)
    num_quant = int(size * quant_ratio)
    num_normal = size - num_panic - num_quant
    
    for i in range(num_panic):
        agents.append(create_panic_retail_agent(
            f"panic_{i}", use_local_reasoner=True
        ))
    
    for i in range(num_quant):
        agents.append(create_quant_agent(
            f"quant_{i}", use_local_reasoner=True
        ))
    
    for i in range(num_normal):
        agents.append(create_normal_agent(
            f"normal_{i}", use_local_reasoner=True
        ))
    
    return agents


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("CognitiveAgent 认知 Agent 测试")
    print("=" * 60)
    
    # 创建不同类型的 Agent
    panic_agent = create_panic_retail_agent("散户小明", use_local_reasoner=True)
    quant_agent = create_quant_agent("量化1号", use_local_reasoner=True)
    
    # 模拟市场状态
    market = {
        "price": 3000.0,
        "trend": "下跌",
        "panic_level": 0.75,
        "volatility": 0.04,
        "news": "美联储加息超预期"
    }
    
    # 模拟账户状态 - 亏损中
    account = {
        "cash": 40000.0,
        "market_value": 60000.0,
        "pnl_pct": -0.08
    }
    
    print("\n[场景] 市场下跌，恐慌指数 0.75，账户亏损 8%")
    print("-" * 50)
    
    # 测试恐慌散户
    decision1 = panic_agent.make_decision(market, account)
    print(f"\n[恐慌散户 - λ={panic_agent.prospect.lambda_coeff}]")
    print(f"  LLM 建议: {decision1.llm_action}")
    print(f"  前景值: {decision1.prospect_value:.4f}")
    print(f"  最终决策: {decision1.final_action}")
    print(f"  是否覆盖: {decision1.was_overridden}")
    if decision1.override_reason:
        print(f"  覆盖原因: {decision1.override_reason}")
    print(f"  决策偏差: {decision1.decision_bias}")
    
    # 测试量化
    decision2 = quant_agent.make_decision(market, account)
    print(f"\n[量化 Agent - λ={quant_agent.prospect.lambda_coeff}]")
    print(f"  LLM 建议: {decision2.llm_action}")
    print(f"  前景值: {decision2.prospect_value:.4f}")
    print(f"  最终决策: {decision2.final_action}")
    print(f"  是否覆盖: {decision2.was_overridden}")
    print(f"  决策偏差: {decision2.decision_bias}")
    
    # 测试情绪画像
    print(f"\n[情绪画像]")
    profile = panic_agent.get_emotional_profile()
    for k, v in profile.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
