# file: agents/cognition/llm_brain.py
"""
LLM 大脑与推理引擎

负责:
1. 构建 Prompt
2. 调用 DeepSeek/混元 API (通过 ModelRouter)
3. 解析 JSON 决策
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

from core.model_router import ModelRouter

@dataclass
class Decision:
    """决策结果"""
    action: str  # BUY, SELL, HOLD
    quantity: int
    price: Optional[float] = None
    reason: str = ""
    confidence: float = 0.5
    
    # 情绪指标
    fear_level: float = 0.0
    greed_level: float = 0.0
    
    @property
    def final_action(self):
        return self.action
        
    @property
    def final_quantity(self):
        return self.quantity

@dataclass
class ReasoningResult:
    """推理过程完整结果"""
    decision: Decision
    raw_response: str
    thought_chain: str
    model_used: str

class BaseReasoner:
    """推理器基类"""
    def derive_decision(self, market_state: Dict, account_state: Dict) -> ReasoningResult:
        raise NotImplementedError

class LocalReasoner(BaseReasoner):
    """
    本地规则推理器 (无需 LLM)
    用于快速模拟或基准测试
    """
    def derive_decision(self, market_state: Dict, account_state: Dict) -> ReasoningResult:
        # 简单均值回归策略
        price = market_state.get("price", 100)
        ma5 = price 
        # 假设 market_state 里有MA信息，如果没有则简化
        
        # 简单随机漫步 + 动量
        action = "HOLD"
        qty = 0
        import random
        r = random.random()
        
        if r < 0.1:
            action = "BUY"
            qty = 100
        elif r > 0.9:
            action = "SELL"
            qty = 100
            
        decision = Decision(action=action, quantity=qty, reason="Random Rule", confidence=0.5)
        return ReasoningResult(decision, "Local Rule", "None", "local")

class DeepSeekReasoner(BaseReasoner):
    """
    DeepSeek 深度思考推理器
    
    支持:
    - DeepSeek-R1 (CoT)
    - 自动降级
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # 注意: 这里最好共享 Router，但为了兼容旧代码，我们在此实例化
        # 如果 Config 正确加载，这里可以用 Global Router?
        # 为了支持 Parallel Refactoring，我们建议从外部传入 Router，或者使用 Global 单例
        from config import GLOBAL_CONFIG
        self.router = ModelRouter(
            deepseek_key=api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY,
            hunyuan_key=GLOBAL_CONFIG.HUNYUAN_API_KEY,
            zhipu_key=GLOBAL_CONFIG.ZHIPU_API_KEY
        )
        # 提示词模板
        self.system_prompt = """你是一个A股市场的个人投资者，你需要根据市场信息和账户状态做出交易决策。
请严格遵守以下 JSON 格式输出:
{
    "action": "BUY" | "SELL" | "HOLD",
    "quantity": <int>,
    "price": <float, optional limit price>,
    "reason": "<string>",
    "confidence": <float 0-1>,
    "fear_level": <float 0-1>,
    "greed_level": <float 0-1>
}
"""

    def _build_user_prompt(self, market_state: Dict, account_state: Dict) -> str:
        """构建用户提示词"""
        return f"""
当前市场状态:
- 价格: {market_state.get('price')}
- 趋势: {market_state.get('trend')}
- 恐慌指数: {market_state.get('panic_level')}
- 新闻: {market_state.get('news')}

你的账户:
- 现金: {account_state.get('cash'):.2f}
- 持仓: {account_state.get('position')}
- 平均成本: {account_state.get('avg_cost'):.2f}
- 当前浮盈: {account_state.get('pnl_pct')*100:.2f}%

请分析并做出决策。
"""

    def build_messages(self, market_state: Dict, account_state: Dict) -> List[Dict]:
        """构建完整的对话消息"""
        prompt = self._build_user_prompt(market_state, account_state)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

    async def derive_decision_async(self, market_state: Dict, account_state: Dict) -> ReasoningResult:
        """异步获取决策"""
        messages = self.build_messages(market_state, account_state)
        
        # 使用 Router 调用 (Smart Mode)
        # TODO: Mode should be configurable
        priority = self.router.get_model_priority("SMART")
        
        content, reasoning, model = await self.router.call_with_fallback(
            messages=messages,
            priority_models=priority
        )
        
        decision = self._parse_response(content)
        return ReasoningResult(decision, content, reasoning or "", model)
        
    def derive_decision(self, market_state: Dict, account_state: Dict) -> ReasoningResult:
        """同步获取决策 (不推荐，仅兼容)"""
        import asyncio
        # 创建新的 loop 或使用现有 loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.derive_decision_async(market_state, account_state))

    def _parse_response(self, content: str) -> Decision:
        """解析 JSON 响应"""
        try:
            # 尝试提取 JSON Block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = content
                
            data = json.loads(json_str)
            return Decision(
                action=data.get("action", "HOLD").upper(),
                quantity=int(data.get("quantity", 0)),
                price=data.get("price"),
                reason=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.5)),
                fear_level=float(data.get("fear_level", 0.0)),
                greed_level=float(data.get("greed_level", 0.0))
            )
        except Exception as e:
            # 解析失败，保持 HOLD
            return Decision(action="HOLD", quantity=0, reason=f"Parse Error: {e}")
