# file: agents/cognition/llm_brain.py
"""
DeepSeek R1 推理引擎包装器

负责:
1. 调用 DeepSeek API (兼容 OpenAI SDK)
2. 提取 reasoning_content 字段 (思维链 CoT)
3. 解析情绪状态标签
4. 结构化决策输出

CRITICAL: DeepSeek R1 的思维链存储在 response.choices[0].message.reasoning_content

作者: Civitas Economica Team
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import GLOBAL_CONFIG


# ==========================================
# 数据结构
# ==========================================

@dataclass
class EmotionalState:
    """情绪状态解析结果"""
    raw_text: str                  # 原始情绪描述
    fear_level: float = 0.0        # 恐惧程度 [0, 1]
    greed_level: float = 0.0       # 贪婪程度 [0, 1]
    confidence_level: float = 0.5  # 信心程度 [0, 1]
    dominant_emotion: str = "neutral"  # 主导情绪
    
    @property
    def emotion_score(self) -> float:
        """情绪分数：-1 (恐惧) ~ +1 (贪婪)"""
        return self.greed_level - self.fear_level


@dataclass
class Decision:
    """决策结果"""
    action: str       # "BUY" | "SELL" | "HOLD"
    ticker: str = ""
    price: float = 0.0
    quantity: int = 0
    confidence: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "ticker": self.ticker,
            "price": self.price,
            "qty": self.quantity,
            "confidence": self.confidence
        }


@dataclass
class ReasoningResult:
    """
    推理结果
    
    包含完整的思维链 (Chain of Thought) 和结构化决策。
    """
    chain_of_thought: str           # 完整思维链 (reasoning_content)
    decision: Decision              # 结构化决策
    emotional_state: EmotionalState # 情绪状态
    raw_content: str = ""           # 原始响应内容
    model_used: str = ""            # 使用的模型
    inference_time_ms: float = 0.0  # 推理耗时 (毫秒)


# ==========================================
# 系统提示词模板
# ==========================================

SYSTEM_PROMPT_TEMPLATE = """
You are a retail investor holding {symbol}. 
Your current PnL is {pnl_pct:.2%}. 
The market sentiment is {sentiment}.

Based on Prospect Theory, you feel the pain of loss {lambda_coeff}x more than gain.

【重要】请按以下格式回复：

1. 首先，在 <emotional_state> 标签内描述你的情绪状态：
   - 你现在感到恐惧还是贪婪？程度如何？
   - 这个盈亏数字让你有什么感觉？
   - 你的信心水平如何？

2. 然后，输出你的交易决策，必须是严格的 JSON 格式：
```json
{{
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "{symbol}",
    "price": <float>,
    "qty": <int>,
    "confidence": <0.0-1.0>
}}
```

示例回复：
<emotional_state>
我现在持有股票亏损了5%，心里很焦虑。虽然理性告诉我应该止损，但损失厌恶让我不愿意"确认"这笔亏损。我感到恐惧程度约60%，信心只剩40%。
</emotional_state>

```json
{{"action": "HOLD", "ticker": "000001", "price": 10.5, "qty": 0, "confidence": 0.4}}
```
"""

USER_PROMPT_TEMPLATE = """
【市场环境】
- 当前价格: {price:.2f}
- 市场趋势: {trend}
- 恐慌指数: {panic_level:.2f}
- 最新消息: {news}

【账户状态】
- 可用资金: {cash:.2f}
- 持仓市值: {market_value:.2f}
- 浮动盈亏: {pnl_pct:.2%}

【闪回记忆 - 过去的教训】
{memory_context}

请基于你的投资人格和前景理论做出决策。记住：
- 亏损时，你可能倾向于冒险（死扛或补仓）
- 盈利时，你可能倾向于保守（落袋为安）
"""


# ==========================================
# DeepSeek 推理引擎
# ==========================================

class DeepSeekReasoner:
    """
    DeepSeek R1 推理引擎
    
    核心功能:
    1. 调用 DeepSeek API (兼容 OpenAI SDK)
    2. 提取 reasoning_content 字段获取完整思维链
    3. 解析 <emotional_state> 标签
    4. 结构化决策输出
    
    CRITICAL: 
    - API Base URL: https://api.deepseek.com
    - 模型: deepseek-reasoner (R1) 或 deepseek-chat
    - 思维链在 message.reasoning_content 字段
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-reasoner",
        timeout: float = 30.0
    ):
        """
        初始化推理引擎
        
        Args:
            api_key: DeepSeek API Key，默认从环境变量读取
            base_url: API Base URL
            model: 模型名称，支持 "deepseek-reasoner" 和 "deepseek-chat"
            timeout: 请求超时时间 (秒)
        """
        self.api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        
        self.client: Optional[OpenAI] = None
        self._api_healthy = False
        self._last_error: Optional[str] = None
        
        if self.api_key:
            self._init_client()
    
    def _init_client(self) -> None:
        """初始化 OpenAI 兼容客户端"""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            self._api_healthy = True
        except Exception as e:
            self._last_error = str(e)
            self._api_healthy = False
    
    @property
    def is_healthy(self) -> bool:
        return self._api_healthy
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        reraise=True
    )
    def _call_api(self, messages: List[Dict]) -> Any:
        """
        带重试机制的 API 调用
        """
        if not self.client:
            raise RuntimeError("API 客户端未初始化")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.6
        )
        self._api_healthy = True
        return response
    
    def reason(
        self,
        market_state: Dict,
        account_state: Dict,
        symbol: str = "000001",
        lambda_coeff: float = 2.25,
        memory_context: str = ""
    ) -> ReasoningResult:
        """
        执行推理
        
        Args:
            market_state: 市场状态
            account_state: 账户状态
            symbol: 交易标的
            lambda_coeff: 损失厌恶系数
            memory_context: 记忆上下文 (RAG 检索结果)
            
        Returns:
            ReasoningResult: 完整推理结果
        """
        start_time = time.time()
        
        # 构建消息
        pnl_pct = account_state.get('pnl_pct', 0)
        sentiment = self._get_sentiment(market_state)
        
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            symbol=symbol,
            pnl_pct=pnl_pct,
            sentiment=sentiment,
            lambda_coeff=lambda_coeff
        )
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            price=market_state.get('price', 0),
            trend=market_state.get('trend', '未知'),
            panic_level=market_state.get('panic_level', 0),
            news=market_state.get('news', '无'),
            cash=account_state.get('cash', 0),
            market_value=account_state.get('market_value', 0),
            pnl_pct=pnl_pct,
            memory_context=memory_context or "无相关记忆"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self._call_api(messages)
            message = response.choices[0].message
            
            # CRITICAL: 提取思维链 (reasoning_content)
            reasoning_content = getattr(message, 'reasoning_content', '')
            if not reasoning_content:
                # 兼容性处理
                reasoning_content = "(思维过程未捕获)"
            
            # 提取内容
            content = message.content or ""
            
            # 解析情绪状态
            emotional_state = self._extract_emotional_state(content, reasoning_content)
            
            # 解析决策
            decision = self._extract_decision(content)
            
            inference_time = (time.time() - start_time) * 1000
            
            return ReasoningResult(
                chain_of_thought=reasoning_content,
                decision=decision,
                emotional_state=emotional_state,
                raw_content=content,
                model_used=self.model,
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            self._api_healthy = False
            self._last_error = str(e)
            
            # 返回默认结果
            return ReasoningResult(
                chain_of_thought=f"推理失败: {e}",
                decision=Decision(action="HOLD"),
                emotional_state=EmotionalState(raw_text="无法分析"),
                raw_content="",
                model_used=self.model,
                inference_time_ms=(time.time() - start_time) * 1000
            )
    
    def _get_sentiment(self, market_state: Dict) -> str:
        """获取市场情绪描述"""
        panic = market_state.get('panic_level', 0.5)
        trend = market_state.get('trend', '震荡')
        
        if panic > 0.7:
            return f"极度恐慌 (恐慌指数 {panic:.2f})"
        elif panic > 0.5:
            return f"偏悲观，{trend}趋势"
        elif panic < 0.3:
            return f"偏乐观，{trend}趋势"
        else:
            return f"中性，{trend}趋势"
    
    def _extract_emotional_state(
        self, 
        content: str, 
        reasoning: str
    ) -> EmotionalState:
        """
        提取情绪状态
        
        从 <emotional_state> 标签和思维链中分析情绪。
        """
        # 尝试提取 <emotional_state> 标签
        match = re.search(
            r'<emotional_state>(.*?)</emotional_state>',
            content,
            re.DOTALL | re.IGNORECASE
        )
        
        raw_text = match.group(1).strip() if match else ""
        
        # 如果没有标签，从思维链中提取
        if not raw_text and reasoning:
            raw_text = reasoning[:500]  # 取前500字符
        
        # 关键词分析
        fear_keywords = ['恐惧', '害怕', '担心', '焦虑', '恐慌', '不安', '亏损', '风险']
        greed_keywords = ['贪婪', '兴奋', '机会', '盈利', '看涨', '信心', '乐观', '冲']
        
        text_to_analyze = raw_text + reasoning
        
        fear_count = sum(1 for kw in fear_keywords if kw in text_to_analyze)
        greed_count = sum(1 for kw in greed_keywords if kw in text_to_analyze)
        
        total = fear_count + greed_count + 1
        fear_level = fear_count / total
        greed_level = greed_count / total
        
        # 确定主导情绪
        if fear_level > greed_level + 0.2:
            dominant = "fear"
        elif greed_level > fear_level + 0.2:
            dominant = "greed"
        else:
            dominant = "neutral"
        
        # 尝试提取数字化的信心水平
        confidence_match = re.search(r'信心[仅只]?[剩有]?(\d+)[%％]?', text_to_analyze)
        if confidence_match:
            confidence = int(confidence_match.group(1)) / 100
        else:
            confidence = 0.5 + (greed_level - fear_level) * 0.3
        
        return EmotionalState(
            raw_text=raw_text,
            fear_level=min(1.0, fear_level),
            greed_level=min(1.0, greed_level),
            confidence_level=max(0.0, min(1.0, confidence)),
            dominant_emotion=dominant
        )
    
    def _extract_decision(self, content: str) -> Decision:
        """
        提取结构化决策
        """
        # 尝试直接解析 JSON
        try:
            data = json.loads(content)
            return Decision(
                action=data.get('action', 'HOLD').upper(),
                ticker=data.get('ticker', ''),
                price=float(data.get('price', 0)),
                quantity=int(data.get('qty', 0)),
                confidence=float(data.get('confidence', 0.5))
            )
        except:
            pass
        
        # 提取 ```json ``` 块
        try:
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                data = json.loads(match.group(1).strip())
                return Decision(
                    action=data.get('action', 'HOLD').upper(),
                    ticker=data.get('ticker', ''),
                    price=float(data.get('price', 0)),
                    quantity=int(data.get('qty', 0)),
                    confidence=float(data.get('confidence', 0.5))
                )
        except:
            pass
        
        # 兜底：尝试从文本中提取动作
        content_upper = content.upper()
        if 'BUY' in content_upper:
            return Decision(action='BUY')
        elif 'SELL' in content_upper:
            return Decision(action='SELL')
        
        return Decision(action='HOLD')


# ==========================================
# 轻量级本地推理器 (无 API 调用)
# ==========================================

class LocalReasoner:
    """
    本地规则引擎推理器
    
    用于 API 不可用时的兜底，或快速模式下减少 API 调用。
    """
    
    def reason(
        self,
        market_state: Dict,
        account_state: Dict,
        prospect_value: float = 0.0,
        **kwargs
    ) -> ReasoningResult:
        """
        基于规则的本地推理
        """
        pnl_pct = account_state.get('pnl_pct', 0)
        trend = market_state.get('trend', '震荡')
        panic = market_state.get('panic_level', 0.5)
        
        # 简单规则引擎
        reasoning_parts = []
        action = "HOLD"
        confidence = 0.5
        
        # 规则 1: 趋势
        if trend == "上涨":
            reasoning_parts.append("市场上涨趋势，偏多")
            action = "BUY"
            confidence += 0.1
        elif trend == "下跌":
            reasoning_parts.append("市场下跌趋势，偏空")
            action = "SELL"
            confidence += 0.1
        
        # 规则 2: 恐慌指数
        if panic > 0.7:
            reasoning_parts.append(f"恐慌指数 {panic:.2f} 过高，保持谨慎")
            if action == "BUY":
                action = "HOLD"
        
        # 规则 3: 前景理论效用
        if prospect_value < -0.5:
            reasoning_parts.append(f"前景值 {prospect_value:.3f} 极低，恐惧占主导")
            action = "HOLD"
        
        # 构建情绪状态
        emotional_state = EmotionalState(
            raw_text="本地规则引擎分析",
            fear_level=panic,
            greed_level=max(0, 1 - panic),
            confidence_level=confidence,
            dominant_emotion="fear" if panic > 0.5 else "neutral"
        )
        
        return ReasoningResult(
            chain_of_thought="\n".join(reasoning_parts),
            decision=Decision(action=action, confidence=confidence),
            emotional_state=emotional_state,
            model_used="local_rules"
        )


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek R1 推理引擎测试")
    print("=" * 60)
    
    # 使用本地推理器测试 (无 API 调用)
    local = LocalReasoner()
    
    market = {
        "price": 3000.0,
        "trend": "下跌",
        "panic_level": 0.75,
        "news": "美联储加息预期升温"
    }
    
    account = {
        "cash": 50000.0,
        "market_value": 50000.0,
        "pnl_pct": -0.08
    }
    
    result = local.reason(market, account, prospect_value=-0.6)
    
    print(f"\n[本地推理结果]")
    print(f"  决策: {result.decision.action}")
    print(f"  信心: {result.decision.confidence:.2f}")
    print(f"  情绪: fear={result.emotional_state.fear_level:.2f}, greed={result.emotional_state.greed_level:.2f}")
    print(f"  思维链: {result.chain_of_thought}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
