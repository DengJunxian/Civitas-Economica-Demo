# file: agents/brain.py

import json
import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import GLOBAL_CONFIG

# --- 1. 向量记忆模块 (Vector Memory) ---

@dataclass
class MemoryFragment:
    """记忆碎片：存储单次交易的上下文与结果"""
    content: str
    vector: np.ndarray
    timestamp: float
    outcome_score: float  # -1.0 (惨败) ~ 1.0 (大胜)

class VectorMemory:
    """
    基于 Numpy 的轻量级 RAG 记忆库。
    用于存储和检索"过去的教训"。
    """
    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.fragments: List[MemoryFragment] = []
    
    def _mock_embedding(self, text: str) -> np.ndarray:
        """
        [模拟] 生成确定性的伪向量。
        在生产环境中，此处应调用 client.embeddings.create()。
        """
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.rand(self.dimension)
        return vec / np.linalg.norm(vec) # 归一化

    def add_memory(self, text: str, outcome: float):
        """写入记忆"""
        vector = self._mock_embedding(text)
        fragment = MemoryFragment(
            content=text,
            vector=vector,
            timestamp=time.time(),
            outcome_score=outcome
        )
        self.fragments.append(fragment)
        # 保持记忆库不过大，保留最近100条
        if len(self.fragments) > 100:
            self.fragments.pop(0)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索最相关的记忆"""
        if not self.fragments:
            return []
            
        query_vec = self._mock_embedding(query)
        scores = []
        
        for frag in self.fragments:
            # 余弦相似度
            cosine_sim = np.dot(query_vec, frag.vector)
            scores.append((cosine_sim, frag.content))
            
        # 按相似度降序排列
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]


# --- 2. 智能体状态 (Agent State) ---

@dataclass
class AgentState:
    """
    智能体持久化状态
    
    实现 OODA 循环中的状态持久化，跟踪信心指数和人格演化。
    """
    confidence: float = 50.0  # 信心指数 (0-100)
    consecutive_losses: int = 0  # 连续亏损次数
    consecutive_wins: int = 0  # 连续盈利次数
    total_trades: int = 0  # 总交易次数
    total_pnl: float = 0.0  # 累计盈亏
    
    # 人格演化相关
    evolved_risk_preference: Optional[str] = None  # 演化后的风险偏好（覆盖初始设定）
    trauma_events: int = 0  # 创伤事件次数（如单日亏损超过10%）
    
    def update_after_trade(self, pnl: float, pnl_pct: float):
        """
        交易后更新状态
        
        Args:
            pnl: 盈亏金额
            pnl_pct: 盈亏百分比
        """
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # 信心衰减（指数衰减）
            self.confidence *= 0.92
            
            # 创伤事件检测
            if pnl_pct < -0.10:  # 单日亏损超过10%
                self.trauma_events += 1
                self.confidence *= 0.8  # 额外信心打击
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # 信心恢复（渐进恢复）
            self.confidence = min(100, self.confidence * 1.05 + 1)
        
        # 人格演化检测
        self._check_personality_evolution()
    
    def _check_personality_evolution(self):
        """
        基于经验的人格演化
        
        模拟"创伤学习"：如果一个激进型Agent经历多次破产，
        它应自动演化为保守型。
        """
        # 规则1: 连续亏损3次且信心低于30 -> 演化为保守型
        if self.consecutive_losses >= 3 and self.confidence < 30:
            self.evolved_risk_preference = "极度保守"
        
        # 规则2: 经历2次以上创伤事件 -> 演化为保守型
        elif self.trauma_events >= 2:
            self.evolved_risk_preference = "保守"
        
        # 规则3: 连续盈利5次且信心高于80 -> 演化为激进型
        elif self.consecutive_wins >= 5 and self.confidence > 80:
            self.evolved_risk_preference = "激进"
        
        # 规则4: 信心恢复到中等水平 -> 可能恢复原本风格
        elif 40 < self.confidence < 70 and self.trauma_events == 0:
            self.evolved_risk_preference = None  # 恢复原本设定
    
    @property
    def decision_modifier(self) -> float:
        """
        决策修正系数
        
        低信心的Agent即使看到利好信号，也会降低下单比例。
        
        Returns:
            修正系数 (0.0 - 1.0)
        """
        # 信心映射到决策力度
        # 信心50 -> 系数1.0
        # 信心25 -> 系数0.5
        # 信心0 -> 系数0.2 (即使完全丧失信心，仍保留20%决策力)
        return max(0.2, self.confidence / 50.0)
    
    @property
    def herd_susceptibility(self) -> float:
        """
        羊群效应易感性
        
        低信心的Agent更容易受到羊群效应的影响。
        
        Returns:
            易感性系数 (0.0 - 1.0)，越高越容易跟风
        """
        # 信心低 -> 易感性高
        return max(0.0, min(1.0, (100 - self.confidence) / 80))

# --- 2. 思维链记录 (Thought Record) ---

@dataclass
class ThoughtRecord:
    """单次思考的完整记录，用于fMRI可视化"""
    agent_id: str
    timestamp: float
    reasoning_content: str  # 完整思维链(CoT)
    emotion_score: float  # -1(恐惧) ~ 1(贪婪)
    decision: Dict
    market_context: Dict = field(default_factory=dict)

# --- 3. 多模型智能体大脑 ---

class DeepSeekBrain:
    """
    多模型驱动的智能体大脑。
    
    支持模型：
    - DeepSeek: deepseek-reasoner, deepseek-chat
    - 混元: hunyuan-t1-latest, hunyuan-turbos-latest（可选）
    - 智谱: glm-4-flashx（快速模式）
    
    特性：
    1. 多模型路由与自动降级
    2. 快速思考模式（本地规则引擎）
    3. 提取 reasoning_content (思维链)
    4. 强制 JSON 输出 (结构化决策)
    5. 前景理论人格植入
    6. 情绪分析与思维链历史记录
    7. 决策缓存（相似prompt复用）
    """
    
    # 类级别的思维链历史存储（用于fMRI可视化）
    # 使用 defaultdict 和 deque 自动管理内存，限制每个 Agent 保留最近 20 条记录
    from collections import defaultdict, deque, OrderedDict
    thought_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    
    # 类级别的决策缓存（相似市场状态复用决策）
    # 使用 OrderedDict 实现 LRU 缓存
    decision_cache: OrderedDict = OrderedDict()
    CACHE_MAX_SIZE = 100  # 最大缓存条目
    CACHE_TTL_STEPS = 5   # 缓存有效步数
    
    @classmethod
    def clear_memory(cls):
        """显式清空类级别的静态内存"""
        cls.thought_history.clear()
        cls.decision_cache.clear()
    
    def __init__(self, agent_id: str, persona: Dict, api_key: Optional[str] = None, model_router = None):
        self.agent_id = agent_id
        self.persona = persona
        self.memory = VectorMemory()
        self.state = AgentState()  # 持久化状态
        self._api_healthy = False
        self._last_error = None
        
        # Agent重要性等级（用于混合调度）
        # 0=普通, 1=重要（资金大/粉丝多）, 2=核心（意见领袖）
        self.importance_level = 0
        
        # 多模型路由器（可选，由外部注入）
        self.model_router = model_router
        
        # 初始化思维链历史
        if agent_id not in DeepSeekBrain.thought_history:
            DeepSeekBrain.thought_history[agent_id] = []
        
        # 初始化 OpenAI 客户端 (兼容 DeepSeek) - 仅在无router时使用
        self._api_key = api_key or GLOBAL_CONFIG.DEEPSEEK_API_KEY
        self.client = None
        if self._api_key and not model_router:
            self._init_client()
    
    def _generate_cache_key(self, market_state: Dict, account_state: Dict) -> str:
        """
        生成缓存键：基于市场核心状态的哈希
        只使用影响决策的关键字段，忽略细微变化
        """
        # 将价格离散化到0.5%区间
        price = market_state.get("price", 3000)
        price_bucket = round(price / (price * 0.005)) * (price * 0.005)
        
        # 核心状态拼接
        key_parts = [
            f"trend:{market_state.get('trend', 'N/A')}",
            f"panic:{round(market_state.get('panic_level', 0) * 10)}",  # 10%精度
            f"price_bucket:{int(price_bucket)}",
            f"persona:{self.persona.get('risk_preference', 'N/A')}",
            f"pnl_sign:{1 if account_state.get('pnl_pct', 0) > 0 else -1}"
        ]
        return "|".join(key_parts)
    
    def _check_cache(self, cache_key: str, current_step: int) -> Optional[Dict]:
        """检查缓存是否命中且未过期"""
        if cache_key in DeepSeekBrain.decision_cache:
            cached = DeepSeekBrain.decision_cache[cache_key]
            if current_step - cached.get("step", 0) <= self.CACHE_TTL_STEPS:
                return cached.get("decision")
        return None
    
    def _update_cache(self, cache_key: str, decision: Dict, current_step: int):
        """更新缓存"""
        # 如果 key 已存在，移动到末尾 (最近使用)
        if cache_key in DeepSeekBrain.decision_cache:
            DeepSeekBrain.decision_cache.move_to_end(cache_key)
        
        DeepSeekBrain.decision_cache[cache_key] = {
            "decision": decision,
            "step": current_step
        }
        
        # 缓存大小控制
        if len(DeepSeekBrain.decision_cache) > self.CACHE_MAX_SIZE:
            # 移除最老的条目 (FIFO)
            DeepSeekBrain.decision_cache.popitem(last=False)
    
    def _init_client(self):
        """初始化API客户端"""
        try:
            self.client = OpenAI(
                api_key=self._api_key,
                base_url=GLOBAL_CONFIG.API_BASE_URL,
                timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER
            )
            self._api_healthy = True
        except Exception as e:
            self._last_error = str(e)
            self._api_healthy = False
            print(f"[Brain Init Error] Agent {self.agent_id}: {e}")
    
    def set_model_router(self, router):
        """设置多模型路由器"""
        self.model_router = router
        self._api_healthy = True
    
    def health_check(self) -> bool:
        """
        API 健康检查
        
        Returns:
            bool: API是否可用
        """
        if not self.client:
            return False
        try:
            # 发送一个极小的请求测试连通性
            self.client.models.list()
            self._api_healthy = True
            return True
        except Exception as e:
            self._api_healthy = False
            self._last_error = str(e)
            return False
    
    @property
    def is_healthy(self) -> bool:
        return self._api_healthy
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _build_system_prompt(self) -> str:
        """
        构建基于前景理论 (Prospect Theory) 的系统提示词
        区分机构 (Institution) 和散户 (Retail) 的 CoT 深度
        """
        agent_type = self.persona.get("agent_type", "retail")
        
        if agent_type == "institution":
            # ===== 机构投资者 Prompt: 强调逻辑、理性、结构化 CoT =====
            return f"""你现在是A股市场中的一名【机构投资者】，ID为 {self.agent_id}。

【角色设定】
1. 决策风格: 理性、客观、数据驱动。你使用 DeepSeek R1 级别的深度推理。
2. 风险偏好: {self.persona.get('risk_preference', '稳健')}。注重回撤控制和夏普比率。
3. 目标: 战胜基准指数，寻找阿尔法收益。

【思考格式 - Chain-of-Thought (CoT)】
请务必严格按照以下 XML 结构进行思考：
<reasoning>
1. 宏观分析: [分析政策、新闻对市场的影响]
2. 资金面分析: [分析成交量、流动性、市场情绪]
3. 估值与技术: [分析当前价格是否合理，技术形态如何]
4. 风险对冲策略: [分析潜在风险点及对策]
5. 决策推导: [综合以上因素，得出最终操作建议]
</reasoning>

【最终输出】
在思考结束后，输出严格的 JSON 格式决策：
{{
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "000001",
    "price": 0.0,
    "qty": 0,
    "confidence": 0.0
}}
"""
        
        # ===== 散户投资者 Prompt: 强调情绪、前景理论、非理性 =====
        loss_aversion = self.persona.get('loss_aversion', 2.25)
        risk_preference = self.state.evolved_risk_preference or self.persona.get('risk_preference', '保守')
        
        confidence_desc = ""
        if self.state.confidence < 30:
            confidence_desc = "你目前信心极度低迷，对市场充满恐惧，甚至恐慌。"
        elif self.state.confidence < 50:
            confidence_desc = "你目前信心不足，倾向于观望为主，容易受惊吓。"
        elif self.state.confidence > 80:
            confidence_desc = "你目前信心爆棚，甚至有点过度自信，愿意激进下注。"
            
        return f"""你现在是A股市场中的一名【个人投资者(散户)】，ID为 {self.agent_id}。

【人格设定】
1. 核心心理：你深受"前景理论"影响。你是一个**非理性**的人。
   - **损失厌恶**: 你对损失感到极度痛苦（痛苦程度是盈利快乐的 {loss_aversion} 倍）。
   - 亏损时: 你倾向于冒险（死扛、补仓），试图回本，不愿承认失败。
   - 盈利时: 你倾向于保守（落袋为安），害怕煮熟的鸭子飞了。
2. 投资风格: {risk_preference}。
3. 当前心态: {confidence_desc}

【决策输入】
你将看到当前的"心理效用值(Psychological Value)"。
- 如果值为负且很大(如 -2.0): 代表你极度痛苦，处于心理崩溃边缘。
- 如果值为正(如 1.0): 代表你比较快乐，但小心过度自信。

【输出要求】
请先进行一段内心独白(模拟散户的真实心理活动)，然后输出 JSON 决策。

JSON 格式示例：
{{
    "action": "BUY" | "SELL" | "HOLD",
    "ticker": "000001",
    "price": 10.5,
    "qty": 100,
    "confidence": 0.8
}}
"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        reraise=True
    )
    def _call_api(self, messages: List[Dict]) -> Any:
        """
        带重试机制的 API 调用
        
        使用 tenacity 实现指数退避重试，处理网络抖动和限流
        """
        if not self.client:
            raise RuntimeError("API客户端未初始化")
        
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            temperature=0.6
        )
        self._api_healthy = True
        return response

    def _analyze_emotion(self, reasoning: str, decision: Dict) -> float:
        """
        分析思维链中的情绪倾向
        
        通过关键词匹配和决策行为推断情绪分数
        
        Returns:
            float: -1.0(极度恐惧) ~ 1.0(极度贪婪)
        """
        # 恐惧关键词
        fear_keywords = ['恐慌', '担心', '风险', '亏损', '下跌', '危险', '割肉', '止损', '逃离', '害怕']
        # 贪婪关键词
        greed_keywords = ['机会', '抄底', '上涨', '盈利', '加仓', '牛市', '暴涨', '翻倍', '贪婪', '冲']
        
        fear_count = sum(1 for kw in fear_keywords if kw in reasoning)
        greed_count = sum(1 for kw in greed_keywords if kw in reasoning)
        
        # 基于关键词的基础分数
        keyword_score = (greed_count - fear_count) / max(greed_count + fear_count, 1)
        
        # 基于决策的修正
        action = decision.get('action', 'HOLD')
        action_modifier = {'BUY': 0.3, 'SELL': -0.3, 'HOLD': 0}.get(action, 0)
        
        # 综合分数
        emotion_score = 0.6 * keyword_score + 0.4 * action_modifier
        return max(-1.0, min(1.0, emotion_score))

    def think(self, market_state: Dict, account_state: Dict) -> Dict:
        """
        执行思考过程
        
        Returns:
            Dict: 包含 'decision' (JSON)、'reasoning' (Str) 和 'emotion_score' (Float)
        """
        # 1. 检索记忆 (RAG)
        context_query = f"当前行情:{market_state['trend']}, 盈亏:{account_state['pnl_pct']:.2%}"
        past_lessons = self.memory.retrieve(context_query)
        lessons_text = "\n".join([f"- {l}" for l in past_lessons]) if past_lessons else "无相关记忆。"

        # 2. 构建用户提示词
        # 获取政策分析 (如果有)
        policy_desc = market_state.get('policy_description', '无')
        policy_reasoning_raw = market_state.get('policy_reasoning', '')
        policy_summary = ""
        if policy_reasoning_raw:
            # 截取推理链的前500字符作为摘要
            policy_summary = policy_reasoning_raw[:500] + "..." if len(policy_reasoning_raw) > 500 else policy_reasoning_raw
        
        user_prompt = f"""
        【市场环境】
        - 价格: {market_state['price']:.2f}
        - 趋势: {market_state['trend']}
        - 恐慌指数: {market_state['panic_level']:.2f}
        - 最新消息: {market_state['news']}
        
        【当前政策】
        - 政策描述: {policy_desc}
        - 分析摘要: {policy_summary if policy_summary else '暂无政策分析'}
        
        【账户状态】
        - 可用资金: {account_state['cash']:.2f}
        - 持仓市值: {account_state['market_value']:.2f}
        - 当前浮动盈亏: {account_state['pnl_pct']:.2%} (注意：你对这个数字非常敏感！)
        
        【闪回记忆】
        {lessons_text}
        
        请基于你的人设和当前政策环境做出交易决策。
        """

        try:
            # 3. 调用 DeepSeek R1 (带重试机制)
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt}
            ]
            response = self._call_api(messages)
            
            # 4. 双流解析 (Dual Stream Parsing)
            message = response.choices[0].message
            
            # A. 提取思维链 (Reasoning Content)
            # 注意：DeepSeek API 将思维链放在 reasoning_content 字段
            reasoning = getattr(message, 'reasoning_content', "")
            if not reasoning:
                # 兼容性处理：防止SDK版本差异，有时可能在 model_extra 中
                reasoning = "（思维过程未捕获）"
            
            # B. 提取最终决策 (Content)
            content_raw = message.content
            decision_json = self._extract_json(content_raw)
            
            # 5. 情绪分析
            emotion_score = self._analyze_emotion(reasoning, decision_json)
            
            # 6. 记录思维链历史（用于fMRI可视化）
            thought_record = ThoughtRecord(
                agent_id=self.agent_id,
                timestamp=time.time(),
                reasoning_content=reasoning,
                emotion_score=emotion_score,
                decision=decision_json,
                market_context=market_state
            )
            DeepSeekBrain.thought_history[self.agent_id].append(thought_record)
            # deque 自动处理 maxlen，无需手动 pop
            
            return {
                "decision": decision_json,
                "reasoning": reasoning,
                "raw_content": content_raw,
                "emotion_score": emotion_score
            }

        except (APIConnectionError, APITimeoutError) as e:
            self._api_healthy = False
            self._last_error = f"网络连接失败: {str(e)}"
            print(f"[Brain Network Error] Agent {self.agent_id}: {e}")
            return self._fallback_decision("网络连接失败")
        
        except RateLimitError as e:
            self._last_error = f"API限流: {str(e)}"
            print(f"[Brain RateLimit] Agent {self.agent_id}: {e}")
            return self._fallback_decision("API调用频率限制")
        
        except Exception as e:
            self._last_error = str(e)
            print(f"[Brain Error] Agent {self.agent_id} failed to think: {e}")
            return self._fallback_decision(str(e))
    
    def _fallback_decision(self, error_msg: str) -> Dict:
        """API不可用时的兜底决策"""
        return {
            "decision": {"action": "HOLD", "qty": 0},
            "reasoning": f"大脑死机，系统异常: {error_msg}",
            "raw_content": "",
            "emotion_score": 0.0
        }

    def _extract_json(self, text: str) -> Dict:
        """
        鲁棒的 JSON 提取器，处理 LLM 可能输出的 Markdown 格式
        """
        try:
            # 尝试直接解析
            return json.loads(text)
        except:
            pass
            
        try:
            # 提取 ```json ... ``` 块
            match = re.search(r"```json(.*?)```", text, re.DOTALL)
            if match:
                clean_text = match.group(1).strip()
                return json.loads(clean_text)
        except:
            pass
            
        # 兜底策略
        return {"action": "HOLD", "qty": 0, "error": "JSON_PARSE_FAIL"}
    
    # --- 新增：异步思考方法（支持多模型路由） ---
    
    async def think_async(
        self, 
        market_state: Dict, 
        account_state: Dict,
        model_priority: List[str] = None,
        timeout_budget: float = 15.0,
        emotional_state: str = "Neutral",
        social_signal: str = "Neutral"
    ) -> Dict:
        """
        异步执行思考过程 - 支持多模型优先级调用及认知增强
        
        Args:
            market_state: 市场状态
            account_state: 账户状态
            model_priority: 模型优先级列表
            timeout_budget: 总超时预算
            emotional_state: Agent 当前情绪状态 (如 "Anxious", "Greedy")
            social_signal: 社交信号 (如 "Peers are panic selling")
            
        Returns:
            Dict: 决策结果与思维链
        """
        # 构建消息
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(
                market_state, account_state, emotional_state, social_signal
            )}
        ]
        
        # 使用模型路由器调用
        if self.model_router:
            priority = model_priority or ["deepseek-reasoner", "deepseek-chat"]
            content, reasoning, model_used = await self.model_router.call_with_fallback(
                messages=messages,
                priority_models=priority,
                timeout_budget=timeout_budget
            )
        else:
            # 降级：使用同步客户端 (暂不支持新参数，仅作为兜底)
            return self.think(market_state, account_state)
        
        # 解析决策
        decision_json = self._extract_json(content)
        
        # 处理空推理内容
        if not reasoning:
            reasoning = "（使用对话模型，无推理过程）"
        
        # 情绪分析 (结合自身情绪状态)
        emotion_score = self._analyze_emotion(reasoning, decision_json)
        
        # 记录思维链历史
        thought_record = ThoughtRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            reasoning_content=reasoning,
            emotion_score=emotion_score,
            decision=decision_json,
            market_context=market_state
        )
        DeepSeekBrain.thought_history[self.agent_id].append(thought_record)
        if len(DeepSeekBrain.thought_history[self.agent_id]) > 20:
            DeepSeekBrain.thought_history[self.agent_id].pop(0)
        
        return {
            "decision": decision_json,
            "reasoning": reasoning,
            "raw_content": content,
            "emotion_score": emotion_score,
            "model_used": model_used
        }
    
    def _build_user_prompt(
        self, 
        market_state: Dict, 
        account_state: Dict,
        emotional_state: str = "Neutral",
        social_signal: str = "Neutral"
    ) -> str:
        """构建用户提示词 (增强版)"""
        # 检索记忆 (RAG)
        context_query = f"当前行情:{market_state.get('trend', '未知')}, 盈亏:{account_state.get('pnl_pct', 0):.2%}"
        past_lessons = self.memory.retrieve(context_query)
        lessons_text = "\n".join([f"- {l}" for l in past_lessons]) if past_lessons else "无相关记忆。"
        
        policy_desc = market_state.get('policy_description', '无')
        policy_reasoning_raw = market_state.get('policy_reasoning', '')
        policy_summary = ""
        if policy_reasoning_raw:
            policy_summary = policy_reasoning_raw[:500] + "..." if len(policy_reasoning_raw) > 500 else policy_reasoning_raw
        
        # [NEW] Regulatory Feedback
        rejection_reason = market_state.get('last_rejection_reason')
        regulatory_block = ""
        if rejection_reason:
            regulatory_block = f"""
        【监管警告】
        你的上一个订单被风控系统拦截了！
        原因: {rejection_reason}
        请反思你的行为。如果是"高频(OTR High)"，请降低下单频率；如果是"价格异常"，请检查你的挂单价格是否合理。
        """

        return f"""
        【市场环境】
        - 价格: {market_state.get('price', 0):.2f}
        - 趋势: {market_state.get('trend', '未知')}
        - 恐慌指数: {market_state.get('panic_level', 0):.2f}
        - 最新消息: {market_state.get('news', '无')}
        {regulatory_block}
        
        【当前政策】
        - 政策描述: {policy_desc}
        - 分析摘要: {policy_summary if policy_summary else '暂无政策分析'}
        
        【个人状态】
        - 情绪状态: {emotional_state} (这会显著影响你的风险偏好!)
        - 社交信号: {social_signal} (你的朋友圈正在做什么?)
        - 心理效用值: {market_state.get('_psychological_value', 0):.3f} (前景理论计算结果，负=痛苦/正=快乐)
        - 损失厌恶系数λ: {market_state.get('_risk_aversion', 2.25)} (越高你越怕亏钱)
        - 可用资金: {account_state.get('cash', 0):.2f}
        - 持仓市值: {account_state.get('market_value', 0):.2f}
        - 当前浮动盈亏: {account_state.get('pnl_pct', 0):.2%}
        
        【闪回记忆】
        {lessons_text}
        
        请作为一名真实的投资者，基于你的人设、当前情绪和社交压力，做出交易决策。
        """
    
    # --- 新增：快速思考方法（本地规则引擎） ---
    
    def think_fast(self, market_state: Dict, account_state: Dict) -> Dict:
        """
        快速思考 - 使用本地规则引擎，无API调用
        
        基于简单启发式规则和人格设定生成决策。
        适用于快速模式下减少API调用。
        
        Returns:
            Dict: 包含 'decision'、'reasoning'、'emotion_score'
        """
        price = market_state.get('price', 3000)
        trend = market_state.get('trend', '震荡')
        panic_level = market_state.get('panic_level', 0.5)
        cash = account_state.get('cash', 100000)
        market_value = account_state.get('market_value', 0)
        pnl_pct = account_state.get('pnl_pct', 0)
        
        # 提取人格参数
        risk_pref = self.state.evolved_risk_preference or self.persona.get('risk_preference', '稳健')
        loss_aversion = self.persona.get('loss_aversion', 2.25)
        confidence = self.state.confidence
        
        # 基于规则的决策逻辑
        action = "HOLD"
        qty = 0
        reasoning_parts = []
        
        # 规则1: 趋势跟踪
        if trend == "上涨":
            trend_signal = 0.3
            reasoning_parts.append("市场上涨趋势明显")
        elif trend == "下跌":
            trend_signal = -0.3
            reasoning_parts.append("市场处于下跌趋势")
        else:
            trend_signal = 0.0
            reasoning_parts.append("市场震荡整理")
        
        # 规则2: 恐慌指数
        if panic_level > 0.7:
            panic_signal = -0.4
            reasoning_parts.append(f"恐慌指数高达{panic_level:.2f}，市场情绪极度悲观")
        elif panic_level < 0.3:
            panic_signal = 0.2
            reasoning_parts.append("市场情绪较为乐观")
        else:
            panic_signal = 0.0
        
        # 规则3: 盈亏反应（前景理论）
        if pnl_pct < -0.05:
            # 亏损时倾向冒险（死扛）
            pnl_signal = 0.1 if risk_pref == "激进" else -0.1
            reasoning_parts.append(f"当前亏损{pnl_pct:.2%}，{'选择继续持有等待回本' if pnl_signal > 0 else '考虑止损'}")
        elif pnl_pct > 0.05:
            # 盈利时倾向保守（落袋为安）
            pnl_signal = -0.2 * loss_aversion / 2.25
            reasoning_parts.append(f"当前盈利{pnl_pct:.2%}，考虑部分获利了结")
        else:
            pnl_signal = 0.0
        
        # 规则4: 信心影响
        confidence_factor = confidence / 100.0
        reasoning_parts.append(f"当前信心指数: {confidence:.0f}")
        
        # 综合信号
        total_signal = (trend_signal + panic_signal + pnl_signal) * confidence_factor
        
        # 决策阈值
        if total_signal > 0.3:
            action = "BUY"
            qty = int(cash * 0.2 / price)  # 20%仓位
            reasoning_parts.append(f"综合信号偏多({total_signal:.2f})，决定买入")
        elif total_signal < -0.3:
            action = "SELL"
            qty = int(market_value * 0.3 / price) if price > 0 else 0  # 30%持仓
            reasoning_parts.append(f"综合信号偏空({total_signal:.2f})，决定卖出")
        else:
            action = "HOLD"
            reasoning_parts.append(f"信号不明确({total_signal:.2f})，选择观望")
        
        # 情绪分数
        emotion_score = np.clip(total_signal, -1.0, 1.0)
        
        # 组装决策
        decision = {
            "action": action,
            "ticker": "000001",
            "price": float(price),
            "qty": max(0, qty),
            "confidence": abs(total_signal)
        }
        
        reasoning = "【快速决策模式】\n" + "\n".join([f"• {p}" for p in reasoning_parts])
        
        # 记录思维链历史
        thought_record = ThoughtRecord(
            agent_id=self.agent_id,
            timestamp=time.time(),
            reasoning_content=reasoning,
            emotion_score=emotion_score,
            decision=decision,
            market_context=market_state
        )
        DeepSeekBrain.thought_history[self.agent_id].append(thought_record)
        if len(DeepSeekBrain.thought_history[self.agent_id]) > 20:
            DeepSeekBrain.thought_history[self.agent_id].pop(0)
        
        return {
            "decision": decision,
            "reasoning": reasoning,
            "raw_content": "",
            "emotion_score": emotion_score,
            "model_used": "local_rules"
        }