# file: agents/trader_agent.py
"""
TraderAgent 实现 — 基于认知闭环的交易智能体

实现 BaseAgent 定义的 Perceive-Reason-Decide-Act 闭环。
核心特性:
1. 心理画像驱动: 风险厌恶、自信程度、关注广度等个性化参数
2. 模拟 DeepSeek R1 推理: 生成类人思维链 (Chain of Thought)
3. 结构化决策: 输出标准 Limit Order
4. 快慢思考双层架构 (System 1/2): 节约算力，模拟直觉与深思

作者: Civitas Economica Team
"""

import asyncio
import json
import os
import random
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from agents.base_agent import BaseAgent, MarketSnapshot
from agents.brain import DeepSeekBrain
from core.types import Order, OrderSide, OrderType, OrderStatus
from core.behavioral_finance import predict_next_return
from agents.persona import Persona, RiskAppetite
from core.society.network import SocialGraph, SentimentState
from agents.cognition.graph_storage import GraphMemoryBank
from agents.cognition.graph_builder import GraphExtractor
from agents.roles.news_analyst import NewsAnalyst
from agents.roles.quant_analyst import QuantAnalyst
from agents.roles.risk_analyst import RiskAnalyst

class TraderAgent(BaseAgent):
    """
    TraderAgent — 具备完整认知闭环的交易智能体
    """

    def __init__(
        self,
        agent_id: str,
        cash_balance: float = 100_000.0,
        portfolio: Optional[Dict[str, int]] = None,
        psychology_profile: Optional[Dict[str, float]] = None,
        model_router: Optional[Any] = None, # Support Router
        persona: Optional[Persona] = None,
        use_llm: bool = True,
        model_priority: Optional[List[str]] = None,
        news_config: Optional[Dict[str, Any]] = None,
        news_max_articles: int = 5,
        ram_cooldown_steps: int = 3
    ):
        super().__init__(agent_id, cash_balance, portfolio, psychology_profile)
        
        self.use_llm = use_llm
        
        # 人格画像集成
        self.persona = persona if persona else Persona(name=agent_id)
        
        # 社交网络集成
        self.social_node_id: Optional[int] = None
        self.social_graph: Optional[SocialGraph] = None
        
        # Map Persona to legacy psychology profile for compatibility
        self.profile = {
            "risk_aversion": self._map_persona_to_risk_aversion(),
            "confidence_level": 0.5 + (self.persona.overconfidence * 0.5),
            "attention_span": 3 + int(self.persona.patience * 2),
            "loss_sensitivity": self.persona.loss_aversion,
        }
        if psychology_profile:
            self.profile.update(psychology_profile)

        # Initialize Brain
        if agent_id.startswith("Debate_"):
            from agents.debate_brain import DebateBrain
            self.brain = DebateBrain(
                agent_id=agent_id,
                persona={
                    "risk_preference": self.persona.risk_appetite.value,
                    "loss_aversion": self.profile.get("loss_sensitivity", 1.5)
                },
                api_key=None
            )
            if model_router is not None:
                self.brain.set_model_router(model_router)
        else:
            self.brain = DeepSeekBrain(
                agent_id=agent_id,
                persona={
                    "risk_preference": self.persona.risk_appetite.value,
                    "loss_aversion": self.profile.get("loss_sensitivity", 1.5)
                },
                model_router=model_router
            )
        # 注入模型优先级 (如果 Brain 支持)
        if hasattr(self.brain, 'model_priority'):
            self.brain.model_priority = model_priority
        
        # 合规反馈记忆（存储被风控拒绝的原因）
        self.compliance_feedback: List[str] = []
        
        # 快/慢思考模式状态跟踪
        self._last_news_count = 0
        self._last_social_sentiment = "neutral"
        self._fast_mode_consecutive_steps = 0
        
        # GraphRAG 记忆层
        self.graph_memory = GraphMemoryBank(agent_id=self.agent_id)
        self.graph_extractor = GraphExtractor(model_router=self.brain.model_router if hasattr(self, 'brain') else model_router)
        
        # Sync initial confidence
        self.brain.state.confidence = self.profile.get("confidence_level", 0.5) * 100

        # === Phase 1: Analyst Roles ===
        self.news_analyst = NewsAnalyst(config_paths=news_config, max_articles=news_max_articles)
        self.quant_analyst = QuantAnalyst()
        self.risk_analyst = RiskAnalyst()

        # === Risk Alert Meeting (RAM) State ===
        self._portfolio_value_history: List[float] = []
        self._last_cvar: Optional[float] = None
        self._ram_until_step: int = 0
        self._ram_last_trigger: str = ""
        self._ram_cooldown_steps: int = max(1, int(ram_cooldown_steps))

        # Phase 4: Wind-Tunnel + Beliefs
        self._price_history: List[float] = []
        self._wind_tunnel_confidence: float = 0.5
        self._wind_tunnel_records: List[Dict[str, Any]] = []
        self._recent_trade_outcomes: List[Dict[str, Any]] = []
        self._last_belief_reflection_step: int = 0

    def _map_persona_to_risk_aversion(self) -> float:
        mapping = {
            RiskAppetite.CONSERVATIVE: 0.8,
            RiskAppetite.BALANCED: 0.5,
            RiskAppetite.AGGRESSIVE: 0.3,
            RiskAppetite.GAMBLER: 0.1
        }
        return mapping.get(self.persona.risk_appetite, 0.5)

    def bind_social_node(self, node_id: int, graph: SocialGraph):
        """Bind this agent to a node in the Social Graph."""
        self.social_node_id = node_id
        self.social_graph = graph
        # Sync agent ID to graph node
        if node_id in graph.agents:
            graph.agents[node_id].agent_id = self.agent_id
        # 初次绑定时立即同步一次语义画像，确保传播引擎有可用语义向量。
        self.sync_social_semantic_profile()

    def _risk_tilt_from_persona(self) -> float:
        """将离散风险偏好映射到连续风偏分数 [-1, 1]。"""
        mapping = {
            RiskAppetite.CONSERVATIVE: -0.8,
            RiskAppetite.BALANCED: 0.0,
            RiskAppetite.AGGRESSIVE: 0.6,
            RiskAppetite.GAMBLER: 0.9,
        }
        return float(mapping.get(self.persona.risk_appetite, 0.0))

    def _default_focus_topics(self) -> List[str]:
        """
        基于人格生成默认关注主题。
        用于 GraphRAG 尚未形成稳定子图时的冷启动推荐画像。
        """
        horizon_map = {
            "short-term": ["波动", "成交量", "情绪", "热点"],
            "medium-term": ["政策", "流动性", "盈利", "估值"],
            "long-term": ["基本面", "产业趋势", "财政", "利率"],
        }
        risk_map = {
            RiskAppetite.CONSERVATIVE: ["防御", "分红", "稳增长"],
            RiskAppetite.BALANCED: ["均衡配置", "政策", "景气"],
            RiskAppetite.AGGRESSIVE: ["成长", "科技", "弹性"],
            RiskAppetite.GAMBLER: ["题材", "高波动", "杠杆"],
        }

        horizon_key = str(self.persona.investment_horizon.value).lower()
        topics = horizon_map.get(horizon_key, ["政策", "风险"])
        topics = topics + risk_map.get(self.persona.risk_appetite, ["政策", "风险"])
        return list(dict.fromkeys(topics))

    def sync_social_semantic_profile(self) -> None:
        """
        将 TraderAgent 的 GraphRAG 主导叙事同步到社交图节点。

        该方法是“语义驱动 SIRS”的数据入口：
        - dominant_narratives: 取自 GraphMemoryBank
        - focus_topics: 人格冷启动 + 近期叙事融合
        - risk_tilt / historical_risk_bias: 用于 RecSys Hot Score 的风偏匹配
        """
        if self.social_graph is None or self.social_node_id is None:
            return

        dominant = self.graph_memory.get_dominant_narratives(top_k=6)
        default_focus = self._default_focus_topics()
        focus_topics = list(dict.fromkeys(default_focus + dominant[:3]))

        # 将脑状态中的信心和情绪轻量映射为历史风险偏置，模拟“越亏越保守/越赚越激进”。
        confidence = float(getattr(self.brain.state, "confidence", 50.0))
        confidence_bias = np.clip((confidence - 50.0) / 50.0, -1.0, 1.0)

        self.social_graph.update_semantic_profile(
            self.social_node_id,
            dominant_narratives=dominant if dominant else default_focus[:3],
            focus_topics=focus_topics,
            risk_tilt=self._risk_tilt_from_persona(),
            historical_risk_bias=float(confidence_bias) * 0.3,
        )
        self.social_graph.update_holdings(self.social_node_id, self.portfolio)
        self.social_graph.update_bdi_profile(
            self.social_node_id,
            beliefs=dominant if dominant else default_focus[:3],
            desires=focus_topics,
            intentions=[getattr(self, "emotional_state", "Neutral")],
        )

    async def perceive(
        self,
        market_snapshot: MarketSnapshot,
        public_news: List[str],
    ) -> Dict[str, Any]:
        """
        感知阶段: 过滤信息
        """
        # 1. 过滤新闻 (基于 attention_span)
        span = int(self.profile.get("attention_span", 3))
        observed_news = public_news[:span] if public_news else []
        
        portfolio_value = self.get_total_value({market_snapshot.symbol: market_snapshot.last_price})
        initial = getattr(self, 'initial_cash', 100000)
        pnl = portfolio_value - initial
        pnl_pct = pnl / initial if initial > 0 else 0

        if portfolio_value:
            self._portfolio_value_history.append(portfolio_value)
            if len(self._portfolio_value_history) > 200:
                self._portfolio_value_history = self._portfolio_value_history[-200:]

        self._price_history.append(market_snapshot.last_price)
        if len(self._price_history) > 200:
            self._price_history = self._price_history[-200:]

        return {
            "snapshot": market_snapshot,
            "news": observed_news,
            "portfolio_value": portfolio_value,
            "cash": self.cash_balance,
            "pnl_pct": pnl_pct,
            "timestamp": market_snapshot.timestamp
        }

    async def reason(
        self,
        perceived_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        推理阶段: 调用 DeepSeek Brain (Enhanced with Emotion & Social)
        """
        # 计算当前情绪状态
        # Phase 1: Analyst Reports
        analyst_reports = self._collect_analyst_reports(perceived_data)
        perceived_data["analyst_reports"] = analyst_reports

        self.update_emotional_state(perceived_data)

        # RAM 触发判定（先于常规决策）
        self._update_ram_state(
            analyst_reports.get("risk", {}),
            getattr(self, "emotional_state", "Neutral")
        )
        
        # 获取社交信号
        social_signal = self.perceive_social_signal(perceived_data)
        
        # 调用增强版推理方法 (reason_and_act logic integrated here)
        return await self.reason_and_act(perceived_data, self.emotional_state, social_signal)

    def _needs_deep_thinking(self, perceived_data: Dict[str, Any], social_signal: str) -> bool:
        """
        判断是否需要触发慢思考 (System 2 / LLM)
        
        触发条件:
        1. 突发新闻: 新闻数量增加
        2. 盈亏剧烈: 触及止损/止盈阈值 (效用极值)
        3. 社交逆转: 社交信号发生方向性改变
        4. 强制刷新: 连续 N 步快思考后强制刷新一次
        """
        # Condition 1: New News
        news = perceived_data.get("news", [])
        if len(news) > self._last_news_count:
            self._last_news_count = len(news)
            return True
            
        # Condition 2: PnL Extremes (Use Prospect Theory utility approximation)
        pnl_pct = perceived_data.get("pnl_pct", 0)
        # 简化版前景效用: 亏损更敏感
        utility = pnl_pct if pnl_pct > 0 else pnl_pct * 2.25
        if utility < -0.15 or utility > 0.10: # 大亏或大赚
            # 但如果已经在同一状态很久，可能不需要每步都想？
            # 暂时简化为：极端情况始终深思
            return True
            
        # Condition 3: Social Signal Reversal
        # Parse social signal simple sentiment
        current_social = "neutral"
        if "Panic" in social_signal or "selling" in social_signal: current_social = "bearish"
        elif "buy" in social_signal or "FOMO" in social_signal: current_social = "bullish"
        
        if current_social != self._last_social_sentiment:
            self._last_social_sentiment = current_social
            return True
            
        # Condition 4: Forced Refresh (every 10 steps to prevent zombies)
        if self._fast_mode_consecutive_steps > 10:
            return True
            
        return False

    def _fast_think(
        self, 
        perceived_data: Dict[str, Any],
        emotional_state: str,
        social_signal: str
    ) -> Dict[str, Any]:
        """
        快思考 (System 1): 基于规则和启发式的快速反应
        不调用 LLM，极其高效。
        """
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        trend_val = getattr(snapshot, "market_trend", 0)
        pnl_pct = perceived_data.get("pnl_pct", 0)
        
        action = "HOLD"
        qty = 0
        price = 0.0
        
        last_price = snapshot.last_price
        
        # Heuristic 1: Trend Following (Momentum)
        if trend_val > 0.03: # Strong Up
             if self.cash_balance > last_price * 100:
                 action = "BUY"
                 # Buy small amount (10% of cash)
                 qty = int((self.cash_balance * 0.1) / last_price) // 100 * 100
                 price = round(last_price * 1.02, 2) # Aggressive
        elif trend_val < -0.03: # Strong Down
             holding = self.portfolio.get(snapshot.symbol, 0)
             if holding > 0:
                 action = "SELL"
                 # Sell 20% of holding
                 qty = int(holding * 0.2) // 100 * 100
                 price = round(last_price * 0.98, 2) # Aggressive
                 
        # Heuristic 2: Panic Selling (Override)
        if "Panic" in social_signal or emotional_state == "Fearful":
             holding = self.portfolio.get(snapshot.symbol, 0)
             if holding > 0:
                 action = "SELL"
                 qty = int(holding * 0.5) // 100 * 100 # Dump 50%
                 if qty == 0 and holding > 0: qty = holding # Dump all if small
                 price = round(last_price * 0.95, 2) # Panic price
        
        reasoning = f"(System 1) Fast response. Trend={trend_val:.3f}, Emotion={emotional_state}"
        
        return {
            "decision": {
                "action": action,
                "qty": qty,
                "price": price
            },
            "reasoning": reasoning,
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp
        }

    async def _async_extract_and_store(self, text: str, current_time: float, is_news: bool = False):
        """后台异步抽取图谱节点，避免阻塞主交易循环"""
        triplets = await self.graph_extractor.extract_graph(text)
        if not triplets:
            return
            
        topics = set()
        for t in triplets:
            self.graph_memory.add_triplet(t["subject"], t["predicate"], t["target"], t["weight"])
            topics.add(t["subject"])
            
        # 如果是宏观新闻，生成短期胶囊缓存
        if is_news and topics:
            capsule_topic = list(topics)[0] # 取主要概念作为 topic
            summary = f"[{capsule_topic}] 相关动态已发生: {text[:50]}..."
            self.graph_memory.add_capsule(capsule_topic, summary, current_time, ttl_seconds=3600)

    def _collect_analyst_reports(self, perceived_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集三名分析师的 JSON 输出，并对失败调用进行容错降级。
        该机制对应 FINCON/QuantAgents 的“角色解耦 + 分工协作”范式。
        """
        reports: Dict[str, Any] = {}
        snapshot: MarketSnapshot = perceived_data.get("snapshot")

        try:
            reports["news"] = self.news_analyst.analyze()
        except Exception as exc:
            reports["news"] = {
                "analyst": "NewsAnalyst",
                "status": "error",
                "error": str(exc),
                "events": [],
                "sentiment_score": 0.0,
                "sentiment_label": "中性",
            }

        try:
            reports["quant"] = self.quant_analyst.analyze(snapshot)
        except Exception as exc:
            reports["quant"] = {
                "analyst": "QuantAnalyst",
                "status": "error",
                "error": str(exc),
                "csad": 0.0,
                "momentum": 0.0,
                "herding_intensity": 0.0,
            }

        try:
            reports["risk"] = self.risk_analyst.analyze(self._portfolio_value_history)
        except Exception as exc:
            reports["risk"] = {
                "analyst": "RiskAnalyst",
                "status": "error",
                "error": str(exc),
                "cvar": 0.0,
                "max_drawdown": 0.0,
            }

        return reports

    def _update_ram_state(self, risk_report: Dict[str, Any], emotional_state: str) -> None:
        """
        更新 Risk Alert Meeting (RAM) 状态。
        触发条件：CVaR 显著恶化或情绪进入 Fearful。
        对应 FINCON 风控会议机制：在情节内强制避险。
        """
        cvar = float(risk_report.get("cvar", 0.0) or 0.0)
        cvar_drop = None
        trigger_reason = ""

        if self._last_cvar is not None:
            cvar_drop = cvar - self._last_cvar
            if cvar_drop < -abs(self._last_cvar) * 0.5 and cvar < 0:
                trigger_reason = f"CVaR 陡降: {self._last_cvar:.4f} -> {cvar:.4f}"

        if emotional_state == "Fearful":
            trigger_reason = trigger_reason or "情绪 Fearful 触发风控会议"

        if trigger_reason:
            self._ram_until_step = max(self._ram_until_step, self._step_count + self._ram_cooldown_steps)
            self._ram_last_trigger = trigger_reason

        self._last_cvar = cvar
        if cvar_drop is not None:
            risk_report["cvar_drop"] = cvar_drop
        risk_report["ram_active"] = self._ram_until_step >= self._step_count
        risk_report["ram_trigger_reason"] = self._ram_last_trigger

    def _beliefs_file_path(self) -> str:
        base_dir = os.path.join("data", "beliefs")
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"beliefs_{self.agent_id}.json")

    def _persist_beliefs(self, new_beliefs: List[str]) -> None:
        """将交易信仰持久化写入文件，供 System Prompt 长期强化。"""
        path = self._beliefs_file_path()
        beliefs = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    beliefs = data.get("beliefs", [])
            except Exception:
                beliefs = []
        for b in new_beliefs:
            b = str(b).strip()
            if b and b not in beliefs:
                beliefs.append(b)
        beliefs = beliefs[-10:]
        payload = {
            "agent_id": self.agent_id,
            "updated_at": time.time(),
            "beliefs": beliefs,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _reflect_and_persist_beliefs(self) -> None:
        """
        文本梯度下降：让 LLM 从连续盈亏中提炼抽象交易信仰，并写入系统提示。
        对应 FINCON/QuantAgents 中的“概念化语言强化”。
        """
        if not self._recent_trade_outcomes:
            return
        if self._step_count - self._last_belief_reflection_step < 5:
            return

        recent = self._recent_trade_outcomes[-8:]
        prompt = {
            "recent_trades": recent,
            "instruction": "提炼3条高度抽象的交易信仰(Investment Beliefs)，输出JSON数组"
        }
        messages = [
            {"role": "system", "content": "你是资深量化与行为金融分析师。只输出JSON数组，不要包含其他文本。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
        ]

        beliefs: List[str] = []
        try:
            if self.brain.model_router:
                content, _, _ = self.brain.model_router.sync_call_with_fallback(
                    messages=messages,
                    priority_models=["deepseek-chat", "glm-4-flashx"],
                    timeout_budget=20.0
                )
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    beliefs = [str(x) for x in parsed if str(x).strip()]
        except Exception:
            beliefs = []

        if not beliefs:
            # 兜底规则
            if self.brain.state.consecutive_losses >= 3:
                beliefs = ["震荡市中高频追涨导致摩擦成本过高，应降低交易频率"]
            elif self.brain.state.consecutive_wins >= 3:
                beliefs = ["趋势市中顺势而为可提高胜率，但需控制回撤"]
            else:
                beliefs = ["在高不确定性环境中保持仓位纪律优于频繁试错"]

        self._persist_beliefs(beliefs)
        self._last_belief_reflection_step = self._step_count

    def _wind_tunnel_predict(self) -> Dict[str, Any]:
        """
        内部模拟风洞：基于历史价格预测次日走势，作为下单前的拦截信号。
        """
        pred_return = predict_next_return(self._price_history)
        if pred_return is None:
            return {"valid": False}
        return {
            "valid": True,
            "predicted_return": pred_return,
            "confidence": self._wind_tunnel_confidence
        }

    async def reason_and_act(
        self,
        perceived_data: Dict[str, Any],
        emotional_state: str,
        social_signal: str
    ) -> Dict[str, Any]:
        """
        核心认知方法：结合市场感知、情绪、社交信号进行推理决策
        此方法对应 User Request Prompt 3 中的 "reason_and_act"
        
        采用快慢思考双层架构 (System 1/2)
        """
        # 0. Risk Alert Meeting (RAM) override
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        if self._ram_until_step >= self._step_count:
            holding = self.portfolio.get(snapshot.symbol, 0)
            if holding > 0:
                return {
                    "decision": {
                        "action": "SELL",
                        "qty": holding,
                        "price": round(snapshot.last_price * 0.98, 2)
                    },
                    "reasoning": f"[RAM] 风控会议触发，强制避险清仓。原因：{self._ram_last_trigger}",
                    "symbol": snapshot.symbol,
                    "timestamp": snapshot.timestamp,
                    "analyst_reports": perceived_data.get("analyst_reports", {}),
                    "risk_mode": "RAM"
                }
            return {
                "decision": {"action": "HOLD"},
                "reasoning": f"[RAM] 风控会议触发，当前无持仓，保持观望。原因：{self._ram_last_trigger}",
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp,
                "analyst_reports": perceived_data.get("analyst_reports", {}),
                "risk_mode": "RAM"
            }

        # 1. System 1 (Rule-based) Enforcement
        # 如果被标记为非 LLM Agent，强制使用快思考 (模拟计算模式)
        if not self.use_llm:
            return self._fast_think(perceived_data, emotional_state, social_signal)

        # 0.5. System 1 vs System 2 Check (Standard)
        if not self._needs_deep_thinking(perceived_data, social_signal):
            self._fast_mode_consecutive_steps += 1
            return self._fast_think(perceived_data, emotional_state, social_signal)
            
        # Trigger System 2 (Slow Thinking)
        self._fast_mode_consecutive_steps = 0
        
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        news = perceived_data["news"]

        # 每轮推理前同步一次语义画像，使扩散模型能实时读取 GraphRAG 最新主导叙事。
        self.sync_social_semantic_profile()
        
        # 1. 检索 GraphRAG 记忆与知识胶囊
        current_time = snapshot.timestamp
        graph_context = ""
        capsules = self.graph_memory.get_valid_capsules(current_time)
        extracted_keywords = []
        news_text = "; ".join(news) if news else ""
        
        if news_text:
            # 简单的关键词提取逻辑用于查图
            if "利率" in news_text or "降准" in news_text: extracted_keywords.append("利率")
            if "流动性" in news_text: extracted_keywords.append("流动性")
            if "政策" in news_text: extracted_keywords.append("政策")
            if not extracted_keywords: extracted_keywords = ["市场", "风险"]
        
        if capsules:
            graph_context = "【知识胶囊(宏观共识缓存)】\n" + "\n".join(capsules)
        elif extracted_keywords:
            subgraph = self.graph_memory.retrieve_subgraph(extracted_keywords, depth=2)
            if subgraph:
                graph_context = "【私有认知图谱关联】\n" + subgraph
                
        # 如果有新消息且没有命中缓存，后台启动提取以充实图谱
        if news_text and not capsules:
            asyncio.create_task(self._async_extract_and_store(news_text, current_time, is_news=True))
            
        # 2. Construct Market State
        market_state = {
            "price": snapshot.last_price,
            "trend": getattr(snapshot, "market_trend", "震荡"), 
            "panic_level": getattr(snapshot, "panic_level", 0.5), 
            "news": news_text if news_text else "无重大新闻",
            "last_rejection_reason": self.compliance_feedback[-1] if self.compliance_feedback else None,
            "policy_description": getattr(snapshot, "policy_description", ""),
            "policy_tax_rate": getattr(snapshot, "policy_tax_rate", 0.0),
            "policy_news": getattr(snapshot, "policy_news", ""),
            "graph_context": graph_context,
            "analyst_reports": perceived_data.get("analyst_reports", {})
        }
        
        # Map numeric trend to string for Brain
        trend_val = snapshot.market_trend
        if trend_val > 0.02: market_state["trend"] = "上涨"
        elif trend_val < -0.02: market_state["trend"] = "下跌"
        else: market_state["trend"] = "震荡"

        # 2. Construct Account State
        account_state = {
            "cash": perceived_data["cash"],
            "market_value": perceived_data["portfolio_value"] - perceived_data["cash"],
            "pnl_pct": perceived_data["pnl_pct"]
        }

        # 3. Call Brain with Enhanced Context
        try:
            decision_output = await self.brain.think_async(
                market_state=market_state, 
                account_state=account_state,
                emotional_state=emotional_state,
                social_signal=social_signal
            )
            
            # Inject symbol/timestamp for decide phase
            decision_output["symbol"] = snapshot.symbol
            decision_output["timestamp"] = snapshot.timestamp
            
            return decision_output
            
        except Exception as e:
            print(f"Agent {self.agent_id} Brain Error: {e}")
            return {
                "decision": {"action": "HOLD"},
                "reasoning": f"Brain Failure: {e}",
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp
            }

    def update_emotional_state(self, perceived_data: Dict[str, Any]):
        """根据 PnL 和 市场波动更新情绪 (受 Persona 影响)"""
        pnl_pct = perceived_data.get("pnl_pct", 0)
        snapshot = perceived_data.get("snapshot")
        panic_level = getattr(snapshot, "panic_level", 0)
        
        # Adjust sensitivity based on Persona
        loss_threshold = -0.10 / (self.persona.loss_aversion / 2.25) # More averse -> smaller threshold (easier to regret)
        panic_threshold = 0.6 * self.persona.patience # More patient -> higher threshold
        
        if pnl_pct < loss_threshold:
            self.emotional_state = "Regretful" # 悔恨 (大亏)
        elif pnl_pct < loss_threshold / 2:
            self.emotional_state = "Anxious"   # 焦虑 (小亏)
        elif pnl_pct > 0.10:
            self.emotional_state = "Greedy"    # 贪婪 (大赚)
        elif pnl_pct > 0.05:
            self.emotional_state = "Confident" # 自信 (小赚)
        elif panic_level > panic_threshold:
            self.emotional_state = "Fearful"   # 恐惧 (市场恐慌)
        else:
            self.emotional_state = "Neutral"   # 中性
            
    def perceive_social_signal(self, perceived_data: Dict[str, Any]) -> str:
        """
        感知社交信号 (集成 SocialNetwork)
        """
        # 1. 优先使用真实社交网络
        if self.social_graph and self.social_node_id is not None:
            bearish_ratio = self.social_graph.get_bearish_ratio(self.social_node_id)
            
            # Thresholds based on conformity
            panic_threshold = 1.0 - (self.persona.conformity * 0.8) # High conformity -> Low threshold (0.2)
            
            if bearish_ratio > panic_threshold:
                return f"Panic Alert! {bearish_ratio:.0%} of neighbors are selling!"
            elif bearish_ratio > panic_threshold * 0.6:
                return "Neighbors are getting nervous."
                
        # 2. 只有在没有网络连接时，才回退到市场趋势代理
        snapshot = perceived_data.get("snapshot")
        trend = getattr(snapshot, "market_trend", 0)
        
        if trend > 0.05:
            return "Everyone is buying! (FOMO)"
        elif trend < -0.05:
            return "Everyone is panic selling!"
        else:
            return "Market is quiet."

    async def decide(
        self,
        reasoning_output: Dict[str, Any],
    ) -> Optional[Order]:
        """
        决策阶段: 将 Brain 的输出转化为 Order 对象
        """
        decision = reasoning_output.get("decision", {})
        action = decision.get("action", "HOLD")
        
        if action == "HOLD" or not action:
            return None
            
        qty = decision.get("qty", 0)
        # Brain might return JSON with "price" or "limit_price"
        price = decision.get("price", 0.0)
        
        symbol = reasoning_output.get("symbol", "UNKNOWN")
        timestamp = reasoning_output.get("timestamp", time.time())
        side = None

        if action == "BUY":
            side = OrderSide.BUY
        elif action == "SELL":
            side = OrderSide.SELL
        else:
            return None
            
        if qty <= 0:
            return None

        # Phase 4: Wind-Tunnel 拦截
        wind_report = self._wind_tunnel_predict()
        decision["wind_tunnel"] = wind_report
        if wind_report.get("valid"):
            pred = float(wind_report.get("predicted_return", 0.0))
            conf = float(wind_report.get("confidence", 0.5))
            if action == "BUY" and pred < -0.005 and conf > 0.6:
                return None
            if action == "SELL" and pred > 0.005 and conf > 0.6:
                return None
            
        # 资金/持仓 预检查 (虽然 OrderBook 也会查，但 Agent 应有自我认知)
        if side == OrderSide.BUY:
            cost = price * qty
            if cost > self.cash_balance:
                # 资金不足，尝试调整
                if price > 0:
                    qty = int(self.cash_balance / price) // 100 * 100
                else:
                    qty = 0
                if qty <= 0:
                    return None
        elif side == OrderSide.SELL:
            holding = self.portfolio.get(symbol, 0)
            if qty > holding:
                qty = holding
            if qty <= 0:
                return None
        
        # 生成 Order
        order = Order(
            symbol=symbol,
            price=price,
            quantity=qty,
            side=side,
            order_type=OrderType.LIMIT,
            agent_id=self.agent_id,
            timestamp=timestamp
        )
        return order

    async def update_memory(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        """
        更新记忆
        """
        # Updates both Vector Memory (Brain) and Local List
        
        # 1. Update Brain's State (Confidence, PnL)
        pnl = outcome.get("pnl", 0.0)
        pnl_pct = 0.0 # Need total capital to calc pct, simplified here
        self.brain.state.update_after_trade(pnl, pnl_pct)
        
        # 2. Update Vector Memory if significant
        if abs(pnl) > 1000:
             content = f"Decision: {decision}. Outcome: {outcome}"
             score = 1.0 if pnl > 0 else -1.0
             self.brain.memory.add_memory(content, score)
             
             # 【GraphRAG 重构】重大爆仓/盈利提取概念写入图谱
             extract_text = f"交易复盘: 进行了操作 {decision.get('action')}，导致 {pnl} 的盈亏结果。当时信心为 {self.brain.state.confidence}。请提取本次成败的关键因素概念。"
             current_time = time.time()
             asyncio.create_task(self._async_extract_and_store(extract_text, current_time, is_news=False))

        # Phase 4: Wind-Tunnel reward update
        wind = decision.get("wind_tunnel", {})
        if wind.get("valid"):
            pred = float(wind.get("predicted_return", 0.0))
            actual = 0.0
            if len(self._price_history) >= 2:
                prev = self._price_history[-2]
                curr = self._price_history[-1]
                actual = (curr - prev) / prev if prev > 0 else 0.0
            if (pred * actual) > 0 and pnl > 0:
                self._wind_tunnel_confidence = min(1.0, self._wind_tunnel_confidence + 0.05)
            else:
                self._wind_tunnel_confidence = max(0.1, self._wind_tunnel_confidence - 0.02)
            self._wind_tunnel_records.append({
                "pred": pred,
                "actual": actual,
                "pnl": pnl,
                "confidence": self._wind_tunnel_confidence,
                "step": self._step_count
            })
            if len(self._wind_tunnel_records) > 50:
                self._wind_tunnel_records = self._wind_tunnel_records[-50:]

        # Phase 4: Text-based Gradient Descent (Investment Beliefs)
        self._recent_trade_outcomes.append({
            "decision": decision,
            "pnl": pnl,
            "status": outcome.get("status", "UNKNOWN"),
            "timestamp": time.time()
        })
        if len(self._recent_trade_outcomes) > 20:
            self._recent_trade_outcomes = self._recent_trade_outcomes[-20:]

        if self.brain.state.consecutive_losses >= 3 or self.brain.state.consecutive_wins >= 3:
            self._reflect_and_persist_beliefs()
             
        # 3. 处理被风控拒绝的订单（监管认知）
        status = outcome.get("status")
        if status == "REJECTED" or status == OrderStatus.REJECTED:
             reason = outcome.get("reason", "Unknown regulatory rejection")
             self.compliance_feedback.append(reason)
             if len(self.compliance_feedback) > 5:
                 self.compliance_feedback.pop(0)
                 
             # Add to brain memory with negative penalty
             content = f"REGULATORY REJECTION! Order was rejected by Risk Control. Reason: {reason}. Action was: {decision.get('action')}"
             # Higher penalty than normal loss to ensure "respect" for regulation
             self.brain.memory.add_memory(content, -2.0)
             
             # 【GraphRAG 重构】监管教训提取
             asyncio.create_task(self._async_extract_and_store(content, time.time(), is_news=False))
             
             # Also slightly decrease confidence
             self.brain.state.confidence *= 0.95

    # ------------------------------------------
    # 额外方法 (针对 TraderAgent 特定请求)
    # ------------------------------------------
    
    def get_psychology_description(self) -> str:
        """返回心理画像描述"""
        return (
            f"Risk: {self.profile.get('risk_aversion', 0.5):.2f}, "
            f"Conf: {self.brain.state.confidence:.1f}"
        )
    
    def share_opinion(self) -> Dict[str, Any]:
        """
        分享自己的投资观点到社交网络
        
        生成当前 Agent 的情绪/观点信号，供邻居 Agent 接收。
        
        Returns:
            包含 agent_id、sentiment、confidence 的字典
        """
        # 基于最近的决策和情绪确定观点
        sentiment = "neutral"
        if hasattr(self, 'emotional_state'):
            if self.emotional_state in ("Greedy", "Confident"):
                sentiment = "bullish"
            elif self.emotional_state in ("Fearful", "Regretful", "Anxious"):
                sentiment = "bearish"
        
        # 如果绑定了社交网络，更新节点状态
        if self.social_graph and self.social_node_id is not None:
            node = self.social_graph.agents.get(self.social_node_id)
            if node:
                if sentiment == "bearish":
                    node.sentiment_state = SentimentState.INFECTED
                elif sentiment == "bullish":
                    node.sentiment_state = SentimentState.BULLISH
                else:
                    node.sentiment_state = SentimentState.SUSCEPTIBLE
        
        return {
            "agent_id": self.agent_id,
            "sentiment": sentiment,
            "confidence": self.brain.state.confidence / 100.0,
            "emotional_state": getattr(self, 'emotional_state', 'Neutral')
        }
    
    def receive_opinion(self, opinions: List[Dict[str, Any]]) -> str:
        """
        接收邻居的观点，生成社交压力描述
        
        Args:
            opinions: 邻居分享的观点列表
            
        Returns:
            社交压力描述字符串 (用于注入 Prompt)
        """
        if not opinions:
            return "没有收到朋友们的消息。"
        
        bearish_count = sum(1 for o in opinions if o.get("sentiment") == "bearish")
        bullish_count = sum(1 for o in opinions if o.get("sentiment") == "bullish")
        total = len(opinions)
        
        # 计算社交压力
        if bearish_count > total * 0.6:
            return f"朋友圈{bearish_count}/{total}个人在恐慌抛售！你感到巨大的社交压力。"
        elif bullish_count > total * 0.6:
            return f"朋友圈{bullish_count}/{total}个人在兴奋加仓！你感到 FOMO 压力。"
        elif bearish_count > bullish_count:
            return f"朋友圈偏悲观 ({bearish_count}空 vs {bullish_count}多)。"
        elif bullish_count > bearish_count:
            return f"朋友圈偏乐观 ({bullish_count}多 vs {bearish_count}空)。"
        else:
            return "朋友圈观点分歧，情绪比较混乱。"

    def get_social_summary(self) -> str:
        """
        获取绑定社交网络的舆情摘要
        
        Returns:
            社交舆情摘要字符串
        """
        if self.social_graph and self.social_node_id is not None:
            return self.social_graph.generate_social_summary(self.social_node_id)
        return "未加入任何投资社群。"
