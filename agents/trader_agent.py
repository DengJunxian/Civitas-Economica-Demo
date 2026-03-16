# file: agents/trader_agent.py
"""
TraderAgent 瀹炵幇 鈥?鍩轰簬璁ょ煡闂幆鐨勪氦鏄撴櫤鑳戒綋

瀹炵幇 BaseAgent 瀹氫箟鐨?Perceive-Reason-Decide-Act 闂幆銆?
鏍稿績鐗规€?
1. 蹇冪悊鐢诲儚椹卞姩: 椋庨櫓鍘屾伓銆佽嚜淇＄▼搴︺€佸叧娉ㄥ箍搴︾瓑涓€у寲鍙傛暟
2. 妯℃嫙 DeepSeek R1 鎺ㄧ悊: 鐢熸垚绫讳汉鎬濈淮閾?(Chain of Thought)
3. 缁撴瀯鍖栧喅绛? 杈撳嚭鏍囧噯 Limit Order
4. 蹇參鎬濊€冨弻灞傛灦鏋?(System 1/2): 鑺傜害绠楀姏锛屾ā鎷熺洿瑙変笌娣辨€?

浣滆€? Civitas Economica Team
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
from core.behavioral_finance import (
    ReferencePoints,
    ReferenceShiftConfig,
    behavioral_update_step,
    initialize_reference_points,
    predict_next_return,
)
from agents.persona import Persona, RiskAppetite
from core.society.network import SocialGraph, SentimentState
from agents.cognition.graph_storage import GraphMemoryBank
from agents.cognition.graph_builder import GraphExtractor
from agents.roles.news_analyst import NewsAnalyst
from agents.roles.quant_analyst import QuantAnalyst
from agents.roles.risk_analyst import RiskAnalyst

class TraderAgent(BaseAgent):
    """
    TraderAgent 鈥?鍏峰瀹屾暣璁ょ煡闂幆鐨勪氦鏄撴櫤鑳戒綋
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
        
        # 浜烘牸鐢诲儚闆嗘垚
        self.persona = persona if persona else Persona(name=agent_id)
        
        # 绀句氦缃戠粶闆嗘垚
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
        # 娉ㄥ叆妯″瀷浼樺厛绾?(濡傛灉 Brain 鏀寔)
        if hasattr(self.brain, 'model_priority'):
            self.brain.model_priority = model_priority
        
        # 鍚堣鍙嶉璁板繂锛堝瓨鍌ㄨ椋庢帶鎷掔粷鐨勫師鍥狅級
        self.compliance_feedback: List[str] = []
        
        # 蹇?鎱㈡€濊€冩ā寮忕姸鎬佽窡韪?
        self._last_news_count = 0
        self._last_social_sentiment = "neutral"
        self._fast_mode_consecutive_steps = 0
        
        # GraphRAG 璁板繂灞?
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
        self.reference_points: Optional[ReferencePoints] = None
        shift_profile = self.persona.reference_shift_profile() if hasattr(self.persona, "reference_shift_profile") else {}
        self.reference_shift_config = ReferenceShiftConfig(**shift_profile)
        base_risk = self.persona.base_risk_score() if hasattr(self.persona, "base_risk_score") else 0.5
        self.current_risk_appetite: float = float(base_risk)
        self.current_trading_intent: float = 0.0
        self.current_loss_aversion_intensity: float = float(self.persona.loss_aversion)
        self.last_behavioral_state: Dict[str, Any] = {}

    def _map_persona_to_risk_aversion(self) -> float:
        mapping = {
            RiskAppetite.CONSERVATIVE: 0.8,
            RiskAppetite.BALANCED: 0.5,
            RiskAppetite.AGGRESSIVE: 0.3,
            RiskAppetite.GAMBLER: 0.1
        }
        return mapping.get(self.persona.risk_appetite, 0.5)

    @staticmethod
    def _emotion_to_sentiment(emotion: str) -> float:
        mapping = {
            "Greedy": 0.65,
            "Confident": 0.35,
            "Neutral": 0.0,
            "Anxious": -0.35,
            "Regretful": -0.65,
            "Fearful": -0.8,
        }
        return float(mapping.get(str(emotion or "Neutral"), 0.0))

    def _derive_step_sentiment(self, perceived_data: Dict[str, Any], social_signal: str) -> float:
        snapshot: MarketSnapshot = perceived_data.get("snapshot")
        trend = float(getattr(snapshot, "market_trend", 0.0))
        emotion_term = self._emotion_to_sentiment(getattr(self, "emotional_state", "Neutral"))
        signal = (social_signal or "").lower()
        social_term = 0.0
        if "panic" in signal or "selling" in signal or "nervous" in signal:
            social_term -= 0.35
        if "buying" in signal or "fomo" in signal or "bull" in signal:
            social_term += 0.35
        sentiment = 0.45 * emotion_term + 0.35 * np.clip(trend, -1.0, 1.0) + 0.20 * social_term
        return float(np.clip(sentiment, -1.0, 1.0))

    def _estimate_peer_anchor(self, current_price: float) -> float:
        price = float(max(current_price, 1e-6))
        if self.social_graph is None or self.social_node_id is None:
            return price
        bearish_ratio = float(self.social_graph.get_bearish_ratio(self.social_node_id))
        bullish_ratio = float(self.social_graph.get_bullish_ratio(self.social_node_id))
        peer_shift = np.clip((bullish_ratio - bearish_ratio) * 0.05, -0.10, 0.10)
        return float(price * (1.0 + peer_shift))

    def _estimate_policy_anchor(self, snapshot: MarketSnapshot, current_price: float) -> tuple[float, float]:
        price = float(max(current_price, 1e-6))
        text_shock = float(getattr(snapshot, "text_policy_shock", 0.0))
        tax_rate = float(getattr(snapshot, "policy_tax_rate", 0.0))
        regime = str(getattr(snapshot, "text_regime_bias", "neutral")).lower()
        shock = text_shock - tax_rate * 0.5
        if regime in {"bullish", "easing", "supportive"}:
            shock += 0.2
        elif regime in {"bearish", "tightening", "restrictive"}:
            shock -= 0.2
        shock = float(np.clip(shock, -1.0, 1.0))
        return float(price * (1.0 + 0.08 * shock)), shock

    def update_behavioral_state(self, perceived_data: Dict[str, Any], social_signal: str) -> Dict[str, Any]:
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        current_price = float(snapshot.last_price)
        if self.reference_points is None:
            self.reference_points = initialize_reference_points(current_price)

        sentiment = self._derive_step_sentiment(perceived_data, social_signal)
        peer_anchor = self._estimate_peer_anchor(current_price)
        policy_anchor, policy_shock = self._estimate_policy_anchor(snapshot, current_price)
        step = behavioral_update_step(
            sentiment=sentiment,
            current_price=current_price,
            reference_points=self.reference_points,
            base_risk_appetite=self.current_risk_appetite,
            peer_anchor=peer_anchor,
            policy_anchor=policy_anchor,
            policy_shock=policy_shock,
            loss_aversion=float(self.persona.loss_aversion),
            shift_config=self.reference_shift_config,
            reference_weights=self.persona.reference_weights() if hasattr(self.persona, "reference_weights") else None,
        )
        self.reference_points = step.reference_points
        self.current_risk_appetite = float(step.risk_appetite)
        self.current_trading_intent = float(step.trading_intent)
        self.current_loss_aversion_intensity = float(step.loss_aversion_intensity)
        self.last_behavioral_state = {
            "sentiment": float(step.sentiment),
            "risk_appetite": float(step.risk_appetite),
            "trading_intent": float(step.trading_intent),
            "prospect_direction": float(step.prospect_direction),
            "loss_aversion_intensity": float(step.loss_aversion_intensity),
            "weighted_reference_return": float(step.weighted_reference_return),
            "reference_points": {
                "purchase_anchor": float(step.reference_points.purchase_anchor),
                "recent_high_anchor": float(step.reference_points.recent_high_anchor),
                "peer_anchor": float(step.reference_points.peer_anchor),
                "policy_anchor": float(step.reference_points.policy_anchor),
            },
        }
        return dict(self.last_behavioral_state)

    def _apply_behavioral_intent_overlay(
        self,
        decision_payload: Dict[str, Any],
        snapshot: MarketSnapshot,
    ) -> Dict[str, Any]:
        intent = float(self.current_trading_intent)
        risk = float(self.current_risk_appetite)
        decision = decision_payload.setdefault("decision", {})
        action = str(decision.get("action", "HOLD")).upper()
        qty = int(decision.get("qty", 0) or 0)
        price = float(decision.get("price", snapshot.last_price) or snapshot.last_price)

        if action == "HOLD":
            if intent >= 0.55 and self.cash_balance > snapshot.last_price:
                action = "BUY"
                qty = max(100, int((self.cash_balance * (0.06 + 0.16 * risk)) / snapshot.last_price) // 100 * 100)
                price = round(snapshot.last_price * 1.01, 2)
            elif intent <= -0.55:
                holding = int(self.portfolio.get(snapshot.symbol, 0))
                if holding > 0:
                    action = "SELL"
                    qty = max(100, int(holding * (0.25 + 0.45 * abs(intent))) // 100 * 100)
                    qty = min(qty, holding)
                    price = round(snapshot.last_price * 0.99, 2)

        decision["action"] = action
        decision["qty"] = max(0, int(qty))
        decision["price"] = float(price)
        decision_payload["behavioral_state"] = dict(self.last_behavioral_state)
        return decision_payload

    def bind_social_node(self, node_id: int, graph: SocialGraph):
        """Bind this agent to a node in the Social Graph."""
        self.social_node_id = node_id
        self.social_graph = graph
        if node_id in graph.agents:
            graph.agents[node_id].agent_id = self.agent_id
        # Initial sync so social diffusion can consume semantic profile.
        self.sync_social_semantic_profile()

    def _risk_tilt_from_persona(self) -> float:
        """Map discrete risk preference to a continuous tilt score in [-1, 1]."""
        mapping = {
            RiskAppetite.CONSERVATIVE: -0.8,
            RiskAppetite.BALANCED: 0.0,
            RiskAppetite.AGGRESSIVE: 0.6,
            RiskAppetite.GAMBLER: 0.9,
        }
        return float(mapping.get(self.persona.risk_appetite, 0.0))

    def _default_focus_topics(self) -> List[str]:
        """Build a stable cold-start topic profile before graph memory converges."""
        horizon_map = {
            "short-term": ["volatility", "turnover", "sentiment", "hot-theme"],
            "medium-term": ["policy", "liquidity", "earnings", "valuation"],
            "long-term": ["fundamental", "industry-trend", "fiscal", "rates"],
        }
        risk_map = {
            RiskAppetite.CONSERVATIVE: ["defensive", "dividend", "stable-growth"],
            RiskAppetite.BALANCED: ["balanced-allocation", "policy", "cycle"],
            RiskAppetite.AGGRESSIVE: ["growth", "technology", "beta"],
            RiskAppetite.GAMBLER: ["speculation", "high-vol", "leverage"],
        }

        horizon_key = str(self.persona.investment_horizon.value).lower()
        topics = horizon_map.get(horizon_key, ["policy", "risk"])
        topics = topics + risk_map.get(self.persona.risk_appetite, ["policy", "risk"])
        return list(dict.fromkeys(topics))

    def sync_social_semantic_profile(self) -> None:
        """Sync GraphRAG semantic profile to social graph node."""
        if self.social_graph is None or self.social_node_id is None:
            return

        dominant = self.graph_memory.get_dominant_narratives(top_k=6)
        default_focus = self._default_focus_topics()
        focus_topics = list(dict.fromkeys(default_focus + dominant[:3]))
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
        """Perception stage."""
        # 1. 杩囨护鏂伴椈 (鍩轰簬 attention_span)
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
        """Reasoning stage with analyst reports and behavioral context."""
        analyst_reports = self._collect_analyst_reports(perceived_data)
        perceived_data["analyst_reports"] = analyst_reports
        self.update_emotional_state(perceived_data)
        self._update_ram_state(
            analyst_reports.get("risk", {}),
            getattr(self, "emotional_state", "Neutral"),
        )
        social_signal = self.perceive_social_signal(perceived_data)
        perceived_data["behavioral_state"] = self.update_behavioral_state(perceived_data, social_signal)
        return await self.reason_and_act(perceived_data, self.emotional_state, social_signal)

    def _needs_deep_thinking(self, perceived_data: Dict[str, Any], social_signal: str) -> bool:
        """Decide whether to invoke expensive deep-thinking path."""
        news = perceived_data.get("news", [])
        if len(news) > self._last_news_count:
            self._last_news_count = len(news)
            return True

        pnl_pct = perceived_data.get("pnl_pct", 0)
        utility = pnl_pct if pnl_pct > 0 else pnl_pct * 2.25
        if utility < -0.15 or utility > 0.10:
            return True

        current_social = "neutral"
        if "panic" in social_signal.lower() or "selling" in social_signal.lower():
            current_social = "bearish"
        elif "buy" in social_signal.lower() or "fomo" in social_signal.lower():
            current_social = "bullish"

        if current_social != self._last_social_sentiment:
            self._last_social_sentiment = current_social
            return True

        if self._fast_mode_consecutive_steps > 10:
            return True
        return False

    def _fast_think(
        self,
        perceived_data: Dict[str, Any],
        emotional_state: str,
        social_signal: str,
    ) -> Dict[str, Any]:
        """Fast rule-based decision path."""
        snapshot: MarketSnapshot = perceived_data["snapshot"]
        trend_val = getattr(snapshot, "market_trend", 0)

        action = "HOLD"
        qty = 0
        price = 0.0
        last_price = snapshot.last_price
        behavioral_state = perceived_data.get("behavioral_state", self.last_behavioral_state)
        if isinstance(behavioral_state, dict):
            intent_score = float(behavioral_state.get("trading_intent", self.current_trading_intent))
            risk_score = float(behavioral_state.get("risk_appetite", self.current_risk_appetite))
        else:
            intent_score = self.current_trading_intent
            risk_score = self.current_risk_appetite

        if trend_val > 0.03 and self.cash_balance > last_price * 100:
            action = "BUY"
            qty = int((self.cash_balance * (0.06 + 0.18 * risk_score)) / last_price) // 100 * 100
            price = round(last_price * 1.02, 2)
        elif trend_val < -0.03:
            holding = self.portfolio.get(snapshot.symbol, 0)
            if holding > 0:
                action = "SELL"
                qty = int(holding * 0.2) // 100 * 100
                price = round(last_price * 0.98, 2)

        if "panic" in social_signal.lower() or emotional_state == "Fearful":
            holding = self.portfolio.get(snapshot.symbol, 0)
            if holding > 0:
                action = "SELL"
                qty = int(holding * 0.5) // 100 * 100
                if qty == 0 and holding > 0:
                    qty = holding
                price = round(last_price * 0.95, 2)

        if action == "HOLD":
            if intent_score >= 0.55 and self.cash_balance > last_price * 100:
                action = "BUY"
                qty = int((self.cash_balance * (0.05 + 0.18 * risk_score)) / last_price) // 100 * 100
                price = round(last_price * 1.01, 2)
            elif intent_score <= -0.55:
                holding = self.portfolio.get(snapshot.symbol, 0)
                if holding > 0:
                    action = "SELL"
                    qty = int(holding * (0.20 + 0.55 * abs(intent_score))) // 100 * 100
                    if qty <= 0:
                        qty = holding
                    price = round(last_price * 0.99, 2)

        if action == "BUY" and qty > 0:
            qty = max(100, int(qty * (0.85 + 0.6 * risk_score)))
        elif action == "SELL" and qty > 0:
            qty = max(100, int(qty * (0.85 + 0.6 * abs(min(0.0, intent_score)))))

        reasoning = f"(System 1) Fast response. Trend={trend_val:.3f}, Emotion={emotional_state}"
        return {
            "decision": {"action": action, "qty": qty, "price": price},
            "reasoning": reasoning,
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp,
            "behavioral_state": dict(self.last_behavioral_state),
        }

    async def _async_extract_and_store(self, text: str, current_time: float, is_news: bool = False):
        """Background graph extraction to avoid blocking the trade loop."""
        triplets = await self.graph_extractor.extract_graph(text)
        if not triplets:
            return
            
        topics = set()
        for t in triplets:
            self.graph_memory.add_triplet(t["subject"], t["predicate"], t["target"], t["weight"])
            topics.add(t["subject"])
            
        # 濡傛灉鏄畯瑙傛柊闂伙紝鐢熸垚鐭湡鑳跺泭缂撳瓨
        if is_news and topics:
            capsule_topic = list(topics)[0] # 鍙栦富瑕佹蹇典綔涓?topic
            summary = f"[{capsule_topic}] 鐩稿叧鍔ㄦ€佸凡鍙戠敓: {text[:50]}..."
            self.graph_memory.add_capsule(capsule_topic, summary, current_time, ttl_seconds=3600)

    def _collect_analyst_reports(self, perceived_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect analyst reports with defensive fallback payloads."""
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
                "sentiment_label": "neutral",
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
        """Update Risk Alert Meeting (RAM) state machine."""
        cvar = float(risk_report.get("cvar", 0.0) or 0.0)
        cvar_drop = None
        trigger_reason = ""

        if self._last_cvar is not None:
            cvar_drop = cvar - self._last_cvar
            if cvar_drop < -abs(self._last_cvar) * 0.5 and cvar < 0:
                trigger_reason = f"CVaR 闄￠檷: {self._last_cvar:.4f} -> {cvar:.4f}"

        if emotional_state == "Fearful":
            trigger_reason = trigger_reason or "鎯呯华 Fearful 瑙﹀彂椋庢帶浼氳"

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
        """Persist trade beliefs for future prompt reinforcement."""
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
        """Summarize recent outcomes into compact investment beliefs."""
        if not self._recent_trade_outcomes:
            return
        if self._step_count - self._last_belief_reflection_step < 5:
            return

        recent = self._recent_trade_outcomes[-8:]
        prompt = {
            "recent_trades": recent,
            "instruction": "Extract up to 3 abstract investment beliefs as a JSON array.",
        }
        messages = [
            {"role": "system", "content": "You are a quantitative behavioral analyst. Return JSON array only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]

        beliefs: List[str] = []
        try:
            if getattr(self.brain, "model_router", None):
                content, _, _ = self.brain.model_router.sync_call_with_fallback(
                    messages=messages,
                    priority_models=["deepseek-chat", "glm-4-flashx"],
                    timeout_budget=20.0,
                )
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    beliefs = [str(x) for x in parsed if str(x).strip()]
        except Exception:
            beliefs = []

        if not beliefs:
            if self.brain.state.consecutive_losses >= 3:
                beliefs = ["In noisy regimes, reduce turnover and avoid overtrading."]
            elif self.brain.state.consecutive_wins >= 3:
                beliefs = ["Trend-following can work, but drawdown controls must remain active."]
            else:
                beliefs = ["When uncertainty is high, position discipline beats frequent trial-and-error."]

        self._persist_beliefs(beliefs)
        self._last_belief_reflection_step = self._step_count

    def _wind_tunnel_predict(self) -> Dict[str, Any]:
        """Internal wind-tunnel predictor used as pre-trade veto signal."""
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
        """Core cognition method combining market, emotion, and social signals."""
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
                    "reasoning": f"[RAM] 椋庢帶浼氳瑙﹀彂锛屽己鍒堕伩闄╂竻浠撱€傚師鍥狅細{self._ram_last_trigger}",
                    "symbol": snapshot.symbol,
                    "timestamp": snapshot.timestamp,
                    "analyst_reports": perceived_data.get("analyst_reports", {}),
                    "risk_mode": "RAM"
                }
            return {
                "decision": {"action": "HOLD"},
                "reasoning": f"[RAM] 椋庢帶浼氳瑙﹀彂锛屽綋鍓嶆棤鎸佷粨锛屼繚鎸佽鏈涖€傚師鍥狅細{self._ram_last_trigger}",
                "symbol": snapshot.symbol,
                "timestamp": snapshot.timestamp,
                "analyst_reports": perceived_data.get("analyst_reports", {}),
                "risk_mode": "RAM"
            }

        # 1. System 1 (Rule-based) Enforcement
        # 濡傛灉琚爣璁颁负闈?LLM Agent锛屽己鍒朵娇鐢ㄥ揩鎬濊€?(妯℃嫙璁＄畻妯″紡)
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

        # Sync semantic profile before deep reasoning.
        self.sync_social_semantic_profile()
        
        # 1. 妫€绱?GraphRAG 璁板繂涓庣煡璇嗚兌鍥?
        current_time = snapshot.timestamp
        graph_context = ""
        capsules = self.graph_memory.get_valid_capsules(current_time)
        extracted_keywords = []
        news_text = "; ".join(news) if news else ""
        
        if news_text:
            # 绠€鍗曠殑鍏抽敭璇嶆彁鍙栭€昏緫鐢ㄤ簬鏌ュ浘
            if "鍒╃巼" in news_text or "闄嶅噯" in news_text: extracted_keywords.append("鍒╃巼")
            if "liquidity" in news_text.lower():
                extracted_keywords.append("liquidity")
            if "鏀跨瓥" in news_text: extracted_keywords.append("鏀跨瓥")
            if not extracted_keywords: extracted_keywords = ["甯傚満", "椋庨櫓"]
        
        if capsules:
            graph_context = "銆愮煡璇嗚兌鍥?瀹忚鍏辫瘑缂撳瓨)銆慭n" + "\n".join(capsules)
        elif extracted_keywords:
            subgraph = self.graph_memory.retrieve_subgraph(extracted_keywords, depth=2)
            if subgraph:
                graph_context = "銆愮鏈夎鐭ュ浘璋卞叧鑱斻€慭n" + subgraph
                
        # 濡傛灉鏈夋柊娑堟伅涓旀病鏈夊懡涓紦瀛橈紝鍚庡彴鍚姩鎻愬彇浠ュ厖瀹炲浘璋?
        if news_text and not capsules:
            asyncio.create_task(self._async_extract_and_store(news_text, current_time, is_news=True))
            
        # 2. Construct Market State
        market_state = {
            "price": snapshot.last_price,
            "trend": getattr(snapshot, "market_trend", "闇囪崱"), 
            "panic_level": getattr(snapshot, "panic_level", 0.5), 
            "news": news_text if news_text else "no-major-news",
            "last_rejection_reason": self.compliance_feedback[-1] if self.compliance_feedback else None,
            "policy_description": getattr(snapshot, "policy_description", ""),
            "policy_tax_rate": getattr(snapshot, "policy_tax_rate", 0.0),
            "policy_news": getattr(snapshot, "policy_news", ""),
            "text_dominant_topic": getattr(snapshot, "text_dominant_topic", "uncategorized"),
            "text_sentiment_score": getattr(snapshot, "text_sentiment_score", 0.0),
            "text_panic_score": getattr(snapshot, "text_panic_score", 0.0),
            "text_greed_score": getattr(snapshot, "text_greed_score", 0.0),
            "text_policy_shock": getattr(snapshot, "text_policy_shock", 0.0),
            "text_regime_bias": getattr(snapshot, "text_regime_bias", "neutral"),
            "text_impact_paths": getattr(snapshot, "text_impact_paths", []),
            "graph_context": graph_context,
            "analyst_reports": perceived_data.get("analyst_reports", {}),
            "behavioral_state": perceived_data.get("behavioral_state", self.last_behavioral_state),
        }
        
        # Map numeric trend to string for Brain
        trend_val = snapshot.market_trend
        if trend_val > 0.02: market_state["trend"] = "涓婃定"
        elif trend_val < -0.02: market_state["trend"] = "涓嬭穼"
        else: market_state["trend"] = "闇囪崱"

        # 2. Construct Account State
        account_state = {
            "cash": perceived_data["cash"],
            "market_value": perceived_data["portfolio_value"] - perceived_data["cash"],
            "pnl_pct": perceived_data["pnl_pct"],
            "risk_appetite": self.current_risk_appetite,
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
            decision_output = self._apply_behavioral_intent_overlay(decision_output, snapshot)
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
        """Update emotional state from PnL and panic indicators."""
        pnl_pct = perceived_data.get("pnl_pct", 0)
        snapshot = perceived_data.get("snapshot")
        panic_level = getattr(snapshot, "panic_level", 0)
        
        # Adjust sensitivity based on Persona
        loss_threshold = -0.10 / (self.persona.loss_aversion / 2.25) # More averse -> smaller threshold (easier to regret)
        panic_threshold = 0.6 * self.persona.patience # More patient -> higher threshold
        
        if pnl_pct < loss_threshold:
            self.emotional_state = "Regretful" # 鎮旀仺 (澶т簭)
        elif pnl_pct < loss_threshold / 2:
            self.emotional_state = "Anxious"   # 鐒﹁檻 (灏忎簭)
        elif pnl_pct > 0.10:
            self.emotional_state = "Greedy"    # 璐┆ (澶ц禋)
        elif pnl_pct > 0.05:
            self.emotional_state = "Confident" # 鑷俊 (灏忚禋)
        elif panic_level > panic_threshold:
            self.emotional_state = "Fearful"   # 鎭愭儳 (甯傚満鎭愭厡)
        else:
            self.emotional_state = "Neutral"   # 涓€?
            
    def perceive_social_signal(self, perceived_data: Dict[str, Any]) -> str:
        """Perceive social signal from graph neighbors and market trend."""
        # 1. 浼樺厛浣跨敤鐪熷疄绀句氦缃戠粶
        if self.social_graph and self.social_node_id is not None:
            bearish_ratio = self.social_graph.get_bearish_ratio(self.social_node_id)
            
            # Thresholds based on conformity
            panic_threshold = 1.0 - (self.persona.conformity * 0.8) # High conformity -> Low threshold (0.2)
            
            if bearish_ratio > panic_threshold:
                return f"Panic Alert! {bearish_ratio:.0%} of neighbors are selling!"
            elif bearish_ratio > panic_threshold * 0.6:
                return "Neighbors are getting nervous."
                
        # 2. 鍙湁鍦ㄦ病鏈夌綉缁滆繛鎺ユ椂锛屾墠鍥為€€鍒板競鍦鸿秼鍔夸唬鐞?
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
        """Convert reasoning output into an executable order."""
        decision = reasoning_output.get("decision", {})
        action = decision.get("action", "HOLD")
        if action == "HOLD" or not action:
            return None

        qty = int(decision.get("qty", 0) or 0)
        price = float(decision.get("price", 0.0) or 0.0)
        symbol = reasoning_output.get("symbol", "UNKNOWN")
        timestamp = reasoning_output.get("timestamp", time.time())

        if action == "BUY":
            side = OrderSide.BUY
        elif action == "SELL":
            side = OrderSide.SELL
        else:
            return None
        if qty <= 0:
            return None

        wind_report = self._wind_tunnel_predict()
        decision["wind_tunnel"] = wind_report
        if wind_report.get("valid"):
            pred = float(wind_report.get("predicted_return", 0.0))
            conf = float(wind_report.get("confidence", 0.5))
            if action == "BUY" and pred < -0.005 and conf > 0.6:
                return None
            if action == "SELL" and pred > 0.005 and conf > 0.6:
                return None

        if side == OrderSide.BUY:
            cost = price * qty
            if cost > self.cash_balance:
                qty = int(self.cash_balance / price) // 100 * 100 if price > 0 else 0
                if qty <= 0:
                    return None
        elif side == OrderSide.SELL:
            holding = self.portfolio.get(symbol, 0)
            qty = min(qty, int(holding))
            if qty <= 0:
                return None

        return Order(
            symbol=symbol,
            price=price,
            quantity=qty,
            side=side,
            order_type=OrderType.LIMIT,
            agent_id=self.agent_id,
            timestamp=timestamp,
        )

    async def update_memory(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> None:
        """Update memory and reinforcement signals after execution feedback."""
        pnl = float(outcome.get("pnl", 0.0) or 0.0)
        pnl_pct = 0.0
        self.brain.state.update_after_trade(pnl, pnl_pct)

        if abs(pnl) > 1000:
            content = f"Decision: {decision}. Outcome: {outcome}"
            score = 1.0 if pnl > 0 else -1.0
            self.brain.memory.add_memory(content, score)
            extract_text = (
                f"交易复盘: action={decision.get('action')}, pnl={pnl}, confidence={self.brain.state.confidence}"
            )
            asyncio.create_task(self._async_extract_and_store(extract_text, time.time(), is_news=False))

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

        self._recent_trade_outcomes.append({
            "decision": decision,
            "pnl": pnl,
            "status": outcome.get("status", "UNKNOWN"),
            "timestamp": time.time(),
        })
        if len(self._recent_trade_outcomes) > 20:
            self._recent_trade_outcomes = self._recent_trade_outcomes[-20:]

        if self.brain.state.consecutive_losses >= 3 or self.brain.state.consecutive_wins >= 3:
            self._reflect_and_persist_beliefs()

        status = outcome.get("status")
        if status == "REJECTED" or status == OrderStatus.REJECTED:
            reason = outcome.get("reason", "Unknown regulatory rejection")
            self.compliance_feedback.append(reason)
            if len(self.compliance_feedback) > 5:
                self.compliance_feedback.pop(0)
            content = f"REGULATORY REJECTION: {reason}. Action={decision.get('action')}"
            self.brain.memory.add_memory(content, -2.0)
            asyncio.create_task(self._async_extract_and_store(content, time.time(), is_news=False))
            self.brain.state.confidence *= 0.95

    # ------------------------------------------
    # 棰濆鏂规硶 (閽堝 TraderAgent 鐗瑰畾璇锋眰)
    # ------------------------------------------
    
    def get_psychology_description(self) -> str:
        """Return short text summary of agent psychology profile."""
        return (
            f"Risk: {self.profile.get('risk_aversion', 0.5):.2f}, "
            f"Conf: {self.brain.state.confidence:.1f}"
        )

    def share_opinion(self) -> Dict[str, Any]:
        """Share local sentiment/opinion into social network."""
        sentiment = "neutral"
        if hasattr(self, "emotional_state"):
            if self.emotional_state in ("Greedy", "Confident"):
                sentiment = "bullish"
            elif self.emotional_state in ("Fearful", "Regretful", "Anxious"):
                sentiment = "bearish"

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
            "emotional_state": getattr(self, "emotional_state", "Neutral"),
        }

    def receive_opinion(self, opinions: List[Dict[str, Any]]) -> str:
        """Receive neighbor opinions and summarize social pressure."""
        if not opinions:
            return "No social messages received."

        bearish_count = sum(1 for o in opinions if o.get("sentiment") == "bearish")
        bullish_count = sum(1 for o in opinions if o.get("sentiment") == "bullish")
        total = len(opinions)

        if bearish_count > total * 0.6:
            return f"{bearish_count}/{total} neighbors are panic selling."
        if bullish_count > total * 0.6:
            return f"{bullish_count}/{total} neighbors are aggressively buying."
        if bearish_count > bullish_count:
            return f"Circle is bearish ({bearish_count} vs {bullish_count})."
        if bullish_count > bearish_count:
            return f"Circle is bullish ({bullish_count} vs {bearish_count})."
        return "Circle is split and uncertain."

    def get_social_summary(self) -> str:
        """Return social sentiment summary for bound social graph."""
        if self.social_graph and self.social_node_id is not None:
            return self.social_graph.generate_social_summary(self.social_node_id)
        return "Not bound to social graph."

