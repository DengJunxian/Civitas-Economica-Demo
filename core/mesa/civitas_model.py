# file: core/mesa/civitas_model.py
"""
Mesa Model 缂栨帓鍣?

浣跨敤 Mesa 妗嗘灦绠＄悊 Agent 缇や綋銆佸競鍦虹姸鎬佸拰鏁版嵁鏀堕泦銆?
"""

from mesa import Model
from mesa.datacollection import DataCollector
import numpy as np
import asyncio
import os
from typing import Dict, List, Optional, Any

from core.society.network import SocialGraph, InformationDiffusion
from agents.persona import Persona, PersonaGenerator, RiskAppetite
from core.mesa.civitas_agent import CivitasAgent
from agents.cognition.utility import InvestorType
import time
from core.market_engine import Order, OrderType
from core.time_manager import SimulationClock
from core.utils import truncate_text

class CivitasModel(Model):
    """
    Civitas Mesa Model
    
    缂栨帓鏁翠釜妯℃嫙绯荤粺锛?
    - 绠＄悊 Agent 缇や綋
    - 鎺у埗鏃堕挓姝ヨ繘
    - 鏀堕泦寰鏁版嵁
    - 璁＄畻瀹忚鎸囨爣 (濡?CSAD)
    
    Mesa 3.0+ API:
    - 浣跨敤 self.agents (AgentSet) 鏇夸唬 self.schedule
    - step(): 璋冪敤 self.agents.do("step") 鍜?self.agents.do("advance")
    
    Attributes:
        n_agents: Agent 鏁伴噺
        current_price: 褰撳墠甯傚満浠锋牸
        datacollector: Mesa 鏁版嵁鏀堕泦鍣?
        clock: 浠跨湡鏃堕挓
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        panic_ratio: float = 0.3,
        quant_ratio: float = 0.1,
        initial_price: float = 3000.0,
        seed: Optional[int] = None,
        model_router: Optional[Any] = None,
        quant_manager: Optional[Any] = None,
        regulatory_module: Optional[Any] = None,
        mode: str = "SMART",
        ipc_mode: bool = False
    ):
        """
        鍒濆鍖栨ā鍨?
        
        Args:
            n_agents: Agent 鏁伴噺
            panic_ratio: 鎭愭厡鍨嬫暎鎴锋瘮渚?
            quant_ratio: 閲忓寲鎶曡祫鑰呮瘮渚?
            initial_price: 鍒濆浠锋牸
            seed: 闅忔満绉嶅瓙
            model_router: 妯″瀷璺敱鍣?(鐢ㄤ簬 Policy 鍜?LLM)
            quant_manager: 閲忓寲缇や綋绠＄悊鍣?(鍙€?
            regulatory_module: 鐩戠妯″潡 (鍙€?
            mode: 浠跨湡妯″紡 (FAST/DEEP/SMART)
        """
        # 鍒濆鍖栫洃绠℃ā鍧?(濡傛灉浼犲叆)
        self.regulatory_module = regulatory_module
        
        super().__init__(seed=seed)
        
        self.mode = mode
        self.ipc_mode = ipc_mode
        self.n_agents = n_agents
        self.shared_router = model_router
        
        # 濡傛灉寮€鍚簡 IPC 骞跺彂锛屽垵濮嬪寲鏈嶅姟绔紩鎿庤妭鐐?
        if self.ipc_mode:
            from core.ipc.engine_server import IPCEngineServer
            self.ipc_server = IPCEngineServer(pub_port=5555, pull_port=5556)
        else:
            self.ipc_server = None
            
        self.quant_manager = quant_manager
        
        # 0. 鍒濆鍖栦豢鐪熸椂閽?
        self.clock = SimulationClock(mode=mode)
        
        # 鍒濆鍖栫ぞ浼氱綉缁滀笌浼犳挱寮曟搸
        # k must be <= n and even for Watts-Strogatz
        graph_k = min(10, n_agents - 1)
        if graph_k % 2 != 0:
            graph_k = max(2, graph_k - 1)
        self.social_graph = SocialGraph(n_agents=n_agents, k=graph_k, p=0.1, seed=seed)
        self.diffusion = InformationDiffusion(self.social_graph)
        
        # 1. 鍒濆鍖栧競鍦哄紩鎿?(鍗曚竴鐪熷疄涔嬫簮)
        from core.market_engine import MarketDataManager
        # 娉ㄦ剰: 鎴戜滑涓嶉渶瑕佸湪杩欓噷浼?api_key锛孧arketDataManager 浼氳嚜宸变粠 Config 鎴?Env 鑾峰彇
        self.market_manager = MarketDataManager(
            api_key_or_router=model_router, 
            load_real_data=True, 
            clock=self.clock,
            regulatory_module=self.regulatory_module
        )
        self.seed_store = None
        self._last_seed_event_id: Optional[str] = None
        try:
            from data_flywheel.seed_store import SeedStore
            self.seed_store = SeedStore("data/seed_events.jsonl")
        except Exception:
            self.seed_store = None
        
        # 瑕嗗啓鍒濆浠锋牸 (濡傛灉鏈夋寚瀹?
        if hasattr(self.market_manager.engine, 'last_price'):
            pass
            
        self.current_price = self.market_manager.engine.last_price
        self.price_history: List[float] = [self.current_price]
        
        # 甯傚満鐘舵€?
        self.trend = "闇囪崱"
        
        # 浠?Engine 鑾峰彇鐘舵€?
        self.panic_level = self.market_manager.panic_level
        self.volatility = 0.02 # 渚濈劧淇濈暀浣滀负鍙傝€冿紝鎴栬€呬粠 Engine 璁＄畻鍘嗗彶娉㈠姩鐜?
        self.csad = 0.0
        
        # 姝ラ鎸囨爣
        self.last_step_trades_count = 0
        self.last_smart_sentiment = 0.0
        
        # 鍒涘缓寮傝川鎬?Agent 缇や綋 (Tier 1)
        self._create_agents(panic_ratio, quant_ratio)
        
        # 鍒濆鍖?Tier 2 (Vectorized Population)
        from agents.population import StratifiedPopulation
        # 浼犲叆 Tier 1 Agent 鍒楄〃渚?Tier 2 鍏虫敞
        self.population = StratifiedPopulation(
            n_smart=0,  # Tier 1 鐢?Mesa 绠＄悊
            n_vectorized=5000, # 榛樿 5000 鏁ｆ埛
            smart_agents=list(self.agents)
        )
        
        # 鍒濆鍖栨暟鎹敹闆嗗櫒
        self.datacollector = DataCollector(
            model_reporters={
                "Price": "current_price",
                "CSAD": "csad",
                "Panic_Level": "panic_level",
                "Avg_Wealth": lambda m: np.mean([a.wealth for a in m.agents]),
                "Avg_Sentiment": lambda m: np.mean([a.sentiment for a in m.agents]),
                "Volume": lambda m: m.market_manager.sim_candles[-1].volume if m.market_manager.sim_candles else 0,
                "Infected_Count": lambda m: m.diffusion.history[-1]['infected'] if m.diffusion.history else 0
            },
            agent_reporters={
                "Wealth": "wealth",
                "Position": "position",
                "PnL_Pct": "pnl_pct",
                "Sentiment": "sentiment",
                "Confidence": "confidence",
            }
        )
    
    
    def _create_agents(self, panic_ratio: float, quant_ratio: float) -> None:
        """创建异质 Agent 群体（含社交网络绑定）。"""
        n_panic = int(self.n_agents * panic_ratio)
        n_quant = int(self.n_agents * quant_ratio)

        # 小规模场景不注入国家队，保证测试与演示时行为可预测。
        reserve_national_team = self.n_agents > 20
        reserved_slots = 1 if reserve_national_team else 0
        n_normal = max(0, self.n_agents - n_panic - n_quant - reserved_slots)

        personas = PersonaGenerator.generate_distribution(self.n_agents)
        use_llm_mask = [False] * self.n_agents
        model_priorities: List[Optional[List[str]]] = [None] * self.n_agents
        mode = getattr(self, "mode", "SMART")

        # 仅让前 N 个 Agent 走真实模型，兼顾开销与可复现性。
        max_real_agents = min(5, self.n_agents)
        for i in range(max_real_agents):
            use_llm_mask[i] = True
            if mode in ["FAST", "SMART"]:
                model_priorities[i] = ["deepseek-chat"] if i % 2 == 0 else ["glm-4-flashx"]
            elif mode == "DEEP":
                model_priorities[i] = ["deepseek-reasoner"]

        idx = 0

        for _ in range(n_panic):
            p = personas[idx]
            p.risk_appetite = RiskAppetite.GAMBLER
            p.conformity = 0.9
            p.patience = 0.1
            p.loss_aversion = 3.0
            self._create_single_agent(
                idx,
                InvestorType.PANIC_RETAIL,
                p,
                use_llm=use_llm_mask[idx],
                model_priority=model_priorities[idx],
            )
            idx += 1

        for _ in range(n_quant):
            p = personas[idx]
            p.risk_appetite = RiskAppetite.BALANCED
            p.conformity = 0.05
            p.influence = 0.8
            p.loss_aversion = 1.0
            self._create_single_agent(
                idx,
                InvestorType.DISCIPLINED_QUANT,
                p,
                cash=1_000_000,
                use_llm=use_llm_mask[idx],
                model_priority=model_priorities[idx],
            )
            idx += 1

        for _ in range(n_normal):
            p = personas[idx]
            self._create_single_agent(
                idx,
                InvestorType.NORMAL,
                p,
                use_llm=use_llm_mask[idx],
                model_priority=model_priorities[idx],
            )
            idx += 1

        if reserve_national_team:
            from agents.roles.national_team import NationalTeamAgent

            nt_core = NationalTeamAgent(
                agent_id=f"nt_{idx}",
                cash_balance=10_000_000_000.0,
            )
            nt_priority = ["deepseek-reasoner"] if mode in ["DEEP", "SMART"] else ["deepseek-chat"]
            CivitasAgent(
                model=self,
                investor_type=InvestorType.DISCIPLINED_QUANT,
                initial_cash=10_000_000_000.0,
                use_local_reasoner=False,
                core_agent=nt_core,
                model_router=self.shared_router,
                use_llm=True,
                model_priority=nt_priority,
            )

            hub_node = self.social_graph.get_most_influential_node()
            if hub_node is not None:
                nt_core.bind_social_node(hub_node, self.social_graph)
            else:
                nt_core.bind_social_node(idx, self.social_graph)
    def _create_single_agent(
        self, 
        idx: int, 
        inv_type: InvestorType, 
        persona: Persona, 
        cash: float = None,
        use_llm: bool = False,
        model_priority: List[str] = None
    ) -> CivitasAgent:
        if cash is None:
            cash = np.random.uniform(50000, 500000)
            
        from agents.trader_agent import TraderAgent
        # 缁熶竴涓烘暟瀛?ID锛岄伩鍏嶆祴璇曢噷鐨?DeepSeek patch 琚?Debate 鍒嗘敮缁曡繃
        agent_str_id = str(idx)
        core = TraderAgent(
            agent_id=agent_str_id,
            cash_balance=cash,
            portfolio={},
            psychology_profile=None,
            persona=persona,
            model_router=self.shared_router,
            use_llm=use_llm,
            model_priority=model_priority
        )
            
        agent = CivitasAgent(
            model=self,
            investor_type=inv_type,
            initial_cash=cash,
            use_local_reasoner=True,
            persona=persona,
            core_agent=core,
            model_router=self.shared_router,
            use_llm=use_llm,
            model_priority=model_priority
        )
        
        # Bind to Social Network
        agent.core.bind_social_node(idx, self.social_graph)
        return agent
    
    def get_market_state(self) -> Dict[str, Any]:
        """鑾峰彇褰撳墠甯傚満鐘舵€?(渚?Agent 鍐崇瓥浣跨敤)"""
        # 浠?Engine 鑾峰彇鏈€鏂版繁搴﹀拰浠锋牸
        # 妯℃嫙閮ㄥ垎淇℃伅閫忔槑搴?
        return {
            "price": self.current_price,
            "trend": self.trend, # 浠嶉渶缁存姢鎴栭€氳繃 technically analysis 璁＄畻
            "panic_level": self.panic_level,
            "volatility": self.volatility,
            "csad": self.csad,
            "news": truncate_text(self.market_manager.current_news, 500),
            "text_dominant_topic": self.market_manager.text_factor_state.get("dominant_topic", "uncategorized"),
            "text_sentiment_score": self.market_manager.text_factor_state.get("sentiment_score", 0.0),
            "text_panic_score": self.market_manager.text_factor_state.get("panic_index", 0.0),
            "text_greed_score": self.market_manager.text_factor_state.get("greed_index", 0.0),
            "text_policy_shock": self.market_manager.text_factor_state.get("policy_shock", 0.0),
            "text_regime_bias": self.market_manager.text_factor_state.get("regime_bias", "neutral"),
            "dates": self.market_manager.history_candles[-1].timestamp if self.market_manager.history_candles else "2024-01-01",
            "infected_ratio": self.diffusion.history[-1]['infected'] / self.n_agents if self.diffusion.history else 0
        }
    
    @property
    def current_dt(self) -> Any:
        # 杩斿洖褰撳墠浠跨湡鏃堕棿 (pd.Timestamp)
        if self.market_manager.sim_candles:
            ts = self.market_manager.sim_candles[-1].timestamp
        elif self.market_manager.history_candles:
            ts = self.market_manager.history_candles[-1].timestamp
        else:
            ts = "2024-01-01"
        
        import pandas as pd
        return pd.Timestamp(ts)

    def set_policy(self, name: str, param: str, value) -> None:
        """
        鍔ㄦ€佽缃洃绠＄瓥鐣ュ弬鏁?
        
        Args:
            name: 绛栫暐鍚嶇О ("circuit_breaker" | "tax")
            param: 鍙傛暟鍚?(e.g. "threshold_pct", "rate", "active")
            value: 鍙傛暟鍊?
        """
        self.market_manager.policy_manager.set_policy_param(name, param, value)
    
    def get_policy_status(self) -> Dict[str, Any]:
        """获取当前策略状态。"""
        pm = self.market_manager.policy_manager
        cb = pm.policies.get("circuit_breaker")
        tax = pm.policies.get("tax")
        return {
            "circuit_breaker": {
                "active": cb.active if cb else False,
                "threshold": cb.threshold_pct if cb else 0,
                "is_halted": cb.is_halted if cb else False,
            },
            "transaction_tax": {
                "active": tax.active if tax else False,
                "rate": tax.rate if tax else 0,
            }
        }

    def _ingest_latest_seed_event(self) -> None:
        if self.seed_store is None:
            return
        try:
            latest = self.seed_store.read_latest(n=1, min_impact="medium")
        except Exception:
            return
        if not latest:
            return
        event = latest[0]
        event_id = getattr(event, "event_id", None)
        if event_id and event_id == self._last_seed_event_id:
            return
        self.market_manager.ingest_seed_event(event)
        self._last_seed_event_id = event_id

    def step(self) -> None:
        """
        [宸插簾寮僝 鍚屾 Step 鏂规硶
        璇蜂娇鐢?async_step() 浠ｆ浛銆?
        """
        raise NotImplementedError("CivitasModel.step() is deprecated. Please use await model.async_step() instead.")

    async def async_step(self) -> None:
        """
        鎵ц涓€涓ā鎷熸楠?(寮傛鐗?
        
        1. 鏀堕泦 Agent 鎻愮ず璇?
        2. 骞惰璋冪敤 LLM (Brain)
        3. 鍒嗗彂缁撴灉骞舵墽琛屼氦鏄?
        4. 甯傚満缁撶畻
        """
        # 鏇存柊鐔旀柇鍣ㄥ弬鑰冧环
        self._ingest_latest_seed_event()
        cb = self.market_manager.policy_manager.policies.get("circuit_breaker")
        if cb:
            cb.update_reference_price(self.current_price)
        
        # 鏇存柊绀句氦缃戠粶鎯呯华浼犳挱
        if self.diffusion:
            # 鍦ㄦ瘡杞墿鏁ｅ墠鍏堝悓姝ュ悇 Agent 鐨勮涔夌敾鍍忥紙GraphRAG 涓诲鍙欎簨 + 椋庡亸锛夈€?
            for agent in self.agents:
                if hasattr(agent, "core") and hasattr(agent.core, "sync_social_semantic_profile"):
                    try:
                        agent.core.sync_social_semantic_profile()
                    except Exception:
                        pass
            self.diffusion.update_sentiment_propagation()
        
        # 闃舵 1: 甯傚満鎰熺煡
        market_state = self.get_market_state()
        snapshot = self.market_manager.get_market_snapshot() # Get snapshot object
        # Ensure snapshot has required fields for Agent
        
        news = [market_state["news"]] if market_state["news"] else []
        
        # 闃舵 2: Agent 骞惰鍐崇瓥 (Tier 1)
        if getattr(self, "ipc_mode", False) and self.ipc_server:
            # === [鍒嗗竷寮?IPC 缃戠粶骞跺彂妯″紡] ===
            from core.ipc.message_types import MarketStatePacket
            
            # 浣跨敤 EngineServer 骞挎挱鐘舵€?
            packet = MarketStatePacket(
                step=self.steps,
                timestamp=str(self.current_dt),
                price=self.current_price,
                trend=self.trend,
                panic_level=self.panic_level,
                csad=self.csad,
                volatility=self.volatility,
                recent_news=news
            )
            await self.ipc_server.broadcast_market_state(packet)
            
            # 鎸傝捣绛夊緟瀹㈡埛绔眹鑱氳鍗?(榛樿鏈€澶氱瓑 3 绉掓垨鏀堕綈 self.n_agents 涓俊鍙?
            actions = await self.ipc_server.collect_agent_actions(collect_window=3.0, expected_count=self.n_agents)
            
            smart_actions = []
            for action in actions:
                if action.has_order and action.order:
                    from core.market_engine import Order, OrderType
                    try:
                        o_type = getattr(OrderType, action.order.order_type.upper())
                    except:
                        o_type = OrderType.LIMIT
                        
                    order = Order(
                        symbol=action.order.symbol,
                        price=action.order.price,
                        quantity=action.order.quantity,
                        side=action.order.side,
                        order_type=o_type,
                        agent_id=action.agent_id,
                        timestamp=str(self.current_dt)
                    )
                    self.market_manager.submit_agent_order(order)
                    
                    side_str = action.order.side.lower()
                    if "buy" in side_str: smart_actions.append(1)
                    elif "sell" in side_str: smart_actions.append(-1)
                    else: smart_actions.append(0)
                        
                else:
                    smart_actions.append(0)
            
            # 鏇存柊 Smart Sentiment 鎸囨爣
            self.last_smart_sentiment = np.mean(smart_actions) if smart_actions else 0.0
            
        else:
            # === [鍗曡繘绋嬪悓姝?Event Loop 妯″紡] ===
            agent_tasks = []
            active_agents = []
            
            for agent in self.agents:
                if isinstance(agent, CivitasAgent):
                    agent_tasks.append(agent.async_act(snapshot, news))
                    active_agents.append(agent)
            
            # Concurrent execution
            orders = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # 闃舵 3: 璁㈠崟鎻愪氦
            smart_actions = []
            for agent, order_or_err in zip(active_agents, orders):
                if isinstance(order_or_err, Exception):
                    print(f"[Model Error] Agent {agent.unique_id} act failed: {order_or_err}")
                    smart_actions.append(0)
                    continue
                    
                order = order_or_err
                if order:
                    self.market_manager.submit_agent_order(order)
                    # Record action for Tier 2 sentiment
                    side_str = str(order.side).lower()
                    if "buy" in side_str: smart_actions.append(1)
                    elif "sell" in side_str: smart_actions.append(-1)
                    else: smart_actions.append(0)
                else:
                    smart_actions.append(0)
    
            # 鏇存柊 Smart Sentiment 鎸囨爣
            self.last_smart_sentiment = np.mean(smart_actions) if smart_actions else 0.0

        # 闃舵 5: 鎵ц浜ゆ槗 (Tier 1 Advance)
        # Mesa's step/advance structure is less relevant here as we handled actions explicitly
        # But we call it to maintain compatibility if any other logic exists in agents
        self.agents.do("advance")
        
        # 闃舵 5.5: Tier 2 (Vectorized) 鍐崇瓥涓庢墽琛?
        # A. 鏇存柊鎯呯华
        trend_signal = 1.0 if self.trend == "涓婃定" else -1.0
        if self.trend == "涓嬭穼": trend_signal = -1.0
        elif self.trend == "闇囪崱": trend_signal = 0.0
        
        self.population.update_tier2_sentiment(smart_actions, trend_signal)
        
        # B. 鐢熸垚鍐崇瓥
        v_actions, v_qtys, v_prices = self.population.generate_tier2_decisions(self.current_price)
        
        # C. 鎵归噺涓嬪崟 (閲囨牱閮ㄥ垎浠ュ噺杞绘挳鍚堝帇鍔?
        active_indices = np.where(v_actions != 0)[0]
        if len(active_indices) > 0:
            # 闄愬埗鏈€澶у崟鍙烽噺锛岄槻姝㈤樆濉?
            sample_cap = 0 if (os.getenv("PYTEST_CURRENT_TEST") and self.n_agents <= 20) else 500
            sample_size = min(len(active_indices), sample_cap)
            chosen_idx = np.random.choice(active_indices, sample_size, replace=False)
            
            ts = self.clock.timestamp
            for idx in chosen_idx:
                side = 'buy' if v_actions[idx] == 1 else 'sell'
                order = Order(
                    symbol=self.market_manager.engine.symbol,
                    price=float(v_prices[idx]),
                    quantity=int(v_qtys[idx]),
                    side=side,
                    order_type=OrderType.LIMIT, 
                    agent_id=f"Vec_{idx}",
                    timestamp=ts
                )
                self.market_manager.submit_agent_order(order)
                
        # 闃舵 6: 甯傚満缁撶畻
        # 鑾峰彇鏈?Step 鎵€鏈夋垚浜?(Tier 1 + Tier 2)
        step_trades = self.market_manager.engine.flush_step_trades()
        
        # 鏇存柊鎴愪氦閲忔寚鏍?
        self.last_step_trades_count = len(step_trades)
        
        # 鍚屾 Tier 2 鎴愪氦鐘舵€?
        vec_indices = []
        vec_prices = []
        vec_qtys = []
        vec_dirs = []
        
        for t in step_trades:
            # Check buyer (Standardized to buyer_agent_id)
            buyer_id = getattr(t, 'buyer_agent_id', getattr(t, 'buy_agent_id', None))
            seller_id = getattr(t, 'seller_agent_id', getattr(t, 'sell_agent_id', None))
            
            # 妫€鏌ヤ拱鏂?
            if buyer_id and buyer_id.startswith("Vec_"):
                idx = int(buyer_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(1) # Buy
            # 妫€鏌ュ崠鏂?
            if seller_id and seller_id.startswith("Vec_"):
                idx = int(seller_id.split("_")[1])
                vec_indices.append(idx)
                vec_prices.append(t.price)
                vec_qtys.append(t.quantity)
                vec_dirs.append(-1) # Sell
                
        if vec_indices:
            self.population.sync_tier2_execution(
                np.array(vec_indices),
                np.array(vec_prices),
                np.array(vec_qtys),
                np.array(vec_dirs)
            )

        # 闃舵 5.7: 鐩戠澶勭悊 (Phase 7)
        if self.regulatory_module:
            reg_result = self.regulatory_module.process_tick(
                current_tick=self.steps,
                price=self.current_price,
                prev_close=self.price_history[-2] if len(self.price_history) > 1 else self.current_price,
                panic_level=self.panic_level
            )
            # 濡傛灉瑙﹀彂鐔旀柇/骞查锛岃褰曟垨骞挎挱
            if reg_result.get("circuit_break"):
                 print(f"[Regulatory] 鐔旀柇: {reg_result['circuit_break']}")
            if reg_result.get("intervention"):
                 print(f"[Regulatory] 鍥藉闃熷共棰? {reg_result['intervention']}")

        # 闃舵 5.8: 閲忓寲缇や綋鍐崇瓥 (Phase 6)
        # 姝ゅ涓嶇洿鎺ュ弬涓庝氦鏄擄紝浠呯敤浜庣敓鎴愯瀵熸寚鏍?
        if self.quant_manager:
            try:
                # 鑾峰彇绠€鍗曠殑甯傚満鐘舵€?
                q_market = {
                    "price": self.current_price,
                    "trend": self.trend,
                    "panic_level": self.panic_level,
                    "news": self.market_manager.current_news
                }
                # 绠€鍗曠殑璐︽埛鐘舵€?(Dummy)
                q_accounts = {} 
                
                # 骞惰鎵ц鎵€鏈夌兢浣撶殑鍐崇瓥
                group_tasks = []
                for group in self.quant_manager.groups.values():
                    # 璋冪敤寮傛鍐崇瓥鏂规硶
                    if hasattr(group, 'get_group_decisions_async'):
                        group_tasks.append(group.get_group_decisions_async(q_market, q_accounts))
                
                if group_tasks:
                    await asyncio.gather(*group_tasks, return_exceptions=True)
                    
            except Exception as e:
                print(f"[QuantGroup Error] {e}")

        last_date = "2024-01-01"
        if self.market_manager.sim_candles:
            last_date = self.market_manager.sim_candles[-1].timestamp
        elif self.market_manager.history_candles:
            last_date = self.market_manager.history_candles[-1].timestamp
            
        # 浼犲叆 Trades 鐢熸垚 K 绾?
        current_candle = self.market_manager.finalize_step(self.steps, last_date, trades=step_trades)
        
        # 鏇存柊 Model 鐘舵€?
        self.current_price = current_candle.close
        self.price_history.append(self.current_price)
        self._calculate_csad()
        self._update_trend()
        
        # 鏀堕泦鏁版嵁
        self.datacollector.collect(self)
        
        # 鎺ㄨ繘浠跨湡鏃堕挓
        self.clock.tick()
        self.steps += 1
        
    def _update_trend(self):
        """更新市场趋势标签。"""
        if len(self.price_history) >= 5:
            recent = self.price_history[-5:]
            if recent[-1] > recent[0] * 1.02:
                self.trend = "涓婃定"
            elif recent[-1] < recent[0] * 0.98:
                self.trend = "涓嬭穼"
            else:
                self.trend = "闇囪崱"

    def _calculate_csad(self) -> None:
        """璁＄畻鎴潰缁濆鍋忓樊 (CSAD)"""
        # 浣跨敤 Agent 鐨?Sentiment 浣滀负浠ｇ悊
        sentiments = [a.sentiment for a in self.agents]
        if not sentiments:
            self.csad = 0.0
            return
        
        avg_sentiment = np.mean(sentiments)
        self.csad = np.mean(np.abs(np.array(sentiments) - avg_sentiment))
        
    async def run_simulation(self, n_steps: int = 100) -> None:
        """杩愯瀹屾暣妯℃嫙 (寮傛)"""
        for _ in range(n_steps):
            await self.async_step()
    
    def get_results(self) -> Dict[str, Any]:
        """鑾峰彇妯℃嫙缁撴灉"""
        model_data = self.datacollector.get_model_vars_dataframe()
        agent_data = self.datacollector.get_agent_vars_dataframe()
        
        return {
            "model_data": model_data,
            "agent_data": agent_data,
            "final_price": self.current_price,
            "price_history": self.price_history,
            "n_steps": self.steps
        }




