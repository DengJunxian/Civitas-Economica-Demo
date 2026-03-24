# file: agents/population.py

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Iterable, Protocol, runtime_checkable

import numpy as np

from core.exchange.evolution import EvolutionOperators, StrategyGenome

from config import GLOBAL_CONFIG
from agents.brain import DeepSeekBrain
from agents.debate_brain import DebateBrain
from agents.brain import build_runtime_metadata, resolve_feature_flags, stable_json_hash
from agents.persona import Persona, PersonaGenerator


@runtime_checkable
class AgentProtocol(Protocol):
    """Shared agent contract for trader/persona/population compatibility."""

    agent_id: str
    cash_balance: float
    persona: Persona


@runtime_checkable
class PopulationEngine(Protocol):
    """Shared population contract for live, replay, and calibration flows."""

    seed: int
    config_hash: str
    feature_flags: Dict[str, bool]

    def iter_agents(self) -> Iterable[AgentProtocol]: ...

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentProtocol]: ...

    def register_agent(self, agent: AgentProtocol) -> None: ...

    def snapshot_metadata(self) -> Dict[str, Any]: ...

    def build_market_distribution_report(self, personas: Optional[List[Persona]] = None, *, regime: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]: ...

    def render_market_distribution_report(self, personas: Optional[List[Persona]] = None) -> str: ...

# --- Tier 1: 鏅鸿兘浣撳畾涔?---

@dataclass
class SmartAgent:
    """
    Tier 1 鎰忚棰嗚 (Opinion Leader)
    鎷ユ湁鐙珛鐨?DeepSeek 澶ц剳銆佽蹇嗗拰瀹屾暣璐︽埛鐘舵€併€?
    """
    id: str
    brain: DeepSeekBrain
    cash: float
    holdings: int
    cost_basis: float
    persona: Persona = field(default_factory=Persona)
    
    # 褰卞搷鍔涚郴鏁?(鍐冲畾鑳借鐩栧灏?Tier 2 鑺傜偣)
    influence_factor: float = 1.0
    reference_points: Dict[str, float] = field(default_factory=dict)
    risk_appetite: float = 0.5
    trading_intent: float = 0.0
    strategy_genome: Optional[StrategyGenome] = None
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        # 娉? 闇€澶栭儴娉ㄥ叆褰撳墠浠锋牸璁＄畻锛屾澶勪粎涓哄崰浣?
        return 0.0

    @property
    def agent_id(self) -> str:
        return self.id

    @property
    def cash_balance(self) -> float:
        return float(self.cash)

    def profile_signature(self) -> str:
        payload = {
            "agent_id": self.id,
            "cash": float(self.cash),
            "holdings": int(self.holdings),
            "cost_basis": float(self.cost_basis),
            "persona": self.persona.to_dict(),
        }
        return stable_json_hash(payload)

    def snapshot_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.id,
            "cash_balance": float(self.cash),
            "holdings": int(self.holdings),
            "cost_basis": float(self.cost_basis),
            "persona_signature": self.persona.stable_signature(),
            "profile_signature": self.profile_signature(),
        }

# --- Tier 2: 鍚戦噺鍖栫兢浣?---

class StratifiedPopulation:
    """
    鍒嗗眰鏅鸿兘浣撶兢浣撶鐞嗗櫒
    
    鏋舵瀯璁捐:
    - Tier 1: List[SmartAgent] -> 澶嶆潅閫昏緫锛屼綆骞跺彂
    - Tier 2: Numpy Matrix -> 绠€鍗曢€昏緫锛岄珮骞跺彂 (SIMD)
    """
    
    # 鐘舵€佺煩闃靛垪绱㈠紩瀹氫箟
    IDX_CASH = 0
    IDX_HOLDINGS = 1
    IDX_COST = 2
    IDX_SENTIMENT = 3  # -1.0 (鏋佸害鐪嬬┖) ~ 1.0 (鏋佸害鐪嬪)
    IDX_COGNITIVE_TYPE = 4  # 璁ょ煡绫诲瀷: 0=鎶€鏈淳, 1=娑堟伅娲? 2=璺熼娲?
    IDX_CONFIDENCE = 5  # 淇″績鎸囨暟: 0-100
    
    def __init__(
        self,
        n_smart: int = 50,
        n_vectorized: int = 9950,
        api_key: str = None,
        smart_agents: List[Any] = None,
        *,
        seed: Optional[int] = None,
        feature_flags: Optional[Dict[str, bool]] = None,
        agents: Optional[List[AgentProtocol]] = None,
        personas: Optional[List[Persona]] = None,
        market_regime: str = "default",
        market_composition_path: Optional[str | Path] = None,
    ):
        self.n_smart = n_smart
        self.n_vectorized = n_vectorized
        self._api_key = api_key
        self.seed = self._resolve_seed(seed)
        self.feature_flags = resolve_feature_flags(feature_flags)
        self.market_regime = str(market_regime or "default")
        self.market_composition_path = Path(market_composition_path) if market_composition_path else None
        self.market_composition = PersonaGenerator.load_market_composition(self.market_composition_path)
        self._seed_rng(self.seed)
        
        # 1. 鍒濆鍖?Tier 1 (Smart Agents)
        self.compat_agents: List[AgentProtocol] = []
        if smart_agents is not None:
            self.smart_agents = smart_agents
            self.n_smart = len(smart_agents)
            print(f"[*] 浣跨敤澶栭儴浼犲叆鐨?{self.n_smart} 涓?Smart Agents")
        elif personas is not None:
            self.smart_agents = []
            self.n_smart = len(personas)
            self._init_smart_agents_from_personas(personas)
            print(f"[*] 浣跨敤澶栭儴浼犲叆鐨?{self.n_smart} 涓?Smart Agents")
        else:
            self.smart_agents: List[SmartAgent] = []
            self._init_smart_agents()

        if agents:
            for agent in agents:
                self.register_agent(agent)
        
        # 2. 鍒濆鍖?Tier 2 (Vectorized Matrix)
        # Shape: (N, 6) -> [Cash, Holdings, Cost, Sentiment, CognitiveType, Confidence]
        self.state = np.zeros((n_vectorized, 6), dtype=np.float32)
        self._init_vectorized_state()
        # Reference points (purchase / recent high / peer / policy) for Tier-2 agents
        self.reference_points = np.zeros((n_vectorized, 4), dtype=np.float32)
        self.risk_appetite_state = np.full(n_vectorized, 0.5, dtype=np.float32)
        self.trading_intent_state = np.zeros(n_vectorized, dtype=np.float32)
        self.loss_aversion_intensity_state = np.full(n_vectorized, 2.25, dtype=np.float32)
        self._init_reference_points()
        
        # 3. 鏋勫缓绀句細缃戠粶 (Influence Topology)
        # 浣跨敤 Watts-Strogatz 灏忎笘鐣岀綉缁滄ā鎷?鍦堝瓙"鏁堝簲
        # 瀹為檯涓婃垜浠瀯寤轰竴涓簩閮ㄥ浘鐨勭畝鍖栫増锛氭瘡涓?Tier 2 鑺傜偣鍏虫敞 1-3 涓?Tier 1 鑺傜偣
        self.influence_map = self._build_influence_network()
        
        # 4. 鏋勫缓閭诲眳缃戠粶锛堢敤浜庢秾鐜板紡缇婄兢鏁堝簲锛?
        self.neighbor_network = self._build_neighbor_network()
        
        # 5. 鏋勫缓Smart Agent涔嬮棿鐨勭ぞ浜ょ綉缁滐紙澶涔嬮棿浜掔浉褰卞搷锛?
        self.smart_social_network = self._build_smart_social_network()
        
        # 6. 涓婁竴杞甋mart Agent鍐崇瓥璁板綍锛堢敤浜庡奖鍝嶄紶閫掞級
        self.last_smart_actions: Dict[str, Dict] = {}
        self.evolution_ops = EvolutionOperators(mutation_rate=0.20, mutation_scale=0.12)
        self.smart_genomes: Dict[str, StrategyGenome] = {}
        self._init_strategy_genomes()
        self.snapshot_info = self._describe_snapshot()
        self.runtime_metadata = build_runtime_metadata(
            seed=self.seed,
            config=self._build_config_signature(),
            snapshot=self.snapshot_info,
            feature_flags=self.feature_flags,
        )
        self.config_hash = self.runtime_metadata["config_hash"]
        self.data_snapshot_info = dict(self.snapshot_info)
        self.market_distribution_report = self.build_market_distribution_report()

    def _resolve_seed(self, seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        env_seed = os.environ.get("CIVITAS_SEED")
        if env_seed is not None and env_seed.strip():
            try:
                return int(env_seed)
            except ValueError:
                pass
        return 0

    def _seed_rng(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def _build_config_signature(self) -> Dict[str, Any]:
        return {
            "n_smart": int(self.n_smart),
            "n_vectorized": int(self.n_vectorized),
            "api_key_enabled": bool(self._api_key),
            "feature_flags": dict(self.feature_flags),
            "market_regime": self.market_regime,
            "market_composition_path": str(self.market_composition_path) if self.market_composition_path else "",
            "market_composition_hash": stable_json_hash(self.market_composition),
        }

    def _describe_snapshot(self) -> Dict[str, Any]:
        return {
            "smart_agent_count": len(getattr(self, "smart_agents", [])),
            "compat_agent_count": len(getattr(self, "compat_agents", [])),
            "vectorized_shape": list(self.state.shape) if hasattr(self, "state") else [0, 0],
            "market_regime": getattr(self, "market_regime", "default"),
            "market_composition_path": str(self.market_composition_path) if getattr(self, "market_composition_path", None) else "",
        }
        
    def _init_smart_agents(self):
        """Initialize Tier-1 smart agents."""
        if self.n_smart <= 0:
            self.smart_agents = []
            return
        print(f"[*] 鍒濆鍖?{self.n_smart} 浣?DeepSeek 鏅鸿兘浣?..")
        for i in range(self.n_smart):
            persona = self._random_persona_template(i)
            # 鍓?涓狝gent榛樿浣跨敤DebateBrain锛堝惎鐢ㄨ京璁哄姛鑳斤級
            if i < 5:
                brain = DebateBrain(agent_id=f"Debate_{i}", persona=persona, api_key=self._api_key)
                agent = SmartAgent(
                    id=f"Debate_{i}",
                    brain=brain,
                    cash=GLOBAL_CONFIG.DEFAULT_CASH * random.uniform(10, 30),  # 杈╄澶璧勯噾鏇村
                    holdings=int(random.uniform(5000, 80000)),
                    cost_basis=3000.0
                )
            else:
                agent = SmartAgent(
                    id=f"Smart_{i}",
                    brain=DeepSeekBrain(agent_id=f"Smart_{i}", persona=persona, api_key=self._api_key),
                    cash=GLOBAL_CONFIG.DEFAULT_CASH * random.uniform(5, 20),
                    holdings=int(random.uniform(1000, 50000)),
                    cost_basis=3000.0
                )
            
            # 璁剧疆閲嶈鎬х瓑绾э紙鐢ㄤ簬娣峰悎璋冨害锛?
            # 鍓?0%涓烘牳蹇?2)锛?0-30%涓洪噸瑕?1)锛屽叾浣欎负鏅€?0)
            if i < self.n_smart * 0.1:
                agent.brain.importance_level = 2  # 鏍稿績Agent
            elif i < self.n_smart * 0.3:
                agent.brain.importance_level = 1  # 閲嶈Agent
            else:
                agent.brain.importance_level = 0  # 鏅€欰gent
            anchor = float(agent.cost_basis if agent.cost_basis > 0 else 3000.0)
            agent.reference_points = {
                "purchase_anchor": anchor,
                "recent_high_anchor": anchor,
                "peer_anchor": anchor,
                "policy_anchor": anchor,
            }
            self.smart_agents.append(agent)

    def _random_persona_template(self, index: int) -> Persona:
        """Build a deterministic persona template for synthetic agents."""
        if not self.feature_flags.get("population_protocol_v1", True):
            from agents.persona import InvestmentHorizon, RiskAppetite

            return Persona(
                name=f"SmartPersona_{index:03d}",
                risk_appetite=random.choice(list(RiskAppetite)),
                investment_horizon=random.choice(list(InvestmentHorizon)),
                conformity=random.betavariate(2, 2),
                influence=min(0.95, random.paretovariate(3) / 5.0),
                patience=random.uniform(0.2, 0.9),
                loss_aversion=random.uniform(1.5, 3.0),
                overconfidence=random.uniform(0.0, 0.6),
                reference_adaptivity=random.uniform(0.2, 0.8),
            )

        rng = random.Random(self.seed + index)
        weights = PersonaGenerator._resolve_composition_weights(regime=self.market_regime, composition=self.market_composition)
        archetype_key = PersonaGenerator._sample_archetype_key(rng, weights)
        persona = Persona.from_archetype(
            archetype_key,
            name=f"SmartPersona_{index:03d}",
            mutable_state=PersonaGenerator._generate_mutable_state(rng),
        )
        persona.name = persona.name or f"SmartPersona_{index:03d}"
        persona.mutable_state.update_semantic("population_index", index)
        return persona

    def _init_smart_agents_from_personas(self, personas: List[Persona]) -> None:
        self.smart_agents = []
        for idx, persona in enumerate(personas):
            agent = SmartAgent(
                id=persona.name or f"Persona_{idx:03d}",
                brain=DeepSeekBrain(
                    agent_id=persona.name or f"Persona_{idx:03d}",
                    persona={"risk_preference": persona.risk_appetite.value, "loss_aversion": persona.loss_aversion},
                    api_key=self._api_key,
                ),
                cash=GLOBAL_CONFIG.DEFAULT_CASH,
                holdings=int(random.uniform(1000, 10000)),
                cost_basis=3000.0,
                persona=persona,
            )
            self.smart_agents.append(agent)

    def _init_strategy_genomes(self) -> None:
        """Attach a strategy genome to each Tier-1 agent."""
        self.smart_genomes = {}
        for agent in self.smart_agents:
            genome = StrategyGenome.random()
            agent.strategy_genome = genome
            self.smart_genomes[agent.id] = genome

    def evolve_smart_genomes(self, performance_map: Dict[str, float]) -> Dict[str, Any]:
        """
        Run selection / crossover / mutation / local diffusion for smart-agent genomes.
        Returns a compact evolution summary.
        """
        if not self.smart_genomes:
            return {"selected": 0, "offspring": 0, "mutated": 0, "diffused": 0}

        selected_ids = self.evolution_ops.selection(performance_map, survival_rate=0.5)
        if not selected_ids:
            selected_ids = list(self.smart_genomes.keys())[:1]

        offspring_count = max(1, len(self.smart_genomes) // 5)
        spawned: Dict[str, StrategyGenome] = {}
        for idx in range(offspring_count):
            pa = self.smart_genomes[random.choice(selected_ids)]
            pb = self.smart_genomes[random.choice(selected_ids)]
            child = self.evolution_ops.mutation(self.evolution_ops.crossover(pa, pb))
            spawned_id = f"GenomeOffspring_{idx:03d}"
            spawned[spawned_id] = child

        mutated_count = 0
        for aid in selected_ids:
            old = self.smart_genomes.get(aid)
            if old is None:
                continue
            self.smart_genomes[aid] = self.evolution_ops.mutation(old)
            mutated_count += 1

        adjacency = {aid: self.smart_social_network.get(aid, []) for aid in self.smart_genomes.keys()}
        diffused = self.evolution_ops.local_diffusion(self.smart_genomes, adjacency, strength=0.06)

        # Keep Tier-1 population size unchanged: only refresh existing agent genomes.
        for agent in self.smart_agents:
            if agent.id in self.smart_genomes:
                agent.strategy_genome = self.smart_genomes[agent.id]

        return {
            "selected": len(selected_ids),
            "offspring": len(spawned),
            "mutated": mutated_count,
            "diffused": len(diffused),
        }

    def get_agent_by_id(self, agent_id: str) -> Optional[SmartAgent]:
        """鏍规嵁 ID 鑾峰彇 SmartAgent"""
        for agent in self.smart_agents:
            if agent.id == agent_id:
                return agent
        for agent in self.compat_agents:
            candidate_id = getattr(agent, "agent_id", getattr(agent, "id", None))
            if candidate_id == agent_id:
                return agent  # type: ignore[return-value]
        return None

    @property
    def agents(self) -> List[AgentProtocol]:
        return list(self.iter_agents())

    def iter_agents(self) -> Iterable[AgentProtocol]:
        yield from self.smart_agents
        yield from self.compat_agents

    def register_agent(self, agent: AgentProtocol) -> None:
        """Register a trader/persona-compatible agent into the shared engine."""
        if isinstance(agent, SmartAgent):
            self.smart_agents.append(agent)
            return
        if not hasattr(agent, "persona"):
            setattr(agent, "persona", Persona(name=str(getattr(agent, "agent_id", getattr(agent, "id", "Anonymous")))))
        self.compat_agents.append(agent)

    def build_market_distribution_report(
        self,
        personas: Optional[List[Persona]] = None,
        *,
        regime: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        def _extract_persona(agent: Any) -> Optional[Persona]:
            direct = getattr(agent, "persona", None)
            if direct is not None:
                return direct
            core = getattr(agent, "core", None)
            if core is not None:
                return getattr(core, "persona", None)
            return None

        if personas is not None:
            source_personas = personas
        else:
            source_personas = [p for p in (_extract_persona(agent) for agent in self.smart_agents) if p is not None]
            if not source_personas:
                source_personas = [p for p in (_extract_persona(agent) for agent in self.compat_agents) if p is not None]
        report = PersonaGenerator.build_market_distribution_report(
            source_personas,
            regime=regime or self.market_regime,
            seed=self.seed if seed is None else int(seed),
            composition_path=self.market_composition_path,
            snapshot=self.snapshot_metadata() if hasattr(self, "runtime_metadata") else {},
        )
        self.market_distribution_report = report
        return report

    def render_market_distribution_report(self, personas: Optional[List[Persona]] = None) -> str:
        if personas is None and getattr(self, "market_distribution_report", None):
            return str(self.market_distribution_report.get("markdown", ""))
        report = self.build_market_distribution_report(personas=personas)
        return str(report.get("markdown", ""))

    def snapshot_metadata(self) -> Dict[str, Any]:
        payload = dict(self.runtime_metadata)
        signatures: List[str] = []
        for agent in self.smart_agents:
            if hasattr(agent, "profile_signature"):
                signatures.append(str(agent.profile_signature()))
                continue
            core = getattr(agent, "core", None)
            persona = getattr(core, "persona", None)
            if persona is not None and hasattr(persona, "stable_signature"):
                signatures.append(str(persona.stable_signature()))
                continue
            agent_id = getattr(agent, "agent_id", getattr(agent, "id", "unknown"))
            signatures.append(stable_json_hash({"agent_id": agent_id}))
        payload["agent_signatures"] = signatures
        payload["smart_agent_count"] = len(self.smart_agents)
        payload["compat_agent_count"] = len(self.compat_agents)
        payload["market_regime"] = self.market_regime
        payload["market_distribution_report"] = dict(getattr(self, "market_distribution_report", {}) or {})
        return payload

    def _init_vectorized_state(self):
        """Initialize Tier-2 vectorized state matrix."""
        # 璧勯噾: 瀵规暟姝ｆ€佸垎甯?(璐瘜宸窛)
        # 淇: 鎻愰珮鍒濆鐜伴噾浣夸箣涓庢寔浠撲环鍊煎尮閰嶏紝閬垮厤缁撴瀯鎬у崠鍘?
        self.state[:, self.IDX_CASH] = np.random.lognormal(11.5, 0.8, self.n_vectorized)
        
        # 鎸佷粨: 闅忔満鍒嗗竷 (闄嶄綆鍒濆鎸佷粨锛岄伩鍏嶅崠鍘嬭繃澶?
        self.state[:, self.IDX_HOLDINGS] = np.random.randint(0, 1000, self.n_vectorized)
        
        # 鎴愭湰: 鍥寸粫 3000 鐐规尝鍔?
        self.state[:, self.IDX_COST] = np.random.normal(3000, 200, self.n_vectorized)
        
        # 鎯呯华: 鍒濆寰鍋忓ソ (姝ｅ父甯傚満鏁ｆ埛閫氬父鐣ュ亸涔愯)
        # Beta(3,2) 鍧囧€?0.6, 鏄犲皠鍒?[-1,1] 鍚庡潎鍊?0.2, 閫傚害鍋忓
        self.state[:, self.IDX_SENTIMENT] = np.random.beta(3, 2, self.n_vectorized) * 2 - 1
        
        # 璁ょ煡绫诲瀷: 0=鎶€鏈淳(20%), 1=娑堟伅娲?30%), 2=璺熼娲?50%)
        type_probs = np.random.random(self.n_vectorized)
        self.state[:, self.IDX_COGNITIVE_TYPE] = np.where(
            type_probs < 0.2, 0, np.where(type_probs < 0.5, 1, 2)
        )
        
        # 淇″績鎸囨暟: 姝ｆ€佸垎甯冿紝鍧囧€?5 (鐣ュ亸淇″績鍏呰冻)
        self.state[:, self.IDX_CONFIDENCE] = np.clip(
            np.random.normal(55, 15, self.n_vectorized), 0, 100
        )
    
    def _init_reference_points(self) -> None:
        """Initialize reference points for vectorized Tier-2 agents."""
        purchase = np.clip(self.state[:, self.IDX_COST], 1.0, None)
        self.reference_points[:, 0] = purchase
        self.reference_points[:, 1] = purchase
        self.reference_points[:, 2] = purchase
        self.reference_points[:, 3] = purchase
        self.risk_appetite_state = np.clip(self.state[:, self.IDX_CONFIDENCE] / 100.0, 0.0, 1.0).astype(np.float32)
        self.trading_intent_state.fill(0.0)
        self.loss_aversion_intensity_state.fill(2.25)

    def update_behavioral_layer(
        self,
        current_price: float,
        *,
        peer_anchor: Optional[float] = None,
        policy_anchor: Optional[float] = None,
        policy_shock: float = 0.0,
    ) -> None:
        """
        Vectorized behavioral pipeline:
        sentiment -> reference shift -> risk appetite -> trading intent
        """
        price = float(max(current_price, 1e-6))
        sentiment = np.clip(self.state[:, self.IDX_SENTIMENT], -1.0, 1.0)
        confidence = np.clip(self.state[:, self.IDX_CONFIDENCE] / 100.0, 0.0, 1.0)

        purchase = self.reference_points[:, 0]
        recent_high = self.reference_points[:, 1]
        peer = self.reference_points[:, 2]
        policy = self.reference_points[:, 3]

        purchase_decay = 0.03
        recent_high_decay = 0.10
        peer_decay = 0.16
        policy_decay = 0.18

        purchase = purchase * (1.0 - purchase_decay) + price * purchase_decay
        recent_high_target = np.maximum(recent_high, price)
        recent_high = recent_high * (1.0 - recent_high_decay) + recent_high_target * recent_high_decay
        peer_target = float(peer_anchor) if peer_anchor is not None else price * (1.0 + 0.04 * np.mean(sentiment))
        peer = peer * (1.0 - peer_decay) + peer_target * peer_decay
        policy_target = float(policy_anchor) if policy_anchor is not None else price * (1.0 + 0.08 * float(policy_shock))
        policy = policy * (1.0 - policy_decay) + policy_target * policy_decay

        self.reference_points[:, 0] = np.clip(purchase, 1e-6, None)
        self.reference_points[:, 1] = np.clip(recent_high, 1e-6, None)
        self.reference_points[:, 2] = np.clip(peer, 1e-6, None)
        self.reference_points[:, 3] = np.clip(policy, 1e-6, None)

        rel_purchase = (price - self.reference_points[:, 0]) / self.reference_points[:, 0]
        rel_high = (price - self.reference_points[:, 1]) / self.reference_points[:, 1]
        rel_peer = (price - self.reference_points[:, 2]) / self.reference_points[:, 2]
        rel_policy = (price - self.reference_points[:, 3]) / self.reference_points[:, 3]

        weighted_ref = 0.34 * rel_purchase + 0.26 * rel_high + 0.20 * rel_peer + 0.20 * rel_policy
        # 避免 np.where 同时计算两侧分支导致负底数幂次告警
        gain_base = np.clip(weighted_ref, 0.0, None)
        loss_base = np.clip(-weighted_ref, 0.0, None)
        gains = np.power(gain_base, 0.88)
        losses = -2.25 * np.power(loss_base, 0.88)
        losses = np.where(weighted_ref < 0.0, losses, 0.0)
        utility = gains + losses
        direction = np.tanh(utility * 8.0)

        underwater_ratio = (
            (rel_purchase < 0).astype(np.float32) * 0.34
            + (rel_high < 0).astype(np.float32) * 0.26
            + (rel_peer < 0).astype(np.float32) * 0.20
            + (rel_policy < 0).astype(np.float32) * 0.20
        )
        loss_aversion_intensity = np.clip(2.25 * (1.0 + 0.8 * underwater_ratio), 0.5, 6.0)

        base_risk = 0.4 * confidence + 0.6 * self.risk_appetite_state
        risk = base_risk + 0.20 * sentiment + 0.28 * direction - 0.15 * np.maximum(0.0, (loss_aversion_intensity - 1.0) / 5.0)
        risk = np.clip(risk, 0.0, 1.0)

        intent = 0.65 * direction + 0.25 * sentiment + 0.10 * (risk - 0.5) * 2.0
        intent = np.clip(intent, -1.0, 1.0)

        self.risk_appetite_state = risk.astype(np.float32)
        self.trading_intent_state = intent.astype(np.float32)
        self.loss_aversion_intensity_state = loss_aversion_intensity.astype(np.float32)

    def _build_neighbor_network(self, n_neighbors: int = 5) -> np.ndarray:
        """
        鏋勫缓閭诲眳缃戠粶锛堢敤浜庢秾鐜板紡缇婄兢鏁堝簲锛?
        
        姣忎釜鏁ｆ埛涓庡懆鍥磋嫢骞茶妭鐐瑰舰鎴愰偦灞呭叧绯伙紝
        妯℃嫙绀句氦鍦堝瓙鍐呯殑鎯呯华浼犳煋銆?
        
        Returns:
            閭诲眳绱㈠紩鐭╅樀 (N, n_neighbors)
        """
        neighbors = np.zeros((self.n_vectorized, n_neighbors), dtype=np.int32)
        for i in range(self.n_vectorized):
            # 闅忔満閫夋嫨閭诲眳锛堝彲閲嶅閫夋嫨鍚屼竴鑺傜偣琛ㄧず鏇寸揣瀵嗙殑鑱旂郴锛?
            candidates = np.random.randint(0, self.n_vectorized, n_neighbors * 2)
            # 鎺掗櫎鑷繁
            candidates = candidates[candidates != i][:n_neighbors]
            # 琛ラ綈涓嶈冻鐨?
            while len(candidates) < n_neighbors:
                new_neighbor = np.random.randint(0, self.n_vectorized)
                if new_neighbor != i:
                    candidates = np.append(candidates, new_neighbor)
            neighbors[i] = candidates[:n_neighbors]
        return neighbors

    def _build_influence_network(self) -> np.ndarray:
        """
        鏋勫缓褰卞搷鍥捐氨
        Returns:
            adjacency matrix (N_vec, N_smart) 鐨勭瀵嗚〃绀烘垨绱㈠紩鍒楄〃
            杩欓噷绠€鍖栦负: 姣忎釜 Tier 2 鍙湁涓€涓富瑕佸叧娉ㄧ殑 Tier 1 (Guru)
        """
        if self.n_smart <= 0:
            return np.zeros(self.n_vectorized, dtype=np.int32)
        # 甯曠疮鎵樺垎甯冿細灏戞暟澶鎷ユ湁缁濆ぇ澶氭暟绮変笣
        weights = np.random.pareto(a=2.0, size=self.n_smart)
        total = float(weights.sum())
        if total <= 0:
            return np.zeros(self.n_vectorized, dtype=np.int32)
        weights /= total
        
        # 涓烘瘡涓暎鎴峰垎閰嶄竴涓?甯﹀ご澶у摜"
        guru_indices = np.random.choice(
            self.n_smart, 
            size=self.n_vectorized, 
            p=weights
        )
        return guru_indices
    
    def _build_smart_social_network(self) -> Dict[str, List[str]]:
        """
        鏋勫缓Smart Agent涔嬮棿鐨勭ぞ浜ょ綉缁滐紙灏忎笘鐣岀綉缁滐級
        
        姣忎釜澶鍏虫敞2-4涓叾浠栧ぇV锛屽舰鎴愪俊鎭紶閫掔幆璺€?
        浣跨敤闅忔満鍥炬ā鎷熺ぞ浜ゅ獟浣撲笂鐨勪簰鍏冲叧绯汇€?
        
        Returns:
            Dict[agent_id, List[鍏虫敞鐨刟gent_id]]
        """
        if self.n_smart <= 0 or not self.smart_agents:
            return {}
        network = {}
        agent_ids = [a.id for a in self.smart_agents]
        
        for i, agent_id in enumerate(agent_ids):
            # 姣忎釜Agent鍏虫敞2-4涓叾浠朅gent
            max_follow = max(1, min(4, self.n_smart - 1))
            n_follow = random.randint(1, max_follow)
            others = [aid for aid in agent_ids if aid != agent_id]
            if not others:
                network[agent_id] = []
                continue
            
            # 鏉冮噸锛氬€惧悜浜庡叧娉ㄧ紪鍙风浉杩戠殑锛堟ā鎷熷湀瀛愭晥搴旓級
            weights = [1.0 / (1 + abs(j - i)) for j in range(len(others))]
            weights = [w / sum(weights) for w in weights]
            
            follows = random.choices(others, weights=weights, k=n_follow)
            network[agent_id] = list(set(follows))
        
        print(f"[OK] Smart Agent绀句氦缃戠粶鏋勫缓瀹屾垚锛屽钩鍧囧叧娉ㄦ暟: {sum(len(v) for v in network.values()) / len(network):.1f}")
        return network
    
    def get_social_influence_context(self, agent_id: str) -> Dict:
        """
        鑾峰彇鏌愪釜Smart Agent鐨勭ぞ浜ゅ奖鍝嶄笂涓嬫枃
        
        杩斿洖璇gent鎵€鍏虫敞鐨勫叾浠朅gent鐨勬渶杩戝喅绛栦俊鎭紝
        鐢ㄤ簬鍦╬rompt涓敞鍏ョぞ浜ゅ奖鍝嶅洜绱犮€?
        
        Args:
            agent_id: 鐩爣Agent ID
            
        Returns:
            Dict: 鍖呭惈绀句氦褰卞搷淇℃伅鐨勪笂涓嬫枃
        """
        followed = self.smart_social_network.get(agent_id, [])
        
        influences = []
        for fid in followed:
            if fid in self.last_smart_actions:
                action_info = self.last_smart_actions[fid]
                influences.append({
                    "agent_id": fid,
                    "action": action_info.get("action", "HOLD"),
                    "confidence": action_info.get("confidence", 0.5),
                    "emotion": action_info.get("emotion_score", 0.0)
                })
        
        # 璁＄畻绀句氦鍦堟暣浣撴儏缁?
        if influences:
            avg_emotion = sum(i["emotion"] for i in influences) / len(influences)
            bullish_ratio = sum(1 for i in influences if i["action"] == "BUY") / len(influences)
            bearish_ratio = sum(1 for i in influences if i["action"] == "SELL") / len(influences)
        else:
            avg_emotion = 0.0
            bullish_ratio = 0.0
            bearish_ratio = 0.0
        
        return {
            "followed_agents": influences,
            "circle_emotion": avg_emotion,
            "circle_bullish_ratio": bullish_ratio,
            "circle_bearish_ratio": bearish_ratio,
            "influence_strength": len(influences) / max(len(followed), 1)
        }
    
    def record_smart_action(self, agent_id: str, decision: Dict):
        """
        璁板綍Smart Agent鐨勫喅绛栵紝鐢ㄤ簬涓嬩竴杞殑绀句氦褰卞搷
        """
        self.last_smart_actions[agent_id] = {
            "action": decision.get("action", "HOLD"),
            "confidence": decision.get("confidence", 0.5),
            "emotion_score": decision.get("emotion_score", 0.0),
            "reasoning_summary": decision.get("reasoning", "")[:100]
        }

    def calculate_csad(self) -> float:
        """
        璁＄畻甯傚満鎯呯华鐨勪竴鑷存€?(Cross-Sectional Absolute Deviation)
        CSAD 瓒婁綆锛岃鏄庢暎鎴锋儏缁秺瓒嬪悓锛岃秺瀹规槗鍙戠敓缇婄兢鏁堝簲銆?
        """
        sentiments = self.state[:, self.IDX_SENTIMENT]
        mean_sentiment = np.mean(sentiments)
        # 缁濆鍋忓樊鐨勫钩鍧囧€?
        csad = np.mean(np.abs(sentiments - mean_sentiment))
        return csad

    def update_tier2_sentiment(self, smart_actions: List[int], market_trend: float):
        """
        [鍚戦噺鍖朷 娑岀幇寮忔儏缁紶鏌撴洿鏂?
        
        閲囩敤鍩轰簬閭诲眳缃戠粶鐨勬秾鐜版満鍒讹紝鏇夸唬纭紪鐮侀槇鍊硷細
        - 姣忎釜鏁ｆ埛瑙傚療鍏堕偦灞呯殑鎯呯华鐘舵€?
        - 褰撻偦灞呮亹鎱屾瘮渚嬭秴杩囬槇鍊兼椂锛岃"鎰熸煋"
        - 涓嶅悓璁ょ煡绫诲瀷瀵逛笉鍚屼俊鍙风殑鍝嶅簲鏉冮噸涓嶅悓
        
        Parameters:
            smart_actions: Tier 1 鐨勬搷浣滃垪琛?(1=Buy, -1=Sell, 0=Hold)
            market_trend: 甯傚満瓒嬪娍淇″彿 (-1.0 ~ 1.0)
        """
        # 1. 鑾峰彇鏉ヨ嚜澶鐨勪俊鍙?(Local Signal)
        # 维度防御：部分场景下 influence_map 可能与 state 行数不一致，这里统一对齐
        vector_size = int(self.state.shape[0])
        self.n_vectorized = vector_size
        smart_acts = np.array(smart_actions, dtype=np.int32).reshape(-1)
        if smart_acts.size == 0:
            smart_acts = np.array([0], dtype=np.int32)
        influence = np.array(self.influence_map, dtype=np.int32).reshape(-1)
        if influence.size < vector_size:
            pad = np.random.choice(influence if influence.size > 0 else np.array([0]), size=vector_size - influence.size)
            influence = np.concatenate([influence, pad])
        elif influence.size > vector_size:
            influence = influence[:vector_size]
        influence = np.clip(influence, 0, smart_acts.size - 1)
        guru_signals = smart_acts[influence]
        
        # 2. 璁＄畻閭诲眳鎯呯华鐘舵€侊紙娑岀幇寮忕緤缇ゆ晥搴旀牳蹇冿級
        current_sentiment = self.state[:, self.IDX_SENTIMENT]
        neighbor_indices = self.neighbor_network  # (N, n_neighbors)
        
        # 鑾峰彇姣忎釜鑺傜偣鐨勯偦灞呮儏缁?
        neighbor_sentiments = current_sentiment[neighbor_indices]  # (N, n_neighbors)
        
        # 璁＄畻閭诲眳鎭愭厡姣斾緥锛堟儏缁?< -0.3 瑙嗕负鎭愭厡锛?
        neighbor_panic_ratio = np.mean(neighbor_sentiments < -0.3, axis=1)  # (N,)
        
        # 璁＄畻閭诲眳骞冲潎鎯呯华锛堢敤浜庝俊鍙蜂紶閫掞級
        neighbor_avg_sentiment = np.mean(neighbor_sentiments, axis=1)
        
        # 3. 鍩轰簬璁ょ煡绫诲瀷鐨勬潈閲嶅垎閰?
        cognitive_types = self.state[:, self.IDX_COGNITIVE_TYPE]
        confidence = self.state[:, self.IDX_CONFIDENCE] / 100.0  # 褰掍竴鍖栧埌 0-1
        
        # 鍩虹鏉冮噸
        w_personal = np.full(vector_size, 0.4)
        w_guru = np.full(vector_size, 0.2)
        w_neighbor = np.full(vector_size, 0.2)
        w_market = np.full(vector_size, 0.2)
        
        # 鎶€鏈淳 (type=0): 鏇翠緷璧栧競鍦鸿秼鍔匡紝杈冨皯鍙楅偦灞呭奖鍝?
        tech_mask = cognitive_types == 0
        w_personal[tech_mask] = 0.3
        w_market[tech_mask] = 0.4
        w_neighbor[tech_mask] = 0.1
        w_guru[tech_mask] = 0.2
        
        # 娑堟伅娲?(type=1): 鏇翠緷璧栧ぇV淇″彿
        news_mask = cognitive_types == 1
        w_personal[news_mask] = 0.2
        w_guru[news_mask] = 0.4
        w_neighbor[news_mask] = 0.2
        w_market[news_mask] = 0.2
        
        # 璺熼娲?(type=2): 鏇村鏄撳彈閭诲眳褰卞搷
        herd_mask = cognitive_types == 2
        w_personal[herd_mask] = 0.1
        w_neighbor[herd_mask] = 0.5
        w_guru[herd_mask] = 0.2
        w_market[herd_mask] = 0.2
        
        # 4. 娑岀幇寮忕緤缇ゆ晥搴旓細閭诲眳鎭愭厡鏃讹紝鏉冮噸鍔ㄦ€佽皟鏁?
        # 褰撹秴杩?0%鐨勯偦灞呮亹鎱屾椂锛屼釜浜哄垽鏂澶у箙鍓婂急
        panic_threshold = 0.6
        panic_mask = neighbor_panic_ratio > panic_threshold
        
        # 鎭愭厡浼犳煋绯绘暟锛堜俊蹇冧綆鐨勬洿瀹规槗琚紶鏌擄級
        susceptibility = (1.0 - confidence) * 0.8 + 0.2  # 0.2 ~ 1.0
        
        # 璋冩暣鏉冮噸
        w_personal[panic_mask] *= (1.0 - susceptibility[panic_mask] * 0.7)
        w_neighbor[panic_mask] += susceptibility[panic_mask] * 0.4
        
        # 褰掍竴鍖栨潈閲?
        total_w = w_personal + w_guru + w_neighbor + w_market
        w_personal /= total_w
        w_guru /= total_w
        w_neighbor /= total_w
        w_market /= total_w
        
        # 5. 鎵归噺鏇存柊鎯呯华 (Matrix Operation)
        noise = np.random.normal(0, 0.05, vector_size)
        
        new_sentiment = (
            w_personal * current_sentiment +
            w_guru * guru_signals +
            w_neighbor * neighbor_avg_sentiment +
            w_market * market_trend +
            noise
        )
        
        # 鎴柇鍒?[-1, 1]
        self.state[:, self.IDX_SENTIMENT] = np.clip(new_sentiment, -1.0, 1.0)
        
        # 6. 鏇存柊淇″績鎸囨暟锛堢粡鍘嗘亹鎱屽悗淇″績涓嬮檷锛?
        confidence_change = np.where(
            panic_mask,
            -5 * susceptibility,  # 鎭愭厡涓俊蹇冨揩閫熶笅闄?
            0.5  # 姝ｅ父鎯呭喌缂撴參鎭㈠
        )
        self.state[:, self.IDX_CONFIDENCE] = np.clip(
            self.state[:, self.IDX_CONFIDENCE] + confidence_change, 0, 100
        )

    def generate_tier2_decisions(self, current_price: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [鍚戦噺鍖朷 鐢熸垚鏁ｆ埛浜ゆ槗鍐崇瓥
        閬垮厤 10,000 娆?if-else锛岀洿鎺ョ敤姒傜巼鐭╅樀鐢熸垚 mask銆?
        
        Returns:
            actions: (N,) {-1, 0, 1}
            quantities: (N,)
            prices: (N,) 鎸傚崟浠锋牸
        """
        self.update_behavioral_layer(current_price)
        intent = self.trading_intent_state
        risk_appetite = self.risk_appetite_state
        
        # 1. 鐢熸垚涔板崠姒傜巼
        # 鎯呯华 > 0.6 -> 楂樻鐜囦拱鍏? 鎯呯华 < -0.6 -> 楂樻鐜囧崠鍑?
        prob_buy = 1 / (1 + np.exp(-5 * (intent - 0.15))) # Sigmoid shift
        prob_sell = 1 / (1 + np.exp(5 * (intent + 0.15)))
        
        # 闅忔満楠板瓙
        rng = np.random.random(self.n_vectorized)
        
        # 鐢熸垚鍔ㄤ綔 Mask
        buy_mask = rng < prob_buy
        sell_mask = (rng > (1 - prob_sell)) & (~buy_mask) # 浜掓枼
        
        actions = np.zeros(self.n_vectorized, dtype=int)
        actions[buy_mask] = 1
        actions[sell_mask] = -1
        
        # 2. 璁＄畻鏁伴噺 (绠€鍗曠殑璧勯噾姣斾緥娉?
        quantities = np.zeros(self.n_vectorized, dtype=int)
        
        # 涔板叆: 浣跨敤 20%~50% 鍙敤璧勯噾
        buy_ratio = np.random.uniform(0.15, 0.45, size=self.n_vectorized) * (0.7 + 0.6 * risk_appetite)
        avail_cash = self.state[:, self.IDX_CASH]
        # 鍚戦噺鍖栬绠? 璧勯噾 * 姣斾緥 / 鍗曚环 (鍚戜笅鍙栨暣)
        raw_buy_qty = (avail_cash * buy_ratio / current_price).astype(int)
        quantities[buy_mask] = raw_buy_qty[buy_mask]
        
        # 鍗栧嚭: 浣跨敤 50%~100% 鎸佷粨
        sell_ratio = np.random.uniform(0.4, 1.0, size=self.n_vectorized) * (0.8 + 0.4 * np.abs(np.minimum(intent, 0.0)))
        avail_holdings = self.state[:, self.IDX_HOLDINGS]
        raw_sell_qty = (avail_holdings * sell_ratio).astype(int)
        quantities[sell_mask] = raw_sell_qty[sell_mask]
        
        # 3. 杩囨护鏃犳晥鍗?(鏁伴噺涓?)
        valid_mask = quantities > 0
        actions[~valid_mask] = 0
        quantities[~valid_mask] = 0
        
        # 4. 鐢熸垚鎸傚崟浠锋牸 (鍦ㄧ幇浠烽檮杩戞尝鍔?
        # 鏁ｆ埛閫氬父鎸傚競浠锋垨鐣ュソ鐨勪环鏍?
        price_noise = np.random.normal(0, 0.002, self.n_vectorized)
        order_prices = current_price * (1 + price_noise)
        
        return actions, quantities, order_prices

    def sync_tier2_execution(self, executed_indices: np.ndarray, executed_prices: np.ndarray, 
                             executed_qtys: np.ndarray, directions: np.ndarray):
        """
        [鍚戦噺鍖朷 鎴愪氦鍥炴墽澶勭悊
        褰撴挳鍚堝紩鎿庢垚浜ゅ悗锛屾壒閲忔洿鏂扮姸鎬佺煩闃点€?
        """
        if len(executed_indices) == 0:
            return
            
        cost_val = executed_prices * executed_qtys
        
        # 鏇存柊璧勯噾 (涔板叆鍑忥紝鍗栧嚭鍔?
        delta_cash = -1 * directions * cost_val
        # FIX: 浣跨敤 add.at 澶勭悊閲嶅绱㈠紩 (鍚屼竴Agent澶氱瑪鎴愪氦)
        np.add.at(self.state[:, self.IDX_CASH], executed_indices, delta_cash)
        
        # 鏇存柊鎸佷粨
        delta_stock = directions * executed_qtys
        np.add.at(self.state[:, self.IDX_HOLDINGS], executed_indices, delta_stock)
        
        # 鏇存柊鎴愭湰浠?(浠呬拱鍏ユ椂鏇存柊鍔犳潈骞冲潎)
        buy_indices_local = (directions == 1)
        if np.any(buy_indices_local):
            # 鑾峰彇鍏ㄥ眬绱㈠紩
            g_idx = executed_indices[buy_indices_local]
            b_qty = executed_qtys[buy_indices_local]
            b_prc = executed_prices[buy_indices_local]
            
            # 鍘熷鎸佷粨鍜屾垚鏈?
            # 淇: 搴旇鐢ㄦ洿鏂板墠鐨勬暟閲忋€傜敱浜庝笂闈㈠凡缁忓姞浜嗭紝杩欓噷瑕佸噺鍥炲幓绠楁棫鐨?
            cur_qty = self.state[g_idx, self.IDX_HOLDINGS]
            prev_qty = cur_qty - b_qty
            prev_cost = self.state[g_idx, self.IDX_COST]
            
            # 鍔犳潈骞冲潎鍏紡
            # (OldQty * OldCost + BuyQty * BuyPrice) / NewQty
            # 閬垮厤闄や互闆?
            denom = np.where(cur_qty > 0, cur_qty, 1.0)
            new_cost_basis = ((prev_qty * prev_cost) + (b_qty * b_prc)) / denom
            
            self.state[g_idx, self.IDX_COST] = new_cost_basis
