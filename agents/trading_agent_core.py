"""
Civitas-Economica 核心交易智能体模块 (Trading Agent Module)

按照架构的严格面向对象设计 (Strict OOD) 原则：
隔离各模块的职责，如 engine/ (撮合), agents/ (LLM 包装、画像、记忆), policy/ (宏观事件), utils/。

此模块包含基于 Pydantic 的 Persona 数据模型、集成了 ChromaDB 与 LangChain 的 TradingAgent 类，
以及负责人群生成的 AgentFactory 类。
所有代码符合 Python 3.11+, asyncio 标准以及 PEP 8 与 Google-style Docstrings。
"""

import asyncio
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# 配置模块级别的 Logger (按照要求提供清晰的日志追踪)
logger = logging.getLogger("civitas.agents.trading_agent")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(sh)


# ----------------------------------------------------------------------------
# 1. 配置文件加载 (Configuration Driven - 绝对消除魔法数字)
# ----------------------------------------------------------------------------
def _load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载基于 YAML 的全局配置参数文件。

    Args:
        path (str): 配置文件路径，默认到项目根目录下的 config.yaml

    Returns:
        Dict[str, Any]: 解析后的配置字典
    """
    config_path = Path(__file__).parent.parent / path
    if not config_path.exists():
        logger.warning(f"Configuration file {config_path} not found. Using safe fallbacks.")
        return {
            "llm_temperature": 0.7,
            "market_slippage": 0.001,
            "transaction_costs": 0.0005,
            "ebbinghaus_base_decay_rate": 0.1
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

GLOBAL_CONFIG = _load_config()


# ----------------------------------------------------------------------------
# 2. Pydantic 数据模型与枚举类型定义 (Type Hinting & strict validation)
# ----------------------------------------------------------------------------
class CognitiveBias(str, Enum):
    Loss_Aversion = "Loss_Aversion"
    Herding = "Herding"
    Overconfidence = "Overconfidence"
    Rational = "Rational"


class InvestmentHorizon(str, Enum):
    Short_term = "Short_term"
    Medium_term = "Medium_term"
    Long_term = "Long_term"


class Persona(BaseModel):
    """
    用于定义交易智能体心理和策略特征的严格数据模型。
    
    采用 Pydantic 进行基于类型的字段验证。
    """
    model_config = ConfigDict(strict=True)

    risk_tolerance: float = Field(
        ..., ge=0.0, le=1.0, 
        description="代理的风险承受能力 (0.0=极度厌恶风险, 1.0=极端赌徒)"
    )
    cognitive_bias: CognitiveBias = Field(
        ..., 
        description="占主导地位的认知偏差，影响 LLM 对市场信息的解释"
    )
    investment_horizon: InvestmentHorizon = Field(
        ..., 
        description="投资的时间跨度偏好"
    )
    policy_sensitivity: float = Field(
        ..., ge=0.0, le=1.0,
        description="对宏观经济政策和新闻的敏感度"
    )

    def describe(self) -> str:
        """生成供 LLM 提示词 (Prompt) 使用的人格自述文本。"""
        return (f"您的风险承受度为 {self.risk_tolerance} (0-1)。"
                f"您的核心认知偏差是 {self.cognitive_bias.value}。"
                f"您的投资周期专注于 {self.investment_horizon.value}。"
                f"您对宏观政策事件敏感度指标为 {self.policy_sensitivity}。")


class TradeAction(BaseModel):
    """
    LLM 计划输出的标准化交易决策抽象类。
    """
    action: str = Field(..., description="如 BUY, SELL 或 HOLD")
    amount: float = Field(..., description="拟交易的资产数量或资金等价物")
    target_price: Optional[float] = Field(None, description="限价单的目标价格（空为市价）")
    reasoning_chain: str = Field(..., description="基于上下文进行决策的思考过程摘要")


# ----------------------------------------------------------------------------
# 3. 核心实体类：TradingAgent
# ----------------------------------------------------------------------------
class TradingAgent:
    """
    Civitas 经济体中的核心交易智能体。

    不仅具有基础的聊天功能，还是一个具有心理和战略约束的专业金融实体。
    通过 LangChain/原生 LLM API 和 ChromaDB 进行市场分析与记忆存储。
    """

    def __init__(self, agent_id: str, persona: Persona):
        """
        初始化交易代理。

        Args:
            agent_id (str): 系统中独一无二的智能体标识符。
            persona (Persona): 限定该智能体行为逻辑的心理与策略约束模型。
        """
        self.agent_id = agent_id
        self.persona = persona

        # 用于持有与市场交互所需的资金/资产等内部状态
        self.portfolio: Dict[str, float] = {}
        self.cash: float = GLOBAL_CONFIG.get("default_starting_cash", 100000.0)

        # 初始化基于 ChromaDB 包装的 Ebbinghaus 认知记忆银行
        # 延迟到这里局部导入以避免潜在的循环依赖引用
        from agents.ebbinghaus_memory import EbbinghausMemoryBank
        self.memory_bank = EbbinghausMemoryBank(agent_id=self.agent_id)

        logger.info(f"Initialized TradingAgent {self.agent_id} with Persona: "
                    f"[{self.persona.cognitive_bias.value}, Risk: {self.persona.risk_tolerance}]")

    async def retrieve_context(self, current_market_data: Any) -> str:
        """
        异步从 ChromaDB 或本地长期记忆中根据上下文获取最相关的历史信息。

        Args:
            current_market_data (Any): 当期市场行情及宏观情绪。
            
        Returns:
            str: 用于注入给 Prompt 的上下文关联记忆字符串。
        """
        # TODO: 实际与 ChromaDB (Vector Store) 联通
        # 此处使用简单的模拟返回
        await asyncio.sleep(0.01)  # 模拟异步 IO
        decay_rate = GLOBAL_CONFIG.get("ebbinghaus_base_decay_rate", 0.1)
        return f"基于艾宾浩斯认知衰减 ({decay_rate})，之前记忆最深的损失：股票近期发生过大幅回撤。"

    async def generate_trading_decision(self, market_data: Dict[str, Any], retrieved_context: str) -> TradeAction:
        """
        使用带有严格 Pydantic 输出解析的 LLM 模型来产生异步的交易指令。
        
        骨架方法：系统 Prompt 会被动态注入 Persona 的约束参数，再结合环境数据
        与已检索的 ChromaDB 长期记忆让 LLM 一起推演决定。

        Args:
            market_data (Dict[str, Any]): 系统提供的当前市场报价和指令簿深度等信息。
            retrieved_context (str): 智能体对过往市场的自我历史记忆及知识截面。

        Returns:
            TradeAction: Pydantic 经过验证的数据结构实例，用来喂给撮合引擎 (Engine)。
        """
        logger.debug(f"[{self.agent_id}] 开始深入思考 (Reasoning) 并生成交易决策...")

        # --- 动态生成的 System Prompt，预先设定身份 ---
        llm_temp = GLOBAL_CONFIG.get("llm_temperature", 0.7)
        prompt_template = f"""
您是 Civitas-Economica 系统内高度特化的金融交易智能体。请严格基于自己特定的人格属性制定交易决策。

【你的心理与行为特征】
{self.persona.describe()}

【历史记忆截面 (ChromaDB Vector)】
{retrieved_context}

【即时市场与行情数据】
{json.dumps(market_data, indent=2, ensure_ascii=False)}

请作为一个 {self.persona.cognitive_bias.value} 型投资者并结合宏观经济情况进行 {self.persona.investment_horizon.value} 分析。
思考过程中请始终记住：您的决策不能明显违背您的「风险承受度」。

请您严格按照给定的 JSON Schema 返回结果：包含 action (BUY/SELL/HOLD), amount, target_price，
以及 reasoning_chain。
"""
        # 注意: 实际使用 LangChain 时，这里可以调用 `LLMChain` 或 `chat_model.agenerate`。
        # 这里为保持骨架，使用异步延时模拟 LLM 请求响应，并强行构造符合 Pydantic 的返回值。
        
        await asyncio.sleep(0.1)  # 模拟 LLM 推理异步调度耗时
        
        try:
            # 模拟 LLM 经过输出解析器(OutputParser)后得到的数据提取过程
            # 现实操作中，这里是从 LLM 返回文本中提取出的 JSON/Dict
            simulated_response = {
                "action": "BUY" if self.persona.risk_tolerance > 0.5 else "HOLD",
                "amount": self.cash * 0.1,  # 使用10%仓位
                "target_price": market_data.get("current_price", 10.0) * (1 - GLOBAL_CONFIG.get("market_slippage", 0.001)),
                "reasoning_chain": f"作为 {self.persona.cognitive_bias.value} 倾向的参与者，我受限分析了上述行情，认为这符合策略。"
            }
            
            # 使用 Pydantic 的严密校验解析字典。若字段缺失或类别错误将抛出 ValidationError
            action_decision = TradeAction(**simulated_response)
            
            logger.info(f"[{self.agent_id}] 决定执行: {action_decision.action} {action_decision.amount}")
            return action_decision
            
        except ValidationError as ve:
            logger.error(f"[{self.agent_id}] 遇到了 LLM 输出格式幻觉 (Hallucination) 导致 Pydantic 校验失败: {ve}")
            # Fallback 策略：一旦发现解析错误，自动回退到 HOLD 操作，保证仿真程序不崩溃
            return TradeAction(
                action="HOLD", 
                amount=0.0, 
                reasoning_chain="LLM 输出不合规或发生幻觉，已被安全回落机制拦截，执行 HOLD 策略。"
            )
        except Exception as e:
            logger.error(f"[{self.agent_id}] 生成交易决策发生未知异常: {e}")
            return TradeAction(action="HOLD", amount=0.0, reasoning_chain="未知错误导致的系统保护性 HOLD。")


# ----------------------------------------------------------------------------
# 4. 工厂模式设计：AgentFactory
# ----------------------------------------------------------------------------
class AgentFactory:
    """
    负责批量、可控、具统计学意义地创建多种类型交易代理的工厂类。
    
    在启动 Civitas 时，工厂通过调整 Persona 参数的联合分布，生成具备社会分工差异化的人口基础。
    """

    @staticmethod
    def create_diverse_population(num_agents: int) -> List[TradingAgent]:
        """
        生成在统计上逼真的群体。
        例如设定为:
        - 60% 高神经质的噪音交易者 (High-neuroticism noise traders)
        - 30% 低神经质的基本面分析师 (Low-neuroticism fundamental analysts)
        - 10% 宏观对冲基金 (Macro hedge funds)

        Args:
            num_agents (int): 被生成并注入系统仿真空间的总人数。

        Returns:
            List[TradingAgent]: 多样化的并配备不同有效 Persona 心理的 Agent 列表。
        """
        population: List[TradingAgent] = []
        
        count_noise_traders = int(num_agents * 0.6)
        count_fundamental = int(num_agents * 0.3)
        count_hedge = num_agents - count_noise_traders - count_fundamental  # 剩下的10%

        idx = 1
        
        # 1. 产生 60% 的噪音交易员 (高风险倾向，容易跟风或过度自信，短视，受政策影响敏感)
        for _ in range(count_noise_traders):
            persona = Persona(
                risk_tolerance=0.8,
                cognitive_bias=CognitiveBias.Herding,  # 或 Overconfidence
                investment_horizon=InvestmentHorizon.Short_term,
                policy_sensitivity=0.9
            )
            population.append(TradingAgent(agent_id=f"NoiseTrader_{idx}", persona=persona))
            idx += 1
            
        # 2. 产生 30% 的基本面分析师 (低风险/适度风险倾向，理性分析为主，长视，对短期政策敏感较低)
        for _ in range(count_fundamental):
            persona = Persona(
                risk_tolerance=0.3,
                cognitive_bias=CognitiveBias.Rational,
                investment_horizon=InvestmentHorizon.Long_term,
                policy_sensitivity=0.3
            )
            population.append(TradingAgent(agent_id=f"FundamentalAnalyst_{idx}", persona=persona))
            idx += 1
            
        # 3. 产生 10% 宏观对冲基金 (高风险，极其关注政策面，损失厌恶用于风控保护)
        for _ in range(count_hedge):
            persona = Persona(
                risk_tolerance=0.7,
                cognitive_bias=CognitiveBias.Loss_Aversion,
                investment_horizon=InvestmentHorizon.Medium_term,
                policy_sensitivity=0.95
            )
            population.append(TradingAgent(agent_id=f"MacroHedgeFund_{idx}", persona=persona))
            idx += 1

        logger.info(f"成功生成了 {len(population)} 名在统计学上呈特化分布的智能体 (Agents)!")
        return population

