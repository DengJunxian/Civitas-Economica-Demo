# file: core/model_router.py
"""
多模型路由器 - 统一管理 DeepSeek 和 混元 API 调用

设计原则：
1. DeepSeek 为必选，混元为可选增强
2. 支持自动降级和故障转移
3. 实时监控响应时间，智能调度
"""

import asyncio
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI, AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from config import GLOBAL_CONFIG


class ModelType(Enum):
    """模型类型枚举"""
    REASONER = "reasoner"  # 推理模型（CoT）
    CHAT = "chat"  # 对话模型
    FLASH = "flash"  # 快速模型


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    provider: str  # "deepseek" | "hunyuan"
    model_type: ModelType
    base_url: str
    timeout: float
    
    
@dataclass
class ModelStats:
    """模型统计信息"""
    call_count: int = 0
    total_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    
    @property
    def avg_time(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count
    
    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 1.0
        return 1.0 - (self.error_count / self.call_count)


class ModelRouter:
    """
    多模型路由器
    
    职责：
    1. 管理多个API客户端
    2. 根据优先级和可用性选择模型
    3. 自动降级和故障转移
    4. 统计响应时间用于智能调度
    """
    
    # 模型注册表
    MODEL_REGISTRY: Dict[str, ModelInfo] = {
        "deepseek-reasoner": ModelInfo(
            name="deepseek-reasoner",
            provider="deepseek",
            model_type=ModelType.REASONER,
            base_url=GLOBAL_CONFIG.API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER
        ),
        "deepseek-chat": ModelInfo(
            name="deepseek-chat",
            provider="deepseek",
            model_type=ModelType.CHAT,
            base_url=GLOBAL_CONFIG.API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_CHAT
        ),
        "hunyuan-t1-latest": ModelInfo(
            name="hunyuan-t1-latest",
            provider="hunyuan",
            model_type=ModelType.REASONER,
            base_url=GLOBAL_CONFIG.HUNYUAN_API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER
        ),
        "hunyuan-turbos-latest": ModelInfo(
            name="hunyuan-turbos-latest",
            provider="hunyuan",
            model_type=ModelType.CHAT,
            base_url=GLOBAL_CONFIG.HUNYUAN_API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_CHAT
        ),
        # 智谱GLM模型 (快速模式专用)
        "glm-4-flashx": ModelInfo(
            name="glm-4-flashx",
            provider="zhipu",
            model_type=ModelType.FLASH,
            base_url=GLOBAL_CONFIG.ZHIPU_API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_FLASH
        ),
        "glm-4-flashx-250414": ModelInfo(
            name="glm-4-flashx-250414",
            provider="zhipu",
            model_type=ModelType.FLASH,
            base_url=GLOBAL_CONFIG.ZHIPU_API_BASE_URL,
            timeout=GLOBAL_CONFIG.API_TIMEOUT_FLASH
        ),
    }
    
    def __init__(self, deepseek_key: str, hunyuan_key: Optional[str] = None, zhipu_key: Optional[str] = None):
        """
        初始化路由器
        
        Args:
            deepseek_key: DeepSeek API密钥（可选，为空时使用智谱降级）
            hunyuan_key: 混元API密钥（可选）
            zhipu_key: 智谱API密钥（可选，快速模式专用或DeepSeek降级）
        """
        self.deepseek_key = deepseek_key
        self.hunyuan_key = hunyuan_key
        self.zhipu_key = zhipu_key
        
        # 初始化客户端
        self.clients: Dict[str, AsyncOpenAI] = {}
        self._init_clients()
        
        # 统计信息
        self.stats: Dict[str, ModelStats] = {
            model: ModelStats() for model in self.MODEL_REGISTRY
        }
        
        # 可用模型列表（基于API密钥）
        self.available_models = self._get_available_models()
        
        # 降级事件记录（用于前端弹窗提示）
        self.fallback_events: List[Dict] = []
        
        # 标记是否有DeepSeek可用
        self.has_deepseek = bool(deepseek_key)
        self.has_zhipu = bool(zhipu_key)
        
    def _init_clients(self):
        """初始化API客户端"""
        # DeepSeek 客户端（必需）
        if self.deepseek_key:
            self.clients["deepseek"] = AsyncOpenAI(
                api_key=self.deepseek_key,
                base_url=GLOBAL_CONFIG.API_BASE_URL,
                timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER
            )
        
        # 混元客户端（可选）
        if self.hunyuan_key:
            self.clients["hunyuan"] = AsyncOpenAI(
                api_key=self.hunyuan_key,
                base_url=GLOBAL_CONFIG.HUNYUAN_API_BASE_URL,
                timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER
            )
        
        # 智谱客户端（可选，快速模式专用）
        if self.zhipu_key:
            self.clients["zhipu"] = AsyncOpenAI(
                api_key=self.zhipu_key,
                base_url=GLOBAL_CONFIG.ZHIPU_API_BASE_URL,
                timeout=GLOBAL_CONFIG.API_TIMEOUT_FLASH
            )
    
    def _get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        available = []
        
        # DeepSeek 模型（必需）
        if self.deepseek_key:
            available.extend(["deepseek-reasoner", "deepseek-chat"])
        
        # 混元模型（可选）
        if self.hunyuan_key:
            available.extend(["hunyuan-t1-latest", "hunyuan-turbos-latest"])
        
        # 智谱GLM模型（可选，快速模式专用）
        if self.zhipu_key:
            available.extend(["glm-4-flashx", "glm-4-flashx-250414"])
        
        return available
    
    def get_model_priority(self, mode: str) -> List[str]:
        """
        根据模式获取模型优先级列表
        
        Args:
            mode: "SMART" | "FAST" | "DEEP"
            
        Returns:
            按优先级排序的可用模型列表
        """
        if mode == "FAST":
            # 快速模式：仅对话模型
            priority = ["deepseek-chat"]
        elif mode == "DEEP":
            # 深度模式：仅推理模型
            priority = ["deepseek-reasoner"]
        else:  # SMART
            # 智能模式：混合列表 (具体分配在Model层)
            priority = ["deepseek-reasoner", "deepseek-chat"]
        
        # 过滤出可用模型
        return [m for m in priority if m in self.available_models]
    
    async def call_with_fallback(
        self,
        messages: List[Dict],
        priority_models: List[str],
        timeout_budget: float = 15.0,
        fallback_response: Optional[str] = None
    ) -> Tuple[str, Optional[str], str]:
        """
        带降级的模型调用
        
        Args:
            messages: 对话消息列表
            priority_models: 按优先级排序的模型列表
            timeout_budget: 总超时预算（秒）
            
        Returns:
            (content, reasoning_content, model_used)
        """
        start_time = time.time()
        last_error = None
        
        for model_name in priority_models:
            # 检查剩余时间
            elapsed = time.time() - start_time
            remaining = timeout_budget - elapsed
            
            if remaining <= 1.0:
                # 时间不足，使用最后一个模型的降级响应
                break
            
            model_info = self.MODEL_REGISTRY.get(model_name)
            if not model_info:
                continue
            
            client = self.clients.get(model_info.provider)
            if not client:
                continue
            
            # 计算此次调用的超时
            call_timeout = min(model_info.timeout, remaining)
            
            try:
                result = await self._call_model(
                    client, model_name, messages, call_timeout
                )
                
                # 更新统计
                call_time = time.time() - start_time
                self._update_stats(model_name, call_time, success=True)
                
                return result + (model_name,)
                
            except asyncio.TimeoutError:
                last_error = f"{model_name}: 超时"
                self._update_stats(model_name, 0, success=False, error="timeout")
                
                # 记录降级事件（用于前端提示）
                if "deepseek" in model_name and len(priority_models) > 1:
                    next_model = priority_models[priority_models.index(model_name) + 1] if priority_models.index(model_name) + 1 < len(priority_models) else "fallback"
                    self.fallback_events.append({
                        "type": "timeout_fallback",
                        "from_model": model_name,
                        "to_model": next_model,
                        "timestamp": time.time(),
                        "message": f"DeepSeek推理超时，已自动降级到{next_model}"
                    })
                continue
                
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = f"{model_name}: {type(e).__name__}"
                self._update_stats(model_name, 0, success=False, error=str(e))
                
                # 记录降级事件
                if "deepseek" in model_name:
                    self.fallback_events.append({
                        "type": "api_error_fallback",
                        "from_model": model_name,
                        "error": type(e).__name__,
                        "timestamp": time.time(),
                        "message": f"DeepSeek API错误({type(e).__name__})，已自动降级"
                    })
                continue
                
            except Exception as e:
                last_error = f"{model_name}: {str(e)}"
                self._update_stats(model_name, 0, success=False, error=str(e))
                continue
        
        # 全部失败，返回降级响应
        return self._fallback_response(last_error, fallback_content=fallback_response)
    
    async def _call_model(
        self,
        client: AsyncOpenAI,
        model_name: str,
        messages: List[Dict],
        timeout: float
    ) -> Tuple[str, Optional[str]]:
        """
        调用单个模型
        
        Returns:
            (content, reasoning_content)
        """
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.6
            ),
            timeout=timeout
        )
        
        message = response.choices[0].message
        content = message.content or ""
        
        # 提取推理内容（如果有）
        reasoning = getattr(message, 'reasoning_content', None)
        
        return content, reasoning
    
    def _update_stats(
        self, 
        model_name: str, 
        call_time: float, 
        success: bool,
        error: Optional[str] = None
    ):
        """更新模型统计信息"""
        stats = self.stats.get(model_name)
        if stats:
            stats.call_count += 1
            if success:
                stats.total_time += call_time
            else:
                stats.error_count += 1
                stats.last_error = error
    
    def _fallback_response(
        self, 
        error: Optional[str] = None,
        fallback_content: Optional[str] = None
    ) -> Tuple[str, Optional[str], str]:
        """降级响应"""
        content = fallback_content if fallback_content else '{"action": "HOLD", "qty": 0, "error": "MODEL_UNAVAILABLE"}'
        return (
            content,
            f"所有模型调用失败: {error}" if error else "所有模型不可用",
            "fallback"
        )
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            model: {
                "calls": stats.call_count,
                "avg_time": f"{stats.avg_time:.2f}s",
                "success_rate": f"{stats.success_rate:.1%}",
                "last_error": stats.last_error
            }
            for model, stats in self.stats.items()
            if stats.call_count > 0
        }
    
    def get_recommended_model(self, time_budget: float) -> str:
        """
        根据时间预算推荐模型
        
        基于历史响应时间，选择最可能在预算内完成的模型。
        """
        candidates = []
        
        for model_name in self.available_models:
            stats = self.stats.get(model_name)
            if stats and stats.call_count > 0:
                # 使用历史平均时间 + 20% 余量
                estimated_time = stats.avg_time * 1.2
                if estimated_time <= time_budget:
                    candidates.append((model_name, estimated_time))
            else:
                # 新模型，使用默认估计
                model_info = self.MODEL_REGISTRY.get(model_name)
                if model_info:
                    if model_info.model_type == ModelType.REASONER:
                        estimated_time = 15.0  # 推理模型默认估计
                    else:
                        estimated_time = 3.0  # 对话模型默认估计
                    if estimated_time <= time_budget:
                        candidates.append((model_name, estimated_time))
        
        if not candidates:
            # 没有合适的，返回最快的
            return "deepseek-chat" if "deepseek-chat" in self.available_models else self.available_models[0]
        
        # 按预估时间排序，返回最快的
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
