# file: data_flywheel/nlp_processor.py
import json
import logging
import re
from typing import List, Optional, Dict, Any

from core.model_router import ModelRouter
from data_flywheel.schemas import RawArticle, SeedEvent, ExtractedEntity

logger = logging.getLogger(__name__)


class NlpProcessor:
    """
    LLM 驱动的 NLP 处理器
    
    复用 ModelRouter 进行大模型调用，对外源文本进行实体提取与情感分析。
    """

    # 抽取与分析的系统提示词
    SYSTEM_PROMPT = """你是一位资深的金融情报分析师。
你的任务是阅读给定的财经类短新闻或事件文本，并提取关键实体及其影响。
请严格按照以下 JSON Schema 输出，不要输出任何其他内容。

{
  "entities": [
    {
      "name": "实体名称（如 中国人民银行, 房地产, CPI）",
      "entity_type": "company / sector / indicator / policy / person",
      "confidence": 0.8
    }
  ],
  "sentiment": 0.5, // 情感极性分数 (-1.0 到 1.0 的浮点数，-1代表极度利空，1代表极度利好，0代表中性)
  "sentiment_label": "利好 / 利空 / 中性",
  "impact_level": "low / medium / high / critical",
  "affected_sectors": ["受影响板块1", "受影响板块2"],
  "summary": "不超过50字的事件一句话摘要"
}
"""

    def __init__(self, model_router: ModelRouter):
        """
        Args:
            model_router: 传入已初始化的 ModelRouter 实例
        """
        self.router = model_router

    def process(self, article: RawArticle) -> SeedEvent:
        """
        处理单篇文章，利用 LLM 进行信息抽取
        """
        logger.info(f"Processing NLP for article: {article.title[:20]}...")
        
        user_prompt = f"标题：{article.title}\n文本内容：{article.content}"
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # 优先使用 chat 模型（速度快，JSON 能力强），降级至 flash 模型
            content, _, model_used = self.router.sync_call_with_fallback(
                messages=messages,
                priority_models=["deepseek-chat", "glm-4-flashx", "deepseek-reasoner"],
                timeout_budget=20.0
            )

            # 解析 JSON 响应
            data = self._parse_json_response(content)
            if data:
                return self._build_seed_event(article, data)
            else:
                logger.warning(f"Failed to parse LLM output for: {article.title}, using fallback rules.")
                return self._fallback_process(article)

        except Exception as e:
            logger.error(f"NLP processing failed for '{article.title}': {e}")
            return self._fallback_process(article)

    def process_batch(self, articles: List[RawArticle]) -> List[SeedEvent]:
        """批量处理（目前复用同步的 process，未来可改写为 Async 提升吞吐量）"""
        results = []
        for a in articles:
            results.append(self.process(a))
        return results

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """从 LLM 回复中提取 JSON 对象"""
        try:
            # 1. 直接尝试解析
            return json.loads(content)
        except json.JSONDecodeError:
            # 2. 尝试从 markdown 代码块中提取
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 3. 尝试暴力匹配花括号
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
                
        return None

    def _build_seed_event(self, article: RawArticle, data: Dict[str, Any]) -> SeedEvent:
        """将 JSON 字典拼装并绑定回原文信息，生成 SeedEvent"""
        entities = []
        for e in data.get("entities", []):
            if isinstance(e, dict) and "name" in e and "entity_type" in e:
                entities.append(ExtractedEntity(
                    name=e.get("name", "Unknown"),
                    entity_type=e.get("entity_type", "other"),
                    confidence=float(e.get("confidence", 0.5))
                ))

        return SeedEvent(
            source=article.source,
            source_url=article.source_url,
            title=article.title,
            summary=data.get("summary", article.title),
            entities=entities,
            sentiment=float(data.get("sentiment", 0.0)),
            sentiment_label=data.get("sentiment_label", "中性"),
            impact_level=data.get("impact_level", "low"),
            affected_sectors=data.get("affected_sectors", []),
            raw_text=article.content,
            created_at=article.published_at or article.fetched_at
        )

    def _fallback_process(self, article: RawArticle) -> SeedEvent:
        """
        降级处理：当 LLM 不可用或解析失败时，采用基于规则的关键词匹配
        （确保数据管道在弱网或超载情况下的鲁棒性）
        """
        text = f"{article.title} {article.content}"
        
        # 简单的情感极性匹配
        bullish_keywords = ["上涨", "增加", "利好", "支持", "鼓励", "下调准备金", "降准", "降息"]
        bearish_keywords = ["下跌", "减少", "利空", "限制", "禁止", "罚款", "重地", "崩盘"]

        bull_count = sum(1 for k in bullish_keywords if k in text)
        bear_count = sum(1 for k in bearish_keywords if k in text)

        if bull_count > bear_count:
            sentiment = 0.5
            sentiment_label = "利好"
        elif bear_count > bull_count:
            sentiment = -0.5
            sentiment_label = "利空"
        else:
            sentiment = 0.0
            sentiment_label = "中性"

        return SeedEvent(
            source=article.source,
            source_url=article.source_url,
            title=article.title,
            summary=article.title,  # 无法做智能摘要，原样使用标题
            sentiment=sentiment,
            sentiment_label=sentiment_label,
            impact_level="medium" if (bull_count + bear_count) > 0 else "low",
            raw_text=article.content,
            created_at=article.published_at or article.fetched_at
        )

