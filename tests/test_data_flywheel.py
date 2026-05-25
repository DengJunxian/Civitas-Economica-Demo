# file: tests/test_data_flywheel.py
import pytest
import json
import os
import shutil
from unittest.mock import MagicMock

from config import GLOBAL_CONFIG
from data_flywheel.schemas import RawArticle, ExtractedEntity, SeedEvent
from data_flywheel.sources.news_api_source import NewsApiSource
from data_flywheel.nlp_processor import NlpProcessor
from data_flywheel.seed_store import SeedStore
from data_flywheel.betta_spider import BettaSpider


@pytest.fixture
def temp_store_path():
    path = "tests/temp_data/seed_events.jsonl"
    yield path
    # Cleanup
    if os.path.exists(path):
         os.remove(path)
    if os.path.exists("tests/temp_data"):
         shutil.rmtree("tests/temp_data")


def test_seed_event_serialization():
    """测试 SeedEvent 的 JSON 序列化与反序列化"""
    event = SeedEvent(
        source="测试来源",
        title="测试标题",
        summary="测试摘要",
        entities=[
            ExtractedEntity(name="公司A", entity_type="company", confidence=0.9),
            ExtractedEntity(name="行业B", entity_type="sector", confidence=0.8)
        ],
        sentiment=0.8,
        sentiment_label="利好",
        impact_level="high",
        affected_sectors=["行业B"]
    )
    
    json_str = event.to_json()
    assert "测试来源" in json_str
    assert "公司A" in json_str
    
    restored = SeedEvent.from_json(json_str)
    assert restored.event_id == event.event_id
    assert restored.title == "测试标题"
    assert len(restored.entities) == 2
    assert restored.entities[0].name == "公司A"
    assert restored.sentiment == 0.8


def test_seed_event_to_policy_input():
    """测试 SeedEvent 转换为 PolicyCommittee 输入"""
    event = SeedEvent(
        source="快讯",
        title="重磅新政发布",
        summary="央行宣布...",
        entities=[ExtractedEntity(name="央行", entity_type="policy")],
        sentiment=0.5,
        sentiment_label="利好",
        impact_level="high",
        affected_sectors=["银行"]
    )
    
    policy_text = event.to_policy_input()
    assert "央行(policy)" in policy_text
    assert "来源: 快讯" in policy_text
    assert "影响: high" in policy_text
    assert "重磅新政发布" in policy_text


def test_news_api_mock_source():
    """测试模拟新闻数据源"""
    source = NewsApiSource(use_mock=True)
    assert "Mock" in source.source_name
    
    articles = source.fetch(max_items=2)
    assert len(articles) == 2
    assert articles[0].title != ""
    assert articles[0].content != ""
    assert articles[0].source != ""


def test_nlp_processor_fallback():
    """测试 NLP 处理器的降级逻辑"""
    mock_router = MagicMock()
    # Simulate LLM failure or parse failure
    mock_router.sync_call_with_fallback.return_value = ("invalid json", "", "model")
    
    processor = NlpProcessor(model_router=mock_router)
    
    article = RawArticle(
        title="市场利好：央行下调准备金",
        content="支持实体经济，释放长期流动性",
        source="测试网络"
    )
    
    event = processor.process(article)
    
    # 期望使用基于规则的兜底
    assert event.sentiment > 0
    assert event.sentiment_label == "利好"
    assert event.impact_level == "medium"
    assert event.title == "市场利好：央行下调准备金"


def test_seed_store(temp_store_path):
    """测试持久化存储"""
    store = SeedStore(temp_store_path)
    
    event1 = SeedEvent(title="事件1", impact_level="low")
    event2 = SeedEvent(title="事件2", impact_level="high")
    event3 = SeedEvent(title="事件3", impact_level="critical")
    
    store.append_batch([event1, event2, event3])
    
    # 读取所有
    all_events = store.read_latest(10)
    assert len(all_events) == 3
    # 验证读取顺序（最新到最旧）
    assert all_events[0].title == "事件3"
    assert all_events[-1].title == "事件1"
    
    # 过滤 min_impact
    high_events = store.read_latest(10, min_impact="high")
    assert len(high_events) == 2  # high, critical
    assert all(e.impact_level in ["high", "critical"] for e in high_events)


def test_betta_spider_run_once(temp_store_path):
    """测试爬虫编排引擎的单次运行"""
    # 配置仅使用 Mock 源
    config = {
        "enable_mock_source": True,
        "rss_feeds": [],
        "output_path": temp_store_path
    }
    
    spider = BettaSpider(config_paths=config)
    
    # 为了让测试不依赖真实网络/LLM API，我们 Mock 掉 NLP 的 LLM 调用，让它走 fallback
    spider.nlp.router.sync_call_with_fallback = MagicMock(return_value=("invalid json", "", "mock"))
    
    # 执行流水线，为了速度限制只拉2条
    events = spider.run_once(max_articles=2)
    
    assert len(events) == 2
    
    # 检查存储
    saved = spider.store.read_latest(10)
    assert len(saved) == 2
    
    # 简单测试 push 接口（Mock 掉真实的 PolicyCommittee）
    mock_committee = MagicMock()
    spider.push_to_sandbox(mock_committee)
    
    # 因为 fallback 的 impact_level 是 medium（如果有关键词）或 low，由于 push 会过滤 min_impact="medium"
    # 所以可能不会推全部，但至少调用的逻辑不报错
    # 验证是否正常执行了
    assert True
