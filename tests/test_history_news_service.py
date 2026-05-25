from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.event_store import EventStore
from core.history_news import HistoryNewsService


def test_history_news_service_mixed_mode_filters_and_persists(tmp_path):
    seed_path = tmp_path / "seed_events.jsonl"
    rows = [
        {
            "title": "央行发布流动性支持政策，市场预期改善",
            "summary": "央行释放流动性，指数风险偏好回升",
            "source": "seed",
            "source_url": "seed://1",
            "created_at": "2020-02-03T03:00:00Z",
            "sentiment": 0.6,
            "impact_level": "high",
        },
        {
            "title": "地产链条承压，指数短线波动加大",
            "summary": "宏观压力抬升",
            "source": "seed",
            "source_url": "seed://2",
            "created_at": "2020-02-04T03:00:00Z",
            "sentiment": -0.4,
            "impact_level": "medium",
        },
    ]
    seed_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in rows), encoding="utf-8")
    store = EventStore(root_dir=tmp_path / "event_store")
    service = HistoryNewsService(event_store=store, seed_store_path=seed_path)
    service._load_online_rows = lambda start_ts, end_ts: [
        {
            "title": "沪深300震荡走强，宏观政策发力",
            "content": "政策与指数主线修复",
            "source": "online",
            "source_url": "https://example.com/1",
            "published_at": pd.Timestamp("2020-02-03T06:00:00Z"),
            "origin": "online:test",
            "sentiment": 0.2,
            "impact_level": "medium",
            "confidence": 0.7,
        }
    ]

    bundle = service.build_news_bundle(
        start_date="2020-02-03",
        end_date="2020-02-05",
        symbol="sh000300",
        source_strategy="mixed",
        scope="macro_index",
        topk_per_day=8,
        persist=True,
        persist_dataset="history_replay_news",
    )

    assert bundle.source_strategy == "mixed"
    assert bundle.items_by_day
    assert bundle.coverage["days_with_news"] >= 1
    assert bundle.coverage["selected_news_count"] >= 2
    assert bundle.persistence["enabled"] is True
    assert bundle.persistence["dataset_version"] == "history_replay_news"
    assert bundle.persistence["scenario_id"]
    assert bundle.persistence["snapshot_id"]

    frame = store.query_events("history_replay_news", event_types=["news"])
    assert not frame.empty


def test_history_news_service_local_fallback_when_online_empty(tmp_path):
    seed_path = Path(tmp_path) / "seed_events.jsonl"
    seed_path.write_text(
        json.dumps(
            {
                "title": "财政政策发力稳定市场",
                "summary": "指数预期改善",
                "source": "seed",
                "created_at": "2020-03-02T03:00:00Z",
                "sentiment": 0.5,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    service = HistoryNewsService(event_store=EventStore(root_dir=tmp_path / "event_store"), seed_store_path=seed_path)
    service._load_online_rows = lambda start_ts, end_ts: []

    bundle = service.build_news_bundle(
        start_date="2020-03-02",
        end_date="2020-03-04",
        symbol="sh000001",
        source_strategy="mixed",
        scope="macro_index",
        topk_per_day=8,
        persist=False,
    )

    assert bundle.coverage["local_candidates"] >= 1
    assert bundle.coverage["days_with_news"] >= 1
    assert bundle.daily_digests
    digest = bundle.daily_digests[0]
    assert digest.summary
    assert -1.0 <= digest.shock_score <= 1.0


def test_history_news_service_local_cache_backfill_and_reuse(tmp_path):
    seed_path = Path(tmp_path) / "seed_events.jsonl"
    cache_path = Path(tmp_path) / "history_news_cache.jsonl"
    store = EventStore(root_dir=tmp_path / "event_store")
    service = HistoryNewsService(event_store=store, seed_store_path=seed_path, local_cache_path=cache_path)
    service._load_online_rows = lambda start_ts, end_ts: [
        {
            "title": "货币政策边际宽松，风险偏好回升",
            "content": "指数出现修复性反弹",
            "source": "online",
            "source_url": "https://example.com/cache-1",
            "published_at": pd.Timestamp("2020-04-01T06:00:00Z"),
            "origin": "online:test",
            "sentiment": 0.4,
            "impact_level": "medium",
            "confidence": 0.72,
        }
    ]

    first = service.build_news_bundle(
        start_date="2020-04-01",
        end_date="2020-04-03",
        symbol="sh000300",
        source_strategy="mixed",
        scope="macro_index",
        topk_per_day=8,
        persist=False,
    )
    assert first.persistence["local_cache"]["appended_count"] >= 1
    assert cache_path.exists()

    service._load_online_rows = lambda start_ts, end_ts: []
    second = service.build_news_bundle(
        start_date="2020-04-01",
        end_date="2020-04-03",
        symbol="sh000300",
        source_strategy="mixed",
        scope="macro_index",
        topk_per_day=8,
        persist=False,
    )

    assert second.coverage["local_cache_candidates"] >= 1
    assert second.coverage["local_candidates"] >= 1
    assert second.coverage["days_with_news"] >= 1


def test_history_news_service_prioritizes_new_guojiutiao_event(tmp_path):
    seed_path = Path(tmp_path) / "seed_events.jsonl"
    rows = [
        {
            "title": "新国九条发布：加强监管防范风险，推动资本市场高质量发展",
            "summary": "资本市场基础制度迎来重要改革信号",
            "source": "seed",
            "source_url": "seed://priority",
            "created_at": "2024-09-24T03:00:00Z",
            "sentiment": 0.05,
            "impact_level": "high",
        },
        {
            "title": "宏观经济数据例行发布",
            "summary": "市场观望情绪延续",
            "source": "seed",
            "source_url": "seed://normal",
            "created_at": "2024-09-24T04:00:00Z",
            "sentiment": 0.05,
            "impact_level": "medium",
        },
    ]
    seed_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in rows), encoding="utf-8")

    service = HistoryNewsService(event_store=EventStore(root_dir=tmp_path / "event_store"), seed_store_path=seed_path)
    service._load_online_rows = lambda start_ts, end_ts: []
    service._llm_digest = lambda **kwargs: None

    bundle = service.build_news_bundle(
        start_date="2024-09-24",
        end_date="2024-09-24",
        symbol="sh000001",
        source_strategy="local",
        scope="macro_index",
        topk_per_day=2,
        persist=False,
        persist_local_cache=False,
    )

    selected = bundle.items_by_day.get("2024-09-24", [])
    assert selected
    assert "国九条" in str(selected[0].get("title", ""))
    assert bundle.daily_digests
    assert "国九条" in bundle.daily_digests[0].summary
