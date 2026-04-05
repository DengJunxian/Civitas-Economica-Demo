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
