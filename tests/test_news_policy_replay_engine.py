from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtester import BacktestConfig
from core.history_news import DailyNewsDigest, HistoryNewsBundle
from core.news_policy_replay import NewsDrivenPolicyReplayEngine


def _mock_daily_frame(n: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2020-02-03", periods=n, freq="B")
    base = 3500.0 + np.linspace(0.0, 40.0, n)
    wave = np.sin(np.arange(n) / 2.8) * 18.0
    close = base + wave
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 8.0,
            "high": close + 12.0,
            "low": close - 16.0,
            "close": close,
            "volume": 2_000_000 + np.arange(n) * 15_000,
        }
    )


class _DummyNewsService:
    def build_news_bundle(self, **kwargs) -> HistoryNewsBundle:
        frame_dates = pd.bdate_range(kwargs["start_date"], kwargs["end_date"])
        digests = []
        items_by_day = {}
        for idx, day in enumerate(frame_dates):
            key = day.strftime("%Y-%m-%d")
            shock = 0.35 if idx % 2 == 0 else -0.22
            digests.append(
                DailyNewsDigest(
                    date=key,
                    summary=f"{key} 主要新闻：宏观政策与指数风险偏好变化",
                    shock_score=shock,
                    news_count=3,
                    headlines=[f"{key} headline-{i}" for i in range(3)],
                    source_mix={"dummy": 3},
                )
            )
            items_by_day[key] = [
                {
                    "title": f"{key} headline-{i}",
                    "content": "macro and index",
                    "source": "dummy",
                    "published_at": pd.Timestamp(f"{key}T08:00:00Z"),
                }
                for i in range(3)
            ]
        return HistoryNewsBundle(
            source_strategy="mixed",
            scope="macro_index",
            symbol=str(kwargs.get("symbol", "")),
            start_date=str(kwargs["start_date"]),
            end_date=str(kwargs["end_date"]),
            items_by_day=items_by_day,
            daily_digests=digests,
            coverage={
                "window_days": len(frame_dates),
                "days_with_news": len(frame_dates),
                "coverage_rate": 1.0,
                "selected_news_count": len(frame_dates) * 3,
                "online_candidates": len(frame_dates) * 2,
                "local_candidates": len(frame_dates),
                "source_distribution": {"dummy": len(frame_dates) * 3},
            },
            persistence={"enabled": True, "dataset_version": "history_replay_news", "scenario_id": "dummy_s", "snapshot_id": "dummy_p"},
        )


def test_news_policy_replay_engine_outputs_scores_and_digest():
    cfg = BacktestConfig(
        symbol="sh000300",
        benchmark_symbol="sh000300",
        strategy_name="portfolio_system",
        period_days=0,
        lookback=20,
        rebalance_frequency=5,
        policy_text="稳市场政策主线",
        policy_shock=0.3,
        news_source_strategy="mixed",
        news_scope="macro_index",
        news_topk_per_day=8,
        persist_news_events=True,
        auth_score_mode="demo_first",
        random_seed=7,
        feature_flags={"agent_replay": True},
    )
    engine = NewsDrivenPolicyReplayEngine(cfg, news_service=_DummyNewsService())
    frame = _mock_daily_frame(18)
    engine.historical_data = frame
    engine.benchmark_data = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})

    result = engine.run_backtest()

    assert result.total_days == len(frame)
    assert len(result.real_prices) == len(frame)
    assert len(result.simulated_prices) == len(frame)
    assert result.metadata["mode"] == "news_policy_replay"
    assert result.metadata["news_coverage"]["coverage_rate"] == 1.0
    assert len(result.metadata["news_digest"]) == len(frame)
    assert 0.0 <= float(result.metadata["strict_authenticity_score"]) <= 1.0
    assert 0.0 <= float(result.metadata["demo_authenticity_score"]) <= 1.0
    assert result.metadata["score_adjustment_trace"]
    assert result.metadata["auth_score_mode"] == "demo_first"
