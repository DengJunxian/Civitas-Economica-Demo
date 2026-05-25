from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtester import BacktestConfig
from core.history_news import DailyNewsDigest, HistoryNewsBundle
from core.news_policy_replay import NewsDrivenPolicyReplayEngine


def _mock_daily_frame(n: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2020-03-02", periods=n, freq="B")
    close = 3000.0 + np.cumsum(np.linspace(-8.0, 12.0, n))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close - 5.0,
            "high": close + 10.0,
            "low": close - 12.0,
            "close": close,
            "volume": 1_000_000 + np.arange(n) * 10_000,
        }
    )


class _NewsService:
    def build_news_bundle(self, **kwargs) -> HistoryNewsBundle:
        dates = pd.bdate_range(kwargs["start_date"], kwargs["end_date"])
        digests = []
        items = {}
        for idx, day in enumerate(dates):
            key = day.strftime("%Y-%m-%d")
            digests.append(
                DailyNewsDigest(
                    date=key,
                    summary=f"{key} 政策与重大新闻冲击",
                    shock_score=0.25 if idx % 2 == 0 else -0.18,
                    news_count=1,
                    headlines=[f"{key} headline"],
                    source_mix={"fixture": 1},
                )
            )
            items[key] = [{"title": f"{key} headline", "content": "policy news", "source": "fixture", "published_at": day}]
        return HistoryNewsBundle(
            source_strategy="fixture",
            scope="macro_index",
            symbol=str(kwargs.get("symbol", "sh000001")),
            start_date=str(kwargs["start_date"]),
            end_date=str(kwargs["end_date"]),
            items_by_day=items,
            daily_digests=digests,
            coverage={"coverage_rate": 1.0, "days_with_news": len(dates), "selected_news_count": len(dates)},
            persistence={"enabled": False, "dataset_version": "fixture"},
        )


def _run_replay(strict: bool):
    cfg = BacktestConfig(
        symbol="sh000001",
        benchmark_symbol="sh000001",
        strategy_name="portfolio_system",
        period_days=0,
        lookback=8,
        rebalance_frequency=3,
        policy_text="历史回放政策",
        policy_shock=0.25,
        news_source_strategy="fixture",
        news_scope="macro_index",
        auth_score_mode="strict" if strict else "demo_first",
        random_seed=11,
        feature_flags={"strict_history_replay": strict},
    )
    engine = NewsDrivenPolicyReplayEngine(cfg, news_service=_NewsService())
    frame = _mock_daily_frame()
    engine.historical_data = frame
    engine.benchmark_data = frame[["date", "close"]].rename(columns={"close": "benchmark_close"})
    return engine.run_backtest()


def test_history_replay_strict_mode_has_no_hidden_beautify():
    result = _run_replay(strict=True)

    assert result.metadata["history_replay_mode"] == "strict"
    assert result.metadata["strict_mode"] is True
    assert result.metadata["raw_vs_display"]["demo_calibration_alpha"] == 0.0
    assert result.metadata["raw_vs_display"]["demo_adjustment_total_abs"] == 0.0
    assert result.metadata["raw_simulated_prices"] == result.metadata["display_simulated_prices"]


def test_history_replay_demo_mode_preserves_raw_metrics():
    result = _run_replay(strict=False)

    assert result.metadata["history_replay_mode"] == "demo"
    assert result.metadata["demo_mode"] is True
    assert result.metadata["raw_simulated_prices"]
    assert result.metadata["display_simulated_prices"]
    assert "raw_metric_source" in result.metadata["raw_vs_display"]
    assert all("raw_close" in bar and "display_close" in bar for bar in result.simulated_bars)


def test_news_driven_replay_keeps_history_pipeline():
    result = _run_replay(strict=False)

    assert result.metadata["mode"] == "news_policy_replay"
    assert result.metadata["news_digest"]
    assert len(result.real_prices) == len(result.simulated_prices)
