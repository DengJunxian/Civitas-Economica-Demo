from pathlib import Path

import pandas as pd

from core.data.market_data_provider import MarketDataProvider, MarketDataQuery


def test_provider_normalizes_columns_and_applies_cn_calendar(tmp_path):
    provider = MarketDataProvider(
        cache_dir=str(tmp_path / "cache"),
        snapshot_dir=str(tmp_path / "snapshots"),
        provider_priority=("akshare",),
    )

    raw = pd.DataFrame(
        {
            "日期": ["2026-01-02", "2026-01-03", "2026-01-05"],
            "开盘": [10.0, 11.0, 12.0],
            "最高": [10.5, 11.5, 12.5],
            "最低": [9.8, 10.8, 11.8],
            "收盘": [10.2, 11.2, 12.2],
            "成交量": [1000, 1100, 1200],
        }
    )

    provider._fetch_from_akshare = lambda _query: raw.copy()  # type: ignore[method-assign]
    provider._get_cn_trading_days = lambda _s, _e: ["2026-01-02", "2026-01-05"]  # type: ignore[method-assign]

    query = MarketDataQuery(symbol="sh000001", interval="1d", period_days=10, adjust="", market="CN")
    frame = provider.get_ohlcv(query, use_cache=False)

    assert not frame.empty
    assert list(frame.columns) == provider.STANDARD_COLUMNS
    assert frame["date"].tolist() == ["2026-01-02", "2026-01-05"]
    assert frame["provider"].nunique() == 1
    assert frame["provider"].iloc[0] == "akshare"


def test_provider_cache_prevents_refetch(tmp_path):
    provider = MarketDataProvider(
        cache_dir=str(tmp_path / "cache"),
        snapshot_dir=str(tmp_path / "snapshots"),
        provider_priority=("akshare",),
    )

    calls = {"count": 0}

    def fake_fetch(_query):
        calls["count"] += 1
        return pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-05"],
                "open": [10.0, 11.0],
                "high": [10.4, 11.4],
                "low": [9.8, 10.8],
                "close": [10.1, 11.1],
                "volume": [1000, 1200],
            }
        )

    provider._fetch_from_akshare = fake_fetch  # type: ignore[method-assign]
    provider._get_cn_trading_days = lambda _s, _e: ["2026-01-02", "2026-01-05"]  # type: ignore[method-assign]

    query = MarketDataQuery(symbol="sh000001", interval="1d", period_days=10, adjust="", market="CN")
    first = provider.get_ohlcv(query, use_cache=True)
    second = provider.get_ohlcv(query, use_cache=True)

    assert calls["count"] == 1
    assert len(first) == len(second) == 2
    cache_files = list(Path(tmp_path / "cache").glob("*.csv"))
    assert cache_files


def test_provider_snapshot_roundtrip(tmp_path):
    provider = MarketDataProvider(
        cache_dir=str(tmp_path / "cache"),
        snapshot_dir=str(tmp_path / "snapshots"),
        provider_priority=("akshare",),
    )
    frame = pd.DataFrame(
        {
            "datetime": ["2026-01-02 15:00:00"],
            "date": ["2026-01-02"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000],
            "amount": [10200.0],
            "symbol": ["sh000001"],
            "interval": ["1d"],
            "provider": ["akshare"],
            "adjust": [""],
            "market": ["CN"],
        }
    )
    query = MarketDataQuery(symbol="sh000001", interval="1d", period_days=1, adjust="", market="CN")

    snapshot_id = provider.create_snapshot(frame=frame, query=query, provider="akshare", snapshot_name="unit")
    loaded = provider.load_snapshot(snapshot_id)

    assert snapshot_id
    assert not loaded.empty
    assert loaded.iloc[0]["symbol"] == "sh000001"
