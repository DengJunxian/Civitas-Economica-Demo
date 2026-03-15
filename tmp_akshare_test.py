"""
Smoke test for the unified market data provider.
"""

from core.data.market_data_provider import MarketDataProvider, MarketDataQuery


def main() -> None:
    provider = MarketDataProvider()

    daily_query = MarketDataQuery(symbol="sh000001", interval="1d", period_days=30, adjust="", market="CN")
    daily = provider.get_ohlcv(daily_query, use_cache=True)
    print(f"[daily] rows={len(daily)} columns={list(daily.columns)}")
    if not daily.empty:
        print(daily.tail(3)[["date", "open", "high", "low", "close", "provider"]])
        snapshot_id = provider.create_snapshot(daily, daily_query, provider=str(daily.iloc[0]["provider"]), snapshot_name="sh000001_daily")
        print(f"[daily] snapshot={snapshot_id}")

    minute_query = MarketDataQuery(symbol="000001", interval="1m", period_days=1, adjust="qfq", market="CN")
    minute = provider.get_ohlcv(minute_query, use_cache=True)
    print(f"[minute] rows={len(minute)}")
    if not minute.empty:
        print(minute.tail(3)[["datetime", "open", "high", "low", "close", "provider"]])


if __name__ == "__main__":
    main()
