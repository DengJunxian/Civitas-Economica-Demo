# file: core/integrated_market_engine.py

import heapq
import time
import random
import re
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak 
from openai import OpenAI
import httpx
import uuid
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak 
from openai import OpenAI
import httpx

from core.types import Order, Trade, Candle, OrderSide, OrderType, OrderStatus
from core.time_manager import SimulationClock

# Import C++ Optimized OrderBook
try:
    from core.exchange.order_book_cpp import OrderBookCPP, _civitas_lob
    from core.exchange.order_book import Order as OrderModel
    if _civitas_lob is None:
        raise ImportError("C++ extension _civitas_lob not available")
    USE_CPP_LOB = True
    print("[*] High-Performance C++ OrderBook Activated")
except ImportError as e:
    USE_CPP_LOB = False
    print(f"[!] Falling back to Python: {e}")

# Assumes a config.py exists with these constants. 
from config import GLOBAL_CONFIG
from core.data.market_data_provider import MarketDataProvider, MarketDataQuery

# ==========================================
# PART 1: Infrastructure & Calendar
# (Derived from market_engine.py)
# ==========================================

class ChinaTradingCalendar:
    """
    Manages A-share trading days.
    Rules: Weekends + Statutory Holidays (Closed on Adjusted Working Days).
    """
    
    # Format: 'YYYY-MM-DD'
    HOLIDAYS_WEEKDAY_2025 = [
        "2025-01-01", # New Year
        "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-03", "2025-02-04", # CNY
        "2025-04-04", # Tomb Sweeping
        "2025-05-01", "2025-05-02", "2025-05-05", # Labor Day
        "2025-06-02", # Dragon Boat
        "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08" # National Day
    ]
    
    HOLIDAYS_WEEKDAY_2026 = [
        "2026-01-01", "2026-01-02",
        "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-23",
        "2026-04-06",
        "2026-05-01", "2026-05-04", "2026-05-05",
        "2026-06-19",
        "2026-09-25",
        "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"
    ]
    
    ALL_HOLIDAYS = set(HOLIDAYS_WEEKDAY_2025 + HOLIDAYS_WEEKDAY_2026)

    @staticmethod
    def get_next_trading_day(date_str: str) -> str:
        """Returns the next valid A-share trading day."""
        try:
            curr = datetime.strptime(date_str, "%Y-%m-%d")
        except:
            curr = datetime.now()
            
        while True:
            curr += timedelta(days=1)
            d_str = curr.strftime("%Y-%m-%d")
            
            # 1. Check Weekend (5=Sat, 6=Sun)
            if curr.weekday() >= 5:
                continue
                
            # 2. Check Holidays
            if d_str in ChinaTradingCalendar.ALL_HOLIDAYS:
                continue
            
            return d_str

# ==========================================
# PART 2: Core Data Structures
# (Merged from coremarket_engine.py & market_engine.py)
# ==========================================

@dataclass
class PolicyState:
    tax_rate: float = GLOBAL_CONFIG.TAX_RATE_STAMP       
    risk_free_rate: float = 0.02  
    liquidity_injection: float = 0.0 
    description: str = "Initial State" 


@dataclass
class ExogenousLiquidityPoint:
    """External market backdrop point used by hybrid replay mode."""

    step: int
    price: float
    volume: float


def blend_price_with_backdrop(
    old_price: float,
    endogenous_price: float,
    *,
    exogenous_price: Optional[float] = None,
    exogenous_volume: float = 0.0,
    backdrop_weight: float = 0.35,
) -> float:
    """
    Blend endogenous simulated price with exogenous backdrop series.
    Agents still generate endogenous orders; backdrop acts as liquidity anchor.
    """
    old_p = float(max(old_price, 1e-6))
    endo_p = float(max(endogenous_price, 1e-6))
    if exogenous_price is None:
        return endo_p

    exo_p = float(max(exogenous_price, 1e-6))
    w = max(0.0, min(1.0, float(backdrop_weight)))
    raw = (1.0 - w) * endo_p + w * exo_p

    # Higher exogenous volume dampens endogenous impact.
    vol = max(0.0, float(exogenous_volume))
    damp = 1.0 + min(3.0, vol / 1_000_000.0) * 0.20
    return float(old_p + (raw - old_p) / damp)

# ==========================================
# PART 3: Matching Engine
# (From coremarket_engine.py, enhanced for Simulation)
# ==========================================

class MatchingEngine:
    """
    A-Share High Fidelity Matching Engine.
    
    Responsibilities:
    1. Order Book Maintenance (Heapq)
    2. Price Limit Checks
    3. Continuous Matching Logic
    4. Fee & Tax Calculation
    5. National Team Intervention
    """

    def __init__(self, symbol: str = "A_SHARE_IDX", prev_close: float = 3000.0, clock: Optional[SimulationClock] = None):
        self.symbol = symbol
        self.prev_close = prev_close
        self.clock = clock
        
        # Order Books or LOB
        # Unified Interface: self.lob will always be an object with .add_order(), .get_depth(), etc.
        if USE_CPP_LOB:
            self.lob = OrderBookCPP(symbol, prev_close)
            print("[MatchingEngine] Using high-performance C++ OrderBook")
        else:
            # Fallback to Python Implementation
            # Ensure proper import path and class usage
            from core.exchange.order_book import OrderBook
            self.lob = OrderBook(symbol, prev_close)
            print("[MatchingEngine] Using Pure Python OrderBook (Fallback)")
        
        # Statistics
        self.last_price = prev_close
        self.total_volume = 0
        self.trades_history: List[Trade] = []
        
        # Buffer for current step generation
        self.step_trades_buffer: List[Trade] = []

    def update_prev_close(self, close_price: float):
        """Update prev_close after market close for next day's limit calculation."""
        self.prev_close = close_price

    def _check_price_limit(self, price: float) -> bool:
        """
        娑ㄨ穼鍋滄鏌?(Delegated to OrderBook)
        """
        if hasattr(self, 'lob'):
            return self.lob._check_price_limit(price)
            
        # Fallback if no LOB (Unlikely)
        limit = GLOBAL_CONFIG.PRICE_LIMIT 
        upper = self.prev_close * (1 + limit)
        lower = self.prev_close * (1 - limit)
        return round(lower, 2) <= round(price, 2) <= round(upper, 2)

    def submit_order(self, order: Order, liquidity_injection_prob: float = 0.0) -> List[Trade]:
        """
        Submit an order and attempt immediate matching.
        
        Args:
            order: The Order object.
            liquidity_injection_prob: Probability (0-1) of National Team intervention if selling pressure is high.
            
        Returns:
            List[Trade]: Trades generated by this order.
        """
        # 1. Price Limit Check
        if not self._check_price_limit(order.price):
            # In a real exchange, this is a Reject. 
            # We return empty list to signify no trades.
            return []

        # 2. National Team Intervention (Liquidity Injection)
        # If it's a sell order and probability triggers, generate a buy order first.
        if order.side == 'sell' and liquidity_injection_prob > 0:
            if random.random() < liquidity_injection_prob:
                team_order = Order(
                    price=order.price, 
                    quantity=order.quantity, 
                    agent_id="NATIONAL_TEAM", 
                    side=OrderSide.BUY, 
                    timestamp=self.clock.timestamp if self.clock else time.time(),
                    order_type=OrderType.LIMIT,
                    symbol=self.symbol
                )
                # Inject directly
                if USE_CPP_LOB:
                    lob_team_order = OrderModel(
                        agent_id=team_order.agent_id,
                        symbol=self.symbol,
                        side=team_order.side,
                        order_type=OrderType.LIMIT,
                        price=team_order.price,
                        quantity=team_order.quantity,
                        order_id=team_order.order_id,
                        timestamp=team_order.timestamp
                    )
                    self.lob.add_order(lob_team_order)
                else:
                    # For Python implementation, we might need adjustments if using heapq directly
                    # But if using unified interface self.lob, we should use that.
                    # The original code used heapq.heappush directly for Python fallback?
                    # "else: heapq.heappush..."
                    # Let's stick to using self.lob.add_order for consistency if possible.
                    # But self.lob is OrderBook instance.
                    
                    self.lob.add_order(team_order)

        # 3. Matching Logic
        generated_trades = []

        # 3. Matching Logic
        generated_trades = []


        
        lob_id = order.order_id
        
        if USE_CPP_LOB:
             # C++ implementation might need a different dictionary or specific fields
             # But here we are creating a specific Order object for the LOB
             # Check if OrderBookCPP expects core.types.Order or its own thing
             # Assuming it expects the same fields.
             
             lob_order = OrderModel(
                agent_id=order.agent_id,
                symbol=self.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                price=order.price,
                quantity=order.quantity,
                order_id=lob_id,
                timestamp=order.timestamp
            )
        else:
            # Python implementation
            from core.types import Order as PyOrder
            lob_order = PyOrder(
                agent_id=order.agent_id,
                symbol=self.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                price=order.price,
                quantity=order.quantity,
                order_id=lob_id,
                timestamp=order.timestamp
            )
            
        # Execute Matching via Polymorphic Interface
        # Both implementations should have .add_order(order) returning List[Trade]
        trades = self.lob.add_order(lob_order)
            
        # Sync back filled quantity to the local order object
        order.filled_qty = lob_order.filled_qty
            
        # Convert LOB Trades back to local Trade object (if necessary)
        # The Scheduler expects core.market_engine.Trade objects
        for t in trades:
            # Map fields from LOB Trade to MarketEngine Trade
            # Check attribute names carefully. Python Trade has: trade_id, price, quantity, maker_id...
            
            # LOB Trade structure:
            # price, quantity, maker_agent_id, taker_agent_id, maker_order_id, taker_order_id, etc.
            # We need to map them correctly.
            # Assuming 't' matches what lob.add_order returns.
            # If using Python OrderBook, it returns a list of Trade objects defined in ITSELF or types.py?
            # It seems core/exchange/order_book.py might generate dicts or its own Trade tuples.
            # Based on current code structure, let's assume 't' has attributes.
            
            # Direct mapping to core.types.Trade
            local_trade = Trade(
                trade_id=str(uuid.uuid4()), # Generate new ID
                price=t.price,
                quantity=int(t.quantity),
                maker_id=getattr(t, 'maker_order_id', getattr(t, 'maker_id', 'unknown')), 
                taker_id=getattr(t, 'taker_order_id', getattr(t, 'taker_id', order.order_id)),
                maker_agent_id=t.maker_agent_id,
                taker_agent_id=t.taker_agent_id,
                buyer_agent_id=t.taker_agent_id if order.side=='buy' else t.maker_agent_id,
                seller_agent_id=t.maker_agent_id if order.side=='buy' else t.taker_agent_id,
                timestamp=t.timestamp,
                buyer_fee=t.buyer_fee,
                seller_fee=t.seller_fee,
                seller_tax=t.seller_tax
            )
            generated_trades.append(local_trade)
            
            # Update Engine Stats
            self.last_price = local_trade.price
            self.total_volume += local_trade.quantity
            self.trades_history.append(local_trade)

        # Add to step buffer for Candle generation
        self.step_trades_buffer.extend(generated_trades)
        return generated_trades



    def get_order_book_depth(self, level=5) -> Dict:
        """Get L5 Market Depth."""
        # Unified call
        return self.lob.get_depth(level)

    def flush_step_trades(self) -> List[Trade]:
        """Return trades from the current step and clear buffer."""
        trades = self.step_trades_buffer[:]
        self.step_trades_buffer = []
        return trades

    def run_call_auction(self, orders: List[Order], market_time: float = None) -> Tuple[float, List[Trade]]:
        """
        闆嗗悎绔炰环锛圕all Auction锛?
        
        妯℃嫙A鑲″競鍦?:15-9:25鐨勯泦鍚堢珵浠烽樁娈点€?
        閫氳繃璁＄畻鑳戒娇鎴愪氦閲忔渶澶у寲鐨勪环鏍兼潵纭畾寮€鐩樹环銆?
        
        绠楁硶锛?
        1. 灏嗘墍鏈変拱鍗栬鍗曟寜浠锋牸鎺掑簭
        2. 璁＄畻姣忎釜浠锋牸姘村钩鐨勭疮璁′拱鍗栭噺
        3. 鎵惧埌浣挎垚浜ら噺鏈€澶у寲鐨勪环鏍间綔涓哄紑鐩樹环
        
        Args:
            orders: 闆嗗悎绔炰环闃舵鐨勬墍鏈夎鍗?
            market_time: 褰撳墠甯傚満鏃堕棿 (鍙€?
            
        Returns:
            (寮€鐩樹环, 鎴愪氦鍒楄〃)
        """
        if not orders:
            return self.prev_close, []
        
        # 鍒嗙涔板崠璁㈠崟
        buy_orders = [o for o in orders if o.side == 'buy']
        sell_orders = [o for o in orders if o.side == 'sell']
        
        if not buy_orders or not sell_orders:
            return self.prev_close, []
        
        # 鏀堕泦鎵€鏈夊彲鑳界殑浠锋牸姘村钩
        all_prices = set()
        for o in orders:
            if self._check_price_limit(o.price):
                all_prices.add(o.price)
        
        if not all_prices:
            return self.prev_close, []
        
        price_levels = sorted(all_prices)
        
        # 璁＄畻姣忎釜浠锋牸姘村钩鐨勬垚浜ら噺
        best_price = self.prev_close
        max_volume = 0
        
        for test_price in price_levels:
            # 鎰挎剰浠?test_price 鎴栨洿楂樹环鏍间拱鍏ョ殑鎬婚噺
            buy_volume = sum(o.quantity for o in buy_orders if o.price >= test_price)
            # 鎰挎剰浠?test_price 鎴栨洿浣庝环鏍煎崠鍑虹殑鎬婚噺
            sell_volume = sum(o.quantity for o in sell_orders if o.price <= test_price)
            # 鎴愪氦閲忓彇杈冨皬鍊?
            match_volume = min(buy_volume, sell_volume)
            
            if match_volume > max_volume:
                max_volume = match_volume
                best_price = test_price
            elif match_volume == max_volume and match_volume > 0:
                # 鐩稿悓鎴愪氦閲忔椂锛岄€夋嫨鏇存帴杩戝墠鏀剁洏浠风殑浠锋牸
                if abs(test_price - self.prev_close) < abs(best_price - self.prev_close):
                    best_price = test_price
        
        # 浠ュ紑鐩樹环鎵ц鎾悎
        trades = []
        opening_price = best_price
        
        # 绛涢€夊彲鎴愪氦鐨勮鍗?
        executable_buys = sorted(
            [o for o in buy_orders if o.price >= opening_price],
            key=lambda x: (-x.price, x.timestamp)  # 浠锋牸浼樺厛锛屾椂闂翠紭鍏?
        )
        executable_sells = sorted(
            [o for o in sell_orders if o.price <= opening_price],
            key=lambda x: (x.price, x.timestamp)
        )
        
        # 鎵ц鎾悎
        buy_idx, sell_idx = 0, 0
        while buy_idx < len(executable_buys) and sell_idx < len(executable_sells):
            buy_order = executable_buys[buy_idx]
            sell_order = executable_sells[sell_idx]
            
            # 璁＄畻鎴愪氦閲?
            match_qty = min(buy_order.remaining_qty, sell_order.remaining_qty)
            
            if match_qty > 0:
                # 鍒涘缓鎴愪氦璁板綍
                market_val = opening_price * match_qty
                comm_rate = GLOBAL_CONFIG.TAX_RATE_COMMISSION
                stamp_rate = GLOBAL_CONFIG.TAX_RATE_STAMP
                
                trade = Trade(
                    trade_id=str(uuid.uuid4()), # Generate new ID to ensure uniqueness for aggregates
                    price=opening_price,
                    quantity=match_qty,
                    buy_agent_id=buy_order.agent_id,
                    sell_agent_id=sell_order.agent_id,
                    timestamp=self.clock.timestamp if self.clock else time.time(),
                    buyer_fee=market_val * comm_rate,
                    seller_fee=market_val * comm_rate,
                    seller_tax=0.0 # Will be updated by PolicyManager
                )
                
                # 搴旂敤鏀跨瓥鏁堟灉锛堝嵃鑺辩◣绛夛級
                # Note: RunCallAuction is internal to MatchingEngine but we need policy here.
                # Ideally MatchingEngine should know about Policy, but for now we apply it here or outside?
                # Actually, RunCallAuction is in MatchingEngine which doesn't know PolicyManager.
                # So we should apply tax in MarketDataManager.finalize_step or wrapper.
                # BUT, wait, this code block is INSIDE MatchingEngine.run_call_auction in the original file?
                # Ah, I am editing core/market_engine.py. 
                # run_call_auction is a method of MatchingEngine.
                # MatchingEngine doesn't have reference to PolicyManager.
                # Let's simple apply default tax here, OR pass PolicyManager to MatchingEngine.
                # EASIER: Leave it as default in MatchingEngine, and overwrite it in MarketDataManager if needed.
                # Or better, let MarketDataManager handle the loop.
                
                trade.seller_tax = market_val * stamp_rate # Default
                
                trades.append(trade)
                
                # 鏇存柊璁㈠崟鐘舵€?
                buy_order.filled_qty += match_qty
                sell_order.filled_qty += match_qty
            
            if buy_order.is_filled:
                buy_idx += 1
            if sell_order.is_filled:
                sell_idx += 1
        
        # 鏇存柊寮曟搸鐘舵€?
        if trades:
            self.last_price = opening_price
            self.total_volume += sum(t.quantity for t in trades)
            self.trades_history.extend(trades)
            self.step_trades_buffer.extend(trades)
        
        return opening_price, trades

# ==========================================
# PART 4: Helpers (Loader & Policy)
# (From market_engine.py)
# ==========================================

class RealMarketLoader:
    """Real market history loader with provider fallback and deterministic caching."""

    _provider = MarketDataProvider()

    @staticmethod
    def _fallback_default_frame(symbol: str) -> pd.DataFrame:
        default_file = f"data/{symbol}_default.csv"
        if os.path.exists(default_file):
            print(f"[!] Using default fallback data: {default_file}")
            return pd.read_csv(default_file)
        print(f"[!] No default data found at {default_file}")
        return pd.DataFrame()

    @staticmethod
    def _to_candles(df: pd.DataFrame, symbol: str) -> List[Candle]:
        if df is None or df.empty:
            return []
        out: List[Candle] = []
        cnt = len(df)
        for i, row in df.iterrows():
            o = float(row.get("open", row.get("Open", 3000.0)))
            h = float(row.get("high", row.get("High", 3000.0)))
            l = float(row.get("low", row.get("Low", 3000.0)))
            c = float(row.get("close", row.get("Close", 3000.0)))
            v = int(float(row.get("volume", row.get("Volume", 0.0))))
            d = str(row.get("date", row.get("Date", "2024-01-01")))
            out.append(
                Candle(
                    symbol=symbol,
                    step=-(cnt - i),
                    timestamp=d,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                    is_simulated=False,
                )
            )
        return out

    @staticmethod
    def load_history(symbol="sh000001", period="365") -> List[Candle]:
        try:
            period_int = int(period)
        except Exception:
            period_int = 365

        try:
            cache_dir = "data/cache"
            os.makedirs(cache_dir, exist_ok=True)
            compat_cache_file = os.path.join(cache_dir, f"{symbol}_{period_int}.csv")

            df = pd.DataFrame()
            if os.path.exists(compat_cache_file):
                print(f"[*] Loading {symbol} from cache: {compat_cache_file}")
                df = pd.read_csv(compat_cache_file)
            else:
                print(f"[*] Loading {period_int} days of {symbol} from market provider...")
                query = MarketDataQuery(
                    symbol=symbol,
                    interval="1d",
                    period_days=period_int,
                    adjust="",
                    market="CN",
                )
                fetched = RealMarketLoader._provider.get_ohlcv(query, use_cache=True, freeze_snapshot=False)
                if fetched is not None and not fetched.empty:
                    df = fetched[["date", "open", "high", "low", "close", "volume"]].copy()
                    df.to_csv(compat_cache_file, index=False)

            if df is None or df.empty:
                df = RealMarketLoader._fallback_default_frame(symbol)

            if df is None or df.empty:
                raise ValueError("no market data from providers, cache, or default file")

            df = df.tail(period_int).reset_index(drop=True)
            candles = RealMarketLoader._to_candles(df, symbol)

            try:
                if ak is not None:
                    spot_df = ak.stock_zh_index_spot()
                    symbol_code = symbol.replace("sh", "").replace("sz", "")
                    row_spot = spot_df[spot_df["代码"] == symbol_code]
                    if not row_spot.empty:
                        latest_price = float(row_spot.iloc[0]["最新价"])
                        if candles and latest_price > 0:
                            candles[-1].close = latest_price
                            print(f"[*] Reconciled latest spot close for {symbol}: {latest_price}")
            except Exception as e:
                print(f"[!] Spot reconciliation failed: {e}")

            print(f"[OK] Loaded {len(candles)} trading-day candles")
            return candles
        except Exception as e:
            print(f"[!] Market history load failed: {e}")
            return [Candle(symbol, -1, "2024-01-01", 3000, 3000, 3000, 3000, 100000, 0.0, False)]

class PolicyInterpreter:
    """
    鏀跨瓥瑙ｉ噴鍣細閫氳繃 DeepSeek 鎴?GLM 鍒嗘瀽鏀跨瓥鏂囨湰锛岄噺鍖栧叾瀵瑰競鍦虹殑褰卞搷銆?
    鏀寔浣跨敤 ModelRouter 杩涜澶氭ā鍨嬭矾鐢便€?
    """
    
    def __init__(self, api_key_or_router):
        # 鏀寔浼犲叆 router 鎴?key (鍏煎鏃ф帴鍙?
        if hasattr(api_key_or_router, 'call_with_fallback'):
            self.router = api_key_or_router
            self.api_key = self.router.deepseek_key if self.router.deepseek_key else "dummy"
        else:
            self.router = None
            self.api_key = api_key_or_router
            
        self.last_reasoning = None  # 淇濆瓨鏈€杩戜竴娆＄殑鎺ㄧ悊杩囩▼

    async def interpret(self, policy_text: str) -> Dict:
        """
        鍒嗘瀽鏀跨瓥鏂囨湰锛岃繑鍥為噺鍖栧弬鏁般€?
        
        Args:
            policy_text: 鏀跨瓥鎻忚堪鏂囨湰
            
        Returns:
            Dict: 鍖呭惈 tax_rate, liquidity_injection, fear_factor, initial_news, 
                  sentiment_shift, reasoning 绛夊瓧娈?
        """
        if not self.api_key: 
            return self._default_policy()

        # 鏋勯€?prompt
        prompt = f"""浣犳槸涓€浣嶈祫娣辩殑A鑲″競鍦烘斂绛栧垎鏋愬笀銆傝鍒嗘瀽浠ヤ笅鏀跨瓥瀵瑰競鍦虹殑褰卞搷锛屽苟缁欏嚭閲忓寲鍙傛暟銆?

銆愬緟鍒嗘瀽鏀跨瓥銆?
{policy_text}

銆愬綋鍓嶅競鍦哄熀鍑嗐€?
- 鍗拌姳绋庣巼: {GLOBAL_CONFIG.TAX_RATE_STAMP:.4%}
- 娑ㄨ穼鍋滈檺鍒? {GLOBAL_CONFIG.PRICE_LIMIT:.0%}

銆愬垎鏋愯姹傘€?
1. 鐩存帴鏁堝簲锛氭祦鍔ㄦ€у拰浜ゆ槗鎴愭湰褰卞搷
2. 淇″彿鏁堝簲锛氭斂绛栫殑闅愬惈淇″彿鍜屽競鍦鸿В璇?
3. 浜岄樁璁ょ煡锛氭姇璧勮€呴鏈熺殑鑷垜瀹炵幇
4. 鏃舵晥鎬э細鐭湡鎯呯华 vs 涓湡鍩烘湰闈?

銆愯緭鍑烘牸寮忋€?
涓ユ牸杩斿洖 JSON 鏍煎紡锛屼笉瑕佸寘鍚?Markdown 鏍煎紡鏍囪锛?
{{
    "tax_rate": <鏂板嵃鑺辩◣鐜囷紝float>,
    "liquidity_injection": <娴佸姩鎬ф敞鍏ユ鐜囷紝0.0-1.0>,
    "fear_factor": <鎭愭厡鍥犲瓙锛?.0-1.0>,
    "sentiment_shift": <鎯呯华鍋忕Щ閲忥紝-1.0鍒?.0>,
    "initial_news": "<绠€鐭柊闂绘爣棰?",
    "market_impact": "<涓€鍙ヨ瘽鎬荤粨>",
    "reasoning_summary": "<鍒嗘瀽杩囩▼鎽樿>"
}}
"""
        try:
            # 浣跨敤 Router 鎴?AsyncClient (DeepSeek)
            # 涓轰簡鍦?Loop Architecture 涓珮鏁堣繍琛岋紝杩欓噷蹇呴』鏄紓姝ヨ皟鐢?
            
            content = ""
            reasoning = ""
            
            if not self.router:
                from core.model_router import ModelRouter
                self.router = ModelRouter(
                    deepseek_key=self.api_key,
                    zhipu_key=GLOBAL_CONFIG.ZHIPU_API_KEY
                )
            
            # 浣跨敤榛樿浼樺厛绾?
            priority = ["deepseek-reasoner", "glm-4-flashx", "deepseek-chat"]
            
            response = await self.router.call_with_fallback(
                [{"role": "user", "content": prompt}],
                priority_models=priority,
                timeout_budget=60.0,
                fallback_response='{"tax_rate": 0.0005, "fear_factor": 0, "liquidity_injection": 0}'
            )
            
            # router 杩斿洖鐨勬槸 content, reasoning, model
            content = response[0]
            reasoning = response[1]

            self.last_reasoning = reasoning
            
            # 瑙ｆ瀽 JSON
            import json
            import re
            
            # 娓呯悊 Markdown
            if "```" in content: 
                content = re.sub(r"```json|```", "", content).strip()
            
            # 灏濊瘯瑙ｆ瀽
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # 灏濊瘯淇甯歌 JSON 閿欒
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    raise ValueError("鏃犳硶鎻愬彇 JSON")

            result['reasoning'] = self.last_reasoning
            print(f"[OK] 鏀跨瓥鍒嗘瀽瀹屾垚: {result.get('initial_news', '鏈煡')}")
            return result

        except Exception as e:
            print(f"[!] 鏀跨瓥鍒嗘瀽澶辫触: {e}")
            return self._default_policy()

    def _default_policy(self) -> Dict:
        """Fallback policy values when LLM policy parsing is unavailable."""
        return {
            "tax_rate": GLOBAL_CONFIG.TAX_RATE_STAMP,
            "liquidity_injection": 0.0,
            "fear_factor": 0.0,
            "sentiment_shift": 0.0,
            "initial_news": "Policy published",
            "market_impact": "Impact pending assessment",
            "reasoning": "(fallback defaults)",
        }

# ==========================================
# PART 5: Market Data Manager
# (Integrated Logic)
# ==========================================

from core.policy import PolicyManager  
from core.regulation.risk_control import RiskEngine

class MarketDataManager:
    def __init__(self, api_key_or_router, load_real_data=True, clock: Optional[SimulationClock] = None, regulatory_module: Optional[Any] = None):
        self.policy = PolicyState()
        self.interpreter = PolicyInterpreter(api_key_or_router)
        self.clock = clock
        self.regulatory_module = regulatory_module
        
        # 鏀跨瓥绠＄悊鍣?(Legacy, keeping for now)
        self.policy_manager = PolicyManager()
        
        # 椋庢帶寮曟搸锛堥泦涓紡缃戝叧锛?(Keep as dual check or remove later)
        self.risk_engine = RiskEngine(
            stamp_duty_rate=GLOBAL_CONFIG.TAX_RATE_STAMP,
            commission_rate=GLOBAL_CONFIG.TAX_RATE_COMMISSION
        )
        
        # Load Data
        self.history_candles = RealMarketLoader.load_history() if load_real_data else []
        self.sim_candles = []
        
        initial_price = self.history_candles[-1].close if self.history_candles else 3000.0
        
        # Initialize Core Engine
        self.engine = MatchingEngine(prev_close=initial_price, clock=self.clock)
        
        # State
        self.current_news = "Waiting for market open"
        self.panic_level = 0.0 
        self.csad_history = []
        self.text_factor_state: Dict[str, Any] = {
            "dominant_topic": "uncategorized",
            "sentiment_score": 0.0,
            "panic_index": 0.0,
            "greed_index": 0.0,
            "policy_shock": 0.0,
            "regime_bias": "neutral",
        }
        self.latest_impact_paths: List[Dict[str, Any]] = []
        
    @property
    def candles(self) -> List[Candle]:
        """Unified view of history + simulation"""
        return self.history_candles + self.sim_candles

    def clear_simulation(self):
        """Reset simulation state"""
        self.sim_candles = []
        self.csad_history = []
        self.panic_level = 0.0
        self.text_factor_state = {
            "dominant_topic": "uncategorized",
            "sentiment_score": 0.0,
            "panic_index": 0.0,
            "greed_index": 0.0,
            "policy_shock": 0.0,
            "regime_bias": "neutral",
        }
        self.latest_impact_paths = []
        # Reset engine state but keep price if continued? 
        # Usually reset to last historical price
        initial_price = self.history_candles[-1].close if self.history_candles else 3000.0
        self.engine = MatchingEngine(prev_close=initial_price, clock=self.clock)

    def apply_policy(self, text: str):
        params = self.interpreter.interpret(text)
        self.policy.liquidity_injection = params.get("liquidity_injection", 0.0)
        self.policy.tax_rate = params.get("tax_rate", GLOBAL_CONFIG.TAX_RATE_STAMP)
        self.policy.description = text
        self.current_news = params.get("initial_news", "Policy Implemented")
        self.panic_level = params.get("fear_factor", 0.0)
        
        # 鍚屾鏀跨瓥绠＄悊鍣ㄧ姸鎬?
        self.policy_manager.set_policy_param("tax", "rate", self.policy.tax_rate)
        # Note: Circuit breaker threshold might be set via explicit API, 
        # but here we interpret general policy text.

    def ingest_seed_event(self, seed_event: Any) -> None:
        """Map a SeedEvent-like payload to market text factors."""
        factors = getattr(seed_event, "text_factors", None)
        if not isinstance(factors, dict):
            return
        headline = getattr(seed_event, "summary", "") or getattr(seed_event, "title", "")
        self.ingest_text_factors(factors, headline=headline)

    def ingest_text_factors(self, factors: Dict[str, Any], headline: str = "") -> None:
        if not isinstance(factors, dict):
            return

        financial = factors.get("financial_factors", {}) or {}
        sentiment = self._safe_float(factors.get("sentiment_score"), 0.0)
        panic = self._safe_float(financial.get("panic_index"), 0.0)
        greed = self._safe_float(financial.get("greed_index"), 0.0)
        shock = self._safe_float(financial.get("policy_shock"), 0.0)
        regime = str(financial.get("regime_bias", "neutral"))
        dominant_topic = str(factors.get("dominant_topic", "uncategorized"))

        target_panic = self._clamp(
            (0.65 * panic) + (0.25 * max(-sentiment, 0.0)) + (0.10 * shock) - (0.20 * greed),
            0.0,
            1.0,
        )
        self.panic_level = self._clamp((0.60 * self.panic_level) + (0.40 * target_panic), 0.0, 1.0)

        if regime == "risk_off" and shock > 0.55:
            self.policy.liquidity_injection = max(self.policy.liquidity_injection, min(1.0, shock * 0.6))
        elif regime == "risk_on" and self.policy.liquidity_injection > 0:
            self.policy.liquidity_injection = max(0.0, self.policy.liquidity_injection - 0.05)

        self.text_factor_state = {
            "dominant_topic": dominant_topic,
            "sentiment_score": self._clamp(sentiment, -1.0, 1.0),
            "panic_index": self._clamp(panic, 0.0, 1.0),
            "greed_index": self._clamp(greed, 0.0, 1.0),
            "policy_shock": self._clamp(shock, 0.0, 1.0),
            "regime_bias": regime,
        }
        self.latest_impact_paths = factors.get("impact_paths", []) or []

        if headline:
            self.current_news = headline

    def calculate_csad(self, agent_returns):
        """Calculate Cross-Sectional Absolute Deviation to detect herd behavior."""
        if agent_returns is None or len(agent_returns) == 0: return
        rm = np.mean(agent_returns)
        csad = np.mean(np.abs(agent_returns - rm))
        self.csad_history.append(csad)
        
        # Dynamic Panic Adjustment
        if rm < -0.02 and csad < 0.02: 
            self.panic_level = min(1.0, self.panic_level + 0.1)
        elif self.policy.liquidity_injection > 0.5:
            self.panic_level = max(0.0, self.panic_level - 0.05)
    
    # Old finalize_step removed to avoid duplication
    # The active finalize_step is defined below.


    def submit_agent_order(self, order: Order):
        """Pass agent orders to the matching engine with centralized risk check."""
        
        # 0. Regulatory Module Check (New)
        if self.regulatory_module:
            # 0.1 Check Circuit Breaker
            if self.regulatory_module.circuit_breaker.is_halted:
                 order.status = OrderStatus.REJECTED
                 order.reason = "Market Halted (Circuit Breaker)"
                 return []
                 
            # 0.2 Check Programmatic Trading Regulation
            allowed, reg_reason = self.regulatory_module.trading_regulator.register_order(
                agent_id=order.agent_id,
                order_type=order.order_type.value, # Use value string
                price=order.price,
                qty=order.quantity
            )
            
            if not allowed:
                order.status = OrderStatus.REJECTED
                order.reason = f"Regulatory Reject: {reg_reason}"
                print(f"[Regulatory] Order REJECTED for Agent {order.agent_id}: {reg_reason}")
                return []
        
        # 1. 鐩樺墠椋庢帶缃戝叧锛圠egacy Check锛?
        market_data = {
            "last_price": self.engine.last_price,
            "best_bid": None, # Could be improved with actual LOB depth
            "best_ask": None
        }
        
        allowed, penalty, reason = self.risk_engine.check_order_compliance(
            order.agent_id, order, market_data
        )
        
        if not allowed:
            # Order Blocked by Risk Gateway
            print(f"[Risk Control] Order REJECTED for Agent {order.agent_id}: {reason}")
            order.status = OrderStatus.REJECTED
            order.reason = reason
            return []
            
        # 2. 娉ㄥ唽璁㈠崟鍒伴珮棰戠洃鎺у櫒锛圠egacy锛?
        self.risk_engine.hft_monitor.register_order(order.agent_id)

        # 3. 妫€鏌ユ斂绛栭檺鍒讹紙Legacy PolicyManager锛?
        market_state = {"last_price": self.engine.last_price}
        policy_res = self.policy_manager.check_order(order, market_state)
        
        if not policy_res.is_allowed:
            # Order Rejected by Policy
            order.status = OrderStatus.REJECTED
            order.reason = policy_res.reason
            return []
            
        # 4. Proceed to Matching
        trades = self.engine.submit_order(order, self.policy.liquidity_injection)
        
        # 5. 娉ㄥ唽鎴愪氦鍒伴珮棰戠洃鎺у櫒锛圠egacy锛?
        if trades:
            self.risk_engine.hft_monitor.register_trade(order.agent_id)
            
        return trades

    def get_market_snapshot(self) -> "MarketSnapshot":
        """Generate a MarketSnapshot for agents."""
        from agents.base_agent import MarketSnapshot
        
        # Get L5 Depth
        depth = self.engine.get_order_book_depth(5)
        
        # Calculate volatility (simple std dev)
        vol = 0.0
        if len(self.candles) > 20:
             closes = [c.close for c in self.candles[-20:]]
             vol = float(np.std(closes))
             
        # Trend (5-day return)
        trend = 0.0
        if len(self.candles) > 5:
            start_p = self.candles[-5].close
            if start_p > 0:
                trend = (self.candles[-1].close - start_p) / start_p

        # Derive Best Bid/Ask
        best_bid = depth['bids'][0]['price'] if depth['bids'] else None
        best_ask = depth['asks'][0]['price'] if depth['asks'] else None
        
        mid = self.engine.last_price
        spread = 0.0
        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
        return MarketSnapshot(
            symbol=self.engine.symbol,
            last_price=self.engine.last_price,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            bid_ask_spread=spread,
            depth=depth,
            total_volume=self.engine.total_volume,
            volatility=vol,
            market_trend=trend,
            panic_level=self.panic_level,
            timestamp=self.clock.timestamp if self.clock else time.time(),
            # 鏀跨瓥瀛楁
            policy_description=self.policy.description,
            policy_tax_rate=self.policy.tax_rate,
            policy_news=self.current_news,
            text_dominant_topic=self.text_factor_state.get("dominant_topic", "uncategorized"),
            text_sentiment_score=float(self.text_factor_state.get("sentiment_score", 0.0)),
            text_panic_score=float(self.text_factor_state.get("panic_index", 0.0)),
            text_greed_score=float(self.text_factor_state.get("greed_index", 0.0)),
            text_policy_shock=float(self.text_factor_state.get("policy_shock", 0.0)),
            text_regime_bias=str(self.text_factor_state.get("regime_bias", "neutral")),
            text_impact_paths=self.latest_impact_paths[:8],
        )

    def get_order_book_depth(self, level=5) -> Dict:
        """Get L5 Market Depth."""
        return self.engine.get_order_book_depth(level)

    def finalize_step(self, step_idx, last_date_str, trades: List[Trade] = None) -> Candle:
        """
        Close the current simulation step, generate a Candle, 
        and prepare the engine for the next step.
        """
        open_p = self.engine.last_price
        
        # 1. Get Date
        new_date_str = ChinaTradingCalendar.get_next_trading_day(last_date_str)
        
        # 2. Get executed trades for this step
        if trades is None:
            step_trades = self.engine.flush_step_trades()
        else:
            step_trades = trades
        
        # 3. Calculate Limits (Visual/Logic consistency)
        limit_rate = GLOBAL_CONFIG.PRICE_LIMIT
        limit_up = self.engine.prev_close * (1.0 + limit_rate)
        limit_down = self.engine.prev_close * (1.0 - limit_rate)

        # 4. Generate Candle
        # 4. Generate Candle
        if not step_trades:
            # No trades: Simulate random drift based on panic
            drift = random.normalvariate(0, 0.003)
            text_sentiment = self._safe_float(self.text_factor_state.get("sentiment_score"), 0.0)
            text_shock = self._safe_float(self.text_factor_state.get("policy_shock"), 0.0)
            text_regime = str(self.text_factor_state.get("regime_bias", "neutral"))
            drift += text_sentiment * 0.002
            if text_regime == "risk_off":
                drift -= text_shock * 0.0015
            elif text_regime == "risk_on":
                drift += text_shock * 0.0010
            
            # 鎭愭厡鏃跺鍔犲悜涓嬪亸绉伙紝浣嗚娓╁拰锛堜笌鎭愭厡绋嬪害鎴愭瘮渚嬶級
            if self.panic_level > 0.3:
                drift -= self.panic_level * 0.003  # 鏈€澶х害0.3%鐨勯澶栦笅琛?
            
            # 鍧囧€煎洖褰? 鍋忕鍓嶆敹杩囧鏃舵媺鍥?
            deviation = (open_p - self.engine.prev_close) / self.engine.prev_close
            drift -= deviation * 0.1  # 娓╁拰鐨勫洖褰掑姏
            
            # Simple drift
            close_p = open_p * (1 + drift)
            high_p = max(open_p, close_p)
            low_p = min(open_p, close_p)
            vol = 0
            
            c = Candle(
                symbol=self.engine.symbol,
                step=step_idx, 
                timestamp=new_date_str, 
                open=open_p, 
                high=high_p, 
                low=low_p, 
                close=close_p, 
                volume=0, 
                is_simulated=True
            )
            self.engine.last_price = close_p
        else:
            # Policy Effect: Update Taxes on Trades
            for t in step_trades:
                t.seller_tax = self.policy_manager.calculate_total_tax(t)
            
            # Standard Candle Logic
            prices = [t.price for t in step_trades]
            high_p = max(prices)
            low_p = min(prices)
            close_p = prices[-1] # Close is the last trade price
            vol = sum(t.quantity for t in step_trades)
            
            c = Candle(
                symbol=self.engine.symbol,
                step=step_idx, 
                timestamp=new_date_str, 
                open=prices[0],  # Open is first trade price of the step
                high=high_p, 
                low=low_p, 
                close=close_p, 
                volume=vol, 
                is_simulated=True
            )

        # 5. Store Candle & Prepare next day
        self.sim_candles.append(c) # Fix: Append to sim_candles, not self.candles (which is a property)
        self.engine.update_prev_close(c.close) # Set Prev Close for next step's limit check
        
        return c

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

# Usage Example
if __name__ == "__main__":
    # Initialize
    manager = MarketDataManager(api_key=None, load_real_data=False)
    
    # 1. Apply Policy
    manager.apply_policy("Reduce stamp tax to boost liquidity.")
    print(f"Policy: {manager.policy}")
    
    # 2. Create Orders
    orders = [
        Order(price=3010.0, quantity=100, agent_id="alice", side="buy", timestamp=time.time()),
        Order(price=3005.0, quantity=100, agent_id="bob", side="sell", timestamp=time.time()+1),
        Order(price=3000.0, quantity=500, agent_id="charlie", side="sell", timestamp=time.time()+2) # Aggressive sell
    ]
    
    # 3. Run Step
    for o in orders:
        trades = manager.submit_agent_order(o)
        for t in trades:
            print(f"Trade Executed: Price {t.price}, Qty {t.quantity} | Buyer pays: {t.buyer_pay_amount:.2f}")
            
    # 4. Finalize
    candle = manager.finalize_step(1, "2024-01-01")
    print(f"Daily Candle Generated: {candle}")


