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
    from core.exchange.order_book_cpp import OrderBookCPP
    from core.exchange.order_book import Order as OrderModel
    USE_CPP_LOB = True
    print("[*] High-Performance C++ OrderBook Activated")
except ImportError as e:
    USE_CPP_LOB = False
    print(f"[!] Falling back to Python: {e}")

# Assumes a config.py exists with these constants. 
from config import GLOBAL_CONFIG

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
        涨跌停检查 (Delegated to OrderBook)
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

    def run_call_auction(self, orders: List[Order]) -> Tuple[float, List[Trade]]:
        """
        集合竞价（Call Auction）
        
        模拟A股市场9:15-9:25的集合竞价阶段。
        通过计算能使成交量最大化的价格来确定开盘价。
        
        算法：
        1. 将所有买卖订单按价格排序
        2. 计算每个价格水平的累计买卖量
        3. 找到使成交量最大化的价格作为开盘价
        
        Args:
            orders: 集合竞价阶段的所有订单
            
        Returns:
            (开盘价, 成交列表)
        """
        if not orders:
            return self.prev_close, []
        
        # 分离买卖订单
        buy_orders = [o for o in orders if o.side == 'buy']
        sell_orders = [o for o in orders if o.side == 'sell']
        
        if not buy_orders or not sell_orders:
            return self.prev_close, []
        
        # 收集所有可能的价格水平
        all_prices = set()
        for o in orders:
            if self._check_price_limit(o.price):
                all_prices.add(o.price)
        
        if not all_prices:
            return self.prev_close, []
        
        price_levels = sorted(all_prices)
        
        # 计算每个价格水平的成交量
        best_price = self.prev_close
        max_volume = 0
        
        for test_price in price_levels:
            # 愿意以 test_price 或更高价格买入的总量
            buy_volume = sum(o.quantity for o in buy_orders if o.price >= test_price)
            # 愿意以 test_price 或更低价格卖出的总量
            sell_volume = sum(o.quantity for o in sell_orders if o.price <= test_price)
            # 成交量取较小值
            match_volume = min(buy_volume, sell_volume)
            
            if match_volume > max_volume:
                max_volume = match_volume
                best_price = test_price
            elif match_volume == max_volume and match_volume > 0:
                # 相同成交量时，选择更接近前收盘价的价格
                if abs(test_price - self.prev_close) < abs(best_price - self.prev_close):
                    best_price = test_price
        
        # 以开盘价执行撮合
        trades = []
        opening_price = best_price
        
        # 筛选可成交的订单
        executable_buys = sorted(
            [o for o in buy_orders if o.price >= opening_price],
            key=lambda x: (-x.price, x.timestamp)  # 价格优先，时间优先
        )
        executable_sells = sorted(
            [o for o in sell_orders if o.price <= opening_price],
            key=lambda x: (x.price, x.timestamp)
        )
        
        # 执行撮合
        buy_idx, sell_idx = 0, 0
        while buy_idx < len(executable_buys) and sell_idx < len(executable_sells):
            buy_order = executable_buys[buy_idx]
            sell_order = executable_sells[sell_idx]
            
            # 计算成交量
            match_qty = min(buy_order.remaining_qty, sell_order.remaining_qty)
            
            if match_qty > 0:
                # 创建成交记录
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
                
                # [NEW] Apply Policy Effects (Tax)
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
                
                # 更新订单状态
                buy_order.filled_qty += match_qty
                sell_order.filled_qty += match_qty
            
            if buy_order.is_filled:
                buy_idx += 1
            if sell_order.is_filled:
                sell_idx += 1
        
        # 更新引擎状态
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
    """真实市场数据加载器，使用 akshare 获取上证指数历史数据。"""
    
    @staticmethod
    def load_history(symbol="sh000001", period="365") -> List[Candle]:
        """
        加载上证指数历史K线数据。
        
        Args:
            symbol: 指数代码，默认上证指数
            period: 加载天数，默认365天
        """
        try:
            cache_dir = "data/cache"
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{symbol}_{period}.csv")
            
            if os.path.exists(cache_file):
                print(f"[*] Loading {symbol} from cache: {cache_file}")
                df = pd.read_csv(cache_file)
            else:
                print(f"[*] Loading {period} days of {symbol} from akshare...")
                try:
                     df = ak.stock_zh_index_daily(symbol=symbol)
                     if df is not None and not df.empty:
                         df.to_csv(cache_file, index=False)
                except Exception as e:
                    print(f"[!] Akshare failed: {e}")
                    df = None

            # Fallback to default data if everything else fails
            if df is None or df.empty:
                default_file = "data/sh000001_default.csv"
                if os.path.exists(default_file):
                    print(f"[!] Using default fallback data: {default_file}")
                    df = pd.read_csv(default_file)
                else:
                    print(f"[!] No default data found at {default_file}")
            
            if df is None or df.empty: 
                raise ValueError(f"未获取到数据且无缓存或默认文件。请检查网络或确保 {default_file} 存在。")
            
            df = df.tail(int(period)).reset_index(drop=True)
            candles = []
            cnt = len(df)
            for i, row in df.iterrows():
                o = float(row.get('open', row.get('Open', 3000)))
                h = float(row.get('high', row.get('High', 3000)))
                l = float(row.get('low', row.get('Low', 3000)))
                c = float(row.get('close', row.get('Close', 3000)))
                v = int(row.get('volume', row.get('Volume', 0)))
                d = str(row.get('date', row.get('Date', '2024-01-01')))
                
                candles.append(Candle(
                    symbol=symbol,
                    step=-(cnt - i), 
                    timestamp=d, 
                    open=o, high=h, low=l, close=c, 
                    volume=v, 
                    is_simulated=False
                ))
            print(f"[OK] 成功加载 {len(candles)} 个交易日数据")
            return candles
        except Exception as e:
            print(f"[!] 数据加载失败: {e}")
            return [Candle(symbol, -1, "2024-01-01", 3000, 3000, 3000, 3000, 100000, 0.0, False)]

class PolicyInterpreter:
    """
    政策解释器：通过 DeepSeek 或 GLM 分析政策文本，量化其对市场的影响。
    支持使用 ModelRouter 进行多模型路由。
    """
    
    def __init__(self, api_key_or_router):
        # 支持传入 router 或 key (兼容旧接口)
        if hasattr(api_key_or_router, 'call_with_fallback'):
            self.router = api_key_or_router
            self.api_key = self.router.deepseek_key if self.router.deepseek_key else "dummy"
        else:
            self.router = None
            self.api_key = api_key_or_router
            
        self.last_reasoning = None  # 保存最近一次的推理过程

    async def interpret(self, policy_text: str) -> Dict:
        """
        分析政策文本，返回量化参数。
        
        Args:
            policy_text: 政策描述文本
            
        Returns:
            Dict: 包含 tax_rate, liquidity_injection, fear_factor, initial_news, 
                  sentiment_shift, reasoning 等字段
        """
        if not self.api_key: 
            return self._default_policy()

        # 构造 prompt
        prompt = f"""你是一位资深的A股市场政策分析师。请分析以下政策对市场的影响，并给出量化参数。

【待分析政策】
{policy_text}

【当前市场基准】
- 印花税率: {GLOBAL_CONFIG.TAX_RATE_STAMP:.4%}
- 涨跌停限制: {GLOBAL_CONFIG.PRICE_LIMIT:.0%}

【分析要求】
1. 直接效应：流动性和交易成本影响
2. 信号效应：政策的隐含信号和市场解读
3. 二阶认知：投资者预期的自我实现
4. 时效性：短期情绪 vs 中期基本面

【输出格式】
严格返回 JSON 格式，不要包含 Markdown 格式标记：
{{
    "tax_rate": <新印花税率，float>,
    "liquidity_injection": <流动性注入概率，0.0-1.0>,
    "fear_factor": <恐慌因子，0.0-1.0>,
    "sentiment_shift": <情绪偏移量，-1.0到1.0>,
    "initial_news": "<简短新闻标题>",
    "market_impact": "<一句话总结>",
    "reasoning_summary": "<分析过程摘要>"
}}
"""
        try:
            # 使用 Router 或 AsyncClient (DeepSeek)
            # 为了在 Loop Architecture 中高效运行，这里必须是异步调用
            
            content = ""
            reasoning = ""
            
            if self.router:
                # 使用 GLM-4-FlashX 快速响应 或者 DeepSeek (Router 内部 handle 异步)
                # 默认优先级
                priority = ["deepseek-reasoner", "deepseek-chat", "glm-4-flashx"]
                
                response = await self.router.call_with_fallback(
                    [{"role": "user", "content": prompt}],
                    priority_models=priority,
                    timeout_budget=60.0
                )
                
                # router 返回的是 content, reasoning, model
                content = response[0]
                reasoning = response[1]
                
            else:
                # 直接使用 AsyncOpenAI
                from openai import AsyncOpenAI
                
                client = AsyncOpenAI(
                    api_key=self.api_key, 
                    base_url=GLOBAL_CONFIG.API_BASE_URL,
                    timeout=GLOBAL_CONFIG.API_TIMEOUT_REASONER
                )
                
                print(f"[PolicyInterpreter] 异步调用 deepseek-reasoner...")
                resp = await client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                message = resp.choices[0].message
                content = message.content
                reasoning = getattr(message, 'reasoning_content', '')

            self.last_reasoning = reasoning
            
            # 解析 JSON
            import json
            import re
            
            # 清理 Markdown
            if "```" in content: 
                content = re.sub(r"```json|```", "", content).strip()
            
            # 尝试解析
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # 尝试修复常见 JSON 错误
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    raise ValueError("无法提取 JSON")

            result['reasoning'] = self.last_reasoning
            print(f"[OK] 政策分析完成: {result.get('initial_news', '未知')}")
            return result

        except Exception as e:
            print(f"[!] 政策分析失败: {e}")
            return self._default_policy()

    def _default_policy(self) -> Dict:
        """默认政策参数（API不可用时使用）"""
        return {
            "tax_rate": GLOBAL_CONFIG.TAX_RATE_STAMP, 
            "liquidity_injection": 0.0, 
            "fear_factor": 0.0,
            "sentiment_shift": 0.0,
            "initial_news": "政策已发布", 
            "market_impact": "影响待评估",
            "reasoning": "(使用默认参数)"
        }

# ==========================================
# PART 5: Market Data Manager
# (Integrated Logic)
# ==========================================

from core.policy import PolicyManager  
from core.regulation.risk_control import RiskEngine  # [NEW]

class MarketDataManager:
    def __init__(self, api_key_or_router, load_real_data=True, clock: Optional[SimulationClock] = None):
        self.policy = PolicyState()
        self.interpreter = PolicyInterpreter(api_key_or_router)
        self.clock = clock
        
        # [NEW] Policy Manager
        self.policy_manager = PolicyManager()
        
        # [NEW] Risk Engine (Centralized Gateway)
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
        
    @property
    def candles(self) -> List[Candle]:
        """Unified view of history + simulation"""
        return self.history_candles + self.sim_candles

    def clear_simulation(self):
        """Reset simulation state"""
        self.sim_candles = []
        self.csad_history = []
        self.panic_level = 0.0
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
        
        # [NEW] Sync with PolicyManager
        self.policy_manager.set_policy_param("tax", "rate", self.policy.tax_rate)
        # Note: Circuit breaker threshold might be set via explicit API, 
        # but here we interpret general policy text.

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
        # 1. [NEW] Pre-Matching Risk Gateway (Extreme Speed Check)
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
            
        # 2. [NEW] Register order in HFT monitor (for OTR calculation)
        self.risk_engine.hft_monitor.register_order(order.agent_id)

        # 3. [NEW] Check Policy (Circuit Breaker, etc.)
        market_state = {"last_price": self.engine.last_price}
        policy_res = self.policy_manager.check_order(order, market_state)
        
        if not policy_res.is_allowed:
            # Order Rejected by Policy
            order.status = OrderStatus.REJECTED
            order.reason = policy_res.reason
            return []
            
        # 4. Proceed to Matching
        trades = self.engine.submit_order(order, self.policy.liquidity_injection)
        
        # 5. [NEW] Register trades in HFT monitor (for OTR calculation)
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
            timestamp=self.clock.timestamp if self.clock else time.time()
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
            drift = random.normalvariate(0, 0.005)
            if self.panic_level > 0.5: drift -= 0.01
            
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