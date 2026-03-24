# file: core/exchange/order_book.py
"""
High-Performance Central Limit Order Book (CLOB)
Inspired by ABIDES (Agent-Based Interactive Discrete Event Simulation).

Implementation Details:
- Uses `heapq` (Binary Heap) for O(log N) order insertion and retrieval.
- Lazy deletion for cancelled orders to maintain high performance.
- Supports Limit and Market orders.
- Price-Time Priority matching.

Data Structures:
- Bids: Max-Heap (simulated using Min-Heap with negated prices).
- Asks: Min-Heap.
"""

import heapq
import time
import uuid
from typing import Dict, List, Optional, Tuple

from config import GLOBAL_CONFIG
from core.types import Order, Trade, OrderSide, OrderType, OrderStatus
from core.utils import PriceQuantizer

class OrderBook:
    """
    ABIDES-style Order Book using Heaps for O(log N) matching.
    """
    
    def __init__(self, symbol: str = "A_SHARE_IDX", prev_close: float = 3000.0):
        self.symbol = symbol
        self.prev_close = prev_close
        
        # Heaps store tuples: 
        # Bids: (-price, timestamp, order_id) -> Max-Heap by price, Min by time
        # Asks: (price, timestamp, order_id)  -> Min-Heap by price, Min by time
        self.bids: List[Tuple[float, float, str]] = []
        self.asks: List[Tuple[float, float, str]] = []
        
        # Order Registry: order_id -> Order object
        self.orders: Dict[str, Order] = {}
        
        # Statistics
        self.last_price: float = prev_close
        self.total_volume: int = 0
        self.trades_history: List[Trade] = []
        self._step_trades: List[Trade] = []
        self.submitted_orders: int = 0
        self.cancelled_orders: int = 0
        self.trade_count: int = 0

    def _get_dynamic_limit(self) -> float:
        """Get dynamic price limits based on symbol."""
        limit = GLOBAL_CONFIG.PRICE_LIMIT
        sym = str(self.symbol)
        if sym.startswith("688") or sym.startswith("300"): limit = 0.20
        elif sym.startswith("8"): limit = 0.30
        elif "ST" in sym.upper(): limit = 0.05
        return limit

    def get_limit_prices(self) -> Tuple[float, float]:
        """Get (Lower Limit, Upper Limit) price."""
        limit = self._get_dynamic_limit()
        return PriceQuantizer.get_limit_prices(self.prev_close, limit)

    def _check_price_limit(self, price: float) -> bool:
        lower, upper = self.get_limit_prices()
        return lower <= price <= upper

    def _would_cross_book(self, order: Order) -> bool:
        if order.side == OrderSide.BUY:
            best_ask = self.get_best_ask()
            return best_ask is not None and order.price >= best_ask
        best_bid = self.get_best_bid()
        return best_bid is not None and order.price <= best_bid

    def estimate_available_qty(self, side: OrderSide, price: float) -> int:
        """Estimate immediate executable quantity against the opposite book."""
        if side == OrderSide.BUY:
            candidates = [
                o for o in self.orders.values()
                if o.side == OrderSide.SELL and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and o.price <= price
            ]
            candidates.sort(key=lambda o: (o.price, o.timestamp))
        else:
            candidates = [
                o for o in self.orders.values()
                if o.side == OrderSide.BUY and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and o.price >= price
            ]
            candidates.sort(key=lambda o: (-o.price, o.timestamp))
        return int(sum(max(0, o.remaining_qty) for o in candidates))

    def estimate_queue_position(self, order: Order) -> int:
        """Approximate the queue ahead of an incoming order."""
        if order.side == OrderSide.BUY:
            ahead = [
                o for o in self.orders.values()
                if o.side == OrderSide.BUY
                and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and (o.price > order.price or (o.price == order.price and o.timestamp < order.timestamp))
            ]
        else:
            ahead = [
                o for o in self.orders.values()
                if o.side == OrderSide.SELL
                and o.status not in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]
                and (o.price < order.price or (o.price == order.price and o.timestamp < order.timestamp))
            ]
        return int(sum(max(0, o.remaining_qty) for o in ahead))

    def modify_order(
        self,
        order_id: str,
        *,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        timestamp: Optional[float] = None,
        original_order: Optional[Order] = None,
    ) -> Optional[Order]:
        """Create a cancel-replace order without mutating heap keys in place."""
        source = original_order or self.orders.get(order_id)
        if source is None or source.status in [OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.REJECTED]:
            return None

        self.cancel_order(order_id)
        replacement = Order(
            symbol=source.symbol,
            price=float(price if price is not None else source.price),
            quantity=int(quantity if quantity is not None else source.remaining_qty or source.quantity),
            side=source.side,
            order_type=source.order_type,
            agent_id=source.agent_id,
            timestamp=float(timestamp if timestamp is not None else time.time()),
            metadata=dict(source.metadata),
        )
        replacement.metadata.update(
            {
                "replaces_order_id": order_id,
                "replace_reason": "modify_order",
            }
        )
        return replacement

    def add_order(self, order: Order) -> List[Trade]:
        """
        Add an order to the book and attempt to match immediately.
        """
        if order.symbol != self.symbol:
            order.status = OrderStatus.REJECTED
            return []
            
        # 1. Price Validation
        if order.order_type == OrderType.LIMIT:
            if not self._check_price_limit(order.price):
                order.status = OrderStatus.REJECTED
                return []
        
        # 2. Market Order Handling
        if order.order_type == OrderType.MARKET:
            lower, upper = self.get_limit_prices()
            order.price = upper if order.side == OrderSide.BUY else lower

        # 2.5 Post-only / FOK pre-checks
        if order.order_type == OrderType.POST_ONLY and self._would_cross_book(order):
            order.status = OrderStatus.REJECTED
            order.reason = "post-only order would take liquidity"
            return []
        if order.order_type == OrderType.FOK:
            available = self.estimate_available_qty(order.side, order.price)
            if available < order.quantity:
                order.status = OrderStatus.REJECTED
                order.reason = "insufficient liquidity for FOK"
                return []

        # 3. Register Order
        self.orders[order.order_id] = order
        order.status = OrderStatus.PENDING
        self.submitted_orders += 1
        
        # 4. Attempt Matching (Incoming vs Book)
        trades = []
        if order.side == OrderSide.BUY:
            trades = self._match_incoming_buy(order)
        else:
            trades = self._match_incoming_sell(order)
        
        # 5. If not filled, add to heap
        resting_allowed = order.order_type in {OrderType.LIMIT, OrderType.POST_ONLY}
        if not order.is_filled and resting_allowed and order.status not in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._push_to_heap(order)
        elif not order.is_filled and order.status not in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            # IOC / MARKET remainders are not allowed to rest.
            order.reason = order.reason or "unfilled remainder canceled"
            
        return trades

    def _match_incoming_buy(self, order: Order) -> List[Trade]:
        """Match incoming Buy order against Asks heap."""
        trades = []
        while order.remaining_qty > 0 and self.asks:
            # Clean top of Asks
            self._clean_heap_top(OrderSide.SELL)
            if not self.asks: break
            
            # Peek Best Ask: (price, ts, id)
            ask_price, ask_ts, ask_id = self.asks[0]
            
            # Check Match
            if order.price >= ask_price:
                ask_order = self.orders[ask_id]
                trade = self._execute_trade(order, ask_order)
                trades.append(trade)
                
                # If ask filled, _clean_heap_top will remove it next iter
                # But we should probably pop it now if filled to save ops?
                # _clean_heap_top handles it safely.
            else:
                break
        return trades

    def _match_incoming_sell(self, order: Order) -> List[Trade]:
        """Match incoming Sell order against Bids heap."""
        trades = []
        while order.remaining_qty > 0 and self.bids:
            # Clean top of Bids
            self._clean_heap_top(OrderSide.BUY)
            if not self.bids: break
            
            # Peek Best Bid: (-price, ts, id)
            neg_bid_price, bid_ts, bid_id = self.bids[0]
            bid_price = -neg_bid_price
            
            # Check Match
            if order.price <= bid_price:
                bid_order = self.orders[bid_id]
                trade = self._execute_trade(bid_order, order)
                trades.append(trade)
            else:
                break
        return trades

    def _push_to_heap(self, order: Order):
        """Push order to the appropriate heap."""
        if order.side == OrderSide.BUY:
            # Negate price for Max-Heap behavior using Min-Heap
            heapq.heappush(self.bids, (-order.price, order.timestamp, order.order_id))
        else:
            heapq.heappush(self.asks, (order.price, order.timestamp, order.order_id))

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order. Uses lazy deletion (marks status, removed later).
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            if not order.is_filled:
                order.status = OrderStatus.CANCELLED
                self.cancelled_orders += 1
                # We do NOT remove from self.orders yet, or the heap pop will fail to find it.
                return True
        return False

    def _clean_heap_top(self, side: OrderSide):
        """
        Remove invalid (cancelled/filled) orders from the top of the heap.
        """
        heap = self.bids if side == OrderSide.BUY else self.asks
        
        while heap:
            # Peek top
            price_key, timestamp, order_id = heap[0]
            
            # Check validity
            if order_id not in self.orders:
                heapq.heappop(heap)
                continue
                
            order = self.orders[order_id]
            if order.status == OrderStatus.CANCELLED or order.is_filled:
                heapq.heappop(heap)
                # Cleanup dict to free memory
                if order.status == OrderStatus.CANCELLED or order.is_filled:
                    if order_id in self.orders: del self.orders[order_id]
            else:
                break

    def _execute_trade(self, bid_order: Order, ask_order: Order) -> Trade:
        """
        Execute a single trade between two valid orders.
        """
        # Determine Execution Price (Maker's Price)
        # If Bid is older (Maker), price is Bid Price.
        # If Ask is older (Maker), price is Ask Price.
        # However, basic rule is usually 'Incoming order trades at resting order's price'.
        # Since we are inside match(), we treat the one on the book 'first' effectively as maker?
        # Actually in a continuous double auction, the 'resting' order sets the price. 
        # Here both satisfy the condition. Who was there first?
        # Comparing timestamps:
        if bid_order.timestamp < ask_order.timestamp:
            exec_price = bid_order.price
            maker, taker = bid_order, ask_order
        else:
            exec_price = ask_order.price
            maker, taker = ask_order, bid_order
            
        exec_qty = min(bid_order.remaining_qty, ask_order.remaining_qty)
        
        # Update Orders
        bid_order.filled_qty += exec_qty
        ask_order.filled_qty += exec_qty
        
        for o in [bid_order, ask_order]:
            if o.is_filled: o.status = OrderStatus.FILLED
            else: o.status = OrderStatus.PARTIAL
            
        # Calculate Fees
        notional = exec_price * exec_qty
        comm = notional * GLOBAL_CONFIG.TAX_RATE_COMMISSION
        stamp = notional * GLOBAL_CONFIG.TAX_RATE_STAMP
        
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            price=exec_price,
            quantity=exec_qty,
            maker_id=maker.order_id,
            taker_id=taker.order_id,
            maker_agent_id=maker.agent_id,
            taker_agent_id=taker.agent_id,
            buyer_agent_id=bid_order.agent_id,
            seller_agent_id=ask_order.agent_id,
            timestamp=max(bid_order.timestamp, ask_order.timestamp, time.time()), # Trade time is now/latest
            buyer_fee=comm,
            seller_fee=comm,
            seller_tax=stamp
        )
        
        # Update Stats
        self.last_price = exec_price
        self.total_volume += exec_qty
        self.trades_history.append(trade)
        self._step_trades.append(trade)
        self.trade_count += 1

        return trade

    def get_best_bid(self) -> Optional[float]:
        self._clean_heap_top(OrderSide.BUY)
        return -self.bids[0][0] if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        self._clean_heap_top(OrderSide.SELL)
        return self.asks[0][0] if self.asks else None

    def get_spread(self) -> Optional[float]:
        """
        获取当前买一卖一价差。
        当任一侧盘口为空时返回 None。
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return float(best_ask - best_bid)

    def get_depth(self, levels: int = 5) -> Dict:
        """
        Get L2 Market Depth. 
        NOTE: This is expensive O(N log N) if heaps are large, as we must copy and sort to find top N valid orders.
        """
        # Create copies to pop from without disturbing main heaps
        # Or iterate directly?
        # heapq.nsmallest is efficient.
        
        # Bids matched by smallest (-price, timestamp) -> Highest Price
        # We need to filter cancelled ones.
        
        valid_bids = []
        # We can't easily iterate heap in order without popping.
        # But we can use nsmallest on the list safely? 
        # The list is a heap, nsmallest works on any iterable but optimized for heaps?
        # Actually nsmallest on a heap is not necessarily skipping the valid check.
        
        # Strategy: Pop from a copy until we get `levels` valid orders.
        # Bids
        temp_bids = self.bids[:] # Copy list O(N)
        # It's already heapified.
        final_bids = []
        while temp_bids and len(final_bids) < levels:
            p, t, oid = heapq.heappop(temp_bids)
            if oid in self.orders and self.orders[oid].status not in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
                final_bids.append({"price": -p, "qty": self.orders[oid].remaining_qty})
                
        # Asks
        temp_asks = self.asks[:]
        final_asks = []
        while temp_asks and len(final_asks) < levels:
            p, t, oid = heapq.heappop(temp_asks)
            if oid in self.orders and self.orders[oid].status not in [OrderStatus.CANCELLED, OrderStatus.FILLED]:
                final_asks.append({"price": p, "qty": self.orders[oid].remaining_qty})
                
        return {"bids": final_bids, "asks": final_asks}

    def flush_step_trades(self) -> List[Trade]:
        t = self._step_trades[:]
        self._step_trades = []
        return t

    def update_prev_close(self, close: float):
        self.prev_close = close

    def get_activity_stats(self) -> Dict[str, int]:
        return {
            "submitted_orders": int(self.submitted_orders),
            "cancelled_orders": int(self.cancelled_orders),
            "trade_count": int(self.trade_count),
        }

    def clear(self):
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
        self.submitted_orders = 0
        self.cancelled_orders = 0
        self.trade_count = 0
