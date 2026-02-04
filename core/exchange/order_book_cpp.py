import time
from typing import List, Optional, Dict, Tuple
from core.exchange.order_book import OrderBook, Order, Trade, OrderStatus, OrderSide

try:
    import _civitas_lob
except ImportError:
    _civitas_lob = None

class OrderBookCPP(OrderBook):
    """
    C++ Optimized OrderBook Wrapper
    
    Inherits from OrderBook to ensure interface compatibility.
    Delegate core operations to _civitas_lob.LimitOrderBook.
    """
    def __init__(self, symbol: str = "A_SHARE_IDX", prev_close: float = 3000.0):
        super().__init__(symbol, prev_close)
        if _civitas_lob is None:
            raise ImportError("C++ extension _civitas_lob not found. Please build with setup.py.")
        self._cpp_lob = _civitas_lob.LimitOrderBook(symbol)
        
    def add_order(self, order: Order) -> List[Trade]:
        # 1. Validation (reusing base class logic or simplified)
        if order.symbol != self.symbol:
            order.status = OrderStatus.REJECTED
            return []
            
        if order.order_type == "limit":
             # Optimization: Check primitive limit directly locally if needed, but keeping it simple
             pass # C++ doesn't check limits in current logic, relying on pre-check or implementing it in C++
             # For speed, we skip redundant Python checks if we trust caller or C++ handles it.
             # But here we stick to verifying.
             if not self._check_price_limit(order.price):
                order.status = OrderStatus.REJECTED
                return []
        
        if order.order_type == "market":
             lower, upper = self.get_limit_prices()
             # Convert to aggressive limit order
             order.price = upper if order.side == OrderSide.BUY else lower

        # 2. Call C++ Exploded Interface (Fast)
        # Handle Enum to string conversion
        side_str = order.side.value if hasattr(order.side, 'value') else str(order.side)
        type_str = order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type)

        # Returns (filled_qty, status_str, list_of_trade_tuples)
        filled_qty, status_str, trades_tuples = self._cpp_lob.add_order_exploded(
             str(order.order_id),
             str(order.agent_id),
             float(order.timestamp),
             str(order.symbol),
             side_str,
             type_str,
             float(order.price),
             float(order.quantity)
        )
        
        # 3. Sync status back to Python Order
        order.filled_qty = int(filled_qty)
        
        # Map string status back to Enum
        # Optimization: Manual mapping or cached
        if status_str == "filled":
             order.status = OrderStatus.FILLED
        elif status_str == "partial":
             order.status = OrderStatus.PARTIAL
        elif status_str == "cancelled":
             order.status = OrderStatus.CANCELLED
        else:
             order.status = OrderStatus.PENDING # Default/Rejected
        
        # 4. Convert Trades (Deferred or Lazy could be better but we need them now)
        trades = []
        for t_tuple in trades_tuples:
            # Unpack tuple
            (tid, price, qty, mid, tid2, maid, taid, ts, bfee, sfee, stax) = t_tuple
            
            t = Trade(
                trade_id=tid,
                price=price,
                quantity=int(qty),
                maker_id=mid,
                taker_id=tid2,
                maker_agent_id=maid,
                taker_agent_id=taid,
                timestamp=ts,
                buyer_fee=bfee,
                seller_fee=sfee,
                seller_tax=stax
            )
            trades.append(t)
            
            # Update history/stats in Python layer for compatibility
            self.last_price = t.price
            self.total_volume += int(t.quantity)
            self.trades_history.append(t)
            self._step_trades.append(t)

        return trades

    def cancel_order(self, order_id: str) -> bool:
        return self._cpp_lob.cancel_order(str(order_id))
    
    # Override data access methods
    def get_best_bid(self) -> Optional[float]:
        p = self._cpp_lob.get_best_bid()
        return p if p > 0 else None

    def get_best_ask(self) -> Optional[float]:
        p = self._cpp_lob.get_best_ask()
        return p if p > 0 else None
        
    def get_depth(self, levels: int = 5) -> Dict:
        # C++ returns {"bids": [[price, qty], ...], ...} pairs
        depth = self._cpp_lob.get_depth(levels)
        
        bids_formatted = []
        for price, qty in depth["bids"]:
            bids_formatted.append({"price": price, "qty": qty})
            
        asks_formatted = []
        for price, qty in depth["asks"]:
            asks_formatted.append({"price": price, "qty": qty})
            
        return {"bids": bids_formatted, "asks": asks_formatted}

    def clear(self):
        super().clear()
        self._cpp_lob.clear()
