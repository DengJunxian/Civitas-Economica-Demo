import pytest
import time
import uuid
from core.exchange.order_book import OrderBook
from core.types import Order, OrderSide, OrderType, OrderStatus

@pytest.fixture
def ob():
    return OrderBook(symbol="TEST", prev_close=100.0)

def create_order(side, price, qty, order_type=OrderType.LIMIT):
    return Order(
        symbol="TEST",
        price=price,
        quantity=qty,
        side=side,
        order_type=order_type,
        agent_id="test_agent",
        timestamp=time.time(),
        order_id=str(uuid.uuid4())
    )

def test_add_limit_order(ob):
    order = create_order(OrderSide.BUY, 99.0, 10)
    ob.add_order(order)
    assert len(ob.bids) == 1
    assert len(ob.orders) == 1
    assert ob.bids[0][0] == -99.0 # Max Heap stored as negative

def test_match_limit_orders(ob):
    # 1. Add Sell Order (Maker)
    sell = create_order(OrderSide.SELL, 100.0, 10)
    ob.add_order(sell)
    
    # 2. Add Buy Order (Taker) matching price
    buy = create_order(OrderSide.BUY, 100.0, 5)
    trades = ob.add_order(buy)
    
    assert len(trades) == 1
    t = trades[0]
    assert t.price == 100.0
    assert t.quantity == 5
    assert buy.status == OrderStatus.FILLED
    assert sell.status == OrderStatus.PARTIAL
    assert sell.remaining_qty == 5

def test_price_time_priority(ob):
    # Sellers at same price
    s1 = create_order(OrderSide.SELL, 101.0, 10)
    s1.timestamp = 1000
    ob.add_order(s1)
    
    s2 = create_order(OrderSide.SELL, 101.0, 10)
    s2.timestamp = 2000
    ob.add_order(s2)
    
    # Buy matches both
    b = create_order(OrderSide.BUY, 101.0, 15)
    trades = ob.add_order(b)
    
    assert len(trades) == 2
    # s1 should trade first (earlier timestamp)
    assert trades[0].maker_id == s1.order_id
    assert trades[0].quantity == 10
    
    assert trades[1].maker_id == s2.order_id
    assert trades[1].quantity == 5

def test_market_order(ob):
    # Setup Book
    ob.add_order(create_order(OrderSide.SELL, 100.0, 10))
    ob.add_order(create_order(OrderSide.SELL, 101.0, 10))
    
    # Market Buy
    m = create_order(OrderSide.BUY, 0, 15, OrderType.MARKET)
    trades = ob.add_order(m)
    
    assert len(trades) == 2
    assert trades[0].price == 100.0
    assert trades[1].price == 101.0
    assert m.is_filled

def test_cancel_order(ob):
    o = create_order(OrderSide.BUY, 99.0, 10)
    ob.add_order(o)
    
    assert ob.cancel_order(o.order_id)
    assert o.status == OrderStatus.CANCELLED
    
    # Validate it's removed effectively from match/depth
    # Add matching sell
    s = create_order(OrderSide.SELL, 99.0, 10)
    trades = ob.add_order(s)
    
    assert len(trades) == 0 # Should not match cancelled order
    assert len(ob.bids) == 0 # Should be cleaned during match attempt

def test_get_depth(ob):
    ob.add_order(create_order(OrderSide.BUY, 100.0, 10))
    ob.add_order(create_order(OrderSide.BUY, 99.0, 20))
    ob.add_order(create_order(OrderSide.SELL, 102.0, 5))
    
    depth = ob.get_depth(5)
    assert len(depth['bids']) == 2
    assert depth['bids'][0]['price'] == 100.0
    assert depth['bids'][1]['price'] == 99.0
    assert len(depth['asks']) == 1
    assert depth['asks'][0]['price'] == 102.0
