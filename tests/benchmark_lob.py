import time
import random
import uuid
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from core.exchange.order_book import OrderBook, Order
try:
    from core.exchange.order_book_cpp import OrderBookCPP
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("[-] OrderBookCPP not available. Please build the extension.")

def generate_random_orders(n=10000):
    orders = []
    symbol = "TEST"
    sides = ["buy", "sell"]
    for i in range(n):
        side = sides[random.randint(0, 1)]
        price = round(random.uniform(90.0, 110.0), 2)
        qty = random.randint(1, 100)
        order = Order.create(
            agent_id=f"agent_{i}",
            symbol=symbol,
            side=side,
            order_type="limit",
            price=price,
            quantity=qty
        )
        orders.append(order)
    return orders

def run_benchmark(lob_cls, orders, name="LOB"):
    lob = lob_cls("TEST", 100.0)
    
    start_time = time.time()
    for order in orders:
        # Clone order to ensure fresh state if reusing list (though orders are objects, we modify status)
        # For fair bench, we assume creating orders is outside timing or fast enough
        # We just pass the object.
        lob.add_order(order)
    end_time = time.time()
    
    duration = end_time - start_time
    ops = len(orders) / duration
    print(f"[{name}] Processed {len(orders)} orders in {duration:.4f}s ({ops:,.0f} orders/sec)")
    return ops

def main():
    print("="*60)
    print("LOB Benchmark: Python vs C++")
    print("="*60)
    
    N = 100000
    print(f"Generating {N} random orders...")
    orders = generate_random_orders(N)
    
    # Deep copy needed because add_order modifies state (filled_qty)
    # Actually, we can just recreate orders or let them rely on logic not to fail.
    # But Python OrderBook modifies order status.
    # Let's regenerate for second run to be fair.
    orders_cpp = generate_random_orders(N)
    
    print("\nRunning Python OrderBook Benchmark...")
    py_ops = run_benchmark(OrderBook, orders, "Python")
    
    if HAS_CPP:
        print("\nRunning C++ OrderBook Benchmark...")
        cpp_ops = run_benchmark(OrderBookCPP, orders_cpp, "C++")
        
        speedup = cpp_ops / py_ops
        print(f"\nSpeedup: {speedup:.2f}x")
        
        if speedup > 5.0:
            print("[SUCCESS] C++ implementation is significantly faster.")
        else:
            print("[WARNING] Speedup is lower than expected.")
            
        # Verify correctness (basic)
        # We can check best bid/ask of both
        # Note: Since random orders are different, we can't compare state directly.
        # But we validated integration earlier.
    else:
        print("\n[!] Skipping C++ Benchmark.")

if __name__ == "__main__":
    main()
