
import random
import numpy as np

# Mock class
class Engine:
    def __init__(self):
        self.prev_close = 3000.0
        self.last_price = 3000.0

class Market:
    def __init__(self):
        self.engine = Engine()
        self.panic_level = 0.0 # Initial panic
        self.prices = [3000.0]

    def step(self, has_trades=False):
        open_p = self.engine.last_price
        
        if not has_trades:
            # New Logic
            drift = random.normalvariate(0, 0.003)
            
            if self.panic_level > 0.3:
                drift -= self.panic_level * 0.003
            
            # Mean reversion
            deviation = (open_p - self.engine.prev_close) / self.engine.prev_close
            drift -= deviation * 0.1
            
            close_p = open_p * (1 + drift)
        else:
            close_p = open_p # Assume constant for simplicity in this test
            
        self.engine.last_price = close_p
        self.prices.append(close_p)
        return close_p

# Simulation 1: No Trades, Low Panic (Should be stable)
m1 = Market()
print("\n--- Simulation 1: Low Panic, No Trades ---")
for _ in range(10):
    p = m1.step(False)
    print(f"Day {_}: {p:.2f}")

# Simulation 2: High Panic, No Trades (Should drift down but not crash 1% daily)
print("\n--- Simulation 2: High Panic (0.8), No Trades ---")
m2 = Market()
m2.panic_level = 0.8
for _ in range(10):
    p = m2.step(False)
    print(f"Day {_}: {p:.2f}")

# Simulation 3: Check drop percentage
drop = (m2.prices[-1] - m2.prices[0]) / m2.prices[0]
print(f"\n10 Day Drop (High Panic): {drop:.2%}")
assert abs(drop) < 0.10, "Drop is too severe!"
print("Test Passed")
