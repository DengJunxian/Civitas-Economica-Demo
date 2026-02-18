
import pytest
import numpy as np
from unittest.mock import MagicMock
from core.market_engine import MatchingEngine
from agents.population import StratifiedPopulation
from core.types import Order

def test_market_engine_auction_signature():
    """验证 run_call_auction 签名修复"""
    engine = MatchingEngine()
    # 应该接受 market_time 参数而不报错
    try:
        engine.run_call_auction([], market_time=12345.6)
    except TypeError as e:
        pytest.fail(f"MatchingEngine.run_call_auction signature error: {e}")

def test_tier2_sync_execution():
    """验证 Tier 2 资金/持仓同步 (包括重复索引)"""
    # Fix: n_smart=0 leads to error in _build_influence_network. 
    # Mocking internal methods to bypass complex init
    with MagicMock() as mock_init:
        StratifiedPopulation._build_influence_network = MagicMock(return_value=np.zeros(10, dtype=int))
        StratifiedPopulation._build_smart_social_network = MagicMock(return_value={})
        StratifiedPopulation._init_smart_agents = MagicMock()
        
        pop = StratifiedPopulation(n_smart=1, n_vectorized=10)
    
    # 手动设置初始状态
    # Index 0: Cash 10000, Hold 100
    pop.state[0, pop.IDX_CASH] = 10000.0
    pop.state[0, pop.IDX_HOLDINGS] = 100.0
    
    # 模拟成交: 
    # Agent 0 买入 10股 @ 10元 (Cost=100) -> Cash -100, Hold +10
    # Agent 0 买入 20股 @ 10元 (Cost=200) -> Cash -200, Hold +20
    # 两个成交记录指向同一个 Agent 0
    
    indices = np.array([0, 0])
    prices = np.array([10.0, 10.0])
    qtys = np.array([10, 20])
    directions = np.array([1, 1]) # Buy
    
    pop.sync_tier2_execution(indices, prices, qtys, directions)
    
    # 验证结果
    # Cash: 10000 - 100 - 200 = 9700
    expected_cash = 9700.0
    actual_cash = pop.state[0, pop.IDX_CASH]
    
    # Holdings: 100 + 10 + 20 = 130
    expected_hold = 130.0
    actual_hold = pop.state[0, pop.IDX_HOLDINGS]
    
    # 使用 np.add.at 应该能正确处理
    assert abs(actual_cash - expected_cash) < 0.01, f"Cash Sync Failed. Exp: {expected_cash}, Got: {actual_cash}"
    assert abs(actual_hold - expected_hold) < 0.01, f"Holdings Sync Failed. Exp: {expected_hold}, Got: {actual_hold}"

if __name__ == "__main__":
    pytest.main([__file__])
