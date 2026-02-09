# file: core/time_manager.py
"""
仿真时间管理器 (SimulationClock)

提供确定性的仿真时间，替代系统时间 (time.time())，确保仿真可复现。
"""

import time
from datetime import datetime, timedelta
from typing import Optional

class SimulationClock:
    """
    仿真时钟
    
    维护当前仿真时间 (Tick 和 DateTime)。
    """
    def __init__(self, start_time: Optional[datetime] = None):
        if start_time is None:
            # 默认从 2024-01-01 09:30:00 开始
            self._current_time = datetime(2024, 1, 1, 9, 30, 0)
        else:
            self._current_time = start_time
            
        self._tick_count = 0
        self._time_step = timedelta(seconds=3) # 默认每 Tick 3 秒 (类似 A 股快照)
        
    @property
    def now(self) -> datetime:
        """获取当前仿真时间 (datetime)"""
        return self._current_time
    
    @property
    def timestamp(self) -> float:
        """获取当前仿真时间戳 (float)"""
        return self._current_time.timestamp()
    
    @property
    def ticks(self) -> int:
        """获取当前 Tick 数"""
        return self._tick_count
        
    def tick(self, steps: int = 1) -> None:
        """推进仿真时间"""
        self._current_time += self._time_step * steps
        self._tick_count += steps
        
    def set_time(self, new_time: datetime) -> None:
        """强制设置时间 (用于跳跃或初始化)"""
        self._current_time = new_time
        
    def set_time_step(self, seconds: float) -> None:
        """设置时间步长"""
        self._time_step = timedelta(seconds=seconds)

# 全局单例 (可选，但推荐在 Model 中传递实例)
# _global_clock = SimulationClock()
