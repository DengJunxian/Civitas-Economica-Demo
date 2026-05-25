# 校准与历史验证说明

## 1. 校准目标

校准不是让模型完美预测每一天收盘价，而是让仿真在固定 replay window 下符合政策冲击后的方向、波动、成交和行为事实。比赛答辩时应强调：这是政策风洞，不是投资预测器。

## 2. 主 benchmark

主 benchmark 使用上证指数 `sh000001`：

- 政策敏感度强。
- 大盘代表性高。
- 历史数据和新闻事件容易解释。
- 适合展示真实指数与仿真路径对齐。

其他指数可作为场景对照，但主线建议保持上证指数。

## 3. 参数集与复现 ID

每个校准参数集必须保存：

- `parameter_set_id`
- `parameter_hash`
- `seed`
- `config_hash`
- `data_snapshot_hash`

实现路径：

- `core/calibration/parameter_store.py`
- `core/reproducibility.py`
- `core/experiment_registry.py`

示例：

```python
from core.calibration.parameter_store import ParameterStore

store = ParameterStore()
pset = store.from_mapping(
    "demo_default",
    {"risk_aversion": 0.7, "liquidity_sensitivity": 0.4},
    seed=42,
    config_hash="...",
    data_snapshot_hash="...",
)
store.save(pset)
```

## 4. 历史回测如何证明真实性

验证链路：

1. 固定 replay window。
2. 冻结数据 snapshot hash。
3. 冻结 config hash 和 random seed。
4. 用 persona prior 初始化异质智能体。
5. regime router 根据市场状态切换行为权重。
6. 至少保留 heuristic baseline 作为参照。
7. 输出 scorecard。

scorecard 至少包含：

- 上证指数路径误差：RMSE、MAE、相关性。
- 事件窗方向命中率。
- 风险指标：最大回撤、波动、VaR、CVaR、Sharpe、Sortino。
- 微观结构指标：spread、depth imbalance、latency、cancel-to-trade。
- 行为金融指标：CSAD、panic、disposition proxy、herding。

## 5. 样本外与固定窗口

答辩建议使用固定 replay window，因为现场最重要的是稳定复现。研究延展时可补样本外窗口：

- 固定窗：展示稳定结果和完整证据链。
- 样本外：展示泛化能力。
- 压力窗：展示极端新闻/政策冲击下的风险暴露。

## 6. 运行命令

```bash
python scripts/run_calibration_experiment.py --seed 42 --output-dir outputs/calibration_demo
pytest -q tests/test_calibration_pipeline.py tests/test_replay_calibration.py tests/test_reproducibility.py
```

