# Evaluation Suite 说明

## 1. 评估目标

统一 evaluation suite 的目标是从评委视角解释“仿真结果为什么可信”。评估不是只看收益率，而是同时看路径、事件、风险、微观结构和行为金融。

## 2. 指标体系

| 类别 | 指标 | 解释 |
| --- | --- | --- |
| 路径拟合 | RMSE、MAE、相关性 | 真实上证指数与仿真路径的距离 |
| 事件窗 | 方向命中率、窗口收益差 | 政策/新闻后方向是否一致 |
| 风险 | 最大回撤、波动、VaR、CVaR、Sharpe、Sortino | 市场稳定性和尾部风险 |
| 微观结构 | spread、depth imbalance、cancel-to-trade、latency | 撮合和订单簿真实性 |
| 行为金融 | CSAD、herding、panic、disposition proxy | 异质投资者和情绪传播 |
| 工程复现 | config hash、snapshot hash、seed、parameter_set_id | 是否可稳定复现 |

## 3. Trade Tape 到 K 线

评估链路要求：

1. agent 生成订单意图。
2. session-aware engine 撮合订单。
3. 生成 canonical trade tape。
4. `core/exchange/bar_builder.py` 从 trade tape 聚合 OHLCV。
5. 前端展示真实指数和仿真路径。
6. event marker layer 标注政策、新闻、监管动作和谣言/辟谣。

这能防止“直接画曲线”的普通 demo 质疑。

## 4. Scorecard

`core/eval/replay_scorecard.py` 和 `ui/components/scorecard_panel.py` 承担统一 scorecard 展示。前端应突出：

- 主 benchmark：上证指数。
- replay window。
- 路径误差。
- 事件窗命中。
- 风险和行为指标。
- 复现元数据。

## 5. 场景对比

场景对比用于回答“政策是否真的改变了市场路径”：

- 基线：不实施政策。
- 政策方案 A：标准强度。
- 政策方案 B：加强或风险压力版本。
- 监管干预：限速、信息澄清、流动性支持等候选动作。

输出应包含差分指标和 Pareto 位置，而不是只给单一路径。

## 6. 测试命令

```bash
pytest -q tests/test_eval_suite.py tests/test_trade_tape.py tests/test_trade_tape_bar_builder.py
pytest -q tests/test_deterministic_replay.py tests/test_performance_smoke.py
```

