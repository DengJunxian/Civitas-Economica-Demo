# 动态政策实验台架构升级说明

## 启动动态政策实验台

1. 启动 Streamlit 主应用后进入“政策实验室 / Policy Lab”。
2. 选择上证指数 `sh000001` 作为主图基准，配置基础政策、强度、半衰期和总交易日。
3. 点击开始后，可使用暂停、恢复、单步推进或运行到结束。

程序化入口仍保留：

```python
from core.policy_session import PolicySession

session = PolicySession.create(agents=[], total_days=20, base_policy="下调印花税并释放流动性")
session.advance(1)
session.append_news_event("盘中出现重大新闻冲击", effective_day=2, strength=1.2)
session.advance(1)
```

## 运行中追加重大新闻/新政策

`PolicySession` 保留 `enqueue_policy` / `append_policy`，并新增通用运行时事件：

- `append_event(...)`
- `append_news_event(...)`
- `append_macro_event(...)`
- `append_regime_shift(...)`
- `append_rumor_event(...)`
- `append_refute_event(...)`
- `append_regulatory_action(...)`

事件会进入 `ScenarioEventQueue`，写入 `EventStore`，并在 `advance()` 时生成 `EventDigest`。Digest 会进入宏观层、社会情绪层、agent belief、市场层和报告 payload。

## 历史回放严格/展示双模式

历史回放主流程仍然使用 `HistoryNewsService` 与 `NewsDrivenPolicyReplayEngine`：

- 严格模式：设置 `feature_flags={"strict_history_replay": True}` 或 `auth_score_mode="strict"`，不进行隐藏展示校准。
- 答辩展示模式：默认 `demo_first`，保留展示友好的价格显示，但 raw 指标独立保存在 `metadata["raw_simulated_prices"]` 与 bar 的 `raw_close`。

环境变量：

- `CIVITAS_STRICT_HISTORY_REPLAY=1`
- `CIVITAS_HISTORY_REPLAY_DEMO_MODE=1`

## 查看 discovered metrics

目标发现入口：

```python
from core.objective_discovery import discover_objectives

payload = discover_objectives(path_frame, reports=[latest_step_report])
```

Policy Lab 和监管优化页会输出同一 schema：

- `ranked_metrics`
- `top_metrics`
- `pareto_frontier`
- `composite_score`
- `weight_decomposition`
- `stability_heatmap`
- `shanghai_index_metric`

上证指数收益 `shanghai_index_return` 保留在候选池和主图中，但综合评分会同时考虑微观结构、行为金融、宏观融资和生态稳定指标。

## Feature Flags

- `CIVITAS_VECTOR_FAST_AGENTS`：预留向量化 fast agents 接口。
- `CIVITAS_BATCHED_SLOW_AGENTS`：预留 batched slow inference 接口，默认 CPU fallback。
- `CIVITAS_MULTI_SYMBOL_V1`：预留多 symbol / 多 book 扩展接口。
- `CIVITAS_STRICT_HISTORY_REPLAY`：历史回放严格评测。
- `CIVITAS_HISTORY_REPLAY_DEMO_MODE`：历史回放答辩展示模式。

## 迁移说明

保留的旧逻辑：

- `MarketEnvironment` 主循环 stage order。
- `SimulationRunner` 独立撮合子进程路径。
- `PolicyInterpretationEngine` 和 `AgentBelief`。
- `PolicySession` 追加政策 API。
- `HistoryNewsService` / `EventStore` / `NewsDrivenPolicyReplayEngine` 历史回放链路。
- 上证指数 K 线主图。

显式标注为 demo-only 的旧逻辑：

- `NewsDrivenPolicyReplayEngine` 中向真实价格拉回的展示校准现在仅在 demo mode 使用，并写入 `raw_vs_display`。
- `ui/history_replay.py` 的展示评分与严格评分分层显示，不再覆盖 raw metrics。

raw vs display 拆分：

- 严格评测读 `raw_simulated_prices` / `raw_close`。
- 答辩展示读 `display_simulated_prices` / `display_close`。
- UI 报告同时展示 `history_replay_mode`、`strict_mode`、`demo_mode` 和 raw/display adjustment。

## 性能基准

本地基准脚本：

```bash
python scripts/benchmark_simulation.py --agents 32 --ticks 12
```

输出包括 tick latency、decisions per second、LLM calls per tick、cache hit rate、fast/slow agent count 和 batch throughput。测试只检查趋势字段存在，不写死绝对性能阈值。

