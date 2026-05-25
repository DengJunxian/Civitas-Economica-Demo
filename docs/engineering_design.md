# 工程设计说明

## 1. 设计目标

第八批增量改造的工程目标是让比赛答辩现场稳定可复现：

- 无 API Key 时能跑 mock/offline tests。
- 网络不可用时使用 cached 或 synthetic 数据 fallback。
- C++ 扩展不可用时有 Python fallback 或清晰错误提示。
- Streamlit 前端 import 不依赖密钥。
- LLM provider 缺 key 不导致项目启动失败。
- 每次实验都能追溯 config、数据快照、参数集和报告编号。

## 2. 模块分层

| 层级 | 关键路径 | 职责 |
| --- | --- | --- |
| 前端展示 | `app.py`, `ui/*` | 政策实验、历史验证、监管优化、材料导出 |
| 政策解释 | `policy/*`, `core/experiment_events.py` | 自然语言政策到结构化政策包 |
| LLM 路由 | `core/llm/*`, `core/model_router.py` | DeepSeek/智谱/离线兜底 |
| 多智能体 | `agents/*`, `engine/*` | persona、regime、快慢智能体、订单意图 |
| 市场内核 | `core/exchange/*` | A 股会话、撮合、trade tape、K 线聚合 |
| 历史验证 | `core/eval/*`, `core/backtester.py`, `ui/history_replay.py` | replay、scorecard、真实路径对照 |
| 监管优化 | `core/optimization/*`, `ui/regulator_optimization.py` | 多目标搜索、Pareto、推荐方案 |
| 复现登记 | `core/reproducibility.py`, `core/experiment_registry.py` | seed、hash、registry、report experiment_id |

## 3. 统一复现协议

`core/reproducibility.py` 提供统一协议：

- `seed_everything(seed)`：设置 Python、numpy 和子进程继承的 `PYTHONHASHSEED`。
- `config_hash(config)`：对配置做稳定 JSON hash，并自动屏蔽 key/token/password。
- `dataset_snapshot_hash(dataset)`：对 DataFrame、文件或结构化对象生成快照 hash。
- `parameter_set_id(params)`：校准参数集 ID。
- `experiment_id(...)`：根据 module、config hash、snapshot hash、seed、parameter_set_id 生成稳定实验 ID。
- `llm_call_record_from_response(...)`：只保存 provider、model、fallback_chain、latency、ok，不保存完整密钥、prompt 或 completion。

`core/experiment_registry.py` 提供：

- `create_experiment_record`
- `append_experiment_record`
- `load_experiment_registry`
- `attach_experiment_to_report`

报告导出在 `ui/reporting.py` 中补齐 `experiment_id` 和 `report_experiment_id`。

## 4. CI 和离线容错

新增 `.github/workflows/ci.yml`，在无 API Key 环境下运行：

- `python -c "import app; assert hasattr(app, 'main')"`
- `pytest -q`

关键 fallback：

- 市场数据：`core/data/market_data_provider.py` 在 akshare/yfinance/Ashare 不可用时生成 deterministic synthetic OHLCV，provider 标为 `synthetic`。
- LLM：`core/llm/router.py` 缺 key 或 provider 失败时返回 offline deterministic stub。
- C++ 扩展：`core/exchange/order_book_cpp.py` 给出清晰 ImportError；默认 Python `OrderBook` 可继续运行。`setup.py` 在 Windows 使用 MSVC `/std:c++17`，在 macOS/Linux 使用 `-std=c++17`，避免平台编译参数不匹配。
- 报告导出：DOCX/PDF 依赖缺失时返回 capability 和 disabled reason，JSON/CSV 仍可用。

## 5. 性能基准

`scripts/benchmark_simulation.py` 是轻量 benchmark：

- agent_count
- tick_latency_ms
- decisions_per_second
- llm_calls_per_tick
- cache_hit_rate
- fast_agent_count
- slow_agent_count
- batch_scenario_throughput

测试入口：

```bash
pytest -q tests/test_performance_smoke.py tests/perf/test_sim_benchmark.py
```

该 benchmark 不设硬阈值，只用于趋势观察，避免不同评委机器上因 CPU 差异误报失败。

在线 DeepSeek 计时使用：

```bash
python scripts/benchmark_deepseek_online.py --llm-agents 4
```

该脚本用于赛前测量“一轮完整、不回退”的云端推理耗时，只读取本地 `.env` 或 shell 环境变量，不持久化密钥和 prompt。

## 6. 外部体系借鉴说明

工程注释和文档明确采用“借鉴抽象，不绑定依赖”的策略：

- ABIDES：exchange kernel、agent latency、message passing、market replay。
- Mesa：ABM 兼容层定位。
- PettingZoo：标准化 observation/action/reward 的接口思想。
- Ray RLlib：大规模训练思路，保持可选。
- FinRL：数据、环境、agent、评估流程。
- JAX-LOB：高性能订单簿思路，当前不强制迁移 JAX。
- Optuna/BoTorch/Ax/Nevergrad：黑箱和多目标优化思想，优先轻量实现。

## 7. 安全与日志

- 源码不硬编码 API Key。
- `.env` 已在 `.gitignore` 中。
- `.env.example` 只放占位符。
- LLM 日志通过 `redact_sensitive` 或复现模块本地 redaction 屏蔽 Authorization/Bearer/API key。
- LLM call records 不保存 prompt、completion、headers 或工具原始输入。
