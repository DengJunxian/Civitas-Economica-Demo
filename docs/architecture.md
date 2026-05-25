# 架构说明：政策风洞推演沙箱

## 1. 核心问题

Civitas Economica 不是把几个图表拼在一起的普通 demo。它要回答的是：当监管部门、财政部门或市场管理者发布一段政策文本后，不同投资者、机构、流动性提供者和舆情传播节点会如何反应，市场价格、成交、风险和行为金融指标会怎样演化，监管者能否在多目标约束下找到更优干预方案。

因此项目的核心对象不是单个聊天机器人，而是一个可回放、可校准、可导出的政策风洞：

1. 政策文本进入系统。
2. LLM 和规则解析器把文本转成结构化政策包。
3. 多智能体根据 persona prior、市场状态、新闻事件和政策冲击形成订单意图。
4. A 股 session-aware 撮合引擎生成 trade tape。
5. K 线、风险指标、行为金融指标和 scorecard 从 trade tape 聚合。
6. 历史 replay window 与真实上证指数路径对齐验证。
7. 监管优化器输出 Pareto 方案和推荐干预。
8. 实验 registry、config hash、snapshot hash、parameter_set_id 和 report experiment_id 支撑复现。

## 2. 政策文本如何变成结构化政策包

政策入口主要经过 `policy/structured.py`、`policy/interpretation_engine.py`、`core/experiment_events.py` 和 `ui/policy_lab.py`：

- 文本抽取：识别政策类型、作用目标、强度、半衰期、渠道和影响方向。
- 结构化政策包：包含 event、macro/social/market delta、scope、confidence、lag、decay 等字段。
- 运行时追加：在会话运行中可以追加重大新闻、新政策、监管事件、谣言和辟谣事件。
- 事件队列：`ScenarioEventQueue` 将事件按生效日和衰减曲线注入仿真。

这一步让自然语言政策不再只是展示文案，而是可执行的冲击向量。

## 3. AI 路由与多智能体

LLM 路由位于 `core/llm/router.py`：

- 慢思考：DeepSeek v4 pro，适合政策解释、复杂推理和答辩材料生成。
- 快思考：DeepSeek v4 flash thinking 或 non-thinking，适合高频 belief update、短文本抽取和微观决策。
- 兜底：智谱 GLM-4-flashx，兼容旧智谱功能。
- 离线兜底：无 API Key 或网络失败时返回 deterministic stub，保证前端和测试不崩。

多智能体异质性来自三层：

- Persona prior：风险偏好、资金规模、投资风格、羊群倾向、基准跟踪压力。
- Regime router：不同市场状态下切换风险偏好和决策权重。
- 两速智能体：慢智能体用于深度研判，快智能体用于低延迟订单/情绪更新。

## 4. A 股 Session-Aware Engine

项目需要 A 股 session-aware engine，因为政策影响在中国市场不是连续 24 小时交易，而是受到开盘集合竞价、上午盘、午休、下午盘、涨跌停、最小价格变动、T+1 等规则约束。

相关路径：

- `core/exchange/session_rules.py`
- `core/exchange/a_share_session.py`
- `core/exchange/market_rules.py`
- `core/exchange/market_kernel.py`
- `core/exchange/trade_tape.py`
- `core/exchange/bar_builder.py`

trade tape 是价格事实源。K 线必须由成交记录聚合，而不是直接手写价格曲线。这样评委追问“价格从哪来”时，可以落到订单、成交、撮合和聚合链路。

## 5. 为什么上证指数是主指标

上证指数 `sh000001` 被设为主 benchmark，是因为它具有政策敏感性、市场代表性和历史可获得性。项目也支持深证成指、创业板指等对照，但答辩主线优先使用上证指数，便于解释政策冲击对大盘风险偏好和市场稳定目标的影响。

验证维度包括：

- 上证指数路径误差：RMSE、MAE、相关性。
- 事件窗方向命中率：政策或新闻后窗口内涨跌方向是否一致。
- 风险指标：回撤、波动、VaR/CVaR、Sharpe、Sortino。
- 微观结构指标：spread、深度不平衡、撤单成交比、延迟。
- 行为金融指标：CSAD、处置效应、恐慌度、羊群强度。

## 6. 历史验证如何证明仿真真实性

历史验证不是证明模型能精确预测市场，而是证明它能在固定 replay window 下稳定复现合理的方向、波动和行为事实。

系统使用：

- cached/synthetic fallback 数据保障断网可运行。
- 固定 replay window 做样本外或固定窗测试。
- scorecard 聚合路径拟合、事件窗、风险、微观结构、行为金融指标。
- 真实指数路径与仿真路径同图展示，并叠加事件标记。

这使项目从“好看的演示”升级为“有验证闭环的政策沙箱”。

## 7. 借鉴的工程抽象

项目没有直接照搬外部项目，但借鉴了以下抽象：

- ABIDES / ABIDES-Gym：借鉴 agent latency、exchange kernel、message passing、market replay 思路，用在撮合、事件注入和 trade tape 回放。
- Mesa：借鉴 ABM 组织方式。`core/mesa/*` 是兼容层，不是项目唯一运行核心。
- PettingZoo：借鉴 observation/action/reward 标准化接口，方便多智能体环境扩展。
- Ray RLlib：借鉴大规模训练与 rollout 思路，但作为可选方向，不作为运行必需依赖。
- FinRL：借鉴数据、环境、agent、评估的金融 RL 工具链流程。
- JAX-LOB：借鉴高性能订单簿模拟思想。当前保留 C++/pybind11 路线，并提供 Python fallback 或清晰错误。
- Optuna / BoTorch / Ax / Nevergrad：借鉴黑箱优化和多目标参数搜索思想，优先轻量、可维护方案。

## 8. 复现链路

第八批改造新增：

- `core/reproducibility.py`：统一 random seed、config hash、dataset snapshot hash、parameter_set_id、LLM call metadata。
- `core/experiment_registry.py`：实验登记、registry JSONL、report experiment_id。
- `tests/test_reproducibility.py`
- `tests/test_deterministic_replay.py`
- `tests/test_performance_smoke.py`

每个实验应保留：

- `experiment_id`
- `config_hash`
- `data_snapshot_hash`
- `parameter_set_id`
- `random_seed`
- `git_commit`
- LLM provider/model/fallback_chain/latency
- report experiment_id

