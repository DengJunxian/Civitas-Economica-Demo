# LLM Provider 与路由说明

## 1. 角色分工

比赛答辩口径：

- 慢思考：DeepSeek v4 pro，用于复杂政策理解、因果链解释、答辩材料生成。
- 快思考：DeepSeek v4 flash thinking / non-thinking，用于短文本抽取、agent belief update、低延迟订单意图。
- 兜底：智谱 GLM-4-flashx，用于 DeepSeek 失败后的在线 fallback，并兼容旧智谱功能。
- 离线兜底：deterministic stub，用于无 key、断网、CI 和现场兜底。

## 2. 路由链

`core/llm/router.py` 中的主要链路：

慢路由 `mode="slow"`：

1. `deepseek:deepseek-v4-pro:thinking=true`
2. `deepseek:deepseek-v4-flash:thinking=false`
3. `zhipu:glm-4-flashx`
4. `offline:deterministic_stub`

快路由 `mode="fast"`：

- 对 `belief_update`、`short_extraction`、`sentiment_tick`、`agent_micro_decision` 等低延迟任务，优先 DeepSeek v4 flash non-thinking。
- 其他快任务可先走 flash thinking，再回退 flash non-thinking，最后 GLM-4-flashx。

旧功能兼容：

- `ZHIPUAI_API_KEY` 和 `ZHIPU_API_KEY` 都可用。
- `core/model_router.py` 仍保留旧调度接口和本地 cache。

## 3. `.env` 配置

复制 `.env.example` 后本地填写，不提交 `.env`：

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
ZHIPUAI_API_KEY=your_zhipu_api_key_here
ZHIPU_API_KEY=your_zhipu_api_key_here
LLM_DEFAULT_PROVIDER=auto
LLM_TIMEOUT_SECONDS=20
LLM_MAX_RETRIES=2
CIVITAS_AGENT_LLM_CALL_TIMEOUT_SECONDS=9
CIVITAS_RANDOM_SEED=42
```

无 key 时不影响启动：

```bash
python -m streamlit run app.py --server.port 8501
pytest -q tests/test_llm_router.py tests/test_performance_smoke.py
```

在线答辩前可做一次严格 DeepSeek 计时：

```bash
python scripts/benchmark_deepseek_online.py --llm-agents 4
```

该脚本只读取本地 `.env` 或 shell 环境变量，不保存也不打印密钥；它会直接调用 DeepSeek，不启用 GLM 或 deterministic stub，以便测量“一轮完整、不回退”的真实耗时。

## 4. 不保存敏感输入

复现记录只保存：

- provider
- model
- fallback_chain
- latency_ms
- ok
- error_type
- timestamp

不会保存：

- 完整 API Key
- Authorization header
- prompt 原文
- completion 原文
- 用户输入全文
- 工具调用原始敏感参数

相关测试：

```bash
pytest -q tests/test_llm_router.py tests/test_reproducibility.py
```

## 5. 答辩解释建议

评委问“为什么不用一个模型直接生成结果”时，可以回答：

系统把 LLM 当作政策解释和智能体认知组件，而不是直接让模型编造市场结果。价格路径来自 agent 决策、订单撮合、trade tape 和 K 线聚合。LLM 路由只影响政策解释、信念更新和叙事生成，并且所有调用都有 provider/model/fallback_chain/latency 的可复现记录。
