# 比赛演示与答辩 Playbook

## 1. 演示前检查

```bash
python -m pip install -r requirements.txt
pytest -q tests/test_reproducibility.py tests/test_deterministic_replay.py tests/test_performance_smoke.py
python -c "import app; assert hasattr(app, 'main')"
python -m streamlit run app.py --server.port 8501
```

如果没有 API Key，可以直接演示。系统会进入 offline deterministic fallback。

## 2. `.env` 配置

`.env` 不提交 git。参考 `.env.example`：

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

赛前在线自检：

```bash
python setup.py build_ext --inplace
python scripts/benchmark_simulation.py --agents 6 --ticks 1
python scripts/benchmark_deepseek_online.py --llm-agents 4
```

第一条确保当前 Python/系统平台有匹配的 pybind11 扩展；第二条确认本地 C++ 撮合性能路径；第三条确认 DeepSeek 在线一轮仿真可在不回退的情况下完成。

## 3. 推荐演示路线

1. 总览首页：一句话定位为“政策风洞推演沙箱”。
2. 政策试验台：选择默认政策模板，运行推演。
3. 展示结构化政策包：说明政策文本如何变成可执行参数。
4. 展示 trade tape 聚合 K 线：说明不是手画曲线。
5. 展示真实上证指数与仿真路径：说明 benchmark 和历史验证。
6. 追加重大新闻或监管事件：说明动态事件能力。
7. 展示 scorecard：路径、事件窗、风险、微观结构、行为金融。
8. 展示监管优化：Pareto front 和推荐方案。
9. 展示复现信息面板：experiment_id、config_hash、snapshot hash、seed、parameter_set_id。
10. 一键导出答辩报告：DOCX/PDF/JSON/CSV/Parquet 至少三种可用，依赖缺失时仍保留 JSON/CSV。

## 4. 8 分钟答辩节奏

| 时间 | 内容 |
| --- | --- |
| 0:00-0:40 | 项目定位：政策风洞，不是普通 demo |
| 0:40-1:40 | 政策文本到结构化政策包 |
| 1:40-3:00 | 多智能体、persona prior、regime router、快慢 LLM |
| 3:00-4:20 | A 股 session-aware 撮合、trade tape、K 线 |
| 4:20-5:40 | 上证指数历史验证与 scorecard |
| 5:40-6:50 | 动态追加事件和监管优化 Pareto |
| 6:50-7:40 | 可复现与 CI：seed、hash、registry、fallback |
| 7:40-8:00 | 导出报告和总结亮点 |

## 5. 常见追问

问：为什么不是普通 demo？

答：普通 demo 往往直接生成图表。本项目把政策文本转成结构化冲击，由异质智能体生成订单，经 A 股会话撮合生成 trade tape，再聚合 K 线，并用上证指数 replay window 做 scorecard 验证。

问：大模型是不是在编造结果？

答：不是。LLM 负责政策结构化、认知解释和部分 agent 决策。市场结果来自订单、撮合、trade tape 和评估链路。LLM 调用只保存 provider/model/fallback_chain/latency，不保存敏感输入。

问：现场没网怎么办？

答：无 API Key 或网络失败时，LLM 走 deterministic stub，市场数据走 cached 或 synthetic fallback，内置场景仍可复现。

问：C++ 扩展不可用怎么办？

答：默认 Python `OrderBook` 可运行。C++ pybind11 扩展用于性能优化，缺失时给出清晰错误，CI 不依赖它；赛前应在当前机器执行 `python setup.py build_ext --inplace` 生成与 Python 版本匹配的扩展文件。

问：为什么用上证指数？

答：上证指数政策敏感、代表性强、历史资料充分，适合作为主 benchmark。其他指数可作为对照。
