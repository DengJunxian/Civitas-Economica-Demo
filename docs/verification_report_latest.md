# 一次性整改验证报告（latest）

生成日期：2026-05-25

## 通过项

- `pytest -q tests/test_reproducibility.py tests/test_deterministic_replay.py tests/test_performance_smoke.py tests/test_llm_router.py tests/test_market_data_provider.py tests/test_experiment_manifest.py tests/test_reporting_bundle_export.py`
  - 结果：`26 passed`
- `pytest -q`
  - 结果：`410 passed, 1 skipped in 412.30s (0:06:52)`
- `python -c "import app; assert hasattr(app, 'main')"`
  - 结果：通过（同等能力已纳入 `tests/test_reproducibility.py`）

## 故障注入验证

- `tests/test_performance_smoke.py::test_market_data_provider_uses_synthetic_fallback_when_network_unavailable`
  - 结果：通过，akshare/yfinance/Ashare 全失败时使用 provider=`synthetic`
- `tests/test_performance_smoke.py::test_llm_router_no_key_smoke_uses_mockable_offline_path`
  - 结果：通过，无 DeepSeek/智谱 key 时进入 offline deterministic stub
- `tests/test_reproducibility.py::test_streamlit_frontend_imports_without_api_keys`
  - 结果：通过，前端 import 不依赖 API Key

## 风险项

1. 当前环境未配置真实 `DEEPSEEK_API_KEY`/`ZHIPUAI_API_KEY`，本轮验证覆盖了路由顺序、fallback、无 key mock/offline 路径，但没有实际调用线上模型。
2. C++ `_civitas_lob` 扩展在当前环境不是全量测试必需项；Python `OrderBook` fallback 已覆盖，C++ parity 测试在扩展缺失时按 `pytest.importorskip` 跳过。
3. 全量测试通过，但 GitHub Actions 运行时间可能受依赖安装和机器性能影响，当前 workflow 设置为 20 分钟超时。

## 证据文件

- 复现模块：`core/reproducibility.py`
- 实验登记模块：`core/experiment_registry.py`
- 新增测试：`tests/test_reproducibility.py`、`tests/test_deterministic_replay.py`、`tests/test_performance_smoke.py`
- 架构说明：`docs/architecture.md`
- 工程设计说明：`docs/engineering_design.md`
- LLM 路由说明：`docs/llm_provider.md`
- 演示 Playbook：`docs/demo_playbook.md`
