# 一次性整改验证报告（latest）

生成日期：2026-04-05

## 通过项

- `venv\\Scripts\\python.exe -m compileall -q app.py core\\model_router.py core\\runtime_mode.py ui\\reporting.py ui\\policy_lab.py ui\\history_replay.py verify_config.py`
  - 结果：通过
- `venv\\Scripts\\python.exe -m pytest -q`
  - 结果：`334 passed`
- `powershell -ExecutionPolicy Bypass -File scripts\\check_competition_delivery.ps1 -FullTest -ReportPath outputs\\competition_materials\\latest_check.json`
  - 结果：通过（含 WARN，不阻塞）
- `venv\\Scripts\\python.exe verify_config.py`
  - 结果：通过，失败退出码机制生效
- `venv\\Scripts\\python.exe -m streamlit run app.py --server.headless true --server.port 8512`
  - 结果：烟测启动成功（进程存活后手动终止）

## 故障注入验证

- `powershell -ExecutionPolicy Bypass -File scripts\\check_competition_delivery.ps1 -Strict`
  - 结果：按预期失败（WARN 被视为 FAIL，exit code = 1）

## 风险项

1. 当前环境未配置 `DEEPSEEK_API_KEY`/`ZHIPU_API_KEY`，在线优先能力无法在本机完全实测。
2. `feedparser` 依赖已补齐，但上游库仍会产生 `DeprecationWarning`；不影响功能，可在后续版本统一做 warning 治理。

## 证据文件

- 机器可读验收报告：`outputs/competition_materials/latest_check.json`
- 赛道条款-证据矩阵：`docs/ai_track_clause_evidence_matrix.md`
- 评委视角审阅报告：`docs/reviewer_assessment_ai_track.md`
- 冗余代码三分法清单：`docs/redundancy_value_triage.md`
