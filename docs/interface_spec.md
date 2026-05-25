# 接口说明

## 1. 用户可见入口

- Web 入口：`app.py`
- 命令行备用入口：`main.py`
- 推荐启动脚本：
  - `scripts/start_competition_demo.ps1`
  - `scripts/start_competition_demo.bat`

推荐启动命令：

```powershell
python -m streamlit run app.py --server.port 8501
```

## 2. 页面交互接口

系统当前比赛版提供 4 个一级页面入口：

- `总览首页`
- `政策试验台`
- `历史回测`
- `高级分析`

它们分别承载：

- 项目定位与演示导航
- 政策输入与仿真执行
- 历史验证与拟真对照
- AI 证据链、行为诊断、监管优化与材料导出

## 3. 场景包接口

比赛场景目录位于 `demo_scenarios/<scenario_name>/`。

标准场景应包含以下文件：

- `initial_config.yaml`
- `analyst_manager_output.json`
- `narration.json`
- `metrics.csv`

`metrics.csv` 至少应包含以下字段：

- `step`
- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `csad`
- `panic_level`

## 4. 内部仿真通信接口

隔离撮合执行器通过文件系统 IPC 与主流程通信。

支持的命令类型包括：

- `submit_intent`
- `advance_time`
- `get_snapshot`
- `shutdown`

核心实现文件：

- `simulation_ipc.py`
- `simulation_runner.py`

## 5. 输出接口

比赛材料主输出目录：

- `outputs/competition_materials/`

运行过程中常见输出文件：

- `outputs/stylized_facts_report.json`
- `outputs/ecology_metrics.csv`
- `outputs/market_abuse_report.json`
- `outputs/intervention_effect_report.json`
- `outputs/history_reports/*`
- `outputs/policy_reports/*`

## 6. 导出能力接口

若运行环境已安装以下依赖，则支持正式文档导出：

- `python-docx`：用于 `docx` 导出
- `reportlab`：用于 `pdf` 导出
- `kaleido`：用于图表 PNG 导出

若缺失相关依赖，系统会保留主流程运行能力，但导出按钮可能部分禁用。
