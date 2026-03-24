# Civitas Economica：数治观澜

面向中国大学生计算机设计大赛人工智能应用赛道的多智能体金融政策风洞沙盘。系统围绕“政策输入 -> 智能体决策 -> 市场撮合 -> 风险评估 -> 可视化解释 -> 材料导出”构建完整闭环，用于演示政策冲击、传言扰动、监管干预等情景下的市场演化过程。

项目支持本地离线演示。评委在没有云端 API Key 的情况下，也可以直接加载内置场景完成展示、回放、分析与导出。

## 作品简介

- 多智能体政策推演：输入政策或载入场景，观察价格、成交量、恐慌度、羊群度等指标变化。
- 历史回放与基准对照：支持历史因子回测和 Agent 级重放，对比“真实市场”和“仿真市场”。
- 行为金融诊断：输出 CSAD、风险热度、事件链和行为模式解释。
- 监管优化：针对市场稳定、流动性与成本目标进行动作搜索和对照评估。
- 比赛材料导出：可生成比赛摘要、演示脚本、图表索引与合规材料。

## 已完成的本地实机验证

以下验证于 `2026-03-24` 在当前仓库完成：

- `python -m pip check`：通过
- `python -m compileall -q app.py main.py config.py simulation_ipc.py simulation_runner.py regulator_agent.py agents core engine policy ui tests`：通过
- `pytest -q`：`295 passed`
- `python -m streamlit run app.py --server.headless true --server.port 8502`：启动成功，`HTTP 200`
- `powershell -ExecutionPolicy Bypass -File scripts\check_competition_delivery.ps1`：通过

## 1. 硬件环境和操作系统

说明：除一般 PC 计算机外，无额外专用硬件强制要求；如需本地大模型推理，建议增加 GPU 与存储空间。

| 项目 | 最低要求 | 推荐配置 | 说明 |
| --- | --- | --- | --- |
| CPU | 4 核 x86_64 | Intel i5 / Ryzen 5 及以上 | 一般演示与测试可运行 |
| 内存 | 8 GB | 16 GB 及以上 | 全量测试和多页面切换更稳定 |
| 磁盘 | 5 GB 可用空间 | 10 GB 及以上 | 用于依赖、缓存和输出文件 |
| GPU | 非必须 | 8 GB+ 显存 | 仅本地模型推理时建议 |
| 操作系统 | Windows 10/11 64 位 | Windows 11 64 位 | 当前仓库已实测 Windows |
| 浏览器 | Edge / Chrome | 最新稳定版 | 用于访问 Streamlit 页面 |

重要说明：

- 仓库已包含 Windows + Python 3.14 对应的预编译扩展：`_civitas_lob.cp314-win_amd64.pyd`
- 如果评委机器使用 `Windows + Python 3.14.x`，通常无需重新编译撮合扩展
- 如果 Python 版本与该二进制不匹配，需要安装 Visual Studio Build Tools 后执行重编译

## 2. 开发平台（含开源/第三方工具）

确保评委按本节安装工具后能够打开工程文件并完成运行。

| 类别 | 名称 | 版本要求 | 用途 |
| --- | --- | --- | --- |
| 编程语言 | Python | 推荐 `3.14.x`，兼容 `3.11+` | 项目主运行环境 |
| 包管理 | pip | `24+` | 安装依赖 |
| 虚拟环境 | `venv` | Python 内置 | 隔离运行环境 |
| 版本管理 | Git | 任意稳定版本 | 获取与管理源码 |
| 编辑器 | VS Code / PyCharm | 任意稳定版本 | 打开工程 |
| Web 前端 | Streamlit | 见 requirements | 比赛演示界面 |
| C++ 编译工具 | Visual Studio Build Tools 2022 | 含 MSVC v143 | 仅在重编译扩展时使用 |

### 2.1 本地部署必须安装的 requirements

以下为当前项目完整依赖清单：

```txt
mesa>=3.0.0
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
openai>=1.0.0
pyyaml>=6.0.0
matplotlib>=3.7.0
sortedcontainers>=2.4.0
networkx>=3.0.0
akshare>=1.18.0
yfinance>=0.2.40
httpx>=0.27.0
tenacity>=8.2.0
scipy>=1.11.0
pytest>=7.0.0
pytest-asyncio>=1.0.0
nest_asyncio>=1.5.0
pyzmq>=26.0.0
pybind11>=2.10.0
setuptools>=42.0.0
kaleido==0.2.1
python-docx>=1.1.0
reportlab>=4.0.0
```

### 2.2 第三方工具说明

- `Mesa`：多智能体建模与仿真
- `Streamlit`、`Plotly`：交互页面与图表展示
- `pandas`、`numpy`、`scipy`、`matplotlib`、`networkx`：数据分析与指标计算
- `openai`、`httpx`、`tenacity`：模型接口与重试控制
- `akshare`、`yfinance`：金融数据支持
- `pyzmq`、`nest_asyncio`：异步与 IPC 支持
- `pybind11`、`setuptools`：C++ 撮合引擎绑定与编译
- `python-docx`、`reportlab`、`kaleido`：比赛材料与报告导出

## 3. 运行环境和安装说明

说明：以下步骤按“评委机器最小可复现路径”编写，保证安装后可直接运行。

### 3.1 运行环境

- 单机本地部署，集中式运行
- 默认 Web 端口：`8501`
- IPC 端口：`5555`、`5556`
- 推荐浏览器：Edge 或 Chrome
- 离线演示无需联网；联网模式仅在调用外部模型 API 时需要访问公网

### 3.2 安装步骤

```powershell
cd C:\Users\Deng Junxian\Desktop\Civitas_new

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

如需尽量与已验证环境保持一致，可改用：

```powershell
pip install -r requirements-lock.txt
```

### 3.3 运行时配置

项目支持离线演示，下列环境变量均为可选：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek Key"
$env:ZHIPU_API_KEY="你的 智谱 Key"
$env:CIVITAS_INFERENCE_MODE="lite"
$env:CIVITAS_LOCAL_MODEL_PATH="D:\models\xxx.gguf"
$env:CIVITAS_VLLM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

示例模板见 `.env.example`。

### 3.4 启动方式

比赛推荐入口：

```powershell
python -m streamlit run app.py --server.port 8501
```

一键脚本启动：

```powershell
scripts\start_competition_demo.bat
```

或：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_competition_demo.ps1
```

备用入口：

```powershell
python main.py
```

说明：

- `app.py` 是比赛展示推荐入口
- `main.py` 适合开发调试，不建议作为评委首选入口
- 离线演示模式不依赖 API Key

### 3.5 浏览器访问

启动成功后访问：

- [http://127.0.0.1:8501](http://127.0.0.1:8501)

### 3.6 预编译扩展说明

通常无需重编译的条件：

- Windows 64 位
- Python `3.14.x`
- 仓库中的 `_civitas_lob.cp314-win_amd64.pyd` 可正常加载

以下情况需要重编译：

- Python 版本与 `.pyd` 不匹配
- 修改了 `core/exchange/c_core/*`
- 需要在其他环境重新生成本地扩展

重编译命令：

```powershell
python setup.py build_ext --inplace
```

### 3.7 建议赛前执行的检查命令

```powershell
python -m pip check
python -m compileall -q app.py main.py config.py simulation_ipc.py simulation_runner.py regulator_agent.py agents core engine policy ui tests
pytest -q
powershell -ExecutionPolicy Bypass -File scripts\check_competition_delivery.ps1
```

### 3.8 最小演示路径

无需 API Key，建议使用以下内置场景：

- `tax_cut_liquidity_boost`
- `rumor_panic_selloff`
- `regulator_stabilization_intervention`

推荐展示流程：

1. 启动 `app.py`
2. 载入内置场景
3. 展示价格、成交量、风险热度、CSAD 等指标
4. 展示行为诊断与监管优化结果
5. 导出比赛材料或演示脚本

## 系统结构

### 核心目录

- `app.py`：Streamlit 比赛前端入口
- `main.py`：兼容启动入口
- `agents/`：多智能体角色、认知、辩论与报告模块
- `core/`：核心业务逻辑、事件系统、回测、校准、仿真、数据处理
- `engine/`：市场循环与撮合调度
- `policy/`：政策引擎与结构化解析
- `ui/`：页面级功能组件
- `demo_scenarios/`：比赛内置场景
- `data/`：静态配置、模板与样例数据
- `docs/`：用户手册、接口说明、答辩辅助文档
- `tests/`：自动化测试
- `scripts/`：启动与检查脚本

### 技术架构

1. 前端展示层：基于 Streamlit 组织比赛演示流程、指标看板和图表交互。
2. 仿真与策略层：由多智能体、政策引擎、历史重放和监管优化模块组成。
3. 市场执行层：撮合引擎与微观结构模块负责订单簿、成交和价格演化。
4. 分析导出层：生成真实性报告、行为金融指标和比赛材料。

## 常见问题

### 页面无法打开

- 检查 `8501` 端口是否被占用
- 可执行：`scripts\start_competition_demo.ps1 -Port 8510`

### `_civitas_lob` 扩展无法加载

- 优先切换至 Python `3.14.x`
- 或安装 Build Tools 后执行 `python setup.py build_ext --inplace`

### 缺少异步或 ZMQ 相关依赖

- 重新执行 `pip install -r requirements.txt`

### 没有 API Key 是否能演示

- 可以，项目支持离线演示模式

## 交付建议

比赛提交时建议至少包含以下内容：

- 仓库源码
- 本 README
- `requirements.txt` 与 `requirements-lock.txt`
- `.env.example`
- `scripts/start_competition_demo.ps1`
- `scripts/check_competition_delivery.ps1`
- 桌面输出的《设计说明书》《项目作品小结》《README（桌面版）》

