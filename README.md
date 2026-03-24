# Civitas Economica：面向计算机设计大赛人工智能应用赛道的多智能体金融政策风洞沙盘

## 项目简介
Civitas Economica 是一个将大语言模型、多智能体仿真、金融市场微观结构、政策推演与监管优化结合起来的人工智能应用项目。系统围绕“政策输入 -> 智能体决策 -> 市场撮合 -> 风险评估 -> 可视化解释”构建完整闭环，用于演示宏观政策、谣言冲击、监管干预等情景下的市场行为演化。

项目当前提供本地离线演示模式，评委无需配置云端 API Key 即可启动内置场景并完成完整展示；若需要联网扩展，也支持 DeepSeek 与智谱模型接口。

## 已完成的本地实机验证
2026-03-24 在当前仓库环境完成如下验证：

- `python -m pip check`：通过
- `python -m compileall -q app.py main.py config.py simulation_ipc.py simulation_runner.py regulator_agent.py agents core engine policy ui tests`：通过
- `pytest -q`：`279 passed`，总耗时约 `6 分 35 秒`
- `python -m streamlit run app.py --server.headless true --server.port 8501`：本地启动成功，返回 `HTTP 200`
- `powershell -ExecutionPolicy Bypass -File scripts\check_competition_delivery.ps1`：通过

## 主要功能
- 多智能体政策推演：输入政策或载入预设场景，观察价格、成交量、恐慌度、羊群效应等指标变化
- 历史回放与对照：支持因子回测与 Agent 视角重放，便于展示“仿真世界”和“历史世界”的差异
- 行为金融诊断：输出市场情绪、CSAD、风险事件链等行为金融指标
- 监管优化：基于强化学习与 A/B 反事实评估，搜索更优监管动作组合
- 复现实验能力：提供事件仓库、实验清单、校准管线和基准测试模块
- 比赛材料导出：可生成答辩摘要、设计提纲、演示脚本与图表索引

## 1. 硬件环境和操作系统
说明：除一般 PC 计算机外，无额外专用硬件强制要求；若启用本地大模型推理，则建议增加 GPU 与存储空间。

| 项目 | 最低要求 | 推荐配置 | 说明 |
| --- | --- | --- | --- |
| CPU | x86_64 4 核 | Intel i5 / Ryzen 5 及以上 | 一般演示与测试即可运行 |
| 内存 | 8 GB | 16 GB 及以上 | 全量测试与多页面切换更稳定 |
| 磁盘 | 5 GB 可用空间 | 10 GB 及以上 | 依赖安装、输出文件、缓存 |
| GPU | 非必须 | 8 GB+ 显存 | 仅在本地模型推理时建议 |
| 操作系统 | Windows 10/11 64 位 | Windows 11 64 位 | 当前仓库已实测 Windows 路径 |
| 浏览器 | Edge / Chrome | 最新稳定版 | 用于访问 Streamlit 页面 |

重要说明：

- 仓库内已包含 Windows + Python 3.14 对应的预编译二进制：`_civitas_lob.cp314-win_amd64.pyd`
- 若评委机器使用 `Windows + Python 3.14.x`，通常无需重新编译 C++ 撮合扩展
- 若 Python 版本与该二进制不匹配，则需要安装 Visual Studio Build Tools 后执行重编译

## 2. 开发平台（含开源/第三方工具）
说明：以下工具安装完成后，评委即可打开工程并按说明运行。

### 2.1 核心开发工具

| 类别 | 名称 | 版本要求 | 用途 |
| --- | --- | --- | --- |
| 编程语言 | Python | 推荐 `3.14.x`；兼容 `3.11+` | 项目主运行环境 |
| 包管理 | pip | `24+`，实测 `25.3` | 安装依赖 |
| 虚拟环境 | `venv` | Python 内置 | 隔离运行环境 |
| 版本管理 | Git | 任意稳定版本 | 获取/管理项目源码 |
| 代码编辑器 | VS Code / PyCharm | 任意稳定版本 | 打开与编辑工程 |
| C++ 编译工具 | Visual Studio Build Tools 2022 | 含 MSVC v143 | 仅在需要重编译 `_civitas_lob` 时使用 |

### 2.2 本地部署所必须的 Python requirements
以下依赖为项目当前 `requirements.txt` 的完整内容，本地部署时必须安装：

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

### 2.3 依赖用途说明

| 依赖类别 | 主要包 | 功能说明 |
| --- | --- | --- |
| Web 与界面 | `streamlit`, `plotly` | 构建比赛展示页面、交互控件与图表 |
| 多智能体/仿真 | `mesa` | 支撑 Agent 仿真框架 |
| 数据分析 | `pandas`, `numpy`, `scipy`, `matplotlib`, `networkx` | 指标分析、统计计算、图表与网络结构分析 |
| 模型与接口 | `openai`, `httpx`, `tenacity` | 调用外部模型接口与重试控制 |
| 金融数据 | `akshare`, `yfinance` | 外部市场数据支持 |
| 并发通信 | `pyzmq`, `nest_asyncio` | IPC 与异步兼容 |
| C++ 扩展 | `pybind11`, `setuptools` | 撮合引擎绑定与编译 |
| 测试 | `pytest`, `pytest-asyncio` | 单元测试与集成测试 |
| 文档导出 | `kaleido`, `python-docx`, `reportlab` | PNG、Word、PDF 等比赛材料导出 |
| 配置 | `pyyaml` | 场景包与配置文件解析 |

## 3. 运行环境和安装说明（包括运行时需要作的配置）
说明：以下步骤按照“评委机器最小可复现路径”编写。

### 3.1 推荐运行环境
- 单机本地部署，属于集中式架构
- 默认 Web 端口：`8501`
- 实时 IPC 端口：`5555`（PUB）、`5556`（PULL）
- 推荐浏览器：Edge 或 Chrome
- 离线演示无需联网；联网模式仅在调用外部模型 API 时需要访问公网

### 3.2 安装步骤

```powershell
# 1) 进入项目根目录
cd C:\Users\Deng Junxian\Desktop\Civitas_new

# 2) 创建虚拟环境
python -m venv venv

# 3) 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 4) 升级 pip
python -m pip install --upgrade pip

# 5) 安装 requirements
pip install -r requirements.txt
```

若希望尽量与已验证环境保持一致，可使用锁定版本：

```powershell
pip install -r requirements-lock.txt
```

### 3.3 环境变量配置
项目支持离线演示，因此下列配置均为可选项：

```powershell
# 可选：云端模型 API
$env:DEEPSEEK_API_KEY="你的 DeepSeek Key"
$env:ZHIPU_API_KEY="你的 智谱 Key"

# 可选：推理模式
$env:CIVITAS_INFERENCE_MODE="lite"

# 可选：本地模型路径与名称
$env:CIVITAS_LOCAL_MODEL_PATH="D:\models\xxx.gguf"
$env:CIVITAS_VLLM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

模板文件位于：`.env.example`

### 3.4 启动方式

#### 方式 A：比赛推荐启动方式
```powershell
python -m streamlit run app.py --server.port 8501
```

#### 方式 B：一键脚本启动
```powershell
scripts\start_competition_demo.bat
```

或：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_competition_demo.ps1
```

#### 方式 C：备用入口
```powershell
python main.py
```

说明：
- `app.py` 是比赛展示推荐入口
- `main.py` 是兼容入口，适合开发调试，不建议作为评委首选入口
- 离线演示模式不依赖 API Key

### 3.5 浏览器访问
启动成功后访问：

- [http://127.0.0.1:8501](http://127.0.0.1:8501)

### 3.6 预编译 C++ 撮合扩展说明
若满足以下条件，通常无需重编译：

- 操作系统为 Windows 64 位
- Python 版本为 `3.14.x`
- 仓库中的 `_civitas_lob.cp314-win_amd64.pyd` 可正常加载

仅在以下场景需要重编译：
- Python 版本与 `.pyd` 不匹配
- 修改了 `core/exchange/c_core/*`
- 需要在其他环境重新生成本地扩展

重编译命令：

```powershell
python setup.py build_ext --inplace
```

### 3.7 比赛前建议执行的检查命令

```powershell
python -m pip check
python -m compileall -q app.py main.py config.py simulation_ipc.py simulation_runner.py regulator_agent.py agents core engine policy ui tests
pytest -q
powershell -ExecutionPolicy Bypass -File scripts\check_competition_delivery.ps1
```

### 3.8 最小演示路径
无需 API Key，建议使用以下 3 个场景进行展示：

- `tax_cut_liquidity_boost`
- `rumor_panic_selloff`
- `regulator_stabilization_intervention`

推荐流程：
1. 启动 `app.py`
2. 载入内置场景
3. 展示价格、成交量、恐慌度、CSAD 等指标
4. 展示行为诊断与监管优化结果
5. 导出比赛材料或演示脚本

## 项目目录说明
- `app.py`：Streamlit 前端入口
- `main.py`：备用启动入口
- `agents/`：多智能体角色、脑模型、人格与管理器
- `core/`：核心业务模块、事件仓库、实验清单、校准与评估模块
- `engine/`：仿真调度与市场循环逻辑
- `ui/`：前端页面组件
- `policy/`：政策引擎
- `demo_scenarios/`：比赛内置场景包
- `data/`：本地数据、缓存、图结构与快照
- `outputs/`：运行时输出与比赛材料
- `docs/`：部署文档、用户手册、答辩脚本等
- `tests/`：测试代码
- `scripts/`：启动脚本与赛前检查脚本

## 常见问题
### 1. 页面打不开
- 检查 `8501` 端口是否被占用
- 可改用：`scripts\start_competition_demo.ps1 -Port 8510`

### 2. 缺少 `zmq` 或异步测试报错
- 重新执行：`pip install -r requirements.txt`

### 3. Python 版本不匹配导致 `_civitas_lob` 无法加载
- 优先切换到 `Python 3.14.x`
- 或安装 Build Tools 后执行 `python setup.py build_ext --inplace`

### 4. 没有 API Key 能否演示
- 可以，项目支持离线演示模式

## 提交建议
比赛提交时建议至少包含以下文件：

- `README.md`
- `requirements.txt`
- `requirements-lock.txt`
- `.env.example`
- `app.py`
- `scripts/start_competition_demo.bat`
- `scripts/start_competition_demo.ps1`
- `scripts/check_competition_delivery.ps1`
- `桌面\\设计说明书.md`
- `桌面\\项目作品小结.md`
- 其他 `docs/` 下的部署、答辩与说明文档
