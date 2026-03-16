# Civitas Economica Demo

Civitas Economica 是一个多智能体社会经济与金融市场仿真沙箱，核心包含 C++ 限价撮合引擎（LOB）、IPC 调度机制与 Streamlit 可视化前端。

## 项目亮点

- `ManagerAgent + 专家分析师（News / Quant / Risk）` 的协同决策机制。
- LLM 仅输出意图，订单由 FCN（Fundamental-Chartist-Noise）映射层生成，便于可解释与风控。
- 社会网络传播与行为金融指标（如 CSAD、恐慌热度）联动。
- 支持 `DEMO_MODE` 答辩演示与 `LIVE_MODE` 联机运行。
- 内置比赛材料自动生成功能（摘要、提纲、演示脚本、图表索引）。

## 1. 硬件环境和操作系统

说明：一般 PC 即可运行；若启用本地大模型（非 API）会显著提高硬件要求。

- 必须（基础演示）
  - CPU：x86_64 4 核及以上（推荐 Intel i5 / Ryzen 5 及以上）
  - 内存：8 GB 及以上（推荐 16 GB）
  - 磁盘：至少 5 GB 可用空间（依赖安装 + 输出文件）
  - 操作系统：Windows 10/11 64 位（本项目当前已验证）
- 推荐（稳定演示）
  - 内存：16 GB+
  - 磁盘：10 GB+
  - 网络：可访问 Python 包源与外部 API（DeepSeek / 智谱）
- 可选（本地模型推理）
  - GPU：8 GB+ 显存可尝试轻量模型，24 GB+ 适合更完整本地推理
  - 额外磁盘：20 GB+（模型文件缓存）

> 注：仓库已包含 Windows 平台预编译扩展 `_civitas_lob.cp314-win_amd64.pyd`，与 Python 3.14 兼容。

## 2. 开发平台（含开源/第三方工具）

说明：评委按以下工具安装后可直接打开并运行工程。

### 2.1 必装工具

- Python：`3.11 ~ 3.14`（本地已验证 `3.14.2`）
- pip：`24+`（本地已验证 `26.0.1`）
- 虚拟环境：`venv`（Python 内置）
- Git：用于拉取或解压后版本管理（可选但推荐）

### 2.2 Windows 下 C++ 扩展相关（仅在需要重编译时）

- Visual Studio Build Tools 2022（MSVC v143，含 C++ 生成工具）
- 说明：若仅使用仓库内已提供的 `.pyd` 文件，可不重编译。

### 2.3 开源依赖（Python）

核心依赖来自 `requirements.txt`：

- 仿真与前端：`mesa`, `streamlit`, `plotly`
- 数据与科学计算：`pandas`, `numpy`, `scipy`, `matplotlib`
- AI / API：`openai`, `httpx`, `tenacity`
- 金融数据：`akshare`, `yfinance`
- 工程与测试：`pytest`, `nest_asyncio`, `pybind11`, `setuptools`, `pyyaml`

补充依赖（建议显式安装）：

- `pyzmq`：IPC 模块 `core/ipc/*` 运行需要
- `pytest-asyncio`：避免 `pytest.ini` 中 `asyncio_mode` 警告

安装命令（补充项）：

```bash
pip install pyzmq pytest-asyncio
```

### 2.4 第三方服务（按需）

- DeepSeek API（可选）：`DEEPSEEK_API_KEY`
- 智谱 API（可选）：`ZHIPU_API_KEY`

`DEMO_MODE` 可在无 API Key 情况下完成答辩演示。

## 3. 运行环境和安装说明（包括运行时需要作的配置）

说明：以下步骤可保证评委最小化复现。

### 3.1 安装步骤

```bash
# 1) 进入项目根目录
cd C:\Users\Deng Junxian\Desktop\Civitas_new

# 2) 创建并激活虚拟环境（Windows PowerShell）
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3) 安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) 建议安装补充依赖（IPC/测试）
pip install pyzmq pytest-asyncio
```

### 3.2 可选：重编译 C++ 撮合扩展

仅在以下情况需要：
- 本机 Python 版本与仓库内 `.pyd` 不匹配
- 需要自行修改 `core/exchange/c_core/*` 并重新编译

```bash
python setup.py build_ext --inplace
```

### 3.3 环境变量配置（按需）

PowerShell 示例：

```powershell
# 可选：云端 API
$env:DEEPSEEK_API_KEY="你的DeepSeekKey"
$env:ZHIPU_API_KEY="你的智谱Key"

# 可选：推理模式（lite / standard / enterprise）
$env:CIVITAS_INFERENCE_MODE="lite"

# 可选：本地模型路径与 vLLM 模型名
$env:CIVITAS_LOCAL_MODEL_PATH="D:\\models\\xxx.gguf"
$env:CIVITAS_VLLM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

### 3.4 启动方式

推荐启动（无交互、评委复现更稳定）：

```bash
python -m streamlit run app.py --server.port 8501
```

备用启动（会在无 Key 时要求命令行输入，非交互终端可能报 EOF）：

```bash
python main.py
```

说明：
- 当前版本已对 `main.py` 做非交互终端兜底：在无 TTY 环境下会自动跳过 API Key 输入，不会因 `input()` 阻塞。
- 比赛现场建议优先使用 `streamlit run app.py`，避免任何终端交互依赖。

浏览器访问：

- `http://127.0.0.1:8501`

### 3.5 端口与网络要求

- Streamlit 默认端口：`8501`
- IPC（实时模式）默认端口：`5555`（PUB）/ `5556`（PULL）
- 若调用外部模型 API，需要可访问公网

### 3.6 最小演示路径（无 API Key）

1. 启动 `streamlit run app.py`
2. 首页进入“答辩模式”
3. 选择场景：
   - `tax_cut_liquidity_boost`
   - `rumor_panic_selloff`
   - `regulator_stabilization_intervention`
4. 点击自动播放，完成 KPI 与叙事链路演示

### 3.7 演示前检查清单（建议答辩前 5 分钟执行）

```bash
python -m pip check
python -m compileall -q .
pytest -q tests/test_competition_demo_mode.py
python -m streamlit run app.py --server.port 8501
```

通过标准：
- 首页可正常打开（无白屏）
- 答辩模式中场景可一键加载
- 自动播放、下一步、重置按钮可用
- `tax_cut_liquidity_boost` / `rumor_panic_selloff` / `regulator_stabilization_intervention` 三场景可切换

## 测试命令

```bash
# 基础测试
pytest -q

# 示例：答辩模式相关测试
pytest -q tests/test_competition_demo_mode.py
```

## 项目结构

- `agents/`：多智能体实现（manager、analyst、trader 等）
- `core/`：核心引擎、行为金融、IPC、推理路由、回测与绩效模块
- `engine/`：仿真循环与进化逻辑
- `ui/`：Streamlit 页面与组件
- `data_flywheel/`：新闻抓取与事件流处理
- `demo_scenarios/`：答辩演示场景数据
- `tests/`：单元测试与集成测试
- `scripts/`：训练/敏感性分析脚本
- `outputs/`：自动生成的比赛材料与图表产物

## 常见问题

- 运行 `python main.py` 无法输入 API Key
  - 当前版本行为：非交互终端会自动跳过输入并继续运行离线可演示模式。
  - 建议：比赛环境仍优先使用 `python -m streamlit run app.py`。
- `ModuleNotFoundError: No module named 'zmq'`
  - 处理：`pip install pyzmq`
- 测试出现 `Unknown config option: asyncio_mode`
  - 处理：`pip install pytest-asyncio`
- 导出 PNG 按钮不可用
  - 原因：未安装 `kaleido`。
  - 处理：`pip install kaleido`（若当前平台不可安装，可继续使用 CSV/JSON 导出）。

## AI 赛道答辩支撑矩阵

已补充答辩材料映射文件：
- `docs/ai_competition_support_matrix.md`

该文件用于直接支撑：
- 设计说明书（问题定义、AI 技术路线、自主实现边界）
- 作品小结（创新点、工程完成度、可复现性）
- 用户手册（启动方式、演示主链路、常见故障）
- 展示视频脚本（建议镜头顺序与讲解重点）
