# 数治观澜：基于大模型多智能体的金融政策风动推演沙箱

面向中国大学生计算机设计大赛人工智能应用赛道的项目作品。系统围绕“政策输入 -> 智能体决策 -> 市场撮合 -> 风险诊断 -> 材料导出”构建完整闭环，用于演示政策冲击、市场情绪传播、监管干预与历史对照回放等场景下的市场演化过程。

项目支持离线演示。评委在没有云端 API Key 的情况下，也可以直接加载内置场景完成展示、回放、分析与材料导出。

## 一、项目亮点

- 多智能体政策推演：把自然语言政策转成结构化冲击，驱动市场行为变化。
- 历史智能回测工作台：统一承载智能因子回测与历史智能体回放，既能做传统回测，也能做带成交轨迹的历史重放。
- 行为金融诊断：自动输出 CSAD、PGR/PLR、波动聚集、回撤分布等典型事实指标。
- 监管优化与 A/B 推演：对比不同政策/干预方案的效果差异，便于答辩展示。
- 比赛材料导出：可生成摘要、演示脚本、图表索引和报告文件。

## 二、评委快速理解路径

推荐首次演示顺序：

1. 进入 `系统说明`，用 1 分钟理解项目定位与页面结构。
2. 进入 `政策试验台`，直接运行默认模板展示主功能。
3. 进入 `历史智能回测`，切换查看智能因子回测与历史智能体回放，说明项目具备验证能力而非单纯演示界面。
4. 进入 `高级分析` 或 `真实性报告`，回答“为什么可信”“哪里像真、哪里不像真”。
5. 需要总结时，点击导出比赛材料。

## 三、硬件环境和操作系统

填写说明：除一般 PC 计算机外，本项目无额外专用硬件强制要求；如需本地大模型推理，建议增加 GPU 与存储空间。

| 项目 | 最低要求 | 推荐配置 | 说明 |
| --- | --- | --- | --- |
| CPU | 4 核 x86_64 | Intel i5 / Ryzen 5 及以上 | 满足演示、测试与图表渲染 |
| 内存 | 8 GB | 16 GB 及以上 | 全量测试与多页面切换更稳定 |
| 磁盘 | 5 GB 可用空间 | 10 GB 及以上 | 用于依赖安装、缓存与导出文件 |
| GPU | 非必需 | 8 GB+ 显存 | 仅在本地推理大模型时建议 |
| 操作系统 | Windows 10/11 64 位 | Windows 11 64 位 | 当前仓库实测环境为 Windows |
| 浏览器 | Edge / Chrome | 最新稳定版 | 用于访问 Streamlit 页面 |

重要说明：

- 仓库内包含已编译的 Windows + Python 3.14 扩展文件 `_civitas_lob.cp314-win_amd64.pyd`。
- 若评委机器使用 `Windows + Python 3.14.x`，通常无需重新编译撮合扩展。
- 若 Python 版本与该二进制不匹配，需要安装 Visual Studio Build Tools 后执行重新编译。

## 四、开发平台（含开源/第三方工具）

确保评委按本说明安装开发工具软件后能够打开工程文件。

| 类别 | 名称 | 推荐版本 | 用途 |
| --- | --- | --- | --- |
| 编程语言 | Python | 3.14.x | 主运行环境 |
| 包管理 | pip | 24+ | 安装依赖 |
| 虚拟环境 | venv | Python 内置 | 隔离依赖 |
| 前端框架 | Streamlit | 1.53.1（锁定环境） | Web 展示界面 |
| 可视化 | Plotly、Matplotlib | 见 `requirements-lock.txt` | 图表展示与导出 |
| 多智能体建模 | Mesa | 3.4.2（锁定环境） | 智能体建模与仿真 |
| 数据分析 | pandas、numpy、scipy、networkx | 见 `requirements-lock.txt` | 指标计算与图结构分析 |
| 金融数据 | AkShare、yfinance | 见 `requirements-lock.txt` | 指数/市场数据支持 |
| 文档导出 | python-docx、reportlab、kaleido | 见依赖文件 | 导出 Word / PDF / PNG |
| C++ 编译 | Visual Studio Build Tools 2022 | MSVC v143 | 仅在重编译扩展时需要 |
| 版本管理 | Git | 任意稳定版 | 获取与提交工程 |
| 开发工具 | VS Code / PyCharm | 任意稳定版 | 查看、编辑与调试源码 |
| 脚本环境 | PowerShell | Windows 自带 | 启动脚本与交付检查 |

### 本地部署时所需 requirements

基础依赖定义在 `requirements.txt` 中：

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

比赛复现实测环境锁定在 `requirements-lock.txt` 中，推荐评委优先使用该文件安装，以减少环境差异。

## 五、运行环境和安装说明

确保评委按本说明安装作品后能够正常运行作品。

### 1. 目录位置

```powershell
cd C:\Users\Deng Junxian\Desktop\Civitas_new
```

### 2. 创建并激活虚拟环境

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. 安装依赖

常规安装：

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

比赛复现推荐安装：

```powershell
pip install -r requirements-lock.txt
```

### 4. 运行时可选配置

项目支持离线演示。以下环境变量均为可选，仅在接入在线模型时需要：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek Key"
$env:ZHIPU_API_KEY="你的 智谱 Key"
$env:CIVITAS_INFERENCE_MODE="lite"
$env:CIVITAS_LOCAL_MODEL_PATH="D:\models\xxx.gguf"
$env:CIVITAS_VLLM_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
```

模板可参考 `.env.example`。

### 5. 启动方式

推荐入口：

```powershell
python -m streamlit run app.py --server.port 8501
```

快捷脚本：

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

- `app.py` 是比赛展示推荐入口。
- `main.py` 更适合开发调试，不建议作为评委首选入口。
- 离线演示模式不依赖 API Key。

### 6. 浏览器访问

启动成功后访问：

- [http://127.0.0.1:8501](http://127.0.0.1:8501)

### 7. C++ 扩展重编译说明

以下情况下需要重新编译：

- Python 版本与 `_civitas_lob.cp314-win_amd64.pyd` 不匹配。
- 修改了 `core/exchange/c_core/*`。
- 在其他操作系统/解释器环境中重新部署。

重新编译命令：

```powershell
python setup.py build_ext --inplace
```

如需编译，请先安装 Visual Studio Build Tools 2022（含 MSVC v143）。

## 六、项目目录说明

```text
app.py                         Streamlit 比赛前端入口
main.py                        备用启动入口
agents/                        多智能体角色、认知与报告模块
core/                          核心仿真、回测、市场、推理、监管与系统运行逻辑
engine/                        市场仿真循环与撮合调度
policy/                        政策结构化解析与政策引擎
ui/                            页面级组件与图表界面
demo_scenarios/                比赛内置场景
data/                          策略模板与样例数据
docs/                          使用说明、接口说明、答辩辅助材料
scripts/                       启动、检查、实验脚本
theme/                         界面主题样式
requirements.txt              常规依赖
requirements-lock.txt         锁定依赖
```

## 七、推荐演示场景

无需 API Key，推荐优先使用以下内置场景：

- `tax_cut_liquidity_boost`
- `rumor_panic_selloff`
- `regulator_stabilization_intervention`

推荐演示流程：

1. 启动 `app.py`。
2. 在 `系统说明` 中快速说明项目定位。
3. 切换到 `政策试验台` 运行默认模板。
4. 切换到 `历史回放` 说明验证能力。
5. 切换到 `高级分析` 或 `真实性报告` 展示证据链和行为金融指标。
6. 导出比赛材料或报告。

## 八、已完成的本地实机验证

以下命令已在当前仓库完成验证：

```powershell
python -m pip check
python -m compileall -q app.py main.py config.py simulation_ipc.py simulation_runner.py regulator_agent.py agents core engine policy ui tests
pytest -q
powershell -ExecutionPolicy Bypass -File scripts\check_competition_delivery.ps1
```

本次复核结果：

- `python -m pip check`：通过
- `python -m compileall ...`：通过
- `pytest -q`：`308 passed`
- `scripts/check_competition_delivery.ps1`：通过
- `python -m streamlit run app.py --server.headless true --server.port 8501`：启动成功，页面可访问

## 九、常见问题

### 1. 页面无法打开

- 检查 `8501` 端口是否被占用。
- 可改用：`scripts\start_competition_demo.ps1 -Port 8510`

### 2. `_civitas_lob` 扩展无法加载

- 优先切换到 Python `3.14.x`
- 或安装 Build Tools 后执行 `python setup.py build_ext --inplace`

### 3. 没有 API Key 是否还能演示

- 可以。项目支持离线演示模式。

### 4. 历史数据拉取失败

- 优先检查网络连接。
- 若只做比赛答辩，可直接使用内置场景完成展示。

## 十、提交建议

比赛提交时建议至少包含以下内容：

- 仓库源代码
- 本 README
- `requirements.txt` 与 `requirements-lock.txt`
- `.env.example`
- `scripts/start_competition_demo.ps1`
- `scripts/check_competition_delivery.ps1`
- 桌面输出的《设计说明书》《项目作品小结》《README（桌面版）》

## 十一、免责声明

本项目仅供教学、科研、竞赛展示与仿真分析使用，不构成任何投资建议。
