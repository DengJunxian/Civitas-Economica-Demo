# 比赛材料信息底稿（增强版）

说明：本底稿基于当前项目的真实代码、配置、输出文件、测试结果与浏览器验收痕迹提取，不以 README 或目录名作唯一依据。凡仓库内无法确认但比赛材料必须填写的信息，统一标注为【待补充】；凡可给出建议写法但尚无直接证据定稿的信息，统一标注为【待确认】。

证据来源摘要：
- 核心入口与页面结构：[C:\Users\Deng Junxian\Desktop\Civitas_new\app.py](C:\Users\Deng%20Junxian\Desktop\Civitas_new\app.py)
- 答辩模式与演示链路：[C:\Users\Deng Junxian\Desktop\Civitas_new\ui\demo_wind_tunnel.py](C:\Users\Deng%20Junxian\Desktop\Civitas_new\ui\demo_wind_tunnel.py)
- 场景与 DEMO_MODE：[C:\Users\Deng Junxian\Desktop\Civitas_new\core\competition_demo.py](C:\Users\Deng%20Junxian\Desktop\Civitas_new\core\competition_demo.py)
- AI 路由与回退：[C:\Users\Deng Junxian\Desktop\Civitas_new\core\model_router.py](C:\Users\Deng%20Junxian\Desktop\Civitas_new\core\model_router.py)
- 行为金融诊断页：[C:\Users\Deng Junxian\Desktop\Civitas_new\ui\behavioral_diagnostics.py](C:\Users\Deng%20Junxian\Desktop\Civitas_new\ui\behavioral_diagnostics.py)
- 历史回测页：[C:\Users\Deng Junxian\Desktop\Civitas_new\ui\backtest_panel.py](C:\Users\Deng%20Junxian\Desktop\Civitas_new\ui\backtest_panel.py)
- 安装验证：`pip install -r requirements.txt` 成功，`pip install pyzmq pytest-asyncio` 成功
- 测试验证：`pytest -q tests/test_competition_demo_mode.py` 通过，共 5 项
- 浏览器验收痕迹：已通过 Streamlit 启动探活（HTTP 200）与答辩模式功能冒烟验证关键入口、KPI、A/B Compare、证据流与按钮状态。

---

## 一、作品情况表字段底稿

### 1. 参赛编号
- 参赛编号：【待补充】
- 缺失原因：仓库、配置、文档、输出文件中均无此信息。

### 2. 作品名称（3 个候选，偏竞赛风格）
- 候选 1：数治观澜：多智能体社会经济智能仿真与决策支持平台
- 候选 2：智策市镜：面向政策冲击分析的 AI 社会经济市场沙箱
- 候选 3：Civitas-Economica：基于多智能体与大模型的金融社会系统仿真平台
- 推荐名称：数治观澜：多智能体社会经济智能仿真与决策支持平台
- 推荐理由：兼顾中文竞赛表达、项目现有 `PROJECT_NAME` 中的“数治观澜”语义，以及“AI + 决策支持 + 仿真平台”定位。

### 3. 作品类型（最匹配项）
- 作品类型：软件应用 / Web 可视化智能决策系统
- 赛道匹配建议：人工智能与大数据应用
- 判断依据：系统由 Streamlit Web 前端、多智能体决策层、仿真引擎、行为金融诊断与数据报告输出构成，不是纯算法论文型作品，也不是纯硬件作品。

### 4. 作者信息
- 学校：【待补充】
- 姓名：【待补充】
- 身份证：【待补充】
- 院系：【待补充】
- 专业：【待补充】
- 年级：【待补充】
- 邮箱：【待补充】
- 电话：【待补充】
- 缺失原因：仓库内无成员实名信息，无法从代码、Git 状态或文档可靠推断。

### 5. 指导教师信息
- 姓名：【待补充】
- 院系：【待补充】
- 邮箱：【待补充】
- 电话：【待补充】
- 缺失原因：仓库内无指导教师实名信息。

### 6. 系统环境要求和安装说明

#### 6.1 硬件环境和操作系统
- 已证实环境：Windows 10/11 64 位可运行
- 已证实 Python 环境：`venv/pyvenv.cfg` 显示 Python `3.14.2`
- 已证实扩展环境：仓库包含 `_civitas_lob.cp314-win_amd64.pyd`，说明当前工程已适配 Windows + Python 3.14 的预编译扩展
- 建议写法：
  - CPU：4 核及以上
  - 内存：8 GB 及以上，推荐 16 GB
  - 磁盘：5 GB 及以上可用空间
  - 操作系统：Windows 10/11 64 位
- 说明：以上 CPU/内存为建议写法，其中操作系统与 Python 版本已有实际验证；CPU/内存下限为【待确认】的竞赛说明口径，不应写成已正式压测结论。

#### 6.2 开发平台（含开源/第三方工具）
- 编程语言：Python、C++
- 前端框架：Streamlit
- 可视化组件：Plotly
- 仿真框架：Mesa
- 数据处理：Pandas、NumPy、SciPy
- AI 接口相关：OpenAI SDK 兼容 DeepSeek / 智谱接口
- 金融数据工具：AkShare、yfinance
- 测试工具：pytest
- 构建相关：setuptools、pybind11
- 建议补装：pyzmq、pytest-asyncio
- 已证实依据：
  - `requirements.txt` 已安装成功
  - `setup.py` 存在 C++ 扩展构建定义
  - `core/ipc/*` 引用 `zmq`
  - `pytest.ini` 使用 `asyncio_mode`

#### 6.3 运行环境和安装说明
- 已证实推荐启动方式：
  - `python -m streamlit run app.py --server.port 8501`
- 已证实现象：
  - `python main.py` 在无交互终端会因请求输入 `DEEPSEEK_API_KEY` 导致 `EOFError`
- 已证实端口：
  - Streamlit：`8501`
  - IPC：`5555` / `5556`
- 已证实可选环境变量：
  - `DEEPSEEK_API_KEY`
  - `ZHIPU_API_KEY`
  - `CIVITAS_INFERENCE_MODE`
  - `CIVITAS_LOCAL_MODEL_PATH`
  - `CIVITAS_VLLM_MODEL`
- 已证实安装过程：
  - `pip install -r requirements.txt` 成功
  - `pip install pyzmq pytest-asyncio` 成功
- 建议写法：

```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyzmq pytest-asyncio
python -m streamlit run app.py --server.port 8501
```

- 运行时注意事项：
  - DEMO_MODE 可在无 API Key 情况下完成答辩演示
  - 若需完整 AI 在线推理，需提供 DeepSeek 或智谱 Key
  - 若需导出 PNG，当前需额外安装 `kaleido`
- 风险提示：
  - 当前“导出 PNG”按钮在浏览器验收中为禁用状态，原因是缺少 `kaleido`，不能写成“已支持完整 PNG 导出”

### 7. 超链接
- 展示视频地址：【待补充】
- Web 在线访问地址：【待补充】
- 缺失原因：仓库中未发现公网地址、部署说明或视频链接。

---

## 二、作品小结素材底稿

### 1. 作品名称
- 推荐使用：数治观澜：多智能体社会经济智能仿真与决策支持平台

### 2. 学生信息
- 学生信息：【待补充】

### 3. 指导教师
- 指导教师：【待补充】

### 4. 作品简介（约 250 字）
本作品面向人工智能与大数据应用赛道，构建了一个融合多智能体建模、大模型决策、市场微结构仿真与可视化答辩演示的社会经济智能仿真平台。系统以前端答辩界面为入口，支持“答辩模式、专家模式、历史回测、行为金融诊断、系统说明”五类功能入口；在核心链路上，以 Analyst、Manager、Market 三段式结构组织智能决策过程，将政策冲击、舆情事件与市场状态转化为可解释、可验证的结构化结果。系统底层采用 Python 控制层与 C++ 限价撮合引擎协同运行，并通过多模型路由、超时回退、本地缓存等机制提升现场稳定性。项目不仅能够完成智能仿真与交互展示，还能自动生成比赛材料和分析产物，适合用于政策推演、金融行为研究、智能决策演示与教学展示。

### 5. 作品效果图清单（建议 9 张）

1. 图名：系统首页总览
- 图注：系统首页展示五个核心入口，分别服务于主线答辩、专家追问、回测验证、行为金融分析和系统说明。
- 在哪里截：首页顶部五入口区域
- 截图位置：`app.py` 运行后首页；浏览器快照已见五入口

2. 图名：答辩模式一键载入
- 图注：答辩模式支持一键加载场景、自动推进时间线，适合比赛现场 5 分钟展示闭环。
- 在哪里截：答辩模式页顶部按钮区
- 截图位置：场景选择、一键载入、自动播放、暂停、下一步、重置

3. 图名：核心 KPI 实时展示
- 图注：系统实时展示流动性、波动率、羊群度、风险预警与监管动作等核心指标。
- 在哪里截：答辩模式 KPI 卡片区域
- 截图位置：KPI 五卡并排区域

4. 图名：三段式智能决策链
- 图注：系统按 Analyst、Manager、Market 三阶段展示智能分析、决策汇总与市场反馈。
- 在哪里截：答辩模式“三段式叙事”区域
- 截图位置：三个并列 JSON/结构化卡片

5. 图名：政策 A/B 世界对比
- 图注：通过 A/B World Compare 展示不同政策干预路径下的市场差异，体现反事实分析能力。
- 在哪里截：答辩模式 A/B Compare 区域
- 截图位置：市场对比图表

6. 图名：决策证据流
- 图注：系统将原始思维链转化为结构化证据流，体现 AI 决策的可解释性与合规表达。
- 在哪里截：答辩模式底部证据流页签
- 截图位置：Analyst Cards 或 Manager Final Card

7. 图名：专家模式图谱
- 图注：专家模式提供社会传播热图、LOB 深度、政策传导链和风险时间轴，用于应对评委追问。
- 在哪里截：专家模式页面中部
- 截图位置：【待截图，需现场启动后截取】

8. 图名：行为金融诊断页
- 图注：自动读取输出报告，展示 CSAD、PGR/PLR、波动聚集、回撤分布等行为金融指标。
- 在哪里截：行为金融诊断页面
- 截图位置：顶部指标卡与原始报告区域

9. 图名：比赛材料自动生成
- 图注：系统可自动生成比赛摘要、设计提纲、演示脚本与图表索引，便于竞赛材料整理。
- 在哪里截：侧边栏“自动生成比赛材料”触发后的结果区
- 截图位置：最近生成文件 JSON 列表

### 6. 设计思想

#### 6.1 项目背景
- 传统金融市场展示类作品通常偏静态可视化，难以体现“智能决策”
- 纯大模型应用又容易停留在问答层，缺少完整业务闭环
- 本项目希望解决“AI 决策如何进入真实仿真流程，并在比赛现场稳定演示”的问题

#### 6.2 设计构思与创意
- 以“政策/事件输入 -> 智能决策 -> 市场反馈 -> 可解释展示 -> 材料沉淀”为主线
- 用 Analyst、Manager、Market 三段式结构替代黑盒式 AI 展示
- 将 DEMO_MODE 设计成比赛友好模式，在无 API Key 情况下仍能跑通主线

#### 6.3 项目模块与功能
- 首页五入口：答辩模式、专家模式、历史回测、行为金融诊断、系统说明
- 答辩模式：一键场景、自动时间线、KPI、A/B Compare、证据流
- 专家模式：专家图谱与深入分析
- 历史回测：策略评估、风险图、报告导出、政策 A/B 自动比较
- 行为金融诊断：读取输出报告并做结构化展示

#### 6.4 技术运用与特色
- Streamlit 构建比赛友好的可视化交互界面
- Python 控制层组织多智能体仿真与数据流
- C++ LOB 引擎提升底层撮合性能
- ModelRouter 提供多模型路由、缓存和回退机制，提高现场稳定性
- 自动生成比赛材料，提升成果沉淀能力

### 7. 指导教师自评（约 250 字草稿）
该作品围绕人工智能与大数据应用的典型场景，完成了从智能分析、市场仿真、行为诊断到可视化展示与材料输出的完整系统设计，体现出较强的综合工程能力与应用创新意识。项目并未停留在简单调用大模型接口，而是结合多智能体角色分工、结构化证据表达、市场微结构建模与反事实对比分析，形成了较清晰的 AI 赋能业务闭环。系统前端设计充分考虑比赛现场展示需求，答辩模式、专家模式和材料自动生成功能能够较好支撑展示与答辩。底层同时引入 Python 与 C++ 协同实现，体现出团队在系统架构、算法组织、交互设计和工程实现方面的扎实能力。若后续进一步补强量化指标和展示材料统一性，作品具备较好的竞赛表现力和推广价值。

说明：此处仅为草稿，正式内容需由指导教师本人审核确认。

---

## 三、设计说明书章节素材底稿

### 1. 简介

#### 1.1 作品创意 / 项目背景
- 关键词：人工智能、多智能体、社会经济仿真、金融市场、行为金融、决策支持
- 已证实项目定位：多智能体社会经济与金融市场仿真沙箱，带有比赛展示前端
- 可用背景表述：
  - 面向复杂社会经济环境，传统市场分析系统往往只能给出单一结果，难以解释政策冲击、舆情传播与群体行为对市场的联动影响。
  - 本项目将大模型、多智能体建模与市场微结构仿真结合，构建一个可解释、可展示、可扩展的智能决策演示平台。

#### 1.2 项目实施计划
- 已证实现状：
  - 已具备答辩场景、页面、测试与输出链路
  - 已存在比赛材料生成模块
- 【待确认】建议写法：
  - 第一阶段：需求分析与总体架构设计
  - 第二阶段：多智能体与市场仿真核心开发
  - 第三阶段：答辩界面与可视化重构
  - 第四阶段：测试验证与竞赛材料整理
- 说明：仓库内没有完整时间计划表，只能作为建议模板，不应写成已被证实的精确排期。

### 2. 总体设计

#### 2.1 系统功能

##### 2.1.1 功能概述
- 系统提供面向比赛展示和研究验证的五大功能入口
- 系统完成从场景输入到市场反馈再到报告输出的完整链路
- 系统支持 DEMO_MODE 场景化展示与历史回测验证

##### 2.1.2 功能说明
- 答辩模式：一键加载场景、自动播放、三段式叙事、A/B 世界对比、证据流
- 专家模式：关键指标、决策证据、热图、LOB 深度、政策传导链、风险时间轴
- 历史回测：策略回测、风险收益展示、报告导出、Qlib 导出、政策 A/B 比较
- 行为金融诊断：读取 `stylized_facts_report.json`、`parameter_sensitivity.csv`、`intervention_effect_report.json`
- 系统说明：展示架构、重构重点与运行约束

#### 2.2 系统软硬件平台

##### 2.2.1 系统开发平台（含开源/第三方工具）
- Python 3.14.2
- pip 26.0.1
- Streamlit、Mesa、Plotly、Pandas、NumPy、SciPy
- OpenAI SDK 兼容 DeepSeek/智谱接口
- pybind11 + setuptools
- C++ 编译环境：Visual Studio Build Tools 2022【待确认是否在最终材料中写明版本号】

##### 2.2.2 系统运行平台
- Windows 10/11 64 位
- 运行入口：`streamlit run app.py`
- 默认端口：8501
- IPC 端口：5555/5556
- 可选外部 API：DeepSeek、智谱

#### 2.3 关键技术
- 多智能体协同决策
- 大模型多路由与回退机制
- C++ 限价撮合引擎
- IPC 进程间通信
- 行为金融指标建模
- 反事实 A/B 比较
- 比赛材料自动生成

#### 2.4 作品特色
- 不只是“展示图表”，而是“构建了一个可运行的智能仿真闭环”
- 不只是“调用大模型”，而是实现了结构化决策、缓存、回退和稳定运行
- 不只是“开发系统”，而是为比赛场景专门设计了答辩模式与专家模式

### 3. 详细设计

#### 3.1 系统结构设计

##### 3.1.1 技术架构
- 前端展示层：Streamlit 页面与比赛交互
- 业务控制层：Python 控制器、仿真调度、数据组织
- 智能决策层：Analyst / Manager / Risk 等角色与 ModelRouter
- 仿真执行层：IPC 与 C++ / Python 撮合引擎
- 输出沉淀层：JSON、CSV、HTML、比赛材料 Markdown

##### 3.1.2 功能模块设计
- 场景模块：`demo_scenarios/*`
- 页面模块：`ui/*`
- 仿真模块：`engine/*`、`simulation_runner.py`
- AI 模块：`core/model_router.py`、`core/inference/*`
- 诊断与报告模块：`ui/behavioral_diagnostics.py`、`core/tear_sheet.py`

##### 3.1.3 关键功能 / 算法设计
- 多模型调用策略：按优先级调用，超时或失败则切换模型，最终回退到 deterministic stub
- 场景回放策略：通过 `bootstrap_competition_demo` 和 `advance_competition_demo` 逐步推进
- 反事实比较策略：在演示中构造 B 世界，比较政策干预前后路径
- 回测比较策略：历史回测页支持生成政策 A/B 自动比较报告

#### 3.2 数据结构设计

##### 3.2.1 存储数据
- 场景配置：`demo_scenarios/*/initial_config.yaml`
- 结构化决策输出：`analyst_manager_output.json`
- 叙事数据：`narration.json`
- 市场指标：`metrics.csv`
- 模型缓存：`data/cache/model_router_cache.json`
- 诊断输出：`outputs/stylized_facts_report.json`
- A/B 干预输出：`outputs/intervention_effect_report.json`
- 敏感性输出：`outputs/parameter_sensitivity.csv`

##### 3.2.2 接口（模块接口、系统间接口）
- 页面与场景接口：
  - `list_competition_scenarios`
  - `load_competition_scenario`
  - `bootstrap_competition_demo`
  - `advance_competition_demo`
- AI 调用接口：
  - `call_with_fallback`
  - `call_with_schema`
  - `sync_call_with_fallback`
- IPC 接口：
  - `submit_intent`
  - `advance_time`
  - `get_snapshot`
  - `shutdown`

##### 3.2.3 关键数据结构
- `DemoScenario`
- `IPCEnvelope`
- `BufferedIntent`
- `ModelInfo`
- `ModelStats`
- `BacktestResult`

#### 3.3 系统界面设计

##### 3.3.1 界面设计风格
- 深色主题
- 强调 KPI 卡片、结构化信息卡和图表联动
- 更偏比赛展示型而不是后台表单型

##### 3.3.2 主要功能页面
- 首页五入口
- 答辩模式页
- 专家模式页
- 历史回测页
- 行为金融诊断页
- 系统说明页

##### 3.3.3 Web 网站页面结构设计
- 顶部：入口切换按钮
- 左侧：全局面板、默认场景、比赛材料生成
- 主区域：
  - 答辩模式：按钮区、KPI、叙事、A/B、图表、证据流
  - 专家模式：指标、证据流、专家图谱
  - 回测页：参数表单、指标图、下载按钮、A/B 比较
  - 诊断页：指标卡、原始报告、下载按钮

### 4. 系统安装及使用说明
- 见第一部分“系统环境要求和安装说明”
- 推荐现场演示启动命令：

```bash
python -m streamlit run app.py --server.port 8501
```

- 不建议把 `python main.py` 写为默认演示启动方式，因其在无交互终端下会请求输入 API Key 并报错
- 若需完整 IPC / 回测 / 测试环境，建议补装：

```bash
pip install pyzmq pytest-asyncio
```

- 若需 PNG 导出能力，建议补装：

```bash
pip install kaleido
```

### 5. 总结
- 作品的核心价值在于把 AI 决策、多智能体仿真、市场微结构和比赛答辩展示结合为统一系统
- 当前系统已形成可演示、可测试、可输出材料的闭环
- 后续若进一步增强量化指标和界面统一性，竞赛表现力会更强

### 6. 附录

#### 6.1 名词定义
- DEMO_MODE：面向比赛的离线演示模式
- LIVE_MODE：实时运行模式
- LOB：限价订单簿
- CSAD：衡量羊群行为的离散度指标
- PGR / PLR：处置效应相关指标
- A/B Compare：干预与非干预路径的对照比较

#### 6.2 参考资料
- DeepSeek / 智谱开放平台文档【待补充具体链接】
- Streamlit 官方文档【待补充具体链接】
- Mesa 官方文档【待补充具体链接】
- Plotly 官方文档【待补充具体链接】

#### 6.3 源代码清单
- 顶层入口：`app.py`、`main.py`
- 页面模块：`ui/`
- 智能体模块：`agents/`
- 核心模块：`core/`
- 仿真调度：`simulation_runner.py`、`simulation_ipc.py`
- 测试：`tests/`
- 场景：`demo_scenarios/`
- 输出：`outputs/`

---

## 四、答辩与 PPT 额外素材底稿

### 1. 项目一句话定位
- 这是一个面向政策冲击与市场行为分析的多智能体 AI 社会经济仿真与决策支持平台。

### 2. 项目三句话介绍版
- 本项目将大模型、多智能体和金融市场仿真结合，构建了一个可解释、可演示的社会经济智能沙箱。
- 系统能够把政策或舆情事件转化为智能分析、市场反馈与行为诊断结果，并通过 Web 界面完成闭环展示。
- 为适应竞赛现场环境，项目还设计了 DEMO_MODE、A/B Compare 和比赛材料自动生成机制。

### 3. 项目总览框架
- 输入层：政策场景、市场状态、新闻事件、历史数据
- 智能层：Analyst、Manager、Risk、ModelRouter
- 仿真层：订单意图、IPC 调度、LOB 撮合、指标生成
- 展示层：答辩模式、专家模式、历史回测、行为金融诊断
- 输出层：JSON/CSV/HTML 报告与比赛材料

### 4. 细分领域框架
- AI 决策：多模型路由、结构化输出、回退机制
- 社会经济仿真：多智能体交互、市场状态推进、政策冲击
- 金融市场机制：订单簿、成交、流动性、波动率
- 行为金融分析：CSAD、PGR/PLR、波动聚集、回撤分析
- 竞赛展示：五入口页面、自动演示、材料生成

### 5. 项目精华部分清单
- 答辩模式一键演示
- 三段式智能决策链
- A/B World Compare
- 决策证据流
- ModelRouter 的稳定性设计
- C++ LOB + Python 协同架构
- 比赛材料自动生成

### 6. 创新点候选
- 将 AI 决策、市场仿真和可解释展示打通为一个完整闭环
- 在比赛场景下实现无需 API Key 的稳定 DEMO_MODE
- 采用结构化证据流替代原始思维链展示
- 融合行为金融指标与反事实 A/B 比较
- 引入多模型路由、缓存与 deterministic stub 提升现场可用性

### 7. 作品特色候选
- 工程完整度高
- 展示链路清晰
- 具备可解释 AI 表达
- 具备研究与教学双场景适用性
- 可自动沉淀比赛文档与图表素材

### 8. AI 技术路径候选表达
- AI 用来解决“复杂市场环境下如何进行结构化分析与决策”的问题
- 输入包括：市场快照、新闻事件、持仓状态、情绪与风险信号
- 处理包括：模型路由、超时回退、本地缓存、结构化 JSON 约束
- 输出包括：动作建议、置信度、风险提示、证据卡片
- 输出再进入 Manager、市场仿真、页面展示与材料生成，不停留在聊天结果层
- 第三方提供的是基础模型能力；自研部分是路由、缓存、业务映射、可解释展示和比赛化封装

### 9. 可量化指标候选
- 已证实候选指标：
  - 场景数量：3 个正式答辩场景
  - 页面入口：5 个主要入口
  - 演示测试：`tests/test_competition_demo_mode.py` 5 项通过
  - 演示链路：支持自动推进时间线
  - 报告输出：存在 `stylized_facts_report.json`、`intervention_effect_report.json`、`parameter_sensitivity.csv`
- 【待确认】可进一步整理的指标：
  - 平均演示时长
  - 单场景步骤数
  - 图表数量
  - 回测策略数量
  - 材料生成文件数量

### 10. 答辩时可展示的默认流程
1. 首页介绍五入口
2. 进入答辩模式，一键载入场景
3. 展示 KPI 与自动讲解
4. 展示 Analyst -> Manager -> Market 三段式结构
5. 展示 A/B Compare
6. 展示决策证据流
7. 切换到专家模式或行为金融诊断页回答追问
8. 演示自动生成比赛材料

### 11. 适合 PPT 呈现的图示建议

#### 11.1 总体架构图
- 内容：展示层、智能层、仿真层、输出层四层结构
- 作用：开场建立整体认知

#### 11.2 功能框架图
- 内容：五入口对应功能与作用
- 作用：让评委快速理解系统不是单页演示

#### 11.3 技术路线图
- 内容：Streamlit、Python 控制层、ModelRouter、IPC、C++ LOB
- 作用：强调技术含量

#### 11.4 数据流图
- 内容：场景配置、叙事 JSON、指标 CSV、输出报告的流向
- 作用：支撑设计说明书中的数据结构章节

#### 11.5 模块关系图
- 内容：UI、agents、core、engine、outputs 之间关系
- 作用：突出系统化设计

#### 11.6 AI 处理链路图
- 内容：输入 -> 模型路由 -> 回退 -> 结构化输出 -> 进入业务流程
- 作用：突出“驾驭 AI 解决问题”

#### 11.7 演示流程图
- 内容：首页 -> 答辩模式 -> KPI -> 三段式叙事 -> A/B Compare -> 证据流 -> 材料生成
- 作用：适合现场答辩讲稿与 PPT 串讲

---

## 五、当前明确缺口汇总

- 作者实名信息：【待补充】
- 指导教师信息：【待补充】
- 参赛编号：【待补充】
- 展示视频地址：【待补充】
- Web 在线访问地址：【待补充】
- 项目实施计划中的精确时间线：【待确认】
- 参考资料中的正式链接列表：【待补充】
- 专家模式、行为金融诊断页、材料生成结果的最终比赛截图：【待补截图】

---

## 六、已证实结论清单（便于后续引用）

- 已证实系统是 `Streamlit + Python 控制层 + C++ LOB` 的多智能体仿真系统
- 已证实存在五入口比赛界面
- 已证实存在 3 个正式答辩场景
- 已证实存在答辩模式自动推进、A/B Compare、决策证据流、比赛材料自动生成
- 已证实 `requirements.txt` 可以安装成功
- 已证实 `tests/test_competition_demo_mode.py` 通过
- 已证实默认演示入口应为 `streamlit run app.py`
- 已证实当前 PNG 导出依赖 `kaleido`，缺失时按钮禁用
