# 数治观澜

基于大模型多智能体的金融政策风洞推演沙箱。项目用于演示政策输入、智能体决策、市场撮合、风险诊断、历史回放与材料导出等核心流程。

## 保留内容

本仓库已整理为答辩运行版，仅保留运行和演示所需内容：

- `app.py`：Streamlit 展示入口。
- `core/`、`agents/`、`engine/`、`policy/`、`ui/`：核心仿真、智能体、撮合、政策解释与界面代码。
- `data/`、`demo_scenarios/`：离线演示数据和内置场景。
- `theme/`、`static/`、`.streamlit/`：界面主题和静态资源。
- `scripts/start_competition_demo.ps1`、`scripts/start_competition_demo.bat`：Windows 快捷启动脚本。
- `requirements.txt`、`requirements-lock.txt`：依赖清单。

测试目录、历史输出、答辩资料、提交打包脚本、基准测试脚本和本地缓存已清理。运行后生成的 `outputs/`、`tmp/`、`artifacts/`、`data/cache/` 等目录会被 Git 忽略。

## 运行环境

- Python 3.11 及以上，推荐使用独立虚拟环境。
- Windows 10/11、macOS 或 Linux 均可运行；答辩推荐使用 Chrome 或 Edge 访问 Streamlit 页面。
- 在线模型 API Key 可选。未配置时系统会自动使用离线回退链路，保证可以完成现场演示。

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果需要完全复现锁定环境，可改用：

```bash
pip install -r requirements-lock.txt
```

## 启动项目

推荐命令：

```bash
python -m streamlit run app.py --server.port 8501
```

Windows 快捷脚本：

```powershell
scripts\start_competition_demo.bat
```

或：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_competition_demo.ps1
```

浏览器打开：

```text
http://127.0.0.1:8501
```

## 可选配置

项目会自动读取本地 `.env`，但该文件不会提交到 Git。可参考 `.env.example` 配置在线模型：

```bash
DEEPSEEK_API_KEY=your_key
ZHIPUAI_API_KEY=your_key
LLM_DEFAULT_PROVIDER=auto
CIVITAS_INFERENCE_MODE=lite
```

没有 API Key 时无需额外处理，项目会进入离线可演示模式。

## 答辩演示建议

1. 打开总览首页，说明系统定位和模块结构。
2. 进入政策试验台，运行默认政策模板。
3. 查看历史回放或高级分析，展示市场路径、风险指标和行为金融诊断。
4. 使用导出功能生成现场需要的报告材料。

## 快速自检

```bash
python -c "import app; assert hasattr(app, 'main')"
python -m streamlit run app.py --server.port 8501
```
