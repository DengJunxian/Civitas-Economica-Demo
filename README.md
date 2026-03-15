# Civitas Economica Demo

Civitas Economica is a multi-agent socio-economic and financial market sandbox with a C++ limit order book (LOB) matching engine, IPC-based simulation control, and a Streamlit UI for visualization and analysis.

This repository focuses on a manager-analyst agent architecture, intent-first trading decisions, behavioral finance indicators, dynamic social influence, and macro-evolution cycles.

## Highlights

- ManagerAgent plus specialist analysts (News, Quant, Risk) with Risk Alert Meeting (RAM) overrides.
- LLM outputs intent only; FCN (Fundamental-Chartist-Noise) mapping generates concrete limit orders.
- Dynamic social graph with BDI fields and portfolio Jaccard similarity weighted edges.
- Evolution loop (selection, innovation, mutation) triggered by day/week cycles.
- Macro shock injection hooks for policy stress testing.
- Wind-tunnel prediction and belief reinforcement persisted to disk.
- IPC boundary to a C++ LOB engine for fast matching and realistic microstructure.

## New Analytics/Portfolio Extensions

- Reusable risk and performance engine in [core/performance.py](core/performance.py) (empyrical-compatible with fallback).
- Standardized tear sheet and scenario comparison in [core/tear_sheet.py](core/tear_sheet.py).
- Unified portfolio construction layer in [core/portfolio/construction.py](core/portfolio/construction.py) (equal/inv-vol/mean-variance/HRP).
- Optional NDlib/EoN diffusion adapters in [core/society/network.py](core/society/network.py).
- CPT profile helpers in [core/behavioral_finance.py](core/behavioral_finance.py).

## Quickstart

Create an environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit UI:

```bash
python main.py
```

Optional: provide an API key if prompted (e.g., `DEEPSEEK_API_KEY`).

## Configuration

Runtime configuration lives in `config.yaml`. Key sections include:

- `data_flywheel`: news pipeline parameters and output path.
- Market settings: slippage, transaction costs, price impact.
- LLM parameters: model name, temperature.

The news pipeline reuses the BettaSpider flow under `data_flywheel/` (crawl + NLP + SeedEvent output).

## Testing

```bash
pytest -q
```

## 如何现场演示（答辩模式）

1. 启动应用：
```bash
python main.py
```
2. 在左侧选择 `答辩场景`（可选：
   `tax_cut_liquidity_boost` / `rumor_panic_selloff` / `regulator_stabilization_intervention`）。
3. 点击 `答辩模式` 按钮：
   - 自动切换到 `DEMO_MODE`
   - 自动加载预生成的 analyst/manager JSON、narration、metrics
   - 自动推进并回放关键讲解
4. 在 `📈 市场走势` 页查看预计算指标轨迹；在 `🏠 系统导览` 查看当前 narration 与结构化输出。
5. 演示结束后点击 `LIVE_MODE` 恢复实时模式。

说明：
- `DEMO_MODE` 不要求输入 DeepSeek/智谱 API Key。
- 若非关键网络依赖失败，系统会显示黄色告警并继续运行，不会中断演示。

## Project Structure

- `agents/`: agent implementations (manager, analysts, trading core, brains).
- `core/`: behavioral finance, scheduling, policy, society network, utilities.
- `engine/`: simulation loop and evolution cycle integration.
- `data_flywheel/`: news ingest pipeline (BettaSpider, NLP, SeedEvent).
- `simulation_runner.py`: IPC runner and LOB integration.
- `ui/`: Streamlit UI components.
- `data/`: persisted artifacts such as seed events and beliefs.

## Notes

- The LOB engine uses a compiled extension when available (`_civitas_lob`); it falls back to a pure Python implementation if needed.
- Beliefs are persisted per agent under `data/beliefs/` and injected into system prompts on load.
