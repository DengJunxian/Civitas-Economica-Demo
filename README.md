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
