# Data, Models, and Third-Party Components

## 1. Built-in demo data

The competition showcase is primarily driven by local scenario packs stored in `demo_scenarios/`.

Each complete scenario contains:

- `initial_config.yaml`
- `analyst_manager_output.json`
- `narration.json`
- `metrics.csv`

These files are the main offline demo inputs and should be included in any competition submission package.

## 2. Local supporting data

- `data/seed_events.jsonl`: seed events used by the data and graph pipeline
- `data/event_graph.graphml`: local event graph artifact
- `data/cache/`: router and cache artifacts created during runtime

## 3. Optional live data sources

The project can use third-party market data libraries in live or extended workflows:

- `AkShare`
- `yfinance`

These are optional for the competition demo path and should not be treated as mandatory for offline reproduction.

## 4. Model and AI routing

The project uses a routed model architecture instead of a single hard-coded model call.

- Router implementation: `core/model_router.py`
- Providers supported in code:
  - DeepSeek
  - Zhipu
- Fallback behavior:
  - local cache reuse
  - deterministic stub when no model is available

This means the project can still demonstrate its reasoning chain and engineering workflow even when online model access is unavailable.

## 5. Model weights

- No large model weights are bundled in this repository.
- Optional local model use is exposed through environment variables:
  - `CIVITAS_LOCAL_MODEL_PATH`
  - `CIVITAS_VLLM_MODEL`
- These settings are not required for the standard competition demo.

## 6. Matching engine source

- Bundled binary: `_civitas_lob.cp314-win_amd64.pyd`
- Source code: `core/exchange/c_core/bindings.cpp`, `core/exchange/c_core/lob.cpp`
- Build script: `setup.py`

This should be explicitly stated in submission materials so judges know the binary is self-built from included source, not an opaque external artifact.

## 7. Major third-party open source components

- Streamlit: UI framework
- Plotly: charts and interactive visualizations
- Mesa: agent-based modeling support
- OpenAI Python SDK: provider-compatible model client
- pyzmq: IPC support in the ZMQ-related modules
- pandas / numpy / scipy / matplotlib: data analysis and statistics
- networkx: graph representation
- AkShare / yfinance: optional market data access
- pybind11: C++ extension binding layer

## 8. Submission note

For competition submission, include this document together with `requirements.txt`, `requirements-lock.txt`, `.env.example`, and the source files under `core/exchange/c_core/`.
