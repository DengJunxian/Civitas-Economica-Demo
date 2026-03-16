# Interface Specification

## 1. User-facing interfaces

- Web UI entry: `app.py`
- Terminal launcher: `main.py`
- Recommended start scripts:
  - `scripts/start_competition_demo.ps1`
  - `scripts/start_competition_demo.bat`

## 2. Scenario pack interface

Competition scenarios are loaded from `demo_scenarios/<scenario_name>/`.

Required files:

- `initial_config.yaml`
- `analyst_manager_output.json`
- `narration.json`
- `metrics.csv`

Required metric columns:

- `step`
- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `csad`
- `panic_level`

## 3. Internal runner command interface

The isolated matching runner communicates through file-system IPC.

Supported command types:

- `submit_intent`
- `advance_time`
- `get_snapshot`
- `shutdown`

Core implementation files:

- `simulation_ipc.py`
- `simulation_runner.py`

## 4. Output interface

Main export directory:

- `outputs/competition_materials/`

Generated files:

- `competition_summary.md`
- `design_outline.md`
- `demo_script_10min.md`
- `figures/index.json`

Additional reports generated during runtime:

- `outputs/stylized_facts_report.json`
- `outputs/ecology_metrics.csv`
- `outputs/market_abuse_report.json`
- `outputs/intervention_effect_report.json`
