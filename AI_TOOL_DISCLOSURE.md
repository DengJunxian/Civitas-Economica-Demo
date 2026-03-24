# AI_TOOL_DISCLOSURE

## Scope
This document discloses AI-assisted components and usage boundaries in Civitas-Economica-Demo.

## AI Usage Categories
| Category | Where Used | Purpose | Determinism Controls |
| --- | --- | --- | --- |
| LLM-driven smart agents | `agents/brain.py`, `agents/trader_agent.py` | Narrative reasoning, decision support | Seeded fallbacks + feature flags |
| Structured policy parsing | `policy/structured.py`, `ui/policy_lab.py` | Convert policy text to structured channels | Parser mode + config hash |
| Behavioral diagnostics | `core/behavioral_finance.py`, `ui/history_replay.py` | Stylized-facts realism scoring | Fixed seed + snapshot metadata |
| Regulator optimization support | `regulator_agent.py` | Policy action search under objectives | Seed + action-space hash |

## AI Tooling Principles
1. Feature flags gate high-risk pathways by default.
2. Legacy interfaces remain available for UI/API compatibility.
3. Experiments must emit reproducibility metadata:
   - `git commit`
   - `config hash`
   - `seed`
   - `dataset snapshot id`

## Human Oversight
- Final policy recommendations are support outputs, not autonomous mandates.
- Structured reports expose mechanism chain for review:
  `policy -> transmission -> behavior -> matching -> market result`.

## Change Log Requirement
- Any newly added AI-driven module must update this disclosure and corresponding tests.
