# Policy Simulation Realism Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make policy-driven multi-agent market simulation more realistic by closing the loop from policy text to heterogeneous belief updates, balance-sheet constraints, order-book pressure, trade execution, and post-trade feedback.

**Architecture:** Keep the current layered design, but strengthen the causal coupling between layers. The main upgrade is to move from "policy changes sentiment, sentiment changes action" to "policy changes funding conditions, risk constraints, narratives, inventory pressure, and execution style, which then reshape the order book and feed back into future cognition."

**Tech Stack:** Python, Streamlit, Mesa, NetworkX, custom order book / market kernel, pytest, existing DeepSeek / GLM routing.

---

## Current realism gaps

Based on the current codebase, the project already has strong pieces in place:

- `core/policy_committee.py` already supports multi-role policy parsing and parameter bounding.
- `policy/interpretation_engine.py` and `engine/simulation_loop.py` already support policy package to agent belief transmission.
- `agents/trader_agent.py` already has System 1 / System 2 routing, behavioral finance state, execution plans, and social-graph hooks.
- `core/society/network.py` already has SIRS-style semantic contagion and dynamic edge weights.
- `simulation_runner.py`, `core/exchange/market_kernel.py`, and `core/market_engine.py` already support intent buffering and matching.

The simulation still risks looking "fake" for four structural reasons:

1. Policy impact is still too direct and too synchronous. It needs lag, partial pass-through, and heterogeneous balance-sheet channels.
2. Agent cognition is richer than agent constraints. Real markets are often moved less by opinions than by margin, risk budgets, inventory, and forced execution.
3. Execution plans exist, but market impact is still too close to "decision equals fill." Queue position, liquidity withdrawal, cancel waves, and slippage feedback need to matter more.
4. Social contagion is already modeled, but it should be coupled more tightly to unrealized PnL, realized fills, drawdown trauma, and observed liquidation events.

## Success criteria

The upgrade is complete only if the simulator can consistently produce the following under policy shocks and stress scenarios:

- Transmission lag: policy effects diffuse over several ticks instead of fully landing in one tick.
- Cross-sectional heterogeneity: the same policy produces different reactions across retail, market maker, leveraged, and institution cohorts.
- Microstructure deformation: spread widening, depth thinning, order cancellation waves, and slippage jump under stress.
- Endogenous cascades: forced selling and sentiment contagion amplify each other without hard-coded price crashes.
- Stylized facts: volatility clustering, fat tails, drawdown clustering, and order-flow imbalance become visible in outputs.
- Reproducibility: the same seed remains stable, while different seeds show realistic path dispersion rather than arbitrary randomness.

## Task 1: Build a realism baseline and acceptance harness

**Files:**
- Modify: `engine/simulation_loop.py`
- Modify: `core/market_metrics.py`
- Modify: `tests/test_market_microstructure_realism.py`
- Modify: `tests/test_policy_transmission_layers.py`
- Add: `tests/test_policy_market_realism_pipeline.py`

**Plan:**
1. Extend end-of-step reporting so each tick records spread, top-of-book depth, cancel ratio, slippage, forced-liquidation volume, and policy pass-through ratio.
2. Add a single realism scorecard helper that compares scenario outputs against target stylized-fact ranges.
3. Create deterministic tests for three scenario families: easing policy, rumor panic, and leverage unwind.
4. Fail tests when policy shock has no lag, when stress does not widen spread / reduce depth, or when all agent cohorts react identically.

**Verification:**
- `pytest tests/test_market_microstructure_realism.py -q`
- `pytest tests/test_policy_transmission_layers.py -q`
- `pytest tests/test_policy_market_realism_pipeline.py -q`

## Task 2: Make policy transmission multi-channel, lagged, and state-dependent

**Files:**
- Modify: `core/macro/government.py`
- Modify: `policy/interpretation_engine.py`
- Modify: `engine/simulation_loop.py`
- Modify: `core/policy_committee.py`
- Modify: `tests/test_policy_transmission.py`
- Modify: `tests/test_multi_agent_heterogeneity.py`

**Plan:**
1. Split policy shock into separate channels: funding, transaction cost, regulatory constraint, narrative guidance, and liquidity support.
2. For each agent cohort, compute channel-specific pass-through using balance-sheet sensitivity, risk appetite, institution type, and policy reaction delay.
3. Add lag structure so some effects are immediate, some delayed, and some only activate after market confirmation or media diffusion.
4. Make committee output include confidence, disagreement tags, and conservative fallback severity so downstream agents can weight policy certainty.
5. Store a full policy transmission trace per tick for later diagnosis and report generation.

**Verification:**
- `pytest tests/test_policy_transmission.py -q`
- `pytest tests/test_multi_agent_heterogeneity.py -q`
- `pytest tests/test_policy_transmission_layers.py -q`

## Task 3: Strengthen trader constraints with balance sheets, leverage, and trauma memory

**Files:**
- Modify: `agents/trader_agent.py`
- Modify: `core/behavioral_finance.py`
- Modify: `core/regulation/risk_control.py`
- Modify: `agents/persona.py`
- Modify: `tests/test_trader_agent.py`
- Modify: `tests/test_risk_engine.py`

**Plan:**
1. Add per-agent state for leverage usage, maintenance margin distance, liquidity preference, inventory pressure, and drawdown trauma.
2. Tie System 2 triggering not only to sentiment and shocks, but also to risk events such as margin breach probability, VaR jump, and abnormal slippage.
3. Make prospect-theory reference points update from actual fills, unrealized PnL, recent highs, peer anchors, and policy anchors with decaying memory.
4. Add forced deleveraging and rights stripping logic: once thresholds are breached, the execution layer can override agent preference and submit liquidation plans.
5. Persist a trauma-memory variable so agents hit by prior crashes become more fragile or more defensive in later scenarios.

**Verification:**
- `pytest tests/test_trader_agent.py -q`
- `pytest tests/test_risk_engine.py -q`
- `pytest tests/test_behavioral_simulation.py -q`

## Task 4: Upgrade from intention realism to execution realism

**Files:**
- Modify: `agents/trader_agent.py`
- Modify: `simulation_runner.py`
- Modify: `core/exchange/market_kernel.py`
- Modify: `core/market_engine.py`
- Modify: `tests/test_trader_execution_plan.py`
- Modify: `tests/test_market_kernel_timeline.py`

**Plan:**
1. Expand `ExecutionPlan` usage so intent is translated into child-order schedules, cancel-replace behavior, and urgency-dependent order types more often.
2. Add queue / latency realism: order arrival delay, cancel delay, and phase-dependent execution behavior should materially affect fills.
3. Allow market makers and institutions to withdraw or reprice liquidity when volatility, toxicity, or rumor intensity spikes.
4. Add execution feedback: missed fills, adverse selection, and slippage should update confidence, urgency, and next-tick order style.
5. Model liquidation and panic selling as a sequence of urgent child orders instead of one-step notional dumps.

**Verification:**
- `pytest tests/test_trader_execution_plan.py -q`
- `pytest tests/test_market_kernel_timeline.py -q`
- `pytest tests/test_market_microstructure_realism.py -q`

## Task 5: Couple social contagion to holdings, PnL, and observed market stress

**Files:**
- Modify: `core/society/network.py`
- Modify: `agents/trader_agent.py`
- Modify: `engine/simulation_loop.py`
- Modify: `tests/test_behavioral_simulation.py`

**Plan:**
1. Make infection pressure depend not only on semantic similarity, but also on shared exposure, correlated holdings, and observed liquidation events in the neighborhood.
2. Make recovery and immunity decay depend on realized market stabilization, not only time-in-state.
3. Let nodes broadcast different narrative force depending on recent PnL, credibility, and influence rank, so "loud but wrong" actors lose transmission power over time.
4. Feed matching outcomes back into the graph each tick: large sweep trades, cancel waves, and sharp slippage should strengthen panic propagation.
5. Add dynamic re-wiring rules so stressed markets form temporary echo chambers and concentration of influence.

**Verification:**
- `pytest tests/test_behavioral_simulation.py -q`
- `pytest tests/test_multi_agent_heterogeneity.py -q`

## Task 6: Add calibration, replay, and anti-fake diagnostics

**Files:**
- Modify: `engine/simulation_loop.py`
- Modify: `core/agent_replay.py`
- Modify: `agents/report/report_agent.py`
- Modify: `ui/policy_lab.py`
- Add: `outputs/realism_calibration_report.json` (runtime artifact)

**Plan:**
1. Add scenario replay comparisons against historical or pseudo-historical stress templates, focusing on spread, depth, turnover burst, and rebound shape.
2. Generate calibration reports that explain whether a scenario is "too smooth," "too synchronous," or "too deterministic."
3. Surface the realism diagnostics in the UI so judges can see not only what happened, but why the simulator considers the path credible.
4. Make the post-run report explicitly separate model inference from deterministic facts to reduce LLM overclaiming.

**Verification:**
- `pytest tests/test_simulation_runner.py -q`
- `pytest tests/test_policy_lab_session.py -q`

## Recommended implementation order

1. Task 1 first, so realism has measurable acceptance criteria.
2. Task 2 and Task 3 next, because realism mainly depends on transmission plus constraints.
3. Task 4 after that, because microstructure realism only matters once beliefs and constraints are credible.
4. Task 5 next, so cascades become endogenous rather than scripted.
5. Task 6 last, for calibration and answerability in demos and defense.

## Notes for this repository

- Prefer evolving the existing architecture instead of replacing it. The current codebase already has the right modules; the problem is coupling strength and validation depth.
- The biggest credibility jump will not come from adding more LLM calls. It will come from making the LLM output pass through deterministic balance-sheet, risk, and execution constraints.
- For答辩表达, the most persuasive line is: "政策不是直接改价格，而是先改资金成本、风险约束、叙事传播和流动性供给，再由交易与撮合共同涌现价格路径。"

