# UI Alignment And Curve Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the policy demo and market-trend experience with the reference screenshot set, while making simulated market curves materially less extreme and more like a policy-perturbed real index path.

**Architecture:** Keep the current `PolicySession + MarketEnvironment + SimulationRunner` stack, but split the work into two layers. First, calibrate the market-path generator so that policy shocks shift trend, volatility, panic, and volume inside a realistic envelope instead of creating outsized one-step moves. Second, reshape the front-end into a clearer “wind tunnel / market trend / agent fMRI / behavioral finance” presentation that matches the reference version’s information hierarchy.

**Tech Stack:** Python 3.14, Streamlit, pandas, Plotly, pytest, existing `PolicySession` and `MarketEnvironment` realism pipeline

---

### Task 1: Define The Realism Envelope For Non-Extreme Policy Runs

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\docs\plans\2026-04-04-ui-alignment-and-curve-calibration.md`
- Optional Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\docs\user_manual.md`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_market_realism_pipeline.py`

**Intent:**
Before touching the curve logic, define what “not too extreme” means in measurable terms.

**Target constraints:**
- Standard policy-lab runs should look like an index path, not a crash simulator.
- Daily return magnitude should usually remain in a narrow envelope unless the selected scenario is explicitly extreme.
- Panic and CSAD should drift, not spike every few bars.
- Volume should react to policy and stress, but not explode without corresponding narrative signals.

**Acceptance targets to encode in tests:**
- Default sessions should have lower realized volatility than the current baseline.
- The first 10 to 20 bars should preserve continuity and not overreact to one policy parse.
- Standard scenarios should stay inside a reasonable cumulative return / max drawdown band.
- Extreme behavior should remain available only in dedicated “wind tunnel / panic / abuse” scenarios.

**Verification:**
- Add regression tests that compare default-session volatility and drawdown against a bounded envelope instead of only asserting “non-zero movement”.

### Task 2: Calibrate PolicySession Fallback Curve Generation

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\policy_lab.py`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_lab_session.py`

**Problem observed:**
The fallback path in `_policy_session_row()` currently compounds policy pressure, order-flow proxies, and noise with coefficients that are too strong for a default index-style chart.

**Planned changes:**
- Reduce the weight of `policy_signal`, `net_flow`, and panic feedback on `daily_return`.
- Lower the noise floor and make it scale more gently with reference volatility.
- Add stronger mean reversion toward the reference trend so the curve stays anchored.
- Smooth `panic_level` and `csad` with tighter carry-over and smaller dependence on raw daily returns.
- Add a per-day return clamp for normal mode so one step cannot create an outsized candle.

**Design choice:**
Treat default policy-lab playback as “policy perturbs a market regime”, not “policy fully determines price”.

**Verification:**
- Add tests for bounded daily moves, smoother panic evolution, and less extreme cumulative drift in default sessions.

### Task 3: Calibrate Deep / Session-Runner Simulation Output For Default Display

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\engine\simulation_loop.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\core\policy_session.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\core\runtime_mode.py`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_market_realism_pipeline.py`

**Problem observed:**
Even when the UI is improved, the endogenous path can still feel too dramatic because current price formation gives large influence to order imbalance, sentiment, and policy interpretation.

**Planned changes:**
- Introduce a “default display calibration” profile for normal policy-lab sessions.
- Lower effective pass-through from policy shock to price in non-extreme modes.
- Reduce the influence of rumor and spoof-like disturbances unless the chosen scenario enables them explicitly.
- Strengthen backdrop blending or regime anchoring so the displayed index path keeps realistic inertia.
- Add a display-facing smoothing layer for the chart if needed, but only after reducing the source volatility first.

**Important rule:**
Do not fake stability purely in the chart layer. First calm the generator, then only use light display smoothing if still necessary.

**Verification:**
- Extend realism tests to assert lower volatility / drawdown under normal mode while keeping stress scenarios more reactive.

### Task 4: Split “Normal Policy Evaluation” From “Extreme Wind Tunnel” Presentation

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\app.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\policy_lab.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\dashboard.py`
- Optional Create: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\wind_tunnel_stage_demo.py`

**Why:**
Your screenshots show two different presentation intents:
- one is a staged, dramatic “政策解构 -> 社会恐慌 -> 订单簿与熔断” wind tunnel,
- the other is a calmer market-trend / behavioral-finance / agent-fMRI workspace.

The current project blends those intents too much, which makes the default chart feel overly theatrical.

**Planned changes:**
- Keep the dramatic three-stage sequence as a dedicated demo path.
- Keep `政策试验台` as the default calmer policy evaluation workspace.
- Move “extreme event storytelling” out of the normal trend chart so default runs are visually and statistically milder.

**Verification:**
- Review page entry flow and ensure users can clearly choose “normal evaluation” vs “extreme wind tunnel”.

### Task 5: Rebuild Market Trend Presentation To Match The Reference Version

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\ui\policy_lab.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\theme\competition_dark.css`
- Optional Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\app.py`
- Test: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_lab_session.py`

**Reference cues to emulate:**
- Left sidebar keeps policy input and parsed policy result visible.
- Center area focuses on the chart first, not on too many diagnostics.
- K-line + MA + volume + CSAD layout is calm, stable, and readable.
- Behavioral finance and Agent fMRI are separate explanatory views, not mixed into the main trend panel.

**Planned UI changes:**
- Reorder the page so “政策输入 / 分析结果” visually sits closer to the trend chart.
- Keep a simpler top summary strip for price, panic, and simulation days.
- Move detailed transmission cards and narrative blocks below the trend chart or into expandable sections.
- Make CSAD a low-amplitude companion chart like the reference version, not a dramatic alert area by default.

**Verification:**
- Add helper-level tests for view-model ordering and chart-section composition where feasible.

### Task 6: Add Calibration Knobs And Scenario Defaults

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\config.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\core\runtime_mode.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\data\policy_templates.json`
- Optional Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\demo_scenarios\*.json`

**Planned changes:**
- Add explicit knobs for:
  - policy pass-through strength
  - maximum daily move in normal mode
  - panic persistence
  - CSAD sensitivity
  - backdrop blending weight
- Give normal templates conservative defaults.
- Reserve high-drama defaults for named panic / rumor / intervention scenarios only.

**Verification:**
- Ensure template switching changes both narrative style and statistical behavior, not just text labels.

### Task 7: Regression And Demo Validation

**Files:**
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_lab_session.py`
- Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\tests\test_policy_market_realism_pipeline.py`
- Optional Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\README.md`
- Optional Modify: `C:\Users\Deng Junxian\Desktop\Civitas_new\docs\user_manual.md`

**Validation checklist:**
- Default policy-lab chart no longer shows cliff-like or crash-like movement without an explicitly extreme scenario.
- MA lines remain believable and the price path stays close to an index-style envelope.
- Volume and CSAD react, but do not dominate the whole visual story.
- Wind-tunnel mode still supports stronger drama when the demo explicitly needs it.
- Agent fMRI and behavior-finance pages remain connected to the same session output.

**Commands to run:**
- `pytest tests/test_policy_lab_session.py -v`
- `pytest tests/test_policy_market_realism_pipeline.py -v`
- `python -m compileall -q app.py ui engine core tests`

**Execution note:**
Implement in this order:
1. curve calibration,
2. mode split,
3. UI restructuring,
4. final regression.
