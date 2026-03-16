# User Manual

## 1. Who this is for

This manual is written for competition reviewers, instructors, and new team members who need to run and demonstrate the system quickly.

## 2. Primary entry point

- Main UI: `app.py`
- Recommended launch command: `python -m streamlit run app.py --server.port 8501`
- Recommended shortcut: `scripts\start_competition_demo.bat`

## 3. Main pages

### 答辩模式

- Purpose: fastest competition showcase path
- Best for: first 3 to 5 minutes of the defense
- Inputs: built-in scenario packs from `demo_scenarios/`
- Outputs: KPI cards, narration, A/B comparison, exported competition materials

### 专家模式

- Purpose: show the structured evidence flow behind decisions
- Best for: explaining why the result is not a black box
- Outputs: decision evidence flow, LOB view, propagation view, policy chain, risk timeline

### 历史回测

- Purpose: show that the logic can be evaluated beyond a single scripted scenario
- Best for: answering "how do you validate effectiveness?"

### 行为金融诊断

- Purpose: explain CSAD, panic heat, herding, and market behavior signals
- Best for: answering "why did the system produce this result?"

### 系统说明

- Purpose: show the architecture, workflow, and engineering structure
- Best for: quick technical summary when judges ask for implementation details

## 4. Recommended demo operation sequence

1. Open the app and stay on the home page for orientation.
2. Enter `答辩模式`.
3. Choose `tax_cut_liquidity_boost` or `regulator_stabilization_intervention`.
4. Start auto-play and explain the KPI changes.
5. Switch to `专家模式`.
6. Switch to `行为金融诊断`.
7. Switch to `历史回测`.
8. Return to the sidebar and generate competition materials.

## 5. What gets exported

When you click the competition material export action, the system generates files under `outputs/competition_materials/`.

- `competition_summary.md`
- `design_outline.md`
- `demo_script_10min.md`
- `figures/index.json`

## 6. Recommended talking points during operation

- The AI part is not only text generation; it influences structured decisions and simulation outcomes.
- The order matching path is isolated from the reasoning path.
- The system can still demonstrate its core value without an internet connection.
- The exported materials make the project easier to hand off and review.

## 7. What not to do during the defense

- Do not use `python main.py` as the primary entry point in front of judges.
- Do not rely on live API calls for the first showcase.
- Do not click incomplete auxiliary scenario folders.
- Do not spend time scrolling raw logs or long reasoning text.
