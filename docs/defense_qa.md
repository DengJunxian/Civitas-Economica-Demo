# Defense Q&A

## 1. What problem does this project solve?

It turns policy shocks, rumor events, and market behavior into a reproducible AI-driven simulation workflow so reviewers can see decision logic, market reaction, and behavioral diagnostics in one system.

## 2. Where is the AI in this project?

The AI is in the analysis and decision layer: model routing, analyst/manager reasoning, structured decision output, and the mapping from evidence to trading intent and risk constraints.

## 3. Why is this not just a model API wrapper?

Because the model output is only one stage. The project also includes decision schemas, simulation scheduling, isolated matching, behavioral metrics, backtesting, and visual explanation layers.

## 4. What happens if no API key is available?

The project can still run offline in `DEMO_MODE` using built-in scenario packs and deterministic fallback behavior. This is deliberate so the core demo is stable during defense.

## 5. What is the most competition-friendly innovation point?

The strongest point is the closed loop: input event -> structured multi-agent reasoning -> market matching -> behavioral diagnostics -> explainable visual output.

## 6. How do you explain the result instead of only showing charts?

Use the expert mode and behavioral diagnostics page. They show evidence flow, CSAD, panic level, herding, and the policy transmission chain rather than only the final price line.

## 7. How do you prove the project is engineered, not only designed?

The repository contains complete code, tests, scenario files, deployment scripts, a C++ matching engine source path, and a reproducible demo route. The full local verification also passes.

## 8. How do you validate correctness?

The repository includes automated tests across simulation, diagnostics, backtesting, policy chain, and demo mode. We report the latest pass count from `scripts/check_competition_delivery.ps1 -FullTest -ReportPath ...` instead of hard-coding a fixed number.

## 9. Why does the project include a C++ matching engine?

To separate high-frequency order book execution from the slower reasoning layer and make the market execution path more realistic and stable.

## 10. What are the current limitations?

The easiest reproduction path is still Windows plus Python 3.14 because the bundled binary targets that environment. Live API features also depend on network stability.

## 11. If judges ask for future work, what should you say?

You can mention cross-platform packaging, more formal dataset/version governance, more benchmark scenarios, and stronger live data replay integration.

## 12. If judges challenge whether the demo is scripted, how should you respond?

Say that the competition path intentionally uses curated scenarios for stable defense delivery, while the codebase also contains live routing, simulation, backtesting, and diagnostic modules beyond the scripted demo path.
