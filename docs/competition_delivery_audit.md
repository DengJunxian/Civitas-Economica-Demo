# Competition Delivery Audit

Audit perspective: provincial / national style review for the AI application category of the Computer Design Competition.

## 1. What was verified in the repository

The following checks were executed locally on 2026-03-24:

- `python -m pip check`: passed
- `python -m compileall -q app.py main.py config.py simulation_ipc.py simulation_runner.py regulator_agent.py agents core engine policy ui tests`: passed
- `pytest -q`: passed, `279 passed` in the latest full run
- `import app`: passed
- `main.check_environment()`: passed in a non-interactive terminal without API keys
- `python -m streamlit run app.py --server.headless true --server.port 8501`: returned `HTTP 200`

## 2. Engineering fixes completed during this audit

- Fixed a Windows IPC response-file race in `simulation_ipc.py` that could downgrade isolated matching to fallback mode during tests.
- Added a regression test in `tests/test_simulation_runner.py` for temporary response-file locking.
- Added missing reproducibility dependencies to `requirements.txt`.
- Added `requirements-lock.txt` for the verified local environment.
- Rewrote `.env.example` into a readable and submission-ready template.
- Added one-click startup scripts and a preflight delivery check script.
- Added Windows asyncio + pyzmq compatibility handling for IPC runtime warnings.
- Narrowed the preflight `compileall` scope to project source directories, avoiding temporary directory scan noise.
- Updated the demo script to match the current five-entry UI and added backup demo paths.
- Added delivery docs, deployment docs, user manual, structure guide, interface spec, data/model provenance note, and defense Q&A.
- Added the contest-facing design specification and project summary documents.
- Added `readme.txt` for compressed-package distribution where Markdown rendering may be unavailable.

## 3. A. Engineering deliverable completeness

### Current status

- README: now sufficient as a root index, but still not the only document reviewers should rely on
- Run instructions: present in `README.md` and `docs/deployment_guide.md`
- Deployment instructions: present
- Environment variable example: present in `.env.example`
- Directory guide: present in `docs/project_structure.md`
- Data / model / third-party provenance: present in `docs/data_model_thirdparty.md`
- Interface specification: present in `docs/interface_spec.md`
- User manual baseline: present in `docs/user_manual.md`
- Additional plain-text delivery note: present in `readme.txt`

### Assessment

The repository now has a recognizable competition delivery package instead of only a development README.

## 4. B. Competition presentation friendliness

### Strong points

- The project has a clear 10-minute main route centered on `答辩模式`.
- The built-in scenarios make it easy to control narrative rhythm.
- The system can export defense materials directly from the UI.
- The offline path is much safer for on-site review than a live API-first demo.

### Remaining weak points

- The repository still contains both `output/` and `outputs/`, which is easy to misread under time pressure.
- The full test suite takes several minutes, so pre-defense validation should use the quick script first and full validation earlier.
- Live API features are still risky for a first showcase.

### Best showcase functions

- `答辩模式`
- `专家模式` decision evidence flow
- `行为金融诊断`
- competition material export

### Flows to avoid live

- `main.py` as the main demo entry
- incomplete auxiliary scenarios
- anything that depends on the network for the first two minutes

## 5. C. AI project persuasiveness

### Positive judgment

- AI participates in a core link of the product, not only a peripheral assistant link.
- The project is stronger than a pure shell because it combines model routing, structured agent reasoning, matching, diagnostics, and backtesting.
- Input -> reasoning -> intent -> matching -> metrics -> visualization is a coherent chain.
- The behavioral diagnostics page helps explain outcomes and business value.

### What still needs stronger defense wording

- You should explicitly emphasize that the competition demo uses curated scenarios for stability, while the repository still contains live and extended paths.
- You should distinguish between "model intelligence" and "system intelligence": the model is one component, the engineered closed loop is the real work.

## 6. D. Reproduction and handoff ability

### Current judgment

- Difficulty on a new Windows machine: medium
- Difficulty on a non-Windows or non-Python-3.14 machine: medium to high

### Why reproduction is now better than before

- One-click start script exists
- Preflight check script exists
- Lock file exists
- Scenario packs are bundled
- Documentation set is much more complete

### Remaining handoff bottlenecks

- The easiest path is still Windows-focused because the bundled binary targets Python 3.14 on Windows.
- There is no packaged installer or portable release build yet.
- There is no dedicated demo account concept because the system is primarily local and scenario-driven.

## 7. E. Top 10 pre-competition risks

1. Python version does not match the bundled `_civitas_lob` binary.
   Mitigation: use Python 3.14 on Windows or rebuild via `python setup.py build_ext --inplace`.
   Fix type: code and environment.

2. Missing dependencies on a fresh machine.
   Mitigation: install from `requirements.txt` or `requirements-lock.txt`, then run `scripts/check_competition_delivery.ps1`.
   Fix type: documentation and environment.

3. On-site network instability breaks live API features.
   Mitigation: use `DEMO_MODE` and local scenario packs for the main presentation.
   Fix type: demo strategy.

4. Streamlit default port is occupied.
   Mitigation: run `scripts/start_competition_demo.ps1 -Port 8510`.
   Fix type: operation script.

5. Reviewer clicks an incomplete auxiliary scenario.
   Mitigation: only use the three complete main scenarios and follow the demo script.
   Fix type: documentation and demo strategy.

6. Directory names `output/` and `outputs/` cause confusion.
   Mitigation: explicitly state that `outputs/` is the authoritative delivery output path.
   Fix type: documentation.

7. Live explanation sounds like a shell around a model API.
   Mitigation: focus the defense on the closed loop and the isolated matching path.
   Fix type: defense strategy.

8. A single page failure stops the entire demo.
   Mitigation: prepare backup path A and backup path B from `docs/demo_script.md`.
   Fix type: demo strategy.

9. Windows IPC timing issues affect stability.
   Mitigation: the temporary file-lock retry fix has been applied in `simulation_ipc.py`; keep using the validated environment.
   Fix type: code.

10. Reviewers ask about provenance of data, models, and third-party components.
   Mitigation: bring `docs/data_model_thirdparty.md` into the submission package and cite included source paths.
   Fix type: documentation.

## 8. Overall conclusion

The project is now at a much more competition-ready level than a normal course project. It has a clear scenario, a coherent AI and simulation chain, complete runnable code, an offline-safe demo route, and a much better handoff package.

It is not yet at the strongest "excellent national-final delivery" level in three areas:

- cross-platform reproducibility is still weaker than the Windows path
- the directory/output naming is not fully cleaned up
- the defense narrative must deliberately frame the project as an engineered AI system, not only a visualization demo with model calls

If you keep the presentation centered on the curated demo path and use the newly added scripts and docs, the current repository already has a credible provincial / national style delivery foundation.
