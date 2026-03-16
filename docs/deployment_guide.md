# Deployment Guide

## 1. Recommended competition environment

- OS: Windows 10 or Windows 11, 64-bit
- Python: 3.14.x recommended for the bundled `_civitas_lob.cp314-win_amd64.pyd`
- Memory: 8 GB minimum, 16 GB recommended
- Network: optional for live API mode, not required for offline demo mode

## 2. Fastest judge-friendly startup path

### Option A: one-click start

1. Open the project root.
2. Install dependencies with `pip install -r requirements.txt`.
3. Double-click `scripts\start_competition_demo.bat`.
4. Open `http://127.0.0.1:8501`.

### Option B: PowerShell startup

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\start_competition_demo.ps1
```

## 3. Locked environment installation

If you want to reproduce the validated local environment more closely:

```powershell
pip install -r requirements-lock.txt
```

This lock file was generated from the local machine used for the current verification pass.

## 4. Offline demo mode

- `DEMO_MODE` works without `DEEPSEEK_API_KEY` and `ZHIPU_API_KEY`.
- The main competition demo should use the built-in scenarios in `demo_scenarios/`.
- Recommended scenarios:
  - `tax_cut_liquidity_boost`
  - `rumor_panic_selloff`
  - `regulator_stabilization_intervention`

## 5. Live mode and optional environment variables

Use `.env.example` as the configuration template.

- `DEEPSEEK_API_KEY`: optional, enables DeepSeek online models
- `ZHIPU_API_KEY`: optional, enables Zhipu online models
- `CIVITAS_INFERENCE_MODE`: optional, `lite | standard | enterprise`
- `CIVITAS_LOCAL_MODEL_PATH`: optional, local model path
- `CIVITAS_VLLM_MODEL`: optional, model name for enterprise mode

## 6. C++ matching engine note

- The repository already contains `_civitas_lob.cp314-win_amd64.pyd`.
- If the judge machine uses Python 3.14 on Windows, recompilation is usually unnecessary.
- Rebuild only when you change `core/exchange/c_core/*` or use an incompatible Python version:

```powershell
python setup.py build_ext --inplace
```

## 7. Preflight check before defense

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_competition_delivery.ps1
```

Use `-FullTest` when time permits:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_competition_delivery.ps1 -FullTest
```

## 8. Common failure recovery

- If port `8501` is occupied, run `scripts\start_competition_demo.ps1 -Port 8510`.
- If API keys are missing, continue with offline demo mode.
- If the browser page fails, use `outputs/competition_materials/` as the backup presentation package.
- If the C++ extension mismatches the Python version, either switch to Python 3.14 or rebuild the extension.
