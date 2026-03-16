param(
    [int]$Port = 8501,
    [switch]$UseSystemPython
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param(
        [string]$RepoRoot,
        [bool]$PreferSystemPython
    )

    if (-not $PreferSystemPython) {
        $candidates = @(
            (Join-Path $RepoRoot "venv\Scripts\python.exe"),
            (Join-Path $RepoRoot ".venv\Scripts\python.exe")
        )
        foreach ($candidate in $candidates) {
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python executable not found. Install Python 3.11+ first."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path (Join-Path $repoRoot "app.py"))) {
    throw "app.py not found. Run this script from the project repository."
}

$pythonExe = Resolve-PythonExe -RepoRoot $repoRoot -PreferSystemPython:$UseSystemPython.IsPresent
$env:PYTHONUTF8 = "1"

Write-Host "[Civitas] Project root: $repoRoot"
Write-Host "[Civitas] Python: $pythonExe"
Write-Host "[Civitas] Streamlit port: $Port"
Write-Host "[Civitas] Demo mode does not require API keys."
Write-Host "[Civitas] Open http://127.0.0.1:$Port after startup."

& $pythonExe -m streamlit run app.py --server.port $Port
