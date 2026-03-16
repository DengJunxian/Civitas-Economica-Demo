param(
    [switch]$FullTest
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExe {
    param([string]$RepoRoot)

    $candidates = @(
        (Join-Path $RepoRoot "venv\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Python executable not found. Install Python 3.11+ first."
}

function Add-Check {
    param(
        [string]$Name,
        [string]$Status,
        [string]$Detail
    )

    $script:checks += [pscustomobject]@{
        Check  = $Name
        Status = $Status
        Detail = $Detail
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$pythonExe = Resolve-PythonExe -RepoRoot $repoRoot
$script:checks = @()

Write-Host "[Civitas] Repo root: $repoRoot"
Write-Host "[Civitas] Python: $pythonExe"

$requiredFiles = @(
    "README.md",
    "readme.txt",
    ".env.example",
    "app.py",
    "requirements.txt",
    "requirements-lock.txt",
    "docs\competition_delivery_audit.md",
    "docs\deployment_guide.md",
    "docs\user_manual.md",
    "docs\project_structure.md",
    "docs\interface_spec.md",
    "docs\data_model_thirdparty.md",
    "docs\defense_qa.md",
    "docs\demo_script.md",
    "scripts\start_competition_demo.ps1"
)

foreach ($relativePath in $requiredFiles) {
    if (Test-Path $relativePath) {
        Add-Check -Name "file:$relativePath" -Status "PASS" -Detail "present"
    } else {
        Add-Check -Name "file:$relativePath" -Status "FAIL" -Detail "missing"
    }
}

$scenarioRoot = Join-Path $repoRoot "demo_scenarios"
$requiredScenarioFiles = @(
    "initial_config.yaml",
    "analyst_manager_output.json",
    "narration.json",
    "metrics.csv"
)

Get-ChildItem $scenarioRoot -Directory | ForEach-Object {
    $missing = @()
    foreach ($name in $requiredScenarioFiles) {
        if (-not (Test-Path (Join-Path $_.FullName $name))) {
            $missing += $name
        }
    }

    if ($missing.Count -eq 0) {
        Add-Check -Name "scenario:$($_.Name)" -Status "PASS" -Detail "complete"
    } elseif ($_.Name -eq "hybrid_replay_abuse_minimal") {
        Add-Check -Name "scenario:$($_.Name)" -Status "WARN" -Detail "auxiliary scenario; not for main competition demo"
    } else {
        Add-Check -Name "scenario:$($_.Name)" -Status "FAIL" -Detail ("missing " + ($missing -join ", "))
    }
}

try {
    & $pythonExe -m pip check | Out-Host
    Add-Check -Name "pip check" -Status "PASS" -Detail "no broken requirements"
} catch {
    Add-Check -Name "pip check" -Status "FAIL" -Detail $_.Exception.Message
}

try {
    & $pythonExe -m compileall -q . | Out-Host
    Add-Check -Name "compileall" -Status "PASS" -Detail "python files compile"
} catch {
    Add-Check -Name "compileall" -Status "FAIL" -Detail $_.Exception.Message
}

$testArgs = if ($FullTest) {
    @("-m", "pytest", "-q")
} else {
    @(
        "-m",
        "pytest",
        "-q",
        "tests/test_competition_demo_mode.py",
        "tests/test_simulation_runner.py",
        "tests/test_refactoring.py"
    )
}

try {
    & $pythonExe @testArgs | Out-Host
    Add-Check -Name "tests" -Status "PASS" -Detail ($testArgs -join " ")
} catch {
    Add-Check -Name "tests" -Status "FAIL" -Detail ($testArgs -join " ")
}

if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    Add-Check -Name "env:DEEPSEEK_API_KEY" -Status "WARN" -Detail "not set; offline demo still works"
} else {
    Add-Check -Name "env:DEEPSEEK_API_KEY" -Status "PASS" -Detail "set"
}

if ([string]::IsNullOrWhiteSpace($env:ZHIPU_API_KEY)) {
    Add-Check -Name "env:ZHIPU_API_KEY" -Status "WARN" -Detail "not set; optional"
} else {
    Add-Check -Name "env:ZHIPU_API_KEY" -Status "PASS" -Detail "set"
}

$checks | Format-Table -AutoSize | Out-Host

$hasFailure = $checks.Status -contains "FAIL"
if ($hasFailure) {
    Write-Error "[Civitas] Competition delivery check failed."
    exit 1
}

Write-Host "[Civitas] Competition delivery check finished without blocking failures."
