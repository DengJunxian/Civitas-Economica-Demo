param(
    [switch]$FullTest,
    [switch]$Strict,
    [string]$ReportPath
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

function Invoke-CheckedCommand {
    param(
        [string]$Name,
        [string]$Detail,
        [scriptblock]$Command
    )

    & $Command
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    if ($exitCode -eq 0) {
        Add-Check -Name $Name -Status "PASS" -Detail $Detail
    } else {
        Add-Check -Name $Name -Status "FAIL" -Detail "$Detail (exit=$exitCode)"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$pythonExe = Resolve-PythonExe -RepoRoot $repoRoot
$script:checks = @()
$compileTargets = @(
    "app.py",
    "main.py",
    "config.py",
    "simulation_ipc.py",
    "simulation_runner.py",
    "regulator_agent.py",
    "agents",
    "core",
    "engine",
    "policy",
    "ui",
    "tests"
)

Write-Host "[Civitas] Repo root: $repoRoot"
Write-Host "[Civitas] Python: $pythonExe"

$requiredFiles = @(
    "README.md",
    ".env.example",
    "app.py",
    "requirements.txt",
    "requirements-lock.txt",
    "AI_TOOL_DISCLOSURE.md",
    "THIRD_PARTY_OPEN_SOURCE_DISCLOSURE.md",
    "docs\user_manual.md",
    "docs\interface_spec.md",
    "docs\data_model_thirdparty.md",
    "docs\defense_qa.md",
    "docs\demo_script.md",
    "docs\ai_track_clause_evidence_matrix.md",
    "docs\reviewer_assessment_ai_track.md",
    "docs\redundancy_value_triage.md",
    "docs\verification_report_latest.md",
    "scripts\build_submission_package.ps1",
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

Invoke-CheckedCommand -Name "pip check" -Detail "no broken requirements" -Command {
    & $pythonExe -m pip check | Out-Host
}

Invoke-CheckedCommand -Name "compileall" -Detail (($compileTargets -join ", ") + " compile") -Command {
    & $pythonExe -m compileall -q @compileTargets | Out-Host
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

Invoke-CheckedCommand -Name "tests" -Detail ($testArgs -join " ") -Command {
    & $pythonExe @testArgs | Out-Host
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

$reportPayload = [pscustomobject]@{
    generated_at = (Get-Date).ToString("s")
    repo_root = $repoRoot
    python = $pythonExe
    strict_mode = [bool]$Strict
    full_test = [bool]$FullTest
    checks = $checks
}

if (-not [string]::IsNullOrWhiteSpace($ReportPath)) {
    $reportTarget = $ReportPath
    if (-not [System.IO.Path]::IsPathRooted($reportTarget)) {
        $reportTarget = Join-Path $repoRoot $reportTarget
    }
    $reportDir = Split-Path -Parent $reportTarget
    if ($reportDir) {
        New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    }
    $reportPayload | ConvertTo-Json -Depth 6 | Set-Content -Path $reportTarget -Encoding utf8
    Write-Host "[Civitas] Machine report written to: $reportTarget"
}

$hasFailure = $checks.Status -contains "FAIL"
if ($Strict -and ($checks.Status -contains "WARN")) {
    Write-Error "[Civitas] Competition delivery check failed (WARN treated as FAIL in -Strict mode)."
    exit 1
}
if ($hasFailure) {
    Write-Error "[Civitas] Competition delivery check failed."
    exit 1
}

Write-Host "[Civitas] Competition delivery check finished without blocking failures."
