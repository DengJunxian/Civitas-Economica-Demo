param(
    [Parameter(Mandatory=$true)]
    [string]$School,
    [Parameter(Mandatory=$true)]
    [string]$ContestId,
    [string]$OutputRoot = "outputs\\submission_packages"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$packageName = "$School-$ContestId-人工智能应用"
$targetRoot = Join-Path $repoRoot $OutputRoot
$packageDir = Join-Path $targetRoot $packageName
New-Item -ItemType Directory -Path $packageDir -Force | Out-Null

$filesToCopy = @{
    "作品小结.md" = "docs/reviewer_assessment_ai_track.md"
    "条款证据矩阵.md" = "docs/ai_track_clause_evidence_matrix.md"
    "冗余代码三分法.md" = "docs/redundancy_value_triage.md"
    "答辩问答.md" = "docs/defense_qa.md"
    "用户手册.md" = "docs/user_manual.md"
    "接口说明.md" = "docs/interface_spec.md"
    "AI工具披露.md" = "AI_TOOL_DISCLOSURE.md"
    "开源引用披露.md" = "THIRD_PARTY_OPEN_SOURCE_DISCLOSURE.md"
    "README.md" = "README.md"
}

$manifest = [ordered]@{
    package_name = $packageName
    generated_at = (Get-Date).ToString("s")
    source_repo = $repoRoot
    files = @()
}

foreach ($displayName in $filesToCopy.Keys) {
    $sourceRel = $filesToCopy[$displayName]
    $sourceAbs = Join-Path $repoRoot $sourceRel
    if (-not (Test-Path $sourceAbs)) {
        Write-Warning "Skip missing file: $sourceRel"
        continue
    }
    $targetName = "$School-$ContestId-$displayName"
    $targetAbs = Join-Path $packageDir $targetName
    Copy-Item -Path $sourceAbs -Destination $targetAbs -Force
    $manifest.files += [ordered]@{
        source = $sourceRel
        target = $targetName
    }
}

$materialsDir = Join-Path $repoRoot "outputs\\competition_materials"
if (Test-Path $materialsDir) {
    $targetMaterials = Join-Path $packageDir "$School-$ContestId-比赛导出材料"
    Copy-Item -Path $materialsDir -Destination $targetMaterials -Recurse -Force
    $manifest.files += [ordered]@{
        source = "outputs/competition_materials"
        target = "$School-$ContestId-比赛导出材料"
    }
}

$manifestPath = Join-Path $packageDir "$School-$ContestId-提交包清单.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding utf8

Write-Host "[Civitas] Submission package ready: $packageDir"
Write-Host "[Civitas] Manifest: $manifestPath"
