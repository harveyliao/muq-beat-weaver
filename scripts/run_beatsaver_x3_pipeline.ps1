$ErrorActionPreference = "Stop"

$repoRoot = "E:\github_repos\muq-beat-weaver"
$beatWeaverRoot = "E:\github_repos\beat-weaver"
$processedDir = Join-Path $beatWeaverRoot "data\processed_beatsaver_x3"
$manifestPath = Join-Path $beatWeaverRoot "data\audio_manifest_beatsaver_x3.json"
$logDir = Join-Path $repoRoot "output\beatsaver_x3_pipeline_logs"

New-Item -ItemType Directory -Force -Path $processedDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string[]]$Arguments
    )

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $logDir "$stamp-$Name.stdout.log"
    $stderr = Join-Path $logDir "$stamp-$Name.stderr.log"
    Write-Host "[$(Get-Date -Format s)] Starting $Name"
    Write-Host "  cwd: $WorkingDirectory"
    Write-Host "  stdout: $stdout"
    Write-Host "  stderr: $stderr"

    $proc = Start-Process `
        -FilePath (Join-Path $WorkingDirectory ".venv\Scripts\python.exe") `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -Wait

    if ($proc.ExitCode -ne 0) {
        throw "$Name failed with exit code $($proc.ExitCode). See $stderr"
    }
}

Invoke-Step `
    -Name "cache-muq" `
    -WorkingDirectory $repoRoot `
    -Arguments @(
        "scripts\cache_muq_subset.py",
        "--audio-manifest", $manifestPath,
        "--processed-dir", "..\beat-weaver\data\processed_beatsaver_x3",
        "--limit", "0"
    )

Invoke-Step `
    -Name "build-muq-beatgrid" `
    -WorkingDirectory $repoRoot `
    -Arguments @(
        "scripts\build_muq_beatgrid_cache.py",
        "--processed-dir", "..\beat-weaver\data\processed_beatsaver_x3"
    )

Invoke-Step `
    -Name "train" `
    -WorkingDirectory $repoRoot `
    -Arguments @(
        "scripts\train_muq_precomputed.py",
        "--config", "configs\muq_frozen_base_bs8_45ep.json",
        "--audio-manifest", $manifestPath,
        "--processed-dir", "..\beat-weaver\data\processed_beatsaver_x3",
        "--output-dir", "output\muq_precomputed_beatsaver_x3_base_bs8"
    )

Write-Host "[$(Get-Date -Format s)] Pipeline completed"
