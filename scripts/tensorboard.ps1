$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$tensorboard = Join-Path $repoRoot ".venv\Scripts\tensorboard.exe"
$logdir = Join-Path $repoRoot "output"

if (-not (Test-Path $tensorboard)) {
    throw "TensorBoard executable not found at $tensorboard"
}

& $tensorboard --logdir $logdir
