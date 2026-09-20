param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        'b0_baseline',
        'a0_lora',
        'a1_lora_focal',
        'a2_lora_freq',
        'a3_lora_fmiseg',
        'a4_lora_freq_focal',
        'a9_frozen_freq_focal'
    )]
    [string]$Experiment,
    [int]$Seed = 1219,
    [ValidateRange(1, 10000)]
    [int]$Epochs = 100,
    [ValidateRange(1, 64)]
    [int]$BatchSize = 2,
    [string]$ResumePath = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$launcher = Join-Path $PSScriptRoot 'run_paper_experiment.ps1'
$logRoot = Join-Path $repoRoot 'runtime_logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -match 'train_model\.py'
}
if ($existing) {
    throw 'A training process is already running; refusing to start another.'
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logRoot "train_${Experiment}_${timestamp}.stdout.log"
$stderr = Join-Path $logRoot "train_${Experiment}_${timestamp}.stderr.log"
$metadata = Join-Path $logRoot 'paper_experiment_current.env'

$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $launcher,
    '-Experiment', $Experiment,
    '-Seed', [string]$Seed,
    '-Epochs', [string]$Epochs,
    '-BatchSize', [string]$BatchSize
)
$resolvedResumePath = ''
if ($ResumePath) {
    $resolvedResumePath = (Resolve-Path -LiteralPath $ResumePath).Path
    $arguments += @('-ResumePath', $resolvedResumePath)
}
$startParameters = @{
    FilePath = 'powershell.exe'
    ArgumentList = $arguments
    WorkingDirectory = $repoRoot
    RedirectStandardOutput = $stdout
    RedirectStandardError = $stderr
    WindowStyle = 'Hidden'
    PassThru = $true
}
$process = Start-Process @startParameters

@(
    "EXPERIMENT=$Experiment"
    "SEED=$Seed"
    "EPOCHS=$Epochs"
    "BATCH_SIZE=$BatchSize"
    "MIOPEN_FIND_MODE=FAST"
    "DETERMINISTIC_BACKEND=1"
    "MIOPEN_ENABLED=0"
    "TRAIN_DROP_LAST=1"
    "RESUME_PATH=$resolvedResumePath"
    "LAUNCHER_PID=$($process.Id)"
    "STARTED_AT=$((Get-Date).ToString('o'))"
    "REPO=$repoRoot"
    "STDOUT=$stdout"
    "STDERR=$stderr"
) | Set-Content -LiteralPath $metadata -Encoding UTF8

Write-Output "Started $Experiment locally (launcher PID $($process.Id))."
Write-Output "Metadata: $metadata"
Write-Output "Stdout: $stdout"
Write-Output "Stderr: $stderr"
