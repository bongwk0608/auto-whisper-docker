param(
    [switch]$Cuda,
    [switch]$Diarization,
    [switch]$DiarizationDryRun,
    [switch]$DiarizationForce
)

$ErrorActionPreference = "Stop"

$whisperArgs = @("compose")
if ($Cuda) {
    $whisperArgs += @("--profile", "cuda", "run", "--rm", "whisper-cuda")
} else {
    $whisperArgs += @("run", "--rm", "whisper")
}

Write-Host "Running Whisper transcription..."
& docker @whisperArgs

if (-not $Diarization) {
    Write-Host "Whisper complete. Diarization was not requested."
    exit 0
}

$diarizationArgs = @(
    "compose",
    "--profile",
    "diarization",
    "run",
    "--rm",
    "diarization-cuda",
    "python",
    "scripts/backfill_diarization.py"
)

if ($DiarizationDryRun) {
    $diarizationArgs += "--dry-run"
}
if ($DiarizationForce) {
    $diarizationArgs += "--force"
}

Write-Host "Running Pyannote diarization..."
& docker @diarizationArgs

