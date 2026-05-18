param(
    [string]$SourceDir = "",
    [string]$Model = "base",
    [string]$OutputFormat = "all",
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvPath = Join-Path $ProjectRoot ".env"
$ExamplePath = Join-Path $ProjectRoot ".env.example"

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-DockerGpu {
    if (-not (Test-CommandExists "docker")) {
        return $false
    }

    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker daemon is not reachable. Writing CPU config for now."
        return $false
    }

    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi *> $null
    return $LASTEXITCODE -eq 0
}

if (-not $SourceDir) {
    $SourceDir = Read-Host "Enter the full audio/video source folder path"
}

if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
    throw "SourceDir does not exist or is not a folder: $SourceDir"
}

$ResolvedSourceDir = (Resolve-Path -LiteralPath $SourceDir).Path -replace "\\", "/"
$Device = "cpu"
$Fp16 = "false"
$NvidiaVisibleDevices = "void"

if ($Cuda) {
    $HasNvidiaSmi = Test-CommandExists "nvidia-smi"
    $HasDockerGpu = $false

    if ($HasNvidiaSmi) {
        Write-Host "NVIDIA GPU detected on host. Testing Docker GPU access..."
        $HasDockerGpu = Test-DockerGpu
    }

    if (-not ($HasNvidiaSmi -and $HasDockerGpu)) {
        throw "CUDA was requested, but NVIDIA Docker GPU access is not working. Run without -Cuda for CPU mode."
    }

    $Device = "auto"
    $Fp16 = "auto"
    $NvidiaVisibleDevices = "all"
}

@"
SOURCE_DIR=$ResolvedSourceDir
WHISPER_MODEL=$Model
WHISPER_OUTPUT_FORMAT=$OutputFormat
WHISPER_LANGUAGE=
WHISPER_TASK=transcribe
WHISPER_DEVICE=$Device
WHISPER_DOWNLOAD_DEVICE=cpu
WHISPER_FP16=$Fp16
WHISPER_CONDITION_ON_PREVIOUS_TEXT=true
WHISPER_VERBOSE=false
SUPPORTED_EXTENSIONS=.mp3,.wav,.m4a,.mp4,.mov,.mkv,.webm,.flac,.ogg,.aac,.wma
NVIDIA_VISIBLE_DEVICES=$NvidiaVisibleDevices
NVIDIA_DRIVER_CAPABILITIES=compute,utility
"@ | Set-Content -LiteralPath $EnvPath -Encoding UTF8

if (-not (Test-Path -LiteralPath $ExamplePath)) {
    Copy-Item -LiteralPath $EnvPath -Destination $ExamplePath
}

Write-Host "Wrote $EnvPath"
Write-Host "Selected WHISPER_DEVICE=$Device"
