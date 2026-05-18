param(
    [string[]]$SourceDirs = @(),
    [string[]]$OutputDirs = @(),
    [string]$SourceDir = "",
    [string]$OutputDir = "",
    [string]$Model = "base",
    [string]$OutputFormat = "all",
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvPath = Join-Path $ProjectRoot ".env"
$OverridePath = Join-Path $ProjectRoot "docker-compose.override.yml"

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

function ConvertTo-ComposePath {
    param([string]$Path)
    return $Path -replace "\\", "/"
}

function ConvertTo-YamlDoubleQuoted {
    param([string]$Value)
    return '"' + ($Value -replace '\\', '\\' -replace '"', '\"') + '"'
}

if ($SourceDir) {
    $SourceDirs += $SourceDir
}
if ($OutputDir) {
    $OutputDirs += $OutputDir
}

while ($SourceDirs.Count -eq 0) {
    $InputValue = Read-Host "Enter the full audio/video source folder path"
    if ($InputValue) {
        $SourceDirs += $InputValue
    }
    $OutputValue = Read-Host "Enter the full transcript output folder path"
    if ($OutputValue) {
        $OutputDirs += $OutputValue
    }

    while ($true) {
        $More = Read-Host "Add another source/output pair? [y/N]"
        if ($More -notmatch "^(y|Y)") {
            break
        }
        $InputValue = Read-Host "Enter the full audio/video source folder path"
        $OutputValue = Read-Host "Enter the full transcript output folder path"
        if ($InputValue) {
            $SourceDirs += $InputValue
        }
        if ($OutputValue) {
            $OutputDirs += $OutputValue
        }
    }
}

if ($SourceDirs.Count -ne $OutputDirs.Count) {
    if ($SourceDirs.Count -eq 1 -and $OutputDirs.Count -eq 0) {
        $OutputDirs += Join-Path $ProjectRoot "output"
    }
}

if ($SourceDirs.Count -ne $OutputDirs.Count) {
    throw "SourceDirs and OutputDirs must contain the same number of paths."
}
if ($SourceDirs.Count -eq 0) {
    throw "At least one source/output pair is required."
}

$ResolvedSourceDirs = @()
$ResolvedOutputDirs = @()

foreach ($Path in $SourceDirs) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Source directory does not exist or is not a folder: $Path"
    }
    $ResolvedSourceDirs += ConvertTo-ComposePath (Resolve-Path -LiteralPath $Path).Path
}

foreach ($Path in $OutputDirs) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
    $ResolvedOutputDirs += ConvertTo-ComposePath (Resolve-Path -LiteralPath $Path).Path
}

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

$Pairs = @()
for ($Index = 0; $Index -lt $ResolvedSourceDirs.Count; $Index++) {
    $Number = $Index + 1
    $Pairs += [ordered]@{
        input = "/inputs/input-{0:d3}" -f $Number
        output = "/outputs/output-{0:d3}" -f $Number
    }
}
$InputOutputPairsJson = ConvertTo-Json -InputObject $Pairs -Compress

@"
SOURCE_DIRS=$($ResolvedSourceDirs -join ";")
OUTPUT_DIRS=$($ResolvedOutputDirs -join ";")
INPUT_OUTPUT_PAIRS=$InputOutputPairsJson
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

$VolumeLines = New-Object System.Collections.Generic.List[string]
for ($Index = 0; $Index -lt $ResolvedSourceDirs.Count; $Index++) {
    $Number = $Index + 1
    $VolumeLines.Add("      - type: bind")
    $VolumeLines.Add("        source: $(ConvertTo-YamlDoubleQuoted -Value $ResolvedSourceDirs[$Index])")
    $VolumeLines.Add("        target: /inputs/input-{0:d3}" -f $Number)
}
for ($Index = 0; $Index -lt $ResolvedOutputDirs.Count; $Index++) {
    $Number = $Index + 1
    $VolumeLines.Add("      - type: bind")
    $VolumeLines.Add("        source: $(ConvertTo-YamlDoubleQuoted -Value $ResolvedOutputDirs[$Index])")
    $VolumeLines.Add("        target: /outputs/output-{0:d3}" -f $Number)
}

@"
services:
  whisper:
    volumes:
$($VolumeLines -join "`n")
  whisper-cuda:
    volumes:
$($VolumeLines -join "`n")
"@ | Set-Content -LiteralPath $OverridePath -Encoding UTF8

Write-Host "Wrote $EnvPath"
Write-Host "Wrote $OverridePath"
Write-Host "Configured $($ResolvedSourceDirs.Count) source/output pair(s)"
Write-Host "Selected WHISPER_DEVICE=$Device"
