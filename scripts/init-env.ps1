param(
    [string[]]$SourceDirs = @(),
    [string[]]$OutputDirs = @(),
    [string]$SourceDir = "",
    [string]$OutputDir = "",
    [string]$SourceListFile = "",
    [string]$Model = "base",
    [string]$OutputFormat = "all",
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvPath = Join-Path $ProjectRoot ".env"
$OverridePath = Join-Path $ProjectRoot "docker-compose.override.yml"
$OverallOutputPath = Join-Path $ProjectRoot "output_overall"

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

function Expand-PathList {
    param([string[]]$Values)
    $Expanded = @()
    foreach ($Value in $Values) {
        foreach ($Item in ($Value -split ",")) {
            $Trimmed = $Item.Trim()
            if ($Trimmed) {
                $Expanded += $Trimmed
            }
        }
    }
    return $Expanded
}

function Read-SourceListFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Source list file does not exist: $Path"
    }

    $Items = @()
    foreach ($Line in Get-Content -LiteralPath $Path) {
        $Trimmed = $Line.Trim()
        if ($Trimmed -and -not $Trimmed.StartsWith("#")) {
            $Items += $Trimmed
        }
    }
    return $Items
}

if ($SourceDir) {
    $SourceDirs += $SourceDir
}
if ($OutputDir) {
    $OutputDirs += $OutputDir
}

$SourceDirs = Expand-PathList -Values $SourceDirs
$OutputDirs = Expand-PathList -Values $OutputDirs
if ($SourceListFile) {
    $SourceDirs += Read-SourceListFile -Path $SourceListFile
}

while ($SourceDirs.Count -eq 0) {
    $InputValue = Read-Host "Enter the full audio/video source folder path"
    if ($InputValue) {
        $SourceDirs += $InputValue
    }

    while ($true) {
        $More = Read-Host "Add another source folder? [y/N]"
        if ($More -notmatch "^(y|Y)") {
            break
        }
        $InputValue = Read-Host "Enter the full audio/video source folder path"
        if ($InputValue) {
            $SourceDirs += $InputValue
        }
    }
}

if ($OutputDirs.Count -eq 0) {
    for ($Index = 0; $Index -lt $SourceDirs.Count; $Index++) {
        $OutputDirs += Join-Path $ProjectRoot "output"
    }
}

if ($SourceDirs.Count -ne $OutputDirs.Count) {
    throw "OutputDirs is optional, but if provided it must contain the same number of paths as SourceDirs."
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

New-Item -ItemType Directory -Path $OverallOutputPath -Force | Out-Null

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
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
WHISPER_VERBOSE=true
FINGERPRINT_MODE=metadata
LOCAL_STAGING=false
LOCAL_STAGING_DIR=/tmp/auto-whisper-staging
OVERALL_OUTPUT_ENABLED=true
OVERALL_OUTPUT_DIR=/overall-output
SUPPORTED_EXTENSIONS=.mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,.mp4,.m4v,.mov,.mkv,.webm,.avi,.wmv,.flv,.ts,.mts,.m2ts,.3gp,.3g2,.mpg,.mpeg,.vob,.ogv
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
  pipeline-cuda:
    volumes:
$($VolumeLines -join "`n")
"@ | Set-Content -LiteralPath $OverridePath -Encoding UTF8

Write-Host "Wrote $EnvPath"
Write-Host "Wrote $OverridePath"
Write-Host "Configured $($ResolvedSourceDirs.Count) source/output pair(s)"
Write-Host "Selected WHISPER_DEVICE=$Device"
