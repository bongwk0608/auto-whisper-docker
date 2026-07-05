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

function Get-ExistingEnvValue {
    param(
        [string]$Key,
        [string]$DefaultValue = ""
    )
    if (Test-Path -LiteralPath $EnvPath -PathType Leaf) {
        foreach ($Line in Get-Content -LiteralPath $EnvPath) {
            if ($Line.StartsWith("$Key=")) {
                return $Line.Substring($Key.Length + 1)
            }
        }
    }
    return $DefaultValue
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

$PyannoteAuthToken = Get-ExistingEnvValue -Key "PYANNOTE_AUTH_TOKEN"
$WhisperLanguage = Get-ExistingEnvValue -Key "WHISPER_LANGUAGE"
$WhisperTask = Get-ExistingEnvValue -Key "WHISPER_TASK" -DefaultValue "transcribe"
$WhisperConditionOnPreviousText = Get-ExistingEnvValue -Key "WHISPER_CONDITION_ON_PREVIOUS_TEXT" -DefaultValue "false"
$WhisperVerbose = Get-ExistingEnvValue -Key "WHISPER_VERBOSE" -DefaultValue "true"
$WhisperOomFallback = Get-ExistingEnvValue -Key "WHISPER_OOM_FALLBACK" -DefaultValue "cpu"
$WhisperWorkerMode = Get-ExistingEnvValue -Key "WHISPER_WORKER_MODE" -DefaultValue "on_oom"
$WhisperStrictResourceCheck = Get-ExistingEnvValue -Key "WHISPER_STRICT_RESOURCE_CHECK" -DefaultValue "false"
$FingerprintMode = Get-ExistingEnvValue -Key "FINGERPRINT_MODE" -DefaultValue "metadata"
$LocalStaging = Get-ExistingEnvValue -Key "LOCAL_STAGING" -DefaultValue "false"
$LocalStagingDir = Get-ExistingEnvValue -Key "LOCAL_STAGING_DIR" -DefaultValue "/tmp/auto-whisper-staging"
$OverallOutputEnabled = Get-ExistingEnvValue -Key "OVERALL_OUTPUT_ENABLED" -DefaultValue "true"
$OverallOutputDir = Get-ExistingEnvValue -Key "OVERALL_OUTPUT_DIR" -DefaultValue "/overall-output"
$SafeOutputFilenames = Get-ExistingEnvValue -Key "SAFE_OUTPUT_FILENAMES" -DefaultValue "auto"
$SupportedExtensions = Get-ExistingEnvValue -Key "SUPPORTED_EXTENSIONS" -DefaultValue ".mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,.mp4,.m4v,.mov,.mkv,.webm,.avi,.wmv,.flv,.ts,.mts,.m2ts,.3gp,.3g2,.mpg,.mpeg,.vob,.ogv"
$PyannoteMetricsEnabled = Get-ExistingEnvValue -Key "PYANNOTE_METRICS_ENABLED" -DefaultValue "0"
$DiarizationBackend = Get-ExistingEnvValue -Key "DIARIZATION_BACKEND" -DefaultValue "pyannote"
$DiarizationModel = Get-ExistingEnvValue -Key "DIARIZATION_MODEL" -DefaultValue "pyannote/speaker-diarization-community-1"
$DiarizationVerbose = Get-ExistingEnvValue -Key "DIARIZATION_VERBOSE" -DefaultValue "false"
$DiarizationProgress = Get-ExistingEnvValue -Key "DIARIZATION_PROGRESS"
$DiarizationTf32 = Get-ExistingEnvValue -Key "DIARIZATION_TF32" -DefaultValue "false"
$DiarizationOomFallback = Get-ExistingEnvValue -Key "DIARIZATION_OOM_FALLBACK" -DefaultValue "cpu"
$DiarizationCudaQuarantineAfterOom = Get-ExistingEnvValue -Key "DIARIZATION_CUDA_QUARANTINE_AFTER_OOM" -DefaultValue "false"
$DiarizationCudaDebugErrors = Get-ExistingEnvValue -Key "DIARIZATION_CUDA_DEBUG_ERRORS" -DefaultValue "false"
$DiarizationWorkerMode = Get-ExistingEnvValue -Key "DIARIZATION_WORKER_MODE" -DefaultValue "always"
$DiarizationGpuMemoryLog = Get-ExistingEnvValue -Key "DIARIZATION_GPU_MEMORY_LOG" -DefaultValue "false"
$DiarizationWorkerTimeoutSeconds = Get-ExistingEnvValue -Key "DIARIZATION_WORKER_TIMEOUT_SECONDS" -DefaultValue "7200"
$DiarizationGpuMemoryWaitSeconds = Get-ExistingEnvValue -Key "DIARIZATION_GPU_MEMORY_WAIT_SECONDS" -DefaultValue "0"
$DiarizationAudioPreprocess = Get-ExistingEnvValue -Key "DIARIZATION_AUDIO_PREPROCESS" -DefaultValue "always"
$DiarizationAudioPreprocessDir = Get-ExistingEnvValue -Key "DIARIZATION_AUDIO_PREPROCESS_DIR" -DefaultValue "/tmp/auto-whisper-diarization"
$DiarizationMinOverlapRatio = Get-ExistingEnvValue -Key "DIARIZATION_MIN_OVERLAP_RATIO" -DefaultValue "0.3"
$DiarizationNumSpeakers = Get-ExistingEnvValue -Key "DIARIZATION_NUM_SPEAKERS"
$DiarizationMinSpeakers = Get-ExistingEnvValue -Key "DIARIZATION_MIN_SPEAKERS"
$DiarizationMaxSpeakers = Get-ExistingEnvValue -Key "DIARIZATION_MAX_SPEAKERS"
$DiarizationOutputDir = Get-ExistingEnvValue -Key "DIARIZATION_OUTPUT_DIR" -DefaultValue "/app/output_pyannote"
$DiarizationOverallOutputDir = Get-ExistingEnvValue -Key "DIARIZATION_OVERALL_OUTPUT_DIR" -DefaultValue "/app/output_pyannote_overall"
$DiarizationCacheDir = Get-ExistingEnvValue -Key "DIARIZATION_CACHE_DIR" -DefaultValue "/app/state/diarization-cache"
$HfHome = Get-ExistingEnvValue -Key "HF_HOME" -DefaultValue "/cache/huggingface"
$TorchHome = Get-ExistingEnvValue -Key "TORCH_HOME" -DefaultValue "/cache/torch"

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
WHISPER_LANGUAGE=$WhisperLanguage
WHISPER_TASK=$WhisperTask
WHISPER_DEVICE=$Device
WHISPER_DOWNLOAD_DEVICE=cpu
WHISPER_FP16=$Fp16
WHISPER_CONDITION_ON_PREVIOUS_TEXT=$WhisperConditionOnPreviousText
WHISPER_VERBOSE=$WhisperVerbose
WHISPER_OOM_FALLBACK=$WhisperOomFallback
WHISPER_WORKER_MODE=$WhisperWorkerMode
WHISPER_STRICT_RESOURCE_CHECK=$WhisperStrictResourceCheck
FINGERPRINT_MODE=$FingerprintMode
LOCAL_STAGING=$LocalStaging
LOCAL_STAGING_DIR=$LocalStagingDir
OVERALL_OUTPUT_ENABLED=$OverallOutputEnabled
OVERALL_OUTPUT_DIR=$OverallOutputDir
SAFE_OUTPUT_FILENAMES=$SafeOutputFilenames
SUPPORTED_EXTENSIONS=$SupportedExtensions
NVIDIA_VISIBLE_DEVICES=$NvidiaVisibleDevices
NVIDIA_DRIVER_CAPABILITIES=compute,utility
PYANNOTE_AUTH_TOKEN=$PyannoteAuthToken
PYANNOTE_METRICS_ENABLED=$PyannoteMetricsEnabled
DIARIZATION_BACKEND=$DiarizationBackend
DIARIZATION_MODEL=$DiarizationModel
DIARIZATION_VERBOSE=$DiarizationVerbose
DIARIZATION_PROGRESS=$DiarizationProgress
DIARIZATION_TF32=$DiarizationTf32
DIARIZATION_OOM_FALLBACK=$DiarizationOomFallback
DIARIZATION_CUDA_QUARANTINE_AFTER_OOM=$DiarizationCudaQuarantineAfterOom
DIARIZATION_CUDA_DEBUG_ERRORS=$DiarizationCudaDebugErrors
DIARIZATION_WORKER_MODE=$DiarizationWorkerMode
DIARIZATION_GPU_MEMORY_LOG=$DiarizationGpuMemoryLog
DIARIZATION_WORKER_TIMEOUT_SECONDS=$DiarizationWorkerTimeoutSeconds
DIARIZATION_GPU_MEMORY_WAIT_SECONDS=$DiarizationGpuMemoryWaitSeconds
DIARIZATION_AUDIO_PREPROCESS=$DiarizationAudioPreprocess
DIARIZATION_AUDIO_PREPROCESS_DIR=$DiarizationAudioPreprocessDir
DIARIZATION_MIN_OVERLAP_RATIO=$DiarizationMinOverlapRatio
DIARIZATION_NUM_SPEAKERS=$DiarizationNumSpeakers
DIARIZATION_MIN_SPEAKERS=$DiarizationMinSpeakers
DIARIZATION_MAX_SPEAKERS=$DiarizationMaxSpeakers
DIARIZATION_OUTPUT_DIR=$DiarizationOutputDir
DIARIZATION_OVERALL_OUTPUT_DIR=$DiarizationOverallOutputDir
DIARIZATION_CACHE_DIR=$DiarizationCacheDir
HF_HOME=$HfHome
TORCH_HOME=$TorchHome
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
if ($Cuda -and -not $PyannoteAuthToken) {
    Write-Host "PYANNOTE_AUTH_TOKEN is blank. Set it in .env before running pipeline-cuda or diarization."
}
