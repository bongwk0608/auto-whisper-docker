param(
    [switch]$SkipContainerChecks
)

$ErrorActionPreference = "Continue"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "== $Title =="
}

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )

    Write-Section $Label
    try {
        $global:LASTEXITCODE = $null
        & $Command
        if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            Write-Host "Command exited with code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "Failed: $($_.Exception.Message)"
    }
}

function Convert-BytesToGiB {
    param([double]$Bytes)
    if ($Bytes -le 0) {
        return "unknown"
    }
    return "{0:N2} GiB" -f ($Bytes / 1GB)
}

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Section "Host Summary"
Write-Host "Project: $ProjectRoot"
Write-Host "Processor: $env:PROCESSOR_IDENTIFIER"
Write-Host "Host logical processors: $env:NUMBER_OF_PROCESSORS"

Invoke-Checked "WSL Status" {
    wsl --status
    wsl -l -v
}

Invoke-Checked "WSL Resource Config" {
    $configPath = Join-Path $env:USERPROFILE ".wslconfig"
    if (Test-Path -LiteralPath $configPath) {
        Write-Host $configPath
        Get-Content -LiteralPath $configPath
    }
    else {
        Write-Host "No .wslconfig found at $configPath"
    }
}

Invoke-Checked "Docker Effective Resources" {
    $info = docker info --format "{{.ServerVersion}}`n{{.NCPU}}`n{{.MemTotal}}`n{{json .Runtimes}}`n{{.DefaultRuntime}}`n{{.OperatingSystem}}`n{{.KernelVersion}}"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker daemon is not reachable from this shell."
        return
    }

    $lines = @($info)
    $serverVersion = $lines[0]
    $cpus = $lines[1]
    $memoryBytes = $lines[2]
    $runtimesJson = $lines[3]
    $defaultRuntime = $lines[4]
    $os = $lines[5]
    $kernel = $lines[6]
    $runtimeNames = try {
        (($runtimesJson | ConvertFrom-Json).PSObject.Properties.Name | Sort-Object) -join ", "
    }
    catch {
        $runtimesJson
    }

    Write-Host "Server Version: $serverVersion"
    Write-Host "CPUs: $cpus"
    Write-Host "Total Memory: $(Convert-BytesToGiB ([double]$memoryBytes))"
    Write-Host "Runtimes: $runtimeNames"
    Write-Host "Default Runtime: $defaultRuntime"
    Write-Host "Operating System: $os"
    Write-Host "Kernel Version: $kernel"
}

Invoke-Checked "GPU And VRAM" {
    nvidia-smi
    Write-Host ""
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
}

Invoke-Checked "Compose Config" {
    docker compose config
}

if (-not $SkipContainerChecks) {
    Invoke-Checked "Container CPU And CUDA" {
        docker compose run --rm whisper python -c "import os, torch; print('CPU:', os.cpu_count()); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
    }

    Invoke-Checked "Container Memory And Disk" {
        docker compose run --rm whisper bash -lc "free -h && nproc && df -h / /input /project-output /root/.cache/whisper"
    }
}

Write-Section "Whisper Recommendation"
Write-Host "For this RTX 3050 Ti 4GB GPU:"
Write-Host "- Use WHISPER_MODEL=base for reliable unattended runs."
Write-Host "- Use WHISPER_MODEL=small when nvidia-smi shows about 3GB or more free VRAM."
Write-Host "- Keep WHISPER_DEVICE=auto and WHISPER_FP16=auto for CUDA runs."
Write-Host "- If CUDA runs out of memory, close GPU-heavy apps or switch to WHISPER_MODEL=base."
Write-Host "- If base still fails, set WHISPER_DEVICE=cpu and WHISPER_FP16=false."
Write-Host ""
Write-Host "Recommended .wslconfig baseline:"
Write-Host "[wsl2]"
Write-Host "memory=12GB"
Write-Host "processors=6"
Write-Host "swap=24GB"
Write-Host "localhostForwarding=true"
