param(
    [hashtable]$Mappings = @{
        X = "\\10.0.0.3\smb_backup"
        Y = "\\10.0.0.3\smb_documents"
        Z = "\\10.0.0.3\smb_media"
    },
    [int]$RetryCount = 30,
    [int]$RetrySeconds = 10,
    [int]$CheckTimeoutSeconds = 5,
    [int]$RepeatMinutes = 1,
    [switch]$RepairConflicts,
    [switch]$Diagnose,
    [switch]$InstallTask,
    [switch]$UninstallTask,
    [string]$TaskName = "AutoWhisperMountNetworkDrives"
)

$ErrorActionPreference = "Stop"

$ScriptPath = $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)
$LogDir = Join-Path $ProjectRoot "logs"
$LogPath = Join-Path $LogDir "mount-windows-network-drives.log"

function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO", "WARN", "ERROR")]
        [string]$Level = "INFO"
    )

    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Line = "[$Timestamp] [$Level] $Message"
    Write-Host $Line

    if (-not (Test-Path -LiteralPath $LogDir -PathType Container)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
}

function Normalize-DriveLetter {
    param([string]$Drive)

    $Letter = $Drive.Trim().TrimEnd(":").ToUpperInvariant()
    if ($Letter -notmatch "^[A-Z]$") {
        throw "Invalid drive letter: $Drive"
    }
    return $Letter
}

function Test-PathReady {
    param(
        [string]$Path,
        [int]$TimeoutSeconds
    )

    $Job = Start-Job -ScriptBlock {
        param([string]$CheckPath)
        Test-Path -LiteralPath $CheckPath -PathType Container -ErrorAction SilentlyContinue
    } -ArgumentList $Path

    try {
        $Completed = Wait-Job -Job $Job -Timeout $TimeoutSeconds
        if (-not $Completed) {
            return $false
        }

        try {
            return [bool](Receive-Job -Job $Job -ErrorAction Stop)
        }
        catch {
            return $false
        }
    }
    finally {
        Remove-Job -Job $Job -Force -ErrorAction SilentlyContinue
    }
}

function Test-IsAdmin {
    $Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $PrincipalCheck = [System.Security.Principal.WindowsPrincipal]::new($Identity)
    return $PrincipalCheck.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-IsUncPath {
    param([string]$Path)

    return (-not [string]::IsNullOrWhiteSpace($Path)) -and $Path -match "^\\\\[^\\]+\\[^\\]+"
}

function Get-SmbMappingRoot {
    param([string]$Letter)

    try {
        $SmbMapping = Get-SmbMapping -LocalPath "${Letter}:" -ErrorAction Stop
        if ($SmbMapping -and $SmbMapping.RemotePath) {
            return $SmbMapping.RemotePath
        }
    }
    catch {
        return $null
    }

    return $null
}

function Get-CimMappingRoot {
    param([string]$Letter)

    try {
        $Disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='${Letter}:'" -ErrorAction Stop
        if ($Disk -and $Disk.ProviderName) {
            return $Disk.ProviderName
        }
    }
    catch {
        return $null
    }

    return $null
}

function Get-PSDriveMappingRoots {
    param([string]$Letter)

    $Drive = Get-PSDrive -Name $Letter -PSProvider FileSystem -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        Root = if ($Drive) { $Drive.Root } else { $null }
        DisplayRoot = if ($Drive) { $Drive.DisplayRoot } else { $null }
    }
}

function Get-NetUseMappingRoot {
    param([string]$Letter)

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $NetOutput = & net.exe use "${Letter}:" 2>&1
    $NetExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference

    if ($NetExitCode -ne 0) {
        return $null
    }

    foreach ($Line in $NetOutput) {
        if ($Line -match "^\s*Remote name\s+(.+?)\s*$") {
            return $Matches[1]
        }
    }

    return $null
}

function Get-DriveMappingSources {
    param([string]$Letter)

    $PSDriveRoots = Get-PSDriveMappingRoots -Letter $Letter
    $Sources = @(
        [pscustomobject]@{ Source = "Get-SmbMapping"; Path = Get-SmbMappingRoot -Letter $Letter },
        [pscustomobject]@{ Source = "Win32_LogicalDisk"; Path = Get-CimMappingRoot -Letter $Letter },
        [pscustomobject]@{ Source = "Get-PSDrive.Root"; Path = $PSDriveRoots.Root },
        [pscustomobject]@{ Source = "Get-PSDrive.DisplayRoot"; Path = $PSDriveRoots.DisplayRoot },
        [pscustomobject]@{ Source = "net use"; Path = Get-NetUseMappingRoot -Letter $Letter }
    )

    return @($Sources | Where-Object { -not [string]::IsNullOrWhiteSpace($_.Path) })
}

function Get-MappedDriveRoot {
    param([string]$Letter)

    $Sources = Get-DriveMappingSources -Letter $Letter
    if ($Sources.Count -gt 0) {
        $UncSource = @($Sources | Where-Object { Test-IsUncPath -Path $_.Path } | Select-Object -First 1)
        if ($UncSource.Count -gt 0) {
            return $UncSource[0].Path
        }

        return $Sources[0].Path
    }

    return $null
}

function Get-DriveDiagnosis {
    param(
        [string]$Letter,
        [string]$UncPath
    )

    $DriveRoot = "${Letter}:\"
    $Sources = Get-DriveMappingSources -Letter $Letter
    $Conflicts = @($Sources | Where-Object { (Test-IsUncPath -Path $_.Path) -and $_.Path.TrimEnd("\") -ine $UncPath.TrimEnd("\") })
    $DriveReady = Test-PathReady -Path $DriveRoot -TimeoutSeconds $CheckTimeoutSeconds
    $ShareReady = Test-PathReady -Path $UncPath -TimeoutSeconds $CheckTimeoutSeconds

    $Status = "Missing"
    if ($Conflicts.Count -gt 0) {
        $Status = "Conflict"
    }
    elseif ($Sources.Count -gt 0 -and $DriveReady -and $ShareReady) {
        $Status = "Ready"
    }
    elseif ($Sources.Count -gt 0 -or $ShareReady) {
        $Status = "Unreachable"
    }

    return [pscustomobject]@{
        Letter = $Letter
        ConfiguredPath = $UncPath
        Status = $Status
        DriveReady = $DriveReady
        ShareReady = $ShareReady
        Sources = $Sources
        Conflicts = $Conflicts
    }
}

function Format-MappingSources {
    param($Sources)

    if (-not $Sources -or $Sources.Count -eq 0) {
        return "none"
    }

    return (($Sources | ForEach-Object { "$($_.Source)='$($_.Path)'" }) -join "; ")
}

function Show-Diagnosis {
    param(
        [System.Collections.Specialized.OrderedDictionary]$NormalizedMappings,
        [string]$Name
    )

    $Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    Write-Host "Windows network drive diagnosis"
    Write-Host "User: $($Identity.Name)"
    Write-Host "Elevated: $(Test-IsAdmin)"
    Write-Host "Process: $([System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName)"
    Write-Host ""

    $Task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($Task) {
        Write-Host "Scheduled Task: $Name"
        Write-Host "  Principal: UserId=$($Task.Principal.UserId); LogonType=$($Task.Principal.LogonType); RunLevel=$($Task.Principal.RunLevel)"
        foreach ($Action in @($Task.Actions)) {
            Write-Host "  Action: $($Action.Execute) $($Action.Arguments)"
        }
        Write-Host "  Settings: MultipleInstances=$($Task.Settings.MultipleInstances); StartWhenAvailable=$($Task.Settings.StartWhenAvailable); ExecutionTimeLimit=$($Task.Settings.ExecutionTimeLimit)"
        Write-Host ""
    }
    else {
        Write-Host "Scheduled Task: $Name not found or not readable in this session."
        Write-Host ""
    }

    foreach ($Letter in $NormalizedMappings.Keys) {
        $Diagnosis = Get-DriveDiagnosis -Letter $Letter -UncPath $NormalizedMappings[$Letter]
        Write-Host "${Letter}: $($Diagnosis.Status)"
        Write-Host "  Configured: $($Diagnosis.ConfiguredPath)"
        Write-Host "  Drive reachable: $($Diagnosis.DriveReady)"
        Write-Host "  Share reachable: $($Diagnosis.ShareReady)"
        Write-Host "  Sources: $(Format-MappingSources -Sources $Diagnosis.Sources)"

        if ($Diagnosis.Status -eq "Conflict") {
            Write-Host "  Conflicts: $(Format-MappingSources -Sources $Diagnosis.Conflicts)"
            Write-Host "  Repair: .\scripts\mount-windows-network-drives.ps1 -RepairConflicts -RetryCount 1"
            Write-Host "  Note: run diagnosis and repair in the same normal or Administrator session that reports the conflict."
        }

        Write-Host ""
    }
}

function Remove-MappedDrive {
    param([string]$Letter)

    Write-Log "Removing stale mapping for ${Letter}:"
    Remove-PSDrive -Name $Letter -Force -ErrorAction SilentlyContinue
    Remove-SmbMapping -LocalPath "${Letter}:" -Force -UpdateProfile -ErrorAction SilentlyContinue

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & net.exe use "${Letter}:" /delete /y 2>$null | Out-Null
    $NetExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference

    if ($NetExitCode -ne 0 -and $NetExitCode -ne 2) {
        throw "net.exe could not remove ${Letter}: mapping. Exit code: $NetExitCode"
    }
}

function Ensure-MappedDrive {
    param(
        [string]$Letter,
        [string]$UncPath,
        [bool]$RepairConflicts
    )

    $DriveRoot = "${Letter}:\"
    $MappingSources = Get-DriveMappingSources -Letter $Letter
    $UncMappingSources = @($MappingSources | Where-Object { Test-IsUncPath -Path $_.Path })
    $ExistingRoot = if ($UncMappingSources.Count -gt 0) { $UncMappingSources[0].Path } elseif ($MappingSources.Count -gt 0) { $MappingSources[0].Path } else { $null }

    if (-not $ExistingRoot -and $RepairConflicts) {
        Remove-MappedDrive -Letter $Letter
    }

    if ($ExistingRoot) {
        if ($ExistingRoot.TrimEnd("\") -ieq $UncPath.TrimEnd("\")) {
            $DriveReady = Test-PathReady -Path $DriveRoot -TimeoutSeconds $CheckTimeoutSeconds
            $ShareReady = Test-PathReady -Path $UncPath -TimeoutSeconds $CheckTimeoutSeconds

            if ($DriveReady -and $ShareReady) {
                Write-Log "${Letter}: is already mapped to $UncPath and reachable."
                return $true
            }

            Write-Log "${Letter}: is mapped to $UncPath but is not reachable yet." "WARN"
            Remove-MappedDrive -Letter $Letter
        }
        else {
            $ConflictDetails = Format-MappingSources -Sources @($MappingSources | Where-Object { (Test-IsUncPath -Path $_.Path) -and $_.Path.TrimEnd("\") -ine $UncPath.TrimEnd("\") })
            if (-not $RepairConflicts) {
                throw "${Letter}: has conflicting mapping(s): $ConflictDetails. Expected '$UncPath'. To repair the affected session, run: .\scripts\mount-windows-network-drives.ps1 -RepairConflicts -RetryCount 1. If this differs between normal and Administrator PowerShell, run diagnosis and repair in the session that reports the conflict."
            }

            Write-Log "${Letter}: has conflicting mapping(s): $ConflictDetails. Expected '$UncPath'. Repairing configured drive mapping." "WARN"
            Remove-MappedDrive -Letter $Letter
        }
    }

    if (-not (Test-PathReady -Path $UncPath -TimeoutSeconds $CheckTimeoutSeconds)) {
        Write-Log "Share $UncPath is not reachable yet." "WARN"
        return $false
    }

    Write-Log "Mapping ${Letter}: to $UncPath"
    try {
        New-PSDrive -Name $Letter -PSProvider FileSystem -Root $UncPath -Persist -Scope Global | Out-Null
    }
    catch {
        if ($RepairConflicts -and $_.Exception.Message -like "*remembered connection*") {
            throw "${Letter}: still has a remembered Windows SMB connection. Rerun this repair from PowerShell as Administrator: .\scripts\mount-windows-network-drives.ps1 -RepairConflicts -RetryCount 1"
        }

        throw
    }

    if (Test-PathReady -Path $DriveRoot -TimeoutSeconds $CheckTimeoutSeconds) {
        Write-Log "${Letter}: is ready."
        return $true
    }

    Write-Log "${Letter}: was mapped but did not become reachable." "WARN"
    return $false
}

function Install-AutoMountTask {
    param([string]$Name)

    $Identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $PrincipalCheck = [System.Security.Principal.WindowsPrincipal]::new($Identity)
    $IsAdmin = $PrincipalCheck.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $IsAdmin) {
        throw "Installing the startup Scheduled Task trigger requires an elevated PowerShell session. Open PowerShell as Administrator and rerun: .\scripts\mount-windows-network-drives.ps1 -InstallTask"
    }

    $PowerShellPath = Join-Path $PSHOME "powershell.exe"
    $ActionArgs = "-WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -RetryCount 1"
    $Action = New-ScheduledTaskAction -Execute $PowerShellPath -Argument $ActionArgs
    $CurrentUser = $Identity.Name
    $LogonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
    $StartupTrigger = New-ScheduledTaskTrigger -AtStartup
    $RepeatTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes)
    $Triggers = @($LogonTrigger, $StartupTrigger, $RepeatTrigger)
    $Principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited
    $Settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
        -MultipleInstances IgnoreNew `
        -StartWhenAvailable

    Register-ScheduledTask `
        -TaskName $Name `
        -Action $Action `
        -Trigger $Triggers `
        -Principal $Principal `
        -Settings $Settings `
        -Description "Mount Auto Whisper Windows network drives." `
        -Force `
        -ErrorAction Stop | Out-Null

    Write-Log "Installed Scheduled Task '$Name' with startup, user logon, and every-$RepeatMinutes-minute recovery triggers."
    Write-Log "Drive letters are user-session scoped; the logon trigger makes mappings visible to normal desktop apps."
}

function Uninstall-AutoMountTask {
    param([string]$Name)

    $Task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if (-not $Task) {
        Write-Log "Scheduled Task '$Name' is not installed."
        return
    }

    Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    Write-Log "Removed Scheduled Task '$Name'."
}

if ($RetryCount -lt 1) {
    throw "RetryCount must be at least 1."
}
if ($RetrySeconds -lt 1) {
    throw "RetrySeconds must be at least 1."
}
if ($CheckTimeoutSeconds -lt 1) {
    throw "CheckTimeoutSeconds must be at least 1."
}
if ($RepeatMinutes -lt 1) {
    throw "RepeatMinutes must be at least 1."
}
if ($InstallTask -and $UninstallTask) {
    throw "Use either -InstallTask or -UninstallTask, not both."
}
if ($Diagnose -and ($InstallTask -or $UninstallTask)) {
    throw "Use -Diagnose by itself, not with -InstallTask or -UninstallTask."
}

if ($InstallTask) {
    Install-AutoMountTask -Name $TaskName
    exit 0
}

if ($UninstallTask) {
    Uninstall-AutoMountTask -Name $TaskName
    exit 0
}

$NormalizedMappings = [ordered]@{}
foreach ($Drive in $Mappings.Keys) {
    $Letter = Normalize-DriveLetter -Drive ([string]$Drive)
    $UncPath = [string]$Mappings[$Drive]

    if ($UncPath -notmatch "^\\\\[^\\]+\\[^\\]+") {
        throw "Mapping for ${Letter}: must be a full UNC share path such as \\server\share. Got: $UncPath"
    }

    $NormalizedMappings[$Letter] = $UncPath.TrimEnd("\")
}

if ($Diagnose) {
    Show-Diagnosis -NormalizedMappings $NormalizedMappings -Name $TaskName
    exit 0
}

for ($Attempt = 1; $Attempt -le $RetryCount; $Attempt++) {
    Write-Log "Network drive mount attempt $Attempt of $RetryCount."
    $Failed = @()

    foreach ($Letter in $NormalizedMappings.Keys) {
        try {
            if (-not (Ensure-MappedDrive -Letter $Letter -UncPath $NormalizedMappings[$Letter] -RepairConflicts $RepairConflicts.IsPresent)) {
                $Failed += "${Letter}:"
            }
        }
        catch {
            Write-Log $_.Exception.Message "ERROR"
            throw
        }
    }

    if ($Failed.Count -eq 0) {
        Write-Log "All requested network drives are ready."
        exit 0
    }

    if ($Attempt -lt $RetryCount) {
        Write-Log "Waiting $RetrySeconds seconds before retrying: $($Failed -join ', ')" "WARN"
        Start-Sleep -Seconds $RetrySeconds
    }
}

Write-Log "Network drives were not ready after $RetryCount attempt(s). Check VPN/network access and Windows Credential Manager credentials." "ERROR"
exit 1
