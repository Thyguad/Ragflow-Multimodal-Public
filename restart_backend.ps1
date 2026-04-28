[CmdletBinding()]
param(
    [string]$DistroName = "Ubuntu-22.04",
    [int]$BaseTimeoutSec = 180,
    [int]$BackendTimeoutSec = 180,
    [switch]$SkipBaseRecreate,
    [switch]$SkipDistroReset
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = $PSScriptRoot
$DockerDir = Join-Path $RepoRoot "docker"
$BackendLogPath = Join-Path $RepoRoot "backend.log"
$BackendErrLogPath = Join-Path $RepoRoot "backend_err.log"
$ServerLogPath = Join-Path $RepoRoot "logs/ragflow_server.log"
$ElasticAuth = "elastic:infini_rag_flow"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    [OK] $Message" -ForegroundColor Green
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host "    [WARN] $Message" -ForegroundColor Yellow
}

function Test-CommandAvailable {
    param([string]$CommandName)
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Missing command: $CommandName"
    }
}

function Invoke-WslBash {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [switch]$IgnoreExitCode
    )

    $output = & wsl -d $DistroName -- bash -lc $Command 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and -not $IgnoreExitCode) {
        $text = ($output | Out-String).Trim()
        if (-not $text) {
            $text = "No output"
        }
        throw "WSL command failed: $text"
    }
    return ($output | Out-String).Trim()
}

function Invoke-DockerComposeBase {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    Push-Location $DockerDir
    try {
        & docker compose -f "docker-compose-base.yml" @Args
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose failed: $($Args -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Get-CurlStatusCode {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [string]$BasicAuth
    )

    $args = @("-s", "-o", "NUL", "-w", "%{http_code}")
    if ($BasicAuth) {
        $args += @("-u", $BasicAuth)
    }
    $args += $Url
    $result = & curl.exe @args 2>$null
    if ($LASTEXITCODE -ne 0) {
        return "000"
    }
    return ($result | Out-String).Trim()
}

function Wait-Until {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Condition,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutSec,
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [int]$IntervalSec = 2
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) {
            Write-Ok $Description
            return
        }
        Start-Sleep -Seconds $IntervalSec
    }

    throw "$Description timed out after ${TimeoutSec}s"
}

function Wait-ForDockerContainer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [int]$TimeoutSec = 120,
        [string]$ExpectedHealth = ""
    )

    Wait-Until -TimeoutSec $TimeoutSec -Description "Container $Name is ready" -Condition {
        $state = (& docker inspect -f '{{.State.Status}}' $Name 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $state -ne "running") {
            return $false
        }

        if (-not $ExpectedHealth) {
            return $true
        }

        $health = (& docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' $Name 2>$null | Out-String).Trim()
        return $health -eq $ExpectedHealth
    }
}

function Show-TailIfExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [int]$Tail = 40
    )

    if (Test-Path $Path) {
        Write-Host ""
        Write-Host "--- $Path (tail $Tail) ---" -ForegroundColor DarkYellow
        Get-Content $Path -Tail $Tail
    }
}

try {
    Write-Step "Checking required commands"
    Test-CommandAvailable -CommandName "docker"
    Test-CommandAvailable -CommandName "wsl"
    Test-CommandAvailable -CommandName "curl.exe"
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not available. Start Docker Desktop first."
    }
    Write-Ok "docker / wsl / curl.exe are available"

    Write-Step "Stopping old backend processes"
    Invoke-WslBash -IgnoreExitCode -Command "pkill -f 'docker/launch_backend_service.sh'"
    Invoke-WslBash -IgnoreExitCode -Command "pkill -f 'api/ragflow_server.py'"
    Invoke-WslBash -IgnoreExitCode -Command "pkill -f 'rag/svr/task_executor.py'"
    Invoke-WslBash -IgnoreExitCode -Command "pkill -f 'image_server.py'"
    Start-Sleep -Seconds 2
    Write-Ok "Old backend processes were cleaned up"

    if (-not $SkipDistroReset) {
        Write-Step "Resetting WSL distro state"
        & wsl --terminate $DistroName 2>$null
        Start-Sleep -Seconds 3
        Write-Ok "Distro $DistroName was terminated"
    }
    else {
        Write-Step "Skipping WSL distro reset"
        Write-WarnMsg "Current WSL state was preserved by -SkipDistroReset"
    }

    if (-not $SkipBaseRecreate) {
        Write-Step "Recreating base containers"
        Invoke-DockerComposeBase @("down", "--remove-orphans")
        Invoke-DockerComposeBase @("up", "-d")
    }
    else {
        Write-Step "Skipping base container recreation"
        Write-WarnMsg "Current base containers were preserved by -SkipBaseRecreate"
    }

    Write-Step "Waiting for base services"
    Wait-ForDockerContainer -Name "ragflow-es-01" -TimeoutSec $BaseTimeoutSec -ExpectedHealth "healthy"
    Wait-ForDockerContainer -Name "ragflow-mysql" -TimeoutSec $BaseTimeoutSec -ExpectedHealth "healthy"
    Wait-ForDockerContainer -Name "ragflow-minio" -TimeoutSec $BaseTimeoutSec
    Wait-ForDockerContainer -Name "ragflow-redis" -TimeoutSec $BaseTimeoutSec
    Wait-Until -TimeoutSec $BaseTimeoutSec -Description "Elasticsearch on port 1200 is reachable" -Condition {
        (Get-CurlStatusCode -Url "http://127.0.0.1:1200/" -BasicAuth $ElasticAuth) -eq "200"
    }

    Write-Step "Starting backend"
    if (-not (Test-Path $BackendLogPath)) {
        New-Item -ItemType File -Path $BackendLogPath -Force | Out-Null
    }
    if (-not (Test-Path $BackendErrLogPath)) {
        New-Item -ItemType File -Path $BackendErrLogPath -Force | Out-Null
    }
    Clear-Content $BackendLogPath -ErrorAction SilentlyContinue
    Clear-Content $BackendErrLogPath -ErrorAction SilentlyContinue
    & (Join-Path $RepoRoot "start_backend.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "start_backend.ps1 failed"
    }

    Write-Step "Waiting for backend ports"
    Wait-Until -TimeoutSec $BackendTimeoutSec -Description "Main API on port 9380 is reachable" -Condition {
        (Get-CurlStatusCode -Url "http://127.0.0.1:9380/") -ne "000"
    }
    Wait-Until -TimeoutSec $BackendTimeoutSec -Description "Image server on port 8000 is reachable" -Condition {
        (Get-CurlStatusCode -Url "http://127.0.0.1:8000/") -ne "000"
    }

    Write-Step "Restart completed"
    Write-Host "API:   http://127.0.0.1:9380" -ForegroundColor Green
    Write-Host "Image: http://127.0.0.1:8000" -ForegroundColor Green
    Write-Host ""
    Write-Host "Follow logs with:" -ForegroundColor Yellow
    Write-Host "  Get-Content `"$BackendErrLogPath`" -Wait"
}
catch {
    Write-Host ""
    Write-Host "Backend restart failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Show-TailIfExists -Path $BackendErrLogPath -Tail 80
    Show-TailIfExists -Path $ServerLogPath -Tail 80
    exit 1
}
