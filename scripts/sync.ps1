# Sync code to / results from a remote GPU cluster via rsync (over SSH).
#
# Uses WSL's rsync — Windows doesn't ship rsync natively. The wrapper
# translates Windows paths to /mnt/c/... so WSL rsync sees the same files.
#
# Connection details (user / host / remote path) live in a *gitignored*
# config file: scripts/sync.config.ps1. Copy sync.config.example.ps1 to
# sync.config.ps1 and fill in your values.
#
# Usage:
#   .\scripts\sync.ps1 push                # local → cluster (code only)
#   .\scripts\sync.ps1 pull                # cluster → local (results + checkpoints)
#   .\scripts\sync.ps1 push -DryRun        # preview without transferring
#   .\scripts\sync.ps1 pull -Only results  # pull only one subtree
#
# rsync flags used:
#   -a    archive (recursive, preserve perms/timestamps/symlinks)
#   -v    verbose (file names)
#   -z    compress in flight
#   --delete (push only) remove remote files that don't exist locally — keeps
#            cluster tree byte-identical to local
#   --info=progress2  human-readable per-transfer progress

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('push', 'pull')]
    [string]$Direction,

    [ValidateSet('all', 'checkpoints', 'results')]
    [string]$Only = 'all',

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# Cluster connection — loaded from a gitignored local config file so the
# host/user never lands on GitHub.
$ConfigPath = Join-Path $PSScriptRoot 'sync.config.ps1'
if (-not (Test-Path $ConfigPath)) {
    throw "Missing $ConfigPath. Copy scripts/sync.config.example.ps1 to scripts/sync.config.ps1 and fill in RemoteUser / RemoteHost / RemoteRoot."
}
. $ConfigPath
foreach ($v in 'RemoteUser','RemoteHost','RemoteRoot') {
    if (-not (Get-Variable -Name $v -Scope Script -ErrorAction SilentlyContinue)) {
        throw "sync.config.ps1 must define `$$v"
    }
}

# Convert the local repo root to a WSL path: C:\Users\... → /mnt/c/Users/...
$LocalWin  = (Get-Location).Path
$LocalWsl  = '/mnt/' + $LocalWin.Substring(0,1).ToLower() + ($LocalWin.Substring(2) -replace '\\','/')

# rsync needs the username escaped — backslash works as-is in single quotes for SSH.
$Remote = "${RemoteUser}@${RemoteHost}:${RemoteRoot}"

# Things we never want crossing the wire either direction.
$ExcludeArgs = @(
    '--exclude=.git/',
    '--exclude=.github/',
    '--exclude=.vscode/',
    '--exclude=.venv/',
    '--exclude=venv/',
    '--exclude=__pycache__/',
    '--exclude=*.pyc',
    '--exclude=*.egg-info/',
    '--exclude=.pytest_cache/',
    '--exclude=.ruff_cache/',
    '--exclude=.mypy_cache/',
    '--exclude=.ipynb_checkpoints/',
    '--exclude=.DS_Store',
    '--exclude=Thumbs.db'
)

$DryArg = if ($DryRun) { '--dry-run' } else { '' }

function Invoke-Rsync {
    param([string[]]$RsyncArgs)
    # Filter out empty strings, then call WSL rsync.
    $filtered = $RsyncArgs | Where-Object { $_ -ne '' }
    & wsl rsync @filtered
    if ($LASTEXITCODE -ne 0) { throw "rsync failed with exit code $LASTEXITCODE" }
}

if ($Direction -eq 'push') {
    # Push code from local → cluster. Excludes outputs (those live on the cluster).
    $pushExcludes = $ExcludeArgs + @(
        '--exclude=checkpoints/',
        '--exclude=results/',
        '--exclude=runs/',
        '--exclude=wandb/',
        '--exclude=mlruns/',
        '--exclude=*.pt',
        '--exclude=*.ckpt',
        '--exclude=data/'
    )
    Write-Host "→ Pushing $LocalWin → $Remote/" -ForegroundColor Cyan
    Invoke-Rsync @(
        '-avz',
        '--delete',                  # mirror local; remove orphaned remote files
        '--info=progress2',
        $DryArg,
        $pushExcludes,
        "$LocalWsl/",                # trailing slash = "contents of"
        "$Remote/"
    )
}
else {
    # Pull artifacts from cluster → local. Default: both checkpoints/ and results/.
    $targets = @()
    if ($Only -in 'all', 'checkpoints') { $targets += 'checkpoints' }
    if ($Only -in 'all', 'results')     { $targets += 'results' }

    foreach ($dir in $targets) {
        $localTarget = Join-Path $LocalWin $dir
        if (-not (Test-Path $localTarget)) {
            New-Item -ItemType Directory -Path $localTarget | Out-Null
        }
        $localTargetWsl = '/mnt/' + $localTarget.Substring(0,1).ToLower() + ($localTarget.Substring(2) -replace '\\','/')
        Write-Host "→ Pulling $Remote/$dir/ → $localTarget" -ForegroundColor Cyan
        Invoke-Rsync @(
            '-avz',
            '--info=progress2',
            $DryArg,
            $ExcludeArgs,
            "$Remote/$dir/",
            "$localTargetWsl/"
        )
    }
}

Write-Host ""
Write-Host "✓ Done." -ForegroundColor Green
