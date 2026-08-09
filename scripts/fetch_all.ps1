[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Workspace = Join-Path $Root "workspace"
New-Item -ItemType Directory -Force $Workspace | Out-Null

$EaiUrl = "https://github.com/embodied-agent-interface/embodied-agent-interface.git"
$EaiRevision = "531c62f8df2cb392bdf1907923c76da41cad4fe6"
$HealUrl = "https://huggingface.co/datasets/Trishna13/HEAL"
$HealRevision = "1887d7f3dbb0e317677821fb39f82eab12a67fd3"

function Get-PinnedRepository {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Revision,
        [Parameter(Mandatory = $true)][string]$Target
    )
    if (-not (Test-Path (Join-Path $Target ".git"))) {
        Write-Host "Cloning $Url"
        & git clone $Url $Target
        if ($LASTEXITCODE -ne 0) { throw "git clone failed: $Url" }
    }
    $Dirty = & git -C $Target status --porcelain
    if ($Dirty) {
        throw "Refusing to change a modified upstream checkout: $Target"
    }
    & git -C $Target fetch --depth 1 origin $Revision
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed: $Url@$Revision" }
    & git -C $Target checkout --detach $Revision
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed: $Revision" }
    Write-Host "Pinned $Target at $Revision"
}

Get-PinnedRepository `
    -Url $EaiUrl `
    -Revision $EaiRevision `
    -Target (Join-Path $Workspace "embodied-agent-interface")

Get-PinnedRepository `
    -Url $HealUrl `
    -Revision $HealRevision `
    -Target (Join-Path $Workspace "HEAL_dataset")

Write-Host "Download complete. Run scripts/setup_env.ps1 next."

