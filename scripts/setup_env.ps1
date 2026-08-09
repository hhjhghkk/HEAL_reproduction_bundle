[CmdletBinding()]
param(
    [switch]$WithModel
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvPrefix = Join-Path $Root "workspace\.conda-heal"
$EaiSource = Join-Path $Root "workspace\embodied-agent-interface"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Install Miniconda/Anaconda, then rerun this script."
}
if (-not (Test-Path (Join-Path $EaiSource "setup.py"))) {
    throw "EAI source is missing. Run scripts/fetch_all.ps1 first."
}
if (-not (Test-Path (Join-Path $EnvPrefix "python.exe"))) {
    & conda create --prefix $EnvPrefix python=3.10 pip -y
    if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed." }
}

& conda run --prefix $EnvPrefix python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& conda run --prefix $EnvPrefix python -m pip install `
    -e $EaiSource `
    -r (Join-Path $Root "requirements_core.txt")
if ($LASTEXITCODE -ne 0) { throw "Core dependency installation failed." }

if ($WithModel) {
    # CUDA 12.8 wheels support the RTX 50-series GPU used for this reproduction.
    & conda run --prefix $EnvPrefix python -m pip install `
        --index-url https://download.pytorch.org/whl/cu128 `
        torch==2.8.0
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
    & conda run --prefix $EnvPrefix python -m pip install `
        -r (Join-Path $Root "requirements_model.txt")
    if ($LASTEXITCODE -ne 0) { throw "Model dependency installation failed." }
}

Write-Host "Environment ready: $EnvPrefix"
Write-Host "Validate with: conda run --prefix `"$EnvPrefix`" python scripts/validate_reproduction.py"

