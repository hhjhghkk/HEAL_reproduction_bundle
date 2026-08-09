[CmdletBinding()]
param(
    [ValidateSet("virtualhome", "behavior")]
    [string]$Dataset = "virtualhome",
    [ValidateSet("generate_prompts", "evaluate_results")]
    [string]$Mode = "generate_prompts",
    [string]$ResponsePath = "",
    [string]$OutputDir = "outputs\eai"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$EnvPrefix = Join-Path $Root "workspace\.conda-heal"
$Arguments = @(
    "run", "--prefix", $EnvPrefix,
    "eai-eval",
    "--dataset", $Dataset,
    "--eval-type", "goal_interpretation",
    "--mode", $Mode,
    "--output-dir", (Join-Path $Root $OutputDir)
)
if ($ResponsePath) {
    $Arguments += @("--llm-response-path", (Join-Path $Root $ResponsePath))
}
& conda @Arguments
if ($LASTEXITCODE -ne 0) { throw "EAI goal interpretation failed." }

