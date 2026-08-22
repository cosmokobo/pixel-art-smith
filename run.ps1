<#
.SYNOPSIS
    Direct launcher for PixelArtSmith (GUI / CLI) on Windows.
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $ScriptDir ".venv"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] Setting up virtual environment..." -ForegroundColor Cyan
    & python -m venv $VenvDir
    $VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
    & $VenvPip install --upgrade pip --quiet
    & $VenvPip install -r (Join-Path $ScriptDir "requirements.txt") --quiet
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython (Join-Path $ScriptDir "main.py") @ScriptArgs
