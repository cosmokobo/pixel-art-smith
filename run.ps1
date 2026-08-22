<#
.SYNOPSIS
    Direct launcher for PixelArtSmith (GUI / CLI) on Windows.
    Guarantees live source execution and automatic dependency cross-verification.
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
$ReqFile = Join-Path $ScriptDir "requirements.txt"

$NeedInstall = $false

if (-not (Test-Path $VenvPython)) {
    Write-Host "[INFO] Creating dedicated virtual environment at: $VenvDir" -ForegroundColor Cyan
    & python -m venv $VenvDir
    $NeedInstall = $true
} else {
    # Check if core dependencies import cleanly
    $checkCmd = "import PIL, numpy, cv2, sklearn"
    & $VenvPython -c $checkCmd 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Dependencies in $VenvDir are missing or incomplete. Reinstalling..." -ForegroundColor Yellow
        $NeedInstall = $true
    }
}

if ($NeedInstall) {
    Write-Host "[INFO] Installing/verifying dependencies from requirements.txt..." -ForegroundColor Cyan
    & $VenvPip install --upgrade pip --quiet
    & $VenvPip install -r $ReqFile --quiet
    Write-Host "[SUCCESS] Dedicated virtual environment is fully synchronized." -ForegroundColor Green
}

$env:PYTHONPATH = "$ScriptDir;$env:PYTHONPATH"
& $VenvPython (Join-Path $ScriptDir "main.py") @ScriptArgs
