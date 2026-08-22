<#
.SYNOPSIS
    Builds the PixelArtSmith standalone executable using PyInstaller.
#>
[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = $ScriptDir
while ($RepoRoot -ne [System.IO.Path]::GetPathRoot($RepoRoot) -and
       -not (Test-Path (Join-Path $RepoRoot "AGENTS.md")) -and
       -not (Test-Path (Join-Path $RepoRoot ".git"))) {
    $RepoRoot = Split-Path -Parent $RepoRoot
}

if ((Test-Path (Join-Path $RepoRoot "AGENTS.md")) -or (Test-Path (Join-Path $RepoRoot ".gitmodules"))) {
    $BuildRoot = Join-Path $RepoRoot "build\pixel-art-smith"
} else {
    $BuildRoot = Join-Path $ScriptDir "dist"
}

Write-Host "[INFO] Building PixelArtSmith -> $BuildRoot" -ForegroundColor Cyan

$VenvDir = Join-Path $ScriptDir ".venv"
$PythonExe = "python"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] Creating virtual environment at $VenvDir..." -ForegroundColor Cyan
    & $PythonExe -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"
$VenvPyInstaller = Join-Path $VenvDir "Scripts\pyinstaller.exe"

& $VenvPip install --upgrade pip --quiet
& $VenvPip install -r (Join-Path $ScriptDir "requirements.txt") --quiet

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

Write-Host "[INFO] Running PyInstaller..." -ForegroundColor Cyan
& $VenvPyInstaller `
    --name "pixel-art-smith" `
    --onefile `
    --clean `
    --noconfirm `
    --distpath $BuildRoot `
    --workpath (Join-Path $ScriptDir "build_temp") `
    --specpath $ScriptDir `
    --copy-metadata "pymatting" `
    --copy-metadata "rembg" `
    --copy-metadata "onnxruntime" `
    --copy-metadata "tqdm" `
    --copy-metadata "jsonschema" `
    --hidden-import "pixel_art_smith" `
    --hidden-import "pixel_art_smith.core" `
    --hidden-import "pixel_art_smith.cli" `
    --hidden-import "pixel_art_smith.gui" `
    --hidden-import "PIL" `
    --hidden-import "cv2" `
    --hidden-import "numpy" `
    --hidden-import "sklearn" `
    --hidden-import "rembg" `
    --hidden-import "customtkinter" `
    (Join-Path $ScriptDir "main.py")

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $ScriptDir "build_temp")
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $ScriptDir "pixel-art-smith.spec")

Write-Host "[SUCCESS] PixelArtSmith built successfully to: $BuildRoot\pixel-art-smith.exe" -ForegroundColor Green
