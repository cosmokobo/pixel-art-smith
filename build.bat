@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ===================================================
echo  PixelArtSmith - Windows PyInstaller Build
echo ===================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1"

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Build completed successfully.
) else (
    echo [ERROR] Build failed with exit code %ERRORLEVEL%.
)
exit /b %ERRORLEVEL%
