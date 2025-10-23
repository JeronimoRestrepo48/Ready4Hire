@echo off
REM Ready4Hire Management Script for Windows (Batch)
REM Simple launcher for PowerShell script
REM Author: Ready4Hire Team

cd /d "%~dp0"

REM Check if PowerShell is available
where powershell >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell no encontrado.
    echo Por favor instala PowerShell para usar este script.
    pause
    exit /b 1
)

REM Run PowerShell script
echo Iniciando Ready4Hire Management System...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0ready4hire.ps1" %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Error al ejecutar el script.
    echo Si ves un error de politica de ejecucion, ejecuta:
    echo    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    echo.
    pause
)

