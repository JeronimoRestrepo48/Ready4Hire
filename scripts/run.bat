@echo off
REM ==============================================================================
REM Ready4Hire - Script de Inicio Completo (Windows Batch)
REM ==============================================================================
REM Levanta todo el stack: Ollama -> FastAPI (DDD v2) -> Blazor WebApp
REM 
REM Uso:
REM   run.bat              # Modo normal
REM   run.bat stop         # Detener todos los servicios
REM   run.bat status       # Ver estado de servicios
REM
REM Author: Ready4Hire Team
REM Version: 2.0.0 (DDD Architecture)
REM ==============================================================================

setlocal enabledelayedexpansion

REM Configuración de directorios
set "SCRIPT_DIR=%~dp0"
set "INTEGRATION_ROOT=%SCRIPT_DIR%.."
set "READY4HIRE_DIR=%INTEGRATION_ROOT%\Ready4Hire"
set "WEBAPP_DIR=%INTEGRATION_ROOT%\WebApp"
set "LOGS_DIR=%READY4HIRE_DIR%\logs"

REM Archivos de log
set "OLLAMA_LOG=%LOGS_DIR%\ollama.log"
set "API_LOG=%LOGS_DIR%\ready4hire_api.log"
set "WEBAPP_LOG=%LOGS_DIR%\webapp.log"

REM Configuración de servicios
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=ready4hire:latest"
if not defined API_HOST set "API_HOST=0.0.0.0"
if not defined API_PORT set "API_PORT=8001"
if not defined WEBAPP_PORT set "WEBAPP_PORT=5214"

REM Colores (usando caracteres especiales)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "NC=[0m"

REM Parsear argumentos
if "%1"=="stop" goto :stop_services
if "%1"=="status" goto :show_status
if "%1"=="help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="/?" goto :show_help

REM ==============================================================================
REM Main - Iniciar servicios
REM ==============================================================================

echo.
echo %CYAN%  ____                _       _  _   _   _  _          %NC%
echo %CYAN% ^|  _ \ ___  __ _  __^| ^|_   _^| ^|^| ^| ^| ^| ^| ^|^(_^)_ __ ___ %NC%
echo %CYAN% ^| ^|_^) / _ \/ _` ^|/ _` ^| ^| ^| ^| ^|^| ^|_^| ^|_^| ^|^| ^| '__/ _ \%NC%
echo %CYAN% ^|  _ ^<  __/ ^(_^| ^| ^(_^| ^| ^|_^| ^|__   _^|  _  ^|^| ^| ^| ^|  __/%NC%
echo %CYAN% ^|_^| \_\___^|\__,_^|\__,_^|\__, ^|  ^|_^| ^|_^| ^|_^|^|_^|_^|  \___^|%NC%
echo %CYAN%                        ^|___/                          %NC%
echo.
echo %GREEN%Sistema de Entrevistas Tecnicas con IA%NC%
echo %YELLOW%Version 2.0.0 - DDD Architecture%NC%
echo.

REM Crear directorio de logs
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

call :start_ollama
call :start_api
call :start_webapp
call :show_summary

goto :eof

REM ==============================================================================
REM Función: Iniciar Ollama
REM ==============================================================================
:start_ollama
echo.
echo %CYAN%============================================================%NC%
echo %CYAN%  1/4 - Iniciando Ollama Server%NC%
echo %CYAN%============================================================%NC%
echo.

REM Verificar si Ollama está instalado
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%X Ollama no esta instalado%NC%
    echo %CYAN%i Descargar desde: https://ollama.com/download/windows%NC%
    exit /b 1
)

REM Verificar si ya está corriendo
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %errorlevel% equ 0 (
    echo %GREEN%√ Ollama ya esta corriendo%NC%
) else (
    echo %CYAN%i Iniciando servidor Ollama...%NC%
    start /B ollama serve > "%OLLAMA_LOG%" 2>&1
    timeout /t 3 /nobreak >nul
    
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
    if !errorlevel! equ 0 (
        echo %GREEN%√ Ollama iniciado correctamente%NC%
    ) else (
        echo %RED%X Error al iniciar Ollama. Ver logs: %OLLAMA_LOG%%NC%
        exit /b 1
    )
)

REM Verificar/descargar modelo
echo %CYAN%i Verificando modelo %OLLAMA_MODEL%...%NC%
ollama list | find /I "%OLLAMA_MODEL%" >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%√ Modelo %OLLAMA_MODEL% ya esta descargado%NC%
) else (
    echo %CYAN%i Descargando modelo %OLLAMA_MODEL% (esto puede tardar varios minutos)...%NC%
    ollama pull %OLLAMA_MODEL%
    echo %GREEN%√ Modelo descargado correctamente%NC%
)

REM Test de conectividad
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%√ Ollama respondiendo correctamente%NC%
) else (
    echo %YELLOW%! Ollama no responde en el endpoint esperado%NC%
)

goto :eof

REM ==============================================================================
REM Función: Iniciar API Python
REM ==============================================================================
:start_api
echo.
echo %CYAN%============================================================%NC%
echo %CYAN%  2/4 - Iniciando API Python (FastAPI DDD v2)%NC%
echo %CYAN%============================================================%NC%
echo.

cd /d "%READY4HIRE_DIR%"

REM Verificar archivo principal
if not exist "app\main_v2.py" (
    echo %RED%X app\main_v2.py no encontrado%NC%
    exit /b 1
)

REM Verificar entorno virtual
set "VENV_ACTIVATED=false"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    set "VENV_ACTIVATED=true"
    echo %GREEN%√ Virtual environment activado%NC%
) else if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
    set "VENV_ACTIVATED=true"
    echo %GREEN%√ Virtual environment activado%NC%
) else (
    echo %YELLOW%! No se encontro virtual environment, usando Python del sistema%NC%
)

REM Verificar dependencias
python -c "import fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo %RED%X Dependencias no instaladas%NC%
    echo %CYAN%i Instalar con: pip install fastapi uvicorn%NC%
    echo %CYAN%i O con requirements: pip install -r app\requirements.txt%NC%
    exit /b 1
)
echo %GREEN%√ Dependencias verificadas%NC%

REM Detener API anterior si existe
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%API_PORT%" ^| find "LISTENING"') do (
    echo %YELLOW%! API ya corriendo en puerto %API_PORT%. Deteniendo...%NC%
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

REM Iniciar API
echo %CYAN%i Iniciando API en puerto %API_PORT%...%NC%
set "PYTHONPATH=%READY4HIRE_DIR%;%PYTHONPATH%"

start /B python -m uvicorn app.main_v2:app --host %API_HOST% --port %API_PORT% > "%API_LOG%" 2>&1

timeout /t 5 /nobreak >nul

REM Verificar inicio
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%API_PORT%" ^| find "LISTENING"') do (
    echo %GREEN%√ API iniciada correctamente (PID: %%a^)%NC%
    
    REM Health check
    echo %CYAN%i Verificando health endpoint...%NC%
    timeout /t 3 /nobreak >nul
    curl -s http://localhost:%API_PORT%/health >nul 2>&1
    if !errorlevel! equ 0 (
        echo %GREEN%√ API respondiendo correctamente%NC%
    ) else (
        curl -s http://localhost:%API_PORT%/api/v2/health >nul 2>&1
        if !errorlevel! equ 0 (
            echo %GREEN%√ API respondiendo correctamente%NC%
        ) else (
            echo %YELLOW%! API puede tardar unos segundos mas en cargar%NC%
            echo %CYAN%i Ver logs: type %API_LOG%%NC%
        )
    )
    goto :api_started
)

echo %RED%X Error al iniciar API. Ver logs: %API_LOG%%NC%
type "%API_LOG%" | more
exit /b 1

:api_started
cd /d "%SCRIPT_DIR%"
goto :eof

REM ==============================================================================
REM Función: Iniciar WebApp
REM ==============================================================================
:start_webapp
echo.
echo %CYAN%============================================================%NC%
echo %CYAN%  3/4 - Iniciando WebApp (Blazor)%NC%
echo %CYAN%============================================================%NC%
echo.

if not exist "%WEBAPP_DIR%" (
    echo %YELLOW%! Directorio WebApp no encontrado. Saltando...%NC%
    goto :eof
)

REM Verificar dotnet
where dotnet >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%! dotnet no esta instalado. Saltando WebApp...%NC%
    echo %CYAN%i Instalar .NET SDK: https://dotnet.microsoft.com/download%NC%
    goto :eof
)

cd /d "%WEBAPP_DIR%"

REM Verificar proyecto
if not exist "Ready4Hire.csproj" (
    echo %RED%X Proyecto Blazor no encontrado%NC%
    cd /d "%SCRIPT_DIR%"
    goto :eof
)

REM Compilar
echo %CYAN%i Compilando proyecto Blazor...%NC%
dotnet build >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%√ Compilacion exitosa%NC%
) else (
    echo %RED%X Error en compilacion%NC%
    dotnet build
    cd /d "%SCRIPT_DIR%"
    goto :eof
)

REM Detener WebApp anterior si existe
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%WEBAPP_PORT%" ^| find "LISTENING"') do (
    echo %YELLOW%! WebApp ya corriendo en puerto %WEBAPP_PORT%. Deteniendo...%NC%
    taskkill /F /PID %%a >nul 2>&1
    timeout /t 2 /nobreak >nul
)

REM Iniciar WebApp
echo %CYAN%i Iniciando WebApp en puerto %WEBAPP_PORT%...%NC%
start /B dotnet run --urls="http://localhost:%WEBAPP_PORT%" > "%WEBAPP_LOG%" 2>&1

timeout /t 5 /nobreak >nul

for /f "tokens=5" %%a in ('netstat -aon ^| find ":%WEBAPP_PORT%" ^| find "LISTENING"') do (
    echo %GREEN%√ WebApp iniciada correctamente (PID: %%a^)%NC%
    goto :webapp_started
)

echo %YELLOW%! WebApp puede estar tardando en iniciar. Ver logs: %WEBAPP_LOG%%NC%

:webapp_started
cd /d "%SCRIPT_DIR%"
goto :eof

REM ==============================================================================
REM Función: Mostrar resumen
REM ==============================================================================
:show_summary
echo.
echo %CYAN%============================================================%NC%
echo %CYAN%  4/4 - Ready4Hire Iniciado Correctamente%NC%
echo %CYAN%============================================================%NC%
echo.
echo %GREEN%Servicios Ready4Hire en Ejecucion:%NC%
echo.
echo   %CYAN%Ollama LLM%NC%
echo     └─ URL: http://localhost:11434
echo     └─ Modelo: %OLLAMA_MODEL%
echo     └─ Log: %OLLAMA_LOG%
echo.
echo   %CYAN%API REST (FastAPI v2 - DDD)%NC%
echo     └─ URL: http://localhost:%API_PORT%
echo     └─ Docs: http://localhost:%API_PORT%/docs
echo     └─ ReDoc: http://localhost:%API_PORT%/redoc
echo     └─ Health: http://localhost:%API_PORT%/health
echo     └─ Log: %API_LOG%
echo.

for /f "tokens=5" %%a in ('netstat -aon ^| find ":%WEBAPP_PORT%" ^| find "LISTENING"') do (
    echo   %CYAN%WebApp (Blazor)%NC%
    echo     └─ URL: http://localhost:%WEBAPP_PORT%
    echo     └─ Log: %WEBAPP_LOG%
    echo.
)

echo %YELLOW%Comandos utiles:%NC%
echo   Ver logs en tiempo real:
echo     %GREEN%powershell Get-Content -Tail 20 -Wait %API_LOG%%NC%
echo.
echo   Detener servicios:
echo     %GREEN%run.bat stop%NC%
echo.
echo   Ver estado:
echo     %GREEN%run.bat status%NC%
echo.
echo   Abrir en navegador:
echo     %GREEN%start http://localhost:%API_PORT%%NC%
echo.
echo %GREEN%√ Sistema Ready4Hire listo para usar!%NC%
echo.
goto :eof

REM ==============================================================================
REM Función: Detener servicios
REM ==============================================================================
:stop_services
echo.
echo %CYAN%============================================================%NC%
echo %CYAN%  Deteniendo Servicios Ready4Hire%NC%
echo %CYAN%============================================================%NC%
echo.

REM Detener Ollama
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %errorlevel% equ 0 (
    echo %CYAN%i Deteniendo Ollama...%NC%
    taskkill /F /IM ollama.exe >nul 2>&1
    echo %GREEN%√ Ollama detenido%NC%
) else (
    echo %YELLOW%! Ollama no estaba corriendo%NC%
)

REM Detener API
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%API_PORT%" ^| find "LISTENING"') do (
    echo %CYAN%i Deteniendo API Python (puerto %API_PORT%)...%NC%
    taskkill /F /PID %%a >nul 2>&1
    echo %GREEN%√ API detenida%NC%
    goto :api_stopped
)
echo %YELLOW%! API no estaba corriendo en puerto %API_PORT%%NC%
:api_stopped

REM Detener WebApp
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%WEBAPP_PORT%" ^| find "LISTENING"') do (
    echo %CYAN%i Deteniendo WebApp (puerto %WEBAPP_PORT%)...%NC%
    taskkill /F /PID %%a >nul 2>&1
    echo %GREEN%√ WebApp detenida%NC%
    goto :webapp_stopped
)
echo %YELLOW%! WebApp no estaba corriendo en puerto %WEBAPP_PORT%%NC%
:webapp_stopped

echo.
echo %GREEN%√ Todos los servicios detenidos%NC%
goto :eof

REM ==============================================================================
REM Función: Mostrar estado
REM ==============================================================================
:show_status
echo.
echo %CYAN%============================================================%NC%
echo %CYAN%  Estado de Servicios Ready4Hire%NC%
echo %CYAN%============================================================%NC%
echo.

REM Ollama
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %errorlevel% equ 0 (
    echo %GREEN%√ Ollama: RUNNING%NC%
    ollama list | find /I "NAME"
    ollama list | find /I "%OLLAMA_MODEL%"
) else (
    echo %RED%X Ollama: STOPPED%NC%
)
echo.

REM API
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%API_PORT%" ^| find "LISTENING"') do (
    echo %GREEN%√ API Python: RUNNING (puerto %API_PORT%, PID: %%a^)%NC%
    curl -s http://localhost:%API_PORT%/health 2>nul
    if !errorlevel! neq 0 (
        curl -s http://localhost:%API_PORT%/api/v2/health 2>nul
        if !errorlevel! neq 0 (
            echo %YELLOW%! API no responde al health check%NC%
        )
    )
    goto :api_status_done
)
echo %RED%X API Python: STOPPED%NC%
:api_status_done
echo.

REM WebApp
for /f "tokens=5" %%a in ('netstat -aon ^| find ":%WEBAPP_PORT%" ^| find "LISTENING"') do (
    echo %GREEN%√ WebApp: RUNNING (puerto %WEBAPP_PORT%, PID: %%a^)%NC%
    goto :webapp_status_done
)
echo %RED%X WebApp: STOPPED%NC%
:webapp_status_done
echo.

goto :eof

REM ==============================================================================
REM Función: Mostrar ayuda
REM ==============================================================================
:show_help
echo Ready4Hire - Script de Inicio (Windows Batch)
echo.
echo Uso: run.bat [OPCION]
echo.
echo Opciones:
echo   (sin opcion)   Iniciar todos los servicios
echo   stop           Detener todos los servicios
echo   status         Ver estado de servicios
echo   help           Mostrar esta ayuda
echo.
goto :eof
