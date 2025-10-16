@echo off
setlocal enabledelayedexpansion

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
REM Version: 2.2.0 (DDD Architecture + Auto-install Ollama)
REM ==============================================================================

REM Configuracion de directorios
set "SCRIPT_DIR=%~dp0"
set "INTEGRATION_ROOT=%SCRIPT_DIR%.."
set "READY4HIRE_DIR=%INTEGRATION_ROOT%\Ready4Hire"
set "WEBAPP_DIR=%INTEGRATION_ROOT%\WebApp"
set "LOGS_DIR=%READY4HIRE_DIR%\logs"

REM Archivos de log
set "OLLAMA_LOG=%LOGS_DIR%\ollama.log"
set "FASTAPI_LOG=%LOGS_DIR%\fastapi.log"
set "BLAZOR_LOG=%LOGS_DIR%\blazor.log"

REM Configuracion de puertos
set "API_PORT=8000"
set "WEBAPP_PORT=5000"
set "OLLAMA_PORT=11434"

REM Modelo de Ollama
set "OLLAMA_MODEL="

REM Colores ANSI
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "CYAN=[36m"
set "NC=[0m"

REM ==============================================================================
REM Funciones auxiliares
REM ==============================================================================

goto :main

:print_header
echo.
echo %CYAN%============================================================%NC%
echo %CYAN%  %~1%NC%
echo %CYAN%============================================================%NC%
echo.
exit /b 0

:print_success
echo %GREEN%√ %~1%NC%
exit /b 0

:print_error
echo %RED%X %~1%NC%
exit /b 0

:print_warning
echo %YELLOW%! %~1%NC%
exit /b 0

:print_info
echo %CYAN%i %~1%NC%
exit /b 0

REM ==============================================================================
REM Verificar directorios
REM ==============================================================================

:check_directories
if not exist "%READY4HIRE_DIR%" (
    call :print_error "Directorio Ready4Hire no encontrado: %READY4HIRE_DIR%"
    exit /b 1
)

if not exist "%WEBAPP_DIR%" (
    call :print_error "Directorio WebApp no encontrado: %WEBAPP_DIR%"
    exit /b 1
)

if not exist "%LOGS_DIR%" (
    mkdir "%LOGS_DIR%"
    call :print_info "Directorio de logs creado: %LOGS_DIR%"
)

exit /b 0

REM ==============================================================================
REM Detectar mejor modelo disponible
REM ==============================================================================

:detect_best_model
call :print_info "Detectando mejor modelo disponible..."

REM Lista de modelos en orden de prioridad
set MODELS=ready4hire:latest llama3.2:3b llama3:latest llama3

for %%M in (%MODELS%) do (
    ollama list | findstr /C:"%%M" >nul 2>&1
    if !errorlevel! equ 0 (
        set "OLLAMA_MODEL=%%M"
        call :print_success "Usando modelo: %%M"
        goto :verify_model
    )
)

REM Si ninguno existe, usar el primero disponible
call :print_warning "Ningun modelo prioritario encontrado"
call :print_info "Buscando cualquier modelo disponible..."

for /f "skip=1 tokens=1" %%M in ('ollama list 2^>nul') do (
    set "OLLAMA_MODEL=%%M"
    call :print_success "Usando modelo: %%M"
    goto :verify_model
)

REM No hay modelos instalados
call :print_error "No hay modelos de Ollama instalados"
echo.
call :print_info "Descarga uno con:"
echo   ollama pull llama3.2:3b    (recomendado, ~2GB)
echo   ollama pull llama3:latest  (mas grande, ~4.7GB)
echo.
exit /b 1

:verify_model
call :print_info "Verificando modelo %OLLAMA_MODEL%..."
ollama list | findstr /C:"%OLLAMA_MODEL%" >nul 2>&1
if !errorlevel! equ 0 (
    call :print_success "Modelo %OLLAMA_MODEL% verificado"
    exit /b 0
) else (
    call :print_error "Modelo %OLLAMA_MODEL% no encontrado"
    exit /b 1
)

REM ==============================================================================
REM Paso 1: Iniciar Ollama Server
REM ==============================================================================

:start_ollama
call :print_header "1/4 - Iniciando Ollama Server"

REM Verificar si Ollama esta instalado
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    call :print_warning "Ollama no esta instalado"
    call :print_info "Instalando Ollama automaticamente..."
    
    REM Descargar instalador con PowerShell
    set "INSTALLER_URL=https://ollama.com/download/OllamaSetup.exe"
    set "INSTALLER_PATH=%TEMP%\OllamaSetup.exe"
    
    call :print_info "Descargando Ollama..."
    powershell -Command "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_PATH%'" 2>nul
    
    if !errorlevel! neq 0 (
        call :print_error "Error al descargar Ollama"
        call :print_info "Por favor instala manualmente desde:"
        echo   https://ollama.com/download/windows
        exit /b 1
    )
    
    call :print_success "Descarga completada"
    
    REM Ejecutar instalador
    call :print_info "Ejecutando instalador..."
    call :print_warning "Acepta los permisos de administrador"
    
    start /wait "" "%INSTALLER_PATH%" /S
    
    if !errorlevel! neq 0 (
        call :print_error "Error durante la instalacion"
        del "%INSTALLER_PATH%" >nul 2>&1
        exit /b 1
    )
    
    REM Esperar instalacion
    timeout /t 5 /nobreak >nul
    
    REM Limpiar
    del "%INSTALLER_PATH%" >nul 2>&1
    
    REM Agregar al PATH
    set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
    
    REM Verificar instalacion
    where ollama >nul 2>&1
    if !errorlevel! neq 0 (
        call :print_error "Ollama no se instalo correctamente"
        call :print_info "Reinicia tu terminal o instala manualmente"
        exit /b 1
    )
    
    call :print_success "Ollama instalado correctamente"
)

REM Verificar si Ollama ya esta corriendo
curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Ollama ya esta corriendo"
    goto :detect_model
)

REM Iniciar Ollama
call :print_info "Iniciando servidor Ollama..."
start /B cmd /c "ollama serve > "%OLLAMA_LOG%" 2>&1"

REM Esperar a que Ollama este listo
set MAX_WAIT=30
set WAIT_COUNT=0

:wait_ollama
timeout /t 1 /nobreak >nul
curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Ollama iniciado correctamente"
    goto :detect_model
)

set /a WAIT_COUNT+=1
if %WAIT_COUNT% lss %MAX_WAIT% goto :wait_ollama

call :print_error "Timeout esperando a Ollama"
exit /b 1

:detect_model
call :detect_best_model
if %errorlevel% neq 0 exit /b 1

curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Ollama respondiendo correctamente"
) else (
    call :print_error "Ollama no responde"
    exit /b 1
)

exit /b 0

REM ==============================================================================
REM Paso 2: Iniciar FastAPI Backend
REM ==============================================================================

:start_fastapi
call :print_header "2/4 - Iniciando Backend FastAPI"

cd /d "%READY4HIRE_DIR%"

REM Verificar entorno virtual
if not exist "venv" (
    call :print_warning "Entorno virtual no encontrado"
    call :print_info "Creando entorno virtual..."
    python -m venv venv
    if !errorlevel! neq 0 (
        call :print_error "Error al crear entorno virtual"
        exit /b 1
    )
    call :print_success "Entorno virtual creado"
)

REM Activar entorno virtual
call :print_info "Activando entorno virtual..."
call venv\Scripts\activate.bat

REM Instalar dependencias
call :print_info "Verificando dependencias..."
pip install -q -r requirements.txt 2>nul

REM Verificar si FastAPI ya esta corriendo
curl -s http://localhost:%API_PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "FastAPI ya esta corriendo"
    exit /b 0
)

REM Iniciar FastAPI
call :print_info "Iniciando FastAPI en puerto %API_PORT%..."
start /B cmd /c "venv\Scripts\python.exe -m uvicorn app.main_v2:app --host 0.0.0.0 --port %API_PORT% --reload > "%FASTAPI_LOG%" 2>&1"

REM Esperar a que FastAPI este listo
set MAX_WAIT=30
set WAIT_COUNT=0

:wait_fastapi
timeout /t 1 /nobreak >nul
curl -s http://localhost:%API_PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "FastAPI iniciado correctamente"
    call :print_info "API: http://localhost:%API_PORT%"
    call :print_info "Docs: http://localhost:%API_PORT%/docs"
    exit /b 0
)

set /a WAIT_COUNT+=1
if %WAIT_COUNT% lss %MAX_WAIT% goto :wait_fastapi

call :print_error "Timeout esperando a FastAPI"
exit /b 1

REM ==============================================================================
REM Paso 3: Iniciar Blazor WebApp
REM ==============================================================================

:start_blazor
call :print_header "3/4 - Iniciando Blazor WebApp"

cd /d "%WEBAPP_DIR%"

REM Verificar dotnet
where dotnet >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error ".NET SDK no esta instalado"
    call :print_info "Descarga desde: https://dotnet.microsoft.com/download"
    exit /b 1
)

REM Verificar si Blazor ya esta corriendo
curl -s http://localhost:%WEBAPP_PORT% >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Blazor ya esta corriendo"
    exit /b 0
)

REM Iniciar Blazor
call :print_info "Iniciando Blazor en puerto %WEBAPP_PORT%..."
start /B cmd /c "dotnet run --urls=http://localhost:%WEBAPP_PORT% > "%BLAZOR_LOG%" 2>&1"

REM Esperar a que Blazor este listo
set MAX_WAIT=60
set WAIT_COUNT=0

:wait_blazor
timeout /t 1 /nobreak >nul
curl -s http://localhost:%WEBAPP_PORT% >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Blazor iniciado correctamente"
    call :print_info "WebApp: http://localhost:%WEBAPP_PORT%"
    exit /b 0
)

set /a WAIT_COUNT+=1
if %WAIT_COUNT% lss %MAX_WAIT% goto :wait_blazor

call :print_error "Timeout esperando a Blazor"
exit /b 1

REM ==============================================================================
REM Resumen Final
REM ==============================================================================

:show_summary
call :print_header "4/4 - Stack Ready4Hire Iniciado"

echo %GREEN%Todos los servicios estan corriendo:%NC%
echo.
echo   %CYAN%Ollama Server:%NC%      http://localhost:%OLLAMA_PORT%
echo   %CYAN%Modelo activo:%NC%      %OLLAMA_MODEL%
echo   %CYAN%FastAPI API:%NC%        http://localhost:%API_PORT%
echo   %CYAN%API Docs:%NC%           http://localhost:%API_PORT%/docs
echo   %CYAN%Blazor WebApp:%NC%      http://localhost:%WEBAPP_PORT%
echo.
echo %YELLOW%Logs disponibles en:%NC%
echo   - %OLLAMA_LOG%
echo   - %FASTAPI_LOG%
echo   - %BLAZOR_LOG%
echo.
echo %CYAN%Para detener:%NC%
echo   run.bat stop
echo.
echo %GREEN%Presiona Ctrl+C para detener%NC%
echo.

pause
exit /b 0

REM ==============================================================================
REM Comando: Detener servicios
REM ==============================================================================

:stop_services
call :print_header "Deteniendo servicios Ready4Hire"

REM Detener FastAPI
call :print_info "Deteniendo FastAPI..."
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :%API_PORT%') do (
    taskkill /PID %%P /F >nul 2>&1
)
call :print_success "FastAPI detenido"

REM Detener Blazor
call :print_info "Deteniendo Blazor..."
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :%WEBAPP_PORT%') do (
    taskkill /PID %%P /F >nul 2>&1
)
call :print_success "Blazor detenido"

REM Detener Ollama
call :print_info "Deteniendo Ollama..."
taskkill /IM ollama.exe /F >nul 2>&1
call :print_success "Ollama detenido"

call :print_success "Todos los servicios detenidos"
exit /b 0

REM ==============================================================================
REM Comando: Ver estado
REM ==============================================================================

:check_status
call :print_header "Estado de servicios Ready4Hire"

REM Ollama
curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%√ Ollama:%NC% Corriendo en puerto %OLLAMA_PORT%
) else (
    echo %RED%X Ollama:%NC% No disponible
)

REM FastAPI
curl -s http://localhost:%API_PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%√ FastAPI:%NC% Corriendo en puerto %API_PORT%
) else (
    echo %RED%X FastAPI:%NC% No disponible
)

REM Blazor
curl -s http://localhost:%WEBAPP_PORT% >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%√ Blazor:%NC% Corriendo en puerto %WEBAPP_PORT%
) else (
    echo %RED%X Blazor:%NC% No disponible
)

echo.
exit /b 0

REM ==============================================================================
REM Main - Punto de entrada
REM ==============================================================================

:main
REM Parsear argumentos
if "%1"=="stop" goto :stop_services
if "%1"=="status" goto :check_status

REM Verificar directorios
call :check_directories
if %errorlevel% neq 0 exit /b 1

REM Iniciar servicios
call :start_ollama
if %errorlevel% neq 0 (
    call :print_error "Error al iniciar Ollama"
    exit /b 1
)

call :start_fastapi
if %errorlevel% neq 0 (
    call :print_error "Error al iniciar FastAPI"
    exit /b 1
)

call :start_blazor
if %errorlevel% neq 0 (
    call :print_error "Error al iniciar Blazor"
    exit /b 1
)

REM Mostrar resumen
call :show_summary

exit /b 0
