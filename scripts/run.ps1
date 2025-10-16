# ==============================================================================
# Ready4Hire - Script de Inicio Completo (Windows PowerShell)
# ==============================================================================
# Levanta todo el stack: Ollama → FastAPI (DDD v2) → Blazor WebApp
# 
# Uso:
#   .\run.ps1              # Modo normal
#   .\run.ps1 -Dev         # Modo desarrollo (con reload)
#   .\run.ps1 -Stop        # Detener todos los servicios
#   .\run.ps1 -Status      # Ver estado de servicios
#
# Author: Ready4Hire Team
# Version: 2.0.0 (DDD Architecture)
# ==============================================================================

param(
    [switch]$Dev,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Help
)

# Configuración
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IntegrationRoot = Split-Path -Parent $ScriptDir
$Ready4HireDir = Join-Path $IntegrationRoot "Ready4Hire"
$WebAppDir = Join-Path $IntegrationRoot "WebApp"
$LogsDir = Join-Path $Ready4HireDir "logs"

# Archivos de log
$OllamaLog = Join-Path $LogsDir "ollama.log"
$ApiLog = Join-Path $LogsDir "ready4hire_api.log"
$WebAppLog = Join-Path $LogsDir "webapp.log"

# Variables de configuración
$OllamaModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "ready4hire:latest" }
$ApiHost = if ($env:API_HOST) { $env:API_HOST } else { "0.0.0.0" }
$ApiPort = if ($env:API_PORT) { [int]$env:API_PORT } else { 8001 }
$WebAppPort = if ($env:WEBAPP_PORT) { [int]$env:WEBAPP_PORT } else { 5214 }

# ==============================================================================
# Funciones de Utilidad
# ==============================================================================

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Write-Warning-Custom {
    param([string]$Text)
    Write-Host "⚠ $Text" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Text)
    Write-Host "ℹ $Text" -ForegroundColor Cyan
}

function Test-PortInUse {
    param([int]$Port)
    $connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq $Port }
    return $connections.Count -gt 0
}

function Stop-ProcessOnPort {
    param([int]$Port)
    $connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -eq $Port }
    foreach ($conn in $connections) {
        $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
        if ($process) {
            Write-Info "Deteniendo proceso: $($process.ProcessName) (PID: $($process.Id))"
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

# ==============================================================================
# Funciones de Control de Servicios
# ==============================================================================

function Stop-Services {
    Write-Header "Deteniendo Servicios Ready4Hire"
    
    # Detener Ollama
    $ollamaProcesses = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollamaProcesses) {
        Write-Info "Deteniendo Ollama..."
        $ollamaProcesses | Stop-Process -Force
        Write-Success "Ollama detenido"
    } else {
        Write-Warning-Custom "Ollama no estaba corriendo"
    }
    
    # Detener API
    if (Test-PortInUse -Port $ApiPort) {
        Write-Info "Deteniendo API Python (puerto $ApiPort)..."
        Stop-ProcessOnPort -Port $ApiPort
        Start-Sleep -Seconds 2
        Write-Success "API detenida"
    } else {
        Write-Warning-Custom "API no estaba corriendo en puerto $ApiPort"
    }
    
    # Detener WebApp
    if (Test-PortInUse -Port $WebAppPort) {
        Write-Info "Deteniendo WebApp (puerto $WebAppPort)..."
        Stop-ProcessOnPort -Port $WebAppPort
        Start-Sleep -Seconds 2
        Write-Success "WebApp detenida"
    } else {
        Write-Warning-Custom "WebApp no estaba corriendo en puerto $WebAppPort"
    }
    
    Write-Success "Todos los servicios detenidos"
    exit 0
}

function Show-Status {
    Write-Header "Estado de Servicios Ready4Hire"
    
    # Ollama
    $ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollamaProcess) {
        Write-Success "Ollama: RUNNING (PID: $($ollamaProcess.Id))"
        try {
            & ollama list 2>$null | Select-String -Pattern "NAME|$OllamaModel"
        } catch {}
    } else {
        Write-Error-Custom "Ollama: STOPPED"
    }
    
    # API
    if (Test-PortInUse -Port $ApiPort) {
        Write-Success "API Python: RUNNING (puerto $ApiPort)"
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:$ApiPort/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            $response | ConvertTo-Json -Depth 3
        } catch {
            try {
                $response = Invoke-RestMethod -Uri "http://localhost:$ApiPort/api/v2/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
                $response | ConvertTo-Json -Depth 3
            } catch {
                Write-Warning-Custom "API no responde al health check"
            }
        }
    } else {
        Write-Error-Custom "API Python: STOPPED"
    }
    
    # WebApp
    if (Test-PortInUse -Port $WebAppPort) {
        Write-Success "WebApp: RUNNING (puerto $WebAppPort)"
    } else {
        Write-Error-Custom "WebApp: STOPPED"
    }
    
    exit 0
}

# ==============================================================================
# Funciones de Inicio
# ==============================================================================

function Start-Ollama {
    Write-Header "1/4 - Iniciando Ollama Server"
    
    # Verificar si Ollama está instalado
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        Write-Error-Custom "Ollama no está instalado"
        Write-Info "Descargar desde: https://ollama.com/download/windows"
        exit 1
    }
    
    # Verificar si ya está corriendo
    $ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollamaProcess) {
        Write-Success "Ollama ya está corriendo (PID: $($ollamaProcess.Id))"
    } else {
        Write-Info "Iniciando servidor Ollama..."
        
        # Crear directorio de logs
        if (-not (Test-Path $LogsDir)) {
            New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
        }
        
        # Iniciar Ollama en background
        $ollamaStartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $ollamaStartInfo.FileName = "ollama"
        $ollamaStartInfo.Arguments = "serve"
        $ollamaStartInfo.RedirectStandardOutput = $true
        $ollamaStartInfo.RedirectStandardError = $true
        $ollamaStartInfo.UseShellExecute = $false
        $ollamaStartInfo.CreateNoWindow = $true
        
        $ollamaProc = New-Object System.Diagnostics.Process
        $ollamaProc.StartInfo = $ollamaStartInfo
        $ollamaProc.Start() | Out-Null
        
        Start-Sleep -Seconds 3
        
        if (Get-Process -Id $ollamaProc.Id -ErrorAction SilentlyContinue) {
            Write-Success "Ollama iniciado correctamente (PID: $($ollamaProc.Id))"
        } else {
            Write-Error-Custom "Error al iniciar Ollama. Ver logs: $OllamaLog"
            exit 1
        }
    }
    
    # Verificar/descargar modelo
    Write-Info "Verificando modelo $OllamaModel..."
    $modelList = & ollama list 2>$null
    if ($modelList -match $OllamaModel) {
        Write-Success "Modelo $OllamaModel ya está descargado"
    } else {
        Write-Info "Descargando modelo $OllamaModel (esto puede tardar varios minutos)..."
        & ollama pull $OllamaModel
        Write-Success "Modelo descargado correctamente"
    }
    
    # Test de conectividad
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
        Write-Success "Ollama respondiendo correctamente"
    } catch {
        Write-Warning-Custom "Ollama no responde en el endpoint esperado"
    }
}

function Start-Api {
    Write-Header "2/4 - Iniciando API Python (FastAPI DDD v2)"
    
    # Cambiar al directorio del proyecto
    Set-Location $Ready4HireDir
    
    # Verificar si existe app/main_v2.py
    if (-not (Test-Path "app\main_v2.py")) {
        Write-Error-Custom "app\main_v2.py no encontrado"
        exit 1
    }
    
    # Verificar entorno virtual
    $venvActivated = $false
    $venvPaths = @("venv\Scripts\Activate.ps1", "..\venv\Scripts\Activate.ps1")
    
    foreach ($venvPath in $venvPaths) {
        if (Test-Path $venvPath) {
            try {
                & $venvPath
                $venvActivated = $true
                Write-Success "Virtual environment activado"
                break
            } catch {
                # Continuar intentando
            }
        }
    }
    
    if (-not $venvActivated) {
        Write-Warning-Custom "No se encontró/activó virtual environment, usando Python del sistema"
    }
    
    # Verificar dependencias
    try {
        python -c "import fastapi, uvicorn" 2>$null
        Write-Success "Dependencias verificadas"
    } catch {
        Write-Error-Custom "Dependencias no instaladas"
        Write-Info "Instalar con: pip install fastapi uvicorn"
        Write-Info "O con requirements: pip install -r app\requirements.txt"
        exit 1
    }
    
    # Detener API anterior si existe
    if (Test-PortInUse -Port $ApiPort) {
        Write-Warning-Custom "API ya corriendo en puerto $ApiPort. Deteniendo..."
        Stop-ProcessOnPort -Port $ApiPort
        Start-Sleep -Seconds 2
    }
    
    # Crear directorio de logs
    if (-not (Test-Path $LogsDir)) {
        New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    }
    
    # Iniciar API
    Write-Info "Iniciando API en puerto $ApiPort..."
    
    # Establecer PYTHONPATH
    $env:PYTHONPATH = "$Ready4HireDir;$env:PYTHONPATH"
    
    $apiArgs = @(
        "-m", "uvicorn", "app.main_v2:app",
        "--host", $ApiHost,
        "--port", $ApiPort
    )
    
    if ($Dev) {
        Write-Info "Modo desarrollo: auto-reload activado"
        $apiArgs += "--reload"
    }
    
    # Iniciar proceso en background
    $apiStartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $apiStartInfo.FileName = "python"
    $apiStartInfo.Arguments = $apiArgs -join " "
    $apiStartInfo.RedirectStandardOutput = $true
    $apiStartInfo.RedirectStandardError = $true
    $apiStartInfo.UseShellExecute = $false
    $apiStartInfo.CreateNoWindow = $true
    $apiStartInfo.WorkingDirectory = $Ready4HireDir
    
    $apiProc = New-Object System.Diagnostics.Process
    $apiProc.StartInfo = $apiStartInfo
    $apiProc.Start() | Out-Null
    
    Start-Sleep -Seconds 5
    
    # Verificar que la API inició
    if (Test-PortInUse -Port $ApiPort) {
        Write-Success "API iniciada correctamente (PID: $($apiProc.Id))"
        
        # Health check
        Write-Info "Verificando health endpoint..."
        Start-Sleep -Seconds 3
        
        try {
            $healthResponse = Invoke-RestMethod -Uri "http://localhost:$ApiPort/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
            Write-Success "API respondiendo correctamente"
            $healthResponse | ConvertTo-Json -Depth 3
        } catch {
            try {
                $healthResponse = Invoke-RestMethod -Uri "http://localhost:$ApiPort/api/v2/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
                Write-Success "API respondiendo correctamente"
                $healthResponse | ConvertTo-Json -Depth 3
            } catch {
                Write-Warning-Custom "API puede tardar unos segundos más en cargar"
                Write-Info "Ver logs: Get-Content -Tail 20 -Wait $ApiLog"
            }
        }
    } else {
        Write-Error-Custom "Error al iniciar API. Ver logs: $ApiLog"
        Get-Content -Tail 20 $ApiLog
        exit 1
    }
    
    Set-Location $ScriptDir
}

function Start-WebApp {
    Write-Header "3/4 - Iniciando WebApp (Blazor)"
    
    # Verificar si existe el directorio WebApp
    if (-not (Test-Path $WebAppDir)) {
        Write-Warning-Custom "Directorio WebApp no encontrado. Saltando..."
        return
    }
    
    # Verificar si dotnet está instalado
    $dotnetCmd = Get-Command dotnet -ErrorAction SilentlyContinue
    if (-not $dotnetCmd) {
        Write-Warning-Custom "dotnet no está instalado. Saltando WebApp..."
        Write-Info "Instalar .NET SDK: https://dotnet.microsoft.com/download"
        return
    }
    
    Set-Location $WebAppDir
    
    # Verificar archivo de proyecto
    if (-not (Test-Path "Ready4Hire.csproj")) {
        Write-Error-Custom "Proyecto Blazor no encontrado"
        Set-Location $ScriptDir
        return
    }
    
    # Compilar
    Write-Info "Compilando proyecto Blazor..."
    $buildOutput = & dotnet build 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Compilación exitosa"
    } else {
        Write-Error-Custom "Error en compilación"
        Write-Host $buildOutput
        Set-Location $ScriptDir
        return
    }
    
    # Detener WebApp anterior si existe
    if (Test-PortInUse -Port $WebAppPort) {
        Write-Warning-Custom "WebApp ya corriendo en puerto $WebAppPort. Deteniendo..."
        Stop-ProcessOnPort -Port $WebAppPort
        Start-Sleep -Seconds 2
    }
    
    # Iniciar WebApp
    Write-Info "Iniciando WebApp en puerto $WebAppPort..."
    
    # Crear directorio de logs
    if (-not (Test-Path $LogsDir)) {
        New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    }
    
    $webAppStartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $webAppStartInfo.FileName = "dotnet"
    $webAppStartInfo.Arguments = "run --urls=`"http://localhost:$WebAppPort`""
    $webAppStartInfo.RedirectStandardOutput = $true
    $webAppStartInfo.RedirectStandardError = $true
    $webAppStartInfo.UseShellExecute = $false
    $webAppStartInfo.CreateNoWindow = $true
    $webAppStartInfo.WorkingDirectory = $WebAppDir
    
    $webAppProc = New-Object System.Diagnostics.Process
    $webAppProc.StartInfo = $webAppStartInfo
    $webAppProc.Start() | Out-Null
    
    Start-Sleep -Seconds 5
    
    if (Test-PortInUse -Port $WebAppPort) {
        Write-Success "WebApp iniciada correctamente (PID: $($webAppProc.Id))"
    } else {
        Write-Warning-Custom "WebApp puede estar tardando en iniciar. Ver logs: $WebAppLog"
    }
    
    Set-Location $ScriptDir
}

function Show-Summary {
    Write-Header "4/4 - Ready4Hire Iniciado Correctamente"
    
    Write-Host ""
    Write-Host "🚀 Servicios Ready4Hire en Ejecución:" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Ollama LLM" -ForegroundColor Cyan
    Write-Host "    └─ URL: http://localhost:11434"
    Write-Host "    └─ Modelo: $OllamaModel"
    Write-Host "    └─ Log: $OllamaLog"
    Write-Host ""
    Write-Host "  API REST (FastAPI v2 - DDD)" -ForegroundColor Cyan
    Write-Host "    └─ URL: http://localhost:$ApiPort"
    Write-Host "    └─ Docs: http://localhost:$ApiPort/docs"
    Write-Host "    └─ ReDoc: http://localhost:$ApiPort/redoc"
    Write-Host "    └─ Health: http://localhost:$ApiPort/health"
    Write-Host "    └─ Log: $ApiLog"
    Write-Host ""
    
    if (Test-PortInUse -Port $WebAppPort) {
        Write-Host "  WebApp (Blazor)" -ForegroundColor Cyan
        Write-Host "    └─ URL: http://localhost:$WebAppPort"
        Write-Host "    └─ Log: $WebAppLog"
        Write-Host ""
    }
    
    Write-Host "📝 Comandos útiles:" -ForegroundColor Yellow
    Write-Host "  Ver logs en tiempo real:"
    Write-Host "    Get-Content -Tail 20 -Wait $ApiLog" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Detener servicios:"
    Write-Host "    .\run.ps1 -Stop" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Ver estado:"
    Write-Host "    .\run.ps1 -Status" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Health check API:"
    Write-Host "    Invoke-RestMethod -Uri http://localhost:$ApiPort/health" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Abrir en navegador:"
    Write-Host "    Start-Process http://localhost:$ApiPort" -ForegroundColor Green
    Write-Host ""
    
    Write-Success "¡Sistema Ready4Hire listo para usar!"
}

# ==============================================================================
# Main
# ==============================================================================

function Main {
    # Manejar parámetros
    if ($Help) {
        Write-Host "Ready4Hire - Script de Inicio (Windows PowerShell)"
        Write-Host ""
        Write-Host "Uso: .\run.ps1 [OPCIÓN]"
        Write-Host ""
        Write-Host "Opciones:"
        Write-Host "  (sin opción)   Iniciar todos los servicios"
        Write-Host "  -Dev           Modo desarrollo (con auto-reload)"
        Write-Host "  -Stop          Detener todos los servicios"
        Write-Host "  -Status        Ver estado de servicios"
        Write-Host "  -Help          Mostrar esta ayuda"
        Write-Host ""
        exit 0
    }
    
    if ($Stop) {
        Stop-Services
        return
    }
    
    if ($Status) {
        Show-Status
        return
    }
    
    # Banner
    Write-Host ""
    Write-Host "  ____                _       _  _   _   _  _          " -ForegroundColor Cyan
    Write-Host " |  _ \ ___  __ _  __| |_   _| || | | | | |(_)_ __ ___ " -ForegroundColor Cyan
    Write-Host " | |_) / _ \/ _\` |/ _\` | | | | || |_| |_| || | '__/ _ \" -ForegroundColor Cyan
    Write-Host " |  _ <  __/ (_| | (_| | |_| |__   _|  _  || | | |  __/" -ForegroundColor Cyan
    Write-Host " |_| \_\___|\__,_|\__,_|\__, |  |_| |_| |_||_|_|  \___|" -ForegroundColor Cyan
    Write-Host "                        |___/                          " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Sistema de Entrevistas Técnicas con IA" -ForegroundColor Green
    Write-Host "Version 2.0.0 - DDD Architecture" -ForegroundColor Yellow
    Write-Host ""
    
    # Ejecutar servicios
    Start-Ollama
    Start-Api
    Start-WebApp
    Show-Summary
}

# Ejecutar
Main
