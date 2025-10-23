# Ready4Hire Management Script for Windows
# PowerShell script to manage all Ready4Hire services
# Author: Ready4Hire Team
# Version: 1.0.0

# Set error action preference
$ErrorActionPreference = "Continue"

# Configuration
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOGS_DIR = Join-Path $SCRIPT_DIR "logs"
$BACKEND_DIR = Join-Path $SCRIPT_DIR "Ready4Hire"
$FRONTEND_DIR = Join-Path $SCRIPT_DIR "WebApp"

# PID files
$BACKEND_PID_FILE = Join-Path $SCRIPT_DIR "backend.pid"
$FRONTEND_PID_FILE = Join-Path $SCRIPT_DIR "frontend.pid"

# Colors
function Write-Success { 
    param($Message)
    Write-Host "[✓] $Message" -ForegroundColor Green 
}

function Write-Error-Custom { 
    param($Message)
    Write-Host "[✗] $Message" -ForegroundColor Red 
}

function Write-Info { 
    param($Message)
    Write-Host "[ℹ] $Message" -ForegroundColor Cyan 
}

function Write-Warning-Custom { 
    param($Message)
    Write-Host "[⚠] $Message" -ForegroundColor Yellow 
}

# Create logs directory
function Initialize-Logs {
    if (-not (Test-Path $LOGS_DIR)) {
        New-Item -ItemType Directory -Path $LOGS_DIR -Force | Out-Null
        Write-Info "Directorio de logs creado: $LOGS_DIR"
    }
}

# Check if PostgreSQL is running
function Test-PostgreSQL {
    Write-Info "Verificando PostgreSQL..."
    
    try {
        $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
        if ($pgService -and $pgService.Status -eq "Running") {
            Write-Success "PostgreSQL está activo"
            return $true
        }
        else {
            Write-Warning-Custom "PostgreSQL no está activo"
            Write-Info "Intenta iniciar PostgreSQL con: net start postgresql-x64-16"
            return $false
        }
    }
    catch {
        Write-Warning-Custom "No se pudo verificar PostgreSQL"
        return $false
    }
}

# Check if Ollama is running
function Test-Ollama {
    Write-Info "Verificando Ollama..."
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "Ollama está activo (puerto 11434)"
            return $true
        }
    }
    catch {
        Write-Warning-Custom "Ollama no está activo o no responde"
        Write-Info "Inicia Ollama desde: Start Menu > Ollama"
        return $false
    }
    return $false
}

# Start Backend
function Start-Backend {
    Write-Info "Iniciando Backend (FastAPI)..."
    
    if (Test-Path $BACKEND_PID_FILE) {
        $pid = Get-Content $BACKEND_PID_FILE
        if (Get-Process -Id $pid -ErrorAction SilentlyContinue) {
            Write-Warning-Custom "Backend ya está corriendo (PID: $pid)"
            return
        }
    }
    
    # Change to backend directory
    Push-Location $BACKEND_DIR
    
    # Activate virtual environment if exists
    $venvPath = Join-Path (Split-Path -Parent $BACKEND_DIR) "venv\Scripts\Activate.ps1"
    if (Test-Path $venvPath) {
        Write-Info "Activando entorno virtual..."
        & $venvPath
    }
    
    # Start backend process
    $logFile = Join-Path $LOGS_DIR "ready4hire_api.log"
    $process = Start-Process -FilePath "python" `
        -ArgumentList "-m", "uvicorn", "app.main_v2_improved:app", "--host", "0.0.0.0", "--port", "8001" `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $logFile `
        -PassThru `
        -NoNewWindow
    
    $process.Id | Out-File $BACKEND_PID_FILE
    
    Pop-Location
    
    Write-Info "Esperando a que el backend inicie (warm-up de modelos)..."
    Start-Sleep -Seconds 12
    
    # Check health
    $retries = 0
    $maxRetries = 5
    while ($retries -lt $maxRetries) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8001/api/v2/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "Backend iniciado (puerto 8001, PID: $($process.Id))"
                return
            }
        }
        catch {
            $retries++
            if ($retries -lt $maxRetries) {
                Write-Info "Reintentando health check ($retries/$maxRetries)..."
                Start-Sleep -Seconds 3
            }
        }
    }
    
    Write-Warning-Custom "Backend tardó en responder. Verifica logs: $logFile"
}

# Start Frontend
function Start-Frontend {
    Write-Info "Iniciando Frontend (Blazor)..."
    
    if (Test-Path $FRONTEND_PID_FILE) {
        $pid = Get-Content $FRONTEND_PID_FILE
        if (Get-Process -Id $pid -ErrorAction SilentlyContinue) {
            Write-Warning-Custom "Frontend ya está corriendo (PID: $pid)"
            return
        }
    }
    
    # Change to frontend directory
    Push-Location $FRONTEND_DIR
    
    # Start frontend process
    $logFile = Join-Path $LOGS_DIR "webapp.log"
    $process = Start-Process -FilePath "dotnet" `
        -ArgumentList "run" `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError $logFile `
        -PassThru `
        -NoNewWindow
    
    $process.Id | Out-File $FRONTEND_PID_FILE
    
    Pop-Location
    
    Write-Info "Esperando a que el frontend inicie..."
    Start-Sleep -Seconds 15
    
    # Check if running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5214" -Method GET -TimeoutSec 5 -ErrorAction Stop
        Write-Success "Frontend iniciado (puerto 5214, PID: $($process.Id))"
    }
    catch {
        Write-Warning-Custom "Frontend aún iniciando. Verifica logs: $logFile"
    }
}

# Stop Backend
function Stop-Backend {
    Write-Info "Deteniendo Backend..."
    
    if (Test-Path $BACKEND_PID_FILE) {
        $pid = Get-Content $BACKEND_PID_FILE
        try {
            $process = Get-Process -Id $pid -ErrorAction Stop
            Stop-Process -Id $pid -Force
            Remove-Item $BACKEND_PID_FILE -Force
            Write-Success "Backend detenido (PID: $pid)"
        }
        catch {
            Write-Warning-Custom "Backend no está corriendo"
            if (Test-Path $BACKEND_PID_FILE) {
                Remove-Item $BACKEND_PID_FILE -Force
            }
        }
    }
    else {
        # Try to kill by name
        $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*" }
        if ($processes) {
            $processes | Stop-Process -Force
            Write-Success "Backend detenido"
        }
        else {
            Write-Warning-Custom "Backend no está corriendo"
        }
    }
}

# Stop Frontend
function Stop-Frontend {
    Write-Info "Deteniendo Frontend..."
    
    if (Test-Path $FRONTEND_PID_FILE) {
        $pid = Get-Content $FRONTEND_PID_FILE
        try {
            $process = Get-Process -Id $pid -ErrorAction Stop
            Stop-Process -Id $pid -Force
            Remove-Item $FRONTEND_PID_FILE -Force
            Write-Success "Frontend detenido (PID: $pid)"
        }
        catch {
            Write-Warning-Custom "Frontend no está corriendo"
            if (Test-Path $FRONTEND_PID_FILE) {
                Remove-Item $FRONTEND_PID_FILE -Force
            }
        }
    }
    else {
        # Try to kill by name
        $processes = Get-Process -Name "dotnet" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*WebApp*" }
        if ($processes) {
            $processes | Stop-Process -Force
            Write-Success "Frontend detenido"
        }
        else {
            Write-Warning-Custom "Frontend no está corriendo"
        }
    }
}

# Start all services
function Start-All {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "Iniciando todos los servicios de Ready4Hire" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    
    Initialize-Logs
    
    # Check dependencies
    $pgOk = Test-PostgreSQL
    $ollamaOk = Test-Ollama
    
    if (-not $pgOk -or -not $ollamaOk) {
        Write-Warning-Custom "Algunos servicios no están disponibles. El sistema puede no funcionar correctamente."
        $response = Read-Host "¿Deseas continuar de todos modos? (s/n)"
        if ($response -ne "s" -and $response -ne "S") {
            Write-Info "Operación cancelada"
            return
        }
    }
    
    # Start services
    Start-Backend
    Start-Frontend
    
    Write-Host ""
    Write-Success "✓ Todos los servicios iniciados"
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host "URLs de acceso:" -ForegroundColor Green
    Write-Host "  🌐 Frontend:  http://localhost:5214" -ForegroundColor White
    Write-Host "  🔧 Backend:   http://localhost:8001" -ForegroundColor White
    Write-Host "  📚 API Docs:  http://localhost:8001/docs" -ForegroundColor White
    Write-Host "  ❤️  Health:    http://localhost:8001/api/v2/health" -ForegroundColor White
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host ""
}

# Stop all services
function Stop-All {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    Write-Host "Deteniendo todos los servicios" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    Write-Host ""
    
    Stop-Backend
    Stop-Frontend
    
    Write-Host ""
    Write-Success "✓ Todos los servicios detenidos"
    Write-Host ""
}

# Restart all services
function Restart-All {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
    Write-Host "Reiniciando todos los servicios" -ForegroundColor Magenta
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
    Write-Host ""
    
    Stop-All
    Start-Sleep -Seconds 2
    Start-All
}

# Show status
function Show-Status {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "📊 Estado del Sistema" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    
    # PostgreSQL
    try {
        $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
        if ($pgService -and $pgService.Status -eq "Running") {
            Write-Host "✅ PostgreSQL: " -NoNewline -ForegroundColor Green
            Write-Host "Activo (puerto 5432)" -ForegroundColor White
        }
        else {
            Write-Host "❌ PostgreSQL: " -NoNewline -ForegroundColor Red
            Write-Host "Inactivo" -ForegroundColor White
        }
    }
    catch {
        Write-Host "❌ PostgreSQL: " -NoNewline -ForegroundColor Red
        Write-Host "Estado desconocido" -ForegroundColor White
    }
    
    # Ollama
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 2 -ErrorAction Stop
        Write-Host "✅ Ollama: " -NoNewline -ForegroundColor Green
        Write-Host "Activo (puerto 11434)" -ForegroundColor White
    }
    catch {
        Write-Host "❌ Ollama: " -NoNewline -ForegroundColor Red
        Write-Host "Inactivo" -ForegroundColor White
    }
    
    # Backend
    if (Test-Path $BACKEND_PID_FILE) {
        $pid = Get-Content $BACKEND_PID_FILE
        if (Get-Process -Id $pid -ErrorAction SilentlyContinue) {
            Write-Host "✅ Backend: " -NoNewline -ForegroundColor Green
            Write-Host "Activo (puerto 8001, PID: $pid)" -ForegroundColor White
        }
        else {
            Write-Host "❌ Backend: " -NoNewline -ForegroundColor Red
            Write-Host "Inactivo" -ForegroundColor White
        }
    }
    else {
        Write-Host "❌ Backend: " -NoNewline -ForegroundColor Red
        Write-Host "Inactivo" -ForegroundColor White
    }
    
    # Frontend
    if (Test-Path $FRONTEND_PID_FILE) {
        $pid = Get-Content $FRONTEND_PID_FILE
        if (Get-Process -Id $pid -ErrorAction SilentlyContinue) {
            Write-Host "✅ Frontend: " -NoNewline -ForegroundColor Green
            Write-Host "Activo (puerto 5214, PID: $pid)" -ForegroundColor White
        }
        else {
            Write-Host "❌ Frontend: " -NoNewline -ForegroundColor Red
            Write-Host "Inactivo" -ForegroundColor White
        }
    }
    else {
        Write-Host "❌ Frontend: " -NoNewline -ForegroundColor Red
        Write-Host "Inactivo" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "ℹ️  URLs de Acceso:" -ForegroundColor Cyan
    Write-Host "   🌐 Frontend:  http://localhost:5214" -ForegroundColor White
    Write-Host "   🔧 Backend:   http://localhost:8001" -ForegroundColor White
    Write-Host "   📚 API Docs:  http://localhost:8001/docs" -ForegroundColor White
    Write-Host "   ❤️  Health:    http://localhost:8001/api/v2/health" -ForegroundColor White
    Write-Host ""
}

# Show logs
function Show-Logs {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "📋 Logs de Ready4Hire" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Backend logs"
    Write-Host "2. Frontend logs"
    Write-Host "3. Ver ambos"
    Write-Host "4. Volver"
    Write-Host ""
    
    $choice = Read-Host "Selecciona una opción"
    
    switch ($choice) {
        "1" {
            $logFile = Join-Path $LOGS_DIR "ready4hire_api.log"
            if (Test-Path $logFile) {
                Get-Content $logFile -Tail 50
            }
            else {
                Write-Warning-Custom "No hay logs de backend"
            }
        }
        "2" {
            $logFile = Join-Path $LOGS_DIR "webapp.log"
            if (Test-Path $logFile) {
                Get-Content $logFile -Tail 50
            }
            else {
                Write-Warning-Custom "No hay logs de frontend"
            }
        }
        "3" {
            Write-Host "`n=== BACKEND ===" -ForegroundColor Yellow
            $backendLog = Join-Path $LOGS_DIR "ready4hire_api.log"
            if (Test-Path $backendLog) {
                Get-Content $backendLog -Tail 25
            }
            
            Write-Host "`n=== FRONTEND ===" -ForegroundColor Yellow
            $frontendLog = Join-Path $LOGS_DIR "webapp.log"
            if (Test-Path $frontendLog) {
                Get-Content $frontendLog -Tail 25
            }
        }
    }
    
    Write-Host ""
    Read-Host "Presiona Enter para continuar"
}

# Check dependencies
function Check-Dependencies {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "🔍 Verificando Dependencias" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✅ Python: " -NoNewline -ForegroundColor Green
        Write-Host "$pythonVersion" -ForegroundColor White
    }
    catch {
        Write-Host "❌ Python: " -NoNewline -ForegroundColor Red
        Write-Host "No encontrado" -ForegroundColor White
    }
    
    # Check .NET
    try {
        $dotnetVersion = dotnet --version 2>&1
        Write-Host "✅ .NET: " -NoNewline -ForegroundColor Green
        Write-Host "$dotnetVersion" -ForegroundColor White
    }
    catch {
        Write-Host "❌ .NET: " -NoNewline -ForegroundColor Red
        Write-Host "No encontrado" -ForegroundColor White
    }
    
    # Check PostgreSQL
    Test-PostgreSQL | Out-Null
    
    # Check Ollama
    Test-Ollama | Out-Null
    
    Write-Host ""
    Read-Host "Presiona Enter para continuar"
}

# Clean logs
function Clean-Logs {
    Write-Host ""
    Write-Warning-Custom "¿Estás seguro de que deseas limpiar todos los logs? (s/n)"
    $response = Read-Host
    
    if ($response -eq "s" -or $response -eq "S") {
        if (Test-Path $LOGS_DIR) {
            Get-ChildItem $LOGS_DIR -Filter *.log | Remove-Item -Force
            Write-Success "Logs limpiados"
        }
        else {
            Write-Info "No hay logs para limpiar"
        }
    }
    else {
        Write-Info "Operación cancelada"
    }
    
    Write-Host ""
    Read-Host "Presiona Enter para continuar"
}

# Show menu
function Show-Menu {
    Clear-Host
    Write-Host ""
    Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║          Ready4Hire - Sistema de Gestión                 ║" -ForegroundColor Cyan
    Write-Host "║                   Windows Version                         ║" -ForegroundColor Cyan
    Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1. 🚀 Iniciar todos los servicios" -ForegroundColor Green
    Write-Host "  2. ⏹️  Detener todos los servicios" -ForegroundColor Yellow
    Write-Host "  3. 🔄 Reiniciar todos los servicios" -ForegroundColor Magenta
    Write-Host "  4. 📊 Ver estado del sistema" -ForegroundColor Cyan
    Write-Host "  5. 📋 Ver logs" -ForegroundColor White
    Write-Host "  6. 🔍 Verificar dependencias" -ForegroundColor Blue
    Write-Host "  7. 🧹 Limpiar logs" -ForegroundColor DarkGray
    Write-Host "  8. ❌ Salir" -ForegroundColor Red
    Write-Host ""
}

# Main menu loop
function Start-InteractiveMenu {
    while ($true) {
        Show-Menu
        $choice = Read-Host "Selecciona una opción (1-8)"
        
        switch ($choice) {
            "1" { Start-All; Read-Host "`nPresiona Enter para continuar" }
            "2" { Stop-All; Read-Host "`nPresiona Enter para continuar" }
            "3" { Restart-All; Read-Host "`nPresiona Enter para continuar" }
            "4" { Show-Status; Read-Host "`nPresiona Enter para continuar" }
            "5" { Show-Logs }
            "6" { Check-Dependencies }
            "7" { Clean-Logs }
            "8" {
                Write-Host ""
                Write-Info "¡Hasta luego!"
                Write-Host ""
                exit 0
            }
            default {
                Write-Warning-Custom "Opción inválida. Por favor selecciona 1-8."
                Start-Sleep -Seconds 2
            }
        }
    }
}

# Main entry point
if ($args.Count -eq 0) {
    # Interactive mode
    Start-InteractiveMenu
}
else {
    # Command line mode
    Initialize-Logs
    
    switch ($args[0]) {
        "start" { Start-All }
        "stop" { Stop-All }
        "restart" { Restart-All }
        "status" { Show-Status }
        "logs" { Show-Logs }
        "check" { Check-Dependencies }
        "clean" { Clean-Logs }
        default {
            Write-Host ""
            Write-Host "Ready4Hire Management Script" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Uso:" -ForegroundColor Yellow
            Write-Host "  .\ready4hire.ps1           - Modo interactivo"
            Write-Host "  .\ready4hire.ps1 start     - Iniciar servicios"
            Write-Host "  .\ready4hire.ps1 stop      - Detener servicios"
            Write-Host "  .\ready4hire.ps1 restart   - Reiniciar servicios"
            Write-Host "  .\ready4hire.ps1 status    - Ver estado"
            Write-Host "  .\ready4hire.ps1 logs      - Ver logs"
            Write-Host "  .\ready4hire.ps1 check     - Verificar dependencias"
            Write-Host "  .\ready4hire.ps1 clean     - Limpiar logs"
            Write-Host ""
        }
    }
}

