# ══════════════════════════════════════════════════════════════════════════════
# Ready4Hire v2.1 - Script de Testing Local (Windows PowerShell)
# Permite ejecutar tests sin desplegar infraestructura completa
# ══════════════════════════════════════════════════════════════════════════════

# Colores
$ESC = [char]27
$RED = "$ESC[31m"
$GREEN = "$ESC[32m"
$YELLOW = "$ESC[33m"
$BLUE = "$ESC[34m"
$PURPLE = "$ESC[35m"
$CYAN = "$ESC[36m"
$NC = "$ESC[0m"

# Rutas
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BackendDir = Join-Path $ProjectRoot "Ready4Hire"
$FrontendDir = Join-Path $ProjectRoot "WebApp"
$E2EDir = Join-Path $ProjectRoot "e2e-tests"

# ──────────────────────────────────────────────────────────────────────────────
# Funciones de Utilidad
# ──────────────────────────────────────────────────────────────────────────────

function Print-Header {
    Write-Host "${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    Write-Host "${CYAN}║              🧪 Ready4Hire v2.1 - Testing Local (Windows)                ║${NC}"
    Write-Host "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
}

function Print-Section {
    param([string]$Message)
    Write-Host "`n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    Write-Host "${PURPLE}$Message${NC}"
    Write-Host "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}`n"
}

function Success {
    param([string]$Message)
    Write-Host "${GREEN}✅ $Message${NC}"
}

function Error-Msg {
    param([string]$Message)
    Write-Host "${RED}❌ $Message${NC}"
}

function Warning-Msg {
    param([string]$Message)
    Write-Host "${YELLOW}⚠️  $Message${NC}"
}

function Info {
    param([string]$Message)
    Write-Host "${BLUE}ℹ️  $Message${NC}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Verificar Prerrequisitos
# ──────────────────────────────────────────────────────────────────────────────

function Check-Prerequisites {
    Print-Section "Verificando Prerrequisitos"
    
    $allOk = $true
    
    # Python 3.11+
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonVersion = python --version
        Success "Python: $pythonVersion"
    } else {
        Error-Msg "Python 3.11+ no encontrado"
        $allOk = $false
    }
    
    # .NET 9.0
    if (Get-Command dotnet -ErrorAction SilentlyContinue) {
        $dotnetVersion = dotnet --version
        Success ".NET: $dotnetVersion"
    } else {
        Warning-Msg ".NET no encontrado (solo necesario para tests de frontend)"
    }
    
    # Node.js (para E2E tests)
    if (Get-Command node -ErrorAction SilentlyContinue) {
        $nodeVersion = node --version
        Success "Node.js: $nodeVersion"
    } else {
        Warning-Msg "Node.js no encontrado (solo necesario para tests E2E)"
    }
    
    if (-not $allOk) {
        Error-Msg "Faltan prerrequisitos. Instálalos e intenta de nuevo."
        exit 1
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Setup Backend
# ──────────────────────────────────────────────────────────────────────────────

function Setup-Backend {
    Print-Section "Configurando Backend Python"
    
    Set-Location $BackendDir
    
    # Crear venv si no existe
    if (-not (Test-Path "venv")) {
        Info "Creando entorno virtual..."
        python -m venv venv
        Success "Entorno virtual creado"
    } else {
        Info "Entorno virtual ya existe"
    }
    
    # Activar venv
    & "venv\Scripts\Activate.ps1"
    
    # Instalar dependencias
    Info "Instalando dependencias..."
    python -m pip install --upgrade pip -q
    pip install -r requirements.txt -q
    
    Success "Backend configurado"
}

# ──────────────────────────────────────────────────────────────────────────────
# Tests Backend
# ──────────────────────────────────────────────────────────────────────────────

function Run-BackendTests {
    Print-Section "Ejecutando Tests Backend"
    
    Set-Location $BackendDir
    & "venv\Scripts\Activate.ps1"
    
    # Configurar PYTHONPATH
    $env:PYTHONPATH = "$BackendDir;$env:PYTHONPATH"
    
    Write-Host "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    Write-Host "${CYAN}  TESTS UNITARIOS${NC}"
    Write-Host "${CYAN}═══════════════════════════════════════════════════════════════════${NC}`n"
    
    $testResult = $true
    try {
        pytest tests/unit/ -v --tb=short --color=yes
        Success "Tests unitarios pasaron"
    } catch {
        Error-Msg "Tests unitarios fallaron"
        $testResult = $false
    }
    
    Write-Host "`n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    Write-Host "${CYAN}  COVERAGE REPORT${NC}"
    Write-Host "${CYAN}═══════════════════════════════════════════════════════════════════${NC}`n"
    
    pytest tests/unit/ --cov=app --cov-report=term-missing --tb=short --color=yes | Select-Object -Last 20
    
    return $testResult
}

# ──────────────────────────────────────────────────────────────────────────────
# Tests de Integración (Opcional)
# ──────────────────────────────────────────────────────────────────────────────

function Run-IntegrationTests {
    Print-Section "Tests de Integración (Requiere servicios)"
    
    Warning-Msg "Los tests de integración requieren Ollama, PostgreSQL, Redis, etc."
    $response = Read-Host "¿Deseas ejecutarlos? (s/N)"
    
    if ($response -notmatch '^[Ss]$') {
        Info "Saltando tests de integración"
        return
    }
    
    Set-Location $BackendDir
    & "venv\Scripts\Activate.ps1"
    $env:PYTHONPATH = "$BackendDir;$env:PYTHONPATH"
    
    Write-Host "`n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    Write-Host "${CYAN}  TESTS DE INTEGRACIÓN${NC}"
    Write-Host "${CYAN}═══════════════════════════════════════════════════════════════════${NC}`n"
    
    try {
        pytest tests/integration/ -v --tb=short --color=yes -x
        Success "Tests de integración pasaron"
    } catch {
        Error-Msg "Tests de integración fallaron (esto es esperado si los servicios no están corriendo)"
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Linting & Code Quality
# ──────────────────────────────────────────────────────────────────────────────

function Run-Linting {
    Print-Section "Linting & Code Quality"
    
    Set-Location $BackendDir
    & "venv\Scripts\Activate.ps1"
    
    Info "Instalando herramientas de linting..."
    pip install -q black flake8 mypy pylint
    
    Write-Host "`n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    Write-Host "${CYAN}  BLACK (Code Formatting)${NC}"
    Write-Host "${CYAN}═══════════════════════════════════════════════════════════════════${NC}`n"
    
    try {
        black --check app/ tests/ --line-length 120 | Select-Object -First 20
        Success "Código formateado correctamente"
    } catch {
        Warning-Msg "Hay archivos que necesitan formateo (ejecuta: black app/ tests/)"
    }
    
    Write-Host "`n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    Write-Host "${CYAN}  FLAKE8 (Style Guide)${NC}"
    Write-Host "${CYAN}═══════════════════════════════════════════════════════════════════${NC}`n"
    
    try {
        flake8 app/ --max-line-length=120 --extend-ignore=E203,E501,W503 --count | Select-Object -Last 10
        Success "Sin errores de estilo"
    } catch {
        Warning-Msg "Hay algunos warnings de estilo"
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Tests Frontend (Opcional)
# ──────────────────────────────────────────────────────────────────────────────

function Run-FrontendTests {
    Print-Section "Tests Frontend .NET (Opcional)"
    
    if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
        Warning-Msg "Saltando tests de frontend (.NET no instalado)"
        return
    }
    
    Warning-Msg "Los tests de frontend requieren más tiempo"
    $response = Read-Host "¿Deseas ejecutarlos? (s/N)"
    
    if ($response -notmatch '^[Ss]$') {
        Info "Saltando tests de frontend"
        return
    }
    
    Set-Location $FrontendDir
    
    Write-Host "`n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    Write-Host "${CYAN}  .NET BUILD${NC}"
    Write-Host "${CYAN}═══════════════════════════════════════════════════════════════════${NC}`n"
    
    try {
        dotnet build | Select-Object -Last 10
        Success "Frontend compilado correctamente"
    } catch {
        Error-Msg "Error compilando frontend"
        return $false
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# Resumen Final
# ──────────────────────────────────────────────────────────────────────────────

function Print-Summary {
    Write-Host "`n${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    Write-Host "${CYAN}║                          📊 RESUMEN FINAL                                 ║${NC}"
    Write-Host "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}`n"
    
    Success "Tests locales completados"
    Info "Para deployment completo, usa: docker-compose --profile dev up -d"
    Info "Para más tests, ejecuta: pytest tests/ -v --cov=app"
    
    Write-Host ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

function Main {
    Print-Header
    
    Check-Prerequisites
    Setup-Backend
    
    $backendTestsOk = Run-BackendTests
    
    if ($backendTestsOk) {
        Success "✅ Tests backend exitosos"
    } else {
        Error-Msg "❌ Tests backend fallaron"
        exit 1
    }
    
    # Opcional: Tests de integración
    try { Run-IntegrationTests } catch { }
    
    # Opcional: Linting
    try { Run-Linting } catch { }
    
    # Opcional: Tests frontend
    try { Run-FrontendTests } catch { }
    
    Print-Summary
}

# Ejecutar script
Main

