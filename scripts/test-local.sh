#!/bin/bash

# ══════════════════════════════════════════════════════════════════════════════
# Ready4Hire v2.1 - Script de Testing Local (Linux/macOS)
# Permite ejecutar tests sin desplegar infraestructura completa
# ══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Rutas
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/Ready4Hire"
FRONTEND_DIR="$PROJECT_ROOT/WebApp"
E2E_DIR="$PROJECT_ROOT/e2e-tests"

# ──────────────────────────────────────────────────────────────────────────────
# Funciones de Utilidad
# ──────────────────────────────────────────────────────────────────────────────

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║              🧪 Ready4Hire v2.1 - Testing Local                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Verificar Prerrequisitos
# ──────────────────────────────────────────────────────────────────────────────

check_prerequisites() {
    print_section "Verificando Prerrequisitos"
    
    local all_ok=true
    
    # Python 3.11+
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | awk '{print $2}')
        success "Python: $python_version"
    else
        error "Python 3.11+ no encontrado"
        all_ok=false
    fi
    
    # .NET 9.0
    if command -v dotnet &> /dev/null; then
        local dotnet_version=$(dotnet --version)
        success ".NET: $dotnet_version"
    else
        warning ".NET no encontrado (solo necesario para tests de frontend)"
    fi
    
    # Node.js (para E2E tests)
    if command -v node &> /dev/null; then
        local node_version=$(node --version)
        success "Node.js: $node_version"
    else
        warning "Node.js no encontrado (solo necesario para tests E2E)"
    fi
    
    if [ "$all_ok" = false ]; then
        error "Faltan prerrequisitos. Instálalos e intenta de nuevo."
        exit 1
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Setup Backend
# ──────────────────────────────────────────────────────────────────────────────

setup_backend() {
    print_section "Configurando Backend Python"
    
    cd "$BACKEND_DIR"
    
    # Crear venv si no existe
    if [ ! -d "venv" ]; then
        info "Creando entorno virtual..."
        python3 -m venv venv
        success "Entorno virtual creado"
    else
        info "Entorno virtual ya existe"
    fi
    
    # Activar venv
    source venv/bin/activate
    
    # Instalar dependencias
    info "Instalando dependencias..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" || true
    
    success "Backend configurado"
}

# ──────────────────────────────────────────────────────────────────────────────
# Tests Backend
# ──────────────────────────────────────────────────────────────────────────────

run_backend_tests() {
    print_section "Ejecutando Tests Backend"
    
    cd "$BACKEND_DIR"
    source venv/bin/activate
    
    # Configurar PYTHONPATH
    export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"
    
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  TESTS UNITARIOS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}\n"
    
    if pytest tests/unit/ -v --tb=short --color=yes 2>&1; then
        success "Tests unitarios pasaron"
    else
        error "Tests unitarios fallaron"
        return 1
    fi
    
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  COVERAGE REPORT${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}\n"
    
    pytest tests/unit/ --cov=app --cov-report=term-missing --tb=short --color=yes 2>&1 | tail -20
}

# ──────────────────────────────────────────────────────────────────────────────
# Tests de Integración (Opcional)
# ──────────────────────────────────────────────────────────────────────────────

run_integration_tests() {
    print_section "Tests de Integración (Requiere servicios)"
    
    warning "Los tests de integración requieren Ollama, PostgreSQL, Redis, etc."
    read -p "¿Deseas ejecutarlos? (s/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        info "Saltando tests de integración"
        return 0
    fi
    
    cd "$BACKEND_DIR"
    source venv/bin/activate
    export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"
    
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  TESTS DE INTEGRACIÓN${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}\n"
    
    if pytest tests/integration/ -v --tb=short --color=yes -x 2>&1; then
        success "Tests de integración pasaron"
    else
        error "Tests de integración fallaron (esto es esperado si los servicios no están corriendo)"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Linting & Code Quality
# ──────────────────────────────────────────────────────────────────────────────

run_linting() {
    print_section "Linting & Code Quality"
    
    cd "$BACKEND_DIR"
    source venv/bin/activate
    
    info "Instalando herramientas de linting..."
    pip install -q black flake8 mypy pylint 2>&1 | grep -v "already satisfied" || true
    
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  BLACK (Code Formatting)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}\n"
    
    if black --check app/ tests/ --line-length 120 2>&1 | head -20; then
        success "Código formateado correctamente"
    else
        warning "Hay archivos que necesitan formateo (ejecuta: black app/ tests/)"
    fi
    
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  FLAKE8 (Style Guide)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}\n"
    
    if flake8 app/ --max-line-length=120 --extend-ignore=E203,E501,W503 --count 2>&1 | tail -10; then
        success "Sin errores de estilo"
    else
        warning "Hay algunos warnings de estilo"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Tests Frontend (Opcional)
# ──────────────────────────────────────────────────────────────────────────────

run_frontend_tests() {
    print_section "Tests Frontend .NET (Opcional)"
    
    if ! command -v dotnet &> /dev/null; then
        warning "Saltando tests de frontend (.NET no instalado)"
        return 0
    fi
    
    warning "Los tests de frontend requieren más tiempo"
    read -p "¿Deseas ejecutarlos? (s/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        info "Saltando tests de frontend"
        return 0
    fi
    
    cd "$FRONTEND_DIR"
    
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  .NET BUILD${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}\n"
    
    if dotnet build 2>&1 | tail -10; then
        success "Frontend compilado correctamente"
    else
        error "Error compilando frontend"
        return 1
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Resumen Final
# ──────────────────────────────────────────────────────────────────────────────

print_summary() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                          📊 RESUMEN FINAL                                 ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}\n"
    
    success "Tests locales completados"
    info "Para deployment completo, usa: docker-compose --profile dev up -d"
    info "Para más tests, ejecuta: pytest tests/ -v --cov=app"
    
    echo ""
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

main() {
    print_header
    
    check_prerequisites
    setup_backend
    
    if run_backend_tests; then
        success "✅ Tests backend exitosos"
    else
        error "❌ Tests backend fallaron"
        exit 1
    fi
    
    # Opcional: Tests de integración
    run_integration_tests || true
    
    # Opcional: Linting
    run_linting || true
    
    # Opcional: Tests frontend
    run_frontend_tests || true
    
    print_summary
}

# Ejecutar script
main "$@"

