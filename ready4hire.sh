#!/bin/bash

# ============================================================================
# Ready4Hire - Script de Control Principal
# ============================================================================

set -e  # Exit on error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Rutas
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/Ready4Hire"
FRONTEND_DIR="$PROJECT_ROOT/WebApp"
LOGS_DIR="$PROJECT_ROOT/logs"

# PIDs
BACKEND_PID_FILE="$PROJECT_ROOT/backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/frontend.pid"

# ============================================================================
# Funciones de Utilidad
# ============================================================================

print_header() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                    🚀 Ready4Hire Control Panel                    ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
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

# ============================================================================
# Verificación de Dependencias
# ============================================================================

check_dependencies() {
    print_section "🔍 Verificando Dependencias"
    
    local deps_ok=true
    
    # Python
    if command -v python3 &> /dev/null; then
        success "Python3: $(python3 --version)"
    else
        error "Python3 no encontrado"
        deps_ok=false
    fi
    
    # .NET
    if command -v dotnet &> /dev/null; then
        success ".NET SDK: $(dotnet --version)"
    else
        error ".NET SDK no encontrado"
        deps_ok=false
    fi
    
    # Ollama
    if command -v ollama &> /dev/null; then
        success "Ollama: Instalado"
    else
        error "Ollama no encontrado"
        deps_ok=false
    fi
    
    # PostgreSQL
    if systemctl is-active --quiet postgresql; then
        success "PostgreSQL: Activo"
    else
        warning "PostgreSQL: No activo (intentando iniciar...)"
        sudo systemctl start postgresql
        if systemctl is-active --quiet postgresql; then
            success "PostgreSQL: Iniciado"
        else
            error "PostgreSQL: Fallo al iniciar"
            deps_ok=false
        fi
    fi
    
    if [ "$deps_ok" = false ]; then
        error "Algunas dependencias faltan. Por favor, instálalas primero."
        exit 1
    fi
    
    success "Todas las dependencias verificadas"
}

# ============================================================================
# Gestión de Logs
# ============================================================================

setup_logs() {
    mkdir -p "$LOGS_DIR"
    
    # Rotar logs si son muy grandes (>10MB)
    for log in "$LOGS_DIR"/*.log; do
        if [ -f "$log" ] && [ $(stat -c%s "$log") -gt 10485760 ]; then
            mv "$log" "$log.old"
            info "Log rotado: $(basename $log)"
        fi
    done
}

# ============================================================================
# Control de Servicios
# ============================================================================

stop_all_services() {
    print_section "🛑 Deteniendo Servicios"
    
    # Backend
    if [ -f "$BACKEND_PID_FILE" ]; then
        local pid=$(cat "$BACKEND_PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null || true
            success "Backend detenido (PID: $pid)"
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Frontend
    if [ -f "$FRONTEND_PID_FILE" ]; then
        local pid=$(cat "$FRONTEND_PID_FILE")
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null || true
            success "Frontend detenido (PID: $pid)"
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    # Killall como backup
    pkill -f "uvicorn.*main_v2_improved" 2>/dev/null || true
    pkill -f "dotnet run" 2>/dev/null || true
    pkill -f "ollama serve" 2>/dev/null || true
    
    sleep 2
    success "Todos los servicios detenidos"
}

start_ollama() {
    info "Iniciando Ollama..."
    
    if pgrep -f "ollama serve" > /dev/null; then
        warning "Ollama ya está ejecutándose"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    nohup ollama serve > "$LOGS_DIR/ollama.log" 2>&1 &
    
    sleep 5
    
    if pgrep -f "ollama serve" > /dev/null; then
        success "Ollama iniciado (puerto 11434)"
    else
        error "Fallo al iniciar Ollama"
        return 1
    fi
}

start_backend() {
    info "Iniciando Backend (FastAPI)..."
    
    cd "$BACKEND_DIR"
    
    # Verificar que existe el módulo
    if [ ! -f "app/main_v2_improved.py" ]; then
        error "Archivo main_v2_improved.py no encontrado"
        return 1
    fi
    
    # Activar venv si existe
    if [ -d "../venv/bin" ]; then
        source ../venv/bin/activate
    fi
    
    # Iniciar backend
    nohup python3 -m uvicorn app.main_v2_improved:app --host 0.0.0.0 --port 8001 > "$LOGS_DIR/ready4hire_api.log" 2>&1 &
    echo $! > "$BACKEND_PID_FILE"
    
    sleep 8
    
    # Verificar health
    if curl -s http://localhost:8001/api/v2/health > /dev/null 2>&1; then
        success "Backend iniciado (puerto 8001, PID: $(cat $BACKEND_PID_FILE))"
        return 0
    else
        error "Backend no responde en health check"
        return 1
    fi
}

start_frontend() {
    info "Iniciando Frontend (Blazor)..."
    
    cd "$FRONTEND_DIR"
    
    # Verificar .env
    if [ ! -f ".env" ]; then
        warning ".env no encontrado en WebApp, creando..."
        cat > .env << EOF
POSTGRES_CONNECTION=Host=localhost;Port=5432;Database=ready4hire;Username=postgres;Password=postgres
EOF
    fi
    
    # Build si es necesario
    if [ ! -d "bin/Debug" ]; then
        info "Compilando frontend..."
        dotnet build > /dev/null 2>&1
    fi
    
    # Iniciar frontend
    nohup dotnet run > "$LOGS_DIR/webapp.log" 2>&1 &
    echo $! > "$FRONTEND_PID_FILE"
    
    sleep 12
    
    # Verificar
    if curl -s http://localhost:5214 > /dev/null 2>&1; then
        success "Frontend iniciado (puerto 5214, PID: $(cat $FRONTEND_PID_FILE))"
        return 0
    else
        error "Frontend no responde"
        return 1
    fi
}

start_all_services() {
    print_section "🚀 Iniciando Servicios"
    
    start_ollama
    start_backend
    start_frontend
    
    print_section "✅ Sistema Iniciado"
}

# ============================================================================
# Estado del Sistema
# ============================================================================

show_status() {
    print_section "📊 Estado del Sistema"
    
    # PostgreSQL
    if systemctl is-active --quiet postgresql; then
        success "PostgreSQL: Activo (puerto 5432)"
    else
        error "PostgreSQL: Inactivo"
    fi
    
    # Ollama
    if pgrep -f "ollama serve" > /dev/null; then
        success "Ollama: Activo (puerto 11434)"
    else
        error "Ollama: Inactivo"
    fi
    
    # Backend
    if [ -f "$BACKEND_PID_FILE" ] && ps -p $(cat "$BACKEND_PID_FILE") > /dev/null 2>&1; then
        if curl -s http://localhost:8001/api/v2/health > /dev/null 2>&1; then
            success "Backend: Activo (puerto 8001, PID: $(cat $BACKEND_PID_FILE))"
        else
            warning "Backend: Proceso activo pero no responde"
        fi
    else
        error "Backend: Inactivo"
    fi
    
    # Frontend
    if [ -f "$FRONTEND_PID_FILE" ] && ps -p $(cat "$FRONTEND_PID_FILE") > /dev/null 2>&1; then
        if curl -s http://localhost:5214 > /dev/null 2>&1; then
            success "Frontend: Activo (puerto 5214, PID: $(cat $FRONTEND_PID_FILE))"
        else
            warning "Frontend: Proceso activo pero no responde"
        fi
    else
        error "Frontend: Inactivo"
    fi
    
    echo ""
    info "URLs de Acceso:"
    echo "   🌐 Frontend:  http://localhost:5214"
    echo "   🔧 Backend:   http://localhost:8001"
    echo "   📚 API Docs:  http://localhost:8001/docs"
    echo "   ❤️  Health:    http://localhost:8001/api/v2/health"
}

# ============================================================================
# Ver Logs
# ============================================================================

view_logs() {
    print_section "📋 Logs del Sistema"
    
    echo "Selecciona el log a ver:"
    echo "  1) Backend (FastAPI)"
    echo "  2) Frontend (Blazor)"
    echo "  3) Ollama"
    echo "  4) Todos"
    echo "  5) Volver"
    
    read -p "Opción: " log_choice
    
    case $log_choice in
        1)
            tail -f "$LOGS_DIR/ready4hire_api.log"
            ;;
        2)
            tail -f "$LOGS_DIR/webapp.log"
            ;;
        3)
            tail -f "$LOGS_DIR/ollama.log"
            ;;
        4)
            tail -f "$LOGS_DIR"/*.log
            ;;
        5)
            return
            ;;
        *)
            error "Opción inválida"
            ;;
    esac
}

# ============================================================================
# Menú Principal
# ============================================================================

show_menu() {
    clear
    print_header
    
    echo -e "${CYAN}Selecciona una opción:${NC}\n"
    echo "  ${GREEN}1)${NC} 🚀 Iniciar todos los servicios"
    echo "  ${GREEN}2)${NC} 🛑 Detener todos los servicios"
    echo "  ${GREEN}3)${NC} 🔄 Reiniciar todos los servicios"
    echo "  ${GREEN}4)${NC} 📊 Ver estado del sistema"
    echo "  ${GREEN}5)${NC} 📋 Ver logs"
    echo "  ${GREEN}6)${NC} 🔍 Verificar dependencias"
    echo "  ${GREEN}7)${NC} 🧹 Limpiar logs"
    echo "  ${GREEN}8)${NC} 🚪 Salir"
    echo ""
}

clean_logs() {
    print_section "🧹 Limpiando Logs"
    
    rm -f "$LOGS_DIR"/*.log
    success "Logs limpiados"
}

main_menu() {
    while true; do
        show_menu
        read -p "Opción: " choice
        
        case $choice in
            1)
                setup_logs
                start_all_services
                show_status
                read -p "Presiona Enter para continuar..."
                ;;
            2)
                stop_all_services
                read -p "Presiona Enter para continuar..."
                ;;
            3)
                stop_all_services
                setup_logs
                start_all_services
                show_status
                read -p "Presiona Enter para continuar..."
                ;;
            4)
                show_status
                read -p "Presiona Enter para continuar..."
                ;;
            5)
                view_logs
                ;;
            6)
                check_dependencies
                read -p "Presiona Enter para continuar..."
                ;;
            7)
                clean_logs
                read -p "Presiona Enter para continuar..."
                ;;
            8)
                echo -e "\n${GREEN}¡Hasta luego!${NC}\n"
                exit 0
                ;;
            *)
                error "Opción inválida"
                sleep 1
                ;;
        esac
    done
}

# ============================================================================
# Script Principal
# ============================================================================

# Si se pasa argumento, ejecutar directamente
if [ $# -gt 0 ]; then
    case $1 in
        start)
            check_dependencies
            setup_logs
            start_all_services
            show_status
            ;;
        stop)
            stop_all_services
            ;;
        restart)
            stop_all_services
            check_dependencies
            setup_logs
            start_all_services
            show_status
            ;;
        status)
            show_status
            ;;
        logs)
            tail -f "$LOGS_DIR"/*.log
            ;;
        *)
            echo "Uso: $0 {start|stop|restart|status|logs}"
            exit 1
            ;;
    esac
else
    # Modo interactivo
    check_dependencies
    main_menu
fi

