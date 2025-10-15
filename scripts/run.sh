#!/bin/bash
# ==============================================================================
# Ready4Hire - Script de Inicio Completo
# ==============================================================================
# Levanta todo el stack: Ollama → FastAPI (DDD v2) → Blazor WebApp
# 
# Uso:
#   ./run.sh              # Modo normal
#   ./run.sh --dev        # Modo desarrollo (con reload)
#   ./run.sh --stop       # Detener todos los servicios
#   ./run.sh --status     # Ver estado de servicios
#
# Author: Ready4Hire Team
# Version: 2.0.0 (DDD Architecture)
# ==============================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorios
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(dirname "$SCRIPT_DIR")"
READY4HIRE_DIR="$INTEGRATION_ROOT/Ready4Hire"
WEBAPP_DIR="$INTEGRATION_ROOT/WebApp"
LOGS_DIR="$READY4HIRE_DIR/logs"

# Archivos de log
OLLAMA_LOG="$LOGS_DIR/ollama.log"
API_LOG="$LOGS_DIR/ready4hire_api.log"
WEBAPP_LOG="$LOGS_DIR/webapp.log"

# Configuración
OLLAMA_MODEL="${OLLAMA_MODEL:-ready4hire:latest}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8001}"
WEBAPP_PORT="${WEBAPP_PORT:-5214}"
DEV_MODE=false

# ==============================================================================
# Funciones de Utilidad
# ==============================================================================

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ==============================================================================
# Funciones de Control de Servicios
# ==============================================================================

stop_services() {
    print_header "Deteniendo Servicios Ready4Hire"
    
    # Detener Ollama
    if pgrep -x "ollama" > /dev/null; then
        print_info "Deteniendo Ollama..."
        pkill -f "ollama serve" || true
        print_success "Ollama detenido"
    else
        print_warning "Ollama no estaba corriendo"
    fi
    
    # Detener API
    if lsof -ti:$API_PORT > /dev/null 2>&1; then
        print_info "Deteniendo API Python (puerto $API_PORT)..."
        lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
        print_success "API detenida"
    else
        print_warning "API no estaba corriendo en puerto $API_PORT"
    fi
    
    # Detener WebApp
    if lsof -ti:$WEBAPP_PORT > /dev/null 2>&1; then
        print_info "Deteniendo WebApp (puerto $WEBAPP_PORT)..."
        lsof -ti:$WEBAPP_PORT | xargs kill -9 2>/dev/null || true
        print_success "WebApp detenida"
    else
        print_warning "WebApp no estaba corriendo en puerto $WEBAPP_PORT"
    fi
    
    print_success "Todos los servicios detenidos"
    exit 0
}

show_status() {
    print_header "Estado de Servicios Ready4Hire"
    
    # Ollama
    if pgrep -x "ollama" > /dev/null; then
        print_success "Ollama: RUNNING"
        ollama list 2>/dev/null | grep -E "NAME|$OLLAMA_MODEL" || true
    else
        print_error "Ollama: STOPPED"
    fi
    
    # API
    if lsof -ti:$API_PORT > /dev/null 2>&1; then
        print_success "API Python: RUNNING (puerto $API_PORT)"
        curl -s http://localhost:$API_PORT/health 2>/dev/null | python3 -m json.tool || \
        curl -s http://localhost:$API_PORT/api/v2/health 2>/dev/null | python3 -m json.tool || \
        print_warning "API no responde al health check"
    else
        print_error "API Python: STOPPED"
    fi
    
    # WebApp
    if lsof -ti:$WEBAPP_PORT > /dev/null 2>&1; then
        print_success "WebApp: RUNNING (puerto $WEBAPP_PORT)"
    else
        print_error "WebApp: STOPPED"
    fi
    
    exit 0
}

# ==============================================================================
# Funciones de Inicio
# ==============================================================================

start_ollama() {
    print_header "1/4 - Iniciando Ollama Server"
    
    # Verificar si Ollama está instalado
    if ! command -v ollama &> /dev/null; then
        print_error "Ollama no está instalado"
        print_info "Instalar con: curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi
    
    # Verificar si ya está corriendo
    if pgrep -x "ollama" > /dev/null; then
        print_success "Ollama ya está corriendo"
    else
        print_info "Iniciando servidor Ollama..."
        mkdir -p "$LOGS_DIR"
        nohup ollama serve > "$OLLAMA_LOG" 2>&1 &
        sleep 3
        
        if pgrep -x "ollama" > /dev/null; then
            print_success "Ollama iniciado correctamente"
        else
            print_error "Error al iniciar Ollama. Ver logs: $OLLAMA_LOG"
            exit 1
        fi
    fi
    
    # Verificar/descargar modelo
    print_info "Verificando modelo $OLLAMA_MODEL..."
    if ollama list | grep -q "$OLLAMA_MODEL"; then
        print_success "Modelo $OLLAMA_MODEL ya está descargado"
    else
        print_info "Descargando modelo $OLLAMA_MODEL (esto puede tardar varios minutos)..."
        ollama pull "$OLLAMA_MODEL"
        print_success "Modelo descargado correctamente"
    fi
    
    # Test de conectividad
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        print_success "Ollama respondiendo correctamente"
    else
        print_warning "Ollama no responde en el endpoint esperado"
    fi
    
    echo ""
}

start_api() {
    print_header "2/4 - Iniciando API Python (FastAPI DDD v2)"
    
    # Ir al directorio del proyecto
    cd "$READY4HIRE_DIR"
    
    # Verificar si existe app/main_v2.py
    if [ ! -f "app/main_v2.py" ]; then
        print_error "app/main_v2.py no encontrado"
        exit 1
    fi
    
    # Verificar entorno virtual (opcional - puede funcionar sin venv)
    VENV_ACTIVATED=false
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        # Intentar activar venv
        if python3 -c "import fastapi, uvicorn" 2>/dev/null || source venv/bin/activate 2>/dev/null; then
            VENV_ACTIVATED=true
            print_success "Virtual environment activado"
        fi
    elif [ -d "../venv" ] && [ -f "../venv/bin/activate" ]; then
        if python3 -c "import fastapi, uvicorn" 2>/dev/null || source ../venv/bin/activate 2>/dev/null; then
            VENV_ACTIVATED=true
            print_success "Virtual environment activado"
        fi
    fi
    
    if [ "$VENV_ACTIVATED" = false ]; then
        print_warning "No se encontró/activó virtual environment, usando Python del sistema"
    fi
    
    # Verificar dependencias (en venv o sistema)
    if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
        print_error "Dependencias no instaladas ni en venv ni en sistema"
        print_info "Instalar con: pip install fastapi uvicorn"
        print_info "O con requirements: pip install -r app/requirements.txt"
        exit 1
    fi
    
    print_success "Dependencias verificadas"
    
    # Detener API anterior si existe
    if lsof -ti:$API_PORT > /dev/null 2>&1; then
        print_warning "API ya corriendo en puerto $API_PORT. Deteniendo..."
        lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Crear directorio de logs
    mkdir -p "$LOGS_DIR"
    
    # Iniciar API
    print_info "Iniciando API en puerto $API_PORT..."
    
    # Establecer PYTHONPATH
    export PYTHONPATH="$READY4HIRE_DIR:$PYTHONPATH"
    
    if [ "$DEV_MODE" = true ]; then
        print_info "Modo desarrollo: auto-reload activado"
        nohup python3 -m uvicorn app.main_v2:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --reload \
            > "$API_LOG" 2>&1 &
    else
        nohup python3 -m uvicorn app.main_v2:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            > "$API_LOG" 2>&1 &
    fi
    
    API_PID=$!
    sleep 30
    
    # Verificar que la API inició
    if lsof -ti:$API_PORT > /dev/null 2>&1; then
        print_success "API iniciada correctamente (PID: $API_PID)"
        
        # Health check (intentar varios endpoints)
        print_info "Verificando health endpoint..."
        sleep 3  # Esperar un poco más para que cargue
        if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
            print_success "API respondiendo correctamente"
            curl -s http://localhost:$API_PORT/health | python3 -m json.tool 2>/dev/null || true
        elif curl -s http://localhost:$API_PORT/api/v2/health > /dev/null 2>&1; then
            print_success "API respondiendo correctamente"
            curl -s http://localhost:$API_PORT/api/v2/health | python3 -m json.tool 2>/dev/null || true
        else
            print_warning "API puede tardar unos segundos más en cargar"
            print_info "Ver logs: tail -f $API_LOG"
        fi
    else
        print_error "Error al iniciar API. Ver logs: $API_LOG"
        tail -20 "$API_LOG"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
}

start_webapp() {
    print_header "3/4 - Iniciando WebApp (Blazor)"
    
    # Verificar si existe el directorio WebApp
    if [ ! -d "$WEBAPP_DIR" ]; then
        print_warning "Directorio WebApp no encontrado. Saltando..."
        return 0
    fi
    
    # Verificar si dotnet está instalado
    if ! command -v dotnet &> /dev/null; then
        print_warning "dotnet no está instalado. Saltando WebApp..."
        print_info "Instalar .NET SDK: https://dotnet.microsoft.com/download"
        return 0
    fi
    
    cd "$WEBAPP_DIR"
    
    # Verificar archivo de proyecto
    if [ ! -f "Ready4Hire.csproj" ]; then
        print_error "Proyecto Blazor no encontrado"
        cd "$SCRIPT_DIR"
        return 1
    fi
    
    # Compilar
    print_info "Compilando proyecto Blazor..."
    if dotnet build > /dev/null 2>&1; then
        print_success "Compilación exitosa"
    else
        print_error "Error en compilación"
        dotnet build
        cd "$SCRIPT_DIR"
        return 1
    fi
    
    # Detener WebApp anterior si existe
    if lsof -ti:$WEBAPP_PORT > /dev/null 2>&1; then
        print_warning "WebApp ya corriendo en puerto $WEBAPP_PORT. Deteniendo..."
        lsof -ti:$WEBAPP_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Iniciar WebApp
    print_info "Iniciando WebApp en puerto $WEBAPP_PORT..."
    mkdir -p "$LOGS_DIR"
    nohup dotnet run --urls="http://localhost:$WEBAPP_PORT" > "$WEBAPP_LOG" 2>&1 &
    WEBAPP_PID=$!
    
    sleep 3
    
    if lsof -ti:$WEBAPP_PORT > /dev/null 2>&1; then
        print_success "WebApp iniciada correctamente (PID: $WEBAPP_PID)"
    else
        print_warning "WebApp puede estar tardando en iniciar. Ver logs: $WEBAPP_LOG"
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
}

show_summary() {
    print_header "4/4 - Ready4Hire Iniciado Correctamente"
    
    echo ""
    echo -e "${GREEN}🚀 Servicios Ready4Hire en Ejecución:${NC}"
    echo ""
    echo -e "  ${BLUE}Ollama LLM${NC}"
    echo -e "    └─ URL: http://localhost:11434"
    echo -e "    └─ Modelo: $OLLAMA_MODEL"
    echo -e "    └─ Log: $OLLAMA_LOG"
    echo ""
    echo -e "  ${BLUE}API REST (FastAPI v2 - DDD)${NC}"
    echo -e "    └─ URL: http://localhost:$API_PORT"
    echo -e "    └─ Docs: http://localhost:$API_PORT/docs"
    echo -e "    └─ ReDoc: http://localhost:$API_PORT/redoc"
    echo -e "    └─ Health: http://localhost:$API_PORT/health"
    echo -e "    └─ Log: $API_LOG"
    echo ""
    
    if lsof -ti:$WEBAPP_PORT > /dev/null 2>&1; then
        echo -e "  ${BLUE}WebApp (Blazor)${NC}"
        echo -e "    └─ URL: http://localhost:$WEBAPP_PORT"
        echo -e "    └─ Log: $WEBAPP_LOG"
        echo ""
    fi
    
    echo -e "${YELLOW}📝 Comandos útiles:${NC}"
    echo -e "  Ver logs en tiempo real:"
    echo -e "    ${GREEN}tail -f $API_LOG${NC}"
    echo ""
    echo -e "  Detener servicios:"
    echo -e "    ${GREEN}$0 --stop${NC}"
    echo ""
    echo -e "  Ver estado:"
    echo -e "    ${GREEN}$0 --status${NC}"
    echo ""
    echo -e "  Health check API:"
    echo -e "    ${GREEN}curl http://localhost:$API_PORT/health${NC}"
    echo ""
    echo -e "  Abrir en navegador:"
    echo -e "    ${GREEN}xdg-open http://localhost:$API_PORT${NC}  # O simplemente abre en tu navegador"
    echo ""
    
    print_success "¡Sistema Ready4Hire listo para usar!"
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    # Parsear argumentos
    case "${1:-}" in
        --stop)
            stop_services
            ;;
        --status)
            show_status
            ;;
        --dev)
            DEV_MODE=true
            ;;
        --help|-h)
            echo "Ready4Hire - Script de Inicio"
            echo ""
            echo "Uso: $0 [OPCIÓN]"
            echo ""
            echo "Opciones:"
            echo "  (sin opción)   Iniciar todos los servicios"
            echo "  --dev          Modo desarrollo (con auto-reload)"
            echo "  --stop         Detener todos los servicios"
            echo "  --status       Ver estado de servicios"
            echo "  --help, -h     Mostrar esta ayuda"
            echo ""
            exit 0
            ;;
    esac
    
    # Banner
    echo ""
    echo -e "${BLUE}"
    echo "  ____                _       _  _   _   _  _          "
    echo " |  _ \ ___  __ _  __| |_   _| || | | | | |(_)_ __ ___ "
    echo " | |_) / _ \/ _\` |/ _\` | | | | || |_| |_| || | '__/ _ \\"
    echo " |  _ <  __/ (_| | (_| | |_| |__   _|  _  || | | |  __/"
    echo " |_| \_\___|\__,_|\__,_|\__, |  |_| |_| |_||_|_|  \___|"
    echo "                        |___/                          "
    echo -e "${NC}"
    echo -e "${GREEN}Sistema de Entrevistas Técnicas con IA${NC}"
    echo -e "${YELLOW}Version 2.0.0 - DDD Architecture${NC}"
    echo ""
    
    # Ejecutar servicios
    start_ollama
    start_api
    start_webapp
    show_summary
}

# Ejecutar
main "$@"
