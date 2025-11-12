#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Ready4Hire - Script de Inicio en Modo Desarrollo
# ══════════════════════════════════════════════════════════════════════════════
#
# Este script inicia todos los servicios en modo desarrollo y muestra
# el estado y URLs de acceso de cada servicio.
#
# Uso: ./run.sh
# ══════════════════════════════════════════════════════════════════════════════

# No usar set -e para permitir verificaciones que pueden fallar

# Colores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_header() { echo -e "${CYAN}$1${NC}"; }

# Banner
echo ""
print_header "╔════════════════════════════════════════════════════════════════╗"
print_header "║         Ready4Hire - Modo Desarrollo                         ║"
print_header "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Verificar Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker no está instalado"
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker no está corriendo o no tienes permisos"
    print_info "Intenta: sudo systemctl start docker"
    exit 1
fi

# Verificar Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose no está instalado"
    exit 1
fi

# Usar docker compose (v2) si está disponible, sino docker-compose (v1)
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

print_info "Usando: $COMPOSE_CMD"
echo ""

# Verificar servicios existentes
print_info "Verificando servicios existentes..."
if $COMPOSE_CMD --profile dev ps 2>/dev/null | grep -q "Up"; then
    print_info "Hay servicios corriendo. Se mantendrán activos."
    print_info "Si deseas reiniciar, primero ejecuta: $COMPOSE_CMD --profile dev down"
    echo ""
else
    print_info "No hay servicios corriendo, iniciando nuevos..."
    echo ""
fi

# Iniciar servicios
print_header "🚀 Iniciando servicios en modo desarrollo..."
echo ""

$COMPOSE_CMD --profile dev up -d

if [ $? -ne 0 ]; then
    print_error "Error al iniciar los servicios"
    exit 1
fi

print_success "Servicios iniciados"
echo ""

# Esperar a que los servicios estén listos
print_info "Esperando a que los servicios estén listos (10 segundos)..."
sleep 10

# Función para verificar salud de un servicio (rápida)
check_service_health() {
    local service=$1
    local url=$2
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f --max-time 2 "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    return 1
}

# Función para obtener estado de un contenedor
get_container_status() {
    local container=$1
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        local status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null)
        if [ "$status" = "running" ]; then
            echo "running"
        else
            echo "$status"
        fi
    else
        echo "not_found"
    fi
}

# Verificar servicios y mostrar URLs
print_header "═══════════════════════════════════════════════════════════════"
print_header "📊 ESTADO DE SERVICIOS Y URLs DE ACCESO"
print_header "═══════════════════════════════════════════════════════════════"
echo ""

# Obtener variables de entorno o usar defaults
ENVIRONMENT=${ENVIRONMENT:-dev}
POSTGRES_PORT=${POSTGRES_PORT:-5432}
REDIS_PORT=${REDIS_PORT:-6379}
QDRANT_PORT=${QDRANT_PORT:-6333}
OLLAMA_PORT=${OLLAMA_PORT:-11434}
API_PORT=${API_PORT:-8001}
WEBAPP_PORT=${WEBAPP_PORT:-5214}

# Servicios Core
# Formato: service_name:container_name:port:description:url:check_http
services=(
    "postgres:ready4hire_postgres_${ENVIRONMENT}:${POSTGRES_PORT}:PostgreSQL Database:postgresql://localhost:${POSTGRES_PORT}:false"
    "redis:ready4hire_redis_${ENVIRONMENT}:${REDIS_PORT}:Redis Cache:redis://localhost:${REDIS_PORT}:false"
    "qdrant:ready4hire_qdrant_${ENVIRONMENT}:${QDRANT_PORT}:Qdrant Vector DB:http://localhost:${QDRANT_PORT}/dashboard:true"
    "ollama:ready4hire_ollama_${ENVIRONMENT}:${OLLAMA_PORT}:Ollama LLM:http://localhost:${OLLAMA_PORT}:true"
    "api:ready4hire_api_${ENVIRONMENT}:${API_PORT}:FastAPI Backend:http://localhost:${API_PORT}/docs:true"
    "webapp:ready4hire_webapp_${ENVIRONMENT}:${WEBAPP_PORT}:Blazor WebApp:http://localhost:${WEBAPP_PORT}:true"
)

all_healthy=true

for service_info in "${services[@]}"; do
    IFS=':' read -r service_name container_name port service_desc url check_http <<< "$service_info"
    
    status=$(get_container_status "$container_name" 2>/dev/null || echo "not_found")
    
    if [ "$status" = "running" ]; then
        # Verificar si el servicio está respondiendo
        if [ "$check_http" = "true" ]; then
            if check_service_health "$service_name" "$url" 2>/dev/null; then
                print_success "$service_desc"
                echo "   📍 URL: $url"
                echo "   🐳 Contenedor: $container_name"
                echo "   🔌 Puerto: $port"
            else
                print_warning "$service_desc (iniciando...)"
                echo "   📍 URL: $url (aún no disponible)"
                echo "   🐳 Contenedor: $container_name"
                echo "   🔌 Puerto: $port"
                all_healthy=false
            fi
        else
            # Para servicios que no tienen HTTP endpoint
            print_success "$service_desc"
            echo "   📍 Conexión: $url"
            echo "   🐳 Contenedor: $container_name"
            echo "   🔌 Puerto: $port"
        fi
    else
        print_error "$service_desc"
        echo "   Estado: $status"
        all_healthy=false
    fi
    echo ""
done

# Resumen de URLs
print_header "═══════════════════════════════════════════════════════════════"
print_header "🌐 URLs DE ACCESO PRINCIPALES"
print_header "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "${CYAN}Frontend WebApp:${NC}     http://localhost:${WEBAPP_PORT}"
echo -e "${CYAN}Backend API:${NC}         http://localhost:${API_PORT}"
echo -e "${CYAN}API Docs (Swagger):${NC}  http://localhost:${API_PORT}/docs"
echo -e "${CYAN}API Health:${NC}           http://localhost:${API_PORT}/api/v2/health"
echo -e "${CYAN}Ollama:${NC}              http://localhost:${OLLAMA_PORT}"
echo -e "${CYAN}Qdrant Dashboard:${NC}    http://localhost:${QDRANT_PORT}/dashboard"
echo ""

# Verificar base de datos
print_header "═══════════════════════════════════════════════════════════════"
print_header "🗄️  VERIFICACIÓN DE BASE DE DATOS"
print_header "═══════════════════════════════════════════════════════════════"
echo ""

if docker exec "ready4hire_postgres_${ENVIRONMENT}" pg_isready -U ready4hire > /dev/null 2>&1; then
    print_success "PostgreSQL está disponible"
    echo "   Host: localhost"
    echo "   Puerto: ${POSTGRES_PORT}"
    echo "   Base de datos: ready4hire_db"
    echo "   Usuario: ready4hire"
else
    print_warning "PostgreSQL aún no está listo (puede estar iniciando)"
fi

echo ""

# Verificar Redis
print_header "═══════════════════════════════════════════════════════════════"
print_header "🔴 VERIFICACIÓN DE REDIS"
print_header "═══════════════════════════════════════════════════════════════"
echo ""

if docker exec "ready4hire_redis_${ENVIRONMENT}" redis-cli -a "Ready4Hire2024!" ping > /dev/null 2>&1; then
    print_success "Redis está disponible"
    echo "   Host: localhost"
    echo "   Puerto: ${REDIS_PORT}"
    echo "   Password: Ready4Hire2024!"
else
    print_warning "Redis aún no está listo (puede estar iniciando)"
fi

echo ""

# Comandos útiles
print_header "═══════════════════════════════════════════════════════════════"
print_header "📝 COMANDOS ÚTILES"
print_header "═══════════════════════════════════════════════════════════════"
echo ""
echo "Ver logs de todos los servicios:"
echo "  $COMPOSE_CMD --profile dev logs -f"
echo ""
echo "Ver logs de un servicio específico:"
echo "  $COMPOSE_CMD --profile dev logs -f [service_name]"
echo ""
echo "Ver estado de servicios:"
echo "  $COMPOSE_CMD --profile dev ps"
echo ""
echo "Detener servicios:"
echo "  $COMPOSE_CMD --profile dev down"
echo ""
echo "Reiniciar un servicio:"
echo "  $COMPOSE_CMD --profile dev restart [service_name]"
echo ""

# Estado final
if [ "$all_healthy" = true ]; then
    print_success "🎉 Todos los servicios están corriendo correctamente!"
    echo ""
    print_info "Puedes acceder a la aplicación en: http://localhost:${WEBAPP_PORT}"
else
    print_warning "⚠️  Algunos servicios aún están iniciando..."
    echo ""
    print_info "Espera unos segundos y verifica el estado con:"
    echo "  $COMPOSE_CMD --profile dev ps"
    echo ""
    print_info "O revisa los logs con:"
    echo "  $COMPOSE_CMD --profile dev logs -f"
fi

echo ""
print_header "═══════════════════════════════════════════════════════════════"
echo ""

