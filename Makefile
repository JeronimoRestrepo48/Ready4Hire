# ============================================================================
# Ready4Hire - Makefile para Control Rápido
# ============================================================================

.PHONY: help start stop restart status logs clean deps build test

# Variables
SCRIPT := ./ready4hire.sh
PYTHON := python3
DOTNET := dotnet
BACKEND_DIR := Ready4Hire
FRONTEND_DIR := WebApp

# ============================================================================
# Comandos Principales
# ============================================================================

help: ## Mostrar esta ayuda
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                   🚀 Ready4Hire - Comandos Make                   ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Comandos de Control:"
	@echo "  make start        - Iniciar todos los servicios"
	@echo "  make stop         - Detener todos los servicios"
	@echo "  make restart      - Reiniciar todos los servicios"
	@echo "  make status       - Ver estado del sistema"
	@echo "  make logs         - Ver logs en tiempo real"
	@echo ""
	@echo "Comandos de Desarrollo:"
	@echo "  make deps         - Instalar dependencias"
	@echo "  make build        - Compilar backend y frontend"
	@echo "  make clean        - Limpiar logs y archivos temporales"
	@echo "  make test         - Ejecutar tests"
	@echo ""
	@echo "Modo Interactivo:"
	@echo "  make menu         - Abrir menú interactivo"
	@echo ""

start: ## Iniciar todos los servicios
	@echo "🚀 Iniciando Ready4Hire..."
	@$(SCRIPT) start

stop: ## Detener todos los servicios
	@echo "🛑 Deteniendo Ready4Hire..."
	@$(SCRIPT) stop

restart: ## Reiniciar todos los servicios
	@echo "🔄 Reiniciando Ready4Hire..."
	@$(SCRIPT) restart

status: ## Ver estado del sistema
	@$(SCRIPT) status

logs: ## Ver logs en tiempo real
	@tail -f logs/*.log

menu: ## Abrir menú interactivo
	@$(SCRIPT)

# ============================================================================
# Comandos de Desarrollo
# ============================================================================

deps: ## Instalar dependencias
	@echo "📦 Instalando dependencias..."
	@echo "  → Python..."
	@cd $(BACKEND_DIR) && pip install -r ../requirements.txt
	@echo "  → .NET..."
	@cd $(FRONTEND_DIR) && $(DOTNET) restore
	@echo "✅ Dependencias instaladas"

build: ## Compilar backend y frontend
	@echo "🔨 Compilando proyecto..."
	@echo "  → Frontend (Blazor)..."
	@cd $(FRONTEND_DIR) && $(DOTNET) build
	@echo "✅ Compilación completada"

clean: ## Limpiar logs y archivos temporales
	@echo "🧹 Limpiando archivos temporales..."
	@rm -rf logs/*.log
	@rm -f backend.pid frontend.pid
	@cd $(FRONTEND_DIR) && $(DOTNET) clean
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Limpieza completada"

test: ## Ejecutar tests
	@echo "🧪 Ejecutando tests..."
	@cd Ready4Hire && $(PYTHON) -m pytest tests/ -v
	@echo "✅ Tests completados"

# ============================================================================
# Comandos de Base de Datos
# ============================================================================

db-migrate: ## Crear migración de base de datos
	@echo "📊 Creando migración..."
	@cd $(FRONTEND_DIR) && $(DOTNET) ef migrations add $(name)

db-update: ## Aplicar migraciones
	@echo "📊 Aplicando migraciones..."
	@cd $(FRONTEND_DIR) && $(DOTNET) ef database update

db-reset: ## Resetear base de datos
	@echo "⚠️  Reseteando base de datos..."
	@cd $(FRONTEND_DIR) && $(DOTNET) ef database drop -f
	@cd $(FRONTEND_DIR) && $(DOTNET) ef database update
	@echo "✅ Base de datos reseteada"

# ============================================================================
# Comandos de Despliegue
# ============================================================================

docker-build: ## Construir imagen Docker
	@echo "🐳 Construyendo imagen Docker..."
	@docker-compose build

docker-up: ## Iniciar con Docker Compose
	@echo "🐳 Iniciando con Docker..."
	@docker-compose up -d

docker-down: ## Detener Docker Compose
	@echo "🐳 Deteniendo Docker..."
	@docker-compose down

docker-logs: ## Ver logs de Docker
	@docker-compose logs -f

# ============================================================================
# Comandos de Utilidad
# ============================================================================

check: ## Verificar dependencias y configuración
	@echo "🔍 Verificando sistema..."
	@command -v python3 >/dev/null 2>&1 && echo "✅ Python3 instalado" || echo "❌ Python3 no encontrado"
	@command -v dotnet >/dev/null 2>&1 && echo "✅ .NET SDK instalado" || echo "❌ .NET SDK no encontrado"
	@command -v ollama >/dev/null 2>&1 && echo "✅ Ollama instalado" || echo "❌ Ollama no encontrado"
	@systemctl is-active --quiet postgresql && echo "✅ PostgreSQL activo" || echo "⚠️  PostgreSQL inactivo"

ps: ## Ver procesos de Ready4Hire
	@echo "📊 Procesos activos:"
	@ps aux | grep -E "(ollama|uvicorn|dotnet run)" | grep -v grep || echo "  (ninguno)"

ports: ## Ver puertos en uso
	@echo "🔌 Puertos en uso:"
	@lsof -i :5432 -i :8001 -i :5214 -i :11434 2>/dev/null || echo "  (ninguno)"

health: ## Verificar health de servicios
	@echo "❤️  Health Check:"
	@curl -s http://localhost:8001/api/v2/health 2>/dev/null | python3 -m json.tool || echo "❌ Backend no responde"

# ============================================================================
# Alias Rápidos
# ============================================================================

up: start ## Alias para 'make start'
down: stop ## Alias para 'make stop'
re: restart ## Alias para 'make restart'
s: status ## Alias para 'make status'
l: logs ## Alias para 'make logs'

# ============================================================================
# Default
# ============================================================================

.DEFAULT_GOAL := help
