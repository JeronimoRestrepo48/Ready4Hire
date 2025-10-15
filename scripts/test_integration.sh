#!/bin/bash
# ==============================================================================
# Ready4Hire - Script de Pruebas de Integración
# ==============================================================================
# Prueba que todos los servicios estén funcionando y comunicándose correctamente
#
# Uso:
#   ./test_integration.sh
#
# Author: Ready4Hire Team
# Version: 1.0.0
# ==============================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
API_PORT=8001
WEBAPP_PORT=5214
OLLAMA_PORT=11434

# Contadores
TESTS_PASSED=0
TESTS_FAILED=0

# ==============================================================================
# Funciones de Utilidad
# ==============================================================================

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_test() {
    echo -e "${YELLOW}⚙${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    ((TESTS_FAILED++))
}

# ==============================================================================
# Pruebas de Servicios
# ==============================================================================

test_ollama() {
    print_header "1/4 - Probando Ollama Server"
    
    print_test "Verificando que Ollama esté corriendo..."
    if pgrep -x "ollama" > /dev/null; then
        print_success "Ollama está corriendo"
    else
        print_error "Ollama no está corriendo"
        return 1
    fi
    
    print_test "Probando endpoint /api/tags..."
    if curl -s http://localhost:$OLLAMA_PORT/api/tags > /dev/null; then
        print_success "Endpoint /api/tags responde"
    else
        print_error "Endpoint /api/tags no responde"
    fi
    
    print_test "Verificando modelo ready4hire:latest..."
    if ollama list | grep -q "ready4hire:latest"; then
        print_success "Modelo ready4hire:latest disponible"
    else
        print_error "Modelo ready4hire:latest no encontrado"
    fi
    
    echo ""
}

test_api() {
    print_header "2/4 - Probando API Python (FastAPI)"
    
    print_test "Verificando que API esté corriendo en puerto $API_PORT..."
    if lsof -ti:$API_PORT > /dev/null 2>&1; then
        print_success "API está corriendo"
    else
        print_error "API no está corriendo"
        return 1
    fi
    
    print_test "Probando endpoint raíz /..."
    RESPONSE=$(curl -s http://localhost:$API_PORT/)
    if echo "$RESPONSE" | grep -q "Ready4Hire API"; then
        print_success "Endpoint raíz responde correctamente"
    else
        print_error "Endpoint raíz no responde como esperado"
    fi
    
    print_test "Probando endpoint /api/v2/health..."
    RESPONSE=$(curl -s http://localhost:$API_PORT/api/v2/health)
    if echo "$RESPONSE" | grep -q "status"; then
        print_success "Endpoint /api/v2/health responde"
        
        # Verificar componentes
        print_test "Verificando componentes de salud..."
        if echo "$RESPONSE" | grep -q "llm_service.*healthy"; then
            print_success "LLM Service: healthy"
        else
            print_error "LLM Service: no healthy"
        fi
        
        if echo "$RESPONSE" | grep -q "STT: ✅"; then
            print_success "Audio STT (Whisper): healthy"
        else
            print_error "Audio STT: no healthy"
        fi
        
        if echo "$RESPONSE" | grep -q "Embeddings: ✅"; then
            print_success "ML Embeddings: healthy"
        else
            print_error "ML Embeddings: no healthy"
        fi
    else
        print_error "Endpoint /api/v2/health no responde"
    fi
    
    print_test "Probando endpoint /docs (Swagger)..."
    if curl -s http://localhost:$API_PORT/docs | grep -q "swagger"; then
        print_success "Documentación Swagger disponible"
    else
        print_error "Documentación Swagger no disponible"
    fi
    
    echo ""
}

test_webapp() {
    print_header "3/4 - Probando WebApp (Blazor)"
    
    print_test "Verificando que WebApp esté corriendo en puerto $WEBAPP_PORT..."
    if lsof -ti:$WEBAPP_PORT > /dev/null 2>&1; then
        print_success "WebApp está corriendo"
    else
        print_error "WebApp no está corriendo"
        return 1
    fi
    
    print_test "Probando endpoint raíz /..."
    RESPONSE=$(curl -s http://localhost:$WEBAPP_PORT/)
    if echo "$RESPONSE" | grep -q "Ready4Hire"; then
        print_success "WebApp responde correctamente"
    else
        print_error "WebApp no responde como esperado"
    fi
    
    print_test "Verificando que WebApp cargue Bootstrap..."
    if echo "$RESPONSE" | grep -q "bootstrap"; then
        print_success "Bootstrap cargado"
    else
        print_error "Bootstrap no cargado"
    fi
    
    print_test "Verificando página de login..."
    if echo "$RESPONSE" | grep -q "login"; then
        print_success "Página de login disponible"
    else
        print_error "Página de login no encontrada"
    fi
    
    echo ""
}

test_integration() {
    print_header "4/4 - Probando Integración API <-> Ollama"
    
    print_test "Verificando que API puede comunicarse con Ollama..."
    HEALTH=$(curl -s http://localhost:$API_PORT/api/v2/health)
    if echo "$HEALTH" | grep -q "llm_service.*healthy"; then
        print_success "API se comunica correctamente con Ollama"
    else
        print_error "API no puede comunicarse con Ollama"
    fi
    
    print_test "Verificando configuración de WebApp..."
    # La WebApp debe estar configurada para usar puerto 8001
    if [ -f "/home/jeronimorestrepoangel/Documentos/Integracion/WebApp/appsettings.json" ]; then
        if grep -q "8001" "/home/jeronimorestrepoangel/Documentos/Integracion/WebApp/appsettings.json"; then
            print_success "WebApp configurada para usar API en puerto 8001"
        else
            print_error "WebApp no está configurada correctamente"
        fi
    fi
    
    echo ""
}

show_summary() {
    print_header "Resumen de Pruebas"
    
    TOTAL=$((TESTS_PASSED + TESTS_FAILED))
    
    echo ""
    echo -e "${GREEN}✓ Pruebas exitosas: $TESTS_PASSED${NC}"
    echo -e "${RED}✗ Pruebas fallidas: $TESTS_FAILED${NC}"
    echo -e "  Total de pruebas: $TOTAL"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 ¡Todas las pruebas pasaron exitosamente!${NC}"
        echo ""
        echo -e "${BLUE}Sistema Ready4Hire completamente funcional:${NC}"
        echo -e "  • Ollama LLM: http://localhost:$OLLAMA_PORT"
        echo -e "  • API REST: http://localhost:$API_PORT"
        echo -e "  • API Docs: http://localhost:$API_PORT/docs"
        echo -e "  • WebApp: http://localhost:$WEBAPP_PORT"
        echo ""
        return 0
    else
        echo -e "${RED}⚠ Algunas pruebas fallaron. Revisa los errores arriba.${NC}"
        echo ""
        return 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    echo ""
    echo -e "${BLUE}"
    echo "  ____                _       _  _   _   _  _          "
    echo " |  _ \ ___  __ _  __| |_   _| || | | | | |(_)_ __ ___ "
    echo " | |_) / _ \/ _\` |/ _\` | | | | || |_| |_| || | '__/ _ \\"
    echo " |  _ <  __/ (_| | (_| | |_| |__   _|  _  || | | |  __/"
    echo " |_| \_\___|\__,_|\__,_|\__, |  |_| |_| |_||_|_|  \___|"
    echo "                        |___/                          "
    echo -e "${NC}"
    echo -e "${GREEN}Pruebas de Integración${NC}"
    echo ""
    
    # Ejecutar pruebas
    test_ollama || true
    test_api || true
    test_webapp || true
    test_integration || true
    show_summary
}

# Ejecutar
main "$@"
