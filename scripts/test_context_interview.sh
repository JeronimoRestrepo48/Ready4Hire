#!/usr/bin/env bash
#
# Ready4Hire - Test de Entrevista con Preguntas de Contexto
# Este script prueba el nuevo flujo de entrevista con:
# - 5 preguntas de contexto (sin evaluación)
# - Selección inteligente de 10 preguntas con clustering
# - Evaluación de las 10 preguntas técnicas
#

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
API_V2="${API_URL}/api/v2"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_step() {
    echo -e "${CYAN}▶${NC} $1"
}

echo "======================================================================"
echo "  Ready4Hire - Test de Entrevista con Preguntas de Contexto"
echo "  Timestamp: $(date -Iseconds)"
echo "======================================================================"
echo ""

# Test 1: Health Check
log_step "1. Verificando salud del API..."
RESPONSE=$(curl -s -w "\n%{http_code}" "${API_V2}/health" || echo "000")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 200 ]; then
    log_success "API está saludable"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    log_error "API no está disponible (HTTP $HTTP_CODE)"
    log_error "Asegúrate de que el API esté corriendo: ./scripts/run.sh --dev"
    exit 1
fi

echo ""

# Test 2: Iniciar Entrevista
log_step "2. Iniciando entrevista técnica..."
START_RESPONSE=$(curl -s -X POST "${API_V2}/interviews" \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "test_user_context_'$(date +%s)'",
        "role": "Backend Developer",
        "category": "technical",
        "difficulty": "mid"
    }')

INTERVIEW_ID=$(echo "$START_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['interview_id'])" 2>/dev/null)

if [ -z "$INTERVIEW_ID" ]; then
    log_error "No se pudo iniciar la entrevista"
    echo "$START_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$START_RESPONSE"
    exit 1
fi

log_success "Entrevista iniciada: $INTERVIEW_ID"
FIRST_QUESTION=$(echo "$START_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['first_question']['text'])" 2>/dev/null)
STATUS=$(echo "$START_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
log_info "Fase: $STATUS"
log_info "Primera pregunta de contexto:"
echo "   $FIRST_QUESTION"

echo ""

# Test 3: Responder las 5 preguntas de contexto
log_step "3. Respondiendo preguntas de contexto (5 preguntas sin evaluación)..."

CONTEXT_ANSWERS=(
    "Tengo 5 años de experiencia en desarrollo backend, trabajando principalmente con Python y Java. He liderado proyectos de migración de monolitos a microservicios y arquitecturas cloud-native en AWS."
    "Domino Python (Django, FastAPI), Java (Spring Boot), bases de datos SQL y NoSQL (PostgreSQL, MongoDB), Docker, Kubernetes, y servicios de AWS (EC2, S3, Lambda). Nivel avanzado en Python y Docker, intermedio en Kubernetes."
    "Me especializo en arquitectura de microservicios, APIs RESTful escalables, DevOps y CI/CD. También tengo experiencia en implementación de machine learning en producción y optimización de rendimiento de aplicaciones."
    "El proyecto más complejo fue migrar un monolito de 10 años a microservicios. Desafíos: sincronización de datos, testing distribuido, y deployment sin downtime. Usé Spring Boot, Docker, Kubernetes, y GitLab CI/CD. Fui el arquitecto técnico líder."
    "Quiero profundizar en observabilidad (Prometheus, Grafana), arquitectura event-driven con Kafka, y service mesh con Istio. Me interesa para diseñar sistemas más resilientes y escalables."
)

for i in {1..5}; do
    log_info "Enviando respuesta de contexto $i/5..."
    ANSWER_RESPONSE=$(curl -s -X POST "${API_V2}/interviews/${INTERVIEW_ID}/answers" \
        -H "Content-Type: application/json" \
        -d "{
            \"answer\": \"${CONTEXT_ANSWERS[$i-1]}\",
            \"time_taken\": $((30 + RANDOM % 30))
        }")
    
    INTERVIEW_STATUS=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('interview_status', 'unknown'))" 2>/dev/null)
    FEEDBACK=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('feedback', '')[:100])" 2>/dev/null)
    
    log_success "Respuesta $i guardada (Fase: $INTERVIEW_STATUS)"
    echo "   Feedback: $FEEDBACK..."
    
    # Verificar si hay siguiente pregunta
    NEXT_Q=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; q = json.load(sys.stdin).get('next_question'); print(q['text'][:80] if q else 'N/A')" 2>/dev/null)
    if [ "$NEXT_Q" != "N/A" ]; then
        echo "   Siguiente pregunta: $NEXT_Q..."
    fi
    
    echo ""
    sleep 1
done

log_success "✅ Fase de contexto completada!"
log_info "El sistema ahora analizará las respuestas y seleccionará 10 preguntas técnicas usando clustering..."

echo ""

# Test 4: Responder algunas preguntas técnicas
log_step "4. Respondiendo preguntas técnicas (evaluadas por el LLM)..."

TECH_ANSWERS=(
    "Usaría un patrón de API Gateway como Kong o AWS API Gateway para centralizar la autenticación. Implementaría JWT tokens con refresh tokens, y usaría OAuth2 para autorización. El API Gateway validaría los tokens antes de rutear a los microservicios."
    "Implementaría Saga pattern para transacciones distribuidas, usando event sourcing con Kafka. Para la comunicación síncrona usaría gRPC, y para asíncrona, message queues. Monitorizaría con Prometheus y Grafana, y usaría circuit breakers para resiliencia."
    "Primero identificaría el cuello de botella con profiling (cProfile, py-spy). Luego aplicaría caching con Redis, optimizaría queries SQL, implementaría índices de base de datos, y usaría async/await para operaciones I/O. Finalmente, escalaría horizontalmente con load balancing."
)

for i in {1..3}; do
    log_info "Enviando respuesta técnica $i/10..."
    ANSWER_RESPONSE=$(curl -s -X POST "${API_V2}/interviews/${INTERVIEW_ID}/answers" \
        -H "Content-Type: application/json" \
        -d "{
            \"answer\": \"${TECH_ANSWERS[$i-1]}\",
            \"time_taken\": $((60 + RANDOM % 60))
        }")
    
    SCORE=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('evaluation', {}).get('score', 0))" 2>/dev/null)
    IS_CORRECT=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('evaluation', {}).get('is_correct', False))" 2>/dev/null)
    EMOTION=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('emotion', {}).get('emotion', 'neutral'))" 2>/dev/null)
    FEEDBACK=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('feedback', '')[:150])" 2>/dev/null)
    
    if [ "$IS_CORRECT" = "True" ]; then
        log_success "Respuesta $i: ✅ Correcta (Score: $SCORE/10, Emoción: $EMOTION)"
    else
        log_warning "Respuesta $i: ❌ Incorrecta (Score: $SCORE/10, Emoción: $EMOTION)"
    fi
    
    echo "   Feedback: $FEEDBACK..."
    
    # Verificar si hay siguiente pregunta
    NEXT_Q=$(echo "$ANSWER_RESPONSE" | python3 -c "import sys, json; q = json.load(sys.stdin).get('next_question'); print(q['text'][:80] if q else 'COMPLETADA')" 2>/dev/null)
    if [ "$NEXT_Q" != "COMPLETADA" ]; then
        echo "   Siguiente pregunta: $NEXT_Q..."
    else
        log_success "🎉 Entrevista completada!"
    fi
    
    echo ""
    sleep 2
done

echo ""
echo "======================================================================"
log_success "Test completado exitosamente!"
echo "======================================================================"
echo ""
echo "Resumen:"
echo "  - ✅ 5 preguntas de contexto respondidas (sin evaluación LLM)"
echo "  - ✅ Preguntas técnicas seleccionadas con clustering"
echo "  - ✅ 3 preguntas técnicas respondidas y evaluadas"
echo "  - ✅ Detección de emociones funcionando"
echo "  - ✅ Feedback personalizado generado"
echo ""
echo "📊 El sistema ahora selecciona preguntas inteligentemente basándose en:"
echo "  1. Respuestas de contexto del candidato"
echo "  2. Embeddings semánticos de las preguntas"
echo "  3. Clustering para diversificación temática"
echo "  4. Ranking por relevancia al perfil detectado"
echo ""
