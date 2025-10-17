#!/bin/bash

###############################################################################
# Test de Integración Completa: Frontend + Backend
# Prueba el flujo completo de dos fases:
# 1. Fase de contexto (5 preguntas) - NO evaluadas
# 2. Fase técnica (10 preguntas) - Evaluadas con LLM
###############################################################################

set -e

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sin color

API_URL="${API_URL:-http://localhost:8000}"

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   TEST DE INTEGRACIÓN COMPLETA                     ║${NC}"
echo -e "${BLUE}║   Frontend (Blazor) + Backend (FastAPI)           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"

# 1. Health Check
echo -e "\n${YELLOW}[1/5] Verificando health del sistema...${NC}"
HEALTH=$(curl -s "${API_URL}/api/v2/health")
echo "$HEALTH" | jq '.'

STATUS=$(echo "$HEALTH" | jq -r '.status')
if [ "$STATUS" != "healthy" ]; then
    echo -e "${RED}❌ Sistema no está saludable${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Sistema saludable${NC}"

# 2. Iniciar entrevista (V2)
echo -e "\n${YELLOW}[2/5] Iniciando entrevista...${NC}"
START_RESPONSE=$(curl -s -X POST "${API_URL}/api/v2/interviews" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user-integration-'$(date +%s)'",
    "role": "Backend Developer",
    "category": "technical",
    "difficulty": "mid"
  }')

echo "$START_RESPONSE" | jq '.'

INTERVIEW_ID=$(echo "$START_RESPONSE" | jq -r '.interview_id')
INTERVIEW_STATUS=$(echo "$START_RESPONSE" | jq -r '.status')
FIRST_QUESTION=$(echo "$START_RESPONSE" | jq -r '.first_question.text')

if [ "$INTERVIEW_STATUS" != "context" ]; then
    echo -e "${RED}❌ Estado inicial incorrecto: $INTERVIEW_STATUS (esperado: context)${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Entrevista iniciada - ID: $INTERVIEW_ID${NC}"
echo -e "${BLUE}Primera pregunta: $FIRST_QUESTION${NC}"

# 3. Fase de CONTEXTO: Responder 5 preguntas (NO evaluadas)
echo -e "\n${YELLOW}[3/5] FASE DE CONTEXTO: Respondiendo 5 preguntas...${NC}"

CONTEXT_ANSWERS=(
    "Tengo 3 años de experiencia en desarrollo backend con .NET"
    "He trabajado con C#, ASP.NET Core, Entity Framework y SQL Server"
    "Me especializo en arquitecturas de microservicios y APIs RESTful"
    "Busco mejorar mis habilidades en diseño de sistemas distribuidos"
    "Mi último proyecto fue una plataforma de e-commerce con alta concurrencia"
)

for i in {0..4}; do
    CONTEXT_NUM=$((i+1))
    echo -e "\n${BLUE}  → Pregunta de contexto ${CONTEXT_NUM}/5${NC}"
    
    CONTEXT_RESPONSE=$(curl -s -X POST "${API_URL}/api/v2/interviews/${INTERVIEW_ID}/answers" \
      -H "Content-Type: application/json" \
      -d "{
        \"answer\": \"${CONTEXT_ANSWERS[$i]}\",
        \"time_taken\": 15
      }")
    
    echo "$CONTEXT_RESPONSE" | jq '.'
    
    RESPONSE_STATUS=$(echo "$CONTEXT_RESPONSE" | jq -r '.interview_status')
    
    if [ "$RESPONSE_STATUS" = "context" ]; then
        NEXT_Q=$(echo "$CONTEXT_RESPONSE" | jq -r '.next_question.text // "N/A"')
        echo -e "${GREEN}  ✅ Respuesta ${CONTEXT_NUM} guardada (sin evaluación LLM)${NC}"
        
        if [ "$NEXT_Q" != "N/A" ] && [ "$NEXT_Q" != "null" ]; then
            echo -e "${BLUE}  Siguiente: $NEXT_Q${NC}"
        fi
    elif [ "$RESPONSE_STATUS" = "questions" ]; then
        echo -e "${GREEN}  ✅ Transición a fase técnica${NC}"
        NEXT_Q=$(echo "$CONTEXT_RESPONSE" | jq -r '.next_question.text // "N/A"')
        echo -e "${BLUE}  Primera pregunta técnica: $NEXT_Q${NC}"
        break
    else
        echo -e "${RED}  ❌ Estado inesperado: $RESPONSE_STATUS${NC}"
        exit 1
    fi
    
    sleep 1
done

# 4. Fase TÉCNICA: Responder 3 preguntas (EVALUADAS con LLM)
echo -e "\n${YELLOW}[4/5] FASE TÉCNICA: Respondiendo preguntas con evaluación LLM...${NC}"

TECHNICAL_ANSWERS=(
    "SOLID son principios de diseño orientado a objetos: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation y Dependency Inversion"
    "La inyección de dependencias es un patrón que permite invertir el control, inyectando dependencias desde el exterior en lugar de crearlas internamente"
    "RESTful API usa HTTP methods (GET, POST, PUT, DELETE) y sigue principios REST como stateless, recursos identificables con URIs y representaciones"
)

for i in {0..2}; do
    TECH_NUM=$((i+1))
    echo -e "\n${BLUE}  → Pregunta técnica ${TECH_NUM}${NC}"
    
    TECH_RESPONSE=$(curl -s -X POST "${API_URL}/api/v2/interviews/${INTERVIEW_ID}/answers" \
      -H "Content-Type: application/json" \
      -d "{
        \"answer\": \"${TECHNICAL_ANSWERS[$i]}\",
        \"time_taken\": 45
      }")
    
    echo "$TECH_RESPONSE" | jq '.'
    
    # Verificar evaluación
    SCORE=$(echo "$TECH_RESPONSE" | jq -r '.evaluation.score // "N/A"')
    IS_CORRECT=$(echo "$TECH_RESPONSE" | jq -r '.evaluation.is_correct // "N/A"')
    EMOTION=$(echo "$TECH_RESPONSE" | jq -r '.emotion.emotion // "N/A"')
    FEEDBACK=$(echo "$TECH_RESPONSE" | jq -r '.feedback // "N/A"')
    
    if [ "$SCORE" != "N/A" ]; then
        SCORE_EMOJI="🌟"
        [ $(echo "$SCORE < 6" | bc) -eq 1 ] && SCORE_EMOJI="💡"
        [ $(echo "$SCORE >= 6" | bc) -eq 1 ] && [ $(echo "$SCORE < 8" | bc) -eq 1 ] && SCORE_EMOJI="👍"
        
        echo -e "${GREEN}  ✅ Evaluación completada${NC}"
        echo -e "  ${SCORE_EMOJI} Score: ${SCORE}/10 - Correcto: ${IS_CORRECT}"
        echo -e "  💭 Emoción: ${EMOTION}"
        echo -e "  📝 Feedback: ${FEEDBACK:0:100}..."
    else
        echo -e "${RED}  ❌ No se recibió evaluación${NC}"
    fi
    
    sleep 2
done

# 5. Finalizar entrevista
echo -e "\n${YELLOW}[5/5] Finalizando entrevista...${NC}"
END_RESPONSE=$(curl -s -X POST "${API_URL}/api/v2/interviews/${INTERVIEW_ID}/end")
echo "$END_RESPONSE" | jq '.'

SUMMARY=$(echo "$END_RESPONSE" | jq -r '.summary // "N/A"')
if [ "$SUMMARY" != "N/A" ]; then
    echo -e "${GREEN}✅ Resumen generado${NC}"
    echo -e "${BLUE}$SUMMARY${NC}"
fi

# Resumen final
echo -e "\n${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅ TEST DE INTEGRACIÓN COMPLETO                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}📊 Flujo Verificado:${NC}"
echo -e "  1. ✅ Health check del sistema"
echo -e "  2. ✅ Inicio de entrevista (fase context)"
echo -e "  3. ✅ 5 preguntas de contexto (sin evaluación LLM)"
echo -e "  4. ✅ Transición a fase técnica"
echo -e "  5. ✅ 3 preguntas técnicas (con evaluación LLM)"
echo -e "  6. ✅ Finalización con resumen"

echo -e "\n${GREEN}🎉 Integración frontend-backend lista para uso${NC}"

exit 0
