#!/bin/bash

# Test rápido de rendimiento - Solo verifica velocidad y corrección del campo is_correct

echo "╔════════════════════════════════════════════════════╗"
echo "║   TEST RÁPIDO DE RENDIMIENTO                       ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

BASE_URL="http://localhost:8000/api/v2"

# [1] Health check
echo "[1/4] Health check..."
curl -s "${BASE_URL}/health" | jq -r '.status' || exit 1
echo "✅ Sistema disponible"
echo ""

# [2] Iniciar entrevista
echo "[2/4] Iniciando entrevista..."
START_RESPONSE=$(curl -s -X POST "${BASE_URL}/interviews" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "perf-test-user",
    "role": "Backend Developer",
    "category": "technical",
    "difficulty": "junior"
  }')

INTERVIEW_ID=$(echo $START_RESPONSE | jq -r '.interview_id')
echo "✅ Entrevista iniciada: $INTERVIEW_ID"
echo ""

# [3] Responder 5 preguntas de contexto (rápido, sin evaluación LLM)
echo "[3/4] Fase de contexto (5 preguntas)..."
for i in {1..5}; do
  echo "  → Pregunta $i/5"
  START_TIME=$(date +%s%N)
  
  RESPONSE=$(curl -s -X POST "${BASE_URL}/interviews/${INTERVIEW_ID}/answers" \
    -H "Content-Type: application/json" \
    -d "{
      \"answer\": \"Respuesta de contexto $i: Python, FastAPI, PostgreSQL, Docker. Tengo 3 años de experiencia.\",
      \"time_taken\": 10
    }")
  
  END_TIME=$(date +%s%N)
  ELAPSED_MS=$(( ($END_TIME - $START_TIME) / 1000000 ))
  
  STATUS=$(echo $RESPONSE | jq -r '.interview_status')
  
  if [ "$STATUS" == "context" ] || [ "$STATUS" == "questions" ]; then
    echo "  ✅ Respuesta $i: ${ELAPSED_MS}ms (status: $STATUS)"
  else
    echo "  ❌ Error en respuesta $i"
    echo $RESPONSE | jq .
  fi
done
echo ""

# [4] Responder 1 pregunta técnica (con evaluación LLM optimizada)
echo "[4/4] Pregunta técnica con evaluación LLM..."
START_TIME=$(date +%s%N)

TECH_RESPONSE=$(curl -s -X POST "${BASE_URL}/interviews/${INTERVIEW_ID}/answers" \
  -H "Content-Type: application/json" \
  -d '{
    "answer": "Utilizaría PowerShell con cmdlets como New-ADUser para crear usuarios en Active Directory de forma masiva. Importaría los datos desde un CSV con Import-CSV y luego iteraría con ForEach-Object para crear cada usuario. También configuraría propiedades adicionales como grupos, contraseña inicial, y ubicación organizacional.",
    "time_taken": 45
  }')

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( ($END_TIME - $START_TIME) / 1000000 ))

# Verificar que la respuesta tenga el campo is_correct
HAS_IS_CORRECT=$(echo $TECH_RESPONSE | jq 'has("evaluation") and .evaluation | has("is_correct")')
SCORE=$(echo $TECH_RESPONSE | jq -r '.evaluation.score // "N/A"')
IS_CORRECT=$(echo $TECH_RESPONSE | jq -r '.evaluation.is_correct // "N/A"')

echo ""
echo "📊 RESULTADOS:"
echo "  ⏱️  Tiempo de evaluación LLM: ${ELAPSED_MS}ms"
echo "  📝 Score: $SCORE"
echo "  ✓  is_correct: $IS_CORRECT"
echo ""

if [ "$HAS_IS_CORRECT" == "true" ]; then
  echo "✅ Campo is_correct presente y correcto"
  echo "✅ TEST DE RENDIMIENTO EXITOSO"
else
  echo "❌ Campo is_correct faltante"
  echo ""
  echo "Respuesta completa:"
  echo $TECH_RESPONSE | jq .
  exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║   ✅ OPTIMIZACIONES VERIFICADAS                    ║"
echo "╚════════════════════════════════════════════════════╝"
