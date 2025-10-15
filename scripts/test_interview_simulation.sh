#!/usr/bin/env bash
#
# Ready4Hire - Interview Simulation Test Script
# Tests the complete interview workflow with the API
#

set -euo pipefail

API_URL="http://localhost:8001"
API_V2="${API_URL}/api/v2"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Test 1: Health Check
test_health() {
    log_info "Testing API health..."
    
    RESPONSE=$(curl -s -w "\n%{http_code}" "${API_V2}/health" || echo "000")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        log_success "API is healthy"
        echo "$BODY" | python3 -m json.tool | grep -E "(status|llm_service|repositories)" || true
        return 0
    else
        log_error "API is not healthy (HTTP $HTTP_CODE)"
        return 1
    fi
}

# Test 2: Start Interview
start_interview() {
    log_info "Starting technical interview..."
    
    USER_ID="test_user_$(date +%s)"
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${API_V2}/interviews" \
        -H "Content-Type: application/json" \
        -d "{
            \"user_id\": \"$USER_ID\",
            \"role\": \"Backend Developer\",
            \"type\": \"technical\",
            \"mode\": \"practice\"
        }" || echo "000")
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 201 ]; then
        INTERVIEW_ID=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin)['interview_id'])" 2>/dev/null)
        
        if [ -n "$INTERVIEW_ID" ]; then
            log_success "Interview started: $INTERVIEW_ID"
            
            QUESTION=$(echo "$BODY" | python3 -c "import sys, json; q = json.load(sys.stdin)['first_question']; print(q.get('text', q.get('question', 'N/A')))" 2>/dev/null)
            echo "   First Question: ${QUESTION:0:100}..."
            
            echo "$INTERVIEW_ID" > /tmp/interview_id.txt
            echo "$USER_ID" > /tmp/user_id.txt
            return 0
        fi
    fi
    
    log_error "Failed to start interview (HTTP $HTTP_CODE)"
    return 1
}

# Test 3: Submit Answer
submit_answer() {
    local INTERVIEW_ID=$(cat /tmp/interview_id.txt 2>/dev/null || echo "")
    local ANSWER="$1"
    
    if [ -z "$INTERVIEW_ID" ]; then
        log_error "No active interview"
        return 1
    fi
    
    log_info "Submitting answer (this may take 1-2 minutes)..."
    echo "   Answer: ${ANSWER:0:80}..."
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        "${API_V2}/interviews/${INTERVIEW_ID}/answers" \
        -H "Content-Type: application/json" \
        -d "{
            \"answer\": \"$ANSWER\",
            \"time_taken\": 30
        }" \
        --max-time 180 || echo "000")
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        SCORE=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('score', 'N/A'))" 2>/dev/null || echo "N/A")
        FEEDBACK=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('feedback', 'N/A')[:100])" 2>/dev/null || echo "N/A")
        
        log_success "Answer evaluated"
        echo "   Score: ${SCORE}/10"
        echo "   Feedback: ${FEEDBACK}..."
        return 0
    else
        log_error "Failed to submit answer (HTTP $HTTP_CODE)"
        return 1
    fi
}

# Test 4: End Interview
end_interview() {
    local INTERVIEW_ID=$(cat /tmp/interview_id.txt 2>/dev/null || echo "")
    local USER_ID=$(cat /tmp/user_id.txt 2>/dev/null || echo "")
    
    if [ -z "$INTERVIEW_ID" ]; then
        log_error "No active interview"
        return 1
    fi
    
    log_info "Ending interview..."
    
    RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
        "${API_V2}/interviews/${INTERVIEW_ID}/end" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\": \"$USER_ID\"}" \
        --max-time 60 || echo "000")
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        OVERALL_SCORE=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('summary', {}).get('overall_score', 'N/A'))" 2>/dev/null || echo "N/A")
        QUESTIONS_ANSWERED=$(echo "$BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('summary', {}).get('questions_answered', 0))" 2>/dev/null || echo "0")
        
        log_success "Interview ended"
        echo "   Overall Score: ${OVERALL_SCORE}/10"
        echo "   Questions Answered: ${QUESTIONS_ANSWERED}"
        return 0
    else
        log_error "Failed to end interview (HTTP $HTTP_CODE)"
        return 1
    fi
}

# Test 5: Get Metrics
test_metrics() {
    log_info "Getting system metrics..."
    
    RESPONSE=$(curl -s -w "\n%{http_code}" "${API_V2}/metrics" || echo "000")
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        log_success "Metrics retrieved"
        echo "$BODY" | python3 -m json.tool | head -10 || true
        return 0
    else
        log_warning "Could not retrieve metrics (HTTP $HTTP_CODE)"
        return 0  # Not critical
    fi
}

# Main test flow
main() {
    echo "======================================================================"
    echo "  Ready4Hire - Interview Simulation Test"
    echo "  Timestamp: $(date -Iseconds)"
    echo "======================================================================"
    echo ""
    
    # Test 1: Health
    if ! test_health; then
        log_error "API is not available. Please start services first:"
        echo "   ./scripts/run.sh --dev"
        exit 1
    fi
    echo ""
    
    # Test 2: Start interview
    if ! start_interview; then
        exit 1
    fi
    echo ""
    
    # Test 3: Submit one answer
    ANSWER1="SOLID principles are five design principles in object-oriented programming: Single Responsibility Principle (each class should have one responsibility), Open-Closed Principle (open for extension, closed for modification), Liskov Substitution Principle (subtypes should be substitutable for their base types), Interface Segregation Principle (clients shouldn't depend on interfaces they don't use), and Dependency Inversion Principle (depend on abstractions, not concretions)."
    
    if ! submit_answer "$ANSWER1"; then
        log_warning "Could not complete full interview flow"
    fi
    echo ""
    
    # Test 4: End interview
    if ! end_interview; then
        log_warning "Could not end interview properly"
    fi
    echo ""
    
    # Test 5: Metrics
    test_metrics
    echo ""
    
    # Cleanup
    rm -f /tmp/interview_id.txt /tmp/user_id.txt
    
    echo "======================================================================"
    log_success "Simulation Complete!"
    echo "======================================================================"
}

main "$@"
