# 🧪 FULL SYSTEM TEST - Ready4Hire
## Complete End-to-End Testing Guide

---

## 📋 Table of Contents
1. [Pre-Test Setup](#pre-test-setup)
2. [Backend Tests (Python)](#backend-tests-python)
3. [Frontend Tests (.NET)](#frontend-tests-net)
4. [Mobile Tests (React Native)](#mobile-tests-react-native)
5. [AI/LLM Tests](#aillm-tests)
6. [Database Tests](#database-tests)
7. [E2E Tests (Playwright)](#e2e-tests-playwright)
8. [Performance Tests](#performance-tests)
9. [Security Tests](#security-tests)
10. [Integration Tests](#integration-tests)

---

## 🚀 Pre-Test Setup

### 1. Start All Services
```bash
# Backend (Python)
cd Ready4Hire
source venv/bin/activate
python -m uvicorn app.main_v2_improved:app --reload &

# Frontend (.NET)
cd ../WebApp
dotnet run &

# Ollama (AI)
ollama serve &
ollama pull llama3.2:3b

# PostgreSQL
docker-compose up postgres -d

# Wait for services to be ready
sleep 10
```

### 2. Verify Services are Running
```bash
curl http://localhost:8000/api/v2/health  # Backend
curl http://localhost:5214                # Frontend
curl http://localhost:11434/api/tags      # Ollama
```

---

## 🐍 Backend Tests (Python)

### Unit Tests
```bash
cd Ready4Hire
pytest tests/unit/ -v --cov=app/domain --cov=app/application
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Performance Tests
```bash
pytest tests/performance/ -v
```

### All Backend Tests
```bash
pytest tests/ -v --cov=app --cov-report=html --cov-report=term
```

**Expected Results:**
- ✅ 50+ tests should pass
- ✅ Coverage: 30%+
- ✅ No critical failures

---

## 🌐 Frontend Tests (.NET)

### Build Test
```bash
cd WebApp
dotnet build --configuration Release
```

### Run Tests (if available)
```bash
dotnet test --verbosity normal
```

### Manual UI Test
```bash
# Open browser
open http://localhost:5214

# Test checklist:
# ✅ Login page loads
# ✅ Registration works
# ✅ Navigation functional
# ✅ Chat interface responsive
# ✅ Gamification displays
```

---

## 📱 Mobile Tests (React Native)

### Install Dependencies
```bash
cd MobileApp
npm install
```

### Lint Check
```bash
npm run lint
```

### Unit Tests
```bash
npm test -- --coverage
```

### Build Test (Android)
```bash
npm run android -- --no-packager
```

**Expected Results:**
- ✅ All tests pass
- ✅ No linting errors
- ✅ Build succeeds

---

## 🤖 AI/LLM Tests

### Test Ollama Connection
```bash
curl http://localhost:11434/api/tags
```

### Test LLM Generation
```bash
curl http://localhost:8000/api/v2/interviews \\
  -X POST \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "test_user",
    "role": "Python Developer",
    "difficulty": "mid",
    "category": "backend"
  }'
```

### Test Evaluation Service
```python
# In Python REPL
from app.container import Container

c = Container(ollama_url="http://localhost:11434")
result = c.evaluation_service.evaluate_answer(
    question="What is Python?",
    answer="Python is a programming language",
    expected_concepts=["programming", "language"],
    keywords=["python"],
    category="technical",
    difficulty="mid",
    role="Developer"
)
print(result)
```

**Expected Results:**
- ✅ Ollama responds
- ✅ Interview creation works
- ✅ Evaluation returns valid scores

---

## 🗄️ Database Tests

### PostgreSQL Connection
```bash
psql -h localhost -U ready4hire_user -d ready4hire_db -c "SELECT version();"
```

### Test Data Insertion
```bash
psql -h localhost -U ready4hire_user -d ready4hire_db << EOF
INSERT INTO "Users" ("Email", "PasswordHash", "DateCreated")
VALUES ('test@test.com', 'hash123', NOW());

SELECT * FROM "Users" WHERE "Email" = 'test@test.com';
EOF
```

### Test Migrations
```bash
cd WebApp
dotnet ef database update
```

**Expected Results:**
- ✅ Database accessible
- ✅ Insertions work
- ✅ Migrations apply cleanly

---

## 🎭 E2E Tests (Playwright)

### Install Playwright
```bash
cd e2e-tests
npm install
npx playwright install
```

### Run All E2E Tests
```bash
npx playwright test
```

### Run with UI
```bash
npx playwright test --ui
```

### Run Specific Test
```bash
npx playwright test tests/01-home.spec.ts
```

### Generate Test Report
```bash
npx playwright show-report
```

**Expected Results:**
- ✅ All E2E tests pass
- ✅ Screenshots captured
- ✅ No timeout errors

---

## ⚡ Performance Tests

### Backend Response Times
```bash
cd Ready4Hire
pytest tests/performance/test_response_times.py -v
```

### Load Test with Apache Bench
```bash
# Health endpoint
ab -n 1000 -c 10 http://localhost:8000/api/v2/health

# Root endpoint
ab -n 1000 -c 10 http://localhost:8000/
```

### Frontend Load
```bash
ab -n 500 -c 5 http://localhost:5214/
```

**Expected Results:**
- ✅ Health endpoint: < 2s average
- ✅ Root endpoint: < 1s average
- ✅ No 500 errors
- ✅ 99th percentile < 5s

---

## 🔒 Security Tests

### Bandit (Python)
```bash
cd Ready4Hire
bandit -c .bandit -r app -f html -o security-report.html
```

### Safety (Dependencies)
```bash
safety check --json
```

### npm audit (Mobile)
```bash
cd MobileApp
npm audit
```

### Trivy (Docker)
```bash
trivy image ready4hire:latest
```

**Expected Results:**
- ✅ No HIGH/CRITICAL vulnerabilities
- ✅ All dependencies up to date
- ✅ Security best practices followed

---

## 🔗 Integration Tests

### Full Interview Flow
```bash
# 1. Start interview
curl -X POST http://localhost:8000/api/v2/interviews \\
  -H "Content-Type: application/json" \\
  -d '{"user_id":"test","role":"Developer","difficulty":"mid","category":"backend"}'

# 2. Answer context questions (5x)
curl -X POST http://localhost:8000/api/v2/interviews/{interview_id}/answers \\
  -H "Content-Type: application/json" \\
  -d '{"answer":"I have 3 years of experience"}'

# 3. Answer technical questions (10x)
# ... repeat ...

# 4. Get results
curl http://localhost:8000/api/v2/interviews/{interview_id}
```

### Gamification Flow
```bash
# Get user stats
curl http://localhost:8000/api/v2/gamification/users/test/stats

# Get leaderboard
curl http://localhost:8000/api/v2/gamification/leaderboard
```

**Expected Results:**
- ✅ Complete flow works
- ✅ Data persists
- ✅ Gamification calculates correctly

---

## ✅ Test Checklist

### Pre-Production Checklist
- [ ] All unit tests pass (50+)
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Performance benchmarks met
- [ ] Security scan clean
- [ ] Database migrations work
- [ ] Frontend builds successfully
- [ ] Mobile app compiles
- [ ] AI/LLM responds correctly
- [ ] Monitoring configured
- [ ] Logs are clean
- [ ] No memory leaks
- [ ] Error handling works
- [ ] CORS configured
- [ ] Authentication works

---

## 📊 Expected Final Results

```
Backend (Python):     ✅ 50/50 tests passing
Frontend (.NET):      ✅ Build successful
Mobile (RN):          ✅ Tests passing
E2E (Playwright):     ✅ All scenarios pass
Performance:          ✅ Within limits
Security:             ✅ No critical issues
Integration:          ✅ All flows work
Database:             ✅ Migrations applied
AI/LLM:              ✅ Generating responses
Monitoring:           ✅ Metrics collecting

Overall Status:       🟢 READY FOR PRODUCTION
```

---

## 🚨 Troubleshooting

### If Tests Fail

1. **Check Services:**
   ```bash
   ps aux | grep -E 'uvicorn|dotnet|ollama|postgres'
   ```

2. **Check Logs:**
   ```bash
   tail -f logs/ready4hire_api.log
   tail -f logs/webapp.log
   ```

3. **Reset Database:**
   ```bash
   cd WebApp
   dotnet ef database drop --force
   dotnet ef database update
   ```

4. **Clear Cache:**
   ```bash
   cd Ready4Hire
   rm -rf __pycache__ .pytest_cache htmlcov/
   ```

---

**Test Duration:** ~15-20 minutes for full suite  
**Last Updated:** 2025-10-24  
**Version:** Ready4Hire v3.3

