# Ready4Hire - Modular AI Interview Simulation System

## Overview

Ready4Hire is now redesigned with a **modular architecture** that separates concerns into distinct layers, enabling easy scaling, component replacement, and independent deployment. The system simulates technical and soft-skills interviews using AI/LLM technology with advanced features like adaptive questioning, emotional analysis, and gamification.

## 🏗️ New Modular Architecture

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │        Routes & Endpoints & Request Handling            │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Core Business Logic Layer                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Interview Service │ Question Service │ Audio Service   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │   Repository Pattern │ Data Models │ File/DB Access    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    LLM Layer                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │    Ollama │ OpenAI │ Anthropic │ Provider Management   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Service Components

- **Backend Service** (Port 8000): Core API and business logic
- **Frontend Service** (Port 3000): Static web interface with Nginx
- **LLM Service** (Port 11434): Ollama or cloud LLM providers
- **Database Service** (Port 5432): Optional PostgreSQL for persistence

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
git clone https://github.com/JeronimoRestrepo48/Ready4Hire.git
cd Ready4Hire
chmod +x setup.sh
./setup.sh full
```

### Option 2: Manual Setup
```bash
# 1. Setup environment
cp .env.example .env

# 2. Deploy with Docker Compose
docker-compose up -d

# 3. Initialize LLM
docker exec -it ready4hire-ollama ollama pull llama3.2

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

## 🐳 Docker Deployment

### Development Environment
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Production Environment
```bash
docker-compose up -d
```

### Individual Services
```bash
# Backend only
docker-compose up -d backend

# Frontend only
docker-compose up -d frontend

# LLM service only
docker-compose up -d ollama
```

## ⚙️ Configuration

### Environment Variables

The system is configured through environment variables for maximum flexibility:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# LLM Configuration - Easy to switch providers!
LLM_PROVIDER=ollama          # or 'openai', 'anthropic'
LLM_MODEL=llama3.2          # or 'gpt-3.5-turbo', 'claude-3-sonnet'
LLM_BASE_URL=http://ollama:11434
LLM_API_KEY=your-api-key    # for commercial providers

# Data Configuration
DATA_REPOSITORY=filesystem   # or 'database'
DATA_PATH=./app/datasets
DATABASE_URL=postgresql://user:pass@db:5432/ready4hire
```

### LLM Provider Switching

**Local Ollama (Default):**
```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_BASE_URL=http://ollama:11434
```

**OpenAI:**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-openai-key
```

**Anthropic:**
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet
LLM_API_KEY=your-anthropic-key
```

The system automatically handles fallbacks between providers!

## 🏢 Scalability Features

### Independent Service Deployment
- Each service can be deployed, updated, and scaled independently
- Load balancing ready
- Microservices architecture

### LLM Flexibility
- Switch between local and cloud LLM providers without code changes
- Automatic fallback mechanism
- Cost optimization through provider selection

### Data Layer Abstraction
- Filesystem storage (default) - no database needed
- PostgreSQL support for advanced features
- Easy migration between storage types

### Container Orchestration
- Docker Compose for simple deployment
- Kubernetes ready for cloud deployment
- Health checks and automatic restarts

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Detailed architecture documentation
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Comprehensive deployment instructions
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when running)

## 🧪 Testing

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# LLM status
curl http://localhost:8000/llm/status

# Available roles
curl http://localhost:8000/get_roles
```

### Service Testing
```bash
# View service status
docker-compose ps

# View service logs
docker-compose logs backend
docker-compose logs frontend
docker-compose logs ollama
```

## 📊 API Endpoints

The modular architecture maintains **100% backward compatibility** with existing endpoints:

### Interview Endpoints
- `POST /start_interview` - Start new interview session
- `POST /next_question` - Get next question
- `POST /answer` - Submit answer and get feedback
- `POST /end_interview` - Complete interview and get summary

### Audio Endpoints
- `POST /stt` - Speech to text conversion
- `POST /tts` - Text to speech synthesis

### Management Endpoints
- `GET /get_roles` - Available interview roles
- `GET /get_levels` - Experience levels
- `GET /get_question_bank` - Question database
- `GET /interview_history` - Session history
- `POST /reset_interview` - Reset session

### System Endpoints
- `GET /health` - System health check
- `GET /llm/status` - LLM provider status
- `POST /llm/test` - Test LLM functionality

## 🔒 Security Features

- Input sanitization and prompt injection protection
- Container security with non-root users
- Network isolation between services
- Security headers and CORS configuration
- Environment-based secrets management

## 🚀 Cloud Deployment

### AWS, Google Cloud, Azure Ready
```bash
# Build for cloud
docker build -f docker/backend.Dockerfile -t your-registry/ready4hire-backend .
docker push your-registry/ready4hire-backend

# Deploy to Kubernetes
kubectl apply -f k8s/
```

### Serverless Support
- AWS Lambda compatible
- Google Cloud Run ready
- Azure Container Instances support

## 📈 Monitoring & Logging

- Health check endpoints for all services
- Structured logging with configurable levels
- Docker health checks
- Prometheus metrics ready (future)

## 🔄 Migration from Legacy Version

The modular architecture is designed for **seamless migration**:

1. **API Compatibility**: All existing endpoints work unchanged
2. **Data Migration**: Automatic data format compatibility
3. **Gradual Migration**: Can run alongside legacy version
4. **Environment Config**: Simple .env configuration

## 🛠️ Development

### Local Development
```bash
# Setup development environment
./setup.sh dev

# Run backend locally
python -m backend.main

# Access services
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

### Adding New LLM Providers
1. Implement `LLMProvider` interface in `backend/llm/providers.py`
2. Add provider configuration
3. Update environment variables
4. Test with `curl http://localhost:8000/llm/test`

### Adding New Data Repositories
1. Implement `DataRepository` interface in `backend/data/repository.py`
2. Update configuration in `create_repository_from_env()`
3. Test with existing endpoints

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Follow the modular architecture patterns
4. Test your changes: `./setup.sh test`
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/JeronimoRestrepo48/Ready4Hire/issues)
- **Documentation**: [docs/](docs/)
- **Setup Problems**: Run `./setup.sh` for automated diagnosis

---

**Ready4Hire v2.0** - Now with modular architecture for enterprise scalability! 🚀