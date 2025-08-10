# Ready4Hire Modular Architecture Documentation

## Overview

Ready4Hire has been redesigned with a modular architecture that separates concerns into distinct layers and enables independent deployment of services. This architecture makes the system scalable, maintainable, and allows for easy replacement of components.

## Architecture Layers

### 1. API Layer (`backend/api/`)
- **Purpose**: Handles HTTP requests and responses
- **Components**:
  - `routes.py`: FastAPI route definitions and request handling
- **Responsibilities**:
  - Input validation and sanitization
  - Request routing
  - Response formatting
  - Error handling
  - Authentication (future)

### 2. Core Business Logic Layer (`backend/core/`)
- **Purpose**: Contains the main business logic and orchestration
- **Components**:
  - `services.py`: Core services for interview management, questions, and audio
- **Responsibilities**:
  - Interview session management
  - Question selection and evaluation
  - Business rule enforcement
  - Workflow orchestration

### 3. Data Layer (`backend/data/`)
- **Purpose**: Handles data access and persistence
- **Components**:
  - `repository.py`: Data access abstractions and implementations
- **Responsibilities**:
  - Data storage and retrieval
  - Data model definitions
  - Database abstraction
  - File system operations

### 4. LLM Layer (`backend/llm/`)
- **Purpose**: Provides abstraction for Large Language Model providers
- **Components**:
  - `providers.py`: LLM provider implementations and management
- **Responsibilities**:
  - LLM provider abstraction
  - Fallback mechanism
  - Provider switching
  - Configuration management

## Service Components

### Backend Service
- **Port**: 8000
- **Technology**: FastAPI + Python
- **Dependencies**: LLM service, data storage
- **Scalability**: Horizontal scaling supported

### Frontend Service
- **Port**: 3000 (80 in container)
- **Technology**: Nginx + Static files
- **Dependencies**: Backend service
- **Scalability**: CDN and load balancer ready

### LLM Service
- **Port**: 11434
- **Technology**: Ollama (default), supports OpenAI, Anthropic
- **Dependencies**: None (self-contained)
- **Scalability**: GPU-based scaling, model switching

### Database Service (Optional)
- **Port**: 5432
- **Technology**: PostgreSQL
- **Dependencies**: None
- **Scalability**: Master-slave replication, sharding

## Configuration Management

### Environment Variables

#### API Configuration
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `API_RELOAD`: Hot reload for development (default: true)

#### LLM Configuration
- `LLM_PROVIDER`: Primary LLM provider (ollama, openai, anthropic)
- `LLM_MODEL`: Model name (e.g., llama3.2, gpt-3.5-turbo)
- `LLM_BASE_URL`: LLM service URL
- `LLM_API_KEY`: API key for commercial providers
- `LLM_TEMPERATURE`: Generation temperature (0.0-1.0)
- `LLM_MAX_TOKENS`: Maximum response tokens
- `LLM_TIMEOUT`: Request timeout in seconds

#### Data Configuration
- `DATA_REPOSITORY`: Storage type (filesystem, database)
- `DATA_PATH`: Path to data files
- `DATABASE_URL`: Database connection string

### Configuration Files
- `.env`: Development configuration
- `.env.example`: Template with all available options
- `docker-compose.yml`: Production deployment
- `docker-compose.dev.yml`: Development deployment

## Deployment Options

### 1. Development Deployment
```bash
# Using Python directly
python -m backend.main

# Using Docker Compose
docker-compose -f docker-compose.dev.yml up
```

### 2. Production Deployment
```bash
# Using Docker Compose
docker-compose up -d

# Individual services
docker-compose up -d backend frontend ollama
```

### 3. Cloud Deployment
- Each service can be deployed independently
- Supports Kubernetes deployment
- Environment-specific configurations
- Load balancing and auto-scaling ready

## LLM Provider Management

### Supported Providers
1. **Ollama** (Default)
   - Local deployment
   - No API costs
   - GPU acceleration
   - Multiple model support

2. **OpenAI**
   - Commercial API
   - High performance
   - Cost per token
   - Rate limiting

3. **Anthropic**
   - Commercial API
   - Claude models
   - Cost per token
   - Rate limiting

### Provider Switching
The system automatically handles provider fallback:
1. Attempts primary provider
2. Falls back to secondary providers
3. Switches active provider on failure
4. Maintains provider health status

### Configuration Example
```bash
# Primary provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-api-key

# Fallback provider
FALLBACK_LLM_MODEL=llama3.2
FALLBACK_LLM_BASE_URL=http://ollama:11434
```

## Data Management

### Repository Pattern
The system uses the repository pattern to abstract data access:

- **FileSystemRepository**: Uses JSON/JSONL files
- **DatabaseRepository**: Uses SQL database (future)
- **Configurable**: Switch via environment variables

### Data Models
- **Question**: Interview question with metadata
- **UserSession**: User interview session state
- **InteractionLog**: User interaction history

### Storage Options
1. **File System** (Default)
   - Simple deployment
   - No database required
   - JSON/JSONL format
   - Version control friendly

2. **Database** (Future)
   - PostgreSQL support
   - ACID transactions
   - Better concurrency
   - Advanced querying

## Monitoring and Health Checks

### Health Endpoints
- `/health`: Overall system health
- `/llm/status`: LLM provider status
- `/llm/test`: LLM functionality test

### Docker Health Checks
- Backend: HTTP health check
- Frontend: Nginx status check
- Database: PostgreSQL connection check

### Logging
- Structured logging
- Configurable log levels
- File and console output
- Error tracking

## Security Considerations

### Input Sanitization
- Prompt injection protection
- HTML/script filtering
- Input length limits
- Pattern-based blocking

### API Security
- Request rate limiting (future)
- API key authentication (configurable)
- CORS configuration
- Security headers

### Container Security
- Non-root users
- Minimal base images
- Security scanning
- Network isolation

## Performance Considerations

### Caching
- LLM response caching (future)
- Static file caching
- Session caching
- Question bank caching

### Scalability
- Horizontal scaling support
- Load balancing ready
- Database connection pooling
- Async request handling

### Resource Management
- Memory limits
- CPU limits
- GPU allocation
- Connection limits

## Migration Guide

### From Legacy Architecture
1. Environment configuration setup
2. Data migration (automatic)
3. Service deployment
4. Testing and validation

### Backward Compatibility
- API endpoints remain the same
- Data format compatibility
- Configuration migration
- Gradual migration support

## Troubleshooting

### Common Issues
1. **LLM Service Unavailable**
   - Check Ollama service status
   - Verify network connectivity
   - Check model availability

2. **Database Connection**
   - Verify connection string
   - Check service dependencies
   - Review network configuration

3. **Frontend Not Loading**
   - Check backend connectivity
   - Verify static file mounting
   - Review proxy configuration

### Debug Mode
Enable debug logging:
```bash
LOG_LEVEL=DEBUG
API_RELOAD=true
```

### Service Logs
```bash
# View service logs
docker-compose logs backend
docker-compose logs frontend
docker-compose logs ollama
```

## Future Enhancements

### Planned Features
- Database repository implementation
- Advanced caching layer
- Metrics and monitoring
- API authentication
- Multi-tenant support
- Advanced LLM orchestration

### Extensibility Points
- Custom LLM providers
- Additional data repositories
- Custom middleware
- Plugin architecture
- Custom evaluation metrics