# Ready4Hire Deployment Guide

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Git
- 4GB+ RAM
- GPU (optional, for local LLM)

### 1. Clone and Setup
```bash
git clone https://github.com/JeronimoRestrepo48/Ready4Hire.git
cd Ready4Hire
cp .env.example .env
```

### 2. Configure Environment
Edit `.env` file:
```bash
# Basic configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
DATA_REPOSITORY=filesystem
```

### 3. Deploy Services
```bash
# Development deployment
docker-compose -f docker-compose.dev.yml up -d

# Production deployment
docker-compose up -d
```

### 4. Initialize LLM
```bash
# Pull the LLM model
docker exec -it ready4hire-ollama ollama pull llama3.2
```

### 5. Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Deployment Options

### Option 1: Development Environment
Best for: Local development, testing, debugging

```bash
# Hot reload enabled, debug logging
docker-compose -f docker-compose.dev.yml up -d
```

Features:
- Hot code reload
- Debug logging
- Volume mounts for development
- Easy debugging

### Option 2: Production Environment
Best for: Production deployment, staging

```bash
# Optimized for production
docker-compose up -d
```

Features:
- Optimized images
- Health checks
- Restart policies
- Resource limits

### Option 3: Cloud Deployment
Best for: Scalable production deployment

See individual service deployment sections below.

## Service Deployment

### Backend Service

#### Standalone Deployment
```bash
# Build image
docker build -f docker/backend.Dockerfile -t ready4hire-backend .

# Run container
docker run -d \
  --name ready4hire-backend \
  -p 8000:8000 \
  -e LLM_PROVIDER=ollama \
  -e LLM_BASE_URL=http://ollama:11434 \
  -v ./app/datasets:/app/app/datasets \
  ready4hire-backend
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ready4hire-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ready4hire-backend
  template:
    metadata:
      labels:
        app: ready4hire-backend
    spec:
      containers:
      - name: backend
        image: ready4hire-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_PROVIDER
          value: "ollama"
        - name: LLM_BASE_URL
          value: "http://ollama-service:11434"
        volumeMounts:
        - name: data-volume
          mountPath: /app/app/datasets
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: ready4hire-data-pvc
```

### Frontend Service

#### Standalone Deployment
```bash
# Build image
docker build -f docker/frontend.Dockerfile -t ready4hire-frontend .

# Run container
docker run -d \
  --name ready4hire-frontend \
  -p 3000:80 \
  ready4hire-frontend
```

#### CDN Deployment
For production, consider deploying static files to a CDN:
```bash
# Upload to AWS S3 + CloudFront
aws s3 sync app/static/ s3://your-bucket/
aws cloudfront create-invalidation --distribution-id YOUR_ID --paths "/*"
```

### LLM Service (Ollama)

#### Local Deployment
```bash
# Pull Ollama image
docker pull ollama/ollama:latest

# Run with GPU support
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama:latest

# Pull models
docker exec -it ollama ollama pull llama3.2
docker exec -it ollama ollama pull mistral
```

#### Cloud LLM Services
For production, consider using cloud LLM services:

**OpenAI Configuration:**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
LLM_API_KEY=your-openai-api-key
```

**Anthropic Configuration:**
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet
LLM_API_KEY=your-anthropic-api-key
```

### Database Service (Optional)

#### PostgreSQL Deployment
```bash
# Run PostgreSQL
docker run -d \
  --name ready4hire-db \
  -p 5432:5432 \
  -e POSTGRES_DB=ready4hire \
  -e POSTGRES_USER=ready4hire_user \
  -e POSTGRES_PASSWORD=ready4hire_password \
  -v postgres-data:/var/lib/postgresql/data \
  postgres:15-alpine
```

#### Configure Backend for Database
```bash
DATA_REPOSITORY=database
DATABASE_URL=postgresql://ready4hire_user:ready4hire_password@database:5432/ready4hire
```

## Environment Configuration

### Development Environment
```bash
# .env.development
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
DATA_REPOSITORY=filesystem
LOG_LEVEL=DEBUG
```

### Production Environment
```bash
# .env.production
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
LLM_PROVIDER=openai
LLM_API_KEY=your-production-api-key
DATA_REPOSITORY=database
DATABASE_URL=postgresql://user:pass@db-host:5432/ready4hire
LOG_LEVEL=INFO
```

### Staging Environment
```bash
# .env.staging
API_HOST=0.0.0.0
API_PORT=8000
LLM_PROVIDER=ollama
LLM_BASE_URL=http://ollama-staging:11434
DATA_REPOSITORY=database
DATABASE_URL=postgresql://user:pass@staging-db:5432/ready4hire
LOG_LEVEL=INFO
```

## Cloud Deployment Examples

### AWS Deployment

#### Using ECS (Elastic Container Service)
```yaml
# task-definition.json
{
  "family": "ready4hire",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ready4hire-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LLM_PROVIDER",
          "value": "openai"
        },
        {
          "name": "LLM_API_KEY",
          "value": "your-api-key"
        }
      ]
    }
  ]
}
```

#### Using Lambda (Serverless)
```python
# serverless deployment with AWS Lambda
import mangum
from backend.main import app

handler = mangum.Mangum(app)
```

### Google Cloud Deployment

#### Using Cloud Run
```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ready4hire-backend
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containers:
      - image: gcr.io/your-project/ready4hire-backend
        ports:
        - containerPort: 8000
        env:
        - name: LLM_PROVIDER
          value: "openai"
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: api-key
```

### Azure Deployment

#### Using Container Instances
```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group ready4hire-rg \
  --name ready4hire-backend \
  --image ready4hire-backend:latest \
  --ports 8000 \
  --environment-variables \
    LLM_PROVIDER=openai \
    LLM_API_KEY=your-api-key
```

## Load Balancing and Scaling

### Nginx Load Balancer
```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

### HAProxy Configuration
```
backend ready4hire_backend
    balance roundrobin
    server backend1 backend1:8000 check
    server backend2 backend2:8000 check
    server backend3 backend3:8000 check
```

### Kubernetes Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ready4hire-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ready4hire-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Monitoring and Logging

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# LLM status
curl http://localhost:8000/llm/status

# Frontend health
curl http://localhost:3000/health
```

### Logging Configuration
```yaml
# docker-compose.yml logging section
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Monitoring with Prometheus
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ready4hire-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
```

## Backup and Recovery

### Data Backup
```bash
# Backup data files
tar -czf backup-$(date +%Y%m%d).tar.gz app/datasets/

# Backup database
pg_dump ready4hire > backup-$(date +%Y%m%d).sql

# Backup Docker volumes
docker run --rm -v ready4hire_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

### Recovery Procedures
```bash
# Restore data files
tar -xzf backup-20240824.tar.gz

# Restore database
psql ready4hire < backup-20240824.sql

# Restore Docker volumes
docker run --rm -v ready4hire_postgres-data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres-backup.tar.gz -C /
```

## Security Configuration

### SSL/TLS Setup
```yaml
# docker-compose.yml with SSL
services:
  nginx:
    image: nginx:alpine
    volumes:
      - ./ssl:/etc/nginx/ssl
      - ./nginx-ssl.conf:/etc/nginx/nginx.conf
    ports:
      - "443:443"
      - "80:80"
```

### Firewall Configuration
```bash
# UFW configuration
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp  # Block direct backend access
ufw deny 11434/tcp # Block direct LLM access
ufw enable
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs backend
docker-compose logs frontend
docker-compose logs ollama

# Check service health
docker-compose ps
```

#### LLM Connection Issues
```bash
# Test LLM connectivity
curl http://localhost:11434/api/tags

# Pull model if missing
docker exec -it ready4hire-ollama ollama pull llama3.2
```

#### Database Connection Issues
```bash
# Test database connection
psql -h localhost -U ready4hire_user -d ready4hire

# Check database logs
docker-compose logs database
```

### Performance Issues

#### High Memory Usage
```bash
# Check container resource usage
docker stats

# Adjust container limits
docker-compose.yml:
  services:
    backend:
      deploy:
        resources:
          limits:
            memory: 2G
```

#### Slow Response Times
```bash
# Enable request logging
LOG_LEVEL=DEBUG

# Check LLM response times
curl -w "@curl-format.txt" http://localhost:8000/llm/test
```

## Maintenance

### Regular Updates
```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Update LLM models
docker exec -it ready4hire-ollama ollama pull llama3.2:latest
```

### Cleanup
```bash
# Remove unused Docker resources
docker system prune -a

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete
```

## Support

For deployment issues:
1. Check the troubleshooting section
2. Review service logs
3. Consult the architecture documentation
4. Open an issue on GitHub

For production deployments, consider:
- Load testing
- Security audits
- Backup strategies
- Monitoring setup
- SLA planning