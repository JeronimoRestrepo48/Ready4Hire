# Deployment Guide - Ready4Hire

## 🎯 Overview

This guide covers the complete deployment process for Ready4Hire, from development to production environments.

## 📋 Prerequisites

### System Requirements

#### Hardware
- **Development**: 
  - CPU: 2+ cores
  - RAM: 8GB minimum, 16GB recommended
  - Storage: 20GB free space
  
- **Production**:
  - CPU: 4+ cores (8+ recommended for ML workloads)
  - RAM: 16GB minimum, 32GB+ for high traffic
  - Storage: 100GB+ SSD
  - GPU: Optional but recommended for LLM/ML (NVIDIA with CUDA support)

#### Software
- **OS**: Linux (Ubuntu 20.04/22.04), macOS 11+, or Windows 10/11 with WSL2
- **Python**: 3.9, 3.10, or 3.11
- **Nginx**: 1.18+
- **Git**: 2.30+
- **Docker** (optional): 20.10+
- **Docker Compose** (optional): 2.0+

### Accounts & Services
- GitHub account (for code access)
- Domain name (for production)
- SSL certificate provider (Let's Encrypt recommended for production)

## 🛠️ Installation Methods

### Method 1: Manual Installation (Recommended for Development)

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/JeronimoRestrepo48/Ready4Hire.git
cd Ready4Hire

# Optional: checkout specific branch or tag
git checkout main  # or specific version tag
```

#### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y nginx python3-dev build-essential ffmpeg portaudio19-dev

# macOS
brew install nginx ffmpeg portaudio

# Windows (via Chocolatey)
choco install nginx ffmpeg
```

#### Step 4: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your preferred editor
nano .env  # or vim, code, etc.
```

**.env file template:**
```env
# Application Settings
API_HOST=0.0.0.0
API_PORT=8001
API_RELOAD=True
DEBUG=True

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama2
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500

# Security
SECRET_KEY=your-secret-key-change-in-production
SSL_CERT_PATH=/path/to/nginx/certs/cert.pem
SSL_KEY_PATH=/path/to/nginx/certs/key.pem
ALLOWED_HOSTS=localhost,ready4hire.local

# Audio Processing
WHISPER_MODEL=base
TTS_ENGINE=pyttsx3
AUDIO_TEMP_DIR=/tmp/ready4hire_audio

# ML Models
RANKNET_MODEL_PATH=app/datasets/ranknet_model.pt
EMBEDDINGS_CACHE_DIR=app/embeddings/cache
USE_GPU=False

# Database (future)
# DATABASE_URL=postgresql://user:pass@localhost/ready4hire

# Logging
LOG_LEVEL=INFO
AUDIT_LOG_PATH=logs/audit_log.jsonl

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST=20
```

#### Step 5: Generate SSL Certificates

```bash
# Navigate to nginx directory
cd nginx

# Make script executable
chmod +x generate-certs.sh

# Generate certificates
./generate-certs.sh

# Go back to project root
cd ..
```

#### Step 6: Configure Nginx

```bash
# Copy Nginx configuration
sudo cp nginx/nginx.conf /etc/nginx/sites-available/ready4hire

# Update certificate paths in the config
sudo nano /etc/nginx/sites-available/ready4hire
# Change paths to match your generated certificates

# Create symbolic link
sudo ln -s /etc/nginx/sites-available/ready4hire /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# If test passes, reload Nginx
sudo systemctl reload nginx
```

#### Step 7: Update /etc/hosts (Optional for local development)

```bash
# Add entry for ready4hire.local
echo "127.0.0.1 ready4hire.local" | sudo tee -a /etc/hosts
```

#### Step 8: Install Ollama (Optional - for local LLM)

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull a model
ollama pull llama2
# Or for smaller/faster: ollama pull llama2:7b-chat
```

#### Step 9: Create Required Directories

```bash
# Create directories for logs and temporary files
mkdir -p logs
mkdir -p app/embeddings/cache
mkdir -p /tmp/ready4hire_audio
```

#### Step 10: Run Database Migrations (Future)

```bash
# When database is implemented
# alembic upgrade head
```

#### Step 11: Start the Application

```bash
# Option A: Using Uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Option B: Using Python module
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

#### Step 12: Verify Installation

```bash
# Check if API is running
curl http://localhost:8001/docs

# Check via HTTPS (with self-signed cert warning)
curl -k https://localhost:8001/docs

# Test start_interview endpoint
curl -X POST "http://localhost:8001/start_interview" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "role": "DevOps Engineer",
    "type": "technical",
    "mode": "practice"
  }'
```

### Method 2: Docker Installation (Recommended for Production)

#### Step 1: Install Docker

```bash
# Ubuntu
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker-compose --version
```

#### Step 2: Create Docker Files

**Dockerfile:**
```dockerfile
# See separate Dockerfile in project root
```

**docker-compose.yml:**
```yaml
# See separate docker-compose.yml in project root
```

#### Step 3: Build and Run

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Method 3: Cloud Deployment

#### AWS Deployment (Elastic Beanstalk)

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.9 ready4hire --region us-east-1

# Create environment
eb create ready4hire-prod

# Deploy
eb deploy

# View logs
eb logs
```

#### Azure Deployment (App Service)

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create --name ready4hire-rg --location eastus

# Create App Service plan
az appservice plan create --name ready4hire-plan --resource-group ready4hire-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group ready4hire-rg --plan ready4hire-plan --name ready4hire-app --runtime "PYTHON:3.9"

# Deploy code
az webapp up --name ready4hire-app --resource-group ready4hire-rg
```

#### Google Cloud Platform (Cloud Run)

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash

# Initialize gcloud
gcloud init

# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/ready4hire

# Deploy to Cloud Run
gcloud run deploy ready4hire \
  --image gcr.io/PROJECT_ID/ready4hire \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🔧 Configuration

### Nginx Production Configuration

For production, update Nginx config with:

1. **Use Let's Encrypt certificates:**
```nginx
ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
```

2. **Enable OCSP Stapling:**
```nginx
ssl_stapling on;
ssl_stapling_verify on;
ssl_trusted_certificate /etc/letsencrypt/live/yourdomain.com/chain.pem;
```

3. **Increase rate limits for production:**
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=1000r/m;
```

### Systemd Service (Linux Production)

Create `/etc/systemd/system/ready4hire.service`:

```ini
[Unit]
Description=Ready4Hire Interview AI Service
After=network.target

[Service]
Type=notify
User=ready4hire
Group=ready4hire
WorkingDirectory=/opt/ready4hire
Environment="PATH=/opt/ready4hire/venv/bin"
ExecStart=/opt/ready4hire/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ready4hire
sudo systemctl start ready4hire
sudo systemctl status ready4hire
```

## 🔒 Security Hardening

### 1. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Fail2ban (optional but recommended)
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 2. SSL Certificate (Production with Let's Encrypt)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal (certbot sets this up automatically)
sudo certbot renew --dry-run
```

### 3. Environment Variables Security

```bash
# Never commit .env to git
echo ".env" >> .gitignore

# Restrict permissions
chmod 600 .env

# In production, use secrets management
# AWS: Secrets Manager, Parameter Store
# Azure: Key Vault
# GCP: Secret Manager
```

### 4. Database Security (When Implemented)

```bash
# Use strong passwords
# Enable SSL/TLS for DB connections
# Restrict network access
# Regular backups
# Encryption at rest
```

## 📊 Monitoring Setup

### 1. Application Logs

```bash
# View application logs
tail -f logs/audit_log.jsonl

# Nginx access logs
sudo tail -f /var/log/nginx/ready4hire_access.log

# Nginx error logs
sudo tail -f /var/log/nginx/ready4hire_error.log
```

### 2. Health Checks

Add to your monitoring system:
- **Endpoint**: `https://yourdomain.com/docs`
- **Expected Status**: 200
- **Frequency**: Every 60 seconds

### 3. Prometheus & Grafana (Optional)

```bash
# Install Prometheus exporter
pip install prometheus-fastapi-instrumentator

# Add to app/main.py:
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# Access metrics at /metrics
```

## 🧪 Testing Deployment

### 1. Smoke Tests

```bash
# Test API is responding
curl -I https://yourdomain.com

# Test start interview
curl -X POST "https://yourdomain.com/api/v2/interviews/start" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","role":"DevOps Engineer","type":"technical"}'

# Test SSL configuration
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com
```

### 2. Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_tests.py --host=https://yourdomain.com
```

### 3. Security Scan

```bash
# Test SSL/TLS configuration
nmap --script ssl-enum-ciphers -p 443 yourdomain.com

# Or use online tools:
# https://www.ssllabs.com/ssltest/
```

## 🔄 Continuous Deployment

### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Ready4Hire

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
      
      - name: Deploy to production
        run: |
          # Add your deployment script here
          ./scripts/deploy.sh
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
```

## 🚨 Troubleshooting

### Issue: Nginx fails to start

```bash
# Check configuration
sudo nginx -t

# Check certificate paths
ls -la /etc/nginx/certs/ready4hire/

# Check port conflicts
sudo netstat -tulpn | grep :443
```

### Issue: Application won't start

```bash
# Check Python version
python3 --version

# Check if virtual environment is activated
which python

# Check dependencies
pip list

# Check logs
tail -n 100 logs/audit_log.jsonl
```

### Issue: SSL certificate errors

```bash
# Regenerate certificates
cd nginx && ./generate-certs.sh

# Verify certificate
openssl x509 -in nginx/certs/cert.pem -text -noout

# Check Nginx configuration
sudo nginx -t
```

### Issue: High memory usage

```bash
# Check memory
free -h

# Limit workers
# In app startup: --workers 2 --port 8001

# Disable GPU if not needed
# In .env: USE_GPU=False
```

## 📞 Support

- **Documentation**: See `/docs` directory
- **Issues**: https://github.com/JeronimoRestrepo48/Ready4Hire/issues
- **Email**: support@ready4hire.com

---

**Last Updated**: 2025-10-14  
**Version**: 1.0
