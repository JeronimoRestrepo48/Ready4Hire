#!/bin/bash

# Ready4Hire Modular Architecture Setup Script
# =============================================

set -e

echo "🚀 Ready4Hire Modular Architecture Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python (for local development)
    if ! command -v python3 &> /dev/null; then
        print_warning "Python 3 is not installed. Docker deployment will still work."
    fi
    
    print_success "Prerequisites check completed"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_success ".env file created"
    else
        print_warning ".env file already exists, skipping..."
    fi
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p app/datasets/sessions
    mkdir -p app/datasets/logs
    
    print_success "Environment setup completed"
}

# Install Python dependencies (for local development)
install_dependencies() {
    if command -v python3 &> /dev/null; then
        print_status "Installing Python dependencies..."
        python3 -m pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_warning "Python not available, skipping dependency installation"
    fi
}

# Test modular backend locally
test_local_backend() {
    if command -v python3 &> /dev/null; then
        print_status "Testing modular backend locally..."
        
        # Set environment for local testing
        export API_HOST=127.0.0.1
        export API_PORT=8001
        export LLM_PROVIDER=ollama
        export LLM_BASE_URL=http://localhost:11434
        export DATA_REPOSITORY=filesystem
        export DATA_PATH=./app/datasets
        
        # Start backend in background
        python3 -m backend.main &
        BACKEND_PID=$!
        
        # Wait for backend to start
        sleep 5
        
        # Test health endpoint
        if curl -f http://localhost:8001/health > /dev/null 2>&1; then
            print_success "Local backend is working!"
        else
            print_warning "Local backend test failed (this is normal if Ollama is not running)"
        fi
        
        # Stop backend
        kill $BACKEND_PID 2>/dev/null || true
        wait $BACKEND_PID 2>/dev/null || true
    else
        print_warning "Python not available, skipping local backend test"
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build backend image
    print_status "Building backend image..."
    docker build -f docker/backend.Dockerfile -t ready4hire-backend .
    
    # Build frontend image
    print_status "Building frontend image..."
    docker build -f docker/frontend.Dockerfile -t ready4hire-frontend .
    
    print_success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_services() {
    local env=${1:-dev}
    
    print_status "Deploying services in $env mode..."
    
    if [ "$env" = "dev" ]; then
        docker-compose -f docker-compose.dev.yml up -d
    else
        docker-compose up -d
    fi
    
    print_success "Services deployed successfully"
}

# Initialize LLM service
init_llm() {
    print_status "Initializing LLM service..."
    
    # Wait for Ollama to be ready
    print_status "Waiting for Ollama service to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker exec ready4hire-ollama-dev ollama list &> /dev/null 2>&1 || \
           docker exec ready4hire-ollama ollama list &> /dev/null 2>&1; then
            break
        fi
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Ollama service did not start in time, you may need to pull models manually"
        return
    fi
    
    # Pull default model
    print_status "Pulling default LLM model (llama3.2)..."
    if docker exec ready4hire-ollama-dev ollama pull llama3.2 2>&1 || \
       docker exec ready4hire-ollama ollama pull llama3.2 2>&1; then
        print_success "LLM model pulled successfully"
    else
        print_warning "Failed to pull LLM model automatically. You can pull it manually later."
    fi
}

# Test deployment
test_deployment() {
    print_status "Testing deployment..."
    
    # Wait for services to be ready
    sleep 10
    
    # Test backend health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend service is healthy"
    else
        print_warning "Backend service health check failed"
    fi
    
    # Test frontend
    if curl -f http://localhost:3000/health > /dev/null 2>&1; then
        print_success "Frontend service is healthy"
    else
        print_warning "Frontend service health check failed"
    fi
    
    # Test API endpoints
    print_status "Testing API endpoints..."
    
    # Test roles endpoint
    if curl -f http://localhost:8000/get_roles > /dev/null 2>&1; then
        print_success "API endpoints are working"
    else
        print_warning "API endpoints test failed"
    fi
    
    print_success "Deployment testing completed"
}

# Show status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    
    echo ""
    print_status "Access Information:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Ollama API: http://localhost:11434"
    
    echo ""
    print_status "Useful Commands:"
    echo "  View logs: docker-compose logs [service]"
    echo "  Stop services: docker-compose down"
    echo "  Restart services: docker-compose restart"
    echo "  Pull LLM model: docker exec ready4hire-ollama ollama pull [model]"
}

# Cleanup function
cleanup() {
    print_status "Stopping any running services..."
    docker-compose down 2>/dev/null || true
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
}

# Main menu
show_menu() {
    echo ""
    echo "Setup Options:"
    echo "1. Full setup (recommended for first time)"
    echo "2. Setup environment only"
    echo "3. Build Docker images"
    echo "4. Deploy development environment"
    echo "5. Deploy production environment"
    echo "6. Test local backend"
    echo "7. Initialize LLM service"
    echo "8. Test deployment"
    echo "9. Show service status"
    echo "10. Cleanup (stop all services)"
    echo "0. Exit"
    echo ""
    read -p "Choose an option (0-10): " choice
}

# Main execution
main() {
    case ${1:-menu} in
        "full")
            check_prerequisites
            setup_environment
            install_dependencies
            build_images
            deploy_services dev
            init_llm
            test_deployment
            show_status
            ;;
        "env")
            setup_environment
            ;;
        "build")
            build_images
            ;;
        "dev")
            deploy_services dev
            ;;
        "prod")
            deploy_services prod
            ;;
        "test-local")
            test_local_backend
            ;;
        "init-llm")
            init_llm
            ;;
        "test")
            test_deployment
            ;;
        "status")
            show_status
            ;;
        "cleanup")
            cleanup
            ;;
        "menu")
            while true; do
                show_menu
                case $choice in
                    1) main full ;;
                    2) main env ;;
                    3) main build ;;
                    4) main dev ;;
                    5) main prod ;;
                    6) main test-local ;;
                    7) main init-llm ;;
                    8) main test ;;
                    9) main status ;;
                    10) main cleanup ;;
                    0) break ;;
                    *) print_error "Invalid option" ;;
                esac
                echo ""
                read -p "Press Enter to continue..."
            done
            ;;
        *)
            echo "Usage: $0 [full|env|build|dev|prod|test-local|init-llm|test|status|cleanup|menu]"
            echo ""
            echo "  full        - Complete setup (recommended)"
            echo "  env         - Setup environment only"
            echo "  build       - Build Docker images"
            echo "  dev         - Deploy development environment"
            echo "  prod        - Deploy production environment"
            echo "  test-local  - Test local backend"
            echo "  init-llm    - Initialize LLM service"
            echo "  test        - Test deployment"
            echo "  status      - Show service status"
            echo "  cleanup     - Stop all services"
            echo "  menu        - Interactive menu"
            ;;
    esac
}

# Trap cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"