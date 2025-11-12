#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Ready4Hire - Secrets Setup Script
# ══════════════════════════════════════════════════════════════════════════════
#
# This script helps create and manage Docker secrets for production deployment.
#
# Usage:
#   ./scripts/setup-secrets.sh create        # Create all secrets
#   ./scripts/setup-secrets.sh rotate        # Rotate secrets
#   ./scripts/setup-secrets.sh list          # List secrets
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SECRETS_DIR="$PROJECT_ROOT/secrets"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Generate secure random string
generate_secret() {
    local length=${1:-32}
    openssl rand -base64 $length | tr -d '\n' | head -c $length
}

# Create secrets directory
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

create_secrets() {
    log "Creating Docker secrets..."
    
    # PostgreSQL Password
    if [ ! -f "$SECRETS_DIR/postgres_password.txt" ]; then
        generate_secret 24 > "$SECRETS_DIR/postgres_password.txt"
        chmod 600 "$SECRETS_DIR/postgres_password.txt"
        log "✅ Created postgres_password"
    else
        warn "postgres_password already exists, skipping..."
    fi
    
    # JWT Secret Key
    if [ ! -f "$SECRETS_DIR/jwt_secret_key.txt" ]; then
        generate_secret 64 > "$SECRETS_DIR/jwt_secret_key.txt"
        chmod 600 "$SECRETS_DIR/jwt_secret_key.txt"
        log "✅ Created jwt_secret_key"
    else
        warn "jwt_secret_key already exists, skipping..."
    fi
    
    # Redis Password (optional)
    if [ ! -f "$SECRETS_DIR/redis_password.txt" ]; then
        generate_secret 32 > "$SECRETS_DIR/redis_password.txt"
        chmod 600 "$SECRETS_DIR/redis_password.txt"
        log "✅ Created redis_password"
    fi
    
    # Create Docker secrets
    if command -v docker &> /dev/null; then
        log "Creating Docker Swarm secrets..."
        
        # Check if in swarm mode
        if docker info | grep -q "Swarm: active"; then
            docker secret create postgres_password "$SECRETS_DIR/postgres_password.txt" 2>/dev/null || warn "Secret already exists"
            docker secret create jwt_secret_key "$SECRETS_DIR/jwt_secret_key.txt" 2>/dev/null || warn "Secret already exists"
            docker secret create redis_password "$SECRETS_DIR/redis_password.txt" 2>/dev/null || warn "Secret already exists"
            
            log "✅ Docker secrets created successfully"
        else
            warn "Docker Swarm not active. Secrets created as files only."
            warn "To use Docker Swarm secrets, run: docker swarm init"
        fi
    fi
    
    log "✅ Secrets setup completed!"
    log "Secrets location: $SECRETS_DIR"
    warn "⚠️  Keep secrets secure! Never commit to version control!"
}

rotate_secrets() {
    warn "⚠️  Rotating secrets will require restarting services!"
    read -p "Continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Rotation cancelled"
        return
    fi
    
    log "Rotating secrets..."
    
    # Backup old secrets
    BACKUP_DIR="$SECRETS_DIR/backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp "$SECRETS_DIR"/*.txt "$BACKUP_DIR/" 2>/dev/null || true
    log "Old secrets backed up to: $BACKUP_DIR"
    
    # Generate new secrets
    generate_secret 24 > "$SECRETS_DIR/postgres_password.txt"
    generate_secret 64 > "$SECRETS_DIR/jwt_secret_key.txt"
    generate_secret 32 > "$SECRETS_DIR/redis_password.txt"
    
    chmod 600 "$SECRETS_DIR"/*.txt
    
    log "✅ Secrets rotated!"
    warn "⚠️  Remember to update services with new secrets!"
}

list_secrets() {
    log "Listing secrets..."
    
    if [ -d "$SECRETS_DIR" ]; then
        echo "File-based secrets:"
        ls -lh "$SECRETS_DIR"/*.txt 2>/dev/null || echo "  No secrets found"
    fi
    
    if command -v docker &> /dev/null && docker info | grep -q "Swarm: active"; then
        echo ""
        echo "Docker Swarm secrets:"
        docker secret ls || echo "  No Docker secrets found"
    fi
}

# Main
case "${1:-create}" in
    create)
        create_secrets
        ;;
    rotate)
        rotate_secrets
        ;;
    list)
        list_secrets
        ;;
    *)
        echo "Usage: $0 {create|rotate|list}"
        exit 1
        ;;
esac

