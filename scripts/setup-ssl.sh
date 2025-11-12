#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Ready4Hire - SSL/TLS Certificate Setup with Let's Encrypt
# ══════════════════════════════════════════════════════════════════════════════
#
# This script sets up SSL certificates using Let's Encrypt and Certbot.
#
# Usage:
#   ./scripts/setup-ssl.sh init <domain> <email>    # Initial setup
#   ./scripts/setup-ssl.sh renew                    # Renew certificates
#   ./scripts/setup-ssl.sh status                  # Check certificate status
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CERTS_DIR="$PROJECT_ROOT/nginx/certs"

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

init_ssl() {
    local domain=${1:-}
    local email=${2:-}
    
    if [ -z "$domain" ] || [ -z "$email" ]; then
        error "Usage: $0 init <domain> <email>"
        exit 1
    fi
    
    log "Setting up SSL certificates for domain: $domain"
    
    # Create certs directory
    mkdir -p "$CERTS_DIR"
    
    # Check if certbot is available
    if ! command -v certbot &> /dev/null; then
        log "Certbot not found. Using Docker container..."
        USE_DOCKER=true
    else
        USE_DOCKER=false
    fi
    
    if [ "$USE_DOCKER" = true ]; then
        log "Using Certbot Docker container..."
        
        # Run certbot in Docker
        docker run -it --rm \
            -v "$CERTS_DIR:/etc/letsencrypt" \
            -v "$CERTS_DIR:/var/lib/letsencrypt" \
            -p 80:80 \
            certbot/certbot certonly \
            --standalone \
            --non-interactive \
            --agree-tos \
            --email "$email" \
            -d "$domain" \
            -d "www.$domain" || {
                error "Failed to obtain certificate"
                exit 1
            }
        
        # Copy certificates to nginx format
        if [ -f "$CERTS_DIR/live/$domain/fullchain.pem" ]; then
            cp "$CERTS_DIR/live/$domain/fullchain.pem" "$CERTS_DIR/fullchain.pem"
            cp "$CERTS_DIR/live/$domain/privkey.pem" "$CERTS_DIR/privkey.pem"
            log "✅ Certificates copied to nginx/certs/"
        fi
    else
        # Use local certbot
        certbot certonly \
            --standalone \
            --non-interactive \
            --agree-tos \
            --email "$email" \
            -d "$domain" \
            -d "www.$domain" || {
                error "Failed to obtain certificate"
                exit 1
            }
        
        # Copy certificates
        if [ -f "/etc/letsencrypt/live/$domain/fullchain.pem" ]; then
            cp "/etc/letsencrypt/live/$domain/fullchain.pem" "$CERTS_DIR/fullchain.pem"
            cp "/etc/letsencrypt/live/$domain/privkey.pem" "$CERTS_DIR/privkey.pem"
            log "✅ Certificates copied to nginx/certs/"
        fi
    fi
    
    log "✅ SSL setup completed!"
    log "Certificates location: $CERTS_DIR"
    warn "⚠️  Remember to configure nginx to use these certificates!"
}

renew_ssl() {
    log "Renewing SSL certificates..."
    
    if [ -d "$CERTS_DIR" ]; then
        # Use Docker certbot for renewal
        docker run --rm \
            -v "$CERTS_DIR:/etc/letsencrypt" \
            -v "$CERTS_DIR:/var/lib/letsencrypt" \
            -p 80:80 \
            certbot/certbot renew \
            --quiet || {
                error "Failed to renew certificates"
                exit 1
            }
        
        # Reload nginx if running
        if docker ps --format '{{.Names}}' | grep -q "nginx"; then
            docker exec ready4hire_nginx_production nginx -s reload || true
            log "✅ Nginx reloaded with new certificates"
        fi
        
        log "✅ Certificate renewal completed!"
    else
        error "Certificates directory not found: $CERTS_DIR"
        exit 1
    fi
}

status_ssl() {
    log "Checking SSL certificate status..."
    
    if [ -f "$CERTS_DIR/fullchain.pem" ]; then
        local domain=$(openssl x509 -in "$CERTS_DIR/fullchain.pem" -noout -subject | sed 's/.*CN=\([^,]*\).*/\1/')
        local expiry=$(openssl x509 -in "$CERTS_DIR/fullchain.pem" -noout -enddate | cut -d= -f2)
        local days_left=$(( ($(date -d "$expiry" +%s) - $(date +%s)) / 86400 ))
        
        echo "Domain: $domain"
        echo "Expires: $expiry"
        echo "Days remaining: $days_left"
        
        if [ $days_left -lt 30 ]; then
            warn "⚠️  Certificate expires in less than 30 days! Run renewal."
        fi
    else
        error "Certificate not found: $CERTS_DIR/fullchain.pem"
        exit 1
    fi
}

# Main
case "${1:-}" in
    init)
        init_ssl "${2:-}" "${3:-}"
        ;;
    renew)
        renew_ssl
        ;;
    status)
        status_ssl
        ;;
    *)
        cat <<EOF
Ready4Hire SSL Setup Script

Usage:
  $0 init <domain> <email>   # Initial SSL setup
  $0 renew                    # Renew certificates
  $0 status                   # Check certificate status

Example:
  $0 init ready4hire.com admin@ready4hire.com

Note: Make sure port 80 is accessible for Let's Encrypt validation.
EOF
        exit 1
        ;;
esac

