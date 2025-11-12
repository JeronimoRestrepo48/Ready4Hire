#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Ready4Hire - Automated Backup Script
# ══════════════════════════════════════════════════════════════════════════════
#
# This script performs automated backups of:
# - PostgreSQL database
# - Redis data (RDB snapshot)
# - Qdrant vector database
# - Docker volumes
#
# Usage:
#   ./scripts/backup.sh                    # Manual backup
#   ./scripts/backup.sh --auto              # Automatic backup (for cron)
#   ./scripts/backup.sh --restore <backup>  # Restore from backup
#
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ready4hire_backup_${TIMESTAMP}"

# Retention policy (keep last N backups)
RETENTION_DAYS=${RETENTION_DAYS:-30}

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Default values if not in .env
POSTGRES_DB="${POSTGRES_DB:-ready4hire_db}"
POSTGRES_USER="${POSTGRES_USER:-ready4hire}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-password}"
POSTGRES_HOST="${DATABASE_HOST:-localhost}"
POSTGRES_PORT="${DATABASE_PORT:-5432}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ══════════════════════════════════════════════════════════════════════════════
# PostgreSQL Backup
# ══════════════════════════════════════════════════════════════════════════════

backup_postgres() {
    log "Backing up PostgreSQL database..."
    
    local backup_file="$BACKUP_DIR/${BACKUP_NAME}_postgres.sql"
    
    # Export PGPASSWORD for non-interactive mode
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    if command -v pg_dump &> /dev/null; then
        if pg_dump \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            -F c \
            -f "$backup_file.gz" \
            --no-owner \
            --no-acl; then
            log "✅ PostgreSQL backup completed: $backup_file.gz"
            echo "$backup_file.gz"
        else
            error "Failed to backup PostgreSQL"
            return 1
        fi
    elif docker ps --format '{{.Names}}' | grep -q "postgres"; then
        # Use Docker if available
        local container_name=$(docker ps --format '{{.Names}}' | grep -i postgres | head -1)
        if docker exec "$container_name" pg_dump \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            -F c > "$backup_file.gz"; then
            log "✅ PostgreSQL backup completed via Docker: $backup_file.gz"
            echo "$backup_file.gz"
        else
            error "Failed to backup PostgreSQL via Docker"
            return 1
        fi
    else
        error "pg_dump not found and no PostgreSQL container running"
        return 1
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Redis Backup
# ══════════════════════════════════════════════════════════════════════════════

backup_redis() {
    log "Backing up Redis data..."
    
    local backup_file="$BACKUP_DIR/${BACKUP_NAME}_redis.rdb"
    
    if docker ps --format '{{.Names}}' | grep -q "redis"; then
        local container_name=$(docker ps --format '{{.Names}}' | grep -i redis | head -1)
        
        # Trigger RDB save
        if [ -n "$REDIS_PASSWORD" ]; then
            docker exec "$container_name" redis-cli --no-auth-warning -a "$REDIS_PASSWORD" BGSAVE
        else
            docker exec "$container_name" redis-cli BGSAVE
        fi
        
        # Wait for save to complete
        sleep 2
        
        # Copy RDB file
        if docker cp "$container_name:/data/dump.rdb" "$backup_file" 2>/dev/null; then
            gzip "$backup_file"
            log "✅ Redis backup completed: $backup_file.gz"
            echo "$backup_file.gz"
        else
            warn "Redis backup file not found, continuing..."
            return 0
        fi
    elif command -v redis-cli &> /dev/null; then
        # Direct redis-cli
        if [ -n "$REDIS_PASSWORD" ]; then
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" BGSAVE
        else
            redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE
        fi
        sleep 2
        
        # Find dump.rdb location
        local rdb_path=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ${REDIS_PASSWORD:+-a "$REDIS_PASSWORD"} CONFIG GET dir | tail -1)
        if [ -n "$rdb_path" ] && [ -f "$rdb_path/dump.rdb" ]; then
            cp "$rdb_path/dump.rdb" "$backup_file"
            gzip "$backup_file"
            log "✅ Redis backup completed: $backup_file.gz"
            echo "$backup_file.gz"
        else
            warn "Redis dump.rdb not found at expected location"
            return 0
        fi
    else
        warn "Redis not available for backup"
        return 0
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Qdrant Backup
# ══════════════════════════════════════════════════════════════════════════════

backup_qdrant() {
    log "Backing up Qdrant vector database..."
    
    local backup_file="$BACKUP_DIR/${BACKUP_NAME}_qdrant.tar.gz"
    
    if docker ps --format '{{.Names}}' | grep -q "qdrant"; then
        local container_name=$(docker ps --format '{{.Names}}' | grep -i qdrant | head -1)
        
        # Create snapshot directory
        local temp_dir=$(mktemp -d)
        
        # Copy Qdrant storage directory
        if docker cp "$container_name:/qdrant/storage" "$temp_dir/qdrant_storage" 2>/dev/null; then
            tar -czf "$backup_file" -C "$temp_dir" qdrant_storage
            rm -rf "$temp_dir"
            log "✅ Qdrant backup completed: $backup_file"
            echo "$backup_file"
        else
            warn "Failed to backup Qdrant storage"
            rm -rf "$temp_dir"
            return 0
        fi
    else
        warn "Qdrant container not running, skipping backup"
        return 0
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Docker Volumes Backup
# ══════════════════════════════════════════════════════════════════════════════

backup_volumes() {
    log "Backing up Docker volumes..."
    
    local backup_file="$BACKUP_DIR/${BACKUP_NAME}_volumes.tar.gz"
    
    local volumes=(
        "ready4hire_postgres_data"
        "ready4hire_redis_data"
        "ready4hire_qdrant_data"
        "ready4hire_ollama_data"
    )
    
    local temp_dir=$(mktemp -d)
    local volumes_found=0
    
    for volume in "${volumes[@]}"; do
        if docker volume inspect "$volume" &>/dev/null; then
            log "  Backing up volume: $volume"
            docker run --rm \
                -v "$volume:/data" \
                -v "$temp_dir:/backup" \
                alpine tar czf "/backup/${volume}.tar.gz" -C /data .
            volumes_found=1
        fi
    done
    
    if [ "$volumes_found" -eq 1 ]; then
        tar -czf "$backup_file" -C "$temp_dir" .
        rm -rf "$temp_dir"
        log "✅ Docker volumes backup completed: $backup_file"
        echo "$backup_file"
    else
        warn "No Docker volumes found to backup"
        rm -rf "$temp_dir"
        return 0
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# Retention Policy
# ══════════════════════════════════════════════════════════════════════════════

cleanup_old_backups() {
    log "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."
    
    find "$BACKUP_DIR" -name "ready4hire_backup_*.gz" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "ready4hire_backup_*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete
    
    log "✅ Cleanup completed"
}

# ══════════════════════════════════════════════════════════════════════════════
# Restore Functions
# ══════════════════════════════════════════════════════════════════════════════

restore_postgres() {
    local backup_file="$1"
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        return 1
    fi
    
    log "Restoring PostgreSQL from $backup_file..."
    warn "⚠️  This will overwrite the current database!"
    
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log "Restore cancelled"
        return 1
    fi
    
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    if [[ "$backup_file" == *.gz ]]; then
        gunzip -c "$backup_file" | pg_restore \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            --clean \
            --if-exists
    else
        pg_restore \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            --clean \
            --if-exists \
            "$backup_file"
    fi
    
    log "✅ PostgreSQL restore completed"
}

# ══════════════════════════════════════════════════════════════════════════════
# Main Functions
# ══════════════════════════════════════════════════════════════════════════════

perform_backup() {
    log "🚀 Starting backup process..."
    log "Backup directory: $BACKUP_DIR"
    
    local backups=()
    
    # Perform backups
    local postgres_backup=$(backup_postgres || echo "")
    [ -n "$postgres_backup" ] && backups+=("$postgres_backup")
    
    local redis_backup=$(backup_redis || echo "")
    [ -n "$redis_backup" ] && backups+=("$redis_backup")
    
    local qdrant_backup=$(backup_qdrant || echo "")
    [ -n "$qdrant_backup" ] && backups+=("$qdrant_backup")
    
    local volumes_backup=$(backup_volumes || echo "")
    [ -n "$volumes_backup" ] && backups+=("$volumes_backup")
    
    # Create backup manifest
    local manifest_file="$BACKUP_DIR/${BACKUP_NAME}_manifest.json"
    cat > "$manifest_file" <<EOF
{
  "backup_name": "$BACKUP_NAME",
  "timestamp": "$TIMESTAMP",
  "created_at": "$(date -Iseconds)",
  "backups": [
$(printf '    "%s"' "${backups[@]}" | sed 's/$/,/' | sed '$s/,$//')
  ],
  "version": "1.0"
}
EOF
    
    log "✅ Backup completed successfully!"
    log "Manifest: $manifest_file"
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Show backup size
    local total_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    log "Total backup size: $total_size"
}

# ══════════════════════════════════════════════════════════════════════════════
# Script Entry Point
# ══════════════════════════════════════════════════════════════════════════════

main() {
    case "${1:-}" in
        --restore)
            if [ -z "${2:-}" ]; then
                error "Backup file required for restore"
                echo "Usage: $0 --restore <backup_file>"
                exit 1
            fi
            restore_postgres "$2"
            ;;
        --auto)
            perform_backup
            ;;
        --help|-h)
            cat <<EOF
Ready4Hire Backup Script

Usage:
  $0                    # Manual backup
  $0 --auto              # Automatic backup (for cron)
  $0 --restore <backup>  # Restore from backup
  $0 --help              # Show this help

Environment Variables:
  BACKUP_DIR           Backup directory (default: ./backups)
  RETENTION_DAYS       Days to keep backups (default: 30)

Cron Example (daily at 2 AM):
  0 2 * * * /path/to/scripts/backup.sh --auto
EOF
            ;;
        *)
            perform_backup
            ;;
    esac
}

main "$@"

