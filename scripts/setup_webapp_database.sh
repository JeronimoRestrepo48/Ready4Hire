#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# Script para crear la base de datos PostgreSQL para WebApp
# ══════════════════════════════════════════════════════════════════════════════

set -e

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_header() { echo -e "${CYAN}$1${NC}"; }

# Variables de configuración
DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}
DB_NAME=${POSTGRES_DB:-ready4hire_db}
DB_USER=${POSTGRES_USER:-ready4hire}
DB_PASSWORD=${POSTGRES_PASSWORD:-password}

echo ""
print_header "╔════════════════════════════════════════════════════════════════╗"
print_header "║         Configuración de Base de Datos WebApp                  ║"
print_header "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Verificar que psql esté instalado
if ! command -v psql &> /dev/null; then
    print_error "psql no está instalado"
    print_info "Instala PostgreSQL client: sudo apt-get install postgresql-client"
    exit 1
fi

# Verificar que dotnet esté instalado
if ! command -v dotnet &> /dev/null; then
    print_error "dotnet no está instalado"
    print_info "Instala .NET SDK desde: https://dotnet.microsoft.com/download"
    exit 1
fi

# Verificar conexión a PostgreSQL como postgres
print_header "1️⃣  Verificando conexión a PostgreSQL..."
echo ""

# Intentar conectar como usuario postgres
if psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -d postgres -c "SELECT 1;" &> /dev/null; then
    print_success "Conexión exitosa a PostgreSQL como usuario 'postgres'"
    ADMIN_USER="postgres"
elif psql -h "$DB_HOST" -p "$DB_PORT" -U $USER -d postgres -c "SELECT 1;" &> /dev/null; then
    print_success "Conexión exitosa a PostgreSQL como usuario '$USER'"
    ADMIN_USER="$USER"
else
    print_error "No se pudo conectar a PostgreSQL"
    print_info "Asegúrate de que PostgreSQL esté corriendo y que tengas permisos"
    print_info "Intenta: sudo -u postgres psql"
    exit 1
fi

echo ""

# Crear usuario/rol si no existe
print_header "2️⃣  Creando usuario/rol '$DB_USER'..."
echo ""

USER_EXISTS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -t -c "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER';" 2>/dev/null | xargs)

if [ "$USER_EXISTS" = "1" ]; then
    print_success "El usuario '$DB_USER' ya existe"
else
    print_info "Creando usuario '$DB_USER'..."
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" &> /dev/null; then
        print_success "Usuario '$DB_USER' creado exitosamente"
    else
        print_error "No se pudo crear el usuario '$DB_USER'"
        exit 1
    fi
fi

# Dar permisos al usuario
print_info "Otorgando permisos al usuario '$DB_USER'..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -c "ALTER USER $DB_USER CREATEDB;" &> /dev/null || true
print_success "Permisos otorgados"

echo ""

# Crear base de datos si no existe
print_header "3️⃣  Creando base de datos '$DB_NAME'..."
echo ""

DB_EXISTS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -t -c "SELECT 1 FROM pg_database WHERE datname='$DB_NAME';" 2>/dev/null | xargs)

if [ "$DB_EXISTS" = "1" ]; then
    print_success "La base de datos '$DB_NAME' ya existe"
else
    print_info "Creando base de datos '$DB_NAME'..."
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" &> /dev/null; then
        print_success "Base de datos '$DB_NAME' creada exitosamente"
    else
        print_error "No se pudo crear la base de datos '$DB_NAME'"
        exit 1
    fi
fi

# Otorgar todos los privilegios al usuario en la base de datos
print_info "Otorgando privilegios al usuario '$DB_USER' en la base de datos..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;" &> /dev/null || true
psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;" &> /dev/null || true
psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $DB_USER;" &> /dev/null || true
psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $DB_USER;" &> /dev/null || true
psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;" &> /dev/null || true
psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_USER;" &> /dev/null || true
print_success "Privilegios otorgados"

echo ""

# Verificar conexión con el nuevo usuario
print_header "4️⃣  Verificando conexión con usuario '$DB_USER'..."
echo ""

export PGPASSWORD="$DB_PASSWORD"

if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &> /dev/null; then
    print_success "Conexión exitosa con usuario '$DB_USER' a la base de datos '$DB_NAME'"
else
    print_error "No se pudo conectar con usuario '$DB_USER'"
    exit 1
fi

echo ""

# Ejecutar migraciones de Entity Framework Core
print_header "5️⃣  Ejecutando migraciones de Entity Framework Core..."
echo ""

cd "$(dirname "$0")/../WebApp" || exit 1

if [ ! -f "Ready4Hire.csproj" ]; then
    print_error "No se encontró el archivo Ready4Hire.csproj"
    print_info "Asegúrate de ejecutar el script desde el directorio raíz del proyecto"
    exit 1
fi

# Verificar que las herramientas de EF Core estén instaladas
if ! dotnet ef --version &> /dev/null; then
    print_info "Instalando herramientas de Entity Framework Core..."
    dotnet tool install --global dotnet-ef &> /dev/null || true
fi

# Configurar cadena de conexión para las migraciones
export POSTGRES_CONNECTION="Host=$DB_HOST;Port=$DB_PORT;Database=$DB_NAME;Username=$DB_USER;Password=$DB_PASSWORD;"

print_info "Ejecutando migraciones..."
if dotnet ef database update --connection "$POSTGRES_CONNECTION" &> /tmp/ef_migration.log; then
    print_success "Migraciones ejecutadas exitosamente"
    
    # Mostrar tablas creadas
    TABLE_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | xargs)
    if [ ! -z "$TABLE_COUNT" ]; then
        echo "   Tablas creadas: $TABLE_COUNT"
    fi
else
    print_error "Error al ejecutar migraciones"
    print_info "Revisa el log: cat /tmp/ef_migration.log"
    cat /tmp/ef_migration.log | tail -20
    exit 1
fi

echo ""

# Verificar tablas creadas
print_header "6️⃣  Verificando tablas creadas..."
echo ""

TABLES=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;" 2>/dev/null)

if [ ! -z "$TABLES" ]; then
    print_success "Tablas en la base de datos:"
    echo "$TABLES" | while read -r table; do
        if [ ! -z "$table" ]; then
            echo "   - $table"
        fi
    done
else
    print_warning "No se encontraron tablas en la base de datos"
fi

echo ""

# Resumen final
print_header "═══════════════════════════════════════════════════════════════"
print_header "📋 RESUMEN"
print_header "═══════════════════════════════════════════════════════════════"
echo ""

print_success "✅ Base de datos configurada exitosamente"
echo ""
echo "   Configuración:"
echo "   - Host: $DB_HOST"
echo "   - Puerto: $DB_PORT"
echo "   - Base de datos: $DB_NAME"
echo "   - Usuario: $DB_USER"
echo ""
print_info "La WebApp ahora debería poder conectarse correctamente"
echo ""

# Mostrar cadena de conexión
print_info "Cadena de conexión:"
echo "   Host=$DB_HOST;Port=$DB_PORT;Database=$DB_NAME;Username=$DB_USER;Password=$DB_PASSWORD;"
echo ""

print_header "═══════════════════════════════════════════════════════════════"
echo ""

