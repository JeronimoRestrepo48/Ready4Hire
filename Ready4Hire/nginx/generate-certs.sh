#!/bin/bash

# Script to generate self-signed SSL certificates for Ready4Hire
# For development and testing purposes only
# For production, use Let's Encrypt or a trusted CA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="${SCRIPT_DIR}/certs"
DAYS_VALID=365
COUNTRY="CO"
STATE="Antioquia"
CITY="Medellin"
ORG="Ready4Hire"
ORG_UNIT="Development"
COMMON_NAME="ready4hire.local"

echo "🔐 Ready4Hire SSL Certificate Generator"
echo "========================================"
echo ""

# Create certs directory if it doesn't exist
if [ ! -d "$CERT_DIR" ]; then
    echo "📁 Creating certificates directory: $CERT_DIR"
    mkdir -p "$CERT_DIR"
fi

# Check if certificates already exist
if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
    read -p "⚠️  Certificates already exist. Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Aborted. Existing certificates preserved."
        exit 0
    fi
    echo "🗑️  Removing existing certificates..."
    rm -f "$CERT_DIR"/*.pem
fi

echo "🔑 Generating private key..."
openssl genrsa -out "$CERT_DIR/key.pem" 4096

echo "📝 Generating certificate signing request..."
openssl req -new -key "$CERT_DIR/key.pem" -out "$CERT_DIR/csr.pem" \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$ORG_UNIT/CN=$COMMON_NAME"

echo "📜 Generating self-signed certificate (valid for $DAYS_VALID days)..."
openssl x509 -req -days $DAYS_VALID -in "$CERT_DIR/csr.pem" \
    -signkey "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -extfile <(printf "subjectAltName=DNS:$COMMON_NAME,DNS:localhost,DNS:*.ready4hire.local,IP:127.0.0.1")

# Create combined PEM (cert + key) for some applications
cat "$CERT_DIR/cert.pem" "$CERT_DIR/key.pem" > "$CERT_DIR/fullchain.pem"

# Set appropriate permissions
chmod 600 "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/cert.pem"
chmod 600 "$CERT_DIR/fullchain.pem"

# Clean up CSR
rm "$CERT_DIR/csr.pem"

echo ""
echo "✅ SSL Certificates generated successfully!"
echo ""
echo "📂 Certificate files:"
echo "   - Private Key: $CERT_DIR/key.pem"
echo "   - Certificate: $CERT_DIR/cert.pem"
echo "   - Full Chain:  $CERT_DIR/fullchain.pem"
echo ""
echo "🔍 Certificate Information:"
openssl x509 -in "$CERT_DIR/cert.pem" -noout -subject -dates -issuer
echo ""
echo "📋 Next Steps:"
echo "   1. Update nginx.conf with correct certificate paths:"
echo "      ssl_certificate     $CERT_DIR/cert.pem;"
echo "      ssl_certificate_key $CERT_DIR/key.pem;"
echo ""
echo "   2. For Nginx system-wide installation:"
echo "      sudo mkdir -p /etc/nginx/certs/ready4hire"
echo "      sudo cp $CERT_DIR/*.pem /etc/nginx/certs/ready4hire/"
echo "      sudo chmod 600 /etc/nginx/certs/ready4hire/key.pem"
echo ""
echo "   3. Reload Nginx:"
echo "      sudo nginx -t"
echo "      sudo systemctl reload nginx"
echo ""
echo "   4. Add to /etc/hosts (if using ready4hire.local):"
echo "      127.0.0.1  ready4hire.local"
echo ""
echo "   5. Trust the certificate (optional, to avoid browser warnings):"
echo "      Linux: sudo cp $CERT_DIR/cert.pem /usr/local/share/ca-certificates/ready4hire.crt && sudo update-ca-certificates"
echo "      macOS: sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $CERT_DIR/cert.pem"
echo ""
echo "⚠️  WARNING: This is a self-signed certificate for DEVELOPMENT only!"
echo "   For PRODUCTION, use Let's Encrypt or a trusted Certificate Authority."
echo ""
