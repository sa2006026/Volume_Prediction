#!/bin/bash

# Cloudflare Tunnel Setup Script for SAM Website
# This script helps set up a Cloudflare tunnel for your Flask app

set -e

TUNNEL_NAME="sam-website"
LOCAL_PORT=5013
CONFIG_DIR="$HOME/.cloudflared"
CONFIG_FILE="$CONFIG_DIR/config.yml"

echo "🚀 Cloudflare Tunnel Setup for SAM Website"
echo "=========================================="
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared is not installed."
    echo "   Install it from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
    exit 1
fi

echo "✅ cloudflared is installed"
echo ""

# Function to setup with custom domain
setup_custom_domain() {
    echo "📋 Setting up tunnel with custom domain..."
    echo ""
    
    # Step 1: Login
    echo "Step 1: Authenticating with Cloudflare..."
    cloudflared tunnel login
    
    # Step 2: Create tunnel
    echo ""
    echo "Step 2: Creating tunnel '$TUNNEL_NAME'..."
    TUNNEL_OUTPUT=$(cloudflared tunnel create "$TUNNEL_NAME" 2>&1)
    TUNNEL_ID=$(echo "$TUNNEL_OUTPUT" | grep -oP '(?<=Created tunnel )[a-f0-9-]+' || echo "")
    
    if [ -z "$TUNNEL_ID" ]; then
        echo "⚠️  Could not extract tunnel ID. Please check the output above."
        echo "   You may need to manually find the tunnel ID."
        read -p "Enter tunnel ID: " TUNNEL_ID
    fi
    
    echo "✅ Tunnel created with ID: $TUNNEL_ID"
    echo ""
    
    # Step 3: Get domain from user
    read -p "Enter your domain/subdomain (e.g., sam.yourdomain.com): " DOMAIN
    
    # Step 4: Create config directory
    mkdir -p "$CONFIG_DIR"
    
    # Find credentials file
    CREDENTIALS_FILE=$(find "$CONFIG_DIR" -name "${TUNNEL_ID}.json" 2>/dev/null | head -n 1)
    if [ -z "$CREDENTIALS_FILE" ]; then
        CREDENTIALS_FILE="$CONFIG_DIR/${TUNNEL_ID}.json"
    fi
    
    # Step 5: Create config file
    echo ""
    echo "Step 3: Creating config file..."
    cat > "$CONFIG_FILE" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $CREDENTIALS_FILE

ingress:
  - hostname: $DOMAIN
    service: http://localhost:$LOCAL_PORT
  - service: http_status:404
EOF
    
    echo "✅ Config file created at: $CONFIG_FILE"
    echo ""
    
    # Step 6: Create DNS record
    echo "Step 4: Creating DNS record..."
    cloudflared tunnel route dns "$TUNNEL_NAME" "$DOMAIN"
    echo "✅ DNS record created"
    echo ""
    
    echo "✅ Setup complete!"
    echo ""
    echo "To run the tunnel, use:"
    echo "  cloudflared tunnel run $TUNNEL_NAME"
    echo ""
    echo "Or install as a service:"
    echo "  sudo cloudflared service install"
    echo "  sudo systemctl start cloudflared"
    echo "  sudo systemctl enable cloudflared"
}

# Function to setup with trycloudflare (temporary URL)
setup_trycloudflare() {
    echo "📋 Setting up temporary tunnel (trycloudflare.com)..."
    echo ""
    echo "This will create a temporary public URL that changes on each restart."
    echo ""
    read -p "Press Enter to start the tunnel..."
    echo ""
    echo "🚀 Starting tunnel..."
    echo "   Your Flask app should be running on http://localhost:$LOCAL_PORT"
    echo ""
    cloudflared tunnel --url "http://localhost:$LOCAL_PORT"
}

# Main menu
echo "Choose setup method:"
echo "1) Custom domain (permanent URL)"
echo "2) Temporary URL (trycloudflare.com - no domain needed)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        setup_custom_domain
        ;;
    2)
        setup_trycloudflare
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
