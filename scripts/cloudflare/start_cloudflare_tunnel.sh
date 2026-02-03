#!/bin/bash

# Quick start script for Cloudflare Tunnel (temporary URL)
# This creates a temporary public URL using trycloudflare.com

LOCAL_PORT=5013

echo "🚀 Starting Cloudflare Tunnel"
echo "=============================="
echo ""
echo "📍 Local service: http://localhost:$LOCAL_PORT"
echo "🌐 Public URL will be shown below..."
echo ""
echo "⚠️  Note: This URL is temporary and changes on each restart"
echo "   For a permanent URL, use setup_cloudflare_tunnel.sh"
echo ""
echo "Press Ctrl+C to stop the tunnel"
echo ""

# Check if Flask app is running
if ! curl -s http://localhost:$LOCAL_PORT > /dev/null 2>&1; then
    echo "⚠️  Warning: Flask app doesn't seem to be running on port $LOCAL_PORT"
    echo "   Make sure to start it first:"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    echo "   cd $PROJECT_ROOT"
    echo "   python3 src/web/sam_website.py"
    echo "   Or use: ./scripts/docker_start.sh gpu"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the tunnel
cloudflared tunnel --url "http://localhost:$LOCAL_PORT"
