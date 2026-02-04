#!/bin/bash

# Switch from direct Flask to Docker while keeping Cloudflare Tunnel running
# This script stops the direct Flask app and starts Docker instead

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_PORT=5013

echo "🔄 Switching to Docker + Cloudflare Tunnel"
echo "==========================================="
echo ""

# Check if something is running on port 5013
if lsof -ti:$LOCAL_PORT > /dev/null 2>&1; then
    echo "⚠️  Port $LOCAL_PORT is in use"
    PID=$(lsof -ti:$LOCAL_PORT | head -1)
    PROCESS=$(ps -p $PID -o comm= 2>/dev/null || echo "unknown")
    echo "   Process: $PROCESS (PID: $PID)"
    echo ""
    read -p "Stop this process and start Docker? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping process on port $LOCAL_PORT..."
        kill $PID 2>/dev/null || sudo kill $PID
        sleep 2
        echo "✅ Process stopped"
    else
        echo "❌ Aborted. Please stop the process manually first."
        exit 1
    fi
fi

# Start Docker
echo ""
echo "Starting Docker container..."
cd "$PROJECT_ROOT"
./scripts/docker_start.sh gpu

# Wait for Docker to be ready
echo ""
echo "Waiting for Docker container to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$LOCAL_PORT > /dev/null 2>&1; then
        echo "✅ Docker container is ready"
        break
    fi
    sleep 1
done

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Status:"
echo "   - Docker: Running on port $LOCAL_PORT"
echo "   - Cloudflare Tunnel: Check with 'cloudflared tunnel list'"
echo "   - Public URL: https://mibseg.com"
echo ""
echo "To verify Cloudflare tunnel is running:"
echo "   cloudflared tunnel run sam-website"
echo ""
