#!/bin/bash

# Start Docker container and Cloudflare Tunnel for mibseg.com
# This script starts both services needed to expose your Dockerized Flask app publicly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TUNNEL_NAME="sam-website"
LOCAL_PORT=5013

echo "🚀 Starting Docker + Cloudflare Tunnel for mibseg.com"
echo "======================================================"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared is not installed."
    echo "   Install it from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
    exit 1
fi

# Check Docker permissions
if ! docker ps &> /dev/null; then
    echo "⚠️  Docker permission issue detected."
    echo ""
    echo "   You need to either:"
    echo "   1. Add your user to docker group:"
    echo "      sudo usermod -aG docker $USER"
    echo "      Then log out and log back in"
    echo ""
    echo "   2. Or run with sudo (not recommended):"
    echo "      sudo $0"
    echo ""
    exit 1
fi

# Check if tunnel exists
if ! cloudflared tunnel list | grep -q "$TUNNEL_NAME"; then
    echo "⚠️  Tunnel '$TUNNEL_NAME' not found."
    echo "   Setting up tunnel first..."
    echo ""
    "$SCRIPT_DIR/cloudflare/setup_cloudflare_tunnel.sh"
    exit 0
fi

# Step 1: Start Docker container
echo "Step 1: Starting Docker container..."
echo ""

cd "$PROJECT_ROOT"

# Determine which mode to use (default: gpu)
MODE=${1:-gpu}

case $MODE in
    cpu)
        COMPOSE_FILE="docker-compose.cpu.yml"
        ;;
    gpu)
        # Check for NVIDIA Docker
        if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            COMPOSE_FILE="docker-compose.gpu.yml"
        else
            echo "⚠️  GPU not available, falling back to CPU mode"
            COMPOSE_FILE="docker-compose.cpu.yml"
        fi
        ;;
    *)
        echo "Usage: $0 [cpu|gpu]"
        exit 1
        ;;
esac

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    docker compose -f "$COMPOSE_FILE" up -d --build
else
    docker-compose -f "$COMPOSE_FILE" up -d --build
fi

if [ $? -ne 0 ]; then
    echo "❌ Failed to start Docker container"
    exit 1
fi

echo "✅ Docker container started"
echo ""

# Wait for Flask app to be ready
echo "Waiting for Flask app to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$LOCAL_PORT > /dev/null 2>&1; then
        echo "✅ Flask app is ready on port $LOCAL_PORT"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Flask app not responding after 30 seconds"
        echo "   Check logs: docker logs sam-segmentation-app-gpu"
    fi
    sleep 1
done
echo ""

# Step 2: Start Cloudflare Tunnel
echo "Step 2: Starting Cloudflare Tunnel..."
echo ""
echo "📍 Local service: http://localhost:$LOCAL_PORT"
echo "🌐 Public URL: https://mibseg.com"
echo ""
echo "⚠️  Tunnel will run in foreground. Press Ctrl+C to stop both services."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    cd "$PROJECT_ROOT"
    if docker compose version &> /dev/null; then
        docker compose -f "$COMPOSE_FILE" down
    else
        docker-compose -f "$COMPOSE_FILE" down
    fi
    echo "✅ Services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start the tunnel (this will run in foreground)
cloudflared tunnel run "$TUNNEL_NAME"
