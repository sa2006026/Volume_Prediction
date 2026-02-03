#!/bin/bash

# Quick start script for Docker SAM Segmentation App

echo "🐳 Docker SAM Segmentation App - Quick Start"
echo "=============================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
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
    echo "      sudo ./docker_start.sh $@"
    echo ""
    echo "   After adding to docker group, you may need to log out/in for changes to take effect."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Determine which mode to use
MODE=${1:-gpu}

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

case $MODE in
    cpu)
        echo "🖥️  Starting in CPU mode..."
        COMPOSE_FILE="$PROJECT_ROOT/docker-compose.cpu.yml"
        ;;
    gpu)
        echo "🎮 Starting in GPU mode..."
        # Check for NVIDIA Docker
        if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            COMPOSE_FILE="$PROJECT_ROOT/docker-compose.gpu.yml"
        else
            echo "⚠️  GPU not available, falling back to CPU mode"
            COMPOSE_FILE="$PROJECT_ROOT/docker-compose.cpu.yml"
        fi
        ;;
    *)
        echo "Usage: $0 [cpu|gpu]"
        exit 1
        ;;
esac

echo ""
echo "📦 Building and starting container..."
echo ""

# Change to project root for docker-compose
cd "$PROJECT_ROOT"

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    docker compose -f "$COMPOSE_FILE" up -d --build
else
    docker-compose -f "$COMPOSE_FILE" up -d --build
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container started successfully!"
    echo ""
    echo "🌐 Application is available at: http://localhost:5013"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs:    docker-compose -f $COMPOSE_FILE logs -f"
    echo "   Stop:         docker-compose -f $COMPOSE_FILE down"
    echo "   Restart:      docker-compose -f $COMPOSE_FILE restart"
    echo ""
else
    echo ""
    echo "❌ Failed to start container. Check logs with:"
    echo "   docker-compose -f $COMPOSE_FILE logs"
    exit 1
fi
