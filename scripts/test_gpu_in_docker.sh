#!/bin/bash

# Test script to verify GPU access in Docker container

echo "🔍 Testing GPU Access in Docker Container"
echo "=========================================="
echo ""

CONTAINER_NAME="sam-segmentation-app-gpu"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "❌ Container $CONTAINER_NAME is not running"
    echo "   Start it with: ./scripts/docker_start.sh gpu"
    exit 1
fi

echo "✅ Container is running"
echo ""

# Test 1: Check nvidia-smi
echo "Test 1: Checking nvidia-smi..."
if docker exec "$CONTAINER_NAME" nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi works - GPU is accessible"
    docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
else
    echo "❌ nvidia-smi failed - GPU not accessible"
    echo "   This usually means:"
    echo "   1. NVIDIA Container Toolkit not installed"
    echo "   2. Docker GPU runtime not configured"
    echo "   3. Container not started with GPU support"
fi
echo ""

# Test 2: Check PyTorch CUDA
echo "Test 2: Checking PyTorch CUDA availability..."
docker exec "$CONTAINER_NAME" python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available - PyTorch will use CPU')
"
echo ""

# Test 3: Check environment variables
echo "Test 3: Checking CUDA environment variables..."
docker exec "$CONTAINER_NAME" env | grep -i cuda
echo ""

# Test 4: Check container logs for SAM initialization
echo "Test 4: Checking SAM model initialization..."
docker logs "$CONTAINER_NAME" 2>&1 | grep -i "device\|cuda\|gpu\|Loading SAM" | tail -10
echo ""

echo "=========================================="
echo "If GPU is not accessible, check:"
echo "1. NVIDIA Container Toolkit installed: docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"
echo "2. Docker daemon configured for GPU: cat /etc/docker/daemon.json"
echo "3. Container started with GPU: docker inspect $CONTAINER_NAME | grep -i nvidia"
