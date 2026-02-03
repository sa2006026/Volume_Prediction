#!/bin/bash

# Quick test to verify PyTorch CUDA in container

CONTAINER_NAME="sam-segmentation-app-gpu"

echo "🔍 Testing PyTorch CUDA in Container"
echo "======================================"
echo ""

if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "❌ Container not running. Start it with: ./scripts/docker_start.sh gpu"
    exit 1
fi

echo "Running PyTorch CUDA test..."
docker exec "$CONTAINER_NAME" python3 << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ CUDA is working!")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test tensor creation on GPU
    try:
        x = torch.randn(3, 3).cuda()
        print(f"✅ Successfully created tensor on GPU: {x.device}")
    except Exception as e:
        print(f"❌ Failed to create tensor on GPU: {e}")
else:
    print("❌ CUDA not available")
    print("Possible issues:")
    print("  1. PyTorch was compiled without CUDA support")
    print("  2. CUDA libraries not accessible in container")
    print("  3. GPU runtime not properly configured")
    sys.exit(1)
EOF

echo ""
echo "Test complete!"
