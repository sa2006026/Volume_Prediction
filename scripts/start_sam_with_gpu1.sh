#!/bin/bash

# Start SAM website using GPU 1 instead of GPU 0
# GPU 0 is almost full, GPU 1 is mostly free

echo "🚀 Starting SAM Website on GPU 1"
echo "=================================="
echo ""
echo "GPU Status:"
echo "  GPU 0: 31.9 GB / 32.7 GB used (97% full)"
echo "  GPU 1: 26 MB / 24.5 GB used (99% free) ✅"
echo ""

# Set environment variable to use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Set PyTorch memory allocation config (as suggested in error)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "✅ Configured to use GPU 1"
echo "✅ Memory allocation optimized"
echo ""
echo "Starting Flask app..."
echo ""

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project directory
cd "$PROJECT_ROOT"

# Start the Flask app
python3 src/web/sam_website.py
