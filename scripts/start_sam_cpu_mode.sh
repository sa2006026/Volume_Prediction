#!/bin/bash

# Start SAM website in CPU mode (no GPU required)
# Useful when GPU memory is full

echo "🚀 Starting SAM Website in CPU Mode"
echo "===================================="
echo ""
echo "⚠️  Note: CPU mode will be slower but uses no GPU memory"
echo ""

# Force CPU mode by hiding GPU
export CUDA_VISIBLE_DEVICES=""

echo "✅ Configured to use CPU only"
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
