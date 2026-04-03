#!/usr/bin/env bash

set -euo pipefail

# Restart the SAM GPU container.
# Use this script from the project root:
#   ./scripts/restart_sam_container.sh

cd "$(dirname "${BASH_SOURCE[0]}")/.."

docker compose -f docker-compose.gpu.yml restart sam-website

