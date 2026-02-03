# Docker Setup Guide for SAM Segmentation Flask App

This guide explains how to run the SAM Segmentation Flask application using Docker.

## Prerequisites

1. **Install Docker**: 
   - Ubuntu/Debian: `sudo apt-get install docker.io docker-compose`
   - Or follow: https://docs.docker.com/get-docker/

2. **For GPU Support** (optional):
   - Install NVIDIA Container Toolkit:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Quick Start

### Option 1: CPU Mode (No GPU Required)

```bash
# Build and run
docker-compose -f docker-compose.cpu.yml up -d

# View logs
docker-compose -f docker-compose.cpu.yml logs -f

# Stop
docker-compose -f docker-compose.cpu.yml down
```

### Option 2: GPU Mode (With NVIDIA GPU)

```bash
# Build and run
docker-compose -f docker-compose.gpu.yml up -d

# View logs
docker-compose -f docker-compose.gpu.yml logs -f

# Stop
docker-compose -f docker-compose.gpu.yml down
```

### Option 3: Standard Mode (Auto-detect)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Manual Docker Commands

### Build the image:

```bash
# GPU version
docker build -t sam-website:latest .

# CPU version
docker build -f Dockerfile.cpu -t sam-website:cpu .
```

### Run the container:

```bash
# GPU version
docker run -d \
  --name sam-app \
  --gpus all \
  -p 5013:5013 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/templates:/app/templates \
  -v $(pwd)/model:/app/model \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  sam-website:latest

# CPU version
docker run -d \
  --name sam-app-cpu \
  -p 5013:5013 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/templates:/app/templates \
  -v $(pwd)/model:/app/model \
  -e CUDA_VISIBLE_DEVICES="" \
  sam-website:cpu
```

## Accessing the Application

Once the container is running, access the application at:
- **Local**: http://localhost:5013
- **Network**: http://YOUR_SERVER_IP:5013

## Useful Commands

### View logs:
```bash
docker-compose logs -f
# or
docker logs -f sam-segmentation-app
```

### Stop the container:
```bash
docker-compose down
# or
docker stop sam-segmentation-app && docker rm sam-segmentation-app
```

### Restart the container:
```bash
docker-compose restart
# or
docker restart sam-segmentation-app
```

### Execute commands inside container:
```bash
docker exec -it sam-segmentation-app bash
```

### Check container status:
```bash
docker ps
docker-compose ps
```

### View resource usage:
```bash
docker stats sam-segmentation-app
```

## Volume Mounts

The following directories are mounted as volumes (data persists outside container):
- `./uploads` → `/app/uploads` - Uploaded images
- `./results` → `/app/results` - Segmentation results
- `./templates` → `/app/templates` - HTML templates
- `./model` → `/app/model` - SAM model files
- `./data` → `/app/data` (read-only) - Data files

## Environment Variables

You can customize behavior with environment variables:

- `CUDA_VISIBLE_DEVICES`: Which GPU to use (e.g., "0", "1", or "" for CPU)
- `PYTORCH_CUDA_ALLOC_CONF`: PyTorch memory allocation config
- `PYTHONPATH`: Python path (set to `/app`)

## Troubleshooting

### Container won't start:
```bash
# Check logs
docker-compose logs

# Check if port is already in use
sudo lsof -i :5013
```

### GPU not detected:
```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check if nvidia-container-toolkit is installed
docker info | grep -i runtime
```

### Permission issues:
```bash
# Add your user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Out of memory:
```bash
# Check container memory usage
docker stats

# Increase Docker memory limit in Docker Desktop settings
```

## Building from Scratch

If you need to rebuild everything:

```bash
# Remove old images
docker-compose down
docker rmi sam-website:latest

# Rebuild without cache
docker-compose build --no-cache

# Start fresh
docker-compose up -d
```

## Integration with Cloudflare Tunnel

If you're using Cloudflare Tunnel, you can run it alongside Docker:

```bash
# In one terminal: Start Docker container
docker-compose up -d

# In another terminal: Start Cloudflare Tunnel
./scripts/cloudflare/start_cloudflare_tunnel.sh
```

The tunnel will forward traffic to `localhost:5013` which is exposed by Docker.

## Notes

- The application runs on port 5013 inside the container
- Data in mounted volumes persists even if container is removed
- GPU support requires NVIDIA Docker runtime
- First build may take 10-15 minutes (downloading base images and dependencies)
