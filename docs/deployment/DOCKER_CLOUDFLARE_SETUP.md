# Running Docker with Cloudflare Tunnel (mibseg.com)

This guide explains how to run your Dockerized Flask app through Cloudflare Tunnel with your domain `mibseg.com`.

## 🚀 Quick Start

### Option 1: Automated Script (Recommended)

```bash
cd /home/mib/Jimmy/Volume_Prediction
./scripts/start_docker_with_cloudflare.sh gpu
```

This script will:
1. Start Docker container
2. Wait for Flask app to be ready
3. Start Cloudflare Tunnel
4. Your app will be available at **https://mibseg.com**

### Option 2: Manual Steps

**Terminal 1 - Start Docker:**
```bash
cd /home/mib/Jimmy/Volume_Prediction
./scripts/docker_start.sh gpu
```

**Terminal 2 - Start Cloudflare Tunnel:**
```bash
cloudflared tunnel run sam-website
```

## 📋 Prerequisites

✅ Your Cloudflare tunnel is already configured:
- Tunnel name: `sam-website`
- Domain: `mibseg.com`
- Points to: `http://localhost:5013`

✅ Docker is configured and working

## 🔄 Switching from Direct Flask to Docker

If you're currently running Flask directly (not Docker), use:

```bash
./scripts/switch_to_docker.sh
```

This will:
1. Stop the direct Flask process
2. Start Docker container
3. Cloudflare tunnel will automatically connect to Docker

## ✅ Verification

### Check Docker is running:
```bash
docker ps | grep sam-segmentation-app-gpu
curl http://localhost:5013
```

### Check Cloudflare Tunnel:
```bash
cloudflared tunnel list
cloudflared tunnel info sam-website
```

### Check Public Access:
```bash
curl https://mibseg.com
```

## 🔧 Running as Services (Production)

### Docker (Auto-restart)
Docker is already configured with `restart: unless-stopped` in `docker-compose.gpu.yml`, so it will auto-restart on reboot.

### Cloudflare Tunnel (System Service)

```bash
# Install as service (if not already)
sudo cloudflared service install

# Start and enable
sudo systemctl start cloudflared
sudo systemctl enable cloudflared

# Check status
sudo systemctl status cloudflared

# View logs
journalctl -u cloudflared -f
```

## 📊 Architecture

```
Internet
   ↓
Cloudflare Edge (mibseg.com)
   ↓
Cloudflare Tunnel (cloudflared)
   ↓
localhost:5013
   ↓
Docker Container (sam-segmentation-app-gpu)
   ↓
Flask App (port 5013)
```

## 🐛 Troubleshooting

### Docker won't start:
```bash
# Check Docker permissions
docker ps

# Check if port 5013 is in use
lsof -i:5013

# View Docker logs
docker logs sam-segmentation-app-gpu
```

### Cloudflare Tunnel won't connect:
```bash
# Check tunnel is running
cloudflared tunnel list

# Check local service is accessible
curl http://localhost:5013

# Check tunnel config
cat ~/.cloudflared/config.yml

# Restart tunnel
cloudflared tunnel run sam-website
```

### Can't access https://mibseg.com:
```bash
# 1. Verify Docker is running
docker ps

# 2. Verify local access works
curl http://localhost:5013

# 3. Verify tunnel is running
cloudflared tunnel list

# 4. Check DNS (wait 1-2 minutes for propagation)
dig mibseg.com
```

## 📝 Useful Commands

```bash
# Start everything
./scripts/start_docker_with_cloudflare.sh gpu

# Stop Docker
docker compose -f docker-compose.gpu.yml down

# View Docker logs
docker logs -f sam-segmentation-app-gpu

# Restart Docker
docker compose -f docker-compose.gpu.yml restart

# Check Cloudflare tunnel status
cloudflared tunnel info sam-website
```

## 🔒 Security Notes

- Your app is automatically protected by Cloudflare's DDoS protection
- HTTPS is automatically provided by Cloudflare
- Consider adding authentication to your Flask app for additional security
- Use Cloudflare Access for advanced access control
