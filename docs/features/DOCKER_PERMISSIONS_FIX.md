# Docker Permissions Fix

## Problem
You're getting this error:
```
permission denied while trying to connect to the Docker daemon socket
```

## Solution

### Step 1: Add your user to the docker group

Run this command (you'll need to enter your password):
```bash
sudo usermod -aG docker $USER
```

### Step 2: Apply the changes

You have two options:

**Option A: Log out and log back in** (Recommended)
- Log out of your session completely
- Log back in
- The changes will take effect

**Option B: Use newgrp** (Quick fix, temporary)
```bash
newgrp docker
```
This starts a new shell with the docker group active. You'll need to run your docker commands in this shell.

### Step 3: Verify it works

Test that Docker works without sudo:
```bash
docker ps
```

If this works without errors, you're all set!

### Step 4: Run the Docker app

Now you can run:
```bash
./scripts/docker_start.sh gpu
# or
./scripts/docker_start.sh cpu
```

## Alternative: Use sudo (Not Recommended)

If you can't add yourself to the docker group, you can use sudo, but it's not recommended:
```bash
sudo docker-compose -f docker-compose.cpu.yml up -d
```

**Note**: Using sudo with Docker can cause file permission issues with mounted volumes.

## Troubleshooting

If you still have issues after adding to docker group:

1. **Check if docker group exists:**
   ```bash
   getent group docker
   ```

2. **Verify your groups:**
   ```bash
   groups
   ```
   You should see "docker" in the list.

3. **Restart Docker service:**
   ```bash
   sudo systemctl restart docker
   ```

4. **Check Docker socket permissions:**
   ```bash
   ls -l /var/run/docker.sock
   ```
   Should show: `srw-rw---- 1 root docker`
