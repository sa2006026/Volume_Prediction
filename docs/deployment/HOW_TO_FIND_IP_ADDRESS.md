# How to Find Your IPv4 Address

## Quick Results

**Your Local/Private IPv4 Address:** `192.168.31.250`  
**Your Public IPv4 Address:** `137.189.62.90`

---

## Understanding IP Addresses

### Local/Private IP Address
- Used within your local network (LAN)
- Usually starts with `192.168.x.x`, `10.x.x.x`, or `172.16-31.x.x`
- Not accessible from the internet
- **Use this for:** Local network connections, router configuration

### Public IP Address
- Your internet-facing address
- Assigned by your ISP
- Accessible from anywhere on the internet
- **Use this for:** Cloudflare setup, remote access, port forwarding

---

## Method 1: Find Local IP Address (Terminal)

### Quick Method:
```bash
hostname -I | awk '{print $1}'
```

### Detailed Method (shows all interfaces):
```bash
ip addr show | grep "inet " | grep -v "127.0.0.1"
```

### Alternative:
```bash
ifconfig | grep "inet " | grep -v "127.0.0.1"
```

**Your local IP:** `192.168.31.250`

---

## Method 2: Find Public IP Address (Terminal)

### Method A (ifconfig.me):
```bash
curl ifconfig.me
```

### Method B (icanhazip.com):
```bash
curl icanhazip.com
```

### Method C (ipify.org):
```bash
curl api.ipify.org
```

### Method D (All in one - tries multiple):
```bash
curl -s ifconfig.me || curl -s icanhazip.com || curl -s api.ipify.org
```

**Your public IP:** `137.189.62.90`

---

## Method 3: Using Python Script

Create a simple script:

```python
#!/usr/bin/env python3
import socket
import urllib.request

# Local IP
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(f"Local IP: {local_ip}")

# Public IP
try:
    public_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
    print(f"Public IP: {public_ip}")
except:
    print("Could not determine public IP")
```

Run it:
```bash
python3 find_ip.py
```

---

## Method 4: Using GUI (if available)

### On Linux with Network Manager:
1. Click network icon in system tray
2. Select "Connection Information" or "Network Settings"
3. Look for "IPv4 Address"

### Using System Settings:
```bash
# GNOME
nmcli device show | grep IP4.ADDRESS

# Or use GUI
gnome-control-center network
```

---

## Method 5: Check Router/Network Info

### Using ip command (most detailed):
```bash
ip addr show
```

This shows all network interfaces with their IP addresses.

### Find default gateway (router IP):
```bash
ip route | grep default
```

---

## Quick Reference Commands

```bash
# Local IP (main interface)
hostname -I | awk '{print $1}'

# All local IPs
ip addr show | grep "inet " | grep -v "127.0.0.1"

# Public IP
curl ifconfig.me

# Network interface details
ip addr show

# Default gateway (router)
ip route | grep default | awk '{print $3}'
```

---

## For Cloudflare Tunnel

**Important:** You typically **DON'T need your IP address** for Cloudflare Tunnel!

Cloudflare Tunnel works by:
1. Your local `cloudflared` connects to Cloudflare's servers
2. Cloudflare routes traffic through the tunnel
3. No port forwarding or IP configuration needed

**However, if you need your IP for other purposes:**

- **Local IP:** `192.168.31.250` - Use for local network access
- **Public IP:** `137.189.62.90` - Use for external access (if not using Cloudflare Tunnel)

---

## Troubleshooting

### "Command not found"
Install missing tools:
```bash
# For ifconfig
sudo apt install net-tools

# For ip (usually pre-installed)
# Already available on most Linux systems
```

### "Can't determine public IP"
- Check internet connection
- Try different service: `curl icanhazip.com`
- Check firewall settings

### "Multiple IPs shown"
- `192.168.x.x` or `10.x.x.x` = Local network IPs
- `172.16-31.x.x` = Docker/container IPs (usually)
- The first `192.168.x.x` is usually your main local IP

---

## Your Current IPs Summary

```
Local/Private IP:  192.168.31.250
Public IP:        137.189.62.90
```

**Note:** Your public IP may change if you don't have a static IP from your ISP.
