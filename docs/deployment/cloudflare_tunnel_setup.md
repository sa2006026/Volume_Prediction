# Cloudflare Tunnel Setup Guide

This guide will help you expose your Flask SAM website (running on localhost:5013) to the public internet using Cloudflare Tunnel.

## Prerequisites

1. A Cloudflare account (free tier works)
2. A domain name added to Cloudflare (or use Cloudflare's free subdomain)
3. `cloudflared` installed (already installed on your system)

## Method 1: Quick Setup (Recommended for Testing)

### Step 1: Login to Cloudflare
```bash
cloudflared tunnel login
```
This will open a browser window for you to authenticate with Cloudflare.

### Step 2: Create a Tunnel
```bash
cloudflared tunnel create sam-website
```
This creates a tunnel named "sam-website" and saves credentials.

### Step 3: Create a Config File
Create a config file at `~/.cloudflared/config.yml`:

```yaml
tunnel: <tunnel-id-from-step-2>
credentials-file: /home/mib/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: sam-website.yourdomain.com
    service: http://localhost:5013
  - service: http_status:404
```

Replace:
- `<tunnel-id-from-step-2>` with the actual tunnel ID
- `sam-website.yourdomain.com` with your actual domain/subdomain

### Step 4: Create DNS Record
```bash
cloudflared tunnel route dns sam-website sam-website.yourdomain.com
```

### Step 5: Run the Tunnel
```bash
cloudflared tunnel run sam-website
```

## Method 2: Using Cloudflare's Free Subdomain (No Domain Required)

If you don't have a domain, you can use Cloudflare's free trycloudflare.com subdomain:

```bash
cloudflared tunnel --url http://localhost:5013
```

This will give you a temporary URL like: `https://random-words-1234.trycloudflare.com`

**Note:** This URL changes every time you restart the tunnel.

## Method 3: Persistent Service (Recommended for Production)

### Step 1: Install as System Service

```bash
sudo cloudflared service install
```

### Step 2: Create Config File
```bash
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

Add the configuration (same as Method 1, Step 3).

### Step 3: Start the Service
```bash
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

### Step 4: Check Status
```bash
sudo systemctl status cloudflared
```

## Security Considerations

1. **Authentication**: Consider adding authentication to your Flask app before exposing it publicly
2. **HTTPS**: Cloudflare Tunnel automatically provides HTTPS
3. **Rate Limiting**: Configure rate limiting in Cloudflare dashboard
4. **Access Rules**: Set up Cloudflare Access rules if needed

## Troubleshooting

- Check tunnel status: `cloudflared tunnel info sam-website`
- View logs: `journalctl -u cloudflared -f` (for service)
- Test locally: Make sure Flask app is running on port 5013
- Check DNS: Ensure DNS record is properly configured

## Updating Flask App for Production

You may want to update the Flask app to:
- Disable debug mode (already done: `debug=False`)
- Use a production WSGI server (gunicorn, uwsgi)
- Add proper error handling
- Configure CORS if needed
