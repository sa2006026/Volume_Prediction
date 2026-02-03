# Permanent URL Setup with Custom Domain - Step by Step Guide

This guide will walk you through setting up a permanent public URL for your SAM website using Cloudflare Tunnel and your own domain.

## Prerequisites Checklist

Before starting, make sure you have:

- [ ] A Cloudflare account (free tier works - sign up at https://dash.cloudflare.com)
- [ ] A domain name (e.g., `yourdomain.com`)
- [ ] Your domain added to Cloudflare (nameservers changed to Cloudflare's)
- [ ] `cloudflared` installed (✅ already installed on your system)
- [ ] Your Flask app running on `localhost:5013`

---

## Method 1: Automated Setup (Easiest)

Simply run the setup script:

```bash
cd /home/mib/Jimmy/Volume_Prediction
./scripts/cloudflare/setup_cloudflare_tunnel.sh
```

Choose option `1` (Custom domain) and follow the prompts. The script will guide you through all steps.

---

## Method 2: Manual Setup (Step by Step)

If you prefer to do it manually or need more control, follow these steps:

### Step 1: Add Your Domain to Cloudflare

1. Go to https://dash.cloudflare.com
2. Click **"Add a Site"**
3. Enter your domain name (e.g., `yourdomain.com`)
4. Choose the **Free** plan
5. Cloudflare will scan your existing DNS records
6. Update your domain's nameservers at your domain registrar to point to Cloudflare's nameservers
   - You'll see instructions like: "Change your nameservers to:"
   - `alice.ns.cloudflare.com`
   - `bob.ns.cloudflare.com`
7. Wait for DNS propagation (usually 5-30 minutes)

### Step 2: Authenticate with Cloudflare

Open a terminal and run:

```bash
cloudflared tunnel login
```

This will:
- Open your browser
- Ask you to select the domain you want to use
- Authenticate and save credentials

**Expected output:**
```
You have successfully logged in.
If you wish to copy your credentials to a server, they have been saved to:
/home/mib/.cloudflared/xxxxx.json
```

### Step 3: Create a Tunnel

Create a new tunnel with a name of your choice:

```bash
cloudflared tunnel create sam-website
```

**Expected output:**
```
Created tunnel sam-website with id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

**Save the tunnel ID** - you'll need it in the next step!

### Step 4: Create Configuration File

Create the config directory if it doesn't exist:

```bash
mkdir -p ~/.cloudflared
```

Create/edit the config file:

```bash
nano ~/.cloudflared/config.yml
```

Add the following content (replace with your actual values):

```yaml
tunnel: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
credentials-file: /home/mib/.cloudflared/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json

ingress:
  - hostname: sam.yourdomain.com
    service: http://localhost:5013
  - service: http_status:404
```

**Replace:**
- `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with your actual tunnel ID (from Step 3)
- `sam.yourdomain.com` with your desired subdomain (e.g., `sam.yourdomain.com` or `app.yourdomain.com`)

**Save the file** (Ctrl+X, then Y, then Enter in nano)

### Step 5: Create DNS Record

Create a DNS record that points your subdomain to the tunnel:

```bash
cloudflared tunnel route dns sam-website sam.yourdomain.com
```

**Replace `sam.yourdomain.com` with your actual subdomain.**

**Expected output:**
```
Successfully created CNAME sam.yourdomain.com which will route to this tunnel
```

### Step 6: Test the Tunnel

First, make sure your Flask app is running:

```bash
cd /home/mib/Jimmy/Volume_Prediction
python3 src/web/sam_website.py
```

In another terminal, run the tunnel:

```bash
cloudflared tunnel run sam-website
```

**Expected output:**
```
2024-01-01T12:00:00Z INF +--------------------------------------------------------------------------------------------+
2024-01-01T12:00:00Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable): |
2024-01-01T12:00:00Z INF |  https://sam.yourdomain.com                                                               |
2024-01-01T12:00:00Z INF +--------------------------------------------------------------------------------------------+
```

### Step 7: Access Your Website

Open your browser and visit: `https://sam.yourdomain.com`

You should see your SAM website! 🎉

**Note:** It may take 1-2 minutes for DNS to propagate. If it doesn't work immediately, wait a bit and try again.

---

## Making It Run Automatically (System Service)

To make the tunnel start automatically and run in the background:

### Step 1: Install as Service

```bash
sudo cloudflared service install
```

### Step 2: Start the Service

```bash
sudo systemctl start cloudflared
```

### Step 3: Enable Auto-Start on Boot

```bash
sudo systemctl enable cloudflared
```

### Step 4: Check Status

```bash
sudo systemctl status cloudflared
```

### Step 5: View Logs (if needed)

```bash
sudo journalctl -u cloudflared -f
```

---

## Troubleshooting

### Issue: "Tunnel not found"
**Solution:** Make sure you're using the correct tunnel name:
```bash
cloudflared tunnel list
```

### Issue: "DNS record already exists"
**Solution:** Delete the old record first:
```bash
cloudflared tunnel route dns delete sam.yourdomain.com
```

### Issue: "Connection refused"
**Solution:** Make sure Flask app is running:
```bash
curl http://localhost:5013
```

### Issue: "Can't access website"
**Solutions:**
1. Wait 1-2 minutes for DNS propagation
2. Check DNS in Cloudflare dashboard
3. Verify tunnel is running: `cloudflared tunnel info sam-website`
4. Check tunnel logs: `sudo journalctl -u cloudflared -f`

### Issue: "Permission denied" when running as service
**Solution:** Make sure config file has correct permissions:
```bash
sudo chown root:root ~/.cloudflared/config.yml
sudo chmod 600 ~/.cloudflared/config.yml
```

---

## Managing Your Tunnel

### List all tunnels:
```bash
cloudflared tunnel list
```

### View tunnel info:
```bash
cloudflared tunnel info sam-website
```

### Delete a tunnel:
```bash
cloudflared tunnel delete sam-website
```

### Update DNS route:
```bash
cloudflared tunnel route dns sam-website new-subdomain.yourdomain.com
```

---

## Security Recommendations

Before going live, consider:

1. **Add Authentication** to your Flask app
2. **Configure Rate Limiting** in Cloudflare dashboard
3. **Set up Cloudflare Access** for additional security
4. **Enable WAF (Web Application Firewall)** in Cloudflare
5. **Use Cloudflare's Bot Fight Mode** to prevent abuse

---

## Example: Complete Setup Session

Here's what a complete setup session looks like:

```bash
# 1. Login to Cloudflare
$ cloudflared tunnel login
# Browser opens, select domain, authenticate

# 2. Create tunnel
$ cloudflared tunnel create sam-website
Created tunnel sam-website with id a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 3. Create config file
$ nano ~/.cloudflared/config.yml
# (Add config as shown in Step 4)

# 4. Create DNS record
$ cloudflared tunnel route dns sam-website sam.yourdomain.com
Successfully created CNAME sam.yourdomain.com which will route to this tunnel

# 5. Run tunnel
$ cloudflared tunnel run sam-website
# Tunnel starts, shows your URL

# 6. Install as service (optional)
$ sudo cloudflared service install
$ sudo systemctl start cloudflared
$ sudo systemctl enable cloudflared
```

---

## Quick Reference

**Start tunnel manually:**
```bash
cloudflared tunnel run sam-website
```

**Start as service:**
```bash
sudo systemctl start cloudflared
```

**Check status:**
```bash
sudo systemctl status cloudflared
```

**View logs:**
```bash
sudo journalctl -u cloudflared -f
```

**Stop tunnel:**
```bash
sudo systemctl stop cloudflared
# or Ctrl+C if running manually
```

---

## Need Help?

- Cloudflare Tunnel Docs: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- Cloudflare Community: https://community.cloudflare.com/
