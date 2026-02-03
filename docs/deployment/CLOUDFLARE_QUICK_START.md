# Cloudflare Tunnel - Quick Start Guide

## 🚀 Fastest Way (Temporary URL - No Setup Required)

If you just want to quickly test or share your app temporarily:

```bash
# 1. Start your Flask app first (in one terminal)
cd /home/mib/Jimmy/Volume_Prediction
python3 src/web/sam_website.py

# 2. In another terminal, start the tunnel
./scripts/cloudflare/start_cloudflare_tunnel.sh
```

This will give you a temporary public URL like: `https://random-words-1234.trycloudflare.com`

**Note:** The URL changes every time you restart the tunnel.

---

## 🌐 Permanent Setup (Custom Domain)

For a permanent public URL with your own domain:

```bash
# Run the setup script
./scripts/cloudflare/setup_cloudflare_tunnel.sh

# Choose option 1 (Custom domain)
# Follow the prompts to:
# - Login to Cloudflare
# - Create tunnel
# - Enter your domain
# - Configure DNS

# Then run the tunnel
cloudflared tunnel run sam-website
```

---

## 🔧 Install as System Service (Recommended for Production)

After setting up with custom domain:

```bash
# Install cloudflared as a service
sudo cloudflared service install

# Start the service
sudo systemctl start cloudflared
sudo systemctl enable cloudflared

# Check status
sudo systemctl status cloudflared

# View logs
journalctl -u cloudflared -f
```

---

## 📋 Prerequisites

1. **Cloudflare Account**: Sign up at https://dash.cloudflare.com (free)
2. **Domain** (for permanent setup): Add your domain to Cloudflare
3. **cloudflared**: Already installed ✅

---

## 🔒 Security Recommendations

Before exposing your app publicly, consider:

1. **Add Authentication**: Protect your Flask app with login
2. **Rate Limiting**: Configure in Cloudflare dashboard
3. **Access Rules**: Set up Cloudflare Access for additional security
4. **HTTPS**: Automatically provided by Cloudflare ✅

---

## 🐛 Troubleshooting

**Tunnel won't start:**
- Check if Flask app is running: `curl http://localhost:5013`
- Check tunnel status: `cloudflared tunnel info sam-website`
- View logs: `journalctl -u cloudflared -f`

**Can't access public URL:**
- Wait 1-2 minutes for DNS propagation
- Check DNS record in Cloudflare dashboard
- Verify tunnel is running: `cloudflared tunnel list`

**Connection refused:**
- Ensure Flask app is running on port 5013
- Check firewall settings
- Verify localhost binding in Flask app

---

## 📚 More Information

See `cloudflare_tunnel_setup.md` for detailed instructions.
