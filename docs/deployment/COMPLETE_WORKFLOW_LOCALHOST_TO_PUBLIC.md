# Complete Workflow: Localhost to Public URL via Cloudflare Tunnel

This document provides a detailed, step-by-step workflow for exposing your local Flask application to the public internet using Cloudflare Tunnel.

---

## 📋 Overview

**Goal:** Make your Flask app running on `localhost:5013` accessible at `https://mibseg.com`

**Method:** Cloudflare Tunnel (no port forwarding, no router configuration needed)

**Time Required:** 10-15 minutes

---

## 🔄 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW OVERVIEW                            │
└─────────────────────────────────────────────────────────────────┘

1. Prerequisites Check
   ↓
2. Domain Setup in Cloudflare
   ↓
3. Authenticate with Cloudflare
   ↓
4. Create Cloudflare Tunnel
   ↓
5. Configure Tunnel
   ↓
6. Create DNS Record
   ↓
7. Start Flask Application
   ↓
8. Run Tunnel
   ↓
9. Test Public Access
   ↓
10. (Optional) Install as Service
```

---

## 📝 Detailed Step-by-Step Workflow

### **PHASE 1: Prerequisites** ⏱️ 5 minutes

#### Step 1.1: Verify Requirements

**Checklist:**
- [ ] Cloudflare account (free tier works)
- [ ] Domain name (e.g., `mibseg.com`)
- [ ] Domain added to Cloudflare
- [ ] `cloudflared` installed
- [ ] Flask app ready to run

**Commands:**
```bash
# Check if cloudflared is installed
cloudflared --version

# Should show version like: cloudflared version 2024.11.0
```

**Expected Result:** ✅ Version number displayed

---

### **PHASE 2: Domain Setup in Cloudflare** ⏱️ 5-30 minutes

#### Step 2.1: Add Domain to Cloudflare

1. **Go to Cloudflare Dashboard:**
   - Visit: https://dash.cloudflare.com
   - Click **"Add a Site"** button

2. **Enter Your Domain:**
   - Type your domain: `mibseg.com`
   - Click **"Add site"**

3. **Choose Plan:**
   - Select **Free** plan
   - Click **"Continue"**

4. **Review DNS Records:**
   - Cloudflare will scan existing DNS records
   - Review and confirm records
   - Click **"Continue"**

#### Step 2.2: Update Nameservers

1. **Get Cloudflare Nameservers:**
   - Cloudflare will show you two nameservers, e.g.:
     - `alice.ns.cloudflare.com`
     - `bob.ns.cloudflare.com`

2. **Update at Domain Registrar:**
   - Log in to your domain registrar (where you bought the domain)
   - Go to DNS/Nameserver settings
   - Replace existing nameservers with Cloudflare's nameservers
   - Save changes

3. **Wait for Propagation:**
   - Usually takes 5-30 minutes
   - Cloudflare dashboard will show "Active" when ready
   - You'll receive an email confirmation

**Expected Result:** ✅ Domain shows as "Active" in Cloudflare dashboard

---

### **PHASE 3: Cloudflare Tunnel Setup** ⏱️ 5 minutes

#### Step 3.1: Authenticate with Cloudflare

**Command:**
```bash
cloudflared tunnel login
```

**What Happens:**
1. Command opens your default web browser
2. Browser shows Cloudflare login page
3. Log in to your Cloudflare account
4. Select the domain you want to use (e.g., `mibseg.com`)
5. Click **"Authorize"**

**Expected Output:**
```
You have successfully logged in.
If you wish to copy your credentials to a server, they have been saved to:
/home/mib/.cloudflared/xxxxx.json
```

**Expected Result:** ✅ Credentials file created in `~/.cloudflared/`

---

#### Step 3.2: Create a Tunnel

**Command:**
```bash
cloudflared tunnel create sam-website
```

**What Happens:**
- Cloudflare creates a new tunnel named "sam-website"
- Generates a unique tunnel ID
- Creates credentials file

**Expected Output:**
```
Created tunnel sam-website with id 0f7ad553-782d-4ce6-80bc-5d994aa8b4da
```

**Important:** Save the tunnel ID! You'll need it for configuration.

**Expected Result:** ✅ Tunnel created, credentials file saved

---

#### Step 3.3: Create Configuration File

**Command:**
```bash
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
```

**File Content:**
```yaml
tunnel: sam-website
credentials-file: /home/mib/.cloudflared/0f7ad553-782d-4ce6-80bc-5d994aa8b4da.json

ingress:
  - hostname: mibseg.com
    service: http://localhost:5013
  - service: http_status:404
```

**Explanation:**
- `tunnel: sam-website` - Name of your tunnel
- `credentials-file` - Path to credentials (use your actual tunnel ID)
- `hostname: mibseg.com` - Your public domain
- `service: http://localhost:5013` - Your local Flask app
- `http_status:404` - Catch-all for unmatched requests

**Save the file:**
- Press `Ctrl+X`
- Press `Y` to confirm
- Press `Enter` to save

**Expected Result:** ✅ Config file created at `~/.cloudflared/config.yml`

---

#### Step 3.4: Create DNS Record

**Command:**
```bash
cloudflared tunnel route dns sam-website mibseg.com
```

**What Happens:**
- Creates a CNAME record in Cloudflare DNS
- Points `mibseg.com` to your tunnel
- Automatically configured in Cloudflare dashboard

**Expected Output:**
```
Successfully created CNAME mibseg.com which will route to this tunnel
```

**Expected Result:** ✅ DNS record created in Cloudflare

**Verify in Dashboard:**
- Go to Cloudflare Dashboard → DNS → Records
- You should see a CNAME record for `mibseg.com` pointing to your tunnel

---

### **PHASE 4: Start Your Application** ⏱️ 1 minute

#### Step 4.1: Start Flask Application

**Command:**
```bash
cd /home/mib/Jimmy/Volume_Prediction
python3 src/web/sam_website.py
```

**Expected Output:**
```
🚀 Starting SAM Interactive Segmentation Website...
📍 Server will be available at: http://localhost:5014
🎯 Features: Upload images, configure SAM parameters, interactive mask management
 * Running on http://127.0.0.1:5013
```

**Verify it's running:**
```bash
# In another terminal
curl http://localhost:5013
```

**Expected Result:** ✅ HTML content returned

**Keep this terminal open!** The Flask app must be running for the tunnel to work.

---

### **PHASE 5: Start the Tunnel** ⏱️ 1 minute

#### Step 5.1: Run the Tunnel

**Command:**
```bash
cloudflared tunnel run sam-website
```

**Expected Output:**
```
2026-01-30T05:15:48Z INF Starting tunnel tunnelID=0f7ad553-782d-4ce6-80bc-5d994aa8b4da
2026-01-30T05:15:48Z INF Version 2024.11.0
2026-01-30T05:15:48Z INF Registered tunnel connection connIndex=0 connection=...
2026-01-30T05:15:48Z INF Registered tunnel connection connIndex=1 connection=...
```

**What to Look For:**
- ✅ `Starting tunnel` - Tunnel is initializing
- ✅ `Registered tunnel connection` - Successfully connected to Cloudflare
- ✅ Multiple connections (connIndex 0, 1, 2, 3) - Redundant connections for reliability

**Keep this terminal open!** The tunnel must be running for public access.

**Expected Result:** ✅ Tunnel running and connected to Cloudflare

---

### **PHASE 6: Test Public Access** ⏱️ 2 minutes

#### Step 6.1: Wait for DNS Propagation

**Wait Time:** 1-2 minutes after creating DNS record

**Why:** DNS changes need time to propagate globally

#### Step 6.2: Test from Browser

1. **Open Browser:**
   - Open any web browser
   - Go to: `https://mibseg.com`

2. **Expected Result:**
   - ✅ Your Flask app loads
   - ✅ Same content as `http://localhost:5013`
   - ✅ HTTPS is automatically enabled

#### Step 6.3: Test from Command Line

**Command:**
```bash
curl https://mibseg.com
```

**Expected Result:** ✅ HTML content returned (same as localhost)

---

### **PHASE 7: Verify Everything Works** ⏱️ 2 minutes

#### Step 7.1: Check Tunnel Status

**Command:**
```bash
cloudflared tunnel info sam-website
```

**Expected Output:**
```
Tunnel ID: 0f7ad553-782d-4ce6-80bc-5d994aa8b4da
Name: sam-website
Created: 2026-01-30T05:13:41Z
Connections: 4 active
```

#### Step 7.2: Check DNS Record

**In Cloudflare Dashboard:**
- Go to: DNS → Records
- Verify CNAME record exists for `mibseg.com`
- Should point to your tunnel

#### Step 7.3: Test Local Access

**Command:**
```bash
curl http://localhost:5013
```

**Expected Result:** ✅ Returns Flask app HTML

#### Step 7.4: Test Public Access

**Command:**
```bash
curl https://mibseg.com
```

**Expected Result:** ✅ Returns same Flask app HTML

---

### **PHASE 8: Make It Permanent (Optional)** ⏱️ 3 minutes

#### Step 8.1: Install Tunnel as Service

**Command:**
```bash
sudo cloudflared service install
```

**What Happens:**
- Installs cloudflared as a systemd service
- Reads config from `~/.cloudflared/config.yml`
- Sets up automatic startup

**Expected Result:** ✅ Service installed

#### Step 8.2: Start the Service

**Command:**
```bash
sudo systemctl start cloudflared
```

**Expected Result:** ✅ Service started

#### Step 8.3: Enable Auto-Start on Boot

**Command:**
```bash
sudo systemctl enable cloudflared
```

**Expected Result:** ✅ Service will start automatically on system boot

#### Step 8.4: Verify Service Status

**Command:**
```bash
sudo systemctl status cloudflared
```

**Expected Output:**
```
● cloudflared.service - cloudflared
     Loaded: loaded
     Active: active (running)
```

**Expected Result:** ✅ Service running

#### Step 8.5: View Service Logs

**Command:**
```bash
sudo journalctl -u cloudflared -f
```

**Expected Result:** ✅ Real-time logs showing tunnel connections

**Now you can:**
- ✅ Close all terminals
- ✅ Reboot your computer
- ✅ Tunnel will keep running automatically

---

## 🔍 Troubleshooting Workflow

### Issue: "Tunnel credentials file doesn't exist"

**Workflow to Fix:**
1. Check if credentials file exists:
   ```bash
   ls -la ~/.cloudflared/*.json
   ```
2. Verify config file points to correct file:
   ```bash
   cat ~/.cloudflared/config.yml
   ```
3. If wrong, update config with correct tunnel ID
4. Re-run tunnel

### Issue: "Connection refused" when accessing public URL

**Workflow to Fix:**
1. Check Flask app is running:
   ```bash
   curl http://localhost:5013
   ```
2. If not running, start Flask app
3. Check tunnel is running:
   ```bash
   cloudflared tunnel info sam-website
   ```
4. If not running, start tunnel
5. Wait 1-2 minutes for DNS propagation
6. Try again

### Issue: "DNS record not found"

**Workflow to Fix:**
1. Check DNS record exists:
   ```bash
   cloudflared tunnel route dns sam-website mibseg.com
   ```
2. Verify in Cloudflare dashboard
3. Wait for DNS propagation (1-2 minutes)

---

## 📊 Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKFLOW SUMMARY                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. ✅ Prerequisites Check                                   │
│     → Cloudflare account, domain, cloudflared              │
│                                                              │
│  2. ✅ Domain Setup                                          │
│     → Add domain to Cloudflare                              │
│     → Update nameservers                                    │
│                                                              │
│  3. ✅ Tunnel Authentication                                 │
│     → cloudflared tunnel login                              │
│                                                              │
│  4. ✅ Create Tunnel                                        │
│     → cloudflared tunnel create sam-website                 │
│                                                              │
│  5. ✅ Configure Tunnel                                      │
│     → Create ~/.cloudflared/config.yml                     │
│                                                              │
│  6. ✅ Create DNS Record                                    │
│     → cloudflared tunnel route dns sam-website mibseg.com   │
│                                                              │
│  7. ✅ Start Flask App                                      │
│     → python3 src/web/sam_website.py                        │
│                                                              │
│  8. ✅ Start Tunnel                                         │
│     → cloudflared tunnel run sam-website                    │
│                                                              │
│  9. ✅ Test Public Access                                  │
│     → https://mibseg.com                                    │
│                                                              │
│  10. ✅ (Optional) Install as Service                      │
│      → sudo cloudflared service install                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Reference Commands

```bash
# 1. Login
cloudflared tunnel login

# 2. Create tunnel
cloudflared tunnel create sam-website

# 3. Create DNS record
cloudflared tunnel route dns sam-website mibseg.com

# 4. Start Flask app
python3 src/web/sam_website.py

# 5. Run tunnel
cloudflared tunnel run sam-website

# 6. Install as service
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared

# 7. Check status
cloudflared tunnel info sam-website
sudo systemctl status cloudflared
```

---

## ✅ Success Checklist

- [ ] Domain added to Cloudflare
- [ ] Nameservers updated
- [ ] Authenticated with Cloudflare
- [ ] Tunnel created
- [ ] Config file created
- [ ] DNS record created
- [ ] Flask app running
- [ ] Tunnel running
- [ ] Public URL accessible
- [ ] (Optional) Installed as service

---

## 📚 Additional Resources

- **Cloudflare Dashboard:** https://dash.cloudflare.com
- **Tunnel Documentation:** https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- **Tunnel Metrics:** Available in Cloudflare dashboard

---

## 🎉 You're Done!

Your localhost application is now publicly accessible at:
**https://mibseg.com**

The tunnel will automatically:
- ✅ Provide HTTPS encryption
- ✅ Handle DNS routing
- ✅ Work behind firewalls/NAT
- ✅ Scale with Cloudflare's global network
