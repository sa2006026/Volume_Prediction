# CUDA Out of Memory Fix for SAM Segmentation

## Problem
```
CUDA out of memory. Tried to allocate 768.00 MiB. 
GPU 0 has a total capacity of 31.47 GiB of which 334.75 MiB is free.
```

## Current GPU Status

**GPU 0 (RTX 5000 Ada):** 31.9 GB / 32.7 GB used (97% full!)
- Process 2363019: 11.31 GiB (gunicorn)
- Process 2367490: 1.38 GiB (gunicorn)
- Process 2381843: 15.19 GiB (gunicorn) ⚠️ Largest!
- Process 2382758: 1.38 GiB (gunicorn)
- Process 4059723: 1.67 GiB (your SAM process)

**GPU 1 (RTX A5000):** 26 MiB / 24.5 GB used (99% free!) ✅

## Solutions

### Solution 1: Use GPU 1 Instead (Recommended)

GPU 1 is almost completely free! Modify your code to use GPU 1.

**Option A: Set CUDA Device Environment Variable**

```bash
# Before running Flask app
export CUDA_VISIBLE_DEVICES=1
python3 src/web/sam_website.py
```

**Option B: Modify Code to Use GPU 1**

Add this to your SAM initialization code.

### Solution 2: Use CPU Mode (No GPU Required)

Disable GPU usage in the web interface or code:

**In Web Interface:**
- Uncheck "Use GPU" option when configuring SAM parameters

**In Code:**
- Set `use_gpu=False` when calling `configure_sam_parameters()`

### Solution 3: Use Smaller SAM Model

Switch to a smaller model that uses less memory:
- `vit_b` (base) - Current, uses ~1.2 GB
- Try smaller model if available, or reduce image size

### Solution 4: Clear GPU Memory

**Kill unnecessary processes (if safe):**
```bash
# Check what processes are doing
ps aux | grep 2381843

# If safe to kill (backup first!)
sudo kill 2381843  # The largest process using 15.19 GiB
```

**Clear PyTorch cache:**
```python
import torch
torch.cuda.empty_cache()
```

### Solution 5: Set Memory Allocation Config

As suggested in the error, set:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 src/web/sam_website.py
```

### Solution 6: Reduce Image Size

Process smaller images or reduce resolution before SAM segmentation.

## Quick Fix Commands

### Use GPU 1:
```bash
export CUDA_VISIBLE_DEVICES=1
cd /home/mib/Jimmy/Volume_Prediction
python3 src/web/sam_website.py
```

### Use CPU:
```bash
# In web interface, uncheck "Use GPU"
# Or modify code default to use_gpu=False
```

### Clear GPU Cache (Python):
```python
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Cleared GPU cache. Free memory: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
```

## Recommended Action

**Best solution:** Use GPU 1 since it's almost empty!

```bash
export CUDA_VISIBLE_DEVICES=1
python3 src/web/sam_website.py
```

Then in the web interface, make sure "Use GPU" is checked - it will use GPU 1 instead of GPU 0.
