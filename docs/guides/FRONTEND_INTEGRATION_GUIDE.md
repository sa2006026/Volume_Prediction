# Frontend Integration Guide - Bounding Box Management

## The Problem

After applying intensity filter, the frontend preview window still shows **all** bounding boxes instead of only showing boxes for masks **inside the intensity range**.

**Root cause**: Frontend is not updating its stored mask list when the intensity filter response comes back.

---

## Solution: Update Frontend State Management

The frontend needs to maintain a **single source of truth** for the current mask list and update it whenever the backend returns a new `masks` array.

### ❌ Current (Incorrect) Frontend Pattern:

```javascript
// WRONG: Storing masks once and never updating
let allMasks = [];  // Set once after segmentation

async function runSegmentation() {
    const response = await fetch('/run_sam_segmentation', {...});
    allMasks = response.masks;  // Store masks
    drawBoundingBoxes(allMasks);
}

async function applyIntensityFilter(min, max) {
    const response = await fetch('/apply_intensity_filter', {
        body: JSON.stringify({min_intensity: min, max_intensity: max})
    });
    
    // ❌ PROBLEM: Still using old allMasks instead of response.masks
    clearBoundingBoxes();
    drawBoundingBoxes(allMasks);  // WRONG - using stale data!
}
```

### ✅ Correct Frontend Pattern:

```javascript
// RIGHT: Update mask list from every backend response
let currentMasks = [];  // This is the single source of truth

async function runSegmentation() {
    const response = await fetch('/run_sam_segmentation', {...});
    
    if (response.success && response.masks) {
        currentMasks = response.masks;  // ✅ Update from backend
        redrawBoundingBoxes();
    }
}

async function applyIntensityFilter(min, max) {
    const response = await fetch('/apply_intensity_filter', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({min_intensity: min, max_intensity: max})
    });
    
    if (response.success && response.masks) {
        currentMasks = response.masks;  // ✅ Update with filtered list
        redrawBoundingBoxes();          // ✅ Redraw using new list
    }
}

async function resetIntensityFilter() {
    const response = await fetch('/reset_intensity_filter', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'}
    });
    
    if (response.success && response.masks) {
        currentMasks = response.masks;  // ✅ Update with full list
        redrawBoundingBoxes();          // ✅ Redraw all boxes
    }
}

function redrawBoundingBoxes() {
    // Clear ALL existing bounding boxes
    clearAllBoundingBoxes();
    
    // Draw ONLY the masks in currentMasks
    currentMasks.forEach(mask => {
        const [x, y, w, h] = mask.bounding_box;
        drawBoundingBox(x, y, w, h, mask.mask_id);
    });
}
```

---

## Complete Frontend Implementation Example

### HTML Structure:
```html
<div class="container">
    <canvas id="imageCanvas"></canvas>
    <canvas id="boxCanvas"></canvas>  <!-- Separate layer for boxes -->
    
    <div id="controls">
        <input type="file" id="imageUpload">
        <button id="runSegmentation">Run Segmentation</button>
        
        <div id="filterControls">
            <label>Min Intensity: <input type="range" id="minIntensity" min="0" max="255" value="0"></label>
            <label>Max Intensity: <input type="range" id="maxIntensity" min="0" max="255" value="255"></label>
            <button id="applyFilter">Apply Intensity Filter</button>
            <button id="resetFilter">Reset Filter</button>
        </div>
    </div>
    
    <div id="maskInfo"></div>
</div>
```

### JavaScript Implementation:

```javascript
class SAMWebInterface {
    constructor() {
        this.imageCanvas = document.getElementById('imageCanvas');
        this.boxCanvas = document.getElementById('boxCanvas');
        this.imageCtx = this.imageCanvas.getContext('2d');
        this.boxCtx = this.boxCanvas.getContext('2d');
        
        // ✅ Single source of truth for current masks
        this.currentMasks = [];
        
        // Current image
        this.currentImage = null;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        document.getElementById('imageUpload').addEventListener('change', 
            (e) => this.handleImageUpload(e));
        document.getElementById('runSegmentation').addEventListener('click', 
            () => this.runSegmentation());
        document.getElementById('applyFilter').addEventListener('click', 
            () => this.applyIntensityFilter());
        document.getElementById('resetFilter').addEventListener('click', 
            () => this.resetIntensityFilter());
        
        // Hover preview
        this.boxCanvas.addEventListener('mousemove', 
            (e) => this.handleMouseMove(e));
    }
    
    // 1️⃣ Upload Image
    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('image', file);
        
        try {
            const response = await fetch('/upload_image', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                // ✅ Backend returns masks: [] to clear everything
                this.currentMasks = data.masks || [];  
                
                await this.loadImage(data.image);
                this.redrawBoundingBoxes();  // Clears all (empty array)
                
                console.log('✅ Image uploaded, bounding boxes cleared');
            }
        } catch (error) {
            console.error('Upload error:', error);
        }
    }
    
    // 2️⃣ Run Segmentation
    async runSegmentation() {
        const params = {
            model_size: 'vit_b',
            crop_layers: 1,
            points_per_side: 32,
            apply_overlap_filter: true,
            overlap_threshold: 0.4
        };
        
        try {
            const response = await fetch('/run_sam_segmentation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            });
            
            const data = await response.json();
            
            if (data.success && data.masks_found) {
                // ✅ Update mask list with segmentation results
                this.currentMasks = data.masks;
                
                await this.loadImage(data.overlay_image);
                this.redrawBoundingBoxes();  // Draw all segmented masks
                
                console.log(`✅ Segmentation complete: ${data.masks_count} masks`);
                this.showMessage(data.message);
            }
        } catch (error) {
            console.error('Segmentation error:', error);
        }
    }
    
    // 3️⃣ Apply Intensity Filter
    async applyIntensityFilter() {
        const minIntensity = document.getElementById('minIntensity').value;
        const maxIntensity = document.getElementById('maxIntensity').value;
        
        try {
            const response = await fetch('/apply_intensity_filter', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    min_intensity: parseInt(minIntensity),
                    max_intensity: parseInt(maxIntensity)
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // ✅ CRITICAL: Update mask list with FILTERED results
                this.currentMasks = data.masks;  // Only masks inside range!
                
                await this.loadImage(data.image);
                this.redrawBoundingBoxes();  // Draw only filtered masks
                
                console.log(`✅ Filter applied: ${data.masks_count} masks kept, ` +
                           `${data.filter_results.filtered_count} filtered out`);
                this.showMessage(data.message);
            }
        } catch (error) {
            console.error('Filter error:', error);
        }
    }
    
    // 4️⃣ Reset Intensity Filter
    async resetIntensityFilter() {
        try {
            const response = await fetch('/reset_intensity_filter', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            const data = await response.json();
            
            if (data.success) {
                // ✅ Update mask list with ALL masks restored
                this.currentMasks = data.masks;
                
                await this.loadImage(data.image);
                this.redrawBoundingBoxes();  // Draw all masks again
                
                console.log(`✅ Filter reset: ${data.masks_count} masks restored`);
                this.showMessage(data.message);
            }
        } catch (error) {
            console.error('Reset error:', error);
        }
    }
    
    // 🎨 Core Drawing Function
    redrawBoundingBoxes() {
        // Step 1: Clear ALL existing bounding boxes
        this.clearAllBoundingBoxes();
        
        // Step 2: Draw ONLY the masks in currentMasks array
        console.log(`Drawing ${this.currentMasks.length} bounding boxes`);
        
        this.currentMasks.forEach(mask => {
            if (mask.bounding_box) {
                const [x, y, w, h] = mask.bounding_box;
                this.drawBoundingBox(x, y, w, h, mask.mask_id);
            }
        });
    }
    
    clearAllBoundingBoxes() {
        this.boxCtx.clearRect(0, 0, this.boxCanvas.width, this.boxCanvas.height);
    }
    
    drawBoundingBox(x, y, w, h, maskId) {
        this.boxCtx.strokeStyle = '#00ff00';  // Green
        this.boxCtx.lineWidth = 2;
        this.boxCtx.strokeRect(x, y, w, h);
        
        // Optional: Draw mask ID
        this.boxCtx.fillStyle = '#00ff00';
        this.boxCtx.font = '12px Arial';
        this.boxCtx.fillText(`#${maskId}`, x + 2, y + 12);
    }
    
    // 🖱️ Hover Preview (separate from main canvas boxes)
    async handleMouseMove(event) {
        const rect = this.boxCanvas.getBoundingClientRect();
        const x = Math.floor(event.clientX - rect.left);
        const y = Math.floor(event.clientY - rect.top);
        
        try {
            const response = await fetch('/get_mask_preview', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({x, y})
            });
            
            const data = await response.json();
            
            if (data.success && data.has_mask) {
                this.showMaskPreview(data.preview_image, data.mask_info);
            } else {
                this.hideMaskPreview();
            }
        } catch (error) {
            // Silently fail for hover
        }
    }
    
    showMaskPreview(previewImage, maskInfo) {
        const infoDiv = document.getElementById('maskInfo');
        infoDiv.innerHTML = `
            <img src="${previewImage}" width="200" height="200">
            <div>
                <strong>Mask #${maskInfo.mask_id}</strong><br>
                Diameter: ${maskInfo.diameter.toFixed(2)}px<br>
                Intensity: ${maskInfo.mean_intensity.toFixed(2)}<br>
                Area: ${maskInfo.area.toFixed(0)}px²<br>
                State: ${maskInfo.state}
            </div>
        `;
        infoDiv.style.display = 'block';
    }
    
    hideMaskPreview() {
        const infoDiv = document.getElementById('maskInfo');
        infoDiv.style.display = 'none';
    }
    
    async loadImage(base64Image) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.imageCanvas.width = img.width;
                this.imageCanvas.height = img.height;
                this.boxCanvas.width = img.width;
                this.boxCanvas.height = img.height;
                
                this.imageCtx.drawImage(img, 0, 0);
                this.currentImage = img;
                resolve();
            };
            img.onerror = reject;
            img.src = base64Image;
        });
    }
    
    showMessage(message) {
        // Show message to user
        console.log(message);
        // You could also update a status div, show a toast, etc.
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    const samInterface = new SAMWebInterface();
    console.log('✅ SAM Web Interface initialized');
});
```

---

## Key Points for Frontend Developer:

### ✅ DO:
1. **Always update `currentMasks`** when backend returns `masks` array
2. **Always call `redrawBoundingBoxes()`** after updating `currentMasks`
3. **Trust the backend** - it sends exactly what should be displayed
4. Use **separate canvas layers** - one for image, one for boxes (easier to clear/redraw)

### ❌ DON'T:
1. **Don't cache the original mask list** and try to filter it on frontend
2. **Don't add your own filtering logic** - backend handles all state management
3. **Don't forget to update mask list** when filter response comes back
4. **Don't draw boxes conditionally** based on old state - always use `currentMasks`

---

## Debugging Checklist

If bounding boxes aren't updating after intensity filter:

```javascript
// Add this debug logging in your applyIntensityFilter function:
async applyIntensityFilter() {
    const response = await fetch('/apply_intensity_filter', {...});
    const data = await response.json();
    
    console.log('=== DEBUG: Intensity Filter Response ===');
    console.log('Success:', data.success);
    console.log('Masks in response:', data.masks ? data.masks.length : 'undefined');
    console.log('First mask:', data.masks ? data.masks[0] : 'none');
    console.log('Current masks BEFORE update:', this.currentMasks.length);
    
    if (data.success && data.masks) {
        this.currentMasks = data.masks;  // Update here
        console.log('Current masks AFTER update:', this.currentMasks.length);
        
        this.redrawBoundingBoxes();
        console.log('Bounding boxes redrawn');
    }
}
```

Expected output after applying filter (100-200 intensity):
```
=== DEBUG: Intensity Filter Response ===
Success: true
Masks in response: 20
First mask: {mask_id: 0, bounding_box: [100, 200, 50, 50], ...}
Current masks BEFORE update: 45
Current masks AFTER update: 20
Bounding boxes redrawn
```

---

## Summary

**The Problem**: Frontend stores mask list once after segmentation and never updates it.

**The Solution**: Update `currentMasks` every time backend returns a `masks` array:
- Upload → `currentMasks = []` (clear all)
- Segmentation → `currentMasks = response.masks` (all masks)
- Apply filter → `currentMasks = response.masks` (filtered subset)
- Reset filter → `currentMasks = response.masks` (all masks back)

Then always redraw from `currentMasks` - **it's the single source of truth**.

