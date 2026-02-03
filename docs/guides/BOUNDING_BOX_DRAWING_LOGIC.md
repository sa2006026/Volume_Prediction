# Bounding Box Drawing Logic - Complete Flow

## Overview
The frontend implements a **"clear-then-draw"** approach for bounding boxes:
1. Clear all existing bounding boxes first
2. Draw new bounding boxes based on the mask list returned by the backend

## Backend Response Format
All endpoints that affect bounding boxes return a consistent format:
```json
{
  "success": true,
  "image": "base64_image_data",
  "masks": [...],           // Array of mask objects with bounding_box data
  "masks_count": N          // Number of masks in the array
}
```

---

## Complete Flow & Expected Behavior

### 1️⃣ Image Upload (`/upload_image`)
**What happens:**
- User uploads a new image
- Backend loads the image and resets all mask data

**Backend returns:**
```json
{
  "success": true,
  "image": "base64_image",
  "masks": [],              // ✅ Empty array
  "masks_count": 0,
  "dimensions": {...}
}
```

**Frontend action:**
```javascript
// On upload success:
clearAllBoundingBoxes();  // Clear any existing boxes
// Don't draw anything (masks array is empty)
displayImage(response.image);
```

---

### 2️⃣ Run Segmentation (`/run_sam_segmentation`)
**What happens:**
- Backend runs SAM segmentation
- Applies overlap filter if enabled
- Returns only masks that passed all filters (state = 'active')

**Backend returns:**
```json
{
  "success": true,
  "overlay_image": "base64_image",
  "masks": [                // ✅ Array of active masks after filtering
    {
      "mask_id": 0,
      "bounding_box": [x, y, w, h],
      "center_x": 100,
      "center_y": 200,
      "diameter": 50,
      "area": 1963,
      "state": "active",
      ...
    },
    ...
  ],
  "masks_count": 45,        // Number of visible masks
  "total_masks": 50         // Total before overlap filtering
}
```

**Frontend action:**
```javascript
// On segmentation success:
clearAllBoundingBoxes();              // Clear all boxes first
displayImage(response.overlay_image);
response.masks.forEach(mask => {
  drawBoundingBox(mask.bounding_box);  // Draw box for each active mask
});
```

---

### 3️⃣ Apply Intensity Filter (`/apply_intensity_filter`)
**What happens:**
- Backend filters masks by intensity range [min_intensity, max_intensity]
- Masks outside range: state changed to `'intensity_filtered'`
- Masks inside range: state remains `'active'`
- Returns only the masks with state = `'active'`

**Backend returns:**
```json
{
  "success": true,
  "image": "base64_image",
  "masks": [                // ✅ Only masks inside intensity range
    {
      "mask_id": 5,
      "bounding_box": [x, y, w, h],
      "mean_intensity": 150,
      "state": "active",
      ...
    },
    ...
  ],
  "masks_count": 20,        // Number of masks inside range
  "filter_results": {
    "kept_count": 20,
    "filtered_count": 25
  },
  "min_intensity": 100,
  "max_intensity": 200
}
```

**Frontend action:**
```javascript
// On intensity filter success:
clearAllBoundingBoxes();              // ✅ Clear ALL existing boxes first
displayImage(response.image);
response.masks.forEach(mask => {
  drawBoundingBox(mask.bounding_box);  // Draw ONLY boxes for masks inside intensity range
});
showFilterInfo(`Kept: ${response.filter_results.kept_count}, Filtered: ${response.filter_results.filtered_count}`);
```

---

### 4️⃣ Reset Intensity Filter (`/reset_intensity_filter`)
**What happens:**
- Backend resets all mask states from `'intensity_filtered'` back to `'active'`
- Returns all masks (they are all active now)

**Backend returns:**
```json
{
  "success": true,
  "image": "base64_image",
  "masks": [                // ✅ All masks (filter removed)
    {
      "mask_id": 0,
      "state": "active",
      "bounding_box": [x, y, w, h],
      ...
    },
    ...
  ],
  "masks_count": 45         // All masks restored
}
```

**Frontend action:**
```javascript
// On reset filter success:
clearAllBoundingBoxes();              // Clear current boxes
displayImage(response.image);
response.masks.forEach(mask => {
  drawBoundingBox(mask.bounding_box);  // Draw boxes for ALL masks again
});
```

---

### 5️⃣ Toggle Individual Mask (`/toggle_mask`)
**What happens:**
- User clicks on a mask to toggle it between 'active' and 'removed'
- Backend updates that specific mask's state
- Returns updated overlay

**Backend returns:**
```json
{
  "success": true,
  "mask_toggled": true,
  "toggle_info": {
    "mask_id": 10,
    "old_state": "active",
    "new_state": "removed"
  },
  "overlay_image": "base64_image"
}
```

**Frontend action:**
```javascript
// On toggle success:
// Option A: Get fresh mask list
fetchAllMasks().then(allMasks => {
  clearAllBoundingBoxes();
  allMasks.filter(m => m.state === 'active').forEach(mask => {
    drawBoundingBox(mask.bounding_box);
  });
});

// Option B: Toggle specific box if you track them
toggleBoundingBoxVisibility(response.toggle_info.mask_id, response.toggle_info.new_state);
```

---

### 6️⃣ Get All Masks (`/get_all_masks`)
**What happens:**
- Returns all masks with their current states
- Useful for refreshing the display or syncing state

**Backend returns:**
```json
{
  "success": true,
  "masks": [                // All masks with their states
    {
      "mask_id": 0,
      "state": "active",
      "bounding_box": [x, y, w, h],
      ...
    },
    {
      "mask_id": 1,
      "state": "removed",     // User toggled off
      "bounding_box": [x, y, w, h],
      ...
    },
    {
      "mask_id": 2,
      "state": "intensity_filtered",  // Filtered out
      "bounding_box": [x, y, w, h],
      ...
    },
    ...
  ],
  "masks_count": 45,
  "summary": {...}
}
```

**Frontend action:**
```javascript
// To refresh display:
clearAllBoundingBoxes();
response.masks.forEach(mask => {
  if (mask.state === 'active') {  // Only draw active masks
    drawBoundingBox(mask.bounding_box);
  }
});
```

---

## Mask States

| State | Description | Show Bounding Box? |
|-------|-------------|-------------------|
| `active` | Mask is visible and included in analysis | ✅ YES |
| `removed` | User manually toggled off | ❌ NO |
| `intensity_filtered` | Filtered out by intensity range | ❌ NO |
| `overlap_filtered` | Filtered out by overlap detection | ❌ NO |

---

## Frontend Implementation Pattern

### Recommended Structure:

```javascript
class BoundingBoxManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.boxes = [];  // Current bounding boxes
  }
  
  // Core method: clear and redraw
  clearAll() {
    this.boxes = [];
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    // Redraw base image
  }
  
  drawFromMaskList(masks) {
    this.clearAll();  // Always clear first
    masks.forEach(mask => {
      if (mask.bounding_box) {
        const [x, y, w, h] = mask.bounding_box;
        this.ctx.strokeStyle = this.getColorForState(mask.state);
        this.ctx.strokeRect(x, y, w, h);
        this.boxes.push({id: mask.mask_id, rect: [x, y, w, h]});
      }
    });
  }
  
  getColorForState(state) {
    // Only active masks should be drawn, but just in case:
    return state === 'active' ? '#00ff00' : '#ff0000';
  }
}

// Usage in your application:
const boxManager = new BoundingBoxManager(canvas);

// 1. Upload image
uploadImage(file).then(response => {
  boxManager.drawFromMaskList(response.masks);  // Empty array = clears all
});

// 2. Run segmentation
runSegmentation(params).then(response => {
  boxManager.drawFromMaskList(response.masks);  // Draw all segmented masks
});

// 3. Apply intensity filter
applyIntensityFilter(min, max).then(response => {
  boxManager.drawFromMaskList(response.masks);  // Draw only filtered masks
});

// 4. Reset filter
resetIntensityFilter().then(response => {
  boxManager.drawFromMaskList(response.masks);  // Draw all masks again
});
```

---

## Key Points

1. ✅ **Always use `masks` array from backend** - it contains exactly what should be displayed
2. ✅ **Always clear before drawing** - prevents stale bounding boxes
3. ✅ **Don't filter on frontend** - backend already filters by state
4. ✅ **Consistent pattern** - every endpoint returns `masks` array
5. ✅ **Empty array = clear all** - upload returns `masks: []` to reset display

---

## Preview Window (Hover) - Separate Logic

The preview window is **independent** from the main canvas bounding boxes:

```javascript
canvas.addEventListener('mousemove', (e) => {
  const {x, y} = getCanvasCoordinates(e);
  
  getMaskPreview(x, y).then(response => {
    if (response.has_mask) {
      // Show preview popup with:
      // - response.preview_image (200x200 crop with red overlay)
      // - response.mask_info (stats, diameter, intensity, etc.)
      showPreview(response.preview_image, response.mask_info);
    } else {
      hidePreview();
    }
  });
});
```

**Note:** Preview endpoint (`/get_mask_preview`) automatically filters out masks with state `'intensity_filtered'` or `'overlap_filtered'`, so hover won't show filtered masks.

---

## Summary

Your "clear-then-draw" logic is now fully supported:
- ✅ Upload image: clears boxes (returns empty array)
- ✅ Segmentation: draws boxes for all segmented masks after filtering
- ✅ Intensity filter: clears and draws only masks inside range
- ✅ Reset filter: clears and draws all masks back
- ✅ All endpoints return `masks` array with exact masks to display

