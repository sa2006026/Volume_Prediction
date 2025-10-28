# Debug Guide: Bounding Boxes Not Updating After Intensity Filter

## Problem
After applying intensity filter, the preview window still shows **ALL** bounding boxes instead of only showing boxes for masks inside the intensity range.

---

## Step-by-Step Debugging

### 1️⃣ Check Backend Response

Open browser DevTools (F12) → Network tab → apply intensity filter

**Look for the `/apply_intensity_filter` request:**

✅ **Correct response should look like:**
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "masks": [
    {"mask_id": 0, "bounding_box": [100, 200, 50, 50], "state": "active", ...},
    {"mask_id": 5, "bounding_box": [300, 400, 45, 45], "state": "active", ...},
    ...
  ],
  "masks_count": 20,
  "filter_results": {
    "kept_count": 20,
    "filtered_count": 25
  }
}
```

**Key things to verify:**
- ✅ `masks` array exists
- ✅ `masks_count` shows fewer masks than before (e.g., 20 instead of 45)
- ✅ All masks in array have `state: "active"`
- ✅ `filter_results` shows some masks were filtered out

---

### 2️⃣ Check Frontend Mask Update

Add console logging to your frontend code:

```javascript
async function applyIntensityFilter() {
    console.log('🔍 Before filter - current masks:', currentMasks.length);
    
    const response = await fetch('/apply_intensity_filter', {...});
    const data = await response.json();
    
    console.log('📥 Filter response:', data);
    console.log('📥 Masks in response:', data.masks ? data.masks.length : 'MISSING!');
    
    if (data.success && data.masks) {
        console.log('⚠️ About to update currentMasks...');
        currentMasks = data.masks;  // THIS LINE IS CRITICAL
        console.log('✅ After update - current masks:', currentMasks.length);
        
        redrawBoundingBoxes();
        console.log('✅ Bounding boxes redrawn');
    } else {
        console.error('❌ No masks in response!');
    }
}
```

**Expected console output:**
```
🔍 Before filter - current masks: 45
📥 Filter response: {success: true, masks: Array(20), ...}
📥 Masks in response: 20
⚠️ About to update currentMasks...
✅ After update - current masks: 20
✅ Bounding boxes redrawn
```

**❌ If you see this, you have a problem:**
```
🔍 Before filter - current masks: 45
📥 Filter response: {success: true, masks: Array(20), ...}
📥 Masks in response: 20
✅ After update - current masks: 45  ← PROBLEM: Not updated!
```

---

### 3️⃣ Check if currentMasks is Being Updated

**Common mistake #1: Not assigning response to currentMasks**
```javascript
// ❌ WRONG - forgot to update
async function applyIntensityFilter() {
    const response = await fetch('/apply_intensity_filter', {...});
    const data = await response.json();
    
    // Missing: currentMasks = data.masks;
    redrawBoundingBoxes();  // Still using old mask list!
}
```

**Common mistake #2: Using a different variable**
```javascript
// ❌ WRONG - updating wrong variable
let allMasks = [];  // Original masks from segmentation
let filteredMasks = [];  // Filtered masks

async function applyIntensityFilter() {
    const data = await fetch(...).then(r => r.json());
    filteredMasks = data.masks;  // Update filteredMasks
    
    redrawBoundingBoxes(allMasks);  // ❌ Still drawing from allMasks!
}
```

**Common mistake #3: Variable scope issue**
```javascript
// ❌ WRONG - shadowing the outer variable
let currentMasks = [];

async function applyIntensityFilter() {
    const data = await fetch(...).then(r => r.json());
    let currentMasks = data.masks;  // ❌ Creates NEW local variable!
    // Outer currentMasks is NOT updated
}
```

**✅ CORRECT:**
```javascript
let currentMasks = [];  // Single source of truth

async function applyIntensityFilter() {
    const data = await fetch(...).then(r => r.json());
    currentMasks = data.masks;  // ✅ Update the outer variable
    redrawBoundingBoxes();  // ✅ Use updated list
}
```

---

### 4️⃣ Check Drawing Function

Verify your `redrawBoundingBoxes()` function:

```javascript
function redrawBoundingBoxes() {
    console.log('🎨 Drawing bounding boxes...');
    console.log('🎨 Drawing from currentMasks:', currentMasks.length, 'masks');
    
    // Step 1: Clear
    clearAllBoundingBoxes();
    console.log('✅ Cleared all boxes');
    
    // Step 2: Draw
    let drawnCount = 0;
    currentMasks.forEach(mask => {
        if (mask.bounding_box) {
            const [x, y, w, h] = mask.bounding_box;
            drawBoundingBox(x, y, w, h);
            drawnCount++;
        }
    });
    
    console.log(`✅ Drew ${drawnCount} bounding boxes`);
}
```

**Expected output after intensity filter (keeping 20 masks):**
```
🎨 Drawing bounding boxes...
🎨 Drawing from currentMasks: 20 masks
✅ Cleared all boxes
✅ Drew 20 bounding boxes
```

**❌ If you see this:**
```
🎨 Drawing bounding boxes...
🎨 Drawing from currentMasks: 45 masks  ← PROBLEM: Old mask count!
✅ Cleared all boxes
✅ Drew 45 bounding boxes
```
→ Your `currentMasks` variable was NOT updated. Go back to step 3.

---

### 5️⃣ Use Debug Endpoint

I've added a debug endpoint to check backend state. Open this URL in your browser:

```
http://localhost:5014/debug_mask_states
```

**Expected response:**
```json
{
  "success": true,
  "total_masks": 45,
  "state_counts": {
    "active": 20,
    "intensity_filtered": 25
  },
  "active_count": 20,
  "intensity_filtered_count": 25,
  "mask_details": [
    {"mask_id": 0, "state": "active", "mean_intensity": 150, ...},
    {"mask_id": 1, "state": "intensity_filtered", "mean_intensity": 50, ...},
    ...
  ]
}
```

**This shows:**
- Total 45 masks exist
- 20 are `active` (inside intensity range)
- 25 are `intensity_filtered` (outside range)
- The `/apply_intensity_filter` endpoint should return only the 20 active ones

---

### 6️⃣ Check if You're Using Multiple Mask Arrays

**Problem:** Some codebases maintain multiple mask arrays:
```javascript
// ❌ WRONG - Multiple sources of truth
let originalMasks = [];  // From segmentation
let filteredMasks = [];  // After intensity filter
let visibleMasks = [];   // After user toggles
```

**This causes confusion** - which array should be used for drawing?

**✅ CORRECT - Single source of truth:**
```javascript
// ✅ RIGHT - One array that gets updated
let currentMasks = [];  // THE ONLY mask list

// Upload: currentMasks = []
// Segmentation: currentMasks = response.masks
// Filter: currentMasks = response.masks
// Reset: currentMasks = response.masks
// Toggle: fetch all masks, then currentMasks = response.masks
```

---

### 7️⃣ Complete Test Sequence

Run this test sequence and verify console output at each step:

```javascript
// Test 1: Upload image
await uploadImage(file);
console.assert(currentMasks.length === 0, '❌ Upload should clear masks');

// Test 2: Run segmentation
await runSegmentation();
console.assert(currentMasks.length > 0, '❌ Segmentation should create masks');
const segmentedCount = currentMasks.length;
console.log(`✅ Segmented ${segmentedCount} masks`);

// Test 3: Apply intensity filter
await applyIntensityFilter(100, 200);
console.assert(currentMasks.length < segmentedCount, 
    '❌ Filter should reduce mask count');
const filteredCount = currentMasks.length;
console.log(`✅ Filtered to ${filteredCount} masks`);

// Test 4: Reset filter
await resetIntensityFilter();
console.assert(currentMasks.length === segmentedCount, 
    '❌ Reset should restore all masks');
console.log(`✅ Reset to ${currentMasks.length} masks`);
```

**Expected output:**
```
✅ Segmented 45 masks
✅ Filtered to 20 masks
✅ Reset to 45 masks
```

---

## Quick Fix Checklist

If bounding boxes aren't updating after intensity filter, check these:

- [ ] Backend returns `masks` array in response ✅
- [ ] Backend `masks` array has fewer items after filter ✅
- [ ] Frontend receives the response successfully ✅
- [ ] Frontend updates `currentMasks = response.masks` ✅
- [ ] `redrawBoundingBoxes()` is called after update ✅
- [ ] `redrawBoundingBoxes()` clears all boxes first ✅
- [ ] `redrawBoundingBoxes()` uses `currentMasks` (not old array) ✅
- [ ] No variable shadowing (e.g., `let currentMasks = ...` in function) ✅
- [ ] No multiple mask arrays causing confusion ✅

---

## Most Likely Issue

Based on your description, **the most likely issue is**:

```javascript
// Frontend is NOT updating its mask list when filter response arrives

// ❌ PROBLEM CODE:
let allMasks = [];  // Set once during segmentation

function runSegmentation() {
    allMasks = response.masks;  // Set here
    drawBoxes(allMasks);
}

function applyFilter() {
    // Filter is applied on backend
    // But allMasks is NEVER updated!
    drawBoxes(allMasks);  // Still has 45 masks from segmentation
}

// ✅ FIX:
let currentMasks = [];

function runSegmentation() {
    currentMasks = response.masks;  // Update
    drawBoxes(currentMasks);
}

function applyFilter() {
    currentMasks = response.masks;  // UPDATE HERE!
    drawBoxes(currentMasks);  // Now has 20 filtered masks
}
```

---

## Summary

**The Fix:** Make sure your frontend JavaScript has this line in the intensity filter handler:

```javascript
async function applyIntensityFilter(min, max) {
    const response = await fetch('/apply_intensity_filter', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({min_intensity: min, max_intensity: max})
    });
    
    const data = await response.json();
    
    if (data.success && data.masks) {
        // 🔥 THIS LINE IS CRITICAL 🔥
        currentMasks = data.masks;  // Update mask list from backend!
        
        redrawBoundingBoxes();  // Redraw using updated list
    }
}
```

Without this line, your frontend will keep using the old mask list from segmentation!

