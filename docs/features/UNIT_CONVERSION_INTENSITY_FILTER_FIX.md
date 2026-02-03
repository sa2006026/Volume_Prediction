# Unit Conversion + Intensity Filter Bug Fix

## Problem Description

When **unit conversion is enabled** and then **intensity filter is applied**, the frontend shows **ALL masks** instead of only the masks inside the intensity range. However, without unit conversion enabled, the intensity filter works correctly.

## Root Cause

The `/apply_intensity_filter` and `/reset_intensity_filter` endpoints were **not applying unit conversion** to the filtered masks before returning them to the frontend. This caused an inconsistency:

### Without Unit Conversion (Working):
```
1. Apply intensity filter
2. Backend returns filtered masks in pixels
3. Frontend receives and displays filtered masks ✓
```

### With Unit Conversion (Broken):
```
1. Enable unit conversion
2. Apply intensity filter
3. Backend returns filtered masks WITHOUT unit conversion
4. Frontend might call /get_all_masks to get unit-converted data
5. /get_all_masks returns ALL masks (including filtered ones) with unit conversion
6. Frontend displays all masks ❌
```

## The Bug

### `/apply_intensity_filter` endpoint:
**Missing**: Unit conversion was not applied to filtered masks before returning

```python
# ❌ BEFORE: No unit conversion applied
for i, (mask_stats, mask_state) in enumerate(...):
    if mask_state == 'active':
        mask_info = mask_stats.copy()
        mask_info['state'] = mask_state
        filtered_masks.append(mask_info)  # Returns pixel values only

return jsonify({
    'masks': filtered_masks  # Missing unit conversion!
})
```

### `/reset_intensity_filter` endpoint:
**Missing**: Same issue - no unit conversion when returning all masks

## The Fix

### File: `sam_website.py`

#### 1. `/apply_intensity_filter` endpoint (lines 1100-1110)

**Added unit conversion block:**
```python
# Add unit conversion information if enabled
if engine.sam_analyzer and engine.sam_analyzer.conversion_enabled:
    converted_masks = []
    for mask in filtered_masks:
        mask_id = mask.get('mask_id', -1)
        if mask_id >= 0:
            converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(mask_id)
            if converted_stats:
                mask.update(converted_stats)
        converted_masks.append(mask)
    filtered_masks = converted_masks
```

#### 2. `/reset_intensity_filter` endpoint (lines 1160-1170)

**Added same unit conversion block:**
```python
# Add unit conversion information if enabled
if engine.sam_analyzer and engine.sam_analyzer.conversion_enabled:
    converted_masks = []
    for mask in all_masks:
        mask_id = mask.get('mask_id', -1)
        if mask_id >= 0:
            converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(mask_id)
            if converted_stats:
                mask.update(converted_stats)
        converted_masks.append(mask)
    all_masks = converted_masks
```

#### 3. Added Debug Logging

Added logging to both endpoints:
```python
print(f"📦 Unit conversion enabled: {engine.sam_analyzer.conversion_enabled if engine.sam_analyzer else False}")
```

## How It Works Now

### With Unit Conversion Enabled:

```
1. Enable unit conversion (e.g., 100 pixels = 50 μm)
2. Run segmentation → 45 masks returned with unit conversion
3. Apply intensity filter (100-200) → 20 active masks
4. Backend:
   - Filters masks by state (20 active)
   - Applies unit conversion to these 20 masks ✅
   - Returns 20 masks with converted units ✅
5. Frontend:
   - Receives 20 masks with unit conversion ✅
   - Clears all bounding boxes ✅
   - Draws 20 bounding boxes ✅
```

## Consistency Across Endpoints

Now all endpoints that return mask data apply unit conversion consistently:

| Endpoint | Returns | Unit Conversion Applied? |
|----------|---------|--------------------------|
| `/run_sam_segmentation` | All segmented masks | ✅ (if enabled) |
| `/get_all_masks` | All masks with states | ✅ (if enabled) |
| `/apply_intensity_filter` | Active masks only | ✅ **NOW FIXED** |
| `/reset_intensity_filter` | All masks restored | ✅ **NOW FIXED** |
| `/get_mask_info` | Single mask at point | ✅ (if enabled) |
| `/get_mask_preview` | Single mask preview | ✅ (if enabled) |
| `/export_mask_csv` | Active masks only | ✅ (if enabled) |

## Testing

### Test Scenario 1: Unit Conversion + Intensity Filter
```
1. Upload image
2. Run segmentation (45 masks)
3. Enable unit conversion (e.g., 100px = 50μm)
4. Apply intensity filter (100-200)
   
Expected Result:
- Backend returns ~20 masks (active only)
- All masks have converted diameter/area values
- Frontend shows 20 bounding boxes
- Server log shows:
  📦 Returning 20 masks to frontend
  📦 Unit conversion enabled: True
```

### Test Scenario 2: Verify Frontend Receives Correct Data
```
1. Enable unit conversion
2. Apply intensity filter
3. Open browser DevTools → Network tab
4. Check /apply_intensity_filter response:
   
Expected Response:
{
  "success": true,
  "masks": [
    {
      "mask_id": 0,
      "diameter": 25.5,  // In μm, not pixels ✓
      "area": 510.7,     // In μm², not pixels² ✓
      "diameter_pixels": 51,  // Also has pixel values
      "state": "active"
    },
    ...
  ],
  "masks_count": 20,
  "total_masks": 45
}
```

### Test Scenario 3: Reset Filter with Unit Conversion
```
1. Have unit conversion enabled and intensity filter applied
2. Click "Reset Filter"
3. Check response

Expected Result:
- Backend returns all 45 masks
- All masks have converted values
- Frontend shows all 45 bounding boxes
- Server log shows:
  📦 Returning 45 masks to frontend
  📦 Unit conversion enabled: True
```

## Debugging

### Check Server Logs

After applying intensity filter, you should see:
```
🔍 Intensity filter applied: 100-200
📊 Total masks: 45, Active: 20, Filtered: 25
📦 Returning 20 masks to frontend
📦 First mask has bounding_box: True
📦 Unit conversion enabled: True  ← Confirms unit conversion is active
```

### Check Browser Network Tab

Look at the `/apply_intensity_filter` response:
- `masks` array should have 20 items (not 45)
- Each mask should have both pixel and unit values if conversion is enabled:
  - `diameter` (in units)
  - `diameter_pixels` (in pixels)
  - `area` (in units²)
  - `area_pixels` (in pixels²)

### Check Frontend State

In browser console:
```javascript
console.log('Current masks:', currentMasks.length);  // Should be 20
console.log('First mask:', currentMasks[0]);
// Should show unit-converted values
```

## What's Fixed

✅ **Consistency**: All endpoints now apply unit conversion when enabled  
✅ **Intensity filter**: Now returns unit-converted values for filtered masks  
✅ **Reset filter**: Now returns unit-converted values for all masks  
✅ **Frontend display**: Receives correct data format regardless of unit conversion state  
✅ **Debugging**: Added logging to track unit conversion status  

## Summary

**The Bug**: `/apply_intensity_filter` and `/reset_intensity_filter` were not applying unit conversion to masks before returning them, causing inconsistency with other endpoints and confusing the frontend.

**The Fix**: Added unit conversion logic (identical to `/get_all_masks`) to both endpoints, ensuring all mask data returned to the frontend has consistent formatting.

**Result**: When unit conversion is enabled and intensity filter is applied, the frontend correctly receives and displays only the filtered masks with their unit-converted values.

