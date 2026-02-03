# Unit Conversion Bug Fix

## Problem Description
When using Unit Conversion for diameter export, the system was calculating the intensity threshold (median) using **ALL masks**, including those that were filtered out by the intensity filter. This caused incorrect grouping of masks into high/low intensity groups.

## Root Cause

In `sam_analyzer.py`, two functions had a bug:
- `get_diameter_data_by_group_with_units()` (line 1097)
- `get_diameter_data_by_group()` (line 1146)

### The Bug:
```python
# ❌ WRONG: Used ALL masks for threshold calculation
intensities = [stats.get('mean_intensity', 0) for stats in self.mask_statistics]
intensity_threshold = np.median(intensities)  # Includes filtered masks!
```

This meant:
1. If you had 45 total masks with 25 filtered out by intensity filter
2. The median calculation used all 45 masks (including the 25 filtered ones)
3. This skewed the threshold, causing incorrect high/low grouping
4. Even though only 20 active masks were exported, they were grouped incorrectly

### Example of the Problem:

**Scenario:**
- Total masks: 45
- After intensity filter (100-200): 20 active masks
- Filtered out: 25 masks (outside 100-200 range)

**Old (buggy) behavior:**
```python
# Median calculated from ALL 45 masks (including filtered ones)
all_intensities = [50, 75, 90, 110, 120, 130, ..., 250, 255]  # 45 values
median = 140  # Skewed by filtered masks

# When grouping the 20 active masks:
# Some masks with intensity 110 would be in low group (< 140)
# Even though among the 20 active masks, 110 might be median or higher
```

**New (fixed) behavior:**
```python
# Median calculated from ONLY 20 active masks
active_intensities = [110, 120, 130, 135, 140, 145, ...]  # 20 values
median = 132.5  # Accurate for active masks only

# When grouping the 20 active masks:
# Mask with intensity 110 correctly goes to low group
# Mask with intensity 145 correctly goes to high group
# Grouping is now correct relative to the active masks
```

## Solution

### Fixed Code:
```python
# ✅ CORRECT: Calculate threshold from active masks only
active_intensities = []
for i, stats in enumerate(self.mask_statistics):
    mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
    if mask_state == 'active':  # Only include active masks
        active_intensities.append(stats.get('mean_intensity', 0))

if not active_intensities:
    return {'high_intensity': [], 'low_intensity': [], 'unit_name': self.unit_name}

intensity_threshold = np.median(active_intensities)  # Correct median!
```

## Changes Made

### File: `src/core/sam_analyzer.py`

#### 1. Function: `get_diameter_data_by_group_with_units()` (lines 1097-1144)
**Before:**
- Line 1108: `intensities = [stats.get('mean_intensity', 0) for stats in self.mask_statistics]`
- Used all masks for median calculation

**After:**
- Lines 1107-1112: Filter to get only active masks' intensities
- Lines 1114-1115: Calculate median from active masks only

#### 2. Function: `get_diameter_data_by_group()` (lines 1146-1188)
**Before:**
- Line 1152: `intensities = [stats.get('mean_intensity', 0) for stats in self.mask_statistics]`
- Used all masks for median calculation

**After:**
- Lines 1156-1161: Filter to get only active masks' intensities
- Lines 1163-1164: Calculate median from active masks only

## Impact

### ✅ What's Fixed:
1. **Intensity threshold is now correct**: Calculated from active masks only
2. **High/Low grouping is accurate**: Masks are grouped relative to active masks' median
3. **Unit conversion respects filtering**: Exported data only includes active masks with correct grouping
4. **Consistency**: Both unit and non-unit export functions now behave identically

### ✅ What Still Works:
1. Masks are still filtered by state (only 'active' masks exported) ✓
2. Unit conversion still works correctly ✓
3. CSV export still filters by state ✓
4. Diameter export still excludes removed/filtered masks ✓

## Testing

### Test Scenario 1: No Filtering
```
1. Upload image
2. Run segmentation (45 masks)
3. Export with unit conversion
   Expected: All 45 masks exported, grouped by their median intensity
```

### Test Scenario 2: With Intensity Filter
```
1. Upload image
2. Run segmentation (45 masks)
3. Apply intensity filter: 100-200 (keeps 20 masks)
4. Export with unit conversion
   Expected: Only 20 masks exported, grouped by median of those 20 masks
```

### Test Scenario 3: Verify Grouping
```
1. Run segmentation
2. Apply intensity filter
3. Export diameter data
4. Check the high/low groups:
   - High group should have intensities >= median of active masks
   - Low group should have intensities < median of active masks
   - NO filtered masks should appear in either group
```

## How to Verify the Fix

### Check Exported Data:
1. Run segmentation
2. Note total masks (e.g., 45)
3. Apply intensity filter (e.g., 100-200)
4. Note active masks (e.g., 20)
5. Export diameter data
6. Verify:
   ```
   - Total exported = 20 (not 45) ✓
   - High group + Low group = 20 ✓
   - No masks outside 100-200 intensity range ✓
   ```

### Check Threshold Calculation:
Add debug logging to verify:
```python
print(f"Total masks: {len(self.mask_statistics)}")
print(f"Active masks: {len(active_intensities)}")
print(f"Intensity threshold (median): {intensity_threshold}")
print(f"High group: {len(high_intensity_diameters)}")
print(f"Low group: {len(low_intensity_diameters)}")
```

Expected output after intensity filter:
```
Total masks: 45
Active masks: 20
Intensity threshold (median): 132.5
High group: 10
Low group: 10
```

## Summary

**The Bug**: Intensity threshold was calculated using ALL masks (including filtered ones), causing incorrect high/low grouping even though only active masks were exported.

**The Fix**: Calculate the intensity threshold (median) using ONLY active masks, ensuring correct grouping relative to the masks being exported.

**Files Changed**: `src/core/sam_analyzer.py`
- `get_diameter_data_by_group_with_units()` - Fixed
- `get_diameter_data_by_group()` - Fixed

**Result**: Unit conversion now correctly exports only active masks with accurate high/low intensity grouping based on the active masks' median intensity.

