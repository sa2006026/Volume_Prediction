# Overlap Filter Enhancement: User-Selectable Mask Removal Mode

## Overview
Added a new feature that allows users to choose whether to remove the **larger** or **smaller** mask when the overlap filter detects duplicate/overlapping masks.

## Changes Made

### Backend Changes (sam_website.py)

1. **Updated `apply_mask_overlap_filter()` method**
   - Added new parameter: `remove_mode: str = 'larger'`
   - Updated logic to remove either larger or smaller mask based on user choice
   - Options: `'larger'` (default) or `'smaller'`

2. **Updated `perform_sam_segmentation()` method**
   - Added new parameter: `overlap_remove_mode: str = 'larger'`
   - Passes the remove mode to `apply_mask_overlap_filter()`

3. **Updated Flask routes**
   - `/run_sam_segmentation`: Now accepts `overlap_remove_mode` parameter
   - `/apply_overlap_filter`: Now accepts `remove_mode` parameter and displays it in the success message

### Frontend Changes (sam_website.html)

1. **Main SAM Configuration Panel**
   - Added radio buttons to choose removal mode during initial segmentation
   - Radio button group name: `overlapRemoveMode`
   - Options: "Larger mask" (default) or "Smaller mask"
   - Located under the overlap filter threshold slider

2. **Overlap Filter Panel (Post-Segmentation)**
   - Added radio buttons to choose removal mode when re-applying the filter
   - Radio button group name: `overlapPanelRemoveMode`
   - Options: "Larger mask" (default) or "Smaller mask"
   - Allows users to experiment with different removal strategies

3. **JavaScript Updates**
   - `runSAMSegmentation()`: Reads the selected remove mode and sends it to backend
   - `applyOverlapFilter()`: Reads the selected remove mode from the overlap panel and sends it to backend

4. **Updated Info Text**
   - Changed description to reflect that users can now choose which mask to remove
   - Updated both the main configuration info and the overlap panel description

## Usage

### During Initial SAM Segmentation:
1. Check the "Overlap Filter" checkbox
2. Set the threshold (e.g., 0.80 = 80% overlap)
3. Choose removal mode:
   - **Larger mask**: Keeps smaller masks (useful for preserving fine details)
   - **Smaller mask**: Keeps larger masks (useful when larger masks are more accurate)
4. Run SAM Segmentation

### After Segmentation (Re-applying Filter):
1. Navigate to the Overlap Filter panel
2. Adjust the threshold if needed
3. Choose the removal mode (Larger or Smaller)
4. Click "Apply Overlap Filter"

## Use Cases

### Remove Larger Mask (Default)
- **When to use**: When you want to preserve smaller, more detailed segmentations
- **Example**: Detecting small droplets where SAM might create both a precise small mask and a larger, less accurate mask around the same object

### Remove Smaller Mask
- **When to use**: When larger masks are more accurate and smaller ones are noise
- **Example**: When SAM creates small spurious masks inside larger, correct segmentations

## Technical Details

### Algorithm
The overlap filter uses mask-based intersection calculation:
1. For each pair of active masks (i, j):
   - Calculate intersection area: `intersection = count_nonzero(mask_i & mask_j)`
   - Calculate overlap ratio: `ratio = intersection / min(area_i, area_j)`
2. If `ratio >= overlap_threshold`:
   - If `remove_mode == 'larger'`: Remove the mask with larger area
   - If `remove_mode == 'smaller'`: Remove the mask with smaller area
3. Mark removed masks as `'overlap_filtered'` state

### Default Behavior
- Default mode: `'larger'` (remove larger mask, keep smaller)
- This preserves the original behavior for backward compatibility
- Users can easily switch to `'smaller'` mode as needed

## Testing Recommendations

1. Test with images containing overlapping objects of different sizes
2. Try both removal modes and compare results
3. Verify that the success message correctly indicates which mode was used
4. Ensure the radio buttons persist their state correctly during the session

## Future Enhancements

Potential improvements:
- Add a "smart" mode that removes based on quality metrics (circularity, intensity, etc.)
- Add preview showing which masks would be removed before applying
- Add statistics showing size distribution of kept vs. removed masks

