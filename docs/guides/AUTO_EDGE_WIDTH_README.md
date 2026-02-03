# Automatic Edge Width Feature

## Overview
This feature automatically calculates the optimal edge width for dark edge detection to prevent the dark ring from including neighboring droplets.

## Changes Made

### Backend (`src/web/sam_website.py`)

1. **New Method: `calculate_optimal_edge_width()`**
   - Located after `extract_dark_edge_pixels()` method
   - Calculates the minimum distance from a mask's contour to the nearest neighboring mask
   - Uses OpenCV's distance transform for accurate distance calculation
   - Returns an optimal edge width that won't overlap with other droplets
   - Includes safety margin of 2 pixels
   - Constrained between min (5px) and max (100px) values

2. **Updated Method: `create_dark_edge_preview()`**
   - Added `auto_edge_width` parameter (default: False)
   - When enabled, automatically calculates optimal edge width before processing
   - Maintains backward compatibility with manual edge width

3. **Updated Method: `get_dark_edge_data_with_units()`**
   - Added `auto_edge_width` parameter (default: False)
   - Automatically calculates edge width when auto mode is enabled
   - Ensures consistency between preview and data extraction

4. **Updated Flask Route: `/get_mask_preview`**
   - Added support for `auto_edge_width` parameter from frontend
   - Passes the auto flag to `create_dark_edge_preview()` and `get_dark_edge_data_with_units()`
   - Returns `edge_width_used` in response so frontend can display the actual width used

### Frontend (`templates/sam_website.html`)

1. **New UI Control: Auto Edge Width Checkbox**
   - Located in the Dark Edge Preview Controls section
   - Checkbox with icon and description
   - When enabled, disables the manual edge width slider (visual feedback with opacity)

2. **Updated JavaScript: `initializeDarkEdgePreviewControls()`**
   - Added event handler for auto edge width toggle
   - Disables/enables manual slider based on auto mode
   - Provides console logging for debugging

3. **Updated JavaScript: `updateMaskPreview()`**
   - Reads the auto edge width checkbox state
   - Sends `auto_edge_width` parameter to backend
   - Displays the actual edge width used when auto mode is enabled

4. **New Preview Info Display**
   - Shows "Edge Width Used: X px (Auto)" in preview panel when auto mode is enabled
   - Highlighted with blue color to distinguish from manual settings
   - Only displayed when dark edges are enabled and auto mode is active

5. **Updated Tip Text**
   - Mentions the auto edge width feature
   - Explains the difference between auto and manual modes

## How It Works

### Algorithm
1. When auto edge width is enabled, the system:
   - Creates a combined mask of all OTHER droplets (excluding the current one)
   - Applies distance transform to calculate distance from each pixel to the nearest other droplet
   - Samples distances along the current mask's contour
   - Finds the minimum distance to any neighboring droplet
   - Subtracts a 2-pixel safety margin
   - Returns this as the optimal edge width

### Benefits
- **Prevents False Positives**: Dark ring won't accidentally include pixels from neighboring droplets
- **Adaptive**: Each droplet gets its own optimal edge width based on its surroundings
- **Safe**: Includes safety margin to ensure no overlap
- **Maintains Accuracy**: Dark edge measurements remain accurate and reliable

## Usage

### For Users
1. Enable "Show Dark Edge Preview" checkbox
2. Check the "Auto Edge Width" checkbox
3. Hover over any mask/droplet
4. The preview will show the dark edge with automatically calculated width
5. The preview info will display "Edge Width Used: X px (Auto)" showing the calculated width

### For Developers
```python
# Backend usage
optimal_width = engine.calculate_optimal_edge_width(mask_id=5)
preview = engine.create_dark_edge_preview(mask_id=5, auto_edge_width=True)
data = engine.get_dark_edge_data_with_units(mask_id=5, auto_edge_width=True)
```

```javascript
// Frontend usage
const response = await fetch('/get_mask_preview', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ 
        mask_id: mask_id,
        show_dark_edges: true,
        auto_edge_width: true,  // Enable auto mode
        darkness_threshold: 80
    })
});
```

## Testing

### Manual Testing Steps
1. Load an image with multiple droplets close together
2. Run SAM segmentation
3. Enable "Show Dark Edge Preview"
4. Test manual mode:
   - Set edge width to 50 pixels (manual slider)
   - Hover over a droplet with close neighbors
   - Observe if dark ring includes neighboring droplets
5. Test auto mode:
   - Enable "Auto Edge Width" checkbox
   - Hover over the same droplet
   - Observe that dark ring stops before neighboring droplets
   - Check the "Edge Width Used" value in preview info

### Expected Behavior
- **Isolated droplets**: Auto mode should use maximum edge width (100px) or less
- **Crowded droplets**: Auto mode should use smaller edge width to avoid neighbors
- **Manual mode**: Should work exactly as before (backward compatible)
- **Performance**: Should be fast due to caching (cache key includes edge_width)

## Technical Details

### Distance Transform
- Uses `cv2.distanceTransform()` with L2 (Euclidean) distance
- Mask size 5 for more accurate distance calculation
- Samples all contour points to find minimum distance

### Caching
- Cached results are keyed by `(mask_id, edge_width, darkness_threshold)`
- Auto mode calculates edge width first, then uses cache normally
- Cache is cleared when image changes or new segmentation is performed

### Backward Compatibility
- All existing functionality preserved
- Auto mode is opt-in (default: False)
- Manual edge width slider still works when auto mode is disabled
- Existing API calls without `auto_edge_width` parameter work unchanged

## Future Enhancements

Possible improvements:
1. Add auto edge width to CSV export ring width parameters
2. Add auto edge width to dark edge filter
3. Add visual indicator on image showing the edge width boundary
4. Add statistics showing distribution of auto-calculated edge widths
5. Allow user to adjust the safety margin (currently fixed at 2 pixels)
