# Pre-Segmentation Filter API Documentation

## Overview
The new pre-segmentation filter allows you to adjust brightness, contrast, and apply advanced pixel intensity filtering **before** running SAM segmentation. This helps clean up the image and improve segmentation results.

## API Endpoints

### 1. Apply Pre-Segmentation Filter
**Endpoint:** `POST /apply_pre_segmentation_filter`

**Description:** Apply brightness, contrast, and pixel intensity filters to prepare the image for SAM segmentation.

**Request Body:**
```json
{
  "brightness": 0,           // -100 to +100 (0 = no change)
  "contrast": 1.0,           // 0.5 to 3.0 (1.0 = no change)
  "min_threshold": -1,       // 0-255 or -1 (not used)
  "max_threshold": -1,       // 0-255 or -1 (not used)
  "filter_mode": "remove_below"  // See filter modes below
}
```

**Filter Modes:**
- `"remove_below"`: Remove (blacken) pixels **below** min_threshold
- `"remove_above"`: Remove (blacken) pixels **above** max_threshold
- `"remove_outside"` or `"keep_range"`: Remove pixels **outside** the range [min_threshold, max_threshold]

**Response:**
```json
{
  "success": true,
  "filtered_image": "data:image/png;base64,...",
  "parameters": {
    "brightness": 0,
    "contrast": 1.0,
    "min_threshold": 50,
    "max_threshold": 200,
    "filter_mode": "remove_outside"
  },
  "message": "Pre-segmentation filter applied successfully..."
}
```

### 2. Reset Pre-Segmentation Filter
**Endpoint:** `POST /reset_pre_segmentation_filter`

**Description:** Reset the image back to its original state, removing all filters.

**Request Body:** None (empty JSON object `{}`)

**Response:**
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "message": "Image reset to original state. All filters cleared."
}
```

## Usage Examples

### Example 1: Remove Dark Background Noise
Remove pixels below intensity 50 to eliminate dark background:

```javascript
fetch('/apply_pre_segmentation_filter', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    brightness: 0,
    contrast: 1.0,
    min_threshold: 50,
    max_threshold: -1,
    filter_mode: 'remove_below'
  })
});
```

### Example 2: Remove Bright Overexposed Areas
Remove pixels above intensity 200 to eliminate overexposed regions:

```javascript
fetch('/apply_pre_segmentation_filter', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    brightness: 0,
    contrast: 1.0,
    min_threshold: -1,
    max_threshold: 200,
    filter_mode: 'remove_above'
  })
});
```

### Example 3: Keep Only Mid-Range Intensities
Keep only pixels within the range [50, 200]:

```javascript
fetch('/apply_pre_segmentation_filter', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    brightness: 0,
    contrast: 1.0,
    min_threshold: 50,
    max_threshold: 200,
    filter_mode: 'keep_range'
  })
});
```

### Example 4: Brighten Image and Remove Dark Areas
Increase brightness by 30 and remove dark pixels below 80:

```javascript
fetch('/apply_pre_segmentation_filter', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    brightness: 30,
    contrast: 1.0,
    min_threshold: 80,
    max_threshold: -1,
    filter_mode: 'remove_below'
  })
});
```

### Example 5: Enhance Contrast and Filter Range
Increase contrast to 1.5x and keep pixels in [60, 180] range:

```javascript
fetch('/apply_pre_segmentation_filter', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    brightness: 0,
    contrast: 1.5,
    min_threshold: 60,
    max_threshold: 180,
    filter_mode: 'keep_range'
  })
});
```

### Example 6: Reset to Original Image
```javascript
fetch('/reset_pre_segmentation_filter', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({})
});
```

## Workflow

1. **Upload Image** → `/upload_image`
2. **Apply Pre-Segmentation Filter** → `/apply_pre_segmentation_filter` (adjust brightness, contrast, filter pixels)
3. **Preview Filtered Image** → The filtered image is returned in the response
4. **Run SAM Segmentation** → `/run_sam_segmentation` (SAM will segment the filtered image)
5. **(Optional) Reset Filter** → `/reset_pre_segmentation_filter` if you want to start over

## Key Features

✅ **Brightness Control**: Adjust image brightness from -100 to +100
✅ **Contrast Control**: Adjust image contrast from 0.5x to 3.0x
✅ **Remove Dark Pixels**: Filter out pixels below a threshold (remove background noise)
✅ **Remove Bright Pixels**: Filter out pixels above a threshold (remove overexposure)
✅ **Range Filtering**: Keep only pixels within a specific intensity range
✅ **Non-Destructive**: Original image is preserved, you can reset anytime
✅ **Pre-Segmentation**: Filters are applied BEFORE SAM segmentation for better results

## Notes

- Filtered pixels are set to **black** (RGB: 0,0,0)
- Intensity thresholds are calculated from **grayscale** values (0-255)
- All filters are applied to the original image, then the result is used for SAM segmentation
- The original image is always preserved and can be restored with the reset endpoint
- You can combine brightness, contrast, and pixel filtering in a single request

