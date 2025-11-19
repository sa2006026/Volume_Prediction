# Filtered Image Export Feature

## Overview

The SAM website now includes the ability to **export the filtered image** after applying intensity thresholds and other adjustments. This allows you to save the processed image for use in other tools or for documentation purposes.

## Location

The export button is located in the **Contrast & Intensity Adjustment** panel:

```
🎨 Contrast & Intensity Adjustment
├─ Brightness: [slider]
├─ Contrast: [slider]
├─ Lower Intensity Threshold: [slider]
├─ Upper Intensity Threshold: [slider]
├─ [Apply Adjustments] [Reset]
└─ [📥 Export Filtered Image]  ← NEW!
   └─ [Associate Masks to Closest Blobs]
```

## How to Use

### Step-by-Step

1. **Upload Image & Run SAM Segmentation**
   - Upload your image
   - Run SAM segmentation
   - Toggle masks as needed

2. **Go to Contrast & Intensity Stage**
   - Click "Go to Contrast & Intensity Stage"
   - Your masks are stored in the backend

3. **Apply Adjustments**
   - Adjust brightness, contrast, and intensity thresholds
   - Click "Apply Adjustments" to see the result
   - Iterate until you're satisfied with the filtered image

4. **Export Filtered Image**
   - Click "📥 Export Filtered Image" button
   - Image automatically downloads as PNG file
   - Filename format: `filtered_image_YYYYMMDD_HHMMSS.png`

## Features

### ✅ What Gets Exported
- The **current filtered/adjusted image** as displayed on canvas
- All adjustments applied (brightness, contrast, intensity thresholds)
- Full resolution (no quality loss from canvas display)
- Original aspect ratio maintained

### 📁 File Format
- **Default Format**: PNG (lossless)
- **Filename**: `filtered_image_20251028_123456.png`
- **Quality**: High quality (PNG compression level 3)

### 🎯 Use Cases

1. **Documentation**
   - Save filtered images for reports
   - Compare before/after processing
   - Archive processing results

2. **External Analysis**
   - Import into ImageJ, FIJI, or other tools
   - Further processing in Photoshop/GIMP
   - Use as input for other pipelines

3. **Quality Control**
   - Review filtering effectiveness offline
   - Share results with colleagues
   - Keep records of processing parameters

4. **Batch Processing Reference**
   - Save successful filtering results
   - Use as template for similar images
   - Document optimal parameter settings

## Technical Details

### Backend Implementation

```python
@app.route('/export_filtered_image', methods=['POST'])
def export_filtered_image():
    # Get last adjusted image (with all filters applied)
    image_to_export = engine.last_adjusted_image or engine.current_image
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"filtered_image_{timestamp}.png"
    
    # Save as PNG with high quality
    cv2.imwrite(temp_path, image_to_export, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    # Send file to client
    return send_file(temp_path, as_attachment=True, download_name=filename)
```

### Frontend Implementation

```javascript
async function exportFilteredImage() {
    const response = await fetch('/export_filtered_image', {
        method: 'POST',
        body: JSON.stringify({ format: 'png' })
    });
    
    // Create download link from blob
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    
    // Cleanup
    window.URL.revokeObjectURL(url);
}
```

## Example Workflows

### Workflow 1: Background Removal
```
1. Upload: sample_image.jpg
2. SAM Segmentation: Detect droplets
3. Adjust Thresholds:
   - Lower: 60 (remove dark background)
   - Upper: -1 (disabled)
4. Apply Adjustments
5. Export Filtered Image → filtered_image_20251028_123456.png
6. Result: Clean image with background removed
```

### Workflow 2: Noise Reduction
```
1. Upload: noisy_microscopy.tif
2. SAM Segmentation: Detect cells
3. Adjust Thresholds:
   - Lower: 40 (remove dark noise)
   - Upper: 230 (remove bright artifacts)
4. Apply Adjustments
5. Export Filtered Image
6. Import to ImageJ for further analysis
```

### Workflow 3: Intensity Range Isolation
```
1. Upload: fluorescence_image.jpg
2. SAM Segmentation: Detect features
3. Adjust Thresholds:
   - Lower: 80
   - Upper: 180
   - Result: Only mid-intensity features visible
4. Export Filtered Image
5. Use for quantitative analysis
```

## File Information

### Exported Image Properties
- **Resolution**: Same as original (or upscaled if ESRGAN was used)
- **Bit Depth**: 8-bit per channel (RGB)
- **Color Space**: BGR → RGB (corrected for standard viewers)
- **Compression**: PNG level 3 (good balance of size/quality)
- **Metadata**: Minimal (no EXIF data embedded)

### File Size Estimates
- **Small image** (512×512): ~200-500 KB
- **Medium image** (1024×1024): ~800 KB - 2 MB
- **Large image** (2048×2048): ~3-8 MB
- **ESRGAN upscaled** (4096×4096): ~12-30 MB

*Size varies based on image complexity and compression*

## Comparison with Other Export Options

| Feature | Export Filtered Image | Download SAM Result | Browser Save Image |
|---------|---------------------|--------------------|--------------------|
| **What's Saved** | Filtered image only | SAM masks overlay | Canvas screenshot |
| **Quality** | Full resolution | Full resolution | Display resolution |
| **Format** | PNG (high quality) | PNG | PNG/JPEG |
| **Includes Adjustments** | ✅ Yes | ❌ No | ✅ Yes |
| **Includes SAM Masks** | ❌ No | ✅ Yes | ✅ Yes (if visible) |
| **Use Case** | External processing | SAM result archive | Quick preview |

## Advanced: Future Enhancements

Possible future improvements:

1. **Multiple Format Support**
   ```html
   <select id="exportFormat">
     <option value="png">PNG (Lossless)</option>
     <option value="jpg">JPEG (Smaller file)</option>
     <option value="tiff">TIFF (16-bit support)</option>
   </select>
   ```

2. **Export with Metadata**
   - Embed processing parameters in EXIF
   - Include timestamp and adjustments used
   - Add custom metadata fields

3. **Batch Export**
   - Export original + filtered side-by-side
   - Export multiple filtered versions
   - Create comparison image

4. **Custom Filename**
   - User-defined filename prefix
   - Include parameters in filename
   - Auto-numbering for series

5. **Export Options Dialog**
   ```
   ┌─────────────────────────────────┐
   │  Export Filtered Image          │
   ├─────────────────────────────────┤
   │  Format: [PNG ▼]                │
   │  Quality: [High ▼]              │
   │  Include metadata: [✓]          │
   │  Filename: [custom_name]        │
   │                                 │
   │  [Cancel]  [Export]             │
   └─────────────────────────────────┘
   ```

## Troubleshooting

### Issue: Export button not visible
**Cause**: Not in Contrast & Intensity stage
**Solution**: Click "Go to Contrast & Intensity Stage" first

### Issue: No adjustments in exported image
**Cause**: Forgot to click "Apply Adjustments"
**Solution**: Always click "Apply Adjustments" before exporting

### Issue: Download not starting
**Cause**: Browser blocked download
**Solution**: Check browser's download settings/permissions

### Issue: Exported image is black
**Cause**: Thresholds too restrictive (removed all pixels)
**Solution**: 
- Reset adjustments
- Use less restrictive thresholds
- Export before applying extreme filters

### Issue: Wrong image exported
**Cause**: Multiple adjustments applied sequentially
**Solution**: The most recent "Apply Adjustments" result is always exported

## Tips & Best Practices

### 1. Preview Before Export
- Always click "Apply Adjustments" to preview first
- Verify the filtered image looks correct on canvas
- Check that desired features are visible

### 2. Name Organization
- Files are auto-named with timestamps
- Organize downloads by project/date
- Consider renaming after export for clarity

### 3. Quality Workflow
```
1. Upload original
2. Apply ESRGAN (optional, for quality)
3. Run SAM segmentation
4. Go to Contrast & Intensity stage
5. Adjust and preview repeatedly
6. Export when satisfied
7. Verify exported file before proceeding
```

### 4. Backup Originals
- Keep original images separate
- Export filtered versions don't replace originals
- Create organized folder structure:
  ```
  project/
  ├── originals/
  │   └── sample_image.jpg
  ├── filtered/
  │   └── filtered_image_20251028_123456.png
  └── results/
      └── sam_segmentation_result.png
  ```

### 5. Document Parameters
- Take screenshots of adjustment settings
- Note threshold values used
- Create processing log for reproducibility

## API Reference

### Endpoint
```
POST /export_filtered_image
```

### Request Body
```json
{
  "format": "png"
}
```

### Response
- **Success**: Binary file download (image/png)
- **Error**: JSON with error message

### Response Headers
```
Content-Type: image/png
Content-Disposition: attachment; filename="filtered_image_20251028_123456.png"
```

## Integration Example

### Using with Python
```python
import requests

# Export filtered image
response = requests.post(
    'http://localhost:5001/export_filtered_image',
    json={'format': 'png'}
)

# Save to file
if response.status_code == 200:
    with open('exported_filtered.png', 'wb') as f:
        f.write(response.content)
    print("Image exported successfully!")
```

### Using with cURL
```bash
curl -X POST \
  http://localhost:5001/export_filtered_image \
  -H "Content-Type: application/json" \
  -d '{"format":"png"}' \
  --output filtered_image.png
```

## Summary

The filtered image export feature provides:

✅ **One-Click Export** - Simple button to download processed image
✅ **High Quality** - Full resolution PNG format
✅ **Timestamped Filenames** - Automatic naming with date/time
✅ **All Adjustments Included** - Brightness, contrast, thresholds applied
✅ **Ready for External Use** - Compatible with all image tools

This feature is essential for:
- Documenting processing results
- Using filtered images in external tools
- Quality control and review
- Archiving processed data
- Sharing results with colleagues

Perfect for workflows where you need to apply intensity filtering and then use the cleaned image in downstream analysis!

