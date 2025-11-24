# Frontend Usage Guide - Pre-Segmentation Filter

## New UI Feature Added ✨

The SAM website now includes a **Pre-Segmentation Filter** panel that appears after you upload an image, allowing you to clean up and enhance the image before running SAM segmentation.

---

## UI Location

The Pre-Segmentation Filter panel appears in the left control panel:

```
┌─────────────────────────────────┐
│  📁 Upload Image for Analysis  │  ← Step 1
├─────────────────────────────────┤
│  🎨 Pre-Segmentation Filter     │  ← NEW! Step 2 (appears after upload)
├─────────────────────────────────┤
│  ⚙️ SAM Configuration           │  ← Step 3
├─────────────────────────────────┤
│  🎯 Run SAM Segmentation        │  ← Step 4
└─────────────────────────────────┘
```

---

## UI Controls

### 1. Brightness Slider
- **Range:** -100 (darkest) to +100 (brightest)
- **Default:** 0 (no change)
- **Use:** Lighten or darken the entire image

### 2. Contrast Slider
- **Range:** 0.5 (low contrast) to 3.0 (high contrast)
- **Default:** 1.0 (no change)
- **Use:** Increase or decrease image contrast

### 3. Filter Mode Dropdown
**Options:**
- **No Pixel Filtering** - Only brightness/contrast (no pixel removal)
- **Remove Dark Pixels (Below Threshold)** - Blacken pixels below min threshold
- **Remove Bright Pixels (Above Threshold)** - Blacken pixels above max threshold
- **Keep Only Range (Between Thresholds)** - Blacken pixels outside [min, max] range

### 4. Min Threshold Slider (conditional)
- **Shows when:** "Remove Dark Pixels" or "Keep Only Range" selected
- **Range:** 0-255
- **Default:** 50
- **Use:** Set minimum intensity to keep

### 5. Max Threshold Slider (conditional)
- **Shows when:** "Remove Bright Pixels" or "Keep Only Range" selected
- **Range:** 0-255
- **Default:** 200
- **Use:** Set maximum intensity to keep

### 6. Action Buttons
- **Apply Filter** - Apply the current filter settings to the image
- **Reset** - Restore the original uploaded image (undo all filters)

---

## Step-by-Step Workflow

### Example 1: Remove Dark Background

1. **Upload your image**
   - The Pre-Segmentation Filter panel appears

2. **Select filter mode**
   - Choose: "Remove Dark Pixels (Below Threshold)"

3. **Adjust Min Threshold**
   - Drag slider to set threshold (e.g., 50)
   - This will remove all pixels with intensity < 50

4. **Click "Apply Filter"**
   - The image updates showing filtered result
   - Success message appears

5. **Run SAM Segmentation**
   - SAM will now segment the filtered image
   - Background noise is eliminated

6. **(Optional) Click "Reset"**
   - Returns to original image if you want to try different settings

---

### Example 2: Keep Specific Intensity Range

1. **Upload your image**

2. **Select filter mode**
   - Choose: "Keep Only Range (Between Thresholds)"

3. **Adjust Both Thresholds**
   - Min Threshold: 60 (removes pixels < 60)
   - Max Threshold: 180 (removes pixels > 180)

4. **Click "Apply Filter"**
   - Only pixels in range [60, 180] remain visible
   - Everything else turns black

5. **Run SAM Segmentation**
   - SAM segments only the objects in your target intensity range

---

### Example 3: Brighten and Filter

1. **Upload your image**

2. **Adjust Brightness**
   - Move brightness slider to +30

3. **Adjust Contrast**
   - Move contrast slider to 1.5

4. **Select filter mode**
   - Choose: "Remove Dark Pixels (Below Threshold)"

5. **Set Min Threshold**
   - Drag slider to 70

6. **Click "Apply Filter"**
   - Image is brightened, contrast enhanced, and dark pixels removed

7. **Run SAM Segmentation**
   - Much better segmentation on the enhanced image!

---

## Visual Feedback

### Success State
When filter is applied successfully:
```
✅ Pre-segmentation filter applied successfully. 
   You can now run SAM segmentation.
```
- Green success banner appears
- Status box shows filter parameters
- Image updates to show filtered result

### Reset State
When reset is clicked:
```
✅ Image reset to original state!
```
- Image returns to original
- All sliders reset to defaults
- Filter status box disappears

---

## UI Behavior Details

### Filter Mode Changes
- When you change the filter mode dropdown, the threshold sliders automatically show/hide:
  - **No Filtering:** Both sliders hidden
  - **Remove Dark:** Only Min Threshold shown
  - **Remove Bright:** Only Max Threshold shown
  - **Keep Range:** Both sliders shown

### Slider Value Display
- All sliders show their current value in real-time
- Values update as you drag the slider (no need to release)

### Button States
- **Apply Filter button:** Always enabled once image is uploaded
- **Reset button:** Always enabled once image is uploaded
- Buttons show loading state while processing

---

## Tips Section

The panel includes a helpful tips section:

```
💡 Filter Tips
• Remove Dark: Eliminate background noise and shadows
• Remove Bright: Eliminate overexposed areas or glare
• Keep Range: Isolate specific intensity ranges
• Tip: Apply filter first, then run SAM segmentation
```

---

## Common Use Cases

### Microscopy Images
**Problem:** Dark noisy background interfering with cell detection

**Solution:**
1. Filter Mode: "Remove Dark Pixels"
2. Min Threshold: 40-60
3. Apply Filter → Run SAM

---

### Droplet Detection
**Problem:** Bright background making droplets hard to detect

**Solution:**
1. Brightness: -20
2. Contrast: 1.3
3. Filter Mode: "Keep Only Range"
4. Min: 50, Max: 150
5. Apply Filter → Run SAM

---

### Fluorescence Imaging
**Problem:** Want to detect only specific intensity signals

**Solution:**
1. Filter Mode: "Keep Only Range"
2. Min Threshold: 80
3. Max Threshold: 200
4. Apply Filter → Run SAM

---

## Integration with Existing Features

### Works Seamlessly With:
- ✅ **SAM Configuration** - Filter applied before SAM runs
- ✅ **Mask Management** - Toggle, remove masks as usual
- ✅ **Intensity Filter** - Can use both pre and post filters
- ✅ **Overlap Filter** - Works with filtered images
- ✅ **Unit Conversion** - Measurements work on filtered masks
- ✅ **CSV Export** - Exports masks from filtered images

### Workflow:
```
Upload → Pre-Filter → SAM Segment → Post-Filter → Export
         ↑________ Reset ________↓
```

---

## Keyboard Shortcuts (Future Enhancement)

Currently none, but could add:
- `Ctrl+Z`: Quick reset
- `Ctrl+Enter`: Apply filter
- Arrow keys: Adjust sliders

---

## Responsive Design

- Panel scrolls independently if controls exceed viewport height
- Works on tablets and large mobile devices
- Sliders adapt to container width

---

## Error Handling

### No Image Loaded
If you try to apply filter without uploading:
```
❌ Please upload an image first!
```

### Network Error
If server communication fails:
```
❌ Error applying filter. Please try again.
```

### Invalid Parameters
Backend validates all parameters and returns helpful errors

---

## Performance Notes

- Filter application takes ~0.5-2 seconds depending on image size
- Loading spinner shows during processing
- Original image is always preserved on server
- No performance impact on SAM segmentation speed

---

## Browser Compatibility

- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Opera

---

## Testing Checklist

- [ ] Upload an image - Pre-filter panel appears
- [ ] Adjust brightness slider - Value updates in real-time
- [ ] Adjust contrast slider - Value updates in real-time
- [ ] Change filter mode - Threshold sliders show/hide correctly
- [ ] Apply filter with "Remove Dark" - Image darkens appropriately
- [ ] Apply filter with "Keep Range" - Both thresholds work
- [ ] Click Reset - Image returns to original
- [ ] Run SAM after filtering - Segmentation uses filtered image
- [ ] Apply multiple filters - Can iterate and refine
- [ ] Check status messages - Success/error messages appear

---

## Screenshots (Conceptual)

### Before Upload
```
┌─────────────────────────────────┐
│  📁 Upload Image for Analysis  │
│  [Drop zone or browse]         │
└─────────────────────────────────┘
```

### After Upload (NEW!)
```
┌─────────────────────────────────┐
│  📁 Upload Image for Analysis  │
│  ✅ image.jpg (2048x2048)      │
├─────────────────────────────────┤
│  🎨 Pre-Segmentation Filter     │ ← NEW!
│  Brightness: [====|====] 0      │
│  Contrast:   [====|====] 1.0    │
│  Filter Mode: [▼ No Filtering]  │
│  [Apply Filter] [Reset]         │
├─────────────────────────────────┤
│  ⚙️ SAM Configuration           │
│  ...                            │
└─────────────────────────────────┘
```

### With Filter Applied
```
┌─────────────────────────────────┐
│  🎨 Pre-Segmentation Filter     │
│  Brightness: [==|======] -20    │
│  Contrast:   [======|==] 1.5    │
│  Filter Mode: [▼ Remove Dark]   │
│  Min Threshold: [====|=] 50     │
│  [Apply Filter] [Reset]         │
│  ✅ Filter applied successfully!│ ← Status
└─────────────────────────────────┘
```

---

## Troubleshooting

**Q: Pre-filter panel doesn't appear**
- A: Make sure image upload was successful
- Check browser console for errors

**Q: Filter has no effect**
- A: Check if thresholds are too extreme
- Try adjusting brightness/contrast first

**Q: Image turns completely black**
- A: Thresholds are too strict or inverted
- Click Reset and use more lenient values

**Q: Can't run SAM after filtering**
- A: Make sure SAM configuration is set
- Check that segmentation button is enabled

---

## Future Enhancements (Roadmap)

- [ ] Live preview (show filter effect before applying)
- [ ] Preset filters (one-click common configurations)
- [ ] Histogram display (visualize intensity distribution)
- [ ] Before/After comparison slider
- [ ] Save/Load filter presets
- [ ] Batch processing (apply same filter to multiple images)
- [ ] Advanced filters (Gaussian blur, edge detection, etc.)

---

## Summary

The new Pre-Segmentation Filter UI provides:
- ✨ Easy-to-use controls for image enhancement
- 🎨 Visual feedback and real-time updates
- 🔄 Non-destructive editing with reset capability
- 📊 Multiple filter modes for different use cases
- 🚀 Seamless integration with existing SAM workflow

**Ready to use!** Just upload an image and the panel will appear automatically.

