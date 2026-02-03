# Quick Start: Dark Edge Analysis

## 🚀 New Workflow (Recommended)

### Step 1: Upload Image
```
Click "Upload Image" → Select your microscopy image
```

### Step 2: Configure Segmentation
```
SAM Parameters:
├─ Model Size: vit_b (default)
├─ Crop Layers: 1-2
└─ Points per Side: 32-64

Dark Edge Analysis: ✅ ENABLE
├─ Auto Edge Width: ✅ ENABLE (recommended)
├─ Edge Width: 5 pixels (if auto disabled)
└─ Darkness Threshold: 80 (adjust based on image)
```

### Step 3: Run Segmentation
```
Click "Run Segmentation"
→ System segments masks AND calculates ring width data
→ All data cached for instant export
```

### Step 4: Set Unit Conversion (Optional)
```
Example: 100 pixels = 10 μm
→ All measurements converted automatically
```

### Step 5: Export CSV
```
Click "Export CSV"
→ Instant export with ALL data including ring width
→ No waiting, no recalculation!
```

## 📊 CSV Output

### With Dark Edge Analysis
```csv
Mask_ID,Center_X_px,Center_Y_px,Diameter_μm,Mean_Intensity,Area_μm²,Circularity,Ring_Width_μm,Dark_Edge_Diameter_μm,Dark_Ratio
0,189.00,379.00,24.46,130.05,390.62,0.827,1.51,26.78,0.604
```

### Without Dark Edge Analysis
```csv
Mask_ID,Center_X_px,Center_Y_px,Diameter_μm,Mean_Intensity,Area_μm²,Circularity
0,189.00,379.00,24.46,130.05,390.62,0.827
```

## ⚙️ Parameter Guide

### Auto Edge Width
- ✅ **Recommended for close-packed droplets**
- Automatically calculates optimal edge width per mask
- Prevents overlap with neighboring droplets

### Edge Width (Manual)
- **5 pixels**: Good for small droplets (20-50 μm)
- **10 pixels**: Good for medium droplets (50-100 μm)
- **20 pixels**: Good for large droplets (>100 μm)

### Darkness Threshold
- **50-80**: Detects subtle dark rings (faint edges)
- **80-120**: Standard detection (most common)
- **120-150**: Only very dark rings (strong edges)

## 🎯 Tips for Best Results

1. **Enable Auto Edge Width** for consistent results
2. **Adjust Darkness Threshold** based on your image contrast
3. **Apply Filters** (overlap, circularity) before export
4. **Set Unit Conversion** for real-world measurements
5. **Check Console Log** for progress and statistics

## ⚡ Performance Comparison

| Task | Old Method | New Method |
|------|------------|------------|
| Segmentation | 10s | 15s |
| Export (50 masks) | 25-50s | 0.05s |
| **Total Time** | **35-60s** | **15s** |

**Result: 2-4x faster overall!** 🚀

## ❓ FAQ

### Q: Do I need to enable dark edge analysis?
**A:** Only if you want ring width data in CSV export. Otherwise, skip it for faster segmentation.

### Q: Can I export without ring width data?
**A:** Yes! If you don't enable dark edge analysis during segmentation, CSV will only include basic mask data.

### Q: What if I forget to enable dark edge analysis?
**A:** Just run segmentation again with it enabled. The cache will be rebuilt.

### Q: Can I change parameters after segmentation?
**A:** You need to re-run segmentation with new parameters. The cache is cleared on each segmentation.

### Q: How do I know if dark edge data is available?
**A:** Check the segmentation result message. It will say "Dark edge data calculated and cached for X masks."

## 🔍 Troubleshooting

### CSV doesn't include ring width columns
→ Enable "Calculate Dark Edges" during segmentation

### Ring width values are 0
→ Increase darkness threshold or check image contrast

### Ring width seems too large
→ Enable "Auto Edge Width" to prevent neighbor overlap

### Export is slow
→ This shouldn't happen! Check console for errors

## 📝 Example Workflow

```
1. Upload: droplet_image.tif
2. Configure:
   - Model: vit_b
   - Points: 32
   - Dark Edge: ✅ ON
   - Auto Width: ✅ ON
   - Threshold: 80
3. Segment → Wait 15s → 50 masks found
4. Convert: 100px = 10μm
5. Export → Instant! → mask_data_filtered_μm_20251210.csv
```

## 🎉 Success!

You now have a CSV file with:
- ✅ Mask positions (center X, Y)
- ✅ Diameters in real units (μm)
- ✅ Mean intensity values
- ✅ Areas in real units (μm²)
- ✅ Circularity scores
- ✅ **Ring width measurements (μm)**
- ✅ **Dark edge diameters (μm)**
- ✅ **Dark pixel ratios**

Ready for analysis in Excel, Python, R, or any data analysis tool!

