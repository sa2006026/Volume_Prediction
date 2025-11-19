# Z-Stack Processing - Complete Index

## 📋 Quick Start

**For ddPCR droplet analysis, use this file:**
👉 **`merged_zstack_MIP.jpg`** (122 KB)

## 📁 All Generated Files

### 🔬 Merged Images (Primary Outputs)

| File | Size | Method | Use Case |
|------|------|--------|----------|
| **merged_zstack_MIP.jpg** ⭐ | 122 KB | Maximum Intensity Projection | **ddPCR droplets (RECOMMENDED)** |
| merged_zstack_EDF.jpg | 141 KB | Extended Depth of Field | Brightfield, all-in-focus |
| merged_zstack_AVG.jpg | 62 KB | Average Projection | Noise reduction |
| merged_zstack_WEIGHTED.jpg | 81 KB | Weighted Average | Balanced approach |

### 🖼️ Preview Images (Half Resolution)
- `merged_zstack_MIP_preview.jpg` (35 KB)
- `merged_zstack_EDF_preview.jpg` (40 KB)
- `merged_zstack_AVG_preview.jpg` (22 KB)
- `merged_zstack_WEIGHTED_preview.jpg` (28 KB)

### 📊 Comparison & Visualization

| File | Size | Description |
|------|------|-------------|
| **recommended_merged_zstack.jpg** | 130 KB | Annotated MIP with recommendation banner |
| **zstack_methods_comparison.jpg** | 415 KB | 2×2 grid comparing all 4 methods |
| zstack_methods_comparison_large.jpg | 709 KB | Larger version for detailed viewing |
| zstack_focal_planes.jpg | 111 KB | Montage showing 6 selected z-planes |
| zstack_focus_plot.jpg | 30 KB | Graph of focus quality across z-stack |
| zstack_process_diagram.jpg | 88 KB | Visual diagram of merging process |

### 📝 Documentation

| File | Description |
|------|-------------|
| **SUMMARY.md** | Complete summary with technical details |
| **README.md** | Detailed documentation and usage guide |
| **INDEX.md** | This file - navigation guide |
| merge_zstack.py | Main merging script |
| compare_methods.py | Creates comparison visualizations |
| visualize_zstack.py | Creates process visualizations |

### 📂 Input Data
- `images/` directory: 20 z-stack planes (z00-z19, 512×512 pixels each)

## 🎯 Which File Should I Use?

### For Droplet Analysis
✅ Use **`merged_zstack_MIP.jpg`**
- Captures all bright droplets across all focal planes
- Standard method for fluorescence microscopy
- Best for counting and quantification

### For Visual Inspection
✅ Use **`zstack_methods_comparison.jpg`**
- See all 4 methods side-by-side
- Compare which method works best for your data

### For Understanding the Process
✅ Use **`zstack_focal_planes.jpg`** and **`zstack_focus_plot.jpg`**
- See how focus changes across z-planes
- Understand the focal distribution

### For Presentations/Publications
✅ Use **`recommended_merged_zstack.jpg`** or **`merged_zstack_MIP.jpg`**
- High quality, publication-ready
- Standard method used in scientific literature

## 🔧 How to Process More Z-Stacks

### Process Different Images
```bash
python3 merge_zstack.py --input /path/to/other/zstack/ --output series2.jpg
```

### Generate Only MIP (Fastest)
```bash
python3 merge_zstack.py --method mip --output quick_mip.jpg
```

### Process TIFF Files
```bash
python3 merge_zstack.py --pattern "*.tif" --output merged_tiff.jpg
```

### Adjust Focus Kernel
```bash
python3 merge_zstack.py --method edf --kernel-size 11 --output fine_edf.jpg
```

## 📈 Processing Stats

- **Input**: 20 z-stack images (z00-z19)
- **Total input size**: ~3.8 MB
- **Total output size**: ~2.2 MB
- **Peak focus plane**: z09
- **Processing time**: ~5-10 seconds
- **Image resolution**: 512×512 pixels

## 🔬 Technical Methods Used

### Maximum Intensity Projection (MIP)
```
For each pixel: result = max(all_z_planes)
```
- **Advantages**: Preserves all bright features
- **Standard in**: Fluorescence, confocal microscopy
- **Best for**: ddPCR droplets, fluorescent particles

### Extended Depth of Field (EDF)
```
For each pixel: result = sharpest_z_plane
Based on: Laplacian variance focus measure
```
- **Advantages**: All-in-focus composite
- **Standard in**: Brightfield microscopy
- **Best for**: Samples with varying depth

### Average Projection
```
For each pixel: result = mean(all_z_planes)
```
- **Advantages**: Noise reduction, smooth result
- **Standard in**: Noise reduction workflows
- **Best for**: Reducing image noise

### Weighted Average
```
For each pixel: result = Σ(weight[z] × pixel[z]) / Σ(weight[z])
Based on: Local variance weighting
```
- **Advantages**: Balances sharpness and smoothness
- **Standard in**: Advanced microscopy workflows
- **Best for**: Complex samples

## 🚀 Next Steps

### 1. Use with SAM Website
```bash
cd /data3/megan_data/Jimmy/Volume_Prediction/src/web
python3 sam_website.py
```
Then upload `merged_zstack_MIP.jpg` for automatic droplet segmentation.

### 2. Manual Analysis
Open `merged_zstack_MIP.jpg` in ImageJ, Fiji, or any image analysis software.

### 3. Batch Processing
Process multiple z-stack series using the provided scripts.

## 📚 References

- **MIP**: Standard practice in confocal microscopy (Pawley, 2006)
- **EDF**: Focus stacking technique (Forster et al., 2004)
- **Focus Measure**: Laplacian variance method (Pech-Pacheco et al., 2000)

## ✅ Quality Checklist

- [x] All 20 z-planes loaded successfully
- [x] 4 merging methods generated
- [x] Comparison visualizations created
- [x] Focus analysis completed (peak at z09)
- [x] Documentation generated
- [x] Preview images created
- [x] Scripts ready for batch processing

## 🎓 Understanding the Results

### Why is z09 the peak focus?
Your sample was positioned such that plane z09 (middle of the stack) had the sharpest overall features. This is ideal as it means your z-stack captured the full depth of field.

### Why use MIP for ddPCR?
ddPCR droplets are fluorescent spheres that may be at different depths. MIP ensures you capture the brightest point of each droplet, regardless of which z-plane it's in focus at.

### What if I see artifacts?
- If MIP has too much background: Try EDF
- If edges look sharp: Try Average or Weighted
- If you need more detail: Adjust EDF kernel size (11-21)

---

**Status**: ✅ Complete  
**Date**: November 12, 2025  
**Input**: 20 z-stack images  
**Output**: 4 merging methods + 6 visualizations + 3 documentation files

