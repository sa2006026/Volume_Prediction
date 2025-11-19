# Z-Stack Image Merger

This tool merges z-stack microscopy images using industry-standard methods.

## Output Files

Successfully processed **20 z-stack images** (z00 to z19) and generated 4 different merged versions:

### 1. **Maximum Intensity Projection (MIP)** ⭐ RECOMMENDED FOR YOUR DATA
- **File**: `merged_zstack_MIP.jpg` (122 KB)
- **Best for**: Fluorescence microscopy, ddPCR droplet imaging
- **Method**: Takes the maximum pixel intensity across all z-planes
- **Use case**: Preserves the brightest features from all focal planes, ideal for visualizing all droplets regardless of their focal plane
- **Why it's best for ddPCR**: Captures all bright droplets across the entire z-stack, ensuring no droplets are missed

### 2. **Extended Depth of Field (EDF) - Focus Stacking**
- **File**: `merged_zstack_EDF.jpg` (141 KB)
- **Best for**: Brightfield microscopy, samples with varying depth
- **Method**: Combines the sharpest regions from each focal plane using Laplacian focus measure
- **Use case**: Creates an all-in-focus image where different regions may have been at different focal planes
- **Notes**: Peak focus detected at z09 (focal plane 9), which had the highest mean focus measure

### 3. **Average Intensity Projection**
- **File**: `merged_zstack_AVG.jpg` (62 KB)
- **Best for**: Noise reduction, smooth visualization
- **Method**: Averages pixel intensities across all z-planes
- **Use case**: Reduces noise but may decrease contrast of dim features
- **Notes**: Smallest file size due to smoothing effect

### 4. **Weighted Average Projection**
- **File**: `merged_zstack_WEIGHTED.jpg` (81 KB)
- **Best for**: Balanced approach between sharpness and noise reduction
- **Method**: Weights each z-plane by its local variance/sharpness
- **Use case**: Emphasizes sharp regions while smoothing out-of-focus areas

## Focus Analysis

Based on the Laplacian focus measure, the focal quality across your z-stack:
- **Best focus**: z09 (focal plane 9) with mean focus = 91,001,053
- **Focus range**: Gradually increases from z00 to z09, then decreases to z19
- **Interpretation**: Your sample has a clear focal center at z09, with droplets distributed across multiple planes

## Usage

### Basic usage (generates all methods):
```bash
python3 merge_zstack.py
```

### Generate only MIP (fastest):
```bash
python3 merge_zstack.py --method mip --output my_merged.jpg
```

### Generate only EDF with custom kernel size:
```bash
python3 merge_zstack.py --method edf --kernel-size 11 --output focused.jpg
```

### Process different directory:
```bash
python3 merge_zstack.py --input /path/to/zstack/ --pattern "*.tif"
```

## Command-Line Options

```
--input, -i       : Input directory (default: images/)
--output, -o      : Output filename (default: merged_zstack.jpg)
--method, -m      : Method to use (mip|edf|average|weighted|all)
--pattern, -p     : File pattern to match (default: *.jpg)
--kernel-size, -k : Kernel size for EDF (default: 15)
```

## Recommendation for Your ddPCR Data

For **ddPCR droplet analysis**, I recommend using:
1. **Primary**: `merged_zstack_MIP.jpg` - Captures all bright droplets across all focal planes
2. **Secondary**: `merged_zstack_EDF.jpg` - If you need the sharpest representation

The MIP method is the gold standard in fluorescence microscopy and is perfect for counting and analyzing droplets that may be at different depths in your sample.

## Preview Files

Each method also generates a half-resolution preview file (e.g., `merged_zstack_MIP_preview.jpg`) for quick viewing without loading the full-resolution image.

## Technical Details

- **Input images**: 20 z-stack planes (512x512 pixels each)
- **Z-range**: z00 to z19 (20 focal planes)
- **Processing time**: ~5-10 seconds for all methods
- **Memory usage**: Efficient processing using NumPy stack operations

## Next Steps

You can now use these merged images for:
1. **SAM segmentation** via `sam_website.py`
2. **Droplet counting and analysis**
3. **Quality assessment** of your ddPCR experiment
4. **Publication figures** (MIP or EDF recommended)

## References

- Maximum Intensity Projection: Standard in confocal and fluorescence microscopy
- Extended Depth of Field: Focus stacking technique from macro photography, adapted for microscopy
- Laplacian variance: Standard focus measure in computational microscopy

