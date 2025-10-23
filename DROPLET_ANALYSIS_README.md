# Droplet Maximum Intensity Analysis

This tool analyzes CSV files from different Z-levels to find droplets that appear at similar locations and tracks their maximum intensity values across the Z-stack.

## Files

- **`find_max_intensity_droplets.py`** - Main analysis library with DropletTracker class
- **`analyze_droplets.py`** - Simple command-line interface
- **`run_droplet_analysis.py`** - Example script for the csv/1 directory

## Quick Usage

### Analyze the default csv/1 directory:
```bash
python3 analyze_droplets.py
```

### Analyze a specific directory:
```bash
python3 analyze_droplets.py /path/to/csv/directory
```

### Advanced usage with custom parameters:
```bash
python3 find_max_intensity_droplets.py csv/1 --threshold 15 --min-appearances 3 --top-n 20
```

## How It Works

1. **Loads CSV files** from the specified directory (expects files named like z00.csv, z01.csv, etc.)
2. **Tracks droplets** across Z-levels by finding droplets within 10 pixels of each other
3. **Finds maximum intensity** for each tracked droplet across all Z-levels
4. **Generates results** showing:
   - Droplet location (x, y coordinates)
   - Maximum intensity value and which Z-level it occurs at
   - Number of Z-levels the droplet appears in
   - Intensity progression across Z-levels

## Output

### Console Output
- Summary of analysis results
- Top droplets ranked by maximum intensity
- Intensity progression for each droplet

### CSV Output
- Detailed tracking results saved to `droplet_max_intensity_analysis.csv`
- Contains all droplet data across Z-levels
- Includes coordinates, intensities, diameters for each Z-level

## Parameters

- **Location Threshold**: Distance in pixels to consider droplets as "same location" (default: 10)
- **Minimum Appearances**: Minimum Z-levels a droplet must appear in to be included (default: 2)
- **Top N**: Number of top droplets to display in console output (default: 10-20)

## Example Results

```
Rank #1 - Track ID: 4
  Base Location: (447.0, 279.0)
  Max Intensity: 56.35 (at z08)
  Appearances: 17 z-levels
  Z-levels: z00, z01, z02, z03, z04, z05, z06, z07, z08, z09, z10, z11, z12, z13, z14, z15, z16
  Intensity progression: z00:15.6 → z01:21.4 → z02:27.6 → z03:36.1 → z04:42.2 → z05:39.0 → z06:44.8 → z07:54.4 → z08:56.4 → z09:55.9 → z10:55.3 → z11:52.5 → z12:48.8 → z13:44.3 → z14:38.2 → z15:32.9 → z16:28.8
```

## CSV File Format Expected

The tool expects CSV files with the following columns:
- `Mask_ID`: Unique identifier for the mask
- `Center_X`: X coordinate of droplet center
- `Center_Y`: Y coordinate of droplet center  
- `Diameter`: Droplet diameter
- `Mean_Intensity`: Mean pixel intensity of the droplet
- `Area`: Droplet area
- `Circularity`: Circularity measure (0-1)

## Requirements

- Python 3.6+
- pandas
- numpy

Install requirements:
```bash
pip install pandas numpy
```
