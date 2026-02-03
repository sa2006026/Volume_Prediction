# Droplet Maximum Diameter Analysis

## Overview

This analysis identifies unique droplets across 16 z-slice images (z_0 through z_15) and finds the maximum diameter for each unique droplet. A droplet is considered unique if its (x,y) location is within a 10-pixel error margin of another droplet.

## Methodology

### Droplet Clustering Algorithm

The analysis uses an iterative clustering approach to identify unique droplets:

1. **Data Loading**: All droplet data from z_0.csv through z_15.csv are loaded and combined
2. **Spatial Clustering**: Droplets are grouped based on their (x,y) coordinates using a 10-pixel error margin
3. **Iterative Expansion**: For each cluster, the algorithm iteratively finds all droplets within 10 pixels of any droplet already in the cluster
4. **Maximum Diameter Selection**: For each cluster, the droplet with the maximum diameter is selected
5. **Result Recording**: The (x,y) coordinates, maximum diameter, and source slide are recorded

### Key Features

- **Error Margin**: 10 pixels (configurable)
- **Clustering Method**: Iterative spatial clustering based on Euclidean distance
- **Max Diameter**: Finds the absolute maximum diameter within each cluster

## Results

### Summary Statistics

- **Total unique droplets found**: 788
- **Average diameter**: 33.43 μm
- **Maximum diameter**: 212.24 μm (found at position 54, 493 in z_1)
- **Minimum diameter**: 5.89 μm
- **Median diameter**: 33.20 μm

### Distribution by Slide

The following table shows how many unique droplets achieved their maximum diameter in each z-slice:

| Slide | Count | Description |
|-------|-------|-------------|
| z_15 | 445 | Highest count - many droplets reach max diameter at this depth |
| z_14 | 103 | Second highest |
| z_0 | 54 | Third highest |
| z_13 | 46 | |
| z_12 | 23 | |
| z_1 | 22 | |
| z_11 | 20 | |
| z_10 | 19 | |
| z_2 | 12 | |
| z_4 | 11 | |
| z_3 | 9 | |
| z_6 | 7 | |
| z_5 | 5 | |
| z_7 | 4 | |
| z_8 | 4 | |
| z_9 | 4 | |

### Key Findings

1. **Depth Distribution**: The majority of droplets (445 out of 788, ~56%) reach their maximum diameter at z_15, suggesting this is the deepest focal plane where most droplets are at their widest cross-section.

2. **Focal Plane Analysis**: The distribution suggests that most droplets are oriented such that their maximum diameter appears near the bottom of the imaging stack (z_15).

3. **Outliers**: The largest droplet (212.24 μm) appears in z_1, suggesting some large droplets may be positioned differently in the imaging volume.

## Output Files

### max_diameter_droplets.csv

Contains the results with the following columns:

- **Center_X_px**: X-coordinate of the droplet center (in pixels)
- **Center_Y_px**: Y-coordinate of the droplet center (in pixels)
- **Diameter_μm**: Maximum diameter found for this unique droplet (in micrometers)
- **slide**: The z-slice where this maximum diameter was observed

## Usage

To run the analysis:

```bash
cd /data3/megan_data/Jimmy
python3 analyze_max_diameter.py
```

The script will:
1. Load all z-slice CSV files
2. Identify unique droplets
3. Find maximum diameters
4. Export results to `max_diameter_droplets.csv`
5. Display summary statistics

## Code Files

- **process_droplets.py**: Core clustering and maximum diameter finding algorithm
- **analyze_max_diameter.py**: Main script to run the analysis
- **max_diameter_droplets.csv**: Output file with results

## Technical Details

### Error Handling

- NaN values in diameter measurements are automatically filtered out
- Empty clusters are skipped
- Files with non-standard naming patterns (e.g., z_0_bf_anchor_occupancy.csv) are excluded

### Performance

- Uses numpy and pandas for efficient computation
- Scipy's cdist for fast distance calculations
- Processes ~6000+ droplets across 16 slices in seconds

## Future Improvements

Possible enhancements:
1. Adjustable error margin parameter
2. 3D visualization of droplet positions and diameters
3. Statistical analysis of diameter distributions
4. Export to additional formats (Excel, JSON)
5. Interactive visualization dashboard

## Author

Generated for analysis of ddPCR droplet data
Date: November 2024

