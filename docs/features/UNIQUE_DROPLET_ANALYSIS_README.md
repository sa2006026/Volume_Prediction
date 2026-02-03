# Unique Droplet Maximum Intensity Analysis

This tool analyzes CSV files from different Z-levels to find ALL unique droplet locations. For droplets that appear at the same location (within 5 pixels), it keeps only the one with the highest intensity.

## Key Difference from Previous Analysis

- **Previous tool**: Tracked specific droplets across Z-levels (found 8 droplets that appeared in all 17 Z-levels)
- **This tool**: Finds ALL unique locations across ALL Z-levels (found 103 unique locations from 673 total droplets)

## Files

- **`find_unique_max_intensity_droplets.py`** - Main analysis library with UniqueDropletAnalyzer class
- **`analyze_unique_droplets.py`** - Simple command-line interface

## Quick Usage

### Analyze the default csv/1 directory:
```bash
python3 analyze_unique_droplets.py
```

### Analyze a specific directory:
```bash
python3 analyze_unique_droplets.py /path/to/csv/directory
```

### Advanced usage with custom parameters:
```bash
python3 find_unique_max_intensity_droplets.py csv/1 --threshold 10 --top-n 25
```

## How It Works

1. **Loads ALL droplets** from all CSV files (673 droplets total)
2. **Groups by location** - finds droplets within 5 pixels of each other
3. **Keeps maximum intensity** - for each location group, keeps only the droplet with highest intensity
4. **Results in unique locations** - 103 unique locations (84.7% reduction in duplicates)

## Analysis Results Summary

From your CSV files:
- **Total droplets found**: 673 across all Z-levels
- **Unique locations**: 103 (after removing duplicates)
- **Duplicates removed**: 570 (84.7% reduction)
- **Maximum intensity**: 56.35 (at location 447.0, 279.0 from z08)

## Top Findings

**Rank #1**: Location (447.0, 279.0)
- Max intensity: **56.35** from z08
- Found in **22 different instances** across Z-levels
- Intensity range: 12.3 - 56.4

**Rank #2**: Location (381.0, 121.0)  
- Max intensity: **55.55** from z09
- Found in **20 different instances** across Z-levels
- Intensity range: 13.9 - 55.6

## Output Files

### Console Output
- Summary statistics (total droplets, unique locations, reduction percentage)
- Top droplets ranked by maximum intensity
- Details for each location (coordinates, max intensity, source Z-level, number of instances)

### CSV Output: `unique_max_intensity_droplets.csv`
Contains columns:
- `Rank`: Ranking by intensity
- `Center_X`, `Center_Y`: Droplet coordinates
- `Max_Intensity`: Highest intensity found at this location
- `Source_Z_Level`: Which Z-level had the maximum intensity
- `Diameter`, `Area`, `Circularity`: Properties from the max intensity droplet
- `Droplets_At_Location`: How many droplets were found at this location
- `Z_Levels_At_Location`: All Z-levels where droplets were found
- `All_Intensities`: All intensity values found at this location

## Parameters

- **Location Threshold**: 5 pixels (default) - distance to consider droplets as "same location"
- **Top N**: Number of top results to display in console (default: 20)

## Use Cases

This analysis is perfect for:
- **Finding all unique droplet locations** across a Z-stack
- **Identifying optimal Z-level** for each droplet location
- **Removing duplicate measurements** from the same physical droplet
- **Getting comprehensive droplet inventory** with maximum intensity values

## Requirements

- Python 3.6+
- pandas
- numpy

Install requirements:
```bash
pip install pandas numpy
```
