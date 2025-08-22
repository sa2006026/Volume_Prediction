python3 statistical_light_extractor.py image/overfocus.jpg --std-multiplier 2.0 --output results/mult_2.0/ --filename "2.0"


# Statistical Light Extractor - User Manual

## 📖 Overview

The **Statistical Light Extractor** is a Python tool for extracting light intensity pixels from images using various statistical methods. It provides multiple algorithms to identify bright regions in images with customizable parameters for different use cases.

## 🚀 Quick Start

### Basic Usage
```bash
python3 statistical_light_extractor.py image/your_image.jpg
```

### With Custom Parameters, Output Path, and Filename
```bash
# Default filename
python3 statistical_light_extractor.py image/your_image.jpg --method basic --std-multiplier 2.0 --output /path/to/your/output/

# Custom filename
python3 statistical_light_extractor.py image/your_image.jpg --method basic --std-multiplier 2.0 --output /path/to/your/output/ --filename my_custom_name
```


## 📋 Command Line Arguments

### Required Arguments
- **`image_path`** - Path to the input image file

### Optional Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--method` | str | `basic` | Statistical method to use |
| `--std-multiplier` | float | `1.5` | Standard deviation multiplier (basic method) |
| `--sensitivity` | float | `0.5` | Sensitivity for adaptive method (0.0-1.0) |
| `--z-threshold` | float | `1.5` | Z-score threshold (zscore method) |
| `--iqr-multiplier` | float | `1.5` | IQR multiplier (iqr method) |
| `--output` | str | None | **Output directory path** (creates if doesn't exist) |
| `--filename` | str | None | **Custom output filename** (without extension) |
| `--visualize` | flag | False | Show visualization |

## 🔬 Available Methods

### 1. Basic Method (`--method basic`)
**Formula:** `threshold = mean + (std_multiplier × std)`

**Parameters:**
- `--std-multiplier` (float, default: 1.5): Higher values = more selective
- Additional options in code: `use_median`, `min_threshold`, `max_threshold`

**Best for:** General purpose, good starting point

**Example:**
```bash
# Conservative (fewer bright pixels)
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 2.5

# Aggressive (more bright pixels)
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 0.5
```

### 2. Adaptive Method (`--method adaptive`)
Automatically adjusts based on image characteristics (contrast, brightness distribution).

**Parameters:**
- `--sensitivity` (float, 0.0-1.0, default: 0.5): Overall sensitivity
  - 0.0 = very selective
  - 1.0 = very inclusive

**Best for:** Images with varying contrast levels

**Example:**
```bash
# High sensitivity
python3 statistical_light_extractor.py image.jpg --method adaptive --sensitivity 0.8

# Low sensitivity
python3 statistical_light_extractor.py image.jpg --method adaptive --sensitivity 0.2
```

### 3. Z-Score Method (`--method zscore`)
Uses statistical z-scores to identify outliers (bright pixels).

**Parameters:**
- `--z-threshold` (float, default: 1.5): Z-score threshold
  - 1.0 = captures ~16% of brightest pixels
  - 2.0 = captures ~2.5% of brightest pixels
  - 3.0 = captures ~0.1% of brightest pixels

**Best for:** Statistical outlier detection, robust to noise

**Example:**
```bash
# Very selective
python3 statistical_light_extractor.py image.jpg --method zscore --z-threshold 2.5

# More inclusive
python3 statistical_light_extractor.py image.jpg --method zscore --z-threshold 1.0
```

### 4. IQR Method (`--method iqr`)
Uses Interquartile Range to identify outliers.

**Parameters:**
- `--iqr-multiplier` (float, default: 1.5): IQR multiplier
  - Higher values = more selective
  - Standard outlier detection uses 1.5

**Best for:** Robust outlier detection, less sensitive to extreme values

**Example:**
```bash
# Standard outlier detection
python3 statistical_light_extractor.py image.jpg --method iqr --iqr-multiplier 1.5

# More aggressive
python3 statistical_light_extractor.py image.jpg --method iqr --iqr-multiplier 1.0
```

### 5. Custom Method (`--method custom`)
Flexible formula with customizable parameters.

**Formula:** `threshold = (base_value + offset) + multiplier × (std ^ power)`

**Parameters (code only):**
- `base_method`: "mean", "median", "q75", "q90", "q95"
- `multiplier`: Multiplier for the spread measure
- `offset`: Offset to add to base value
- `power`: Power to raise std to (1.0=linear, 0.5=sqrt, 2.0=quadratic)

**Best for:** Fine-tuning and experimental purposes

### 6. Parameter Sweep (`--method sweep`)
Automatically tests multiple parameter values to find optimal settings.

**Example:**
```bash
python3 statistical_light_extractor.py image.jpg --method sweep
```

## 💾 Output Files

### Specifying Output Location
You can specify any output directory using the `--output` parameter:

```bash
# Save to current directory's 'output' folder
python3 statistical_light_extractor.py image.jpg --output ./output/

# Save to specific absolute path
python3 statistical_light_extractor.py image.jpg --output /home/user/my_results/

# Save to relative path
python3 statistical_light_extractor.py image.jpg --output ../results/experiment1/

# Save with custom filename
python3 statistical_light_extractor.py image.jpg --output ./output/ --filename my_experiment_result
```

### Output File Format
When using `--output` parameter, the script saves:

#### **Default filename pattern:**
- **Pattern**: `{image_name}_statistical_{method}.png`
- **Example**: `overfocus_statistical_basic.png`

#### **Custom filename (using `--filename`):**
- **Pattern**: `{custom_name}.png`
- **Example**: `my_experiment_result.png`

#### **File properties:**
- **Format**: PNG image where white pixels = detected light areas
- **Location**: Automatically created in specified directory

**⚠️ Note:** If no `--output` is specified, no files are saved (results only shown in terminal)

## 📊 Understanding Results

### Console Output
```
Image Statistics:
  mean: 105.80
  std: 22.89
  median: 105.00
  dynamic_range: 255.00

Extraction Results:
  Method: statistical_basic
  Parameters: {'std_multiplier': 1.5}
  Threshold: 140.1
  Light pixels: 11,534 (4.4%)
```

### Key Metrics
- **Threshold**: Pixel intensity cutoff value
- **Light pixels**: Number of pixels classified as "light"
- **Percentage**: Proportion of image classified as light

## 🎯 Choosing the Right Method and Parameters

### By Image Type

**High Contrast Images** (clear bright/dark separation):
```bash
# Use basic method with moderate multiplier
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 1.5
```

**Low Contrast Images** (subtle lighting differences):
```bash
# Use adaptive method with high sensitivity
python3 statistical_light_extractor.py image.jpg --method adaptive --sensitivity 0.7
```

**Noisy Images**:
```bash
# Use robust z-score method
python3 statistical_light_extractor.py image.jpg --method zscore --z-threshold 2.0
```

### By Desired Output

**Want ~5% of image as light pixels:**
```bash
# Start with basic method, std_multiplier 1.5-2.0
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 1.8
```

**Want only brightest highlights:**
```bash
# Use high threshold
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 3.0
```

**Want to capture subtle lighting:**
```bash
# Use lower threshold
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 0.5
```

## 📁 Output Path Examples

### Different Output Locations
```bash
# Save to default 'output' directory in current folder
python3 statistical_light_extractor.py image.jpg --output ./output/

# Save to specific project folder
python3 statistical_light_extractor.py image.jpg --output /home/jimmy/projects/results/

# Save to dated folder
python3 statistical_light_extractor.py image.jpg --output ./results/$(date +%Y-%m-%d)/

# Save to method-specific folder
python3 statistical_light_extractor.py image.jpg --method basic --output ./results/basic_method/
python3 statistical_light_extractor.py image.jpg --method adaptive --output ./results/adaptive_method/

# Save with absolute path
python3 statistical_light_extractor.py image.jpg --output /home/jimmy/code/2Dto3D_2/custom_output/
```

### Organizing Results by Parameters
```bash
# Organize by multiplier values with custom filenames
python3 statistical_light_extractor.py image.jpg --std-multiplier 1.0 --output ./results/ --filename mult_1_0_result
python3 statistical_light_extractor.py image.jpg --std-multiplier 1.5 --output ./results/ --filename mult_1_5_result
python3 statistical_light_extractor.py image.jpg --std-multiplier 2.0 --output ./results/ --filename mult_2_0_result

# Organize by image type with descriptive names
python3 statistical_light_extractor.py bright_image.jpg --output ./results/ --filename bright_image_analysis
python3 statistical_light_extractor.py dark_image.jpg --output ./results/ --filename dark_image_analysis

# Organize experiments with timestamps and descriptions
python3 statistical_light_extractor.py image.jpg --method basic --output ./results/ --filename "experiment_1_basic_default"
python3 statistical_light_extractor.py image.jpg --method adaptive --output ./results/ --filename "experiment_2_adaptive_high_sens"
```

### Custom Filename Examples
```bash
# Descriptive experiment names
python3 statistical_light_extractor.py image.jpg --filename "test_conservative_threshold"
python3 statistical_light_extractor.py image.jpg --filename "trial_3_aggressive_detection"

# Project-specific naming
python3 statistical_light_extractor.py photo.jpg --filename "project_alpha_mask_v1"
python3 statistical_light_extractor.py scan.jpg --filename "medical_scan_bright_regions"

# Include parameters in filename for reference
python3 statistical_light_extractor.py image.jpg --std-multiplier 2.5 --filename "analysis_std2p5_conservative"
```

## 🔧 Advanced Usage Examples

### 1. Batch Processing
```bash
# Process multiple images
for img in images/*.jpg; do
    python3 statistical_light_extractor.py "$img" --method basic --std-multiplier 1.5 --output results/
done
```

### 2. Parameter Optimization
```bash
# Find optimal parameters
python3 statistical_light_extractor.py image.jpg --method sweep

# Test specific parameters
python3 multiplier_comparison.py image.jpg "0.5,1.0,1.5,2.0,2.5"
```

### 3. Different Methods Comparison
```bash
# Test all methods
python3 statistical_light_extractor.py image.jpg --method basic --output results/
python3 statistical_light_extractor.py image.jpg --method adaptive --output results/
python3 statistical_light_extractor.py image.jpg --method zscore --output results/
python3 statistical_light_extractor.py image.jpg --method iqr --output results/
```

## 🐍 Python API Usage

### Basic Usage
```python
from statistical_light_extractor import StatisticalLightExtractor

# Initialize
extractor = StatisticalLightExtractor("path/to/image.jpg")

# Get image statistics
stats = extractor.get_image_statistics()
print(f"Mean brightness: {stats['mean']:.1f}")

# Extract light pixels
mask, info = extractor.extract_statistical_basic(std_multiplier=1.5)
print(f"Light percentage: {info['light_percentage']:.1f}%")

# Clean up the mask
cleaned_mask = extractor.morphological_cleanup(mask)

# Save result
cv2.imwrite("output_mask.png", cleaned_mask)
```

### Advanced Usage
```python
# Compare multiple methods
methods_and_params = [
    ("basic", {"std_multiplier": 1.5}),
    ("adaptive", {"sensitivity": 0.5}),
    ("zscore", {"z_threshold": 2.0})
]

extractor.visualize_comparison(methods_and_params, "comparison.png")

# Parameter sweep
sweep_results = extractor.parameter_sweep("basic", {"std_multiplier": [1.0, 1.5, 2.0, 2.5]})
best_params = sweep_results['best_result']['parameters']
```

## 🛠️ Dependencies

```bash
pip install opencv-python numpy matplotlib
```

## ⚠️ Common Issues and Solutions

### Issue: No output file generated
**Solution:** Make sure to use `--output` parameter
```bash
python3 statistical_light_extractor.py image.jpg --output ./results/
```

### Issue: Too many/few bright pixels detected
**Solutions:**
- **Too many**: Increase `std_multiplier` or decrease `sensitivity`
- **Too few**: Decrease `std_multiplier` or increase `sensitivity`

### Issue: Image not found
**Solution:** Check file path and format (supports common formats: jpg, png, bmp, etc.)

### Issue: Poor results on specific image types
**Solutions:**
- **Dark images**: Try `adaptive` method with high sensitivity
- **High contrast**: Use `basic` method with moderate multiplier
- **Noisy images**: Use `zscore` method for robustness

## 📈 Performance Tips

1. **Start with parameter sweep** to find optimal settings
2. **Use basic method first** as a baseline
3. **Compare multiple methods** for best results
4. **Adjust parameters incrementally** (0.1-0.5 steps)
5. **Visual inspection** is crucial - numbers don't tell the whole story

## 🔍 Example Workflows

### Workflow 1: New Image Analysis
```bash
# Step 1: Get baseline with default settings
python3 statistical_light_extractor.py image.jpg --output results/

# Step 2: Run parameter sweep
python3 statistical_light_extractor.py image.jpg --method sweep

# Step 3: Test recommended parameters
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 2.0 --output results/

# Step 4: Compare with other methods
python3 statistical_light_extractor.py image.jpg --method adaptive --sensitivity 0.6 --output results/
```

### Workflow 2: Fine-tuning for Specific Target
```bash
# Goal: Extract exactly 5% of image as light pixels

# Step 1: Test range of multipliers
python3 multiplier_comparison.py image.jpg "1.0,1.2,1.4,1.6,1.8,2.0"

# Step 2: Narrow down based on results
python3 multiplier_comparison.py image.jpg "1.4,1.5,1.6"

# Step 3: Final extraction with optimal parameter
python3 statistical_light_extractor.py image.jpg --method basic --std-multiplier 1.5 --output final/
```

---

## 📞 Support

For issues or questions:
1. Check image format and path
2. Verify dependencies are installed
3. Try different methods if results are unsatisfactory
4. Use parameter sweep for automatic optimization

Happy light extraction! 🌟
