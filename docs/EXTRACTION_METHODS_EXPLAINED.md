# Statistical Light Extraction Methods - Technical Explanation

## 🧠 Overview

The **Statistical Light Extractor** uses statistical analysis to identify and extract bright/light regions from images. Instead of using fixed thresholds, it analyzes the pixel intensity distribution and applies mathematical formulas to determine optimal cutoff points.

## 🔬 Core Concept

All methods follow this general approach:
1. **Analyze** the image's grayscale intensity distribution
2. **Calculate** statistical measures (mean, std, percentiles, etc.)
3. **Determine** a threshold value using mathematical formulas
4. **Apply** the threshold to create a binary mask (bright vs. dark pixels)

---

## 📊 Method 1: Basic Statistical (`extract_statistical_basic`)

### 🎯 **Core Formula:**
```
threshold = base_value + (std_multiplier × standard_deviation)
```

### 🔧 **How It Works:**
1. **Calculate image statistics** (mean, standard deviation)
2. **Choose base value**: mean or median
3. **Add deviation**: base + (multiplier × std)
4. **Apply threshold**: pixels ≥ threshold = light

### 📈 **Mathematical Example:**
```
Image stats: mean=105.8, std=22.9
std_multiplier=1.5

threshold = 105.8 + (1.5 × 22.9) = 140.15
→ Pixels with intensity ≥ 140 are classified as "light"
```

### ⚙️ **Parameters:**
- **`std_multiplier`**: Controls selectivity
  - **0.5**: Very inclusive (captures ~30% of image)
  - **1.5**: Balanced (captures ~4-8% of image)  
  - **3.0**: Very selective (captures ~0.1% of image)
- **`use_median`**: More robust against outliers
- **`min/max_threshold`**: Safety bounds

### ✅ **Best For:**
- General purpose extraction
- Images with normal brightness distribution
- When you want predictable behavior

---

## 📊 Method 2: Adaptive Statistical (`extract_statistical_adaptive`)

### 🎯 **Core Formula:**
```
threshold = percentile_base + (adaptive_multiplier × std)
where adaptive_multiplier = 2.0 × (1.0 - effective_sensitivity)
```

### 🔧 **How It Works:**
1. **Analyze image contrast** (contrast_ratio = std/mean)
2. **Adjust sensitivity** based on contrast level
3. **Use percentile** instead of mean as base
4. **Calculate adaptive threshold** that responds to image characteristics

### 📈 **Mathematical Example:**
```
Image stats: percentile_75=118, std=22.9, contrast_ratio=0.22
sensitivity=0.5, contrast_boost=1.0

Low contrast detected (0.22 < 0.3)
→ effective_sensitivity = 0.5 × 1.0 = 0.5
→ std_multiplier = 2.0 × (1.0 - 0.5) = 1.0

threshold = 118 + (1.0 × 22.9) = 140.9
```

### ⚙️ **Parameters:**
- **`sensitivity`**: Overall inclusiveness (0.0-1.0)
  - **0.0**: Maximum selectivity
  - **1.0**: Maximum inclusiveness
- **`contrast_boost`**: Extra sensitivity for low-contrast images
- **`percentile_base`**: Base percentile (75=upper quartile, 90=top decile)

### ✅ **Best For:**
- Images with varying contrast levels
- Automatic parameter adjustment
- When you don't know optimal settings

---

## 📊 Method 3: Z-Score Statistical (`extract_statistical_zscore`)

### 🎯 **Core Formula:**
```
z_score = |pixel_value - center| / scale
threshold condition: (pixel > center) AND (z_score ≥ z_threshold)
```

### 🔧 **How It Works:**
1. **Calculate center** (mean or median) and **scale** (std or MAD)
2. **Compute z-scores** for each pixel (how many standard deviations from center)
3. **Apply threshold** to z-scores, only for pixels above center
4. **Create binary mask** based on statistical outliers

### 📈 **Mathematical Example:**
```
Robust mode: center=median=105, MAD=16.3, scale=24.15
z_threshold=1.5

For a pixel with intensity 150:
z_score = |150 - 105| / 24.15 = 1.86
Since pixel > center AND z_score ≥ 1.5 → classified as "light"
```

### ⚙️ **Parameters:**
- **`z_threshold`**: Standard deviations from center
  - **1.0**: Captures ~16% of extreme values
  - **2.0**: Captures ~2.5% of extreme values  
  - **3.0**: Captures ~0.1% of extreme values
- **`robust`**: Use median+MAD instead of mean+std for outlier resistance

### ✅ **Best For:**
- Statistical outlier detection
- Noisy images (robust mode)
- When you want statistically rigorous thresholds

---

## 📊 Method 4: IQR Statistical (`extract_statistical_iqr`)

### 🎯 **Core Formula:**
```
IQR = Q75 - Q25
upper_threshold = Q75 + (iqr_multiplier × IQR)
```

### 🔧 **How It Works:**
1. **Calculate quartiles** (Q25, Q75) - the 25th and 75th percentiles
2. **Compute IQR** (Interquartile Range) = Q75 - Q25
3. **Determine outlier threshold** using classic outlier detection formula
4. **Classify pixels** above threshold as bright outliers

### 📈 **Mathematical Example:**
```
Image stats: Q25=94, Q75=118
IQR = 118 - 94 = 24
iqr_multiplier=1.5

upper_threshold = 118 + (1.5 × 24) = 154
→ Pixels with intensity ≥ 154 are classified as "light"
```

### ⚙️ **Parameters:**
- **`iqr_multiplier`**: Outlier detection sensitivity
  - **1.5**: Standard outlier detection (most common)
  - **1.0**: More aggressive outlier detection
  - **2.0**: Conservative outlier detection
- **`upper_only`**: Only detect bright outliers (vs. both bright and dark)

### ✅ **Best For:**
- Robust outlier detection
- Images with extreme values or noise
- Classical statistical approach

---

## 📊 Method 5: Custom Statistical (`extract_statistical_custom`)

### 🎯 **Core Formula:**
```
threshold = (base_value + offset) + multiplier × (std ^ power)
```

### 🔧 **How It Works:**
1. **Choose base method** (mean, median, q75, q90, q95)
2. **Apply offset** to shift the base value
3. **Calculate powered standard deviation** (linear, square root, quadratic)
4. **Combine** with multiplier for final threshold

### 📈 **Mathematical Example:**
```
base_method="q90", offset=10, multiplier=1.2, power=0.5
Image stats: q90=131, std=22.9

threshold = (131 + 10) + 1.2 × (22.9^0.5) = 141 + 1.2 × 4.78 = 146.74
```

### ⚙️ **Parameters:**
- **`base_method`**: Starting point
  - **"mean"**: Average intensity
  - **"q75"**: Upper quartile (75th percentile)
  - **"q90"**: 90th percentile (quite bright)
  - **"q95"**: 95th percentile (very bright)
- **`multiplier`**: Scale factor for spread
- **`offset`**: Shift base value up/down
- **`power`**: Transform standard deviation
  - **1.0**: Linear (normal)
  - **0.5**: Square root (compressed)
  - **2.0**: Quadratic (amplified)

### ✅ **Best For:**
- Fine-tuning and experimentation
- Non-standard distributions
- Research and optimization

---

## 🔄 Processing Pipeline

### 1. **Image Preprocessing**
```python
# Convert to grayscale for intensity analysis
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
```

### 2. **Statistical Analysis**
```python
stats = {
    'mean': np.mean(gray_image),
    'std': np.std(gray_image),
    'median': np.median(gray_image),
    'q25': np.percentile(gray_image, 25),
    'q75': np.percentile(gray_image, 75),
    # ... more statistics
}
```

### 3. **Threshold Calculation**
Each method applies its specific formula using the calculated statistics.

### 4. **Binary Mask Creation**
```python
binary_mask = (gray_image >= threshold).astype(np.uint8) * 255
```

### 5. **Morphological Cleanup** (Optional)
```python
# Remove noise and fill gaps
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
```

---

## 🎯 Choosing the Right Method

### **By Image Characteristics:**

| Image Type | Best Method | Reasoning |
|------------|-------------|-----------|
| **High contrast** | Basic | Clear distinction between bright/dark |
| **Low contrast** | Adaptive | Automatically adjusts sensitivity |
| **Noisy** | Z-Score (robust) | Resistant to outliers |
| **Unknown** | Parameter Sweep | Tests multiple approaches |

### **By Statistical Preference:**

| Goal | Method | Why |
|------|--------|-----|
| **Simple & reliable** | Basic | Well-understood, predictable |
| **Automatic tuning** | Adaptive | Self-adjusting parameters |
| **Statistically rigorous** | Z-Score | Based on standard deviation theory |
| **Classical outlier detection** | IQR | Traditional statistical method |
| **Maximum flexibility** | Custom | Fully customizable formula |

---

## 📈 Performance Characteristics

### **Computational Complexity:**
- **Basic**: O(n) - single pass through image
- **Adaptive**: O(n) - adds percentile calculation
- **Z-Score**: O(n) - per-pixel z-score calculation
- **IQR**: O(n log n) - requires sorting for percentiles
- **Custom**: O(n) - varies by power function

### **Memory Usage:**
All methods use O(n) memory for intermediate calculations and mask storage.

### **Accuracy vs. Speed:**
- **Fastest**: Basic, Custom
- **Most Accurate**: Z-Score (robust), IQR
- **Best Balance**: Adaptive

---

## 🔍 Example Results Comparison

For the same image (`overfocus.jpg` with mean=105.8, std=22.9):

| Method | Parameters | Threshold | Light % | Use Case |
|--------|------------|-----------|---------|----------|
| Basic | std_mult=1.5 | 140.1 | 4.4% | General purpose |
| Adaptive | sens=0.5 | 131.7 | 9.4% | Auto-adjustment |
| Z-Score | z_thresh=2.0 | Variable | 4.4% | Statistical rigor |
| IQR | iqr_mult=1.5 | 154.0 | 1.2% | Outlier detection |
| Custom | q90, mult=1.0 | 131.0 | 9.8% | Fine-tuning |

This shows how different methods can produce very different results on the same image, allowing you to choose based on your specific needs.

---

## 🛠️ Implementation Notes

### **Robustness Features:**
- **Threshold clamping**: Prevents invalid values (0-255 range)
- **Division by zero protection**: Handles edge cases
- **Input validation**: Ensures parameter bounds
- **Graceful degradation**: Falls back to simpler methods if needed

### **Extension Points:**
The framework is designed to easily add new statistical methods by following the same pattern:
1. Calculate image statistics
2. Apply mathematical formula
3. Create binary mask
4. Return mask and metadata

This modular design makes it easy to experiment with new approaches while maintaining consistent interfaces and output formats.
