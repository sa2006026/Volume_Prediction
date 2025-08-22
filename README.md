# 2D to 3D Image Processing Project

This project provides various tools and algorithms for extracting light intensity pixels from images, analyzing droplet rings, and performing 2D to 3D conversions.

## 📁 Project Structure

```
2Dto3D_2/
├── src/                           # Main source code
│   ├── core/                      # Core extraction algorithms
│   │   ├── statistical_light_extractor.py    # Statistical light extraction
│   │   ├── adaptive_light_extraction.py      # Adaptive light extraction
│   │   ├── high_coverage_extractor.py        # High coverage extraction
│   │   ├── circle_extractor.py               # Circle detection
│   │   └── droplet_ring_predictor.py         # Ring prediction
│   ├── analysis/                  # Analysis and measurement tools
│   │   ├── max_distance_finder.py            # Distance analysis
│   │   ├── simple_max_distance.py            # Simple distance analysis
│   │   ├── ring_width_analyzer.py            # Ring width analysis
│   │   ├── simple_ring_analyzer.py           # Simple ring analysis
│   │   └── circle_drawer.py                  # Circle visualization
│   ├── utils/                     # Utility functions (empty)
│   └── web/                       # Web server components
│       ├── pixel_removal_server.py           # Interactive web server
│       └── test_server.py                    # Server testing
├── tools/                         # Standalone tools and demos
│   ├── demo.py                    # Basic usage demonstration
│   ├── simple_statistical_usage.py           # Simple usage examples
│   ├── parameter_tuning_demo.py              # Parameter tuning demo
│   └── multiplier_comparison.py              # Compare multipliers
├── tests/                         # Test scripts
│   ├── test_extraction.py         # Extraction testing
│   ├── overfocus_1_multiplier_test.py        # Overfocus testing
│   └── adaptive_multiplier_test.py           # Adaptive testing
├── docs/                          # Documentation
│   ├── USER_MANUAL.md             # User manual
│   ├── EXTRACTION_METHODS_EXPLAINED.md      # Technical explanations
│   └── ADAPTIVE_EXTRACTION_EXPLAINED.md     # Adaptive methods
├── data/                          # Input data
│   ├── input/                     # Input images
│   │   ├── BF image/              # Bright field images
│   │   └── Fluorescent image/     # Fluorescent images
│   └── sample/                    # Sample images (empty)
├── results/                       # Generated results (organized by type)
├── output/                        # General output files
├── custom_output/                 # Custom output files
├── templates/                     # Web templates
│   └── pixel_removal.html         # Pixel removal interface
├── config/                        # Configuration files
│   └── server.log                 # Server logs
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Core Extraction Algorithms

```bash
# Statistical light extraction
python3 src/core/statistical_light_extractor.py data/input/overfocus.jpg --method basic --std-multiplier 2.0

# Adaptive light extraction
python3 src/core/adaptive_light_extraction.py data/input/overfocus.jpg --sensitivity 0.5

# High coverage extraction
python3 src/core/high_coverage_extractor.py
```

### Analysis Tools

```bash
# Distance analysis
python3 src/analysis/max_distance_finder.py data/input/overfocus.jpg

# Ring analysis
python3 src/analysis/ring_width_analyzer.py data/input/overfocus.jpg

# Circle detection
python3 src/core/circle_extractor.py data/input/overfocus.jpg
```

### Web Interface

```bash
# Start the interactive pixel removal server
python3 src/web/pixel_removal_server.py

# Test the server
python3 src/web/test_server.py
```

### Demos and Tools

```bash
# Basic usage demo
python3 tools/demo.py

# Parameter tuning demonstration
python3 tools/parameter_tuning_demo.py

# Compare different multipliers
python3 tools/multiplier_comparison.py
```

## 📖 Documentation

- **[User Manual](docs/USER_MANUAL.md)** - Comprehensive usage guide
- **[Extraction Methods](docs/EXTRACTION_METHODS_EXPLAINED.md)** - Technical explanation of extraction methods
- **[Adaptive Methods](docs/ADAPTIVE_EXTRACTION_EXPLAINED.md)** - Adaptive extraction techniques

## 🔧 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Dependencies include:
- opencv-python>=4.5.0
- numpy>=1.19.0
- matplotlib>=3.3.0
- scikit-learn>=0.24.0
- scipy>=1.6.0

## 📊 Output Structure

- **results/** - Organized by processing type (e.g., `circle_extraction/`, `ring_analysis/`)
- **output/** - General output files
- **custom_output/** - User-defined custom outputs

## 🧪 Testing

Run tests to verify functionality:

```bash
# Basic extraction test
python3 tests/test_extraction.py

# Multiplier tests
python3 tests/overfocus_1_multiplier_test.py
python3 tests/adaptive_multiplier_test.py
```

## 🌐 Web Interface

The project includes an interactive web interface for pixel removal:

1. Start the server: `python3 src/web/pixel_removal_server.py`
2. Open your browser to `http://localhost:5000`
3. Upload an image and interactively select regions to remove

## 📈 Image Paths

All scripts now reference images in the `data/input/` directory. Update image paths in scripts if your images are located elsewhere.
