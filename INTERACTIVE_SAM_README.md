# Interactive SAM Segmentation with Mask Removal

A web-based interface for SAM (Segment Anything Model) image segmentation with interactive mask removal functionality.

## Features

### 🎯 Core Functionality
- **Upload Image**: Support for PNG, JPG, JPEG, TIFF, BMP formats (up to 50MB)
- **SAM Model Selection**: Choose between SAM-B (fast), SAM-L (balanced), SAM-H (best quality)
- **Parameter Control**: Adjust segmentation parameters for optimal results
- **Interactive Mask Removal**: Click on any mask to remove it from the segmentation
- **Real-time Visualization**: See changes immediately as you remove masks

### 🛠 SAM Configuration

1. **Auto-Detected Model**:
   - System automatically detects and loads the best available SAM model
   - Supports SAM-B, SAM-L, or SAM-H depending on availability
   - Priority: SAM-B (fastest) → SAM-L → SAM-H

2. **Pre-Configured Filters**:
   - `Area Filter`: Removes masks that are too small or too large
   - `Edge Filter`: Removes masks touching image borders  
   - `Circularity Filter`: Keeps only reasonably circular shapes (droplets)

### 🖱 Interactive Features

1. **Hover to Inspect**:
   - Hover over any mask to see details (ID, area, stability score)
   - Visual highlighting shows the exact mask boundaries

2. **Click to Remove**:
   - Click on any mask to remove it from the segmentation
   - Immediate visual feedback and statistics update

3. **Reset Functionality**:
   - Restore all removed masks with a single click
   - Start over with the original segmentation

4. **Download Results**:
   - Download the final segmented image with your modifications

## Installation & Setup

### Prerequisites
- Python 3.8+
- SAM model files in `/model/` directory
- Required Python packages (see requirements.txt)

### Quick Start

1. **Navigate to the project directory**:
   ```bash
   cd /home/jimmy/code/2Dto3D_2
   ```

2. **Run the server**:
   ```bash
   python3 run_interactive_sam.py
   ```

3. **Open your browser**:
   ```
   http://localhost:5004
   ```

### Manual Server Start
If you prefer to run the server directly:
```bash
python3 src/web/interactive_sam_server.py
```

## Usage Guide

### Step 1: Upload Image
- Drag and drop an image or click "Browse Files"
- Supported formats: PNG, JPG, JPEG, TIFF, BMP
- Maximum file size: 50MB

### Step 2: Review SAM Configuration
- **Model**: Automatically detected and optimized for available hardware
- **Filters**: Pre-configured for optimal droplet detection
- **Method**: Automatic mask generation with quality filtering

### Step 3: Run Segmentation
- Click "Run SAM Segmentation"
- Wait for processing (30-90 seconds depending on settings)
- View the segmented image with colored masks

### Step 4: Refine Results
- **Hover** over masks to see details
- **Click** on unwanted masks to remove them
- **Reset** to restore all masks if needed
- **Download** the final result

## Technical Details

### Backend Endpoints

- `GET /`: Serve the interactive interface
- `GET /health`: Server health check and model status
- `POST /segment`: Perform SAM segmentation
- `POST /remove_masks`: Update visualization after mask removal

### Frontend Features

- Responsive design for desktop and mobile
- Real-time progress tracking
- Interactive tooltips and highlights
- Drag-and-drop file upload
- Error handling and user feedback

### Model Support

The system supports multiple SAM model backends:
- **PyTorch**: Standard SAM implementation
- **ONNX**: Optimized inference (if available)
- **TensorRT**: GPU-accelerated inference (if available)

## File Structure

```
2Dto3D_2/
├── templates/
│   └── interactive_sam.html          # Frontend interface
├── src/web/
│   └── interactive_sam_server.py     # Backend server
├── run_interactive_sam.py            # Startup script
├── model/                            # SAM model files
└── INTERACTIVE_SAM_README.md         # This documentation
```

## Troubleshooting

### Common Issues

1. **"SAM model not loaded"**:
   - Ensure model files are in the `/model/` directory
   - Check file permissions and paths

2. **Slow processing**:
   - Use SAM-B model for faster results
   - Reduce `points_per_side` parameter
   - Reduce `crop_n_layers` to 0

3. **Out of memory errors**:
   - Resize images before upload
   - Use smaller model (SAM-B)
   - Reduce segmentation parameters

4. **Server connection errors**:
   - Check if port 5004 is available
   - Verify Python environment and dependencies

### Performance Tips

- **Model Priority**: System automatically selects SAM-B for speed, falls back to SAM-L/SAM-H
- **Optimized Settings**: Pre-configured for droplet detection with balanced speed/quality
- **Image Size**: Smaller images (< 2MB) process faster while maintaining quality

## Development

### Adding Features
The modular design allows easy extension:
- Frontend: Modify `interactive_sam.html`
- Backend: Add endpoints to `interactive_sam_server.py`
- Models: Add support in `get_sam_analyzer()` function

### API Integration
The backend provides JSON APIs that can be integrated with other applications:
```python
# Example segmentation request
{
    "image": "base64_encoded_image",
    "sam_model": "l",
    "crop_n_layers": 1,
    "points_per_side": 32,
    "stability_score_thresh": 0.95
}
```

## License & Credits

Based on the SAM (Segment Anything Model) by Meta AI Research.
Interactive interface and mask removal functionality developed for the 2Dto3D project.

---

**Happy Segmenting! 🎯✨**
