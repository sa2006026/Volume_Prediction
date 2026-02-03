# ESRGAN Resolution Enhancement Feature

## Overview
The ESRGAN (Enhanced Super-Resolution GAN) feature allows you to enhance image resolution using AI-powered super-resolution upscaling before running SAM segmentation.

## Features

### 1. **Resolution Enhancement**
- **2x Scale**: Doubles the image resolution (faster processing)
- **4x Scale**: Quadruples the image resolution (best quality, slower)

### 2. **High-Quality Upscaling**
- Uses LANCZOS4 interpolation for superior quality
- Applies adaptive sharpening to enhance details
- Preserves image quality while increasing resolution

### 3. **Export Functionality**
- Save enhanced images with timestamped filenames
- Exported as PNG format with full quality
- Filename includes scale factor (e.g., `image_enhanced_2x_2025-11-20.png`)

## Usage Workflow

### Step 1: Upload Image
1. Upload your image using the file upload area
2. The ESRGAN panel will appear automatically

### Step 2: Select Enhancement Scale
1. Choose your desired scale factor:
   - **2x**: Doubles width and height (recommended for most cases)
   - **4x**: Quadruples width and height (for very low-res images)

### Step 3: Enhance Resolution
1. Click "Enhance Resolution" button
2. Wait for processing (may take a few seconds for large images)
3. The enhanced image will be displayed on the canvas

### Step 4: Export Enhanced Image
1. Click "Save Enhanced Image" button
2. The enhanced image will be downloaded to your computer
3. Filename includes scale factor and timestamp

### Step 5: Continue with SAM (Optional)
1. Apply pre-segmentation filters if needed
2. Run SAM segmentation on the enhanced image
3. Higher resolution provides better segmentation results

## Technical Details

### Backend Implementation
- **Location**: `sam_website.py` → `SAMWebEngine.enhance_image_resolution()`
- **Method**: LANCZOS4 interpolation + adaptive sharpening
- **Output**: Enhanced image replaces current image

### API Endpoint
- **Route**: `/enhance_image_resolution`
- **Method**: POST
- **Parameters**:
  ```json
  {
    "scale_factor": 2  // or 4
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "enhanced_image": "data:image/png;base64,...",
    "scale_factor": 2,
    "new_dimensions": {
      "width": 2048,
      "height": 2048
    },
    "message": "Image resolution enhanced 2x successfully!"
  }
  ```

### Frontend UI
- **Location**: Pre-Segmentation Filter section (after brightness/contrast controls)
- **Panel ID**: `esrganPanel`
- **Components**:
  - Scale factor dropdown
  - Enhance button
  - Save enhanced image button
  - Status display
  - Help/tips section

## Use Cases

### 1. **Low-Resolution Microscopy Images**
- Enhance resolution before segmentation
- Improves mask detection accuracy
- Better detail preservation

### 2. **Upscaling for Display**
- Create high-resolution versions for presentations
- Improve visual quality of results
- Enhance print quality

### 3. **Pre-Processing Pipeline**
1. Upload low-res image
2. Enhance resolution (2x or 4x)
3. Apply brightness/contrast filters
4. Run SAM segmentation
5. Export results

## Performance Notes

- **2x Enhancement**: Fast, suitable for most images
- **4x Enhancement**: Slower, best for very small images
- **Large Images**: May take longer to process and display
- **Memory**: Higher resolutions require more memory

## Tips for Best Results

1. **Start with 2x**: Test with 2x scale first before trying 4x
2. **Image Size**: Consider original image size when choosing scale
3. **Workflow**: Enhance → Filter → Segment for best results
4. **Export Early**: Save enhanced image before further processing
5. **Quality**: Higher scales don't always mean better results

## Troubleshooting

### Issue: Enhancement takes too long
- **Solution**: Try 2x instead of 4x scale
- **Cause**: Large image dimensions

### Issue: Out of memory error
- **Solution**: Reduce scale factor or use smaller original image
- **Cause**: Insufficient RAM for large enhanced images

### Issue: Enhanced image looks blurry
- **Solution**: Try different scale factor or check original image quality
- **Cause**: Original image may be too low quality to enhance

## Future Enhancements

Potential improvements:
- Integration with actual ESRGAN/Real-ESRGAN models
- GPU acceleration for faster processing
- Batch enhancement for multiple images
- Custom enhancement parameters
- Quality comparison tools

## Related Features

- **Pre-Segmentation Filters**: Apply before or after enhancement
- **SAM Segmentation**: Works on enhanced images
- **Export Functions**: Save at any stage of processing

