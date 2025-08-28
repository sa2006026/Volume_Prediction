#!/usr/bin/env python3
"""
Interactive SAM Segmentation Server
Web interface for SAM-based image segmentation with interactive mask removal
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask.json.provider import DefaultJSONProvider
import cv2
import numpy as np
import os
import base64
import io
from PIL import Image
import json
from datetime import datetime
import sys
from werkzeug.utils import secure_filename
import tempfile
import time
import torch

# Add the parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sam_analyzer import SAMAnalyzer

# Try to import advanced SAM configuration from mask_grouping_server
try:
    mask_grouping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'mask_grouping_server')
    sys.path.append(mask_grouping_path)
    from sam_config import SAMConfig, SAMBackend
    from onnx_sam_wrapper import load_onnx_sam
    try:
        from tensorrt_sam_wrapper_cpu import load_tensorrt_sam
        print("🔧 Using CPU-compatible TensorRT wrapper for optimal performance")
    except ImportError:
        try:
            from tensorrt_sam_wrapper import load_tensorrt_sam
            print("⚠️ Using standard TensorRT wrapper - may encounter CUDA conflicts")
        except ImportError:
            load_tensorrt_sam = None
            print("❌ TensorRT wrapper not available")
    
    ADVANCED_SAM_AVAILABLE = True
    print("✅ Advanced SAM features available (ONNX/TensorRT)")
except ImportError as e:
    ADVANCED_SAM_AVAILABLE = False
    print(f"⚠️ Advanced SAM features not available: {e}")
    SAMConfig = None
    SAMBackend = None

class NumpyJSONProvider(DefaultJSONProvider):
    """Custom JSON provider to handle numpy arrays and other special types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__, template_folder='../../templates')
app.json = NumpyJSONProvider(app)

# Global variables for SAM models
sam_analyzers = {}
current_backend = SAMBackend.PYTORCH if ADVANCED_SAM_AVAILABLE else None

def get_sam_analyzer():
    """Get or create SAM analyzer"""
    global sam_analyzers
    
    if 'default' not in sam_analyzers:
        print("🔄 Loading SAM model...")
        
        # Use the existing SAMAnalyzer which automatically finds and loads the best model
        sam_analyzers['default'] = SAMAnalyzer()
        
        print("✅ SAM model loaded successfully")
    
    return sam_analyzers['default']



def process_image_data(image_data):
    """Process base64 image data and return numpy array"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        return image_array
        
    except Exception as e:
        print(f"❌ Error processing image data: {e}")
        return None

def numpy_to_base64(image_array, format='PNG'):
    """Convert numpy array to base64 string"""
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/{format.lower()};base64,{image_base64}"
        
    except Exception as e:
        print(f"❌ Error converting image to base64: {e}")
        return None

@app.route('/')
def index():
    """Serve the interactive SAM interface"""
    return render_template('interactive_sam.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check if SAM model is available
        sam_loaded = 'default' in sam_analyzers
        
        return jsonify({
            'status': 'healthy',
            'sam_loaded': sam_loaded,
            'model_type': 'auto-detected',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'sam_loaded': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/segment', methods=['POST'])
def segment():
    """Perform SAM segmentation on uploaded image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        start_time = time.time()
        
        # Extract parameters
        sam_model = data.get('sam_model', 'b')
        crop_n_layers = data.get('crop_n_layers', 1)
        points_per_side = data.get('points_per_side', 32)
        pred_iou_thresh = data.get('pred_iou_thresh', 0.88)
        stability_score_thresh = data.get('stability_score_thresh', 0.95)
        crop_n_points_downscale_factor = data.get('crop_n_points_downscale_factor', 1)
        min_mask_region_area = data.get('min_mask_region_area', 0)
        
        print(f"🔄 Starting SAM segmentation with model: {sam_model}")
        print(f"   Parameters: crop_n_layers={crop_n_layers}, points_per_side={points_per_side}")
        
        # Process image
        image_array = process_image_data(data['image'])
        if image_array is None:
            return jsonify({'success': False, 'error': 'Failed to process image data'})
        
        # Get SAM analyzer
        sam_analyzer = get_sam_analyzer()
        
        # Load image into analyzer
        sam_analyzer.load_image(image_array)
        
        # Generate masks using SAM segmentation
        try:
            mask_stats = sam_analyzer.segment_droplets(method="sam")
            masks = sam_analyzer.masks
            
            print(f"✅ Generated {len(masks)} masks")
            
        except Exception as e:
            print(f"❌ Error during mask generation: {e}")
            return jsonify({'success': False, 'error': f'Mask generation failed: {str(e)}'})
        
        # Process masks and create visualization
        processed_masks = []
        combined_mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            # Get statistics from the SAMAnalyzer
            if i < len(mask_stats):
                stats = mask_stats[i]
                bbox = stats['bounding_box']  # [x, y, w, h]
                area = stats['area']
                stability_score = stats.get('circularity', 0.0)  # Use circularity as stability measure
            else:
                # Fallback calculation
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    bbox = [x, y, w, h]
                else:
                    bbox = [0, 0, mask.shape[1], mask.shape[0]]
                area = np.sum(mask > 0)
                stability_score = 0.0
            
            # Add mask to visualization
            combined_mask[mask > 0] = (i % 255) + 1
            
            processed_masks.append({
                'id': i,
                'bbox': bbox,
                'area': int(area),
                'stability_score': float(stability_score)
            })
        
        # Create segmented image visualization
        segmented_image = create_segmented_visualization(image_array, combined_mask)
        segmented_image_b64 = numpy_to_base64(segmented_image)
        
        # Convert original image to base64 for storage
        original_image_b64 = numpy_to_base64(image_array)
        
        processing_time = time.time() - start_time
        
        result = {
            'success': True,
            'total_masks': len(masks),
            'processing_time': processing_time,
            'segmented_image': segmented_image_b64,
            'original_image': original_image_b64,
            'masks': processed_masks,
            'configuration': {
                'sam_model': 'auto-detected',
                'method': 'sam',
                'filters_applied': 'area, edge, circularity'
            }
        }
        
        print(f"✅ Segmentation completed in {processing_time:.1f}s")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in segmentation endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/remove_masks', methods=['POST'])
def remove_masks():
    """Remove specified masks and regenerate visualization"""
    try:
        data = request.get_json()
        
        if not data or 'original_image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Process original image
        original_image = process_image_data(data['original_image'])
        if original_image is None:
            return jsonify({'success': False, 'error': 'Failed to process original image'})
        
        remaining_masks = data.get('masks', [])
        
        print(f"🔄 Regenerating visualization with {len(remaining_masks)} remaining masks")
        
        # Create new combined mask with only remaining masks
        combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        for i, mask_data in enumerate(remaining_masks):
            # Note: This is a simplified approach. In a full implementation,
            # you would need to store the actual mask arrays and regenerate
            # the visualization properly.
            bbox = mask_data['bbox']
            x, y, w, h = bbox
            combined_mask[y:y+h, x:x+w] = (i % 255) + 1
        
        # Create updated segmented image
        updated_image = create_segmented_visualization(original_image, combined_mask)
        updated_image_b64 = numpy_to_base64(updated_image)
        
        result = {
            'success': True,
            'updated_image': updated_image_b64,
            'remaining_masks': len(remaining_masks)
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in remove_masks endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_mask_preview', methods=['POST'])
def get_mask_preview():
    """Get mask preview for hover display at specific coordinates"""
    try:
        # Get SAM analyzer
        sam_analyzer = get_sam_analyzer()
        
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        if sam_analyzer is None or not hasattr(sam_analyzer, 'masks') or not sam_analyzer.masks:
            return jsonify({
                'success': True,
                'has_mask': False,
                'preview_image': None,
                'mask_info': None
            })
        
        # Find which mask contains this point
        for i, mask in enumerate(sam_analyzer.masks):
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
                # Generate preview for this mask
                preview_image = sam_analyzer.create_mask_preview(i, preview_size=(200, 200))
                
                if preview_image is not None:
                    # Convert to base64
                    preview_base64 = numpy_to_base64(preview_image)
                    
                    # Get mask info
                    if i < len(sam_analyzer.mask_statistics):
                        mask_info = sam_analyzer.mask_statistics[i].copy()
                        mask_info['mask_id'] = i
                        mask_info['state'] = (sam_analyzer.mask_states[i] 
                                            if hasattr(sam_analyzer, 'mask_states') and i < len(sam_analyzer.mask_states) else 'active')
                    else:
                        mask_info = {'mask_id': i, 'state': 'active'}
                    
                    return jsonify({
                        'success': True,
                        'has_mask': True,
                        'preview_image': f"data:image/png;base64,{preview_base64}",
                        'mask_info': mask_info,
                        'coordinates': {'x': x, 'y': y}
                    })
        
        return jsonify({
            'success': True,
            'has_mask': False,
            'preview_image': None,
            'mask_info': None,
            'coordinates': {'x': x, 'y': y}
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def create_segmented_visualization(image, mask):
    """Create a visualization of the segmented image with colored masks"""
    try:
        # Create colored overlay
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        # Generate colors for different mask regions
        unique_values = np.unique(mask)
        for value in unique_values:
            if value == 0:  # Background
                continue
            
            # Generate a color based on the mask value
            color = [
                int((value * 67) % 255),
                int((value * 131) % 255), 
                int((value * 199) % 255)
            ]
            
            colored_mask[mask == value] = color
        
        # Blend with original image
        alpha = 0.6
        result = image.copy()
        
        # Apply colored overlay where masks exist
        mask_exists = mask > 0
        result[mask_exists] = (
            alpha * colored_mask[mask_exists] + 
            (1 - alpha) * result[mask_exists]
        ).astype(np.uint8)
        
        return result
        
    except Exception as e:
        print(f"❌ Error creating segmented visualization: {e}")
        return image

if __name__ == '__main__':
    print("🚀 Starting Interactive SAM Segmentation Server...")
    
    # Pre-load the SAM model
    try:
        get_sam_analyzer()  # Load the auto-detected model
        print("✅ SAM model pre-loaded")
    except Exception as e:
        print(f"⚠️ Failed to pre-load SAM model: {e}")
    
    app.run(host='0.0.0.0', port=5004, debug=True)
