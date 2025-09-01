#!/usr/bin/env python3
"""
SAM Interactive Segmentation Website
A dedicated web interface for SAM-based image segmentation with interactive mask management
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
    """Custom JSON provider to handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
templates_dir = os.path.join(project_root, 'templates')

app = Flask(__name__, template_folder=templates_dir)
app.json = NumpyJSONProvider(app)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

class SAMWebEngine:
    """Engine for handling SAM segmentation with configurable parameters"""
    
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.image_path = None
        self.sam_analyzer = None
        self.current_model_size = "vit_b"
        self.current_crop_layers = 1
        self.current_points_per_side = 32
        self.current_backend = "pytorch"  # Default to PyTorch
        self.performance_mode = False
        self.use_gpu = True
        self.sam_config = None
        self.output_dir = "results/sam_segmentation"
        self.current_stage = "segmentation"  # Add stage management: "segmentation", "blob_analysis", "removal"
        self.stored_masks = []  # Store masks for next stage processing
        self.stored_masks_stage = None  # Track which stage masks are stored for
        self.last_adjusted_image = None  # Keep last adjusted image for downstream stages
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize advanced SAM configuration if available
        if ADVANCED_SAM_AVAILABLE:
            self._initialize_advanced_sam_config()
    
    def set_stage(self, stage: str):
        """Set the current processing stage to control interactions"""
        allowed_stages = ["segmentation", "blob_analysis", "removal"]
        if stage in allowed_stages:
            self.current_stage = stage
            print(f"🔧 Stage changed to: {stage}")
            return True
        return False
    
    def get_current_stage(self) -> str:
        """Get the current processing stage"""
        return self.current_stage
    
    def is_mask_interaction_allowed(self) -> bool:
        """Check if mask interactions are allowed in current stage"""
        # Only allow mask interactions during segmentation stage
        return self.current_stage == "segmentation"
    
    def get_stored_masks(self):
        """Get stored masks for next stage processing"""
        return self.stored_masks
    
    def has_stored_masks(self) -> bool:
        """Check if there are stored masks available"""
        return len(self.stored_masks) > 0
    
    def _initialize_advanced_sam_config(self):
        """Initialize advanced SAM configuration with ONNX/TensorRT support"""
        try:
            self.sam_config = SAMConfig(
                backend=SAMBackend.AUTO,
                model_type=self.current_model_size,
                use_gpu=self.use_gpu,
                performance_mode=self.performance_mode
            )
            print(f"🔧 Advanced SAM config initialized: {self.sam_config}")
        except Exception as e:
            print(f"⚠️ Failed to initialize advanced SAM config: {e}")
            self.sam_config = None
    
    def get_available_backends(self):
        """Get list of available SAM backends"""
        if not ADVANCED_SAM_AVAILABLE:
            return ["pytorch"]
        
        backends = ["pytorch"]
        
        # Check ONNX availability
        try:
            if self.sam_config and self.sam_config.get_onnx_model_path():
                backends.append("onnx")
        except:
            pass
        
        # Check TensorRT availability
        try:
            if load_tensorrt_sam and self.sam_config and self.sam_config.get_tensorrt_model_path():
                backends.append("tensorrt")
        except:
            pass
        
        return backends
    
    def load_image(self, image_path: str):
        """Load image for SAM processing with resolution optimization"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Check image dimensions and optimize for SAM processing
        height, width = self.original_image.shape[:2]
        print(f"📏 Loaded image: {width}x{height}")
        
        # For very large images (like 2048x2048), we might need to adjust SAM parameters
        if width > 1500 or height > 1500:
            print("🔧 Large image detected, optimizing SAM parameters for better performance")
            # Adjust default parameters for large images
            self.current_points_per_side = 64  # More points for better coverage
            self.current_crop_layers = 2       # More crop layers for large images
        
        self.current_image = self.original_image.copy()
        
        # Initialize SAM analyzer
        self.sam_analyzer = SAMAnalyzer()
        self.sam_analyzer.load_image(self.current_image.copy())
        
        return True
    
    def configure_sam_parameters(self, model_size: str = "vit_b", 
                                crop_layers: int = 1, 
                                points_per_side: int = 32,
                                backend: str = "pytorch",
                                performance_mode: bool = False,
                                use_gpu: bool = True):
        """Configure SAM model parameters with advanced backend support"""
        self.current_model_size = model_size
        self.current_crop_layers = crop_layers
        self.current_points_per_side = points_per_side
        self.current_backend = backend
        self.performance_mode = performance_mode
        self.use_gpu = use_gpu
        
        # Update advanced SAM config if available
        if ADVANCED_SAM_AVAILABLE and self.sam_config:
            try:
                # Update configuration
                backend_enum = getattr(SAMBackend, backend.upper(), SAMBackend.PYTORCH)
                self.sam_config.backend = backend_enum
                self.sam_config.model_type = model_size
                self.sam_config.use_gpu = use_gpu
                self.sam_config.performance_mode = performance_mode
                print(f"🔧 Updated SAM config: {backend} backend, {model_size} model")
            except Exception as e:
                print(f"⚠️ Failed to update advanced config: {e}")
        
        # Update SAM analyzer if it exists
        if self.sam_analyzer:
            # Reinitialize with new parameters
            self.sam_analyzer = SAMAnalyzer()
            self.sam_analyzer.load_image(self.current_image.copy())
            
            # Update the mask generator with new parameters
            if self.sam_analyzer.sam_initialized and self.sam_analyzer.sam_model:
                from segment_anything import SamAutomaticMaskGenerator
                
                # Get performance-optimized parameters
                if performance_mode:
                    stability_thresh = 0.8
                    min_area = 500
                else:
                    stability_thresh = 0.85
                    min_area = 100
                
                self.sam_analyzer.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam_analyzer.sam_model,
                    points_per_side=points_per_side,
                    crop_n_layers=crop_layers,
                    min_mask_region_area=min_area,
                    stability_score_thresh=stability_thresh,
                    box_nms_thresh=0.7
                )
        
        return True
    
    def perform_sam_segmentation(self):
        """Perform SAM segmentation with current parameters"""
        if self.sam_analyzer is None:
            raise ValueError("No image loaded")
        
        # Configure parameters before segmentation
        self.configure_sam_parameters(
            self.current_model_size, 
            self.current_crop_layers, 
            self.current_points_per_side
        )
        
        # Perform segmentation
        mask_stats = self.sam_analyzer.segment_droplets(method="sam")
        
        if not mask_stats:
            return None, None, []
        
        # Create overlay visualization
        overlay_image = self.sam_analyzer.create_mask_overlay(
            show_labels=False,
            alpha=0.3
        )
        
        # Get summary statistics
        summary = self.sam_analyzer.get_segmentation_summary()
        
        return overlay_image, summary, mask_stats
    
    def get_mask_at_point(self, x: int, y: int):
        """Get mask information at specific coordinates"""
        if self.sam_analyzer is None:
            return None
        
        return self.sam_analyzer.get_mask_at_point(x, y)
    
    def toggle_mask_at_point(self, x: int, y: int):
        """Toggle mask state at specific coordinates"""
        if self.sam_analyzer is None:
            return None
        
        # Check if mask interactions are allowed in current stage
        if not self.is_mask_interaction_allowed():
            return None, None
        
        toggle_result = self.sam_analyzer.toggle_mask_state(x, y)
        
        if toggle_result:
            # Create updated visualization
            overlay_image = self.sam_analyzer.create_mask_overlay(
                show_labels=False,
                alpha=0.3
            )
            return toggle_result, overlay_image
        
        return None, None
    
    def get_all_masks_info(self):
        """Get information about all current masks"""
        if self.sam_analyzer is None or not self.sam_analyzer.masks:
            return [], {}
        
        # Get all mask statistics with states
        all_masks = []
        for i, (mask_stats, mask_state) in enumerate(zip(
            self.sam_analyzer.mask_statistics, 
            self.sam_analyzer.mask_states
        )):
            mask_info = mask_stats.copy()
            mask_info['state'] = mask_state
            mask_info['quality_score'] = mask_stats['circularity'] * 0.6 + (1.0 - abs(1.0 - mask_stats['aspect_ratio'])) * 0.4
            all_masks.append(mask_info)
        
        # Get summary with state counts
        summary = self.sam_analyzer.get_segmentation_summary()
        active_count = sum(1 for state in self.sam_analyzer.mask_states if state == 'active')
        removed_count = sum(1 for state in self.sam_analyzer.mask_states if state == 'removed')
        summary['active_masks'] = active_count
        summary['removed_masks'] = removed_count
        
        return all_masks, summary
    
    def reset_all_masks(self):
        """Reset all masks to active state"""
        if self.sam_analyzer is None or not self.sam_analyzer.masks:
            return None
        
        # Reset all mask states
        self.sam_analyzer.mask_states = ['active'] * len(self.sam_analyzer.masks)
        
        # Create updated visualization
        overlay_image = self.sam_analyzer.create_mask_overlay(
            show_labels=False,
            alpha=0.3
        )
        
        return overlay_image
    
    def get_image_as_base64(self, image=None):
        """Convert image to base64 for web display"""
        if image is None:
            image = self.current_image
        
        if image is None:
            return None
        
        # Convert BGR to RGB for proper web display
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"
    
    def get_mask_preview_at_point(self, x: int, y: int):
        """Get mask preview at specific coordinates for hover display"""
        if self.sam_analyzer is None or not self.sam_analyzer.masks:
            return None, None
        
        # Find which mask contains this point
        for i, mask in enumerate(self.sam_analyzer.masks):
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
                # Create a focused preview showing only this specific blob
                preview_image = self._create_blob_focused_preview(i, x, y, preview_size=(200, 200))
                
                if preview_image is not None:
                    # Convert to base64
                    preview_base64 = self.get_image_as_base64(preview_image)
                    
                    # Get mask info
                    mask_info = self.sam_analyzer.mask_statistics[i].copy()
                    mask_info['mask_id'] = i
                    mask_info['state'] = (self.sam_analyzer.mask_states[i] 
                                        if i < len(self.sam_analyzer.mask_states) else 'active')
                    
                    return preview_base64, mask_info
        
        return None, None
    
    def _create_blob_focused_preview(self, mask_id: int, x: int, y: int, preview_size: tuple = (200, 200)):
        """Create a focused preview showing the specific blob being hovered"""
        if self.sam_analyzer is None or mask_id >= len(self.sam_analyzer.masks):
            return None
        
        mask = self.sam_analyzer.masks[mask_id]
        if mask is None:
            return None
        
        # Get the bounding box of the mask
        mask_stats = self.sam_analyzer.mask_statistics[mask_id]
        if not mask_stats or 'bounding_box' not in mask_stats:
            return None
        
        x1, y1, w, h = mask_stats['bounding_box']
        
        # Add some padding around the blob
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(mask.shape[1], x1 + w + 2 * padding)
        y2 = min(mask.shape[0], y1 + h + 2 * padding)
        
        # Extract the region containing the blob
        blob_region = self.current_image[y1:y2, x1:x2]
        mask_region = mask[y1:y2, x1:x2]
        
        if blob_region.size == 0:
            return None
        
        # Create a composite image showing the blob with mask overlay
        preview = blob_region.copy()
        
        # Apply mask overlay (semi-transparent red)
        mask_overlay = np.zeros_like(preview)
        mask_overlay[mask_region > 0] = [0, 0, 255]  # Red overlay for mask
        
        # Blend the overlay
        alpha = 0.3
        preview = cv2.addWeighted(preview, 1-alpha, mask_overlay, alpha, 0)
        
        # Resize to preview size
        preview = cv2.resize(preview, preview_size)
        
        return preview
    
    def apply_image_adjustments(self, brightness: int = 0, contrast: float = 1.0, intensity_threshold: int = -1):
        """Apply brightness, contrast, and intensity threshold adjustments to the image"""
        if self.original_image is None:
            return None
        
        # Start with the original image
        adjusted_image = self.original_image.copy().astype(np.float32)
        
        # Apply brightness adjustment (-100 to +100)
        if brightness != 0:
            adjusted_image = adjusted_image + brightness
        
        # Apply contrast adjustment (0.5 to 3.0, where 1.0 is no change)
        if contrast != 1.0:
            # Apply contrast around the middle value (128)
            adjusted_image = 128 + contrast * (adjusted_image - 128)
        
        # Clip values to valid range
        adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
        
        # Apply intensity threshold if specified
        if intensity_threshold > 0:
            # Create mask for pixels below threshold
            gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            threshold_mask = gray_image < intensity_threshold
            
            # Set pixels below threshold to black
            adjusted_image[threshold_mask] = [0, 0, 0]
        
        # Persist last adjusted image for next-stage processing
        self.last_adjusted_image = adjusted_image.copy()
        # Also update current image so frontend redraws align
        self.current_image = adjusted_image.copy()
        
        return adjusted_image
    
    def get_active_masks_region(self):
        """Get the combined region of all active masks"""
        if self.sam_analyzer is None or not self.sam_analyzer.masks:
            return None
        
        # Create combined mask of all active masks
        combined_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        
        for i, (mask, state) in enumerate(zip(self.sam_analyzer.masks, self.sam_analyzer.mask_states)):
            if state == 'active':
                combined_mask[mask > 0] = 255
        
        return combined_mask
    
    def apply_adjustments_to_masked_region(self, brightness: int = 0, contrast: float = 1.0, intensity_threshold: int = -1):
        """Apply image adjustments only to the active masked regions"""
        if self.original_image is None:
            return None
        
        # Get the combined active mask region
        active_mask = self.get_active_masks_region()
        if active_mask is None:
            return self.original_image.copy()
        
        # Apply adjustments to the entire image first
        adjusted_image = self.apply_image_adjustments(brightness, contrast, intensity_threshold)
        if adjusted_image is None:
            return self.original_image.copy()
        
        # Create result image starting with original
        result_image = self.original_image.copy()
        
        # Apply adjusted pixels only where masks are active
        result_image[active_mask > 0] = adjusted_image[active_mask > 0]
        
        # Persist last adjusted image
        self.last_adjusted_image = result_image.copy()
        self.current_image = result_image.copy()
        
        return result_image

    def associate_masks_to_blobs(self):
        """Find closest blob to each stored mask center on the adjusted image."""
        # Use last adjusted image if available; otherwise fall back to current image
        work_image = self.last_adjusted_image if self.last_adjusted_image is not None else self.current_image
        if work_image is None or not self.has_stored_masks():
            return None, []
        
        # Convert to grayscale and binarize to find blobs
        gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
        # Otsu threshold as default blob extraction
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours (blobs)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blob_centers = []
        for idx, cnt in enumerate(contours):
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                blob_centers.append((idx, (cx, cy)))
        
        associations = []
        # Prepare overlay visualization
        overlay = work_image.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        
        # Draw blob centers
        for blob_id, (cx, cy) in blob_centers:
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)
        
        # For each stored mask, find nearest blob by Euclidean distance
        for mask in self.stored_masks:
            # Try to read mask center from stored dict structure
            mx, my = 0, 0
            if isinstance(mask, dict):
                if 'center' in mask and isinstance(mask['center'], (list, tuple)) and len(mask['center']) >= 2:
                    mx, my = int(mask['center'][0]), int(mask['center'][1])
                elif 'center_x' in mask and 'center_y' in mask:
                    mx, my = int(float(mask['center_x'])), int(float(mask['center_y']))
            
            closest_blob_id = -1
            closest_center = (0, 0)
            closest_dist = float('inf')
            for blob_id, (cx, cy) in blob_centers:
                d = (mx - cx) ** 2 + (my - cy) ** 2
                if d < closest_dist:
                    closest_dist = d
                    closest_blob_id = blob_id
                    closest_center = (cx, cy)
            
            associations.append({
                'mask_center': {'x': mx, 'y': my},
                'blob_id': int(closest_blob_id),
                'blob_center': {'x': int(closest_center[0]), 'y': int(closest_center[1])},
                'distance_px': float(np.sqrt(closest_dist)) if closest_dist < float('inf') else None
            })
            
            # Draw mask center and connection line
            cv2.circle(overlay, (mx, my), 4, (0, 0, 255), -1)
            if closest_blob_id >= 0:
                cv2.line(overlay, (mx, my), closest_center, (255, 0, 0), 1)
        
        return overlay, associations

# Global engine instance
engine = SAMWebEngine()

@app.route('/')
def index():
    """Main SAM segmentation page"""
    return render_template('sam_website.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Upload and load an image file"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'})
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image file.'})
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(project_root, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file with secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(upload_dir, filename)
        file.save(upload_path)
        
        # Load the uploaded image
        engine.load_image(upload_path)
        image_base64 = engine.get_image_as_base64()
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'image_path': upload_path,
            'dimensions': {
                'width': int(engine.current_image.shape[1]),
                'height': int(engine.current_image.shape[0])
            },
            'message': 'Image uploaded successfully! Configure SAM parameters and run segmentation.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_sam_config', methods=['GET'])
def get_sam_config():
    """Get available SAM backends and configuration options"""
    try:
        available_backends = engine.get_available_backends()
        
        config_info = {
            'available_backends': available_backends,
            'current_backend': engine.current_backend,
            'advanced_features_available': ADVANCED_SAM_AVAILABLE,
            'model_sizes': ['vit_b', 'vit_l', 'vit_h'],
            'performance_mode_available': ADVANCED_SAM_AVAILABLE
        }
        
        if ADVANCED_SAM_AVAILABLE and engine.sam_config:
            config_info['current_config'] = {
                'backend': engine.current_backend,
                'model_type': engine.current_model_size,
                'use_gpu': engine.use_gpu,
                'performance_mode': engine.performance_mode
            }
        
        return jsonify({
            'success': True,
            'config': config_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/run_sam_segmentation', methods=['POST'])
def run_sam_segmentation():
    """Run SAM segmentation with specified parameters"""
    try:
        data = request.get_json()
        model_size = data.get('model_size', 'vit_b')
        crop_layers = data.get('crop_layers', 1)
        points_per_side = data.get('points_per_side', 32)
        backend = data.get('backend', 'pytorch')
        performance_mode = data.get('performance_mode', False)
        use_gpu = data.get('use_gpu', True)
        
        if engine.current_image is None:
            return jsonify({'success': False, 'error': 'No image loaded. Please upload an image first.'})
        
        # Configure and run SAM segmentation with advanced parameters
        engine.configure_sam_parameters(
            model_size=model_size, 
            crop_layers=crop_layers, 
            points_per_side=points_per_side,
            backend=backend,
            performance_mode=performance_mode,
            use_gpu=use_gpu
        )
        overlay_image, summary, mask_stats = engine.perform_sam_segmentation()
        
        if overlay_image is None:
            return jsonify({
                'success': True,
                'masks_found': False,
                'message': 'No masks detected with current parameters. Try adjusting the settings.'
            })
        
        # Convert overlay to base64
        overlay_base64 = engine.get_image_as_base64(overlay_image)
        
        return jsonify({
            'success': True,
            'masks_found': True,
            'overlay_image': overlay_base64,
            'masks_count': len(mask_stats),
            'summary': summary,
            'masks': mask_stats,
            'parameters': {
                'model_size': model_size,
                'crop_layers': crop_layers,
                'points_per_side': points_per_side,
                'backend': backend,
                'performance_mode': performance_mode,
                'use_gpu': use_gpu
            },
            'message': f'SAM segmentation completed! Found {len(mask_stats)} masks using {backend} backend.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_mask_info', methods=['POST'])
def get_mask_info():
    """Get mask information at specific coordinates"""
    try:
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        mask_info = engine.get_mask_at_point(int(x), int(y))
        
        if mask_info:
            # Add quality score
            mask_info['quality_score'] = mask_info['circularity'] * 0.6 + (1.0 - abs(1.0 - mask_info['aspect_ratio'])) * 0.4
            
            return jsonify({
                'success': True,
                'has_mask': True,
                'mask_info': mask_info
            })
        else:
            return jsonify({
                'success': True,
                'has_mask': False,
                'mask_info': None
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/toggle_mask', methods=['POST'])
def toggle_mask():
    """Toggle mask state at specific coordinates"""
    try:
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        # Check current stage before allowing mask interactions
        current_stage = engine.get_current_stage()
        if not engine.is_mask_interaction_allowed():
            return jsonify({
                'success': True,
                'mask_toggled': False,
                'stage_blocked': True,
                'current_stage': current_stage,
                'message': f'Mask interactions not allowed during {current_stage} stage'
            })
        
        toggle_result, overlay_image = engine.toggle_mask_at_point(int(x), int(y))
        
        if toggle_result:
            overlay_base64 = engine.get_image_as_base64(overlay_image)
            
            return jsonify({
                'success': True,
                'mask_toggled': True,
                'toggle_info': toggle_result,
                'overlay_image': overlay_base64,
                'stage_blocked': False,
                'current_stage': current_stage,
                'message': f"Mask {toggle_result['mask_id'] + 1} {toggle_result['new_state']}"
            })
        else:
            return jsonify({
                'success': True,
                'mask_toggled': False,
                'stage_blocked': False,
                'current_stage': current_stage,
                'message': 'No mask found at clicked location'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_all_masks', methods=['POST'])
def get_all_masks():
    """Get information about all masks"""
    try:
        masks, summary = engine.get_all_masks_info()
        
        return jsonify({
            'success': True,
            'masks_count': len(masks),
            'masks': masks,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_all_masks', methods=['POST'])
def reset_all_masks():
    """Reset all masks to active state"""
    try:
        overlay_image = engine.reset_all_masks()
        
        if overlay_image is not None:
            overlay_base64 = engine.get_image_as_base64(overlay_image)
            
            return jsonify({
                'success': True,
                'overlay_image': overlay_base64,
                'message': 'All masks restored to active state'
            })
        else:
            return jsonify({'success': False, 'error': 'No masks to reset'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_mask_preview', methods=['POST'])
def get_mask_preview():
    """Get mask preview for hover display at specific coordinates"""
    try:
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        preview_base64, mask_info = engine.get_mask_preview_at_point(int(x), int(y))
        
        if preview_base64 and mask_info:
            return jsonify({
                'success': True,
                'has_mask': True,
                'preview_image': preview_base64,
                'mask_info': mask_info,
                'coordinates': {'x': x, 'y': y}
            })
        else:
            return jsonify({
                'success': True,
                'has_mask': False,
                'preview_image': None,
                'mask_info': None,
                'coordinates': {'x': x, 'y': y}
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply_image_adjustments', methods=['POST'])
def apply_image_adjustments():
    """Apply brightness, contrast, and intensity threshold adjustments"""
    try:
        data = request.get_json()
        brightness = data.get('brightness', 0)
        contrast = data.get('contrast', 1.0)
        intensity_threshold = data.get('intensity_threshold', -1)
        apply_to_masks_only = data.get('apply_to_masks_only', False)
        
        if engine.original_image is None:
            return jsonify({'success': False, 'error': 'No image loaded'})
        
        # Apply adjustments
        # Always apply to whole image per latest UX requirement
        adjusted_image = engine.apply_image_adjustments(
            brightness=int(brightness),
            contrast=float(contrast),
            intensity_threshold=int(intensity_threshold)
        )
        
        if adjusted_image is None:
            return jsonify({'success': False, 'error': 'Failed to apply adjustments'})
        
        # Convert to base64
        adjusted_base64 = engine.get_image_as_base64(adjusted_image)
        
        return jsonify({
            'success': True,
            'adjusted_image': adjusted_base64,
            'parameters': {
                'brightness': brightness,
                'contrast': contrast,
                'intensity_threshold': intensity_threshold,
                'apply_to_masks_only': False
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/set_processing_stage', methods=['POST'])
def set_processing_stage():
    """Set the current processing stage to control interactions"""
    try:
        data = request.get_json()
        stage = data.get('stage', 'segmentation')
        
        success = engine.set_stage(stage)
        
        if success:
            return jsonify({
                'success': True,
                'stage': stage,
                'mask_interactions_allowed': engine.is_mask_interaction_allowed(),
                'message': f'Stage set to: {stage}'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid stage: {stage}'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_current_stage', methods=['GET'])
def get_current_stage():
    """Get the current processing stage and interaction permissions"""
    try:
        stage = engine.get_current_stage()
        return jsonify({
            'success': True,
            'stage': stage,
            'mask_interactions_allowed': engine.is_mask_interaction_allowed()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_mask_state', methods=['POST'])
def update_mask_state():
    """Update mask state in the backend"""
    try:
        data = request.get_json()
        mask_id = data.get('mask_id', 0)
        new_state = data.get('state', 'active')
        
        if engine.sam_analyzer is None or mask_id >= len(engine.sam_analyzer.mask_states):
            return jsonify({'success': False, 'error': 'Invalid mask ID'})
        
        # Update the mask state in the backend
        engine.sam_analyzer.mask_states[mask_id] = new_state
        
        return jsonify({
            'success': True,
            'mask_id': mask_id,
            'new_state': new_state,
            'message': f'Mask {mask_id + 1} state updated to {new_state}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/store_masks_for_next_stage', methods=['POST'])
def store_masks_for_next_stage():
    """Store mask information in backend for next processing stage"""
    try:
        data = request.get_json()
        masks_data = data.get('masks', [])
        stage = data.get('stage', 'blob_analysis')
        
        if not masks_data:
            return jsonify({'success': False, 'error': 'No mask data provided'})
        
        # Store masks in the engine for next stage processing
        engine.stored_masks = masks_data
        engine.stored_masks_stage = stage
        
        # Log the storage
        print(f"🔒 Stored {len(masks_data)} masks for {stage} stage")
        
        return jsonify({
            'success': True,
            'masks_stored': len(masks_data),
            'stage': stage,
            'message': f'Successfully stored {len(masks_data)} masks for {stage} stage'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_stored_masks', methods=['GET'])
def get_stored_masks():
    """Get stored masks for next stage processing"""
    try:
        if engine.has_stored_masks():
            return jsonify({
                'success': True,
                'masks': engine.get_stored_masks(),
                'stage': engine.stored_masks_stage,
                'count': len(engine.get_stored_masks())
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No stored masks available'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/associate_masks_to_blobs', methods=['POST'])
def associate_masks_to_blobs():
    """Associate stored masks to closest blobs in the last adjusted image and return overlay + mapping."""
    try:
        overlay, associations = engine.associate_masks_to_blobs()
        if overlay is None:
            return jsonify({'success': False, 'error': 'No adjusted image or stored masks available'})
        
        overlay_base64 = engine.get_image_as_base64(overlay)
        return jsonify({
            'success': True,
            'overlay_image': overlay_base64,
            'associations': associations,
            'count_masks': len(associations)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_active_masks_count', methods=['POST'])
def get_active_masks_count():
    """Get count of active masks for next stage validation"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({
                'success': True,
                'active_count': 0,
                'total_count': 0,
                'can_proceed': False
            })
        
        active_count = sum(1 for state in engine.sam_analyzer.mask_states if state == 'active')
        total_count = len(engine.sam_analyzer.masks)
        
        return jsonify({
            'success': True,
            'active_count': active_count,
            'total_count': total_count,
            'can_proceed': active_count > 0
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(os.path.join(project_root, 'uploads'), exist_ok=True)
    
    print("🚀 Starting SAM Interactive Segmentation Website...")
    print("📍 Server will be available at: http://localhost:5001")
    print("🎯 Features: Upload images, configure SAM parameters, interactive mask management")
    print()
    
    try:
        app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
