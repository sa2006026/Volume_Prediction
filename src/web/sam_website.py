#!/usr/bin/env python3
"""
SAM Interactive Segmentation Website
A dedicated web interface for SAM-based image segmentation with interactive mask management
"""

from flask import Flask, render_template, request, jsonify, send_file
try:
    from flask.json.provider import DefaultJSONProvider
except ImportError:
    # For older Flask versions
    from flask.json import JSONEncoder as DefaultJSONProvider
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
try:
    app.json = NumpyJSONProvider(app)
except TypeError:
    # For older Flask versions, use JSONEncoder
    app.json_encoder = NumpyJSONProvider
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
        self.stored_masks = []  # Store masks for analysis
        self.last_adjusted_image = None  # Keep last adjusted image
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize advanced SAM configuration if available
        if ADVANCED_SAM_AVAILABLE:
            self._initialize_advanced_sam_config()
    
    def get_stored_masks(self):
        """Get stored masks for analysis"""
        return self.stored_masks
    
    def has_stored_masks(self) -> bool:
        """Check if there are stored masks available"""
        return len(self.stored_masks) > 0
    
    def is_intensity_filter_active(self) -> bool:
        """Check if intensity filtering is currently active"""
        if self.sam_analyzer is None or not self.sam_analyzer.mask_states:
            return False
        
        # Check if any masks are in 'intensity_filtered' state
        return any(state == 'intensity_filtered' for state in self.sam_analyzer.mask_states)
    
    def is_overlap_filter_active(self) -> bool:
        """Check if overlap filtering is currently active"""
        if self.sam_analyzer is None or not self.sam_analyzer.mask_states:
            return False
        
        # Check if any masks are in 'overlap_filtered' state
        return any(state == 'overlap_filtered' for state in self.sam_analyzer.mask_states)
    
    def is_any_filter_active(self) -> bool:
        """Check if any quality filtering is currently active"""
        return self.is_intensity_filter_active() or self.is_overlap_filter_active()
    
    def create_clean_filtered_overlay(self):
        """
        Create a clean overlay using the clear-and-rebuild approach:
        1. Clear all existing bounding boxes from preview
        2. Add back only the masks that meet the intensity filter criteria
        
        Returns:
            Clean overlay image with only qualifying masks and bounding boxes
        """
        if self.sam_analyzer is None:
            return None
        
        # Always use the filtered overlay method for consistency
        # It will handle both filtered and non-filtered states correctly
        return self.sam_analyzer.create_filtered_mask_overlay(
            show_labels=False,
            alpha=0.3
        )
    
    def apply_mask_overlap_filter(self, overlap_threshold: float = 0.8, remove_mode: str = 'larger'):
        """Apply mask-based overlap filtering instead of bbox-based.

        Logic: For any two active masks i, j, compute the intersection area of
        the binary masks. If intersection / min(area_i, area_j) >= overlap_threshold,
        mark the mask as 'overlap_filtered' based on remove_mode.

        Args:
            overlap_threshold: Ratio in [0,1]. E.g., 0.8 => 80% of smaller mask overlapped
            remove_mode: 'larger' to remove larger mask, 'smaller' to remove smaller mask

        Returns:
            Dict summary with kept_count and removed_count.
        """
        if self.sam_analyzer is None or not self.sam_analyzer.masks:
            return {
                'success': False,
                'error': 'No masks available for filtering',
                'kept_count': 0,
                'removed_count': 0
            }

        masks = self.sam_analyzer.masks
        # Ensure mask_states aligns with masks length
        if not hasattr(self.sam_analyzer, 'mask_states') or not self.sam_analyzer.mask_states:
            self.sam_analyzer.mask_states = ['active'] * len(masks)
        else:
            # Extend or trim to match
            if len(self.sam_analyzer.mask_states) < len(masks):
                self.sam_analyzer.mask_states += ['active'] * (len(masks) - len(self.sam_analyzer.mask_states))
            elif len(self.sam_analyzer.mask_states) > len(masks):
                self.sam_analyzer.mask_states = self.sam_analyzer.mask_states[:len(masks)]

        states = self.sam_analyzer.mask_states

        # Precompute areas for active masks
        areas = []
        bin_masks = []
        for m in masks:
            bm = (m > 0)
            bin_masks.append(bm)
            areas.append(int(np.count_nonzero(bm)))

        to_remove = set()
        n = len(bin_masks)
        for i in range(n):
            if states[i] != 'active' or i in to_remove or areas[i] == 0:
                continue
            mi = bin_masks[i]
            ai = areas[i]
            for j in range(i + 1, n):
                if states[j] != 'active' or j in to_remove or areas[j] == 0:
                    continue
                mj = bin_masks[j]
                aj = areas[j]
                # Intersection count
                inter = int(np.count_nonzero(mi & mj))
                if inter == 0:
                    continue
                base = min(ai, aj)
                if base == 0:
                    continue
                ratio = inter / float(base)
                if ratio >= float(overlap_threshold):
                    # Remove mask based on remove_mode
                    if remove_mode == 'smaller':
                        # Remove the smaller mask between i and j
                        # If ai >= aj: j is smaller, remove j
                        # If ai < aj: i is smaller, remove i
                        remove_idx = j if ai >= aj else i
                    else:  # 'larger' (default)
                        # Remove the larger mask between i and j
                        # If ai >= aj: i is larger, remove i
                        # If ai < aj: j is larger, remove j
                        remove_idx = i if ai >= aj else j
                    to_remove.add(remove_idx)
                    
                    # CRITICAL FIX: If current mask i is marked for removal, 
                    # stop comparing it with other masks
                    if remove_idx == i:
                        break

        # Apply removals by updating states
        removed_count = 0
        for idx in to_remove:
            if states[idx] == 'active':
                states[idx] = 'overlap_filtered'
                removed_count += 1

        kept_count = len([s for s in states if s == 'active'])
        return {
            'success': True,
            'kept_count': kept_count,
            'removed_count': removed_count
        }

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
    
    def perform_sam_segmentation(self, apply_overlap_filter: bool = True, overlap_threshold: float = 0.8, overlap_remove_mode: str = 'larger'):
        """Perform SAM segmentation with current parameters and optional overlap filtering"""
        if self.sam_analyzer is None:
            raise ValueError("No image loaded")
        
        # Configure parameters before segmentation
        self.configure_sam_parameters(
            self.current_model_size, 
            self.current_crop_layers, 
            self.current_points_per_side
        )
        
        # Perform segmentation with configurable overlap filtering
        mask_stats = self.sam_analyzer.segment_droplets(
            method="sam", 
            apply_overlap_filter=apply_overlap_filter, 
            overlap_threshold=overlap_threshold
        )
        
        if not mask_stats:
            return None, None, []
        
        # Apply mask-based overlap filter (overrides any bbox-based logic inside analyzer)
        if apply_overlap_filter:
            self.apply_mask_overlap_filter(overlap_threshold, overlap_remove_mode)
        
        # Create overlay visualization using clean approach
        overlay_image = self.create_clean_filtered_overlay()
        
        # Get summary statistics
        summary = self.sam_analyzer.get_segmentation_summary()
        
        return overlay_image, summary, mask_stats
    
    def get_mask_at_point(self, x: int, y: int):
        """Get mask information at specific coordinates"""
        if self.sam_analyzer is None:
            return None
        
        mask_info = self.sam_analyzer.get_mask_at_point(x, y)
        
        # If any filter is active, only return info for non-filtered masks
        if mask_info and self.is_any_filter_active():
            mask_id = mask_info.get('mask_id', -1)
            if mask_id >= 0 and mask_id < len(self.sam_analyzer.mask_states):
                mask_state = self.sam_analyzer.mask_states[mask_id]
                if mask_state in ['intensity_filtered', 'overlap_filtered']:
                    return None  # Don't return info for filtered masks
        
        return mask_info
    
    def toggle_mask_at_point(self, x: int, y: int):
        """Toggle mask state at specific coordinates"""
        if self.sam_analyzer is None:
            return None, None
        
        # Check if mask interactions are allowed in current stage
        if not self.is_mask_interaction_allowed():
            return None, None
        
        toggle_result = self.sam_analyzer.toggle_mask_state(x, y)
        
        if toggle_result:
            # Create updated visualization using clean clear-and-rebuild approach
            overlay_image = self.create_clean_filtered_overlay()
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
        
        # Create updated visualization using clean approach
        overlay_image = self.create_clean_filtered_overlay()
        
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
                # Check if any filter is active and skip filtered masks
                if self.is_any_filter_active():
                    mask_state = (self.sam_analyzer.mask_states[i] 
                                if i < len(self.sam_analyzer.mask_states) else 'active')
                    if mask_state in ['intensity_filtered', 'overlap_filtered']:
                        continue  # Skip filtered masks
                
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
    
    def get_mask_preview_by_id(self, mask_id: int):
        """Get mask preview by mask ID directly"""
        print(f"   🔍 get_mask_preview_by_id called with mask_id={mask_id}")
        
        if self.sam_analyzer is None or not self.sam_analyzer.masks:
            print(f"   ❌ No sam_analyzer or no masks")
            return None, None
        
        print(f"   📊 Total masks available: {len(self.sam_analyzer.masks)}")
        
        if mask_id < 0 or mask_id >= len(self.sam_analyzer.masks):
            print(f"   ❌ mask_id {mask_id} out of range [0, {len(self.sam_analyzer.masks)-1}]")
            return None, None
        
        # Check if any filter is active and skip filtered masks
        if self.is_any_filter_active():
            mask_state = (self.sam_analyzer.mask_states[mask_id] 
                        if mask_id < len(self.sam_analyzer.mask_states) else 'active')
            if mask_state in ['intensity_filtered', 'overlap_filtered']:
                print(f"   ❌ Mask {mask_id} is filtered out (state: {mask_state})")
                return None, None  # Skip filtered masks
        
        # Get mask center for preview generation
        mask_stats = self.sam_analyzer.mask_statistics[mask_id]
        center_x = int(mask_stats.get('center_x', 0))
        center_y = int(mask_stats.get('center_y', 0))
        print(f"   📍 Mask center: ({center_x}, {center_y})")
        
        # Create a focused preview showing only this specific blob
        preview_image = self._create_blob_focused_preview(mask_id, center_x, center_y, preview_size=(200, 200))
        
        if preview_image is not None:
            print(f"   ✅ Preview image created successfully")
            # Convert to base64
            preview_base64 = self.get_image_as_base64(preview_image)
            
            # Get mask info
            mask_info = mask_stats.copy()
            mask_info['mask_id'] = mask_id
            mask_info['state'] = (self.sam_analyzer.mask_states[mask_id] 
                                if mask_id < len(self.sam_analyzer.mask_states) else 'active')
            
            return preview_base64, mask_info
        else:
            print(f"   ❌ _create_blob_focused_preview returned None")
        
        return None, None
    
    def _create_blob_focused_preview(self, mask_id: int, x: int, y: int, preview_size: tuple = (200, 200)):
        """Create a focused preview showing the specific blob (actual mask, not just bbox)"""
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
        
        # Create a composite image showing the blob with actual mask overlay (not just bbox)
        preview = blob_region.copy()
        
        # Create overlay (semi-transparent Red for the mask region)
        mask_overlay = np.zeros_like(preview)
        mask_area = (mask_region > 0)
        mask_overlay[mask_area] = [0, 0, 255]  # Red
        
        # Blend the overlay with the region
        alpha = 0.4
        preview = cv2.addWeighted(preview, 1 - alpha, mask_overlay, alpha, 0)
        
        # Resize to preview size
        preview = cv2.resize(preview, preview_size)
        
        return preview
    
    def apply_pre_segmentation_filter(self, brightness: int = 0, contrast: float = 1.0, 
                                      min_threshold: int = -1, max_threshold: int = -1, 
                                      filter_mode: str = 'remove_below'):
        """
        Apply pre-segmentation image processing with advanced pixel filtering.
        This prepares the image before SAM segmentation.
        
        Args:
            brightness: Brightness adjustment (-100 to +100, 0 = no change)
            contrast: Contrast adjustment (0.5 to 3.0, 1.0 = no change)
            min_threshold: Minimum intensity threshold (0-255, -1 = not used)
            max_threshold: Maximum intensity threshold (0-255, -1 = not used)
            filter_mode: How to apply thresholds:
                - 'remove_below': Remove pixels below min_threshold (set to black)
                - 'remove_above': Remove pixels above max_threshold (set to black)
                - 'remove_outside': Remove pixels outside [min_threshold, max_threshold] range
                - 'keep_range': Keep only pixels within [min_threshold, max_threshold] range (same as remove_outside)
        
        Returns:
            Adjusted image with filters applied
        """
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
        
        # Apply pixel intensity filtering based on mode
        if min_threshold >= 0 or max_threshold >= 0:
            # Convert to grayscale for intensity calculation
            gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            
            if filter_mode == 'remove_below' and min_threshold >= 0:
                # Remove pixels below minimum threshold
                threshold_mask = gray_image < min_threshold
                adjusted_image[threshold_mask] = [0, 0, 0]
            
            elif filter_mode == 'remove_above' and max_threshold >= 0:
                # Remove pixels above maximum threshold
                threshold_mask = gray_image > max_threshold
                adjusted_image[threshold_mask] = [0, 0, 0]
            
            elif filter_mode in ['remove_outside', 'keep_range']:
                # Remove pixels outside the range [min_threshold, max_threshold]
                if min_threshold >= 0 and max_threshold >= 0:
                    threshold_mask = (gray_image < min_threshold) | (gray_image > max_threshold)
                    adjusted_image[threshold_mask] = [0, 0, 0]
                elif min_threshold >= 0:
                    # Only min specified, remove below
                    threshold_mask = gray_image < min_threshold
                    adjusted_image[threshold_mask] = [0, 0, 0]
                elif max_threshold >= 0:
                    # Only max specified, remove above
                    threshold_mask = gray_image > max_threshold
                    adjusted_image[threshold_mask] = [0, 0, 0]
        
        # Persist adjusted image for SAM segmentation
        self.last_adjusted_image = adjusted_image.copy()
        self.current_image = adjusted_image.copy()
        
        return adjusted_image
    
    def apply_image_adjustments(self, brightness: int = 0, contrast: float = 1.0, intensity_threshold: int = -1):
        """Apply brightness, contrast, and intensity threshold adjustments to the image
        (Legacy method - kept for backward compatibility)"""
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
    
    def reset_to_original_image(self):
        """Reset the current image back to the original uploaded image"""
        if self.original_image is None:
            return None
        
        self.current_image = self.original_image.copy()
        self.last_adjusted_image = None
        
        return self.current_image
    
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
        
        # Track which blobs are closest to masks (will be filled with green)
        closest_blob_ids = set()
        
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
            
            # Track this blob as closest to a mask
            if closest_blob_id >= 0:
                closest_blob_ids.add(closest_blob_id)
            
            associations.append({
                'mask_center': {'x': mx, 'y': my},
                'blob_id': int(closest_blob_id),
                'blob_center': {'x': int(closest_center[0]), 'y': int(closest_center[1])},
                'distance_px': float(np.sqrt(closest_dist)) if closest_dist < float('inf') else None
            })
            
            # Draw mask center and connection line
            cv2.circle(overlay, (mx, my), 4, (0, 0, 255), -1)
            if closest_blob_id >= 0:
                cv2.line(overlay, (mx, my), closest_center, (255, 200, 100), 3)  # Light blue, thicker line
        
        # Fill only the closest blobs with green
        for blob_id in closest_blob_ids:
            cnt = contours[blob_id]
            cv2.fillPoly(overlay, [cnt], (0, 255, 0))
        
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
            'masks': [],  # Empty mask list - frontend should clear all bounding boxes
            'masks_count': 0,
            'clear_and_redraw': True,  # Explicit flag for frontend to clear drawings
            'clear_preview': True,     # Explicitly clear preview overlay as well
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
        apply_overlap_filter = data.get('apply_overlap_filter', True)
        overlap_threshold = data.get('overlap_threshold', 0.8)
        overlap_remove_mode = data.get('overlap_remove_mode', 'larger')
        
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
        overlay_image, summary, mask_stats = engine.perform_sam_segmentation(
            apply_overlap_filter=apply_overlap_filter,
            overlap_threshold=overlap_threshold,
            overlap_remove_mode=overlap_remove_mode
        )
        
        if overlay_image is None:
            return jsonify({
                'success': True,
                'masks_found': False,
                'message': 'No masks detected with current parameters. Try adjusting the settings.'
            })
        
        # Convert overlay to base64
        # To avoid double-drawing boxes in the frontend, return the base image (no boxes)
        overlay_base64 = engine.get_image_as_base64()
        
        # Filter masks by state for response so only masks that passed all filters are returned
        filtered_mask_stats = [s for s in mask_stats if s.get('state', 'active') not in ['intensity_filtered', 'overlap_filtered']]
        visible_masks_count = len(filtered_mask_stats)
        total_masks_count = len(mask_stats)
        
        return jsonify({
            'success': True,
            'masks_found': True,
            'overlay_image': overlay_base64,
            'masks_count': visible_masks_count,
            'total_masks': total_masks_count,
            'summary': summary,
            'masks': filtered_mask_stats,
            'parameters': {
                'model_size': model_size,
                'crop_layers': crop_layers,
                'points_per_side': points_per_side,
                'backend': backend,
                'performance_mode': performance_mode,
                'use_gpu': use_gpu
            },
            'message': f'SAM segmentation completed! Showing {visible_masks_count} masks (of {total_masks_count}) using {backend} backend.'
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
            
            # Add unit conversion information if enabled
            if engine.sam_analyzer and engine.sam_analyzer.conversion_enabled:
                mask_id = mask_info.get('mask_id', -1)
                if mask_id >= 0:
                    converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(mask_id)
                    if converted_stats:
                        mask_info.update(converted_stats)
            
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
        
        # Allow mask interactions
        
        toggle_result, overlay_image = engine.toggle_mask_at_point(int(x), int(y))
        
        if toggle_result:
            overlay_base64 = engine.get_image_as_base64(overlay_image)
            
            return jsonify({
                'success': True,
                'mask_toggled': True,
                'toggle_info': toggle_result,
                'overlay_image': overlay_base64,
                'message': f"Mask {toggle_result['mask_id'] + 1} {toggle_result['new_state']}"
            })
        else:
            return jsonify({
                'success': True,
                'mask_toggled': False,
                'message': 'No mask found at clicked location'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_all_masks', methods=['POST'])
def get_all_masks():
    """Get information about all masks"""
    try:
        masks, summary = engine.get_all_masks_info()
        
        # Add unit conversion information if enabled
        if engine.sam_analyzer and engine.sam_analyzer.conversion_enabled:
            converted_masks = []
            for mask in masks:
                mask_id = mask.get('mask_id', -1)
                if mask_id >= 0:
                    converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(mask_id)
                    if converted_stats:
                        mask.update(converted_stats)
                converted_masks.append(mask)
            masks = converted_masks
            
            # Add conversion info to summary
            conversion_info = engine.sam_analyzer.get_conversion_info()
            summary['conversion_info'] = conversion_info
        
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
    """Get mask preview for hover display at specific coordinates or by mask ID"""
    try:
        data = request.get_json()
        
        # Support both coordinate-based and ID-based lookup
        if 'mask_id' in data:
            # Direct mask ID lookup
            mask_id = data.get('mask_id')
            print(f"🔍 Preview request for mask_id: {mask_id}")
            preview_base64, mask_info = engine.get_mask_preview_by_id(int(mask_id))
            print(f"   Result: preview={'available' if preview_base64 else 'None'}, mask_info={'available' if mask_info else 'None'}")
        else:
            # Coordinate-based lookup (legacy)
            x = data.get('x', 0)
            y = data.get('y', 0)
            print(f"🔍 Preview request for coordinates: ({x}, {y})")
            preview_base64, mask_info = engine.get_mask_preview_at_point(int(x), int(y))
            print(f"   Result: preview={'available' if preview_base64 else 'None'}, mask_info={'available' if mask_info else 'None'}")
        
        if preview_base64 and mask_info:
            # Add unit conversion information if enabled
            if engine.sam_analyzer and engine.sam_analyzer.conversion_enabled:
                mask_id = mask_info.get('mask_id', -1)
                if mask_id >= 0:
                    converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(mask_id)
                    if converted_stats:
                        mask_info.update(converted_stats)
            
            return jsonify({
                'success': True,
                'has_mask': True,
                'preview_image': preview_base64,
                'mask_info': mask_info
            })
        else:
            print(f"   ❌ No preview available")
            return jsonify({
                'success': True,
                'has_mask': False,
                'preview_image': None,
                'mask_info': None
            })
        
    except Exception as e:
        print(f"❌ Error in get_mask_preview: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply_pre_segmentation_filter', methods=['POST'])
def apply_pre_segmentation_filter():
    """Apply pre-segmentation image processing with brightness, contrast, and advanced pixel filtering"""
    try:
        data = request.get_json()
        brightness = data.get('brightness', 0)
        contrast = data.get('contrast', 1.0)
        min_threshold = data.get('min_threshold', -1)
        max_threshold = data.get('max_threshold', -1)
        filter_mode = data.get('filter_mode', 'remove_below')
        
        if engine.original_image is None:
            return jsonify({'success': False, 'error': 'No image loaded. Please upload an image first.'})
        
        # Apply pre-segmentation filter
        adjusted_image = engine.apply_pre_segmentation_filter(
            brightness=int(brightness),
            contrast=float(contrast),
            min_threshold=int(min_threshold),
            max_threshold=int(max_threshold),
            filter_mode=str(filter_mode)
        )
        
        if adjusted_image is None:
            return jsonify({'success': False, 'error': 'Failed to apply pre-segmentation filter'})
        
        # Convert to base64
        adjusted_base64 = engine.get_image_as_base64(adjusted_image)
        
        return jsonify({
            'success': True,
            'filtered_image': adjusted_base64,
            'parameters': {
                'brightness': brightness,
                'contrast': contrast,
                'min_threshold': min_threshold,
                'max_threshold': max_threshold,
                'filter_mode': filter_mode
            },
            'message': 'Pre-segmentation filter applied successfully. You can now run SAM segmentation on the filtered image.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_pre_segmentation_filter', methods=['POST'])
def reset_pre_segmentation_filter():
    """Reset image to original state before any pre-segmentation filters"""
    try:
        if engine.original_image is None:
            return jsonify({'success': False, 'error': 'No image loaded'})
        
        # Reset to original image
        reset_image = engine.reset_to_original_image()
        
        if reset_image is None:
            return jsonify({'success': False, 'error': 'Failed to reset image'})
        
        # Convert to base64
        reset_base64 = engine.get_image_as_base64(reset_image)
        
        return jsonify({
            'success': True,
            'image': reset_base64,
            'message': 'Image reset to original state. All filters cleared.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply_image_adjustments', methods=['POST'])
def apply_image_adjustments():
    """Apply brightness, contrast, and intensity threshold adjustments (Legacy endpoint)"""
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


@app.route('/get_stored_masks', methods=['GET'])
def get_stored_masks():
    """Get stored masks for next stage processing"""
    try:
        if engine.has_stored_masks():
            return jsonify({
                'success': True,
                'masks': engine.get_stored_masks(),
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

@app.route('/apply_intensity_filter', methods=['POST'])
def apply_intensity_filter():
    """Apply intensity filter to filter out masks based on intensity range"""
    try:
        data = request.get_json()
        min_intensity = data.get('min_intensity', 0)
        max_intensity = data.get('max_intensity', 255)
        
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No masks available for filtering'})
        
        # Apply intensity filter with min/max range
        filter_results = engine.sam_analyzer.apply_intensity_filter(
            min_intensity=float(min_intensity),
            max_intensity=float(max_intensity)
        )
        
        if not filter_results['success']:
            return jsonify(filter_results)
        
        # Return base image only to avoid double-drawing; frontend will draw latest boxes
        base_image_b64 = engine.get_image_as_base64()
        
        # Get filtered mask list (only active masks that passed the filter)
        filtered_masks = []
        total_masks = len(engine.sam_analyzer.mask_statistics)
        
        for i, (mask_stats, mask_state) in enumerate(zip(
            engine.sam_analyzer.mask_statistics, 
            engine.sam_analyzer.mask_states
        )):
            # Only include masks that are active (not filtered out)
            if mask_state == 'active':
                mask_info = mask_stats.copy()
                mask_info['state'] = mask_state
                filtered_masks.append(mask_info)
        
        # Add unit conversion information if enabled
        if engine.sam_analyzer and engine.sam_analyzer.conversion_enabled:
            converted_masks = []
            for mask in filtered_masks:
                mask_id = mask.get('mask_id', -1)
                if mask_id >= 0:
                    converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(mask_id)
                    if converted_stats:
                        mask.update(converted_stats)
                converted_masks.append(mask)
            filtered_masks = converted_masks
        
        # Log for debugging
        print(f"🔍 Intensity filter applied: {min_intensity}-{max_intensity}")
        print(f"📊 Total masks: {total_masks}, Active: {len(filtered_masks)}, Filtered: {total_masks - len(filtered_masks)}")
        print(f"📦 Returning {len(filtered_masks)} masks to frontend")
        if filtered_masks:
            print(f"📦 First mask has bounding_box: {'bounding_box' in filtered_masks[0]}")
        print(f"📦 Unit conversion enabled: {engine.sam_analyzer.conversion_enabled if engine.sam_analyzer else False}")
        
        return jsonify({
            'success': True,
            'image': base_image_b64,
            'filter_results': filter_results,
            'masks': filtered_masks,  # Add filtered mask list for frontend to draw bounding boxes
            'masks_count': len(filtered_masks),
            'total_masks': total_masks,  # Add total count for comparison
            'filtered_count': total_masks - len(filtered_masks),
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'clear_and_redraw': True,  # Flag to tell frontend to clear all boxes first
            'message': f'Intensity filter applied! Kept: {filter_results["kept_count"]}, Filtered: {filter_results["filtered_count"]}'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_intensity_filter', methods=['POST'])
def reset_intensity_filter():
    """Reset intensity filter to unfiltered state"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No masks available'})
        
        # Reset intensity filter
        engine.sam_analyzer.reset_intensity_filter()
        
        # Return base image only; frontend redraws according to current states
        base_image_b64 = engine.get_image_as_base64()
        
        # Get all masks after reset (all should be active now)
        all_masks = []
        for i, (mask_stats, mask_state) in enumerate(zip(
            engine.sam_analyzer.mask_statistics, 
            engine.sam_analyzer.mask_states
        )):
            mask_info = mask_stats.copy()
            mask_info['state'] = mask_state
            all_masks.append(mask_info)
        
        # Add unit conversion information if enabled
        if engine.sam_analyzer and engine.sam_analyzer.conversion_enabled:
            converted_masks = []
            for mask in all_masks:
                mask_id = mask.get('mask_id', -1)
                if mask_id >= 0:
                    converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(mask_id)
                    if converted_stats:
                        mask.update(converted_stats)
                converted_masks.append(mask)
            all_masks = converted_masks
        
        # Log for debugging
        print(f"🔄 Intensity filter reset")
        print(f"📦 Returning {len(all_masks)} masks to frontend")
        print(f"📦 Unit conversion enabled: {engine.sam_analyzer.conversion_enabled if engine.sam_analyzer else False}")
        
        return jsonify({
            'success': True,
            'image': base_image_b64,
            'masks': all_masks,  # Return all masks for frontend to redraw all bounding boxes
            'masks_count': len(all_masks),
            'clear_and_redraw': True,  # Flag to tell frontend to clear all boxes first
            'message': 'Intensity filter reset - all masks are unfiltered'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_intensity_statistics', methods=['POST'])
def get_intensity_statistics():
    """Get intensity statistics for all masks"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No masks available'})
        
        statistics = engine.sam_analyzer.get_intensity_statistics()
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/debug_mask_states', methods=['GET'])
def debug_mask_states():
    """Debug endpoint to show current mask states (for troubleshooting)"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({
                'success': False,
                'error': 'No masks available',
                'total_masks': 0
            })
        
        # Get state breakdown
        state_counts = {}
        mask_details = []
        
        for i, (mask_stats, mask_state) in enumerate(zip(
            engine.sam_analyzer.mask_statistics,
            engine.sam_analyzer.mask_states
        )):
            # Count states
            state_counts[mask_state] = state_counts.get(mask_state, 0) + 1
            
            # Add to details
            mask_details.append({
                'mask_id': i,
                'state': mask_state,
                'mean_intensity': mask_stats.get('mean_intensity', 0),
                'bounding_box': mask_stats.get('bounding_box', [0, 0, 0, 0])
            })
        
        return jsonify({
            'success': True,
            'total_masks': len(engine.sam_analyzer.masks),
            'state_counts': state_counts,
            'active_count': state_counts.get('active', 0),
            'intensity_filtered_count': state_counts.get('intensity_filtered', 0),
            'overlap_filtered_count': state_counts.get('overlap_filtered', 0),
            'removed_count': state_counts.get('removed', 0),
            'mask_details': mask_details[:10]  # First 10 masks for debugging
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/apply_overlap_filter', methods=['POST'])
def apply_overlap_filter():
    """Apply overlap filter to remove duplicate masks"""
    try:
        data = request.get_json()
        overlap_threshold = data.get('overlap_threshold', 0.8)
        remove_mode = data.get('remove_mode', 'larger')
        
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No masks available for filtering'})
        
        # Apply mask-based overlap filter (intersection over smaller mask area)
        filter_results = engine.apply_mask_overlap_filter(
            overlap_threshold=float(overlap_threshold),
            remove_mode=str(remove_mode)
        )
        
        if not filter_results['success']:
            return jsonify(filter_results)
        
        # Create updated overlay using clean clear-and-rebuild approach
        overlay_image = engine.create_clean_filtered_overlay()
        
        overlay_base64 = engine.get_image_as_base64(overlay_image)
        
        remove_text = "larger" if remove_mode == 'larger' else "smaller"
        
        return jsonify({
            'success': True,
            'image': overlay_base64,
            'filter_results': filter_results,
            'overlap_threshold': overlap_threshold,
            'remove_mode': remove_mode,
            'message': f'Overlap filter applied (removing {remove_text} masks)! Kept: {filter_results["kept_count"]}, Removed: {filter_results["removed_count"]} duplicate masks'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_overlap_filter', methods=['POST'])
def reset_overlap_filter():
    """Reset overlap filter to unfiltered state"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No masks available'})
        
        # Reset overlap filter
        engine.sam_analyzer.reset_overlap_filter()
        
        # Create updated overlay using clean approach
        overlay_image = engine.create_clean_filtered_overlay()
        
        overlay_base64 = engine.get_image_as_base64(overlay_image)
        
        return jsonify({
            'success': True,
            'image': overlay_base64,
            'message': 'Overlap filter reset - all duplicate masks are restored'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_diameter_data', methods=['POST'])
def export_diameter_data():
    """Export diameter data by intensity groups (excluding removed masks)"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No masks available'})
        
        # Use unit-converted data if conversion is enabled
        if engine.sam_analyzer.conversion_enabled:
            diameter_data = engine.sam_analyzer.get_diameter_data_by_group_with_units()
            unit_name = diameter_data.get('unit_name', 'pixels')
        else:
            diameter_data = engine.sam_analyzer.get_diameter_data_by_group()
            unit_name = 'pixels'
        
        # Format data for export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diameter_export_{timestamp}.txt"
        
        export_lines = []
        export_lines.append(f"Diameter Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        export_lines.append("=" * 50)
        export_lines.append(f"Units: {unit_name}")
        export_lines.append("")
        
        # High intensity group
        high_intensity = diameter_data.get('high_intensity', [])
        export_lines.append(f"HIGH INTENSITY GROUP (Red) - {len(high_intensity)} masks:")
        if high_intensity:
            export_lines.extend([f"{i+1:3d}. {diameter:.2f} {unit_name}" for i, diameter in enumerate(high_intensity)])
        else:
            export_lines.append("  No masks in this group")
        export_lines.append("")
        
        # Low intensity group
        low_intensity = diameter_data.get('low_intensity', [])
        export_lines.append(f"LOW INTENSITY GROUP (Blue) - {len(low_intensity)} masks:")
        if low_intensity:
            export_lines.extend([f"{i+1:3d}. {diameter:.2f} {unit_name}" for i, diameter in enumerate(low_intensity)])
        else:
            export_lines.append("  No masks in this group")
        export_lines.append("")
        
        # Summary statistics
        all_diameters = high_intensity + low_intensity
        if all_diameters:
            export_lines.append("SUMMARY STATISTICS:")
            export_lines.append(f"Total masks exported: {len(all_diameters)}")
            export_lines.append(f"High intensity: {len(high_intensity)} ({len(high_intensity)/len(all_diameters)*100:.1f}%)")
            export_lines.append(f"Low intensity: {len(low_intensity)} ({len(low_intensity)/len(all_diameters)*100:.1f}%)")
            export_lines.append(f"Average diameter: {np.mean(all_diameters):.2f} {unit_name}")
            export_lines.append(f"Min diameter: {np.min(all_diameters):.2f} {unit_name}")
            export_lines.append(f"Max diameter: {np.max(all_diameters):.2f} {unit_name}")
        
        export_text = "\n".join(export_lines)
        
        return jsonify({
            'success': True,
            'data': export_text,
            'filename': filename,
            'high_count': len(high_intensity),
            'low_count': len(low_intensity),
            'total_count': len(all_diameters),
            'unit_name': unit_name,
            'conversion_enabled': engine.sam_analyzer.conversion_enabled
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/set_pixel_conversion', methods=['POST'])
def set_pixel_conversion():
    """Set pixel-to-unit conversion ratio"""
    try:
        data = request.get_json()
        pixel_distance = data.get('pixel_distance', 0)
        unit_distance = data.get('unit_distance', 0)
        unit_name = data.get('unit_name', 'μm')
        
        if engine.sam_analyzer is None:
            return jsonify({'success': False, 'error': 'No image loaded'})
        
        success = engine.sam_analyzer.set_pixel_to_unit_conversion(
            pixel_distance=float(pixel_distance),
            unit_distance=float(unit_distance),
            unit_name=str(unit_name)
        )
        
        if success:
            conversion_info = engine.sam_analyzer.get_conversion_info()

            # Build currently visible masks (active only), with unit conversion applied
            visible_masks = []
            if engine.sam_analyzer.mask_statistics and engine.sam_analyzer.mask_states:
                for i, (stats, state) in enumerate(zip(
                    engine.sam_analyzer.mask_statistics,
                    engine.sam_analyzer.mask_states
                )):
                    if state == 'active':
                        mask_info = stats.copy()
                        mask_info['state'] = state
                        # Apply unit conversion details to this mask
                        converted_stats = engine.sam_analyzer.get_mask_statistics_with_units(i)
                        if converted_stats:
                            mask_info.update(converted_stats)
                        visible_masks.append(mask_info)
            
            return jsonify({
                'success': True,
                'conversion_info': conversion_info,
                'masks': visible_masks,                 # Return only active masks for redraw
                'masks_count': len(visible_masks),
                'clear_and_redraw': True,               # Frontend should clear boxes and redraw
                'clear_preview': True,                   # Clear preview overlay too
                'message': f'Conversion set: {pixel_distance} pixels = {unit_distance} {unit_name}'
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid conversion parameters'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_conversion_info', methods=['POST'])
def get_conversion_info():
    """Get current pixel-to-unit conversion settings"""
    try:
        if engine.sam_analyzer is None:
            return jsonify({'success': False, 'error': 'No image loaded'})
        
        conversion_info = engine.sam_analyzer.get_conversion_info()
        
        return jsonify({
            'success': True,
            'conversion_info': conversion_info
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_conversion', methods=['POST'])
def reset_conversion():
    """Reset pixel-to-unit conversion settings"""
    try:
        if engine.sam_analyzer is None:
            return jsonify({'success': False, 'error': 'No image loaded'})
        
        engine.sam_analyzer.reset_conversion()
        
        return jsonify({
            'success': True,
            'message': 'Pixel-to-unit conversion reset'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_diameter_excel', methods=['POST'])
def export_diameter_excel():
    """Export diameter data as Excel file with separate columns for high/low intensity groups"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No masks available'})
        
        # Use unit-converted data if conversion is enabled
        if engine.sam_analyzer.conversion_enabled:
            diameter_data = engine.sam_analyzer.get_diameter_data_by_group_with_units()
            unit_name = diameter_data.get('unit_name', 'pixels')
        else:
            diameter_data = engine.sam_analyzer.get_diameter_data_by_group()
            unit_name = 'pixels'
        
        # Get diameter lists
        high_intensity = diameter_data.get('high_intensity', [])
        low_intensity = diameter_data.get('low_intensity', [])
        
        # Create Excel-like data structure (CSV format for simplicity)
        max_length = max(len(high_intensity), len(low_intensity))
        
        # Create CSV content
        csv_lines = []
        csv_lines.append(f"High Intensity Diameter ({unit_name}),Low Intensity Diameter ({unit_name})")
        
        for i in range(max_length):
            high_value = f"{high_intensity[i]:.2f}" if i < len(high_intensity) else ""
            low_value = f"{low_intensity[i]:.2f}" if i < len(low_intensity) else ""
            csv_lines.append(f"{high_value},{low_value}")
        
        csv_content = "\n".join(csv_lines)
        
        # Format filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diameter_export_{timestamp}.csv"
        
        return jsonify({
            'success': True,
            'data': csv_content,
            'filename': filename,
            'high_count': len(high_intensity),
            'low_count': len(low_intensity),
            'total_count': len(high_intensity) + len(low_intensity),
            'unit_name': unit_name,
            'conversion_enabled': engine.sam_analyzer.conversion_enabled,
            'content_type': 'text/csv'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export_mask_csv', methods=['POST'])
def export_mask_csv():
    """Export mask information as CSV with center location, diameter, and pixel intensity"""
    try:
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No mask data available. Please run segmentation first.'})
        
        if not engine.sam_analyzer.mask_statistics:
            return jsonify({'success': False, 'error': 'No mask statistics available. Please run segmentation first.'})
        
        # Create CSV content - only export active masks (not filtered out)
        csv_lines = []
        
        # Determine if we should use units
        use_units = engine.sam_analyzer.conversion_enabled
        unit_name = engine.sam_analyzer.unit_name if use_units else "pixels"
        area_unit = f"{unit_name}²" if use_units else "pixels²"
        
        # Create header with appropriate units
        if use_units:
            csv_lines.append(f"Mask_ID,Center_X_px,Center_Y_px,Diameter_{unit_name},Mean_Intensity,Area_{area_unit},Circularity")
        else:
            csv_lines.append("Mask_ID,Center_X,Center_Y,Diameter,Mean_Intensity,Area,Circularity")
        
        active_mask_count = 0
        for i, stats in enumerate(engine.sam_analyzer.mask_statistics):
            # Get mask state
            mask_state = engine.sam_analyzer.mask_states[i] if i < len(engine.sam_analyzer.mask_states) else 'active'
            
            # Only export masks that are active (not filtered out)
            if mask_state == 'active':
                # Extract the required data
                mask_id = stats.get('mask_id', i)
                center_x = stats.get('center_x', 0)
                center_y = stats.get('center_y', 0)
                diameter = stats.get('diameter', 0)
                mean_intensity = stats.get('mean_intensity', 0)
                area = stats.get('area', 0)
                circularity = stats.get('circularity', 0)
                
                # Convert to units if conversion is enabled
                if use_units:
                    diameter = engine.sam_analyzer.convert_pixels_to_units(diameter)
                    area = engine.sam_analyzer.convert_area_to_units(area)
                
                # Add row to CSV
                csv_lines.append(f"{mask_id},{center_x:.2f},{center_y:.2f},{diameter:.2f},{mean_intensity:.2f},{area:.2f},{circularity:.3f}")
                active_mask_count += 1
        
        # Check if there are any active masks to export
        if active_mask_count == 0:
            return jsonify({'success': False, 'error': 'No active masks to export. All masks have been filtered out.'})
        
        # Join all lines
        csv_content = "\n".join(csv_lines)
        
        # Generate filename with timestamp and unit info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unit_suffix = f"_{unit_name}" if use_units else "_pixels"
        filename = f'mask_data_filtered{unit_suffix}_{timestamp}.csv'
        
        return jsonify({
            'success': True,
            'data': csv_content,
            'filename': filename,
            'exported_masks': active_mask_count,
            'total_masks': len(engine.sam_analyzer.mask_statistics),
            'units_used': unit_name,
            'conversion_enabled': use_units
        })
    
    except Exception as e:
        print(f"❌ Error in CSV export: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(os.path.join(project_root, 'uploads'), exist_ok=True)
    
    print("🚀 Starting SAM Interactive Segmentation Website...")
    print("📍 Server will be available at: http://localhost:5014")
    print("🎯 Features: Upload images, configure SAM parameters, interactive mask management")
    print()
    
    try:
        app.run(host='127.0.0.1', port=5015, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
