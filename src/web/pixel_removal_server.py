#!/usr/bin/env python3
"""
Interactive Pixel Removal Web Server
Allows users to select areas on an image and remove those pixels through a web interface
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.threshold_extractor import ThresholdExtractor
from core.radial_analyzer import RadialAnalyzer
from core.sam_analyzer import SAMAnalyzer

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

# Get the project root directory (2 levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
templates_dir = os.path.join(project_root, 'templates')

app = Flask(__name__, template_folder=templates_dir)
app.json = NumpyJSONProvider(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class PixelRemovalEngine:
    """Engine for handling pixel removal operations"""
    
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.baseline_image = None  # Image before any contrast adjustments
        self.threshold_base_image = None  # Image without threshold extractions (for independent mode)
        self.image_path = None
        self.removal_history = []
        self.image_states = []  # Store image states for undo functionality
        self.current_contrast = 1.0  # Track current contrast level
        self.current_brightness = 0  # Track current brightness level
        self.threshold_extractor = None  # Threshold extraction instance
        self.radial_analyzer = None  # Radial intensity analysis instance
        self.sam_analyzer = None  # SAM segmentation instance
        self.output_dir = "results/pixel_removal"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_image(self, image_path: str):
        """Load image for processing"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.current_image = self.original_image.copy()
        self.baseline_image = self.original_image.copy()  # Set baseline for contrast adjustments
        self.threshold_base_image = self.original_image.copy()  # Set base for threshold operations
        self.removal_history = []
        self.image_states = [self.original_image.copy()]  # Store original as first state
        self.current_contrast = 1.0  # Reset contrast tracking
        self.current_brightness = 0  # Reset brightness tracking
        
        # Initialize threshold extractor
        self.threshold_extractor = ThresholdExtractor()
        self.threshold_extractor.original_image = self.original_image.copy()
        self.threshold_extractor.current_image = self.current_image.copy()
        
        # Initialize radial analyzer
        self.radial_analyzer = RadialAnalyzer()
        self.radial_analyzer.load_image(self.current_image.copy())
        
        # Initialize SAM analyzer
        self.sam_analyzer = SAMAnalyzer()
        self.sam_analyzer.load_image(self.current_image.copy())
        
        return True
    
    def remove_pixels_in_area(self, selection_data: dict, removal_method: str = "black"):
        """Detect pixel intensity in selected region, then remove all similar pixels in whole image"""
        if self.current_image is None:
            return False
        
        # Parse selection data
        selection_type = selection_data.get('type', 'rectangle')
        coords = selection_data.get('coordinates', [])
        
        # Create mask for the selected area (sample region)
        sample_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        
        if selection_type == 'rectangle' and len(coords) >= 4:
            x1, y1, x2, y2 = coords[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            sample_mask[y1:y2, x1:x2] = 255
            
        elif selection_type == 'circle' and len(coords) >= 3:
            center_x, center_y, radius = coords[:3]
            center_x, center_y, radius = int(center_x), int(center_y), int(radius)
            cv2.circle(sample_mask, (center_x, center_y), radius, 255, -1)
            
        elif selection_type == 'polygon' and len(coords) >= 6:
            # Polygon coordinates as [x1,y1,x2,y2,x3,y3,...]
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    points.append([int(coords[i]), int(coords[i+1])])
            
            if len(points) >= 3:
                points_array = np.array(points, dtype=np.int32)
                cv2.fillPoly(sample_mask, [points_array], 255)
        
        # Check if any pixels are selected in sample area
        if np.sum(sample_mask) == 0:
            return False
        
        # Analyze pixel intensities in the selected region
        gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        sample_intensities = gray_image[sample_mask == 255]
        
        if len(sample_intensities) == 0:
            return False
        
        # Use exact min/max intensity range from sample region
        min_intensity = np.min(sample_intensities)
        max_intensity = np.max(sample_intensities)
        
        # Use exact range - no tolerance, no statistical adjustments
        lower_bound = min_intensity
        upper_bound = max_intensity
        
        # Find all pixels in the entire image with similar intensities
        global_mask = ((gray_image >= lower_bound) & (gray_image <= upper_bound)).astype(np.uint8) * 255
        
        # Optional: Clean up the mask to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, kernel)
        global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, kernel)
        
        # Save current state before applying changes (for undo)
        self.image_states.append(self.current_image.copy())
        
        # Limit the number of states to prevent memory issues (keep last 10)
        if len(self.image_states) > 10:
            self.image_states.pop(0)
        
        # Use the global mask for removal (not just the sample area)
        mask = global_mask
        
        # Apply removal based on method
        if removal_method == "black":
            self.current_image[mask == 255] = [0, 0, 0]  # Black pixels
        elif removal_method == "white":
            self.current_image[mask == 255] = [255, 255, 255]  # White pixels
        elif removal_method == "transparent":
            # Convert to RGBA and set alpha to 0
            if self.current_image.shape[2] == 3:
                alpha_channel = np.ones(self.current_image.shape[:2], dtype=np.uint8) * 255
                self.current_image = cv2.merge([self.current_image, alpha_channel])
            self.current_image[mask == 255, 3] = 0  # Transparent
        elif removal_method == "blur":
            # Apply strong blur to selected area
            blurred = cv2.GaussianBlur(self.current_image, (51, 51), 0)
            self.current_image[mask == 255] = blurred[mask == 255]
        elif removal_method == "noise":
            # Add random noise to selected area
            noise = np.random.randint(0, 256, self.current_image.shape, dtype=np.uint8)
            self.current_image[mask == 255] = noise[mask == 255]
        
        # Save to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.removal_history.append({
            'timestamp': timestamp,
            'method': removal_method,
            'selection': selection_data,
            'sample_pixels': int(np.sum(sample_mask == 255)),
            'affected_pixels': int(np.sum(mask == 255)),
            'intensity_range': f"{int(lower_bound)}-{int(upper_bound)}",
            'min_intensity': int(min_intensity),
            'max_intensity': int(max_intensity)
        })
        
        # Update baseline image after pixel removal to maintain contrast adjustments
        self.baseline_image = self.current_image.copy()
        # Also update threshold base image since this is a non-threshold operation
        self.threshold_base_image = self.current_image.copy()
        
        return True
    
    def _rebuild_threshold_base_image(self):
        """Rebuild threshold base image by replaying non-threshold operations"""
        # Start with original image
        self.threshold_base_image = self.original_image.copy()
        
        # Replay all non-threshold operations from history
        for operation in self.removal_history:
            if operation['method'] == 'contrast_adjustment':
                # Apply contrast adjustment
                contrast_factor = operation['contrast_factor']
                brightness_offset = operation['brightness_offset']
                adjusted = cv2.convertScaleAbs(self.threshold_base_image, 
                                             alpha=contrast_factor, beta=brightness_offset)
                self.threshold_base_image = adjusted
            elif operation['method'] != 'threshold_extraction':
                # For other pixel removal operations, we would need to replay them
                # This is complex, so for now we'll mark that threshold base needs rebuilding
                # In practice, this means independent mode after undo might not work perfectly
                # with mixed operations - this is a limitation we can document
                pass

    def undo_last_operation(self):
        """Undo the last pixel removal operation"""
        if len(self.image_states) > 1:  # Need at least original + one operation
            # Remove the current state and go back to previous
            self.image_states.pop()
            self.current_image = self.image_states[-1].copy()
            
            # Remove last operation from history
            last_operation = None
            if self.removal_history:
                last_operation = self.removal_history.pop()
            
            # Rebuild threshold base image and baseline image
            self._rebuild_threshold_base_image()
            self.baseline_image = self.current_image.copy()
            
            return True, len(self.image_states) > 1  # Return if more undos are possible
        return False, False
    
    def reset_image(self):
        """Reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.baseline_image = self.original_image.copy()  # Reset baseline
            self.threshold_base_image = self.original_image.copy()  # Reset threshold base
            self.removal_history = []
            self.image_states = [self.original_image.copy()]  # Reset states to just original
            self.current_contrast = 1.0  # Reset contrast tracking
            self.current_brightness = 0  # Reset brightness tracking
            return True
        return False
    
    def save_current_image(self, filename: str = None):
        """Save current image"""
        if self.current_image is None:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "processed"
            
            # Get the most recent min/max intensity values from threshold operations
            min_intensity = None
            max_intensity = None
            
            # Look through removal history backwards to find the most recent threshold operation
            for operation in reversed(self.removal_history):
                if 'min_intensity' in operation and 'max_intensity' in operation:
                    min_intensity = operation['min_intensity']
                    max_intensity = operation['max_intensity']
                    break
            
            # Build filename with all adjustment parameters
            contrast_str = f"c{self.current_contrast:.2f}".replace('.', 'p')
            brightness_str = f"b{self.current_brightness:+d}"
            
            # Add intensity thresholds if available
            if min_intensity is not None and max_intensity is not None:
                intensity_str = f"_min{min_intensity}_max{max_intensity}"
            else:
                intensity_str = ""
            
            filename = f"{base_name}_pixel_removed_{contrast_str}_{brightness_str}{intensity_str}_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, self.current_image)
        return output_path
    
    def adjust_contrast(self, contrast_factor: float = 1.0, brightness_offset: int = 0):
        """
        Adjust contrast and brightness of the current image
        Always applies adjustments from the baseline image to avoid cumulative effects
        
        Args:
            contrast_factor: Contrast multiplier (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)
            brightness_offset: Brightness offset (-100 to +100, 0 = no change)
        """
        if self.baseline_image is None:
            return False
        
        # Save current state before applying changes (for undo)
        self.image_states.append(self.current_image.copy())
        
        # Limit the number of states to prevent memory issues (keep last 10)
        if len(self.image_states) > 10:
            self.image_states.pop(0)
        
        # Apply contrast and brightness adjustment from baseline image
        # Formula: new_pixel = contrast_factor * old_pixel + brightness_offset
        adjusted = cv2.convertScaleAbs(self.baseline_image, alpha=contrast_factor, beta=brightness_offset)
        self.current_image = adjusted
        
        # Also update threshold base image with contrast adjustments (non-threshold operation)
        self.threshold_base_image = adjusted.copy()
        
        # Update contrast/brightness tracking
        self.current_contrast = contrast_factor
        self.current_brightness = brightness_offset
        
        # Save to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.removal_history.append({
            'timestamp': timestamp,
            'method': 'contrast_adjustment',
            'contrast_factor': float(contrast_factor),
            'brightness_offset': int(brightness_offset),
            'description': f"Contrast: {contrast_factor:.2f}, Brightness: {brightness_offset:+d}"
        })
        
        return True
    
    def remove_pixels_by_threshold(self, min_intensity: int, max_intensity: int, 
                                 removal_method: str = "black", color_space: str = "gray", 
                                 threshold_mode: str = "cumulative"):
        """
        Remove pixels within specified intensity threshold range
        
        Args:
            min_intensity: Minimum intensity value (0-255)
            max_intensity: Maximum intensity value (0-255)
            removal_method: How to handle removed pixels
            color_space: Color space for analysis ('gray', 'hsv', 'lab')
            threshold_mode: 'cumulative' (builds on current) or 'independent' (from original)
        """
        if self.current_image is None or self.threshold_extractor is None:
            return False
        
        # Save current state before applying changes (for undo)
        self.image_states.append(self.current_image.copy())
        
        # Limit the number of states to prevent memory issues (keep last 10)
        if len(self.image_states) > 10:
            self.image_states.pop(0)
        
        # Perform threshold extraction
        try:
            if threshold_mode == "independent":
                # Independent mode: Reset to threshold base image and apply only current extraction
                # Use threshold base image (image without any threshold extractions)
                source_image = self.threshold_base_image.copy()
                self.threshold_extractor.current_image = source_image
                
                # Perform extraction on the threshold base image
                result_image, extraction_info = self.threshold_extractor.extract_by_intensity_range(
                    min_intensity, max_intensity, removal_method, color_space
                )
                
                # Independent mode: Use the extraction result directly (reset + extract)
                self.current_image = result_image
                
            else:
                # Cumulative mode: Use current image and build on previous operations
                source_image = self.current_image.copy()
                self.threshold_extractor.current_image = source_image
                
                # Perform extraction on the current image
                result_image, extraction_info = self.threshold_extractor.extract_by_intensity_range(
                    min_intensity, max_intensity, removal_method, color_space
                )
                
                # Cumulative mode: directly use the result
                self.current_image = result_image
                # Update baseline image after threshold removal
                self.baseline_image = self.current_image.copy()
                # NOTE: Do NOT update threshold_base_image for threshold operations
                # threshold_base_image should only contain non-threshold modifications
            
            # Save to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.removal_history.append({
                'timestamp': timestamp,
                'method': 'threshold_extraction',
                'threshold_mode': threshold_mode,
                'min_intensity': extraction_info['min_intensity'],
                'max_intensity': extraction_info['max_intensity'],
                'affected_pixels': extraction_info['extracted_pixels'],
                'extraction_percentage': extraction_info['extraction_percentage'],
                'removal_method': removal_method,
                'color_space': color_space,
                'description': f"{threshold_mode.capitalize()} threshold {min_intensity}-{max_intensity} ({removal_method})"
            })
            
            return True
            
        except Exception as e:
            print(f"Threshold extraction error: {e}")
            return False
    
    def get_intensity_histogram(self, color_space: str = "gray"):
        """Get intensity histogram for the current image"""
        if self.current_image is None:
            return None
        
        try:
            if self.threshold_extractor is None:
                self.threshold_extractor = ThresholdExtractor()
            
            self.threshold_extractor.current_image = self.current_image.copy()
            hist, bins = self.threshold_extractor.get_intensity_histogram(color_space)
            
            return {
                'histogram': hist.tolist(),
                'bins': bins.tolist(),
                'color_space': color_space
            }
        except Exception as e:
            print(f"Histogram error: {e}")
            return None
    
    def get_image_as_base64(self):
        """Convert current image to base64 for web display"""
        if self.current_image is None:
            return None
        
        # Convert BGR to RGB for proper web display
        rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{image_base64}"

# Global engine instance
engine = PixelRemovalEngine()

@app.route('/')
def index():
    """Main page"""
    return render_template('pixel_removal.html')

@app.route('/load_image', methods=['POST'])
def load_image():
    """Load an image for processing"""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '')
        
        # Default to overfocus.jpg if no path provided
        if not image_path:
            image_path = '/home/jimmy/code/2Dto3D_2/data/input/BF image/overfocus.jpg'
        
        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join('/home/jimmy/code/2Dto3D_2', image_path)
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': f'Image not found: {image_path}'})
        
        engine.load_image(image_path)
        image_base64 = engine.get_image_as_base64()
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'image_path': image_path,
            'dimensions': {
                'width': int(engine.current_image.shape[1]),
                'height': int(engine.current_image.shape[0])
            },
            'current_contrast': engine.current_contrast,
            'current_brightness': engine.current_brightness
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
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
            'current_contrast': engine.current_contrast,
            'current_brightness': engine.current_brightness
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/remove_pixels', methods=['POST'])
def remove_pixels():
    """Sample pixel intensity from selected area, then remove all similar pixels globally"""
    try:
        data = request.get_json()
        selection_data = data.get('selection', {})
        removal_method = data.get('method', 'black')
        
        success = engine.remove_pixels_in_area(selection_data, removal_method)
        
        if success:
            image_base64 = engine.get_image_as_base64()
            return jsonify({
                'success': True,
                'image': image_base64,
                'history': engine.removal_history[-5:]  # Last 5 operations
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to remove pixels'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/undo_last', methods=['POST'])
def undo_last():
    """Undo last pixel removal operation"""
    try:
        success, can_undo_more = engine.undo_last_operation()
        if success:
            image_base64 = engine.get_image_as_base64()
            return jsonify({
                'success': True,
                'image': image_base64,
                'history': engine.removal_history[-5:],  # Last 5 operations
                'can_undo_more': can_undo_more,
                'message': 'Last operation undone'
            })
        else:
            return jsonify({'success': False, 'error': 'No operations to undo'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_image', methods=['POST'])
def reset_image():
    """Reset to original image"""
    try:
        success = engine.reset_image()
        if success:
            image_base64 = engine.get_image_as_base64()
            return jsonify({
                'success': True,
                'image': image_base64,
                'message': 'Image reset to original',
                'current_contrast': engine.current_contrast,
                'current_brightness': engine.current_brightness
            })
        else:
            return jsonify({'success': False, 'error': 'No original image to reset to'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_image', methods=['POST'])
def save_image():
    """Save current image"""
    try:
        data = request.get_json()
        filename = data.get('filename', None)
        
        saved_path = engine.save_current_image(filename)
        if saved_path:
            return jsonify({
                'success': True,
                'saved_path': saved_path,
                'message': f'Image saved to {saved_path}'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save image'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/adjust_contrast', methods=['POST'])
def adjust_contrast():
    """Adjust contrast and brightness of the current image"""
    try:
        data = request.get_json()
        contrast_factor = data.get('contrast', 1.0)
        brightness_offset = data.get('brightness', 0)
        
        # Validate input ranges
        contrast_factor = max(0.1, min(3.0, float(contrast_factor)))  # Clamp between 0.1 and 3.0
        brightness_offset = max(-100, min(100, int(brightness_offset)))  # Clamp between -100 and +100
        
        success = engine.adjust_contrast(contrast_factor, brightness_offset)
        
        if success:
            image_base64 = engine.get_image_as_base64()
            return jsonify({
                'success': True,
                'image': image_base64,
                'applied_contrast': contrast_factor,
                'applied_brightness': brightness_offset,
                'history': engine.removal_history[-5:],  # Last 5 operations
                'message': f'Contrast adjusted to {contrast_factor:.2f}, brightness {brightness_offset:+d}'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to adjust contrast - no image loaded'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/remove_by_threshold', methods=['POST'])
def remove_by_threshold():
    """Remove pixels within specified intensity threshold range"""
    try:
        data = request.get_json()
        min_intensity = data.get('min_intensity', 0)
        max_intensity = data.get('max_intensity', 255)
        removal_method = data.get('method', 'black')
        color_space = data.get('color_space', 'gray')
        threshold_mode = data.get('threshold_mode', 'cumulative')
        
        # Validate inputs
        min_intensity = max(0, min(255, int(min_intensity)))
        max_intensity = max(0, min(255, int(max_intensity)))
        
        if min_intensity > max_intensity:
            min_intensity, max_intensity = max_intensity, min_intensity
        
        # Validate threshold mode
        if threshold_mode not in ['cumulative', 'independent']:
            threshold_mode = 'cumulative'
        
        success = engine.remove_pixels_by_threshold(min_intensity, max_intensity, removal_method, color_space, threshold_mode)
        
        if success:
            image_base64 = engine.get_image_as_base64()
            return jsonify({
                'success': True,
                'image': image_base64,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'removal_method': removal_method,
                'color_space': color_space,
                'threshold_mode': threshold_mode,
                'history': engine.removal_history[-5:],  # Last 5 operations
                'message': f'{threshold_mode.capitalize()} threshold: pixels removed in range {min_intensity}-{max_intensity}'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to remove pixels by threshold'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_histogram', methods=['POST'])
def get_histogram():
    """Get intensity histogram for current image"""
    try:
        data = request.get_json()
        color_space = data.get('color_space', 'gray')
        
        histogram_data = engine.get_intensity_histogram(color_space)
        
        if histogram_data:
            return jsonify({
                'success': True,
                'histogram_data': histogram_data
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to generate histogram'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_history', methods=['GET'])
def get_history():
    """Get removal history"""
    return jsonify({
        'success': True,
        'history': engine.removal_history,
        'total_operations': len(engine.removal_history)
    })

@app.route('/radial_analysis', methods=['POST'])
def radial_analysis():
    """Perform radial intensity profile analysis"""
    try:
        data = request.get_json()
        detection_method = data.get('detection_method', 'hough')
        step_angle = data.get('step_angle', 15)
        min_radius = data.get('min_radius', 20)
        max_radius = data.get('max_radius', 150)
        
        if engine.current_image is None or engine.radial_analyzer is None:
            return jsonify({'success': False, 'error': 'No image loaded for analysis'})
        
        # Update radial analyzer with current image
        engine.radial_analyzer.load_image(engine.current_image.copy())
        
        # Detect droplets based on method
        if detection_method == 'hough':
            droplets = engine.radial_analyzer.detect_droplets_hough(min_radius, max_radius)
        elif detection_method == 'blob':
            droplets = engine.radial_analyzer.detect_droplets_blob(min_radius, max_radius)
        elif detection_method == 'manual':
            # For manual, we'd need center coordinates from the user
            # For now, use image center as fallback
            height, width = engine.current_image.shape[:2]
            center_x, center_y = width // 2, height // 2
            estimated_radius = min(width, height) // 4
            droplets = engine.radial_analyzer.detect_droplets_manual(center_x, center_y, estimated_radius)
        else:
            return jsonify({'success': False, 'error': f'Unknown detection method: {detection_method}'})
        
        if not droplets:
            return jsonify({
                'success': True,
                'droplets_count': 0,
                'droplets': [],
                'detection_method': detection_method,
                'message': 'No droplets detected'
            })
        
        # Perform radial analysis on detected droplets
        analysis_results = engine.radial_analyzer.analyze_all_droplets(step_angle)
        
        # Get summary statistics
        summary = engine.radial_analyzer.get_analysis_summary()
        
        return jsonify({
            'success': True,
            'droplets_count': len(droplets),
            'droplets': droplets,
            'detection_method': detection_method,
            'analysis_results': analysis_results,
            'summary': summary,
            'message': f'Found {len(droplets)} droplet(s)'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_radial_profile', methods=['POST'])
def get_radial_profile():
    """Get radial profile data for visualization"""
    try:
        if engine.radial_analyzer is None or not engine.radial_analyzer.radial_profiles:
            return jsonify({'success': False, 'error': 'No radial analysis data available'})
        
        # Extract profile data for visualization
        profiles = []
        for result in engine.radial_analyzer.radial_profiles:
            if 'radial_profile' in result:
                profiles.append(result['radial_profile'])
        
        return jsonify({
            'success': True,
            'profile_data': {
                'profiles': profiles,
                'summary': engine.radial_analyzer.get_analysis_summary()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_radial_visualization', methods=['POST'])
def get_radial_visualization():
    """Get visualization image with detected circles and analysis overlay"""
    try:
        data = request.get_json()
        show_radial_lines = data.get('show_radial_lines', True)
        show_rings = data.get('show_rings', True)
        show_labels = data.get('show_labels', True)
        
        if engine.radial_analyzer is None or not engine.radial_analyzer.radial_profiles:
            return jsonify({'success': False, 'error': 'No radial analysis data available'})
        
        # Create visualization
        vis_image = engine.radial_analyzer.create_visualization(
            show_radial_lines=show_radial_lines,
            show_rings=show_rings,
            show_labels=show_labels
        )
        
        if vis_image.size == 0:
            return jsonify({'success': False, 'error': 'Failed to create visualization'})
        
        # Convert to base64 for web display
        rgb_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'visualization_image': f"data:image/png;base64,{image_base64}",
            'droplets_count': len(engine.radial_analyzer.radial_profiles),
            'summary': engine.radial_analyzer.get_analysis_summary()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/sam_segmentation', methods=['POST'])
def sam_segmentation():
    """Perform SAM-based droplet segmentation and auto-display masks"""
    try:
        if engine.current_image is None or engine.sam_analyzer is None:
            return jsonify({'success': False, 'error': 'No image loaded for SAM analysis'})
        
        # Update SAM analyzer with current image
        engine.sam_analyzer.load_image(engine.current_image.copy())
        
        # Perform segmentation using SAM
        mask_stats = engine.sam_analyzer.segment_droplets(method="sam")
        
        if not mask_stats:
            return jsonify({
                'success': True,
                'masks_count': 0,
                'masks': [],
                'overlay_image': None,
                'message': 'No droplets segmented with current filters'
            })
        
        # Automatically create and return mask overlay
        overlay_image = engine.sam_analyzer.create_mask_overlay(
            show_labels=True,
            alpha=0.3
        )
        
        if overlay_image.size > 0:
            # Convert to base64 for web display
            rgb_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            overlay_b64 = f"data:image/png;base64,{image_base64}"
        else:
            overlay_b64 = None
        
        # Get summary statistics
        summary = engine.sam_analyzer.get_segmentation_summary()
        
        return jsonify({
            'success': True,
            'masks_count': len(mask_stats),
            'masks': mask_stats,
            'overlay_image': overlay_b64,
            'summary': summary,
            'message': f'Segmented {len(mask_stats)} high-quality droplet(s)'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_sam_visualization', methods=['POST'])
def get_sam_visualization():
    """Get SAM mask visualization overlay"""
    try:
        data = request.get_json()
        show_labels = data.get('show_labels', True)
        alpha = data.get('alpha', 0.3)
        
        if engine.sam_analyzer is None or not engine.sam_analyzer.masks:
            return jsonify({'success': False, 'error': 'No SAM segmentation data available'})
        
        # Create mask overlay
        overlay_image = engine.sam_analyzer.create_mask_overlay(
            show_labels=show_labels,
            alpha=alpha
        )
        
        if overlay_image.size == 0:
            return jsonify({'success': False, 'error': 'Failed to create mask overlay'})
        
        # Convert to base64 for web display
        rgb_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'overlay_image': f"data:image/png;base64,{image_base64}",
            'masks_count': len(engine.sam_analyzer.masks),
            'summary': engine.sam_analyzer.get_segmentation_summary()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_mask_at_point', methods=['POST'])
def get_mask_at_point():
    """Get mask statistics at a specific point for hover preview"""
    try:
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        if engine.sam_analyzer is None:
            return jsonify({'success': False, 'error': 'No SAM analyzer available'})
        
        # Get mask at point
        mask_stats = engine.sam_analyzer.get_mask_at_point(int(x), int(y))
        
        if mask_stats:
            return jsonify({
                'success': True,
                'has_mask': True,
                'mask_stats': mask_stats
            })
        else:
            return jsonify({
                'success': True,
                'has_mask': False,
                'mask_stats': None
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/toggle_mask', methods=['POST'])
def toggle_mask():
    """Toggle the state of a mask at a specific point"""
    try:
        data = request.get_json()
        x = data.get('x', 0)
        y = data.get('y', 0)
        
        if engine.sam_analyzer is None:
            return jsonify({'success': False, 'error': 'No SAM analyzer available'})
        
        # Toggle mask at point
        toggle_result = engine.sam_analyzer.toggle_mask_state(int(x), int(y))
        
        if toggle_result:
            # Create updated visualization
            overlay_image = engine.sam_analyzer.create_mask_overlay(
                show_labels=True,
                alpha=0.3
            )
            
            if overlay_image.size > 0:
                # Convert to base64 for web display
                rgb_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                overlay_b64 = f"data:image/png;base64,{image_base64}"
            else:
                overlay_b64 = None
            
            return jsonify({
                'success': True,
                'mask_toggled': True,
                'toggle_info': toggle_result,
                'overlay_image': overlay_b64,
                'message': f"Mask {toggle_result['mask_id']+1} {toggle_result['new_state']}"
            })
        else:
            return jsonify({
                'success': True,
                'mask_toggled': False,
                'message': 'No mask found at clicked location'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs(templates_dir, exist_ok=True)
    
    print("🚀 Starting Pixel Removal Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📁 Results will be saved to: results/pixel_removal/")
    print("🎯 Default image: overfocus.jpg")
    print()
    
    try:
        app.run(host='127.0.0.1', port=8000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("📝 Make sure port 5000 is available")
        import traceback
        traceback.print_exc()
