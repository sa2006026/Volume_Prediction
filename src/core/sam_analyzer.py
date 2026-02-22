#!/usr/bin/env python3
"""
SAM (Segment Anything Model) based droplet segmentation using Meta's official SAM model
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import os
import sys
import torch
from pathlib import Path

# Add the sam_droplet directory to path to import SAM functionality
sam_droplet_path = Path(__file__).parent.parent.parent.parent / "sam_droplet"
if sam_droplet_path.exists():
    sys.path.append(str(sam_droplet_path))

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: segment-anything not available. Using fallback segmentation.")
    SAM_AVAILABLE = False

class SAMAnalyzer:
    """Meta's SAM-based droplet segmentation analyzer"""
    
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.masks = []
        self.mask_statistics = []
        self.sam_model = None
        self.mask_generator = None
        self.sam_initialized = False
        self.mask_states = []  # Track active/removed state for each mask
        
        # Unit conversion settings
        self.conversion_enabled = False
        self.pixels_per_unit = 1.0  # pixels per unit (e.g., pixels per μm)
        self.unit_name = "μm"
        self.pixel_distance = 0.0  # reference pixel distance
        self.unit_distance = 0.0   # corresponding unit distance
        
        # Initialize SAM model
        self._initialize_sam()
    
    def _initialize_sam(self):
        """Initialize the Meta SAM model using existing checkpoints"""
        if not SAM_AVAILABLE:
            print("SAM not available, using fallback methods")
            return
            
        try:
            print("Initializing Meta's SAM model...")
            
            # Look for SAM model in the sam_droplet directory
            model_files = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth", 
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            
            # Check local model directory first, then sam_droplet
            local_model_dir = Path(__file__).parent.parent.parent / "model"
            sam_model_dir = sam_droplet_path / "model"
            
            model_dirs = [local_model_dir, sam_model_dir]
            available_model = None
            model_type = None
            
            # Priority order: vit_b (smallest), vit_l, vit_h
            for mtype in ["vit_b", "vit_l", "vit_h"]:
                filename = model_files[mtype]
                for model_dir in model_dirs:
                    model_path = model_dir / filename
                    if model_path.exists():
                        available_model = model_path
                        model_type = mtype
                        print(f"Found SAM model: {mtype} at {model_path}")
                        break
                if available_model:
                    break
            
            if not available_model:
                print("No SAM model checkpoints found in sam_droplet/model directory")
                return
            
            # Initialize model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading SAM model on device: {device}")
            
            self.sam_model = sam_model_registry[model_type](checkpoint=str(available_model))
            self.sam_model.to(device=device)
            
            # Create automatic mask generator with optimized settings for droplets
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=32,           # Good balance for droplet detection
                crop_n_layers=1,              # Multiple crop scales
                min_mask_region_area=1,     # Filter out very small segments
                stability_score_thresh=0.85,  # High quality masks only
                box_nms_thresh=0.7           # Remove overlapping detections
            )
            
            self.sam_initialized = True
            print("Meta SAM model initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize SAM model: {e}")
            self.sam_initialized = False
        
    def load_image(self, image: np.ndarray):
        """Load image for SAM analysis"""
        self.image = image.copy()
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.masks = []
        self.mask_statistics = []
        self.mask_states = []
    
    def _sam_segmentation(self) -> List[np.ndarray]:
        """
        Use Meta's SAM model for automatic segmentation
        """
        if not self.sam_initialized or self.mask_generator is None:
            print("SAM model not initialized, using fallback segmentation")
            return self._fallback_segmentation()
        
        if self.image is None:
            return []
        
        try:
            print("Running SAM automatic mask generation...")
            
            # Use RGB image for SAM (convert from BGR if needed)
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # Generate masks using SAM
            sam_masks = self.mask_generator.generate(rgb_image)
            
            print(f"SAM generated {len(sam_masks)} masks")
            
            # Convert SAM masks to our format with quality filters
            masks = []
            for sam_mask in sam_masks:
                # Extract boolean segmentation mask
                segmentation = sam_mask['segmentation']
                
                # Convert boolean to uint8 mask
                mask = (segmentation * 255).astype(np.uint8)
                
                # Filter by area - only keep reasonably sized droplets
                area = sam_mask.get('area', 0)
                if not (0 <= area <= 5000000):
                    continue
                
                # Edge removal filter - reject masks touching image borders
                if self._mask_touches_edge(mask):
                    continue
                
                # Circularity filter - only keep reasonably circular shapes
                circularity = self._calculate_circularity(mask)
                if circularity < 0.75:
                    continue
                
                masks.append(mask)
            
            print(f"Filtered to {len(masks)} viable droplet masks")
            return masks
            
        except Exception as e:
            print(f"SAM segmentation failed: {e}")
            print("Falling back to traditional methods")
            return self._fallback_segmentation()
    
    def _fallback_segmentation(self) -> List[np.ndarray]:
        """
        Fallback segmentation when SAM is not available
        """
        if self.gray_image is None:
            return []
        
        masks = []
        
        # Use Hough circles as fallback
        circles = cv2.HoughCircles(
            self.gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Create circular mask
                mask = np.zeros_like(self.gray_image)
                cv2.circle(mask, (x, y), r, 255, -1)
                masks.append(mask)
        
        return masks
    
    def _mask_touches_edge(self, mask: np.ndarray) -> bool:
        """
        Check if a mask touches the image edges (with margin for safety)
        
        Args:
            mask: Binary mask (0 or 255)
            
        Returns:
            True if mask touches any edge or is very close to edges
        """
        h, w = mask.shape[:2]
        
        # Use a small margin (2 pixels) from edges for more strict filtering
        margin = 2
        
        # Check if any pixels near the border are non-zero
        top_edge = np.any(mask[:margin, :] > 0)
        bottom_edge = np.any(mask[h-margin:, :] > 0)
        left_edge = np.any(mask[:, :margin] > 0)
        right_edge = np.any(mask[:, w-margin:] > 0)
        
        return top_edge or bottom_edge or left_edge or right_edge
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """
        Calculate circularity of a mask
        
        Args:
            mask: Binary mask (0 or 255)
            
        Returns:
            Circularity value (0-1, where 1 is perfect circle)
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Circularity = 4π * area / perimeter²
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        
        return circularity
    
    def segment_droplets(self, method: str = "sam", apply_overlap_filter: bool = True, overlap_threshold: float = 0.4) -> List[Dict]:
        """
        Segment droplets using Meta's SAM model or fallback methods with automatic quality filtering
        
        Args:
            method: Segmentation method ("sam", "fallback")
            apply_overlap_filter: Whether to automatically apply overlap filtering
            overlap_threshold: Overlap threshold for duplicate detection (0.0-1.0)
            
        Returns:
            List of mask dictionaries with statistics
        """
        if method == "sam":
            masks = self._sam_segmentation()
        else:
            masks = self._fallback_segmentation()
        
        self.masks = masks
        self.mask_statistics = []
        self.mask_states = []
        
        # Calculate statistics for each mask and initialize states
        for i, mask in enumerate(masks):
            stats = self._calculate_mask_statistics(mask, i)
            self.mask_statistics.append(stats)
            self.mask_states.append('active')  # All masks start as active
        
        # Automatically apply overlap filter during segmentation
        if apply_overlap_filter and len(self.masks) > 1:
            print(f"🔧 Applying automatic overlap filter (threshold: {overlap_threshold})")
            overlap_results = self.apply_overlap_filter(overlap_threshold)
            if overlap_results['success']:
                print(f"✅ Overlap filter: Kept {overlap_results['kept_count']}, removed {overlap_results['removed_count']} duplicate masks")
        
        # Add state information to each mask statistic for frontend filtering
        for i, stats in enumerate(self.mask_statistics):
            if i < len(self.mask_states):
                stats['state'] = self.mask_states[i]
            else:
                stats['state'] = 'active'
        
        return self.mask_statistics
    
    def _calculate_mask_statistics(self, mask: np.ndarray, mask_id: int) -> Dict:
        """
        Calculate comprehensive statistics for a mask
        
        Args:
            mask: Binary mask
            mask_id: Unique identifier for the mask
            
        Returns:
            Dictionary with mask statistics
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum enclosing circle
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        diameter = radius * 2
        
        # Circularity
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Solidity (convex hull area vs contour area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Extent (contour area vs bounding box area)
        extent = float(area) / (w * h) if (w * h) > 0 else 0
        
        # Equivalent diameter (diameter of circle with same area)
        equiv_diameter = np.sqrt(4 * area / np.pi)
        
        # Mean intensity within mask (from grayscale)
        mean_intensity = np.mean(self.gray_image[mask > 0]) if np.any(mask > 0) else 0
        
        # Calculate red and green mean intensities from BGR image
        red_mean_intensity = 0
        green_mean_intensity = 0
        if self.image is not None and np.any(mask > 0):
            # OpenCV uses BGR format: channel 0=Blue, 1=Green, 2=Red
            mask_pixels = mask > 0
            red_mean_intensity = np.mean(self.image[:, :, 2][mask_pixels]) if np.any(mask_pixels) else 0
            green_mean_intensity = np.mean(self.image[:, :, 1][mask_pixels]) if np.any(mask_pixels) else 0
        
        return {
            'mask_id': mask_id,
            'area': float(area),
            'perimeter': float(perimeter),
            'center_x': float(center_x),
            'center_y': float(center_y),
            'bounding_box': [int(x), int(y), int(w), int(h)],
            'diameter': float(diameter),
            'radius': float(radius),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'solidity': float(solidity),
            'extent': float(extent),
            'equivalent_diameter': float(equiv_diameter),
            'mean_intensity': float(mean_intensity),
            'red_mean_intensity': float(red_mean_intensity),
            'green_mean_intensity': float(green_mean_intensity),
            'enclosing_circle': [float(circle_x), float(circle_y), float(radius)]
        }
    
    def get_mask_at_point(self, x: int, y: int) -> Optional[Dict]:
        """
        Get mask and statistics at a specific point
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Mask statistics with state info if point is within a mask, None otherwise
        """
        for i, mask in enumerate(self.masks):
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
                mask_data = self.mask_statistics[i].copy()
                mask_data['state'] = self.mask_states[i] if i < len(self.mask_states) else 'active'
                mask_data['clickable'] = True
                return mask_data
        return None
    
    def toggle_mask_state(self, x: int, y: int) -> Optional[Dict]:
        """
        Toggle the state of a mask at a specific point
        
        Args:
            x, y: Coordinates to check
            
        Returns:
            Updated mask info if toggled, None if no mask at point
        """
        for i, mask in enumerate(self.masks):
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0:
                # Toggle state
                current_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
                new_state = 'removed' if current_state == 'active' else 'active'
                
                if i < len(self.mask_states):
                    self.mask_states[i] = new_state
                else:
                    # Extend states if needed
                    while len(self.mask_states) <= i:
                        self.mask_states.append('active')
                    self.mask_states[i] = new_state
                
                return {
                    'mask_id': i,
                    'old_state': current_state,
                    'new_state': new_state,
                    'center_x': self.mask_statistics[i]['center_x'],
                    'center_y': self.mask_statistics[i]['center_y'],
                    'diameter': self.mask_statistics[i]['diameter']
                }
        return None
    
    def create_mask_overlay(self, show_labels: bool = False, alpha: float = 0.3) -> np.ndarray:
        """
        Create visualization overlay with all masks
        
        Args:
            show_labels: Whether to show mask labels
            alpha: Transparency of mask overlay
            
        Returns:
            Image with mask overlay
        """
        if self.image is None:
            return np.array([])
        
        overlay = self.image.copy()
        mask_overlay = np.zeros_like(self.image)
        
        # All masks are red initially
        red_color = (0, 0, 255)  # Red in BGR format (OpenCV uses BGR)
        
        for i, (mask, stats) in enumerate(zip(self.masks, self.mask_statistics)):
            if not stats:
                continue
            
            # Get mask state
            mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            
            if mask_state == 'active':
                color = red_color
                mask_alpha = alpha
                contour_thickness = 2
                text_color = (255, 255, 255)
            elif mask_state in ['intensity_filtered', 'overlap_filtered']:
                # Skip filtered masks - don't display them at all
                continue
            else:  # removed
                color = red_color  # Keep red color but will be dashed
                mask_alpha = alpha * 0.2  # Much more transparent for removed masks
                contour_thickness = 2
                text_color = (200, 200, 200)
            
            # Create colored mask
            colored_mask = np.zeros_like(self.image)
            colored_mask[mask > 0] = color
            
            # Add to overlay
            mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, mask_alpha, 0)
            
            # Draw bounding box instead of contour
            bbox = stats['bounding_box']  # [x, y, w, h]
            x, y, w, h = bbox
            
            if mask_state == 'removed':
                # Draw dashed bounding box for removed masks with shorter dashes for more visibility
                self._draw_dashed_rectangle(overlay, (x, y), (x + w, y + h), color, contour_thickness + 1, dash_length=8)
            else:
                # Draw solid bounding box for active masks
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, contour_thickness)
        
        # Combine overlay with original image
        result = cv2.addWeighted(overlay, 1, mask_overlay, alpha, 0)
        
        return result
    
    def create_filtered_mask_overlay(self, show_labels: bool = False, alpha: float = 0.3) -> np.ndarray:
        """
        Create visualization overlay with clean approach: clear all bounding boxes first,
        then add back only the ones that meet all quality criteria (intensity and overlap filters)
        
        Args:
            show_labels: Whether to show mask labels
            alpha: Transparency of mask overlay
            
        Returns:
            Image with filtered mask overlay (only non-filtered masks)
        """
        if self.image is None:
            return np.array([])
        
        # Step 1: Start with clean original image (no existing bounding boxes)
        overlay = self.image.copy()
        mask_overlay = np.zeros_like(self.image)
        
        # Count masks by state for debugging
        state_counts = {}
        for state in self.mask_states:
            state_counts[state] = state_counts.get(state, 0) + 1
        print(f"📊 Creating overlay with mask states: {state_counts}")
        
        # Step 2: Only add back masks and bounding boxes that pass all filters
        red_color = (0, 0, 255)  # Red in BGR format (OpenCV uses BGR)
        drawn_count = 0
        
        for i, (mask, stats) in enumerate(zip(self.masks, self.mask_statistics)):
            if not stats:
                continue
            
            # Get mask state
            mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            
            # Step 3: Only process masks that are NOT filtered (intensity or overlap)
            # This ensures we only add back the masks that meet all quality criteria
            if mask_state in ['intensity_filtered', 'overlap_filtered']:
                continue  # Skip completely - don't add this mask or its bounding box
            
            drawn_count += 1
            
            # Step 4: Add back the qualifying masks with their visual properties
            if mask_state == 'active':
                color = red_color
                mask_alpha = alpha
                contour_thickness = 2
                text_color = (255, 255, 255)
            else:  # removed
                color = red_color  # Keep red color but will be dashed
                mask_alpha = alpha * 0.2  # Much more transparent for removed masks
                contour_thickness = 2
                text_color = (200, 200, 200)
            
            # Step 5: Create colored mask overlay for qualifying masks
            colored_mask = np.zeros_like(self.image)
            colored_mask[mask > 0] = color
            
            # Add to overlay
            mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, mask_alpha, 0)
            
            # Step 6: Draw bounding box only for qualifying masks
            bbox = stats['bounding_box']  # [x, y, w, h]
            x, y, w, h = bbox
            
            if mask_state == 'removed':
                # Draw dashed bounding box for removed masks
                self._draw_dashed_rectangle(overlay, (x, y), (x + w, y + h), color, contour_thickness + 1, dash_length=8)
            else:
                # Draw solid bounding box for active masks
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, contour_thickness)
        
        # Step 7: Combine the clean overlay with original image
        result = cv2.addWeighted(overlay, 1, mask_overlay, alpha, 0)
        
        print(f"✅ Drew {drawn_count} bounding boxes on overlay (out of {len(self.masks)} total masks)")
        
        return result
    
    def get_segmentation_summary(self) -> Dict:
        """
        Get summary statistics of all segmented masks
        
        Returns:
            Summary dictionary
        """
        if not self.mask_statistics:
            return {}
        
        areas = [stats['area'] for stats in self.mask_statistics]
        diameters = [stats['diameter'] for stats in self.mask_statistics]
        circularities = [stats['circularity'] for stats in self.mask_statistics]
        
        return {
            'total_masks': len(self.mask_statistics),
            'average_area': np.mean(areas) if areas else 0,
            'average_diameter': np.mean(diameters) if diameters else 0,
            'average_circularity': np.mean(circularities) if circularities else 0,
            'min_diameter': np.min(diameters) if diameters else 0,
            'max_diameter': np.max(diameters) if diameters else 0,
            'std_diameter': np.std(diameters) if diameters else 0
        }
    
    def create_mask_preview(self, mask_id: int, preview_size: tuple = (150, 150)) -> Optional[np.ndarray]:
        """
        Create a preview image of a specific mask for hover display
        
        Args:
            mask_id: Index of the mask to preview
            preview_size: Size of the preview image (width, height)
            
        Returns:
            Preview image as numpy array, or None if mask doesn't exist
        """
        if (mask_id < 0 or mask_id >= len(self.masks) or 
            mask_id >= len(self.mask_statistics) or 
            self.image is None):
            return None
        
        mask = self.masks[mask_id]
        stats = self.mask_statistics[mask_id]
        
        # Get bounding box with some padding
        bbox = stats['bounding_box']  # [x, y, w, h]
        x, y, w, h = bbox
        
        # Add padding around the bounding box
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(self.image.shape[1], x + w + padding)
        y_end = min(self.image.shape[0], y + h + padding)
        
        # Extract the region of interest from original image
        roi_image = self.image[y_start:y_end, x_start:x_end].copy()
        roi_mask = mask[y_start:y_end, x_start:x_end]
        
        if roi_image.size == 0 or roi_mask.size == 0:
            return None
        
        # Create mask overlay on the ROI
        mask_overlay = np.zeros_like(roi_image)
        mask_color = (0, 0, 255)  # Red color for the mask
        mask_overlay[roi_mask > 0] = mask_color
        
        # Blend the original ROI with mask overlay
        alpha = 0.4
        preview_image = cv2.addWeighted(roi_image, 1 - alpha, mask_overlay, alpha, 0)
        
        # Draw bounding box on the preview (adjust coordinates to ROI)
        roi_bbox_x = x - x_start
        roi_bbox_y = y - y_start
        cv2.rectangle(preview_image, 
                     (roi_bbox_x, roi_bbox_y), 
                     (roi_bbox_x + w, roi_bbox_y + h), 
                     mask_color, 2)
        
        # Resize to preview size while maintaining aspect ratio
        h_roi, w_roi = preview_image.shape[:2]
        target_w, target_h = preview_size
        
        # Calculate scaling factor to fit within target size
        scale = min(target_w / w_roi, target_h / h_roi)
        new_w = int(w_roi * scale)
        new_h = int(h_roi * scale)
        
        # Resize the preview
        preview_resized = cv2.resize(preview_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a canvas with the target size and center the preview
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas.fill(32)  # Dark gray background
        
        # Calculate position to center the preview
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = preview_resized
        
        # Add a border around the canvas
        border_color = (100, 100, 100)  # Light gray border
        cv2.rectangle(canvas, (0, 0), (target_w - 1, target_h - 1), border_color, 2)
        
        return canvas
    
    def _draw_dashed_contours(self, image, contours, color, thickness, dash_length=8):
        """
        Draw dashed contours for removed masks
        
        Args:
            image: Image to draw on
            contours: List of contours
            color: Color for the dashed lines
            thickness: Line thickness
            dash_length: Length of each dash
        """
        for contour in contours:
            # Convert contour to a list of points
            points = contour.reshape(-1, 2)
            
            # Calculate total perimeter
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Draw dashed line along the contour
            num_points = len(points)
            if num_points < 2:
                continue
            
            # Calculate cumulative distances along the contour
            distances = [0]
            for i in range(1, num_points):
                dist = np.linalg.norm(points[i] - points[i-1])
                distances.append(distances[-1] + dist)
            
            # Add the distance from last point back to first (closed contour)
            final_dist = np.linalg.norm(points[0] - points[-1])
            total_distance = distances[-1] + final_dist
            
            # Draw dashed pattern
            current_distance = 0
            draw_dash = True
            
            while current_distance < total_distance:
                next_distance = current_distance + dash_length
                
                # Find points at current_distance and next_distance
                start_point = self._get_point_at_distance(points, distances, current_distance, total_distance)
                end_point = self._get_point_at_distance(points, distances, next_distance, total_distance)
                
                if draw_dash:
                    cv2.line(image, tuple(start_point.astype(int)), tuple(end_point.astype(int)), 
                            color, thickness)
                
                draw_dash = not draw_dash
                current_distance = next_distance
    
    def _get_point_at_distance(self, points, distances, target_distance, total_distance):
        """
        Get the point at a specific distance along the contour
        
        Args:
            points: Array of contour points
            distances: Cumulative distances
            target_distance: Target distance along contour
            total_distance: Total perimeter length
            
        Returns:
            Point coordinates at the target distance
        """
        # Handle wrap-around for closed contours
        if target_distance >= total_distance:
            # Handle the segment from last point back to first
            excess = target_distance - distances[-1]
            last_point = points[-1]
            first_point = points[0]
            segment_length = np.linalg.norm(first_point - last_point)
            
            if segment_length == 0:
                return first_point
            
            ratio = excess / segment_length
            return last_point + ratio * (first_point - last_point)
        
        # Find the segment containing the target distance
        for i in range(len(distances) - 1):
            if distances[i] <= target_distance <= distances[i + 1]:
                # Interpolate within this segment
                segment_start = distances[i]
                segment_end = distances[i + 1]
                segment_length = segment_end - segment_start
                
                if segment_length == 0:
                    return points[i]
                
                ratio = (target_distance - segment_start) / segment_length
                return points[i] + ratio * (points[i + 1] - points[i])
        
        # Fallback to first point
        return points[0]
    
    def _draw_dashed_circle(self, image, center, radius, color, thickness, dash_length=5):
        """
        Draw a dashed circle for removed mask center points
        
        Args:
            image: Image to draw on
            center: Center point (x, y)
            radius: Circle radius
            color: Color for the circle
            thickness: Line thickness
            dash_length: Length of each dash in degrees
        """
        import math
        
        cx, cy = center
        
        # Draw dashed circle using small arcs
        for angle in range(0, 360, dash_length * 2):
            start_angle = math.radians(angle)
            end_angle = math.radians(angle + dash_length)
            
            # Calculate start and end points
            start_x = int(cx + radius * math.cos(start_angle))
            start_y = int(cy + radius * math.sin(start_angle))
            end_x = int(cx + radius * math.cos(end_angle))
            end_y = int(cy + radius * math.sin(end_angle))
            
            # Draw arc segment (approximate with line)
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
    
    def _draw_dashed_rectangle(self, image, pt1, pt2, color, thickness, dash_length=8):
        """
        Draw a dashed rectangle for removed mask bounding boxes
        
        Args:
            image: Image to draw on
            pt1: Top-left corner (x, y)
            pt2: Bottom-right corner (x, y)
            color: Color for the rectangle
            thickness: Line thickness
            dash_length: Length of each dash in pixels
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw dashed lines for each side of the rectangle
        # Top side
        self._draw_dashed_line(image, (x1, y1), (x2, y1), color, thickness, dash_length)
        # Right side
        self._draw_dashed_line(image, (x2, y1), (x2, y2), color, thickness, dash_length)
        # Bottom side
        self._draw_dashed_line(image, (x2, y2), (x1, y2), color, thickness, dash_length)
        # Left side
        self._draw_dashed_line(image, (x1, y2), (x1, y1), color, thickness, dash_length)
    
    def _draw_dashed_line(self, image, pt1, pt2, color, thickness, dash_length=8):
        """
        Draw a dashed line between two points
        
        Args:
            image: Image to draw on
            pt1: Start point (x, y)
            pt2: End point (x, y)
            color: Color for the line
            thickness: Line thickness
            dash_length: Length of each dash in pixels
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # Unit direction vector
        ux = dx / length
        uy = dy / length
        
        # Draw dashed pattern
        current_distance = 0
        draw_dash = True
        
        while current_distance < length:
            next_distance = min(current_distance + dash_length, length)
            
            # Calculate start and end points for this segment
            start_x = int(x1 + current_distance * ux)
            start_y = int(y1 + current_distance * uy)
            end_x = int(x1 + next_distance * ux)
            end_y = int(y1 + next_distance * uy)
            
            if draw_dash:
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
            
            draw_dash = not draw_dash
            current_distance = next_distance
    
    def apply_intensity_filter(self, min_intensity: float = 0, max_intensity: float = 255) -> Dict:
        """
        Filter masks based on their mean intensity values
        
        Args:
            min_intensity: Minimum intensity threshold (inclusive)
            max_intensity: Maximum intensity threshold (inclusive)
            
        Returns:
            Dictionary with filter results
        """
        if not self.mask_statistics:
            return {'success': False, 'error': 'No masks available for filtering'}
        
        filtered_count = 0
        kept_count = 0
        
        # Apply intensity filter only to masks that have passed other filters
        for i, stats in enumerate(self.mask_statistics):
            # Determine current state; only operate on 'active' or previously 'intensity_filtered' masks
            current_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            if current_state not in ['active', 'intensity_filtered']:
                # Skip masks filtered by other criteria (e.g., overlap)
                continue
            
            mean_intensity = stats.get('mean_intensity', 0)
            
            # Check if intensity is within the specified range
            if min_intensity <= mean_intensity <= max_intensity:
                # Keep/restore mask if it was filtered by intensity previously
                if i < len(self.mask_states):
                    if self.mask_states[i] == 'intensity_filtered':
                        self.mask_states[i] = 'active'
                kept_count += 1
            else:
                # Filter out the mask by intensity
                if i < len(self.mask_states):
                    self.mask_states[i] = 'intensity_filtered'
                else:
                    # Extend states if needed
                    while len(self.mask_states) <= i:
                        self.mask_states.append('active')
                    self.mask_states[i] = 'intensity_filtered'
                filtered_count += 1
        
        return {
            'success': True,
            'filtered_count': filtered_count,
            'kept_count': kept_count,
            'total_count': len(self.mask_statistics),
            'min_intensity': min_intensity,
            'max_intensity': max_intensity
        }
    
    def reset_intensity_filter(self):
        """Reset intensity filter - restore all intensity-filtered masks to active"""
        if not self.mask_states:
            return
        
        for i in range(len(self.mask_states)):
            if self.mask_states[i] == 'intensity_filtered':
                self.mask_states[i] = 'active'
    
    def get_intensity_statistics(self) -> Dict:
        """
        Get intensity statistics for all masks
        
        Returns:
            Dictionary with intensity statistics
        """
        if not self.mask_statistics:
            return {}
        
        intensities = [stats.get('mean_intensity', 0) for stats in self.mask_statistics]
        
        if not intensities:
            return {}
        
        return {
            'min_intensity': float(np.min(intensities)),
            'max_intensity': float(np.max(intensities)),
            'mean_intensity': float(np.mean(intensities)),
            'std_intensity': float(np.std(intensities)),
            'total_masks': len(intensities)
        }
    
    def set_pixel_to_unit_conversion(self, pixel_distance: float, unit_distance: float, unit_name: str = "μm") -> bool:
        """
        Set pixel-to-unit conversion ratio
        
        Args:
            pixel_distance: Distance in pixels (e.g., 100 pixels)
            unit_distance: Corresponding distance in real units (e.g., 50 μm)
            unit_name: Name of the unit (default: μm)
            
        Returns:
            True if conversion was set successfully, False otherwise
        """
        if pixel_distance <= 0 or unit_distance <= 0:
            return False
        
        self.pixel_distance = float(pixel_distance)
        self.unit_distance = float(unit_distance)
        self.unit_name = str(unit_name)
        self.pixels_per_unit = pixel_distance / unit_distance
        self.conversion_enabled = True
        
        print(f"✅ Conversion set: {pixel_distance} pixels = {unit_distance} {unit_name}")
        print(f"   Ratio: {self.pixels_per_unit:.4f} pixels per {unit_name}")
        
        return True
    
    def reset_conversion(self):
        """Reset pixel-to-unit conversion settings"""
        self.conversion_enabled = False
        self.pixels_per_unit = 1.0
        self.unit_name = "μm"
        self.pixel_distance = 0.0
        self.unit_distance = 0.0
        print("🔄 Pixel-to-unit conversion reset")
    
    def get_conversion_info(self) -> Dict:
        """
        Get current conversion settings
        
        Returns:
            Dictionary with conversion information
        """
        return {
            'enabled': self.conversion_enabled,
            'pixels_per_unit': self.pixels_per_unit,
            'unit_name': self.unit_name,
            'pixel_distance': self.pixel_distance,
            'unit_distance': self.unit_distance,
            'conversion_ratio': f"{self.pixel_distance} pixels = {self.unit_distance} {self.unit_name}" if self.conversion_enabled else None
        }
    
    def convert_pixels_to_units(self, pixel_value: float) -> float:
        """
        Convert pixel measurement to real units
        
        Args:
            pixel_value: Value in pixels
            
        Returns:
            Value in real units
        """
        if not self.conversion_enabled or self.pixels_per_unit <= 0:
            return pixel_value
        
        return pixel_value / self.pixels_per_unit
    
    def convert_area_to_units(self, pixel_area: float) -> float:
        """
        Convert pixel area to real units (area conversion is squared)
        
        Args:
            pixel_area: Area in pixels²
            
        Returns:
            Area in real units²
        """
        if not self.conversion_enabled or self.pixels_per_unit <= 0:
            return pixel_area
        
        return pixel_area / (self.pixels_per_unit ** 2)
    
    def get_mask_statistics_with_units(self, mask_id: int) -> Optional[Dict]:
        """
        Get mask statistics with unit conversion applied
        
        Args:
            mask_id: Index of the mask
            
        Returns:
            Mask statistics with converted units, or None if mask doesn't exist
        """
        if mask_id < 0 or mask_id >= len(self.mask_statistics):
            return None
        
        stats = self.mask_statistics[mask_id].copy()
        
        if self.conversion_enabled:
            # Convert linear measurements
            stats['diameter_units'] = self.convert_pixels_to_units(stats['diameter'])
            stats['radius_units'] = self.convert_pixels_to_units(stats['radius'])
            stats['perimeter_units'] = self.convert_pixels_to_units(stats['perimeter'])
            stats['equivalent_diameter_units'] = self.convert_pixels_to_units(stats['equivalent_diameter'])
            
            # Convert area measurements
            stats['area_units'] = self.convert_area_to_units(stats['area'])
            
            # Add unit information
            stats['unit_name'] = self.unit_name
            stats['conversion_enabled'] = True
        else:
            stats['conversion_enabled'] = False
        
        return stats
    
    def get_all_mask_statistics_with_units(self) -> List[Dict]:
        """
        Get all mask statistics with unit conversion applied
        
        Returns:
            List of mask statistics with converted units
        """
        results = []
        for i in range(len(self.mask_statistics)):
            stats = self.get_mask_statistics_with_units(i)
            if stats:
                results.append(stats)
        
        return results
    
    def get_diameter_data_by_group_with_units(self) -> Dict:
        """
        Get diameter data grouped by intensity with unit conversion
        
        Returns:
            Dictionary with high/low intensity diameter lists in real units
        """
        if not self.mask_statistics:
            return {'high_intensity': [], 'low_intensity': [], 'unit_name': self.unit_name}
        
        # Get intensity statistics ONLY from active masks for threshold calculation
        active_intensities = []
        for i, stats in enumerate(self.mask_statistics):
            mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            if mask_state == 'active':
                active_intensities.append(stats.get('mean_intensity', 0))
        
        if not active_intensities:
            return {'high_intensity': [], 'low_intensity': [], 'unit_name': self.unit_name}
        
        intensity_threshold = np.median(active_intensities)
        
        high_intensity_diameters = []
        low_intensity_diameters = []
        
        for i, stats in enumerate(self.mask_statistics):
            # Only include active masks (not filtered out)
            mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            if mask_state != 'active':
                continue
            
            diameter = stats.get('diameter', 0)
            mean_intensity = stats.get('mean_intensity', 0)
            
            # Convert to units if conversion is enabled
            if self.conversion_enabled:
                diameter = self.convert_pixels_to_units(diameter)
            
            if mean_intensity >= intensity_threshold:
                high_intensity_diameters.append(diameter)
            else:
                low_intensity_diameters.append(diameter)
        
        return {
            'high_intensity': high_intensity_diameters,
            'low_intensity': low_intensity_diameters,
            'unit_name': self.unit_name if self.conversion_enabled else 'pixels'
        }
    
    def get_diameter_data_by_group(self) -> Dict:
        """
        Get diameter data grouped by intensity (original pixel values)
        
        Returns:
            Dictionary with high/low intensity diameter lists in pixels
        """
        if not self.mask_statistics:
            return {'high_intensity': [], 'low_intensity': []}
        
        # Get intensity statistics ONLY from active masks for threshold calculation
        active_intensities = []
        for i, stats in enumerate(self.mask_statistics):
            mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            if mask_state == 'active':
                active_intensities.append(stats.get('mean_intensity', 0))
        
        if not active_intensities:
            return {'high_intensity': [], 'low_intensity': []}
        
        intensity_threshold = np.median(active_intensities)
        
        high_intensity_diameters = []
        low_intensity_diameters = []
        
        for i, stats in enumerate(self.mask_statistics):
            # Only include active masks (not filtered out)
            mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            if mask_state != 'active':
                continue
            
            diameter = stats.get('diameter', 0)
            mean_intensity = stats.get('mean_intensity', 0)
            
            if mean_intensity >= intensity_threshold:
                high_intensity_diameters.append(diameter)
            else:
                low_intensity_diameters.append(diameter)
        
        return {
            'high_intensity': high_intensity_diameters,
            'low_intensity': low_intensity_diameters
        }
    
    def apply_overlap_filter(self, overlap_threshold: float = 0.9) -> Dict:
        """
        Filter out overlapping masks by comparing bounding box overlap and removing the larger mask
        
        Args:
            overlap_threshold: Minimum bounding box overlap ratio to consider masks as duplicates (0.0-1.0)
                              Default 0.4 means 90% overlap of the smaller bounding box
            
        Returns:
            Dictionary with filter results including overlap pairs with bounding box information
        """
        if not self.masks or not self.mask_statistics:
            return {'success': False, 'error': 'No masks available for overlap filtering'}
        
        removed_count = 0
        kept_count = 0
        overlap_pairs = []
        
        # Create a list to track which masks to remove
        masks_to_remove = set()
        
        # Compare each mask's bounding box with every other mask's bounding box
        for i in range(len(self.mask_statistics)):
            if i in masks_to_remove:
                continue
                
            bbox_i = self.mask_statistics[i]['bounding_box']
            area_i = self.mask_statistics[i]['area']
            
            for j in range(i + 1, len(self.mask_statistics)):
                if j in masks_to_remove:
                    continue
                    
                bbox_j = self.mask_statistics[j]['bounding_box']
                area_j = self.mask_statistics[j]['area']
                
                # Calculate overlap between bounding boxes
                overlap_ratio = self._calculate_bounding_box_overlap(bbox_i, bbox_j)
                
                if overlap_ratio >= overlap_threshold:
                    # Remove the larger mask (keep the smaller one)
                    if area_i > area_j:
                        masks_to_remove.add(i)
                        overlap_pairs.append({
                            'kept_mask': j,
                            'removed_mask': i,
                            'overlap_ratio': overlap_ratio,
                            'kept_area': area_j,
                            'removed_area': area_i,
                            'kept_bbox': bbox_j,
                            'removed_bbox': bbox_i
                        })
                        break  # Don't check further for mask i since it's being removed
                    else:
                        masks_to_remove.add(j)
                        overlap_pairs.append({
                            'kept_mask': i,
                            'removed_mask': j,
                            'overlap_ratio': overlap_ratio,
                            'kept_area': area_i,
                            'removed_area': area_j,
                            'kept_bbox': bbox_i,
                            'removed_bbox': bbox_j
                        })
        
        # Apply the removal by setting mask states
        for i in range(len(self.mask_statistics)):
            if i in masks_to_remove:
                # Set mask state to 'overlap_filtered'
                if i < len(self.mask_states):
                    self.mask_states[i] = 'overlap_filtered'
                else:
                    # Extend states if needed
                    while len(self.mask_states) <= i:
                        self.mask_states.append('active')
                    self.mask_states[i] = 'overlap_filtered'
                removed_count += 1
            else:
                # Keep the mask active (unless it was already filtered by other means)
                if i < len(self.mask_states):
                    if self.mask_states[i] == 'overlap_filtered':
                        self.mask_states[i] = 'active'
                kept_count += 1
        
        return {
            'success': True,
            'removed_count': removed_count,
            'kept_count': kept_count,
            'total_count': len(self.mask_statistics),
            'overlap_threshold': overlap_threshold,
            'overlap_pairs': overlap_pairs
        }
    
    def _calculate_bounding_box_overlap(self, bbox1: tuple, bbox2: tuple) -> float:
        """
        Calculate the overlap ratio between two bounding boxes
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            Overlap ratio (0.0 to 1.0) based on the smaller bounding box
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate the coordinates of the intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Check if there is an intersection
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas of both bounding boxes
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        # Calculate overlap ratio based on the smaller bounding box
        smaller_area = min(area1, area2)
        overlap_ratio = intersection_area / smaller_area
        
        return overlap_ratio
    
    def reset_overlap_filter(self):
        """Reset overlap filter - restore all overlap-filtered masks to active"""
        for i in range(len(self.mask_states)):
            if self.mask_states[i] == 'overlap_filtered':
                self.mask_states[i] = 'active'