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
                crop_n_layers=3,              # Multiple crop scales
                min_mask_region_area=100,     # Filter out very small segments
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
                if not (100 <= area <= 50000):
                    continue
                
                # Edge removal filter - reject masks touching image borders
                if self._mask_touches_edge(mask):
                    continue
                
                # Circularity filter - only keep reasonably circular shapes
                circularity = self._calculate_circularity(mask)
                if circularity < 0.5:
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
    
    def segment_droplets(self, method: str = "sam") -> List[Dict]:
        """
        Segment droplets using Meta's SAM model or fallback methods
        
        Args:
            method: Segmentation method ("sam", "fallback")
            
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
        
        # Mean intensity within mask
        mean_intensity = np.mean(self.gray_image[mask > 0]) if np.any(mask > 0) else 0
        
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
    
    def create_mask_overlay(self, show_labels: bool = True, alpha: float = 0.3) -> np.ndarray:
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
        
        # Define colors for different masks
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        
        for i, (mask, stats) in enumerate(zip(self.masks, self.mask_statistics)):
            if not stats:
                continue
            
            # Get mask state
            mask_state = self.mask_states[i] if i < len(self.mask_states) else 'active'
            
            if mask_state == 'active':
                color = colors[i % len(colors)]
                mask_alpha = alpha
                contour_thickness = 2
                text_color = (255, 255, 255)
            else:  # removed
                color = (128, 128, 128)  # Gray for removed masks
                mask_alpha = alpha * 0.3  # Much more transparent
                contour_thickness = 1
                text_color = (160, 160, 160)
            
            # Create colored mask
            colored_mask = np.zeros_like(self.image)
            colored_mask[mask > 0] = color
            
            # Add to overlay
            mask_overlay = cv2.addWeighted(mask_overlay, 1, colored_mask, mask_alpha, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if mask_state == 'removed':
                # Draw dashed contour for removed masks
                cv2.drawContours(overlay, contours, -1, color, contour_thickness, cv2.LINE_AA)
            else:
                cv2.drawContours(overlay, contours, -1, color, contour_thickness)
            
            # Draw center point
            center_x = int(stats['center_x'])
            center_y = int(stats['center_y'])
            
            if mask_state == 'active':
                cv2.circle(overlay, (center_x, center_y), 3, color, -1)
            else:
                cv2.circle(overlay, (center_x, center_y), 3, color, 1)  # Hollow circle for removed
            
            # Add labels
            if show_labels:
                if mask_state == 'active':
                    label = f"M{i+1}"
                else:
                    label = f"M{i+1}✕"  # Add X mark for removed
                    
                cv2.putText(overlay, label, (center_x - 10, center_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Add diameter and state
                diameter_text = f"D:{stats['diameter']:.1f}"
                if mask_state == 'removed':
                    diameter_text += " (OFF)"
                    
                cv2.putText(overlay, diameter_text, (center_x - 25, center_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Combine overlay with original image
        result = cv2.addWeighted(overlay, 1, mask_overlay, alpha, 0)
        
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
