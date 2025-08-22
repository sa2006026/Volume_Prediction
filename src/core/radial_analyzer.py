#!/usr/bin/env python3
"""
Radial Intensity Profile Analysis for droplet ring detection
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class RadialAnalyzer:
    """Performs radial intensity profile analysis for droplet ring detection"""
    
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.droplets = []
        self.radial_profiles = []
    
    def load_image(self, image: np.ndarray):
        """Load image for analysis"""
        self.image = image.copy()
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.droplets = []
        self.radial_profiles = []
    
    def detect_droplets_hough(self, min_radius: int = 20, max_radius: int = 150, 
                             param1: int = 50, param2: int = 30) -> List[Dict]:
        """
        Detect droplets using Circular Hough Transform
        
        Args:
            min_radius: Minimum droplet radius
            max_radius: Maximum droplet radius
            param1: Upper threshold for edge detection
            param2: Accumulator threshold for center detection
            
        Returns:
            List of detected droplets with center and radius
        """
        if self.gray_image is None:
            return []
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_image, (9, 9), 2)
        
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_radius * 2,  # Minimum distance between circle centers
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        droplets = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                droplets.append({
                    'center_x': float(x),
                    'center_y': float(y),
                    'radius': float(r),
                    'detection_method': 'hough'
                })
        
        self.droplets = droplets
        return droplets
    
    def detect_droplets_blob(self, min_radius: int = 20, max_radius: int = 150) -> List[Dict]:
        """
        Detect droplets using blob detection
        
        Args:
            min_radius: Minimum droplet radius
            max_radius: Maximum droplet radius
            
        Returns:
            List of detected droplets with center and radius
        """
        if self.gray_image is None:
            return []
        
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by Area
        params.filterByArea = True
        params.minArea = np.pi * min_radius * min_radius
        params.maxArea = np.pi * max_radius * max_radius
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(self.gray_image)
        
        droplets = []
        for kp in keypoints:
            # Estimate radius from blob size
            radius = kp.size / 2
            if min_radius <= radius <= max_radius:
                droplets.append({
                    'center_x': float(kp.pt[0]),
                    'center_y': float(kp.pt[1]),
                    'radius': float(radius),
                    'detection_method': 'blob'
                })
        
        self.droplets = droplets
        return droplets
    
    def detect_droplets_manual(self, center_x: int, center_y: int, radius: int) -> List[Dict]:
        """
        Manually specify droplet center and radius
        
        Args:
            center_x: X coordinate of droplet center
            center_y: Y coordinate of droplet center
            radius: Droplet radius
            
        Returns:
            List with single manually specified droplet
        """
        droplets = [{
            'center_x': float(center_x),
            'center_y': float(center_y),
            'radius': float(radius),
            'detection_method': 'manual'
        }]
        
        self.droplets = droplets
        return droplets
    
    def extract_radial_profile(self, center_x: float, center_y: float, 
                              max_radius: float, step_angle: int = 15) -> Dict:
        """
        Extract radial intensity profile from droplet center
        
        Args:
            center_x: X coordinate of droplet center
            center_y: Y coordinate of droplet center
            max_radius: Maximum radius to extract
            step_angle: Angular step in degrees
            
        Returns:
            Dictionary with radial profile data
        """
        if self.gray_image is None:
            return {}
        
        height, width = self.gray_image.shape
        center_x, center_y = int(center_x), int(center_y)
        max_radius = int(max_radius)
        
        # Generate angles for radial lines
        angles = np.arange(0, 360, step_angle)
        radial_profiles = []
        
        for angle in angles:
            # Convert angle to radians
            angle_rad = np.radians(angle)
            
            # Generate points along radial line
            distances = np.arange(0, max_radius)
            x_coords = center_x + distances * np.cos(angle_rad)
            y_coords = center_y + distances * np.sin(angle_rad)
            
            # Ensure coordinates are within image bounds
            valid_mask = ((x_coords >= 0) & (x_coords < width) & 
                         (y_coords >= 0) & (y_coords < height))
            
            if np.any(valid_mask):
                x_valid = x_coords[valid_mask].astype(int)
                y_valid = y_coords[valid_mask].astype(int)
                distances_valid = distances[valid_mask]
                
                # Extract intensities along the radial line
                intensities = self.gray_image[y_valid, x_valid]
                radial_profiles.append({
                    'angle': angle,
                    'distances': distances_valid,
                    'intensities': intensities
                })
        
        # Average the radial profiles to reduce noise
        if radial_profiles:
            max_dist = min([len(profile['distances']) for profile in radial_profiles])
            averaged_distances = np.arange(max_dist)
            averaged_intensities = np.zeros(max_dist)
            
            for profile in radial_profiles:
                averaged_intensities += profile['intensities'][:max_dist]
            
            averaged_intensities /= len(radial_profiles)
            
            return {
                'center_x': center_x,
                'center_y': center_y,
                'radius': max_radius,
                'distances': averaged_distances.tolist(),
                'intensities': averaged_intensities.tolist(),
                'individual_profiles': radial_profiles
            }
        
        return {}
    
    def analyze_ring_structure(self, profile: Dict) -> Dict:
        """
        Analyze ring structure from radial intensity profile
        
        Args:
            profile: Radial profile dictionary
            
        Returns:
            Dictionary with ring analysis results
        """
        if not profile or 'intensities' not in profile:
            return {}
        
        intensities = np.array(profile['intensities'])
        distances = np.array(profile['distances'])
        
        # Smooth the profile to reduce noise
        smoothed_intensities = ndimage.gaussian_filter1d(intensities, sigma=2)
        
        # Find peaks in the intensity profile
        peaks, properties = find_peaks(
            smoothed_intensities,
            height=np.mean(smoothed_intensities),  # Minimum peak height
            distance=5,  # Minimum distance between peaks
            prominence=np.std(smoothed_intensities) * 0.5  # Minimum prominence
        )
        
        peak_intensities = smoothed_intensities[peaks]
        peak_distances = distances[peaks]
        
        # Estimate ring width
        ring_width = 0
        if len(peaks) >= 2:
            # Simple estimation: distance between first two prominent peaks
            ring_width = float(peak_distances[1] - peak_distances[0])
        
        return {
            'peak_distances': peak_distances.tolist(),
            'peak_intensities': peak_intensities.tolist(),
            'ring_width': ring_width,
            'smoothed_profile': smoothed_intensities.tolist()
        }
    
    def analyze_all_droplets(self, step_angle: int = 15) -> List[Dict]:
        """
        Perform radial analysis on all detected droplets
        
        Args:
            step_angle: Angular step for radial profile extraction
            
        Returns:
            List of analysis results for each droplet
        """
        results = []
        
        for droplet in self.droplets:
            # Extract radial profile
            profile = self.extract_radial_profile(
                droplet['center_x'],
                droplet['center_y'],
                droplet['radius'],
                step_angle
            )
            
            # Analyze ring structure
            ring_analysis = self.analyze_ring_structure(profile)
            
            # Combine results
            result = {
                **droplet,
                'radial_profile': profile,
                'ring_analysis': ring_analysis
            }
            
            # Add ring measurements to droplet info
            if ring_analysis.get('ring_width'):
                result['ring_width'] = ring_analysis['ring_width']
            if ring_analysis.get('peak_intensities'):
                result['peak_intensities'] = ring_analysis['peak_intensities']
            
            results.append(result)
            
        self.radial_profiles = results
        return results
    
    def get_analysis_summary(self) -> Dict:
        """
        Get summary of all analysis results
        
        Returns:
            Summary dictionary with statistics
        """
        if not self.radial_profiles:
            return {}
        
        ring_widths = [r.get('ring_width', 0) for r in self.radial_profiles if r.get('ring_width')]
        
        summary = {
            'total_droplets': len(self.droplets),
            'analyzed_droplets': len(self.radial_profiles),
            'average_ring_width': np.mean(ring_widths) if ring_widths else 0,
            'std_ring_width': np.std(ring_widths) if ring_widths else 0,
            'min_ring_width': np.min(ring_widths) if ring_widths else 0,
            'max_ring_width': np.max(ring_widths) if ring_widths else 0
        }
        
        return summary
    
    def create_visualization(self, show_radial_lines: bool = True, 
                           show_rings: bool = True, show_labels: bool = True) -> np.ndarray:
        """
        Create visualization with detected droplets and analysis results
        
        Args:
            show_radial_lines: Whether to show radial sampling lines
            show_rings: Whether to show detected ring boundaries
            show_labels: Whether to show droplet labels and measurements
            
        Returns:
            Annotated image with visualization overlay
        """
        if self.image is None:
            return np.array([])
        
        # Create a copy of the original image for annotation
        vis_image = self.image.copy()
        
        # Define colors
        droplet_color = (0, 255, 0)      # Green for droplet boundary
        center_color = (255, 0, 0)       # Red for center point
        radial_color = (255, 255, 0)     # Cyan for radial lines
        ring_color = (255, 0, 255)       # Magenta for ring boundaries
        text_color = (255, 255, 255)     # White for text
        
        for i, result in enumerate(self.radial_profiles):
            center_x = int(result['center_x'])
            center_y = int(result['center_y'])
            radius = int(result['radius'])
            
            # Draw droplet boundary circle
            cv2.circle(vis_image, (center_x, center_y), radius, droplet_color, 2)
            
            # Draw center point
            cv2.circle(vis_image, (center_x, center_y), 3, center_color, -1)
            
            # Draw radial sampling lines
            if show_radial_lines and 'radial_profile' in result:
                profile = result['radial_profile']
                if 'individual_profiles' in profile:
                    for line_profile in profile['individual_profiles'][:8]:  # Show first 8 lines
                        angle = line_profile['angle']
                        angle_rad = np.radians(angle)
                        end_x = int(center_x + radius * np.cos(angle_rad))
                        end_y = int(center_y + radius * np.sin(angle_rad))
                        cv2.line(vis_image, (center_x, center_y), (end_x, end_y), radial_color, 1)
            
            # Draw ring boundaries
            if show_rings and 'ring_analysis' in result:
                ring_analysis = result['ring_analysis']
                if 'peak_distances' in ring_analysis:
                    for peak_dist in ring_analysis['peak_distances']:
                        ring_radius = int(peak_dist)
                        if ring_radius > 0 and ring_radius < radius:
                            cv2.circle(vis_image, (center_x, center_y), ring_radius, ring_color, 1)
            
            # Add labels
            if show_labels:
                # Droplet number
                label_pos = (center_x - 20, center_y - radius - 10)
                cv2.putText(vis_image, f"D{i+1}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, text_color, 2, cv2.LINE_AA)
                
                # Ring width if available
                if 'ring_width' in result and result['ring_width'] > 0:
                    width_text = f"W:{result['ring_width']:.1f}px"
                    width_pos = (center_x - 30, center_y + radius + 20)
                    cv2.putText(vis_image, width_text, width_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, text_color, 1, cv2.LINE_AA)
                
                # Center coordinates
                coord_text = f"({center_x},{center_y})"
                coord_pos = (center_x - 40, center_y + radius + 35)
                cv2.putText(vis_image, coord_text, coord_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1, cv2.LINE_AA)
        
        # Add legend
        if self.radial_profiles:
            legend_y = 30
            cv2.putText(vis_image, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, text_color, 2, cv2.LINE_AA)
            
            legend_y += 25
            cv2.circle(vis_image, (20, legend_y), 5, droplet_color, 2)
            cv2.putText(vis_image, "Droplet Boundary", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, text_color, 1, cv2.LINE_AA)
            
            legend_y += 20
            cv2.circle(vis_image, (20, legend_y), 3, center_color, -1)
            cv2.putText(vis_image, "Center Point", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, text_color, 1, cv2.LINE_AA)
            
            if show_rings:
                legend_y += 20
                cv2.circle(vis_image, (20, legend_y), 5, ring_color, 1)
                cv2.putText(vis_image, "Ring Boundaries", (35, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, text_color, 1, cv2.LINE_AA)
        
        return vis_image
