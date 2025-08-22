#!/usr/bin/env python3
"""
Ring Width Analyzer - Extract and measure ring width from images

This script specializes in detecting ring-like structures and measuring their width
using various geometric and statistical approaches.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
import argparse
import os
from typing import Tuple, List, Dict, Any

class RingWidthAnalyzer:
    """
    Specialized analyzer for extracting ring structures and measuring their width
    """
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.bgr_image = self.original_image.copy()
        self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray_image.shape
        
    def extract_ring_mask(self, method: str = "otsu") -> np.ndarray:
        """Extract ring mask using specified method"""
        
        if method == "otsu":
            # Use Otsu thresholding for automatic threshold selection
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == "adaptive":
            # Use adaptive thresholding
            binary_mask = cv2.adaptiveThreshold(
                self.gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 2
            )
            
        elif method == "statistical":
            # Use statistical thresholding
            mean_val = np.mean(self.gray_image)
            std_val = np.std(self.gray_image)
            threshold = mean_val + (1.5 * std_val)
            binary_mask = (self.gray_image >= threshold).astype(np.uint8) * 255
            
        elif method == "high_contrast":
            # Specialized for high contrast images like rings
            # Use a higher threshold to capture only the brightest pixels
            threshold = np.percentile(self.gray_image[self.gray_image > 0], 95)
            binary_mask = (self.gray_image >= threshold).astype(np.uint8) * 255
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return binary_mask
    
    def clean_ring_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Clean the binary mask to isolate ring structure"""
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Keep only the largest connected component (assuming it's the ring)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        if num_labels > 1:  # Background is label 0
            # Find largest component (excluding background)
            largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            cleaned = (labels == largest_component).astype(np.uint8) * 255
            
        return cleaned
    
    def find_ring_center_and_radii(self, binary_mask: np.ndarray) -> Tuple[Tuple[int, int], float, float]:
        """
        Find ring center and inner/outer radii using distance transform and contour analysis
        
        Returns:
            Tuple of (center_point, inner_radius, outer_radius)
        """
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (0, 0), 0, 0
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Method 1: Minimum enclosing circle for outer boundary
        (cx, cy), outer_radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(cx), int(cy))
        
        # Method 2: Find inner radius using distance transform
        # Invert the mask to find the inner boundary
        inverted_mask = cv2.bitwise_not(binary_mask)
        
        # Find the largest "hole" in the ring (inner circle)
        inner_contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        inner_radius = 0
        if inner_contours:
            # Find contour closest to the center
            center_point = np.array([cx, cy])
            best_inner_radius = 0
            
            for contour in inner_contours:
                # Calculate centroid of this contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    contour_cx = int(M["m10"] / M["m00"])
                    contour_cy = int(M["m01"] / M["m00"])
                    contour_center = np.array([contour_cx, contour_cy])
                    
                    # Distance from main center
                    distance = np.linalg.norm(center_point - contour_center)
                    
                    # If this contour is close to the main center, it might be the inner circle
                    if distance < outer_radius * 0.5:  # Within reasonable distance
                        _, radius = cv2.minEnclosingCircle(contour)
                        if radius > best_inner_radius:
                            best_inner_radius = radius
            
            inner_radius = best_inner_radius
        
        return center, inner_radius, outer_radius
    
    def measure_ring_width_radial(self, binary_mask: np.ndarray, center: Tuple[int, int], 
                                 num_rays: int = 360) -> Dict[str, Any]:
        """
        Measure ring width using radial sampling from center
        
        Args:
            binary_mask: Binary mask of the ring
            center: Center point (x, y)
            num_rays: Number of radial rays to sample
            
        Returns:
            Dictionary with width measurements
        """
        
        cx, cy = center
        max_radius = min(self.width, self.height) // 2
        
        ring_widths = []
        inner_radii = []
        outer_radii = []
        
        for angle in np.linspace(0, 2*np.pi, num_rays, endpoint=False):
            # Create ray from center outward
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Sample along this ray
            distances = []
            for r in range(1, max_radius):
                x = int(cx + r * dx)
                y = int(cy + r * dy)
                
                # Check bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    if binary_mask[y, x] > 0:  # Hit the ring
                        distances.append(r)
                else:
                    break
            
            # Analyze the distances to find inner and outer boundaries
            if len(distances) >= 2:
                inner_radius = min(distances)
                outer_radius = max(distances)
                ring_width = outer_radius - inner_radius
                
                ring_widths.append(ring_width)
                inner_radii.append(inner_radius)
                outer_radii.append(outer_radius)
        
        if ring_widths:
            return {
                'mean_width': np.mean(ring_widths),
                'std_width': np.std(ring_widths),
                'min_width': np.min(ring_widths),
                'max_width': np.max(ring_widths),
                'median_width': np.median(ring_widths),
                'mean_inner_radius': np.mean(inner_radii),
                'mean_outer_radius': np.mean(outer_radii),
                'num_measurements': len(ring_widths),
                'all_widths': ring_widths
            }
        else:
            return {
                'mean_width': 0,
                'std_width': 0,
                'min_width': 0,
                'max_width': 0,
                'median_width': 0,
                'mean_inner_radius': 0,
                'mean_outer_radius': 0,
                'num_measurements': 0,
                'all_widths': []
            }
    
    def measure_ring_width_distance_transform(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Alternative method using distance transform to measure ring width
        """
        
        # Calculate distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # Find the maximum distance (should be roughly half the ring width)
        max_distance = np.max(dist_transform)
        
        # Find skeleton of the ring (medial axis)
        skeleton = morphology.skeletonize(binary_mask > 0)
        
        # Get distance values along the skeleton
        skeleton_distances = dist_transform[skeleton]
        
        if len(skeleton_distances) > 0:
            # Ring width is approximately 2 * distance from skeleton to edge
            estimated_widths = skeleton_distances * 2
            
            return {
                'mean_width': np.mean(estimated_widths),
                'std_width': np.std(estimated_widths),
                'min_width': np.min(estimated_widths),
                'max_width': np.max(estimated_widths),
                'median_width': np.median(estimated_widths),
                'max_distance': max_distance,
                'num_measurements': len(estimated_widths)
            }
        else:
            return {
                'mean_width': 0,
                'std_width': 0,
                'min_width': 0,
                'max_width': 0,
                'median_width': 0,
                'max_distance': 0,
                'num_measurements': 0
            }
    
    def analyze_ring_complete(self, extraction_method: str = "otsu") -> Dict[str, Any]:
        """
        Complete ring analysis pipeline
        
        Returns:
            Comprehensive analysis results
        """
        
        # Extract ring mask
        binary_mask = self.extract_ring_mask(extraction_method)
        cleaned_mask = self.clean_ring_mask(binary_mask)
        
        # Find center and radii
        center, inner_radius, outer_radius = self.find_ring_center_and_radii(cleaned_mask)
        
        # Measure width using radial sampling
        radial_measurements = self.measure_ring_width_radial(cleaned_mask, center)
        
        # Measure width using distance transform
        distance_measurements = self.measure_ring_width_distance_transform(cleaned_mask)
        
        # Calculate geometric width (outer - inner radius)
        geometric_width = outer_radius - inner_radius
        
        # Compile results
        results = {
            'extraction_method': extraction_method,
            'image_info': {
                'path': self.image_path,
                'dimensions': (self.width, self.height)
            },
            'ring_center': center,
            'geometric_measurements': {
                'inner_radius': inner_radius,
                'outer_radius': outer_radius,
                'geometric_width': geometric_width
            },
            'radial_measurements': radial_measurements,
            'distance_transform_measurements': distance_measurements,
            'masks': {
                'raw_mask': binary_mask,
                'cleaned_mask': cleaned_mask
            }
        }
        
        return results
    
    def visualize_ring_analysis(self, results: Dict[str, Any], save_path: str = None):
        """
        Create comprehensive visualization of ring analysis
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(self.rgb_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Raw extraction mask
        axes[0, 1].imshow(results['masks']['raw_mask'], cmap='gray')
        axes[0, 1].set_title(f'Raw Extraction ({results["extraction_method"]})')
        axes[0, 1].axis('off')
        
        # Cleaned mask with center and radii
        axes[0, 2].imshow(results['masks']['cleaned_mask'], cmap='gray')
        center = results['ring_center']
        inner_r = results['geometric_measurements']['inner_radius']
        outer_r = results['geometric_measurements']['outer_radius']
        
        # Draw circles
        circle_inner = plt.Circle(center, inner_r, fill=False, color='red', linewidth=2)
        circle_outer = plt.Circle(center, outer_r, fill=False, color='blue', linewidth=2)
        axes[0, 2].add_patch(circle_inner)
        axes[0, 2].add_patch(circle_outer)
        axes[0, 2].plot(center[0], center[1], 'ro', markersize=8)
        axes[0, 2].set_title('Cleaned Mask + Geometry')
        axes[0, 2].axis('off')
        
        # Ring width distribution (radial method)
        if results['radial_measurements']['all_widths']:
            axes[1, 0].hist(results['radial_measurements']['all_widths'], bins=30, alpha=0.7)
            axes[1, 0].axvline(results['radial_measurements']['mean_width'], 
                              color='red', linestyle='--', label=f'Mean: {results["radial_measurements"]["mean_width"]:.1f}')
            axes[1, 0].set_title('Ring Width Distribution (Radial)')
            axes[1, 0].set_xlabel('Width (pixels)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Distance transform visualization
        dist_transform = cv2.distanceTransform(results['masks']['cleaned_mask'], cv2.DIST_L2, 5)
        axes[1, 1].imshow(dist_transform, cmap='hot')
        axes[1, 1].set_title('Distance Transform')
        axes[1, 1].axis('off')
        
        # Summary text
        axes[1, 2].axis('off')
        summary_text = f"""
RING WIDTH ANALYSIS SUMMARY

Geometric Method:
• Inner radius: {inner_r:.1f} pixels
• Outer radius: {outer_r:.1f} pixels  
• Width: {results['geometric_measurements']['geometric_width']:.1f} pixels

Radial Sampling Method:
• Mean width: {results['radial_measurements']['mean_width']:.1f} ± {results['radial_measurements']['std_width']:.1f}
• Min width: {results['radial_measurements']['min_width']:.1f}
• Max width: {results['radial_measurements']['max_width']:.1f}
• Measurements: {results['radial_measurements']['num_measurements']}

Distance Transform Method:
• Mean width: {results['distance_transform_measurements']['mean_width']:.1f}
• Max distance: {results['distance_transform_measurements']['max_distance']:.1f}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Ring analysis visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description='Ring Width Analyzer')
    parser.add_argument('image_path', help='Path to the ring image')
    parser.add_argument('--method', choices=['otsu', 'adaptive', 'statistical', 'high_contrast'],
                       default='otsu', help='Extraction method')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = RingWidthAnalyzer(args.image_path)
        print(f"Loaded image: {args.image_path} ({analyzer.width}x{analyzer.height})")
        
        # Perform complete analysis
        results = analyzer.analyze_ring_complete(args.method)
        
        # Print results
        print(f"\n{'='*60}")
        print("RING WIDTH ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Method: {results['extraction_method']}")
        print(f"Ring center: {results['ring_center']}")
        print(f"\nGeometric Measurements:")
        print(f"  Inner radius: {results['geometric_measurements']['inner_radius']:.1f} pixels")
        print(f"  Outer radius: {results['geometric_measurements']['outer_radius']:.1f} pixels")
        print(f"  Ring width: {results['geometric_measurements']['geometric_width']:.1f} pixels")
        
        print(f"\nRadial Sampling Measurements:")
        rm = results['radial_measurements']
        print(f"  Mean width: {rm['mean_width']:.1f} ± {rm['std_width']:.1f} pixels")
        print(f"  Width range: {rm['min_width']:.1f} - {rm['max_width']:.1f} pixels")
        print(f"  Median width: {rm['median_width']:.1f} pixels")
        print(f"  Number of measurements: {rm['num_measurements']}")
        
        print(f"\nDistance Transform Measurements:")
        dm = results['distance_transform_measurements']
        print(f"  Mean width: {dm['mean_width']:.1f} pixels")
        print(f"  Max distance: {dm['max_distance']:.1f} pixels")
        
        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            
            # Save masks
            mask_path = os.path.join(args.output, f"{base_name}_ring_mask.png")
            cv2.imwrite(mask_path, results['masks']['cleaned_mask'])
            print(f"\nSaved ring mask to: {mask_path}")
        
        # Visualize
        if args.visualize:
            vis_path = None
            if args.output:
                vis_path = os.path.join(args.output, f"{base_name}_ring_analysis.png")
            analyzer.visualize_ring_analysis(results, vis_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
