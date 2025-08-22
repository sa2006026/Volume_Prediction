#!/usr/bin/env python3
"""
Simple Ring Width Analyzer - Extract and measure ring width using only OpenCV and NumPy

This script specializes in detecting ring-like structures and measuring their width
without requiring additional dependencies like scikit-image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Dict, Any

class SimpleRingAnalyzer:
    """
    Ring analyzer using only OpenCV and NumPy
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
        
        print(f"Image loaded: {self.width}x{self.height}")
        print(f"Pixel value range: {np.min(self.gray_image)} - {np.max(self.gray_image)}")
        print(f"Mean intensity: {np.mean(self.gray_image):.1f}")
    
    def extract_ring_mask(self, method: str = "otsu") -> Tuple[np.ndarray, float]:
        """Extract ring mask using specified method"""
        
        if method == "otsu":
            # Use Otsu thresholding for automatic threshold selection
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            threshold_value, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"Otsu threshold: {threshold_value}")
            
        elif method == "high_percentile":
            # Use high percentile for very bright pixels (good for white rings)
            non_zero_pixels = self.gray_image[self.gray_image > 0]
            if len(non_zero_pixels) > 0:
                threshold_value = np.percentile(non_zero_pixels, 90)
            else:
                threshold_value = np.percentile(self.gray_image, 95)
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            print(f"High percentile threshold: {threshold_value}")
            
        elif method == "statistical":
            # Use statistical thresholding
            mean_val = np.mean(self.gray_image)
            std_val = np.std(self.gray_image)
            threshold_value = mean_val + (1.5 * std_val)
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            print(f"Statistical threshold: {threshold_value:.1f}")
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return binary_mask, threshold_value
    
    def clean_ring_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Clean the binary mask to isolate ring structure"""
        
        # Remove small noise with opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes with closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Keep only the largest connected component (assuming it's the ring)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        if num_labels > 1:  # Background is label 0
            # Find largest component (excluding background)
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component = np.argmax(areas) + 1
            cleaned = (labels == largest_component).astype(np.uint8) * 255
            print(f"Largest component area: {areas[largest_component-1]} pixels")
            
        return cleaned
    
    def find_ring_contours(self, binary_mask: np.ndarray) -> Tuple[List, Tuple[int, int], float]:
        """Find ring contours and estimate center"""
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [], (0, 0), 0
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find center using minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(cx), int(cy))
        
        print(f"Ring center: {center}")
        print(f"Enclosing circle radius: {radius:.1f}")
        
        return contours, center, radius
    
    def measure_ring_width_radial(self, binary_mask: np.ndarray, center: Tuple[int, int], 
                                 num_rays: int = 360) -> Dict[str, Any]:
        """
        Measure ring width using radial sampling from center
        """
        
        cx, cy = center
        max_radius = min(self.width//2, self.height//2, int(np.sqrt(self.width**2 + self.height**2)//2))
        
        ring_widths = []
        inner_radii = []
        outer_radii = []
        valid_rays = []
        
        print(f"Sampling {num_rays} rays from center {center}, max radius: {max_radius}")
        
        for i, angle in enumerate(np.linspace(0, 2*np.pi, num_rays, endpoint=False)):
            # Create ray from center outward
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Sample along this ray
            ray_hits = []
            for r in range(1, max_radius):
                x = int(cx + r * dx)
                y = int(cy + r * dy)
                
                # Check bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    if binary_mask[y, x] > 0:  # Hit the ring
                        ray_hits.append(r)
                else:
                    break
            
            # Analyze the hits to find ring boundaries
            if len(ray_hits) >= 2:
                # Find continuous segments
                segments = []
                current_segment = [ray_hits[0]]
                
                for j in range(1, len(ray_hits)):
                    if ray_hits[j] - ray_hits[j-1] <= 2:  # Continuous (allowing small gaps)
                        current_segment.append(ray_hits[j])
                    else:
                        if len(current_segment) >= 2:
                            segments.append(current_segment)
                        current_segment = [ray_hits[j]]
                
                # Add last segment
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                
                # Use the longest segment as the ring
                if segments:
                    longest_segment = max(segments, key=len)
                    inner_radius = min(longest_segment)
                    outer_radius = max(longest_segment)
                    ring_width = outer_radius - inner_radius
                    
                    if ring_width > 0:
                        ring_widths.append(ring_width)
                        inner_radii.append(inner_radius)
                        outer_radii.append(outer_radius)
                        valid_rays.append(i)
        
        print(f"Valid measurements from {len(ring_widths)} out of {num_rays} rays")
        
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
                'all_widths': ring_widths,
                'valid_rays': valid_rays
            }
        else:
            return {
                'mean_width': 0, 'std_width': 0, 'min_width': 0, 'max_width': 0,
                'median_width': 0, 'mean_inner_radius': 0, 'mean_outer_radius': 0,
                'num_measurements': 0, 'all_widths': [], 'valid_rays': []
            }
    
    def measure_ring_width_distance_transform(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        """
        Measure ring width using distance transform
        """
        
        # Calculate distance transform (distance to nearest edge)
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        
        # The maximum distance roughly corresponds to half the ring width
        max_distance = np.max(dist_transform)
        
        # Find all points that are not on the edge (distance > 0)
        interior_points = dist_transform[dist_transform > 0]
        
        if len(interior_points) > 0:
            # Estimate ring width as 2 * average distance from center line to edge
            estimated_widths = interior_points * 2
            
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
                'mean_width': 0, 'std_width': 0, 'min_width': 0, 'max_width': 0,
                'median_width': 0, 'max_distance': 0, 'num_measurements': 0
            }
    
    def analyze_ring_complete(self, extraction_method: str = "otsu") -> Dict[str, Any]:
        """Complete ring analysis pipeline"""
        
        print(f"\n{'='*60}")
        print(f"RING ANALYSIS USING {extraction_method.upper()} METHOD")
        print(f"{'='*60}")
        
        # Extract ring mask
        binary_mask, threshold_value = self.extract_ring_mask(extraction_method)
        cleaned_mask = self.clean_ring_mask(binary_mask)
        
        # Find contours and center
        contours, center, enclosing_radius = self.find_ring_contours(cleaned_mask)
        
        # Measure width using radial sampling
        print(f"\nMeasuring ring width using radial sampling...")
        radial_measurements = self.measure_ring_width_radial(cleaned_mask, center)
        
        # Measure width using distance transform
        print(f"Measuring ring width using distance transform...")
        distance_measurements = self.measure_ring_width_distance_transform(cleaned_mask)
        
        # Compile results
        results = {
            'extraction_method': extraction_method,
            'threshold_value': threshold_value,
            'image_info': {
                'path': self.image_path,
                'dimensions': (self.width, self.height)
            },
            'ring_center': center,
            'enclosing_radius': enclosing_radius,
            'radial_measurements': radial_measurements,
            'distance_transform_measurements': distance_measurements,
            'masks': {
                'raw_mask': binary_mask,
                'cleaned_mask': cleaned_mask
            }
        }
        
        return results
    
    def visualize_ring_analysis(self, results: Dict[str, Any], save_path: str = None):
        """Create comprehensive visualization of ring analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(self.rgb_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Raw extraction mask
        axes[0, 1].imshow(results['masks']['raw_mask'], cmap='gray')
        axes[0, 1].set_title(f'Raw Extraction ({results["extraction_method"]})')
        axes[0, 1].axis('off')
        
        # Cleaned mask with center and measurements
        axes[0, 2].imshow(results['masks']['cleaned_mask'], cmap='gray')
        center = results['ring_center']
        
        # Draw center point
        axes[0, 2].plot(center[0], center[1], 'ro', markersize=8, label='Center')
        
        # Draw some radial lines for visualization
        if results['radial_measurements']['valid_rays']:
            sample_rays = results['radial_measurements']['valid_rays'][::30]  # Every 30th ray
            for ray_idx in sample_rays[:12]:  # Show up to 12 rays
                angle = ray_idx * 2 * np.pi / 360
                dx = np.cos(angle)
                dy = np.sin(angle)
                
                # Draw ray line
                end_x = center[0] + results['enclosing_radius'] * dx
                end_y = center[1] + results['enclosing_radius'] * dy
                axes[0, 2].plot([center[0], end_x], [center[1], end_y], 'r-', alpha=0.3, linewidth=1)
        
        axes[0, 2].set_title('Cleaned Mask + Analysis')
        axes[0, 2].axis('off')
        
        # Ring width distribution (radial method)
        if results['radial_measurements']['all_widths']:
            axes[1, 0].hist(results['radial_measurements']['all_widths'], bins=20, alpha=0.7, color='blue')
            mean_width = results['radial_measurements']['mean_width']
            axes[1, 0].axvline(mean_width, color='red', linestyle='--', 
                              label=f'Mean: {mean_width:.1f}px')
            axes[1, 0].set_title('Ring Width Distribution (Radial)')
            axes[1, 0].set_xlabel('Width (pixels)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No radial\nmeasurements', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Ring Width Distribution (Radial)')
        
        # Distance transform visualization
        dist_transform = cv2.distanceTransform(results['masks']['cleaned_mask'], cv2.DIST_L2, 5)
        im = axes[1, 1].imshow(dist_transform, cmap='hot')
        axes[1, 1].set_title('Distance Transform')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Summary text
        axes[1, 2].axis('off')
        
        rm = results['radial_measurements']
        dm = results['distance_transform_measurements']
        
        summary_text = f"""RING WIDTH ANALYSIS SUMMARY

Extraction Method: {results['extraction_method']}
Threshold Value: {results['threshold_value']:.1f}

Ring Center: ({center[0]}, {center[1]})
Enclosing Radius: {results['enclosing_radius']:.1f} px

RADIAL SAMPLING METHOD:
• Mean width: {rm['mean_width']:.1f} ± {rm['std_width']:.1f} px
• Width range: {rm['min_width']:.1f} - {rm['max_width']:.1f} px
• Median width: {rm['median_width']:.1f} px
• Valid measurements: {rm['num_measurements']}
• Mean inner radius: {rm['mean_inner_radius']:.1f} px
• Mean outer radius: {rm['mean_outer_radius']:.1f} px

DISTANCE TRANSFORM METHOD:
• Mean width: {dm['mean_width']:.1f} px
• Max distance: {dm['max_distance']:.1f} px
• Measurements: {dm['num_measurements']}
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(f'Ring Width Analysis - {os.path.basename(self.image_path)}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Ring analysis visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description='Simple Ring Width Analyzer')
    parser.add_argument('image_path', help='Path to the ring image')
    parser.add_argument('--method', choices=['otsu', 'high_percentile', 'statistical'],
                       default='otsu', help='Extraction method')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = SimpleRingAnalyzer(args.image_path)
        
        # Perform complete analysis
        results = analyzer.analyze_ring_complete(args.method)
        
        # Print results
        print(f"\n{'='*60}")
        print("RING WIDTH MEASUREMENT RESULTS")
        print(f"{'='*60}")
        print(f"Method: {results['extraction_method']}")
        print(f"Threshold: {results['threshold_value']:.1f}")
        print(f"Ring center: {results['ring_center']}")
        print(f"Enclosing radius: {results['enclosing_radius']:.1f} pixels")
        
        print(f"\nRadial Sampling Measurements:")
        rm = results['radial_measurements']
        print(f"  Mean ring width: {rm['mean_width']:.1f} ± {rm['std_width']:.1f} pixels")
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
            
            # Save cleaned mask
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
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
