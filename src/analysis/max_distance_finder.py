#!/usr/bin/env python3
"""
Maximum Distance Finder - Find the maximum distance between white pixels and draw the line

This script finds the two white pixels that are farthest apart and draws a straight line between them.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Dict, Any
from scipy.spatial.distance import pdist, squareform
import time

class MaxDistanceFinder:
    """
    Find maximum distance between white pixels and draw connecting line
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
    
    def detect_white_pixels(self, method: str = "otsu", sample_rate: float = 1.0) -> np.ndarray:
        """
        Detect white pixels using specified method
        
        Args:
            method: Detection method ("otsu", "high_threshold", "percentile")
            sample_rate: Rate of sampling pixels (1.0 = all pixels, 0.1 = 10% sample)
            
        Returns:
            Array of white pixel coordinates [(x, y), ...]
        """
        
        if method == "otsu":
            # Otsu thresholding
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            threshold_value, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == "high_threshold":
            # High threshold for very bright pixels
            threshold_value = 200
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            
        elif method == "percentile":
            # Percentile-based threshold
            non_zero_pixels = self.gray_image[self.gray_image > 0]
            if len(non_zero_pixels) > 0:
                threshold_value = np.percentile(non_zero_pixels, 85)
            else:
                threshold_value = np.percentile(self.gray_image, 95)
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            
        elif method == "pure_white":
            # Only pure white pixels
            threshold_value = 255
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Detection method: {method}")
        print(f"Threshold value: {threshold_value}")
        
        # Find white pixel coordinates
        white_pixels = np.where(binary_mask == 255)
        white_coords = np.column_stack((white_pixels[1], white_pixels[0]))  # (x, y) format
        
        print(f"Total white pixels found: {len(white_coords):,}")
        
        # Sample pixels if needed (for performance with large numbers of pixels)
        if sample_rate < 1.0 and len(white_coords) > 1000:
            n_samples = int(len(white_coords) * sample_rate)
            indices = np.random.choice(len(white_coords), n_samples, replace=False)
            white_coords = white_coords[indices]
            print(f"Sampled to {len(white_coords):,} pixels ({sample_rate*100:.1f}%)")
        
        return white_coords, binary_mask, threshold_value
    
    def find_max_distance_brute_force(self, white_coords: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
        """
        Find maximum distance using brute force approach (accurate but slow)
        
        Returns:
            Tuple of (point1, point2, max_distance)
        """
        
        print("Finding maximum distance using brute force method...")
        start_time = time.time()
        
        if len(white_coords) < 2:
            return (0, 0), (0, 0), 0
        
        max_distance = 0
        max_point1 = (0, 0)
        max_point2 = (0, 0)
        
        # Brute force: check all pairs
        n_points = len(white_coords)
        total_comparisons = n_points * (n_points - 1) // 2
        
        print(f"Checking {total_comparisons:,} point pairs...")
        
        for i in range(n_points):
            if i % 1000 == 0:
                progress = (i * (n_points - i)) / total_comparisons * 100
                print(f"Progress: {progress:.1f}%")
            
            for j in range(i + 1, n_points):
                # Calculate Euclidean distance
                dist = np.sqrt((white_coords[i][0] - white_coords[j][0])**2 + 
                              (white_coords[i][1] - white_coords[j][1])**2)
                
                if dist > max_distance:
                    max_distance = dist
                    max_point1 = tuple(white_coords[i])
                    max_point2 = tuple(white_coords[j])
        
        elapsed_time = time.time() - start_time
        print(f"Brute force completed in {elapsed_time:.2f} seconds")
        print(f"Maximum distance: {max_distance:.2f} pixels")
        print(f"Point 1: {max_point1}")
        print(f"Point 2: {max_point2}")
        
        return max_point1, max_point2, max_distance
    
    def find_max_distance_optimized(self, white_coords: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
        """
        Find maximum distance using optimized approach (faster)
        
        Returns:
            Tuple of (point1, point2, max_distance)
        """
        
        print("Finding maximum distance using optimized method...")
        start_time = time.time()
        
        if len(white_coords) < 2:
            return (0, 0), (0, 0), 0
        
        # Use convex hull to reduce search space
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(white_coords)
            hull_points = white_coords[hull.vertices]
            print(f"Reduced search space from {len(white_coords):,} to {len(hull_points)} hull points")
        except:
            # If convex hull fails, use all points
            hull_points = white_coords
            print("Using all points (convex hull failed)")
        
        # Find maximum distance among hull points
        max_distance = 0
        max_point1 = (0, 0)
        max_point2 = (0, 0)
        
        n_points = len(hull_points)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.sqrt((hull_points[i][0] - hull_points[j][0])**2 + 
                              (hull_points[i][1] - hull_points[j][1])**2)
                
                if dist > max_distance:
                    max_distance = dist
                    max_point1 = tuple(hull_points[i])
                    max_point2 = tuple(hull_points[j])
        
        elapsed_time = time.time() - start_time
        print(f"Optimized method completed in {elapsed_time:.2f} seconds")
        print(f"Maximum distance: {max_distance:.2f} pixels")
        print(f"Point 1: {max_point1}")
        print(f"Point 2: {max_point2}")
        
        return max_point1, max_point2, max_distance
    
    def find_max_distance_scipy(self, white_coords: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
        """
        Find maximum distance using scipy (fastest for small sets)
        
        Returns:
            Tuple of (point1, point2, max_distance)
        """
        
        print("Finding maximum distance using scipy method...")
        start_time = time.time()
        
        if len(white_coords) < 2:
            return (0, 0), (0, 0), 0
        
        # Limit to reasonable number of points for scipy
        if len(white_coords) > 5000:
            print(f"Too many points ({len(white_coords):,}), sampling 5000 points...")
            indices = np.random.choice(len(white_coords), 5000, replace=False)
            coords_sample = white_coords[indices]
        else:
            coords_sample = white_coords
        
        # Calculate all pairwise distances
        distances = pdist(coords_sample, metric='euclidean')
        
        # Find maximum distance
        max_idx = np.argmax(distances)
        max_distance = distances[max_idx]
        
        # Convert back to point indices
        n = len(coords_sample)
        i = 0
        while (i + 1) * (n - i // 2) // 2 <= max_idx:
            i += 1
        i -= 1
        j = max_idx - i * (n - (i + 1) // 2) + i + 1
        
        max_point1 = tuple(coords_sample[i])
        max_point2 = tuple(coords_sample[j])
        
        elapsed_time = time.time() - start_time
        print(f"Scipy method completed in {elapsed_time:.2f} seconds")
        print(f"Maximum distance: {max_distance:.2f} pixels")
        print(f"Point 1: {max_point1}")
        print(f"Point 2: {max_point2}")
        
        return max_point1, max_point2, max_distance
    
    def draw_max_distance_line(self, point1: Tuple[int, int], point2: Tuple[int, int], 
                              max_distance: float, binary_mask: np.ndarray,
                              line_style: str = "detailed") -> np.ndarray:
        """
        Draw the maximum distance line on the image
        
        Args:
            point1, point2: The two farthest points
            max_distance: The distance between them
            binary_mask: The white pixel mask
            line_style: Style of drawing ("simple", "detailed", "highlighted")
            
        Returns:
            Image with drawn line
        """
        
        # Create result image
        result_image = self.bgr_image.copy()
        
        # Line properties
        line_color = (0, 0, 255)      # Red line
        point_color = (0, 255, 0)     # Green points
        text_color = (255, 255, 255)  # White text
        
        if line_style == "simple":
            # Simple line and points
            cv2.line(result_image, point1, point2, line_color, 3)
            cv2.circle(result_image, point1, 5, point_color, -1)
            cv2.circle(result_image, point2, 5, point_color, -1)
            
        elif line_style == "detailed":
            # Detailed line with annotations
            # Draw thick line
            cv2.line(result_image, point1, point2, line_color, 5)
            
            # Draw endpoint circles
            cv2.circle(result_image, point1, 8, point_color, -1)
            cv2.circle(result_image, point2, 8, point_color, -1)
            cv2.circle(result_image, point1, 12, point_color, 2)
            cv2.circle(result_image, point2, 12, point_color, 2)
            
            # Add labels
            cv2.putText(result_image, "P1", 
                       (point1[0] + 15, point1[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            cv2.putText(result_image, "P2", 
                       (point2[0] + 15, point2[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            # Add distance label at midpoint
            mid_x = (point1[0] + point2[0]) // 2
            mid_y = (point1[1] + point2[1]) // 2
            distance_text = f"Max Distance: {max_distance:.1f} px"
            cv2.putText(result_image, distance_text,
                       (mid_x - 100, mid_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
        elif line_style == "highlighted":
            # Highlighted version with background
            # Create overlay
            overlay = result_image.copy()
            
            # Draw thick line on overlay
            cv2.line(overlay, point1, point2, line_color, 8)
            
            # Blend with original
            cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
            
            # Draw bright endpoints
            cv2.circle(result_image, point1, 10, (0, 255, 255), -1)  # Cyan
            cv2.circle(result_image, point2, 10, (0, 255, 255), -1)  # Cyan
            cv2.circle(result_image, point1, 15, (255, 255, 255), 3)  # White border
            cv2.circle(result_image, point2, 15, (255, 255, 255), 3)  # White border
        
        return result_image
    
    def create_distance_analysis_plot(self, point1: Tuple[int, int], point2: Tuple[int, int],
                                    max_distance: float, white_coords: np.ndarray,
                                    binary_mask: np.ndarray) -> np.ndarray:
        """
        Create a detailed analysis plot showing the maximum distance
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        axes[0, 0].imshow(self.rgb_image)
        axes[0, 0].plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', linewidth=3, label='Max Distance')
        axes[0, 0].plot(point1[0], point1[1], 'go', markersize=8, label='Point 1')
        axes[0, 0].plot(point2[0], point2[1], 'bo', markersize=8, label='Point 2')
        axes[0, 0].set_title('Original Image with Max Distance Line')
        axes[0, 0].legend()
        axes[0, 0].axis('off')
        
        # White pixel mask
        axes[0, 1].imshow(binary_mask, cmap='gray')
        axes[0, 1].plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', linewidth=3)
        axes[0, 1].plot(point1[0], point1[1], 'go', markersize=8)
        axes[0, 1].plot(point2[0], point2[1], 'bo', markersize=8)
        axes[0, 1].set_title('White Pixel Mask with Max Distance')
        axes[0, 1].axis('off')
        
        # Distance distribution (sample for visualization)
        if len(white_coords) > 1000:
            sample_coords = white_coords[np.random.choice(len(white_coords), 1000, replace=False)]
        else:
            sample_coords = white_coords
        
        # Calculate distances from point1 to all other points
        distances_from_p1 = np.sqrt((sample_coords[:, 0] - point1[0])**2 + 
                                   (sample_coords[:, 1] - point1[1])**2)
        
        axes[1, 0].hist(distances_from_p1, bins=50, alpha=0.7, color='blue')
        axes[1, 0].axvline(max_distance, color='red', linestyle='--', linewidth=2, label=f'Max: {max_distance:.1f}')
        axes[1, 0].set_title('Distance Distribution from Point 1')
        axes[1, 0].set_xlabel('Distance (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Summary information
        axes[1, 1].axis('off')
        summary_text = f"""MAXIMUM DISTANCE ANALYSIS

Point 1 Coordinates: ({point1[0]}, {point1[1]})
Point 2 Coordinates: ({point2[0]}, {point2[1]})

Maximum Distance: {max_distance:.2f} pixels

Distance in different units:
• Millimeters: {max_distance * 0.264:.2f} mm (assuming 96 DPI)
• Inches: {max_distance / 96:.3f} inches (assuming 96 DPI)

Image Information:
• Total white pixels: {len(white_coords):,}
• Image dimensions: {self.width} × {self.height}
• Maximum possible distance: {np.sqrt(self.width**2 + self.height**2):.1f} px

Line Properties:
• Angle: {np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0])):.1f}°
• Horizontal distance: {abs(point2[0] - point1[0])} px
• Vertical distance: {abs(point2[1] - point1[1])} px
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle(f'Maximum Distance Analysis - {os.path.basename(self.image_path)}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return plot_image
    
    def analyze_max_distance(self, detection_method: str = "otsu", 
                           distance_method: str = "optimized",
                           sample_rate: float = 1.0) -> Dict[str, Any]:
        """
        Complete analysis pipeline
        """
        
        print(f"\n{'='*70}")
        print(f"MAXIMUM DISTANCE ANALYSIS")
        print(f"{'='*70}")
        
        # Detect white pixels
        white_coords, binary_mask, threshold_value = self.detect_white_pixels(detection_method, sample_rate)
        
        if len(white_coords) < 2:
            print("Not enough white pixels found for distance calculation")
            return None
        
        # Find maximum distance
        if distance_method == "brute_force":
            point1, point2, max_distance = self.find_max_distance_brute_force(white_coords)
        elif distance_method == "optimized":
            point1, point2, max_distance = self.find_max_distance_optimized(white_coords)
        elif distance_method == "scipy":
            point1, point2, max_distance = self.find_max_distance_scipy(white_coords)
        else:
            raise ValueError(f"Unknown distance method: {distance_method}")
        
        # Draw results
        result_detailed = self.draw_max_distance_line(point1, point2, max_distance, binary_mask, "detailed")
        result_simple = self.draw_max_distance_line(point1, point2, max_distance, binary_mask, "simple")
        result_highlighted = self.draw_max_distance_line(point1, point2, max_distance, binary_mask, "highlighted")
        
        # Create analysis plot
        analysis_plot = self.create_distance_analysis_plot(point1, point2, max_distance, white_coords, binary_mask)
        
        results = {
            'detection_method': detection_method,
            'distance_method': distance_method,
            'threshold_value': threshold_value,
            'point1': point1,
            'point2': point2,
            'max_distance': max_distance,
            'total_white_pixels': len(white_coords),
            'binary_mask': binary_mask,
            'result_images': {
                'detailed': result_detailed,
                'simple': result_simple,
                'highlighted': result_highlighted
            },
            'analysis_plot': analysis_plot
        }
        
        return results

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Maximum Distance Finder for White Pixels')
    parser.add_argument('image_path', help='Path to the image')
    parser.add_argument('--detection', choices=['otsu', 'high_threshold', 'percentile', 'pure_white'],
                       default='otsu', help='White pixel detection method')
    parser.add_argument('--distance', choices=['brute_force', 'optimized', 'scipy'],
                       default='optimized', help='Distance calculation method')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                       help='Sampling rate for pixels (0.1-1.0, lower = faster but less accurate)')
    parser.add_argument('--output', help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        # Initialize finder
        finder = MaxDistanceFinder(args.image_path)
        
        # Analyze maximum distance
        results = finder.analyze_max_distance(args.detection, args.distance, args.sample_rate)
        
        if results is None:
            print("Analysis failed - not enough white pixels found")
            return 1
        
        # Print summary
        print(f"\n{'='*70}")
        print("MAXIMUM DISTANCE RESULTS")
        print(f"{'='*70}")
        print(f"Detection method: {results['detection_method']}")
        print(f"Distance method: {results['distance_method']}")
        print(f"Threshold value: {results['threshold_value']}")
        print(f"Total white pixels: {results['total_white_pixels']:,}")
        print(f"Maximum distance: {results['max_distance']:.2f} pixels")
        print(f"Point 1: {results['point1']}")
        print(f"Point 2: {results['point2']}")
        
        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            
            # Save result images
            for style, image in results['result_images'].items():
                output_path = os.path.join(args.output, f"{base_name}_max_distance_{style}.png")
                cv2.imwrite(output_path, image)
                print(f"Saved {style} result to: {output_path}")
            
            # Save analysis plot
            analysis_path = os.path.join(args.output, f"{base_name}_distance_analysis.png")
            cv2.imwrite(analysis_path, cv2.cvtColor(results['analysis_plot'], cv2.COLOR_RGB2BGR))
            print(f"Saved analysis plot to: {analysis_path}")
            
            # Save binary mask
            mask_path = os.path.join(args.output, f"{base_name}_white_pixel_mask.png")
            cv2.imwrite(mask_path, results['binary_mask'])
            print(f"Saved white pixel mask to: {mask_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
