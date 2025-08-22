#!/usr/bin/env python3
"""
Simple Maximum Distance Finder - Find max distance between white pixels and draw line

This script finds the two white pixels that are farthest apart and draws a straight line.
"""

import cv2
import numpy as np
import argparse
import os
from typing import Tuple
import time

class SimpleMaxDistanceFinder:
    """
    Simple version to find maximum distance between white pixels
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
    
    def detect_white_pixels(self, method: str = "otsu") -> Tuple[np.ndarray, np.ndarray, float]:
        """Detect white pixels"""
        
        if method == "otsu":
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            threshold_value, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "high_threshold":
            threshold_value = 200
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
        elif method == "percentile":
            non_zero_pixels = self.gray_image[self.gray_image > 0]
            if len(non_zero_pixels) > 0:
                threshold_value = np.percentile(non_zero_pixels, 85)
            else:
                threshold_value = np.percentile(self.gray_image, 95)
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
        else:
            threshold_value = 255
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
        
        # Find white pixel coordinates
        white_pixels = np.where(binary_mask == 255)
        white_coords = np.column_stack((white_pixels[1], white_pixels[0]))  # (x, y) format
        
        print(f"Detection method: {method}")
        print(f"Threshold value: {threshold_value}")
        print(f"White pixels found: {len(white_coords):,}")
        
        return white_coords, binary_mask, threshold_value
    
    def find_max_distance_convex_hull(self, white_coords: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
        """Find maximum distance using convex hull optimization"""
        
        print("Finding maximum distance using convex hull method...")
        start_time = time.time()
        
        if len(white_coords) < 2:
            return (0, 0), (0, 0), 0
        
        # Use convex hull to find extreme points
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(white_coords)
            hull_points = white_coords[hull.vertices]
            print(f"Reduced search from {len(white_coords):,} to {len(hull_points)} hull points")
        except:
            # Fallback: use corner points and sample
            print("ConvexHull not available, using corner sampling")
            # Find extreme points manually
            min_x_idx = np.argmin(white_coords[:, 0])
            max_x_idx = np.argmax(white_coords[:, 0])
            min_y_idx = np.argmin(white_coords[:, 1])
            max_y_idx = np.argmax(white_coords[:, 1])
            
            extreme_indices = [min_x_idx, max_x_idx, min_y_idx, max_y_idx]
            hull_points = white_coords[extreme_indices]
            
            # Add some random samples
            if len(white_coords) > 100:
                sample_size = min(50, len(white_coords) - 4)
                remaining_indices = list(set(range(len(white_coords))) - set(extreme_indices))
                sample_indices = np.random.choice(remaining_indices, sample_size, replace=False)
                sample_points = white_coords[sample_indices]
                hull_points = np.vstack([hull_points, sample_points])
        
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
        print(f"Maximum distance found in {elapsed_time:.2f} seconds")
        print(f"Maximum distance: {max_distance:.2f} pixels")
        print(f"Point 1: {max_point1}")
        print(f"Point 2: {max_point2}")
        
        return max_point1, max_point2, max_distance
    
    def draw_max_distance_line(self, point1: Tuple[int, int], point2: Tuple[int, int], 
                              max_distance: float) -> dict:
        """Draw the maximum distance line on images"""
        
        # Convert points to int
        p1 = (int(point1[0]), int(point1[1]))
        p2 = (int(point2[0]), int(point2[1]))
        
        results = {}
        
        # 1. Simple version - red line on original
        simple_image = self.bgr_image.copy()
        cv2.line(simple_image, p1, p2, (0, 0, 255), 3)  # Red line
        cv2.circle(simple_image, p1, 8, (0, 255, 0), -1)  # Green point
        cv2.circle(simple_image, p2, 8, (0, 255, 0), -1)  # Green point
        results['simple'] = simple_image
        
        # 2. Detailed version with annotations
        detailed_image = self.bgr_image.copy()
        
        # Draw thick line
        cv2.line(detailed_image, p1, p2, (0, 0, 255), 5)
        
        # Draw endpoint circles
        cv2.circle(detailed_image, p1, 12, (0, 255, 0), -1)  # Green fill
        cv2.circle(detailed_image, p2, 12, (0, 255, 0), -1)  # Green fill
        cv2.circle(detailed_image, p1, 16, (255, 255, 255), 2)  # White border
        cv2.circle(detailed_image, p2, 16, (255, 255, 255), 2)  # White border
        
        # Add coordinate labels
        cv2.putText(detailed_image, f"P1({p1[0]},{p1[1]})", 
                   (p1[0] + 20, p1[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(detailed_image, f"P2({p2[0]},{p2[1]})", 
                   (p2[0] + 20, p2[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add distance label at midpoint
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        # Add background rectangle for text
        text = f"Max Distance: {max_distance:.1f} px"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(detailed_image, 
                     (mid_x - text_size[0]//2 - 10, mid_y - text_size[1] - 10),
                     (mid_x + text_size[0]//2 + 10, mid_y + 10),
                     (0, 0, 0), -1)  # Black background
        cv2.putText(detailed_image, text,
                   (mid_x - text_size[0]//2, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        results['detailed'] = detailed_image
        
        # 3. Clean version - line on black background
        clean_image = np.zeros_like(self.bgr_image)
        cv2.line(clean_image, p1, p2, (255, 255, 255), 5)  # White line
        cv2.circle(clean_image, p1, 10, (0, 255, 0), -1)   # Green points
        cv2.circle(clean_image, p2, 10, (0, 255, 0), -1)   # Green points
        results['clean'] = clean_image
        
        # 4. Highlighted version
        highlighted_image = self.bgr_image.copy()
        
        # Create overlay for line
        overlay = highlighted_image.copy()
        cv2.line(overlay, p1, p2, (0, 0, 255), 10)
        cv2.addWeighted(overlay, 0.6, highlighted_image, 0.4, 0, highlighted_image)
        
        # Bright endpoint markers
        cv2.circle(highlighted_image, p1, 15, (0, 255, 255), -1)  # Cyan
        cv2.circle(highlighted_image, p2, 15, (0, 255, 255), -1)  # Cyan
        cv2.circle(highlighted_image, p1, 20, (255, 255, 255), 3)  # White border
        cv2.circle(highlighted_image, p2, 20, (255, 255, 255), 3)  # White border
        
        results['highlighted'] = highlighted_image
        
        return results
    
    def analyze_and_draw_max_distance(self, detection_method: str = "otsu"):
        """Complete analysis pipeline"""
        
        print(f"\n{'='*70}")
        print(f"MAXIMUM DISTANCE ANALYSIS")
        print(f"{'='*70}")
        
        # Detect white pixels
        white_coords, binary_mask, threshold_value = self.detect_white_pixels(detection_method)
        
        if len(white_coords) < 2:
            print("Error: Not enough white pixels found")
            return None
        
        # Find maximum distance
        point1, point2, max_distance = self.find_max_distance_convex_hull(white_coords)
        
        # Draw results
        result_images = self.draw_max_distance_line(point1, point2, max_distance)
        
        # Calculate additional info
        angle = np.degrees(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
        horizontal_dist = abs(point2[0] - point1[0])
        vertical_dist = abs(point2[1] - point1[1])
        
        results = {
            'detection_method': detection_method,
            'threshold_value': threshold_value,
            'point1': point1,
            'point2': point2,
            'max_distance': max_distance,
            'angle': angle,
            'horizontal_distance': horizontal_dist,
            'vertical_distance': vertical_dist,
            'total_white_pixels': len(white_coords),
            'binary_mask': binary_mask,
            'result_images': result_images
        }
        
        return results

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Simple Maximum Distance Finder')
    parser.add_argument('image_path', help='Path to the image')
    parser.add_argument('--detection', choices=['otsu', 'high_threshold', 'percentile', 'pure_white'],
                       default='otsu', help='White pixel detection method')
    parser.add_argument('--output', help='Output directory for results')
    
    args = parser.parse_args()
    
    try:
        # Initialize finder
        finder = SimpleMaxDistanceFinder(args.image_path)
        
        # Analyze maximum distance
        results = finder.analyze_and_draw_max_distance(args.detection)
        
        if results is None:
            return 1
        
        # Print summary
        print(f"\n{'='*70}")
        print("MAXIMUM DISTANCE RESULTS")
        print(f"{'='*70}")
        print(f"Detection method: {results['detection_method']}")
        print(f"Threshold value: {results['threshold_value']}")
        print(f"Total white pixels: {results['total_white_pixels']:,}")
        print(f"")
        print(f"Maximum distance: {results['max_distance']:.2f} pixels")
        print(f"Point 1: {results['point1']}")
        print(f"Point 2: {results['point2']}")
        print(f"")
        print(f"Line properties:")
        print(f"  Angle: {results['angle']:.1f} degrees")
        print(f"  Horizontal distance: {results['horizontal_distance']} pixels")
        print(f"  Vertical distance: {results['vertical_distance']} pixels")
        print(f"")
        print(f"Distance in other units (assuming 96 DPI):")
        print(f"  Millimeters: {results['max_distance'] * 0.264:.2f} mm")
        print(f"  Inches: {results['max_distance'] / 96:.3f} inches")
        print(f"  Centimeters: {results['max_distance'] * 0.0264:.2f} cm")
        
        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            
            # Save all result images
            for style, image in results['result_images'].items():
                output_path = os.path.join(args.output, f"{base_name}_max_distance_{style}.png")
                cv2.imwrite(output_path, image)
                print(f"Saved {style} result to: {output_path}")
            
            # Save binary mask
            mask_path = os.path.join(args.output, f"{base_name}_white_mask.png")
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
