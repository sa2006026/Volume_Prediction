#!/usr/bin/env python3
"""
Circle Drawer - Draw circles based on white pixel detection

This script detects white pixels and fits circles to them, then draws the circles
on the original image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Dict, Any

class CircleDrawer:
    """
    Draw circles based on white pixel detection
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
    
    def detect_white_pixels(self, method: str = "otsu") -> Tuple[np.ndarray, float]:
        """Detect white pixels using various methods"""
        
        if method == "otsu":
            # Otsu thresholding
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            threshold_value, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == "high_threshold":
            # High threshold for very bright pixels
            threshold_value = 200  # Adjust as needed
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            
        elif method == "percentile":
            # Use high percentile
            non_zero_pixels = self.gray_image[self.gray_image > 0]
            if len(non_zero_pixels) > 0:
                threshold_value = np.percentile(non_zero_pixels, 85)
            else:
                threshold_value = np.percentile(self.gray_image, 95)
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            
        elif method == "pure_white":
            # Only pure white pixels (255)
            threshold_value = 255
            binary_mask = (self.gray_image >= threshold_value).astype(np.uint8) * 255
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Detection method: {method}")
        print(f"Threshold value: {threshold_value}")
        print(f"White pixels detected: {np.sum(binary_mask > 0):,}")
        
        return binary_mask, threshold_value
    
    def clean_white_pixels(self, binary_mask: np.ndarray) -> np.ndarray:
        """Clean the binary mask"""
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def fit_circles_to_contours(self, binary_mask: np.ndarray) -> List[Dict]:
        """Fit circles to contours found in the binary mask"""
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        
        for i, contour in enumerate(contours):
            # Filter out very small contours
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
            
            # Fit minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # Calculate circle properties
            circle_area = np.pi * radius * radius
            coverage = area / circle_area if circle_area > 0 else 0
            
            circles.append({
                'center': (int(cx), int(cy)),
                'radius': float(radius),
                'contour_area': float(area),
                'circle_area': float(circle_area),
                'coverage': float(coverage),
                'contour': contour
            })
        
        # Sort circles by area (largest first)
        circles.sort(key=lambda x: x['contour_area'], reverse=True)
        
        print(f"Found {len(circles)} circles")
        for i, circle in enumerate(circles[:5]):  # Show top 5
            print(f"  Circle {i+1}: center=({circle['center'][0]}, {circle['center'][1]}), "
                  f"radius={circle['radius']:.1f}, area={circle['contour_area']:.0f}")
        
        return circles
    
    def fit_circles_hough(self, binary_mask: np.ndarray) -> List[Dict]:
        """Use Hough Circle Transform to detect circles"""
        
        # Apply Hough Circle Transform
        circles_hough = cv2.HoughCircles(
            binary_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,              # Inverse ratio of accumulator resolution
            minDist=50,        # Minimum distance between circle centers
            param1=50,         # Upper threshold for edge detection
            param2=30,         # Accumulator threshold for center detection
            minRadius=10,      # Minimum radius
            maxRadius=500      # Maximum radius
        )
        
        circles = []
        
        if circles_hough is not None:
            circles_hough = np.round(circles_hough[0, :]).astype("int")
            
            for (cx, cy, radius) in circles_hough:
                circles.append({
                    'center': (cx, cy),
                    'radius': float(radius),
                    'method': 'hough'
                })
            
            print(f"Hough Transform found {len(circles)} circles")
            for i, circle in enumerate(circles):
                print(f"  Circle {i+1}: center=({circle['center'][0]}, {circle['center'][1]}), "
                      f"radius={circle['radius']:.1f}")
        else:
            print("No circles found with Hough Transform")
        
        return circles
    
    def draw_circles_on_image(self, circles: List[Dict], 
                             draw_type: str = "contour",
                             colors: List[Tuple] = None) -> np.ndarray:
        """Draw circles on the original image"""
        
        if colors is None:
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Orange
                (128, 0, 255),  # Purple
            ]
        
        # Create a copy to draw on
        result_image = self.bgr_image.copy()
        
        for i, circle in enumerate(circles):
            color = colors[i % len(colors)]
            center = circle['center']
            radius = int(circle['radius'])
            
            if draw_type == "contour" and 'contour' in circle:
                # Draw the original contour
                cv2.drawContours(result_image, [circle['contour']], -1, color, 2)
            
            # Draw the fitted circle
            cv2.circle(result_image, center, radius, color, 2)
            
            # Draw center point
            cv2.circle(result_image, center, 5, color, -1)
            
            # Add label
            label = f"C{i+1}: R={radius}"
            cv2.putText(result_image, label, 
                       (center[0] + radius + 10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_image
    
    def analyze_and_draw_circles(self, 
                                detection_method: str = "otsu",
                                circle_method: str = "contour") -> Dict[str, Any]:
        """Complete pipeline: detect white pixels and draw circles"""
        
        print(f"\n{'='*60}")
        print(f"CIRCLE DRAWING ANALYSIS")
        print(f"{'='*60}")
        
        # Detect white pixels
        binary_mask, threshold_value = self.detect_white_pixels(detection_method)
        cleaned_mask = self.clean_white_pixels(binary_mask)
        
        # Fit circles
        if circle_method == "contour":
            circles = self.fit_circles_to_contours(cleaned_mask)
        elif circle_method == "hough":
            circles = self.fit_circles_hough(cleaned_mask)
        elif circle_method == "both":
            contour_circles = self.fit_circles_to_contours(cleaned_mask)
            hough_circles = self.fit_circles_hough(cleaned_mask)
            circles = contour_circles + hough_circles
        else:
            raise ValueError(f"Unknown circle method: {circle_method}")
        
        # Draw circles
        result_image = self.draw_circles_on_image(circles, draw_type="contour")
        
        # Compile results
        results = {
            'detection_method': detection_method,
            'circle_method': circle_method,
            'threshold_value': threshold_value,
            'circles': circles,
            'masks': {
                'raw_mask': binary_mask,
                'cleaned_mask': cleaned_mask
            },
            'result_image': result_image
        }
        
        return results
    
    def visualize_circle_analysis(self, results: Dict[str, Any], save_path: str = None):
        """Create visualization of circle analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        axes[0, 0].imshow(self.rgb_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # White pixel mask
        axes[0, 1].imshow(results['masks']['cleaned_mask'], cmap='gray')
        axes[0, 1].set_title(f'White Pixel Detection\n({results["detection_method"]})')
        axes[0, 1].axis('off')
        
        # Result with circles
        result_rgb = cv2.cvtColor(results['result_image'], cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(result_rgb)
        axes[1, 0].set_title('Detected Circles')
        axes[1, 0].axis('off')
        
        # Circle information
        axes[1, 1].axis('off')
        
        circle_info = f"""CIRCLE DETECTION RESULTS

Detection Method: {results['detection_method']}
Circle Method: {results['circle_method']}
Threshold: {results['threshold_value']:.1f}

Found {len(results['circles'])} circles:

"""
        
        for i, circle in enumerate(results['circles'][:8]):  # Show top 8
            center = circle['center']
            radius = circle['radius']
            circle_info += f"Circle {i+1}:\n"
            circle_info += f"  Center: ({center[0]}, {center[1]})\n"
            circle_info += f"  Radius: {radius:.1f} pixels\n"
            if 'contour_area' in circle:
                circle_info += f"  Area: {circle['contour_area']:.0f} px²\n"
            circle_info += f"\n"
        
        axes[1, 1].text(0.05, 0.95, circle_info, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle(f'Circle Detection Analysis - {os.path.basename(self.image_path)}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Circle analysis visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Circle Drawer based on White Pixels')
    parser.add_argument('image_path', help='Path to the image')
    parser.add_argument('--detection', choices=['otsu', 'high_threshold', 'percentile', 'pure_white'],
                       default='otsu', help='White pixel detection method')
    parser.add_argument('--circle', choices=['contour', 'hough', 'both'],
                       default='contour', help='Circle fitting method')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize drawer
        drawer = CircleDrawer(args.image_path)
        
        # Analyze and draw circles
        results = drawer.analyze_and_draw_circles(args.detection, args.circle)
        
        # Print summary
        print(f"\n{'='*60}")
        print("CIRCLE DRAWING SUMMARY")
        print(f"{'='*60}")
        print(f"Detection method: {results['detection_method']}")
        print(f"Circle method: {results['circle_method']}")
        print(f"Number of circles found: {len(results['circles'])}")
        
        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            
            # Save result image
            result_path = os.path.join(args.output, f"{base_name}_with_circles.png")
            cv2.imwrite(result_path, results['result_image'])
            print(f"\nSaved result image to: {result_path}")
            
            # Save mask
            mask_path = os.path.join(args.output, f"{base_name}_white_mask.png")
            cv2.imwrite(mask_path, results['masks']['cleaned_mask'])
            print(f"Saved white pixel mask to: {mask_path}")
        
        # Visualize
        if args.visualize:
            vis_path = None
            if args.output:
                vis_path = os.path.join(args.output, f"{base_name}_circle_analysis.png")
            drawer.visualize_circle_analysis(results, vis_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
