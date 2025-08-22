#!/usr/bin/env python3
"""
Circle Extractor - Extract and draw detected circles as separate objects

This script extracts the detected rings/circles and draws them clearly
with various visualization options.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Dict, Any

class CircleExtractor:
    """
    Extract and visualize circles from images
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
    
    def detect_circles(self, method: str = "otsu") -> List[Dict]:
        """Detect circles in the image"""
        
        if method == "otsu":
            # Otsu thresholding
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            threshold_value, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == "high_threshold":
            # High threshold for bright pixels
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
            
        elif method == "adaptive":
            # Adaptive thresholding
            binary_mask = cv2.adaptiveThreshold(
                self.gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 2
            )
            threshold_value = "adaptive"
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Detection method: {method}")
        print(f"Threshold: {threshold_value}")
        
        # Clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < 100:
                continue
            
            # Fit circle to contour
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # Calculate circle properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            circles.append({
                'id': i,
                'center': (int(cx), int(cy)),
                'radius': float(radius),
                'area': float(area),
                'perimeter': float(perimeter),
                'circularity': float(circularity),
                'contour': contour,
                'threshold': threshold_value
            })
        
        # Sort by area (largest first)
        circles.sort(key=lambda x: x['area'], reverse=True)
        
        print(f"Found {len(circles)} circles")
        for i, circle in enumerate(circles[:10]):  # Show top 10
            print(f"  Circle {i+1}: center=({circle['center'][0]}, {circle['center'][1]}), "
                  f"radius={circle['radius']:.1f}, area={circle['area']:.0f}, "
                  f"circularity={circle['circularity']:.3f}")
        
        return circles, cleaned_mask
    
    def create_circle_canvas(self, canvas_type: str = "black") -> np.ndarray:
        """Create a canvas for drawing circles"""
        
        if canvas_type == "black":
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elif canvas_type == "white":
            canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        elif canvas_type == "gray":
            canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 128
        elif canvas_type == "original":
            canvas = self.bgr_image.copy()
        else:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        return canvas
    
    def draw_circles_simple(self, circles: List[Dict], 
                           canvas_type: str = "black",
                           draw_filled: bool = False) -> np.ndarray:
        """Draw circles in simple style"""
        
        canvas = self.create_circle_canvas(canvas_type)
        
        # Color palette
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 255, 255), # White
            (128, 128, 128), # Gray
        ]
        
        for i, circle in enumerate(circles):
            color = colors[i % len(colors)]
            center = circle['center']
            radius = int(circle['radius'])
            
            if draw_filled:
                cv2.circle(canvas, center, radius, color, -1)
            else:
                thickness = max(2, int(radius * 0.02))  # Scale thickness with radius
                cv2.circle(canvas, center, radius, color, thickness)
            
            # Draw center point
            cv2.circle(canvas, center, max(3, int(radius * 0.01)), color, -1)
        
        return canvas
    
    def draw_circles_detailed(self, circles: List[Dict], 
                             canvas_type: str = "black") -> np.ndarray:
        """Draw circles with detailed annotations"""
        
        canvas = self.create_circle_canvas(canvas_type)
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
        ]
        
        for i, circle in enumerate(circles):
            color = colors[i % len(colors)]
            center = circle['center']
            radius = int(circle['radius'])
            
            # Draw circle outline
            thickness = max(2, int(radius * 0.02))
            cv2.circle(canvas, center, radius, color, thickness)
            
            # Draw center point
            cv2.circle(canvas, center, 5, color, -1)
            
            # Draw radius line
            cv2.line(canvas, center, 
                    (center[0] + radius, center[1]), color, thickness)
            
            # Add labels
            label_text = f"C{i+1}"
            font_scale = max(0.5, radius * 0.002)
            cv2.putText(canvas, label_text, 
                       (center[0] + radius + 10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            
            # Add radius label
            radius_text = f"R={radius}"
            cv2.putText(canvas, radius_text,
                       (center[0] + radius + 10, center[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, 1)
        
        return canvas
    
    def draw_circles_contours(self, circles: List[Dict], 
                             canvas_type: str = "black") -> np.ndarray:
        """Draw original contours of circles"""
        
        canvas = self.create_circle_canvas(canvas_type)
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)
        ]
        
        for i, circle in enumerate(circles):
            color = colors[i % len(colors)]
            
            # Draw original contour
            cv2.drawContours(canvas, [circle['contour']], -1, color, 2)
            
            # Draw fitted circle
            center = circle['center']
            radius = int(circle['radius'])
            cv2.circle(canvas, center, radius, color, 1)
            
            # Draw center
            cv2.circle(canvas, center, 3, color, -1)
        
        return canvas
    
    def create_circle_grid(self, circles: List[Dict], 
                          grid_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
        """Create a grid view of individual circles"""
        
        if not circles:
            return np.zeros((grid_size[1], grid_size[0], 3), dtype=np.uint8)
        
        # Calculate grid dimensions
        n_circles = len(circles)
        cols = min(4, n_circles)
        rows = (n_circles + cols - 1) // cols
        
        grid_width = cols * grid_size[0]
        grid_height = rows * grid_size[1]
        grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, circle in enumerate(circles):
            row = i // cols
            col = i % cols
            
            # Extract circle region
            center = circle['center']
            radius = int(circle['radius'])
            
            # Calculate bounding box
            x1 = max(0, center[0] - radius - 10)
            y1 = max(0, center[1] - radius - 10)
            x2 = min(self.width, center[0] + radius + 10)
            y2 = min(self.height, center[1] + radius + 10)
            
            if x2 > x1 and y2 > y1:
                # Extract region
                circle_region = self.rgb_image[y1:y2, x1:x2]
                
                # Resize to grid size
                resized_region = cv2.resize(circle_region, grid_size)
                
                # Place in grid
                start_y = row * grid_size[1]
                end_y = start_y + grid_size[1]
                start_x = col * grid_size[0]
                end_x = start_x + grid_size[0]
                
                grid_canvas[start_y:end_y, start_x:end_x] = cv2.cvtColor(resized_region, cv2.COLOR_RGB2BGR)
                
                # Add label
                cv2.putText(grid_canvas, f"Circle {i+1}", 
                           (start_x + 10, start_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return grid_canvas
    
    def extract_and_visualize_circles(self, 
                                    detection_method: str = "otsu",
                                    visualization_style: str = "detailed") -> Dict[str, Any]:
        """Complete circle extraction and visualization"""
        
        print(f"\n{'='*60}")
        print(f"CIRCLE EXTRACTION AND VISUALIZATION")
        print(f"{'='*60}")
        
        # Detect circles
        circles, binary_mask = self.detect_circles(detection_method)
        
        # Create different visualizations
        visualizations = {}
        
        if visualization_style in ["simple", "all"]:
            visualizations['simple_outline'] = self.draw_circles_simple(circles, "black", False)
            visualizations['simple_filled'] = self.draw_circles_simple(circles, "black", True)
        
        if visualization_style in ["detailed", "all"]:
            visualizations['detailed_black'] = self.draw_circles_detailed(circles, "black")
            visualizations['detailed_white'] = self.draw_circles_detailed(circles, "white")
            visualizations['on_original'] = self.draw_circles_detailed(circles, "original")
        
        if visualization_style in ["contours", "all"]:
            visualizations['contours'] = self.draw_circles_contours(circles, "black")
        
        if visualization_style in ["grid", "all"]:
            visualizations['grid'] = self.create_circle_grid(circles)
        
        results = {
            'detection_method': detection_method,
            'circles': circles,
            'binary_mask': binary_mask,
            'visualizations': visualizations
        }
        
        return results
    
    def create_comprehensive_visualization(self, results: Dict[str, Any], save_path: str = None):
        """Create comprehensive visualization of all circle extractions"""
        
        visualizations = results['visualizations']
        n_vis = len(visualizations)
        
        if n_vis == 0:
            print("No visualizations to show")
            return
        
        # Calculate subplot arrangement
        cols = min(3, n_vis)
        rows = (n_vis + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Show each visualization
        for i, (name, image) in enumerate(visualizations.items()):
            if i < len(axes):
                # Convert BGR to RGB for matplotlib
                if len(image.shape) == 3:
                    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    display_image = image
                
                axes[i].imshow(display_image)
                axes[i].set_title(f'{name.replace("_", " ").title()}')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(visualizations), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Circle Extraction Results - {len(results["circles"])} Circles Found', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Circle Extractor and Visualizer')
    parser.add_argument('image_path', help='Path to the image')
    parser.add_argument('--detection', choices=['otsu', 'high_threshold', 'percentile', 'adaptive'],
                       default='otsu', help='Circle detection method')
    parser.add_argument('--style', choices=['simple', 'detailed', 'contours', 'grid', 'all'],
                       default='detailed', help='Visualization style')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Show comprehensive visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = CircleExtractor(args.image_path)
        
        # Extract and visualize circles
        results = extractor.extract_and_visualize_circles(args.detection, args.style)
        
        # Print summary
        print(f"\n{'='*60}")
        print("CIRCLE EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Detection method: {args.detection}")
        print(f"Visualization style: {args.style}")
        print(f"Circles found: {len(results['circles'])}")
        print(f"Visualizations created: {len(results['visualizations'])}")
        
        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            
            # Save each visualization
            for name, image in results['visualizations'].items():
                output_path = os.path.join(args.output, f"{base_name}_circles_{name}.png")
                cv2.imwrite(output_path, image)
                print(f"Saved {name} to: {output_path}")
            
            # Save binary mask
            mask_path = os.path.join(args.output, f"{base_name}_circle_mask.png")
            cv2.imwrite(mask_path, results['binary_mask'])
            print(f"Saved detection mask to: {mask_path}")
        
        # Show comprehensive visualization
        if args.visualize:
            vis_path = None
            if args.output:
                vis_path = os.path.join(args.output, f"{base_name}_circle_extraction_overview.png")
            extractor.create_comprehensive_visualization(results, vis_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
