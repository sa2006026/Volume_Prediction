#!/usr/bin/env python3
"""
Droplet Ring Predictor - Predict and draw ring circles based on measured ring width

This script uses the measured ring width from a reference droplet to predict
where rings should appear around other droplets in the image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, List, Dict, Any

class DropletRingPredictor:
    """
    Predict ring positions based on measured ring width
    """
    
    def __init__(self, image_path: str, reference_ring_width: float = 47.0):
        """
        Initialize with image path and reference ring width
        
        Args:
            image_path: Path to the image
            reference_ring_width: Measured ring width from reference droplet (pixels)
        """
        self.image_path = image_path
        self.reference_ring_width = reference_ring_width
        
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.bgr_image = self.original_image.copy()
        self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray_image.shape
        
        print(f"Image loaded: {self.width}x{self.height}")
        print(f"Reference ring width: {self.reference_ring_width} pixels")
    
    def detect_droplets(self, method: str = "bright_regions") -> List[Dict]:
        """
        Detect potential droplets in the image
        
        Args:
            method: Detection method ("bright_regions", "dark_regions", "circles", "contours")
            
        Returns:
            List of detected droplets with center and radius
        """
        droplets = []
        
        if method == "bright_regions":
            # Detect bright circular regions (potential droplet centers)
            # Use high threshold to find bright spots
            threshold = np.percentile(self.gray_image[self.gray_image > 0], 90)
            binary_mask = (self.gray_image >= threshold).astype(np.uint8) * 255
            
        elif method == "dark_regions":
            # Detect dark circular regions (potential droplet centers)
            threshold = np.percentile(self.gray_image, 20)
            binary_mask = (self.gray_image <= threshold).astype(np.uint8) * 255
            
        elif method == "adaptive":
            # Use adaptive thresholding
            binary_mask = cv2.adaptiveThreshold(
                self.gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 2
            )
            
        elif method == "edge_circles":
            # Detect circular patterns using edge detection
            edges = cv2.Canny(self.gray_image, 50, 150)
            
            # Use Hough Circle Transform on edges
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=int(self.reference_ring_width * 2),  # Minimum distance between droplets
                param1=50,
                param2=30,
                minRadius=int(self.reference_ring_width * 0.5),  # Minimum droplet radius
                maxRadius=int(self.reference_ring_width * 3)     # Maximum droplet radius
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (cx, cy, radius) in circles:
                    droplets.append({
                        'center': (cx, cy),
                        'radius': float(radius),
                        'method': 'hough_edges'
                    })
            
            print(f"Edge-based circle detection found {len(droplets)} droplets")
            return droplets
            
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and fit circles
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (reasonable droplet size)
            min_area = np.pi * (self.reference_ring_width * 0.5) ** 2  # Min droplet area
            max_area = np.pi * (self.reference_ring_width * 3) ** 2    # Max droplet area
            
            if min_area <= area <= max_area:
                # Fit circle to contour
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                
                # Additional validation
                if self.reference_ring_width * 0.5 <= radius <= self.reference_ring_width * 3:
                    droplets.append({
                        'center': (int(cx), int(cy)),
                        'radius': float(radius),
                        'area': float(area),
                        'method': method
                    })
        
        print(f"Droplet detection ({method}) found {len(droplets)} potential droplets")
        return droplets
    
    def predict_ring_circles(self, droplets: List[Dict], 
                           ring_width_variation: float = 0.1) -> List[Dict]:
        """
        Predict ring circles around detected droplets
        
        Args:
            droplets: List of detected droplets
            ring_width_variation: Allowed variation in ring width (0.1 = ±10%)
            
        Returns:
            List of predicted ring circles
        """
        predicted_rings = []
        
        # Calculate ring parameters based on reference
        base_ring_width = self.reference_ring_width
        width_min = base_ring_width * (1 - ring_width_variation)
        width_max = base_ring_width * (1 + ring_width_variation)
        
        print(f"\nPredicting rings with width: {width_min:.1f} - {width_max:.1f} pixels")
        
        for i, droplet in enumerate(droplets):
            center = droplet['center']
            droplet_radius = droplet['radius']
            
            # Predict ring parameters
            # Ring typically appears at some distance from droplet edge
            ring_distance_options = [
                droplet_radius * 1.2,  # Close to droplet
                droplet_radius * 1.5,  # Medium distance
                droplet_radius * 2.0,  # Far from droplet
            ]
            
            for j, ring_distance in enumerate(ring_distance_options):
                # Inner and outer ring radii
                inner_radius = ring_distance
                outer_radius = ring_distance + base_ring_width
                ring_center_radius = (inner_radius + outer_radius) / 2
                
                # Check if ring fits in image
                if (center[0] - outer_radius >= 0 and 
                    center[0] + outer_radius < self.width and
                    center[1] - outer_radius >= 0 and 
                    center[1] + outer_radius < self.height):
                    
                    predicted_rings.append({
                        'droplet_id': i,
                        'ring_id': j,
                        'center': center,
                        'inner_radius': inner_radius,
                        'outer_radius': outer_radius,
                        'ring_center_radius': ring_center_radius,
                        'ring_width': base_ring_width,
                        'droplet_radius': droplet_radius,
                        'confidence': 1.0 - (j * 0.2)  # Closer rings have higher confidence
                    })
        
        print(f"Predicted {len(predicted_rings)} ring circles around {len(droplets)} droplets")
        return predicted_rings
    
    def validate_predicted_rings(self, predicted_rings: List[Dict]) -> List[Dict]:
        """
        Validate predicted rings against actual image content
        """
        validated_rings = []
        
        # Create a mask for actual bright pixels (potential rings)
        threshold = np.percentile(self.gray_image[self.gray_image > 0], 85)
        bright_mask = (self.gray_image >= threshold).astype(np.uint8) * 255
        
        for ring in predicted_rings:
            center = ring['center']
            inner_r = int(ring['inner_radius'])
            outer_r = int(ring['outer_radius'])
            
            # Create ring mask
            ring_mask = np.zeros_like(self.gray_image)
            cv2.circle(ring_mask, center, outer_r, 255, -1)
            cv2.circle(ring_mask, center, inner_r, 0, -1)
            
            # Check overlap with bright pixels
            overlap = cv2.bitwise_and(bright_mask, ring_mask)
            overlap_pixels = np.sum(overlap > 0)
            ring_area = np.sum(ring_mask > 0)
            
            if ring_area > 0:
                overlap_ratio = overlap_pixels / ring_area
                
                # Keep rings with sufficient overlap
                if overlap_ratio > 0.1:  # At least 10% overlap
                    ring['validation_score'] = overlap_ratio
                    ring['overlap_pixels'] = overlap_pixels
                    validated_rings.append(ring)
        
        print(f"Validated {len(validated_rings)} out of {len(predicted_rings)} predicted rings")
        return validated_rings
    
    def draw_predicted_rings(self, droplets: List[Dict], 
                           predicted_rings: List[Dict],
                           draw_droplets: bool = True,
                           draw_predictions: bool = True) -> np.ndarray:
        """
        Draw droplets and predicted rings on the image
        """
        result_image = self.bgr_image.copy()
        
        # Colors
        droplet_color = (0, 255, 0)      # Green for droplets
        prediction_color = (255, 0, 0)   # Red for predicted rings
        validated_color = (0, 0, 255)    # Blue for validated rings
        
        # Draw droplets
        if draw_droplets:
            for i, droplet in enumerate(droplets):
                center = droplet['center']
                radius = int(droplet['radius'])
                
                # Draw droplet circle
                cv2.circle(result_image, center, radius, droplet_color, 2)
                cv2.circle(result_image, center, 3, droplet_color, -1)
                
                # Label droplet
                cv2.putText(result_image, f"D{i+1}", 
                           (center[0] + radius + 5, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, droplet_color, 1)
        
        # Draw predicted rings
        if draw_predictions:
            for ring in predicted_rings:
                center = ring['center']
                inner_r = int(ring['inner_radius'])
                outer_r = int(ring['outer_radius'])
                
                # Choose color based on validation
                if 'validation_score' in ring:
                    color = validated_color
                    thickness = 3
                else:
                    color = prediction_color
                    thickness = 2
                
                # Draw ring boundaries
                cv2.circle(result_image, center, inner_r, color, thickness)
                cv2.circle(result_image, center, outer_r, color, thickness)
                
                # Draw ring center line
                center_r = int(ring['ring_center_radius'])
                cv2.circle(result_image, center, center_r, color, 1)
                
                # Add ring label
                label_text = f"R{ring['droplet_id']+1}.{ring['ring_id']+1}"
                if 'validation_score' in ring:
                    label_text += f" ({ring['validation_score']:.2f})"
                
                cv2.putText(result_image, label_text,
                           (center[0] + outer_r + 5, center[1] + ring['ring_id'] * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result_image
    
    def analyze_and_predict_rings(self, 
                                detection_method: str = "bright_regions",
                                validate_predictions: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: detect droplets and predict ring circles
        """
        print(f"\n{'='*70}")
        print(f"DROPLET RING PREDICTION ANALYSIS")
        print(f"{'='*70}")
        
        # Detect droplets
        droplets = self.detect_droplets(detection_method)
        
        # Predict rings
        predicted_rings = self.predict_ring_circles(droplets)
        
        # Validate predictions
        if validate_predictions and predicted_rings:
            validated_rings = self.validate_predicted_rings(predicted_rings)
        else:
            validated_rings = predicted_rings
        
        # Draw results
        result_image = self.draw_predicted_rings(droplets, validated_rings)
        
        results = {
            'detection_method': detection_method,
            'reference_ring_width': self.reference_ring_width,
            'droplets': droplets,
            'predicted_rings': predicted_rings,
            'validated_rings': validated_rings,
            'result_image': result_image
        }
        
        return results
    
    def visualize_prediction_analysis(self, results: Dict[str, Any], save_path: str = None):
        """
        Create visualization of ring prediction analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        axes[0, 0].imshow(self.rgb_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Droplet detection
        droplet_image = self.bgr_image.copy()
        for droplet in results['droplets']:
            center = droplet['center']
            radius = int(droplet['radius'])
            cv2.circle(droplet_image, center, radius, (0, 255, 0), 2)
            cv2.circle(droplet_image, center, 3, (0, 255, 0), -1)
        
        axes[0, 1].imshow(cv2.cvtColor(droplet_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Detected Droplets')
        axes[0, 1].axis('off')
        
        # Predicted rings
        result_rgb = cv2.cvtColor(results['result_image'], cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(result_rgb)
        axes[1, 0].set_title('Predicted Ring Circles')
        axes[1, 0].axis('off')
        
        # Analysis summary
        axes[1, 1].axis('off')
        
        summary_text = f"""RING PREDICTION RESULTS

Reference Ring Width: {results['reference_ring_width']:.1f} pixels
Detection Method: {results['detection_method']}

Detected Droplets: {len(results['droplets'])}
Predicted Rings: {len(results['predicted_rings'])}
Validated Rings: {len(results['validated_rings'])}

DROPLET DETAILS:
"""
        
        for i, droplet in enumerate(results['droplets'][:5]):  # Show first 5
            center = droplet['center']
            radius = droplet['radius']
            summary_text += f"D{i+1}: center=({center[0]}, {center[1]}), radius={radius:.1f}\n"
        
        summary_text += f"\nRING PREDICTIONS:\n"
        for ring in results['validated_rings'][:8]:  # Show first 8
            did = ring['droplet_id'] + 1
            rid = ring['ring_id'] + 1
            score = ring.get('validation_score', 0)
            summary_text += f"R{did}.{rid}: inner={ring['inner_radius']:.0f}, outer={ring['outer_radius']:.0f}"
            if score > 0:
                summary_text += f" (score={score:.2f})"
            summary_text += f"\n"
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle(f'Droplet Ring Prediction - {os.path.basename(self.image_path)}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Ring prediction visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Droplet Ring Predictor')
    parser.add_argument('image_path', help='Path to the image')
    parser.add_argument('--ring-width', type=float, default=47.0,
                       help='Reference ring width in pixels (default: 47.0)')
    parser.add_argument('--detection', choices=['bright_regions', 'dark_regions', 'adaptive', 'edge_circles'],
                       default='bright_regions', help='Droplet detection method')
    parser.add_argument('--no-validation', action='store_true', 
                       help='Skip validation of predicted rings')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = DropletRingPredictor(args.image_path, args.ring_width)
        
        # Analyze and predict
        results = predictor.analyze_and_predict_rings(
            args.detection, 
            validate_predictions=not args.no_validation
        )
        
        # Print summary
        print(f"\n{'='*70}")
        print("PREDICTION SUMMARY")
        print(f"{'='*70}")
        print(f"Reference ring width: {args.ring_width} pixels")
        print(f"Detection method: {args.detection}")
        print(f"Droplets found: {len(results['droplets'])}")
        print(f"Rings predicted: {len(results['predicted_rings'])}")
        print(f"Rings validated: {len(results['validated_rings'])}")
        
        # Save results
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            
            # Save result image
            result_path = os.path.join(args.output, f"{base_name}_predicted_rings.png")
            cv2.imwrite(result_path, results['result_image'])
            print(f"\nSaved predicted rings to: {result_path}")
        
        # Visualize
        if args.visualize:
            vis_path = None
            if args.output:
                vis_path = os.path.join(args.output, f"{base_name}_ring_prediction_analysis.png")
            predictor.visualize_prediction_analysis(results, vis_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
