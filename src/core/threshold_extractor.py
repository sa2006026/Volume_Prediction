#!/usr/bin/env python3
"""
Threshold-based Pixel Extraction and Removal

This module provides threshold-based pixel extraction capabilities that can remove
pixels within specified intensity ranges. It supports various threshold methods
and can be used for selective pixel removal based on intensity values.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, Optional, Dict, Any, List

class ThresholdExtractor:
    """
    Threshold-based pixel extractor for removing pixels in specified intensity ranges
    """
    
    def __init__(self, image_path: str = None):
        """
        Initialize the threshold extractor
        
        Args:
            image_path: Path to input image
        """
        self.image_path = image_path
        self.original_image = None
        self.current_image = None
        self.mask = None
        self.extraction_history = []
        
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
    
    def load_image(self, image_path: str):
        """Load image for processing"""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.current_image = self.original_image.copy()
        self.extraction_history = []
        return True
    
    def extract_by_intensity_range(self, min_intensity: int, max_intensity: int, 
                                 removal_method: str = "black", 
                                 color_space: str = "gray") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract/remove pixels within specified intensity range
        
        Args:
            min_intensity: Minimum intensity value (0-255)
            max_intensity: Maximum intensity value (0-255)
            removal_method: How to handle extracted pixels ('black', 'white', 'transparent', 'blur', 'noise')
            color_space: Color space for analysis ('gray', 'hsv', 'lab')
            
        Returns:
            Tuple of (processed_image, extraction_info)
        """
        if self.current_image is None:
            raise ValueError("No image loaded")
        
        # Validate intensity range
        min_intensity = max(0, min(255, int(min_intensity)))
        max_intensity = max(0, min(255, int(max_intensity)))
        
        if min_intensity > max_intensity:
            min_intensity, max_intensity = max_intensity, min_intensity
        
        # Convert to appropriate color space for analysis
        if color_space == "gray":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)[:,:,2]  # Use V channel
        elif color_space == "lab":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)[:,:,0]  # Use L channel
        else:
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for pixels in the intensity range
        self.mask = ((analysis_image >= min_intensity) & (analysis_image <= max_intensity)).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        
        # Count extracted pixels
        extracted_pixels = np.sum(self.mask == 255)
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        extraction_percentage = (extracted_pixels / total_pixels) * 100
        
        # Apply removal method
        result_image = self.current_image.copy()
        
        if removal_method == "black":
            result_image[self.mask == 255] = [0, 0, 0]
        elif removal_method == "white":
            result_image[self.mask == 255] = [255, 255, 255]
        elif removal_method == "transparent":
            # Convert to RGBA if needed
            if result_image.shape[2] == 3:
                alpha_channel = np.ones(result_image.shape[:2], dtype=np.uint8) * 255
                result_image = cv2.merge([result_image, alpha_channel])
            result_image[self.mask == 255, 3] = 0  # Set alpha to 0
        elif removal_method == "blur":
            # Apply Gaussian blur to selected pixels
            blurred = cv2.GaussianBlur(self.current_image, (51, 51), 0)
            result_image[self.mask == 255] = blurred[self.mask == 255]
        elif removal_method == "noise":
            # Replace with random noise
            noise = np.random.randint(0, 256, self.current_image.shape, dtype=np.uint8)
            result_image[self.mask == 255] = noise[self.mask == 255]
        elif removal_method == "mean":
            # Replace with mean color of the image
            mean_color = np.mean(self.current_image.reshape(-1, 3), axis=0).astype(np.uint8)
            result_image[self.mask == 255] = mean_color
        
        # Update current image
        self.current_image = result_image
        
        # Create extraction info
        extraction_info = {
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'extracted_pixels': int(extracted_pixels),
            'total_pixels': int(total_pixels),
            'extraction_percentage': round(extraction_percentage, 2),
            'removal_method': removal_method,
            'color_space': color_space,
            'intensity_range_size': max_intensity - min_intensity + 1
        }
        
        # Add to history
        self.extraction_history.append(extraction_info)
        
        return result_image, extraction_info
    
    def extract_by_percentile_range(self, min_percentile: float, max_percentile: float,
                                  removal_method: str = "black", 
                                  color_space: str = "gray") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract pixels within percentile range of intensities
        
        Args:
            min_percentile: Minimum percentile (0-100)
            max_percentile: Maximum percentile (0-100)
            removal_method: How to handle extracted pixels
            color_space: Color space for analysis
            
        Returns:
            Tuple of (processed_image, extraction_info)
        """
        if self.current_image is None:
            raise ValueError("No image loaded")
        
        # Convert to analysis color space
        if color_space == "gray":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)[:,:,2]
        elif color_space == "lab":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)[:,:,0]
        else:
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate percentile thresholds
        min_intensity = np.percentile(analysis_image, min_percentile)
        max_intensity = np.percentile(analysis_image, max_percentile)
        
        return self.extract_by_intensity_range(
            int(min_intensity), int(max_intensity), removal_method, color_space
        )
    
    def extract_multiple_ranges(self, intensity_ranges: List[Tuple[int, int]], 
                               removal_method: str = "black",
                               color_space: str = "gray") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract pixels from multiple intensity ranges
        
        Args:
            intensity_ranges: List of (min_intensity, max_intensity) tuples
            removal_method: How to handle extracted pixels
            color_space: Color space for analysis
            
        Returns:
            Tuple of (processed_image, extraction_info)
        """
        if self.current_image is None:
            raise ValueError("No image loaded")
        
        if not intensity_ranges:
            raise ValueError("No intensity ranges provided")
        
        # Convert to analysis color space
        if color_space == "gray":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)[:,:,2]
        elif color_space == "lab":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)[:,:,0]
        else:
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Create combined mask from all ranges
        combined_mask = np.zeros(analysis_image.shape[:2], dtype=np.uint8)
        
        total_extracted = 0
        range_info = []
        
        for min_intensity, max_intensity in intensity_ranges:
            # Validate range
            min_intensity = max(0, min(255, int(min_intensity)))
            max_intensity = max(0, min(255, int(max_intensity)))
            
            if min_intensity > max_intensity:
                min_intensity, max_intensity = max_intensity, min_intensity
            
            # Create mask for this range
            range_mask = ((analysis_image >= min_intensity) & (analysis_image <= max_intensity)).astype(np.uint8) * 255
            
            # Add to combined mask
            combined_mask = cv2.bitwise_or(combined_mask, range_mask)
            
            # Track range info
            range_pixels = np.sum(range_mask == 255)
            total_extracted += range_pixels
            range_info.append({
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'pixels': int(range_pixels)
            })
        
        # Clean up combined mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        self.mask = combined_mask
        
        # Apply removal method
        result_image = self.current_image.copy()
        
        if removal_method == "black":
            result_image[self.mask == 255] = [0, 0, 0]
        elif removal_method == "white":
            result_image[self.mask == 255] = [255, 255, 255]
        elif removal_method == "transparent":
            if result_image.shape[2] == 3:
                alpha_channel = np.ones(result_image.shape[:2], dtype=np.uint8) * 255
                result_image = cv2.merge([result_image, alpha_channel])
            result_image[self.mask == 255, 3] = 0
        elif removal_method == "blur":
            blurred = cv2.GaussianBlur(self.current_image, (51, 51), 0)
            result_image[self.mask == 255] = blurred[self.mask == 255]
        elif removal_method == "noise":
            noise = np.random.randint(0, 256, self.current_image.shape, dtype=np.uint8)
            result_image[self.mask == 255] = noise[self.mask == 255]
        
        # Update current image
        self.current_image = result_image
        
        # Calculate statistics
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        final_extracted = np.sum(self.mask == 255)
        extraction_percentage = (final_extracted / total_pixels) * 100
        
        extraction_info = {
            'ranges': range_info,
            'total_extracted_pixels': int(final_extracted),
            'total_pixels': int(total_pixels),
            'extraction_percentage': round(extraction_percentage, 2),
            'removal_method': removal_method,
            'color_space': color_space,
            'num_ranges': len(intensity_ranges)
        }
        
        self.extraction_history.append(extraction_info)
        
        return result_image, extraction_info
    
    def get_intensity_histogram(self, color_space: str = "gray") -> Tuple[np.ndarray, np.ndarray]:
        """
        Get intensity histogram of current image
        
        Args:
            color_space: Color space for analysis
            
        Returns:
            Tuple of (histogram_values, bin_edges)
        """
        if self.current_image is None:
            raise ValueError("No image loaded")
        
        if color_space == "gray":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)[:,:,2]
        elif color_space == "lab":
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)[:,:,0]
        else:
            analysis_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        hist, bins = np.histogram(analysis_image.flatten(), bins=256, range=[0, 256])
        return hist, bins
    
    def visualize_extraction(self, save_path: str = None, show_histogram: bool = True) -> str:
        """
        Create visualization of the extraction process
        
        Args:
            save_path: Path to save visualization
            show_histogram: Whether to include histogram
            
        Returns:
            Path to saved visualization
        """
        if self.current_image is None or self.mask is None:
            raise ValueError("No extraction has been performed")
        
        # Create visualization
        if show_histogram:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            hist_ax = axes[1, 2]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes = axes.reshape(1, -1)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Extraction mask
        axes[0, 1].imshow(self.mask, cmap='gray')
        axes[0, 1].set_title('Extraction Mask\n(White = Extracted)')
        axes[0, 1].axis('off')
        
        # Result image
        if self.current_image.shape[2] == 4:  # RGBA
            # Handle transparent images
            result_display = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2RGBA)
        else:
            result_display = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        axes[0, 2].imshow(result_display)
        axes[0, 2].set_title('Result Image')
        axes[0, 2].axis('off')
        
        if show_histogram:
            # Show histogram with extraction range
            hist, bins = self.get_intensity_histogram()
            hist_ax.plot(bins[:-1], hist, color='blue', alpha=0.7)
            hist_ax.set_title('Intensity Histogram')
            hist_ax.set_xlabel('Intensity Value')
            hist_ax.set_ylabel('Pixel Count')
            hist_ax.grid(True, alpha=0.3)
            
            # Highlight extracted ranges if available
            if self.extraction_history:
                last_extraction = self.extraction_history[-1]
                if 'min_intensity' in last_extraction:
                    min_int = last_extraction['min_intensity']
                    max_int = last_extraction['max_intensity']
                    hist_ax.axvspan(min_int, max_int, alpha=0.3, color='red', 
                                  label=f'Extracted Range: {min_int}-{max_int}')
                    hist_ax.legend()
                elif 'ranges' in last_extraction:
                    # Multiple ranges
                    for i, range_info in enumerate(last_extraction['ranges']):
                        min_int = range_info['min_intensity']
                        max_int = range_info['max_intensity']
                        color = plt.cm.Set1(i % 9)
                        hist_ax.axvspan(min_int, max_int, alpha=0.3, color=color,
                                      label=f'Range {i+1}: {min_int}-{max_int}')
                    hist_ax.legend()
            
            # Hide empty subplots
            for i in range(1, 2):
                for j in range(0, 2):
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        if save_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"threshold_extraction_visualization_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def reset_to_original(self):
        """Reset image to original state"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.mask = None
            self.extraction_history = []
            return True
        return False
    
    def save_result(self, output_path: str):
        """Save current processed image"""
        if self.current_image is not None:
            cv2.imwrite(output_path, self.current_image)
            return True
        return False

def main():
    """Command line interface for threshold extraction"""
    parser = argparse.ArgumentParser(description="Threshold-based pixel extraction and removal")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--min-intensity", type=int, default=0, help="Minimum intensity value (0-255)")
    parser.add_argument("--max-intensity", type=int, default=255, help="Maximum intensity value (0-255)")
    parser.add_argument("--method", choices=['black', 'white', 'transparent', 'blur', 'noise', 'mean'], 
                       default='black', help="Removal method")
    parser.add_argument("--color-space", choices=['gray', 'hsv', 'lab'], default='gray', 
                       help="Color space for analysis")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--visualize", action='store_true', help="Create visualization")
    parser.add_argument("--percentile", action='store_true', help="Use percentile range instead of absolute values")
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = ThresholdExtractor(args.image_path)
        
        # Perform extraction
        if args.percentile:
            result, info = extractor.extract_by_percentile_range(
                args.min_intensity, args.max_intensity, args.method, args.color_space
            )
            print(f"Extracted pixels in percentile range {args.min_intensity}-{args.max_intensity}%")
        else:
            result, info = extractor.extract_by_intensity_range(
                args.min_intensity, args.max_intensity, args.method, args.color_space
            )
            print(f"Extracted pixels in intensity range {args.min_intensity}-{args.max_intensity}")
        
        # Print extraction statistics
        print(f"Extracted pixels: {info['extracted_pixels']:,}")
        print(f"Total pixels: {info['total_pixels']:,}")
        print(f"Extraction percentage: {info['extraction_percentage']:.2f}%")
        print(f"Removal method: {info['removal_method']}")
        print(f"Color space: {info['color_space']}")
        
        # Save result
        if args.output:
            extractor.save_result(args.output)
            print(f"Result saved to: {args.output}")
        else:
            # Auto-generate output filename
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            output_path = f"{base_name}_threshold_{args.min_intensity}_{args.max_intensity}.png"
            extractor.save_result(output_path)
            print(f"Result saved to: {output_path}")
        
        # Create visualization
        if args.visualize:
            viz_path = extractor.visualize_extraction()
            print(f"Visualization saved to: {viz_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
