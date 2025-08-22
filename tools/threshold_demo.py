#!/usr/bin/env python3
"""
Threshold Extraction Demo

This script demonstrates the threshold extraction functionality
with various examples and use cases.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.threshold_extractor import ThresholdExtractor
import cv2
import numpy as np

def demo_basic_threshold_extraction():
    """Demonstrate basic threshold-based pixel removal"""
    print("=== Basic Threshold Extraction Demo ===")
    
    # Use the default image
    image_path = "data/input/overfocus.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please ensure you have an image in the data/input/ directory")
        return
    
    # Initialize extractor
    extractor = ThresholdExtractor(image_path)
    
    # Demo 1: Remove dark pixels (shadows)
    print("\n1. Removing dark pixels (0-50 intensity):")
    result, info = extractor.extract_by_intensity_range(0, 50, "white", "gray")
    print(f"   Removed {info['extracted_pixels']:,} pixels ({info['extraction_percentage']:.2f}%)")
    extractor.save_result("results/threshold_demo_remove_dark.png")
    print("   Result saved: results/threshold_demo_remove_dark.png")
    
    # Reset image
    extractor.reset_to_original()
    
    # Demo 2: Remove bright pixels (highlights)
    print("\n2. Removing bright pixels (200-255 intensity):")
    result, info = extractor.extract_by_intensity_range(200, 255, "black", "gray")
    print(f"   Removed {info['extracted_pixels']:,} pixels ({info['extraction_percentage']:.2f}%)")
    extractor.save_result("results/threshold_demo_remove_bright.png")
    print("   Result saved: results/threshold_demo_remove_bright.png")
    
    # Reset image
    extractor.reset_to_original()
    
    # Demo 3: Remove mid-range pixels with blur
    print("\n3. Blurring mid-range pixels (100-150 intensity):")
    result, info = extractor.extract_by_intensity_range(100, 150, "blur", "gray")
    print(f"   Blurred {info['extracted_pixels']:,} pixels ({info['extraction_percentage']:.2f}%)")
    extractor.save_result("results/threshold_demo_blur_midrange.png")
    print("   Result saved: results/threshold_demo_blur_midrange.png")

def demo_percentile_extraction():
    """Demonstrate percentile-based extraction"""
    print("\n=== Percentile-Based Extraction Demo ===")
    
    image_path = "data/input/overfocus.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    extractor = ThresholdExtractor(image_path)
    
    # Remove darkest 10% of pixels
    print("\n1. Removing darkest 10% of pixels:")
    result, info = extractor.extract_by_percentile_range(0, 10, "white", "gray")
    print(f"   Intensity range: {info['min_intensity']}-{info['max_intensity']}")
    print(f"   Removed {info['extracted_pixels']:,} pixels ({info['extraction_percentage']:.2f}%)")
    extractor.save_result("results/threshold_demo_percentile_dark.png")
    print("   Result saved: results/threshold_demo_percentile_dark.png")
    
    # Reset and remove brightest 5%
    extractor.reset_to_original()
    print("\n2. Removing brightest 5% of pixels:")
    result, info = extractor.extract_by_percentile_range(95, 100, "black", "gray")
    print(f"   Intensity range: {info['min_intensity']}-{info['max_intensity']}")
    print(f"   Removed {info['extracted_pixels']:,} pixels ({info['extraction_percentage']:.2f}%)")
    extractor.save_result("results/threshold_demo_percentile_bright.png")
    print("   Result saved: results/threshold_demo_percentile_bright.png")

def demo_multiple_ranges():
    """Demonstrate multiple range extraction"""
    print("\n=== Multiple Range Extraction Demo ===")
    
    image_path = "data/input/overfocus.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    extractor = ThresholdExtractor(image_path)
    
    # Remove multiple ranges simultaneously
    ranges = [(0, 30), (100, 120), (200, 255)]  # Dark, mid, and bright ranges
    print(f"\n1. Removing multiple ranges: {ranges}")
    result, info = extractor.extract_multiple_ranges(ranges, "transparent", "gray")
    
    print(f"   Total removed pixels: {info['total_extracted_pixels']:,}")
    print(f"   Extraction percentage: {info['extraction_percentage']:.2f}%")
    
    for i, range_info in enumerate(info['ranges']):
        print(f"   Range {i+1} ({range_info['min_intensity']}-{range_info['max_intensity']}): {range_info['pixels']:,} pixels")
    
    extractor.save_result("results/threshold_demo_multiple_ranges.png")
    print("   Result saved: results/threshold_demo_multiple_ranges.png")

def demo_color_spaces():
    """Demonstrate different color space analysis"""
    print("\n=== Color Space Analysis Demo ===")
    
    image_path = "data/input/overfocus.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    extractor = ThresholdExtractor(image_path)
    
    color_spaces = ["gray", "hsv", "lab"]
    
    for color_space in color_spaces:
        print(f"\n{color_space.upper()} color space analysis:")
        
        # Get histogram
        hist, bins = extractor.get_intensity_histogram(color_space)
        total_pixels = np.sum(hist)
        
        # Find intensity range with most pixels (mode range)
        max_hist_idx = np.argmax(hist)
        mode_range = (max(0, max_hist_idx - 10), min(255, max_hist_idx + 10))
        
        print(f"   Most common intensity range: {mode_range[0]}-{mode_range[1]}")
        
        # Remove the most common range
        result, info = extractor.extract_by_intensity_range(
            mode_range[0], mode_range[1], "noise", color_space
        )
        
        print(f"   Removed {info['extracted_pixels']:,} pixels ({info['extraction_percentage']:.2f}%)")
        extractor.save_result(f"results/threshold_demo_{color_space}_mode.png")
        print(f"   Result saved: results/threshold_demo_{color_space}_mode.png")
        
        # Reset for next color space
        extractor.reset_to_original()

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("\n=== Visualization Demo ===")
    
    image_path = "data/input/overfocus.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    extractor = ThresholdExtractor(image_path)
    
    # Perform an extraction
    result, info = extractor.extract_by_intensity_range(50, 150, "black", "gray")
    
    # Create visualization
    viz_path = extractor.visualize_extraction("results/threshold_demo_visualization.png", show_histogram=True)
    print(f"   Visualization saved: {viz_path}")
    print(f"   Shows original image, extraction mask, result, and histogram")

def main():
    """Run all demos"""
    print("🎯 Threshold Extraction Demo")
    print("=" * 50)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Run demos
    demo_basic_threshold_extraction()
    demo_percentile_extraction() 
    demo_multiple_ranges()
    demo_color_spaces()
    demo_visualization()
    
    print("\n✅ All demos completed!")
    print("\nFiles created in results/ directory:")
    print("  - threshold_demo_remove_dark.png")
    print("  - threshold_demo_remove_bright.png") 
    print("  - threshold_demo_blur_midrange.png")
    print("  - threshold_demo_percentile_dark.png")
    print("  - threshold_demo_percentile_bright.png")
    print("  - threshold_demo_multiple_ranges.png")
    print("  - threshold_demo_gray_mode.png")
    print("  - threshold_demo_hsv_mode.png") 
    print("  - threshold_demo_lab_mode.png")
    print("  - threshold_demo_visualization.png")
    
    print("\n💡 You can now use the web interface to perform threshold extraction interactively!")
    print("   Start the server: python3 src/web/pixel_removal_server.py")
    print("   Then go to: http://localhost:8000")

if __name__ == "__main__":
    main()
