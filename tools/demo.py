#!/usr/bin/env python3
"""
Demo script showing how to use the adaptive light extraction code
"""

from src.core.adaptive_light_extraction import AdaptiveLightExtractor
import cv2
import numpy as np

def demo_basic_usage():
    """Demonstrate basic usage of the adaptive light extractor"""
    
    # Initialize with your image
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = AdaptiveLightExtractor(image_path)
    
    print("=== BASIC USAGE DEMO ===")
    print(f"Image loaded: {extractor.width} x {extractor.height} pixels")
    
    # Get image statistics
    stats = extractor.analyze_image_statistics()
    print(f"Mean intensity: {stats['mean_intensity']:.2f}")
    print(f"Contrast ratio: {stats['contrast']:.2f}")
    
    # Auto-select best method
    best_method = extractor.auto_select_method()
    print(f"Recommended method: {best_method}")
    
    # Extract using the recommended method
    if best_method == "otsu":
        mask = extractor.adaptive_threshold_otsu()
    elif best_method == "local":
        mask = extractor.adaptive_threshold_local()
    elif best_method == "percentile":
        mask = extractor.adaptive_threshold_percentile()
    elif best_method == "kmeans":
        mask = extractor.adaptive_threshold_kmeans()
    elif best_method == "hsv":
        mask = extractor.adaptive_threshold_hsv()
    elif best_method == "lab":
        mask = extractor.adaptive_threshold_lab()
    elif best_method == "statistical":
        mask = extractor.adaptive_threshold_statistical()
    
    # Clean up the mask
    cleaned_mask = extractor.morphological_cleanup(mask)
    
    # Get connected components
    filtered_mask, component_info = extractor.extract_connected_components(cleaned_mask)
    
    print(f"Light pixels found: {np.sum(filtered_mask > 0):,}")
    print(f"Connected components: {component_info['total_components']}")
    
    return filtered_mask

def demo_custom_parameters():
    """Demonstrate usage with custom parameters"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = AdaptiveLightExtractor(image_path)
    
    print("\n=== CUSTOM PARAMETERS DEMO ===")
    
    # Try different percentile thresholds
    for percentile in [70, 80, 90, 95]:
        mask = extractor.adaptive_threshold_percentile(percentile=percentile)
        light_pixels = np.sum(mask > 0)
        percentage = (light_pixels / (extractor.width * extractor.height)) * 100
        print(f"Percentile {percentile}%: {light_pixels:,} pixels ({percentage:.1f}%)")
    
    # Try different HSV parameters
    print("\nHSV method with different parameters:")
    for value_thresh in [0.5, 0.6, 0.7, 0.8]:
        mask = extractor.adaptive_threshold_hsv(value_threshold=value_thresh)
        light_pixels = np.sum(mask > 0)
        percentage = (light_pixels / (extractor.width * extractor.height)) * 100
        print(f"Value threshold {value_thresh}: {light_pixels:,} pixels ({percentage:.1f}%)")

def demo_comparison():
    """Demonstrate comparison of all methods"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = AdaptiveLightExtractor(image_path)
    
    print("\n=== METHOD COMPARISON DEMO ===")
    
    # Extract using all methods
    results = extractor.extract_all_methods()
    
    # Sort by percentage of light pixels
    method_results = []
    for method, mask in results.items():
        light_pixels = np.sum(mask > 0)
        percentage = (light_pixels / (extractor.width * extractor.height)) * 100
        method_results.append((method, light_pixels, percentage))
    
    # Sort by percentage (ascending)
    method_results.sort(key=lambda x: x[2])
    
    print("Methods ranked by light pixel percentage (low to high):")
    for method, pixels, percentage in method_results:
        print(f"  {method:12}: {pixels:6,} pixels ({percentage:5.1f}%)")

if __name__ == "__main__":
    demo_basic_usage()
    demo_custom_parameters()
    demo_comparison()
