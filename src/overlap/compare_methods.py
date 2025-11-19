#!/usr/bin/env python3
"""
Compare Z-Stack Merging Methods
Creates a side-by-side comparison of all merging methods
"""

import cv2
import numpy as np
from pathlib import Path


def create_comparison_grid():
    """Create a 2x2 grid comparing all merging methods"""
    
    # Load all merged images
    methods = {
        'MIP (Maximum Intensity)': 'merged_zstack_MIP.jpg',
        'EDF (Focus Stacking)': 'merged_zstack_EDF.jpg',
        'Average Projection': 'merged_zstack_AVG.jpg',
        'Weighted Average': 'merged_zstack_WEIGHTED.jpg'
    }
    
    images = {}
    for name, filename in methods.items():
        img = cv2.imread(filename)
        if img is not None:
            images[name] = img
            print(f"✓ Loaded: {name}")
        else:
            print(f"✗ Failed to load: {filename}")
    
    if len(images) != 4:
        print("❌ Error: Not all methods were generated. Run merge_zstack.py first.")
        return
    
    # Get image dimensions
    h, w = list(images.values())[0].shape[:2]
    
    # Add text labels to each image
    labeled_images = {}
    for name, img in images.items():
        labeled = img.copy()
        
        # Add semi-transparent background for text
        overlay = labeled.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        labeled = cv2.addWeighted(labeled, 0.7, overlay, 0.3, 0)
        
        # Add text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (255, 255, 255)
        
        # Center the text
        text_size = cv2.getTextSize(name, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 35
        
        cv2.putText(labeled, name, (text_x, text_y), font, font_scale, color, thickness)
        labeled_images[name] = labeled
    
    # Create 2x2 grid
    row1 = np.hstack([
        labeled_images['MIP (Maximum Intensity)'],
        labeled_images['EDF (Focus Stacking)']
    ])
    
    row2 = np.hstack([
        labeled_images['Average Projection'],
        labeled_images['Weighted Average']
    ])
    
    grid = np.vstack([row1, row2])
    
    # Add border
    border_color = (100, 100, 100)
    grid = cv2.copyMakeBorder(grid, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
    
    # Save comparison
    output_path = 'zstack_methods_comparison.jpg'
    cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n✅ Comparison grid saved: {output_path}")
    print(f"   Grid size: {grid.shape[1]}x{grid.shape[0]}")
    
    # Also create a larger version for detailed viewing
    scale = 1.5
    large_grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    large_output = 'zstack_methods_comparison_large.jpg'
    cv2.imwrite(large_output, large_grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"   Large version saved: {large_output}")


def create_method_recommendation_image():
    """Create a recommendation guide image"""
    
    # Load MIP image (recommended for ddPCR)
    mip = cv2.imread('merged_zstack_MIP.jpg')
    if mip is None:
        print("❌ Could not load MIP image")
        return
    
    h, w = mip.shape[:2]
    
    # Create an annotated version
    annotated = mip.copy()
    
    # Add recommendation banner at the top
    overlay = annotated.copy()
    banner_height = 80
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 100, 0), -1)
    annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)
    
    # Add recommendation text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    text1 = "RECOMMENDED: Maximum Intensity Projection"
    text2 = "Best for ddPCR droplet counting & analysis"
    
    cv2.putText(annotated, text1, (20, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, text2, (20, 60), font, 0.5, (200, 255, 200), 1)
    
    # Save annotated version
    output_path = 'recommended_merged_zstack.jpg'
    cv2.imwrite(output_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✅ Recommended method saved: {output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Z-Stack Methods Comparison")
    print("=" * 60)
    print()
    
    create_comparison_grid()
    print()
    create_method_recommendation_image()
    
    print()
    print("=" * 60)
    print("✅ Comparison images created!")
    print("=" * 60)

