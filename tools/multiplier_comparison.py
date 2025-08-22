#!/usr/bin/env python3
"""
Multiplier Comparison Script for Statistical Light Extractor

This script creates a visual comparison of different std_multiplier values
to help you choose the best parameter for your image.
"""

import sys
import os
from src.core.statistical_light_extractor import StatisticalLightExtractor

def compare_multipliers(image_path: str, multipliers: list, output_path: str = None):
    """
    Create a visual comparison of different std_multiplier values
    
    Args:
        image_path: Path to the input image
        multipliers: List of std_multiplier values to compare
        output_path: Optional path to save the comparison image
    """
    
    # Initialize extractor
    extractor = StatisticalLightExtractor(image_path)
    print(f"Loaded image: {image_path} ({extractor.width}x{extractor.height})")
    
    # Show image statistics
    stats = extractor.get_image_statistics()
    print(f"\nImage Statistics:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Dynamic Range: {stats['dynamic_range']:.2f}")
    
    # Create method and parameter combinations for comparison
    methods_and_params = []
    
    print(f"\nTesting {len(multipliers)} different multipliers:")
    
    for multiplier in multipliers:
        # Test the multiplier
        mask, info = extractor.extract_statistical_basic(std_multiplier=multiplier)
        
        print(f"  Multiplier {multiplier:4.1f}: {info['light_percentage']:5.1f}% light pixels "
              f"(threshold: {info['threshold_value']:6.1f})")
        
        # Add to comparison list
        methods_and_params.append(("basic", {"std_multiplier": multiplier}))
    
    # Create visualization with multiplier values in filename
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # Create a string representation of multipliers for filename
        multiplier_str = "_".join([f"{m:.1f}".replace(".", "p") for m in multipliers])
        output_path = f"/home/jimmy/code/2Dto3D_2/output/{base_name}_multipliers_{multiplier_str}.png"
    
    print(f"\nGenerating comparison visualization...")
    extractor.visualize_comparison(methods_and_params, output_path)
    
    print(f"Comparison saved to: {output_path}")
    
    return methods_and_params

def main():
    """Main function"""
    
    # Default image path (you can change this)
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please update the image_path in the script or provide it as an argument.")
        return 1
    
    # Define multipliers to test (you can customize these)
    multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        # Parse multipliers from command line
        try:
            multipliers = [float(x) for x in sys.argv[2].split(',')]
        except ValueError:
            print("Error: Multipliers should be comma-separated numbers (e.g., '0.5,1.0,1.5,2.0')")
            return 1
    
    try:
        # Run comparison
        compare_multipliers(image_path, multipliers)
        
        print(f"\n=== RECOMMENDATIONS ===")
        print("- Lower multipliers (0.5-1.0): More light pixels detected, less selective")
        print("- Medium multipliers (1.5-2.0): Balanced selection, good starting point")
        print("- Higher multipliers (2.5-3.0): Fewer light pixels, more selective")
        print("\nChoose based on your specific needs and the results shown above.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
