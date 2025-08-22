#!/usr/bin/env python3
"""
Simple Statistical Light Extractor Usage

Easy-to-use script focusing on the statistical method with optimal parameters.
"""

from src.core.statistical_light_extractor import StatisticalLightExtractor
import cv2
import os

def extract_light_pixels(image_path, 
                        std_multiplier=1.5, 
                        use_median=False,
                        save_result=True):
    """
    Extract light pixels using optimized statistical method
    
    Args:
        image_path: Path to input image
        std_multiplier: Controls selectivity (lower = more selective)
                       Recommended range: 0.5-3.0
                       - 0.5-1.0: Very selective (1-10% light pixels)
                       - 1.0-2.0: Moderate (5-20% light pixels)  
                       - 2.0-3.0: Inclusive (20%+ light pixels)
        use_median: Use median instead of mean as base (more robust)
        save_result: Save the result mask
    
    Returns:
        Tuple of (mask, info_dict)
    """
    
    # Initialize extractor
    extractor = StatisticalLightExtractor(image_path)
    
    # Extract using statistical method
    mask, info = extractor.extract_statistical_basic(
        std_multiplier=std_multiplier,
        use_median=use_median
    )
    
    # Clean up the mask
    cleaned_mask = extractor.morphological_cleanup(mask, kernel_size=3, operations="close")
    
    # Print results
    print(f"Statistical Light Extraction Results:")
    print(f"  Image: {os.path.basename(image_path)} ({extractor.width}x{extractor.height})")
    print(f"  Parameters: std_multiplier={std_multiplier}, use_median={use_median}")
    print(f"  Threshold: {info['threshold_value']:.1f}")
    print(f"  Light pixels: {info['light_pixels']:,} ({info['light_percentage']:.1f}%)")
    
    # Save result if requested
    if save_result:
        output_dir = "/home/jimmy/code/2Dto3D_2/output"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_statistical_light_mask.png")
        cv2.imwrite(output_path, cleaned_mask)
        print(f"  Saved to: {output_path}")
    
    return cleaned_mask, info

def main():
    """Main demonstration"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    
    print("=== STATISTICAL LIGHT EXTRACTION ===\n")
    
    # Test with current optimal parameters (gives ~4.4% light pixels)
    print("1. Optimal parameters (recommended):")
    mask1, info1 = extract_light_pixels(image_path, std_multiplier=1.5, use_median=False)
    
    print("\n" + "="*50 + "\n")
    
    # Test with more selective parameters  
    print("2. More selective (fewer light pixels):")
    mask2, info2 = extract_light_pixels(image_path, std_multiplier=2.0, use_median=False)
    
    print("\n" + "="*50 + "\n")
    
    # Test with less selective parameters
    print("3. Less selective (more light pixels):")
    mask3, info3 = extract_light_pixels(image_path, std_multiplier=1.0, use_median=False)
    
    print("\n" + "="*50 + "\n")
    
    # Test with median base (more robust)
    print("4. Using median base (more robust):")
    mask4, info4 = extract_light_pixels(image_path, std_multiplier=1.5, use_median=True)
    
    print(f"\n{'='*50}")
    print("PARAMETER GUIDE:")
    print("="*50)
    print("std_multiplier parameter effects:")
    print("  0.5-1.0  : Very selective (1-10% light pixels)")
    print("  1.0-1.5  : Moderate selection (5-15% light pixels)")  
    print("  1.5-2.0  : Balanced (3-8% light pixels)")
    print("  2.0-3.0  : Conservative (1-3% light pixels)")
    print("  3.0+     : Very conservative (<1% light pixels)")
    print()
    print("use_median parameter:")
    print("  False : Use mean (standard approach)")
    print("  True  : Use median (more robust to outliers)")
    print()
    print("Recommendation for your image:")
    print("  std_multiplier=1.5, use_median=False")
    print("  This gives ~4.4% light pixels with good precision")

if __name__ == "__main__":
    main()
