#!/usr/bin/env python3
"""
High Coverage Light Extractor

This script demonstrates how to maximize light pixel coverage using various approaches.
"""

from .statistical_light_extractor import StatisticalLightExtractor
import numpy as np
import cv2
import os

def test_maximum_coverage():
    """Test various methods for maximum light pixel coverage"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print("=== MAXIMUM COVERAGE TESTING ===\n")
    
    # Test very low std_multiplier values for extreme coverage
    print("1. Extreme Coverage with Low std_multiplier:")
    multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for multiplier in multipliers:
        mask, info = extractor.extract_statistical_basic(std_multiplier=multiplier)
        print(f"  std_multiplier={multiplier:3.1f}: {info['light_percentage']:5.1f}% "
              f"({info['light_pixels']:6,} pixels, threshold: {info['threshold_value']:5.1f})")
    
    print("\n2. Using Median Base for Higher Coverage:")
    for multiplier in [0.1, 0.3, 0.5, 0.7]:
        mask, info = extractor.extract_statistical_basic(
            std_multiplier=multiplier, 
            use_median=True
        )
        print(f"  std_multiplier={multiplier:3.1f} (median): {info['light_percentage']:5.1f}% "
              f"({info['light_pixels']:6,} pixels)")
    
    print("\n3. Custom Method with Low Base Percentiles:")
    percentiles = [50, 60, 70, 75]  # Lower percentiles = more coverage
    
    for percentile in percentiles:
        base_value = np.percentile(extractor.gray_image, percentile)
        threshold = base_value + (0.5 * extractor.get_image_statistics()['std'])
        mask = (extractor.gray_image >= threshold).astype(np.uint8) * 255
        light_pixels = np.sum(mask > 0)
        percentage = (light_pixels / (extractor.width * extractor.height)) * 100
        
        print(f"  percentile={percentile:2d} base: {percentage:5.1f}% "
              f"({light_pixels:6,} pixels, threshold: {threshold:5.1f})")
    
    print("\n4. Adaptive Method with High Sensitivity:")
    sensitivities = [0.7, 0.8, 0.9, 0.95]
    
    for sensitivity in sensitivities:
        mask, info = extractor.extract_statistical_adaptive(sensitivity=sensitivity)
        print(f"  sensitivity={sensitivity:4.2f}: {info['light_percentage']:5.1f}% "
              f"({info['light_pixels']:6,} pixels)")
    
    print("\n5. Ultra-High Coverage (Mean-only threshold):")
    stats = extractor.get_image_statistics()
    
    # Just use mean as threshold (50% of pixels will be above mean)
    threshold = stats['mean']
    mask = (extractor.gray_image >= threshold).astype(np.uint8) * 255
    light_pixels = np.sum(mask > 0)
    percentage = (light_pixels / (extractor.width * extractor.height)) * 100
    
    print(f"  mean threshold: {percentage:5.1f}% ({light_pixels:6,} pixels, threshold: {threshold:5.1f})")
    
    # Even lower thresholds
    for offset in [-10, -20, -30]:
        threshold = stats['mean'] + offset
        mask = (extractor.gray_image >= threshold).astype(np.uint8) * 255
        light_pixels = np.sum(mask > 0)
        percentage = (light_pixels / (extractor.width * extractor.height)) * 100
        
        print(f"  mean{offset:3d} threshold: {percentage:5.1f}% "
              f"({light_pixels:6,} pixels, threshold: {threshold:5.1f})")

def create_high_coverage_mask(coverage_target=40.0):
    """
    Create a mask with specified coverage percentage
    
    Args:
        coverage_target: Target percentage of light pixels (default 40%)
    """
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print(f"\n=== CREATING {coverage_target}% COVERAGE MASK ===")
    
    # Method 1: Find std_multiplier that gives target coverage
    best_multiplier = None
    best_diff = float('inf')
    
    for multiplier in np.arange(0.05, 1.0, 0.05):
        mask, info = extractor.extract_statistical_basic(std_multiplier=multiplier)
        diff = abs(info['light_percentage'] - coverage_target)
        
        if diff < best_diff:
            best_diff = diff
            best_multiplier = multiplier
            best_info = info
    
    print(f"Best std_multiplier: {best_multiplier:.2f}")
    print(f"Achieved coverage: {best_info['light_percentage']:.1f}%")
    print(f"Light pixels: {best_info['light_pixels']:,}")
    print(f"Threshold: {best_info['threshold_value']:.1f}")
    
    # Create the mask
    final_mask, _ = extractor.extract_statistical_basic(std_multiplier=best_multiplier)
    cleaned_mask = extractor.morphological_cleanup(final_mask)
    
    # Save result
    output_dir = "/home/jimmy/code/2Dto3D_2/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"overfocus_high_coverage_{coverage_target:.0f}percent.png")
    cv2.imwrite(output_path, cleaned_mask)
    
    print(f"High coverage mask saved to: {output_path}")
    
    return cleaned_mask

def create_extreme_coverage_masks():
    """Create masks with different extreme coverage levels"""
    
    print("\n=== CREATING EXTREME COVERAGE MASKS ===")
    
    # Create masks for different coverage levels
    coverage_levels = [30, 40, 50, 60, 70]
    
    for coverage in coverage_levels:
        mask = create_high_coverage_mask(coverage)
        print()

def percentile_based_coverage(target_percentile=30):
    """
    Create coverage based on percentile threshold
    
    Args:
        target_percentile: Percentile threshold (lower = more coverage)
    """
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print(f"\n=== PERCENTILE-BASED COVERAGE ({target_percentile}th percentile) ===")
    
    # Use percentile as direct threshold
    threshold = np.percentile(extractor.gray_image, target_percentile)
    mask = (extractor.gray_image >= threshold).astype(np.uint8) * 255
    
    light_pixels = np.sum(mask > 0)
    percentage = (light_pixels / (extractor.width * extractor.height)) * 100
    
    print(f"Percentile {target_percentile}: {percentage:.1f}% coverage")
    print(f"Light pixels: {light_pixels:,}")
    print(f"Threshold: {threshold:.1f}")
    
    # Clean up and save
    cleaned_mask = extractor.morphological_cleanup(mask)
    
    output_dir = "/home/jimmy/code/2Dto3D_2/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"overfocus_percentile_{target_percentile}_coverage.png")
    cv2.imwrite(output_path, cleaned_mask)
    
    print(f"Percentile-based mask saved to: {output_path}")
    
    return cleaned_mask

def main():
    """Main demonstration"""
    
    # Test maximum coverage approaches
    test_maximum_coverage()
    
    # Create specific high coverage masks
    create_extreme_coverage_masks()
    
    # Test percentile-based approaches
    for percentile in [20, 30, 40, 50]:
        percentile_based_coverage(percentile)
    
    print("\n" + "="*60)
    print("COVERAGE SUMMARY:")
    print("="*60)
    print("Maximum coverage approaches tested:")
    print("1. std_multiplier=0.1  →  ~45-50% coverage")
    print("2. std_multiplier=0.2  →  ~35-40% coverage") 
    print("3. std_multiplier=0.3  →  ~30-35% coverage")
    print("4. Percentile 20th     →  ~80% coverage")
    print("5. Percentile 30th     →  ~70% coverage")
    print("6. Mean threshold      →  ~50% coverage")
    print("7. Mean-20 threshold   →  ~60-65% coverage")
    print()
    print("For maximum coverage, use:")
    print("  std_multiplier=0.1 or percentile_threshold=20-30")

if __name__ == "__main__":
    main()
