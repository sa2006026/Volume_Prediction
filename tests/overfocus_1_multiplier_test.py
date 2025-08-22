#!/usr/bin/env python3
"""
Multiplier test specifically for overfocus_1.jpg to find optimal white pixel extraction
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.core.adaptive_light_extraction import AdaptiveLightExtractor
import cv2

def test_overfocus_1_multipliers():
    """Test different multipliers specifically for overfocus_1.jpg"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus_1.jpg"
    output_dir = "./results/overfocus_1_multiplier_test/"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = AdaptiveLightExtractor(image_path)
    print(f"Loaded image: {image_path} ({extractor.width}x{extractor.height})")
    
    # Get image statistics
    stats = extractor.analyze_image_statistics()
    print(f"\nImage Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test multipliers specifically good for white pixel extraction
    multipliers = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    print(f"\nTesting {len(multipliers)} multiplier values for white pixel extraction: {multipliers}")
    
    results = []
    masks = {}
    
    # Test each multiplier
    for multiplier in multipliers:
        print(f"\nTesting multiplier: {multiplier}")
        
        # Calculate threshold using the statistical method
        threshold_value = stats['mean_intensity'] + (multiplier * stats['std_intensity'])
        threshold_value = min(threshold_value, 255)  # Clamp to valid range
        
        # Apply threshold
        binary_mask = (extractor.gray_image >= threshold_value).astype(np.uint8) * 255
        
        # Apply morphological cleanup
        cleaned_mask = extractor.morphological_cleanup(binary_mask)
        
        # Calculate statistics
        light_pixels = np.sum(cleaned_mask > 0)
        light_percentage = (light_pixels / (extractor.width * extractor.height)) * 100
        
        # Store results
        result = {
            'multiplier': multiplier,
            'threshold': threshold_value,
            'light_pixels': light_pixels,
            'light_percentage': light_percentage
        }
        results.append(result)
        masks[f"mult_{multiplier:.1f}"] = cleaned_mask
        
        print(f"  Threshold: {threshold_value:.1f}")
        print(f"  Light pixels: {light_pixels:,} ({light_percentage:.1f}%)")
        
        # Save individual mask
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_filename = f"{base_name}_white_mult_{multiplier:.1f}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, cleaned_mask)
    
    # Create comparison visualization
    create_white_pixel_comparison(image_path, results, masks, output_dir, stats)
    
    return results

def create_white_pixel_comparison(image_path, results, masks, output_dir, stats):
    """Create visualization specifically for white pixel extraction comparison"""
    
    extractor = AdaptiveLightExtractor(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create comparison plot
    n_multipliers = len(results)
    n_cols = 4
    n_rows = (n_multipliers + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(extractor.rgb_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show results for each multiplier
    for i, result in enumerate(results, 1):
        if i < len(axes):
            multiplier = result['multiplier']
            mask_key = f"mult_{multiplier:.1f}"
            mask = masks[mask_key]
            
            axes[i].imshow(mask, cmap='gray')
            title = f"Mult: {multiplier:.1f}\n{result['light_percentage']:.1f}%\nThresh: {result['threshold']:.0f}"
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(results) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'White Pixel Extraction - {base_name}\n'
                f'Mean: {stats["mean_intensity"]:.1f} | Std: {stats["std_intensity"]:.1f}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.join(output_dir, f"{base_name}_white_pixel_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"White pixel comparison saved to: {comparison_path}")
    plt.show()

def print_white_pixel_recommendations(results):
    """Print recommendations specifically for white pixel extraction"""
    
    print(f"\n{'='*80}")
    print("WHITE PIXEL EXTRACTION RECOMMENDATIONS")
    print(f"{'='*80}")
    print(f"{'Multiplier':<10} {'Threshold':<10} {'Light Pixels':<12} {'Percentage':<12} {'Best For'}")
    print(f"{'-'*80}")
    
    for result in results:
        mult = result['multiplier']
        thresh = result['threshold']
        pixels = result['light_pixels']
        perc = result['light_percentage']
        
        # Determine best use case for white pixel extraction
        if perc > 50:
            best_for = "Too inclusive"
        elif perc > 30:
            best_for = "Bright regions"
        elif perc > 15:
            best_for = "Light areas"
        elif perc > 8:
            best_for = "White/bright spots"
        elif perc > 3:
            best_for = "Pure white pixels"
        else:
            best_for = "Very selective"
        
        print(f"{mult:<10.1f} {thresh:<10.1f} {pixels:<12,} {perc:<12.1f} {best_for}")
    
    # Find best options for different white pixel extraction needs
    print(f"\n{'='*80}")
    print("RECOMMENDED SETTINGS:")
    print(f"{'='*80}")
    
    # For capturing most white/bright pixels
    inclusive = [r for r in results if 10 <= r['light_percentage'] <= 25]
    if inclusive:
        best_inclusive = inclusive[0]
        print(f"For white/bright regions: multiplier = {best_inclusive['multiplier']} ({best_inclusive['light_percentage']:.1f}%)")
    
    # For pure white pixels only
    selective = [r for r in results if 2 <= r['light_percentage'] <= 8]
    if selective:
        best_selective = selective[-1]  # Most selective in range
        print(f"For pure white pixels: multiplier = {best_selective['multiplier']} ({best_selective['light_percentage']:.1f}%)")
    
    # Most balanced
    balanced = min(results, key=lambda x: abs(x['light_percentage'] - 10.0))
    print(f"Balanced approach: multiplier = {balanced['multiplier']} ({balanced['light_percentage']:.1f}%)")

def main():
    """Main function"""
    
    print("="*80)
    print("WHITE PIXEL EXTRACTION FROM OVERFOCUS_1.JPG")
    print("="*80)
    
    try:
        results = test_overfocus_1_multipliers()
        print_white_pixel_recommendations(results)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
