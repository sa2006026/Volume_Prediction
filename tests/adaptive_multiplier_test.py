#!/usr/bin/env python3
"""
Adaptive Light Extraction with Multiplier Range Testing

This script uses the AdaptiveLightExtractor to test different std_multiplier values
from 0.5 to 2.0 with 0.1 intervals for statistical light extraction.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.core.adaptive_light_extraction import AdaptiveLightExtractor
import cv2

def test_multiplier_range(image_path: str, 
                         multiplier_start: float = 0.5, 
                         multiplier_end: float = 2.0, 
                         interval: float = 0.1,
                         output_dir: str = "./results/adaptive_multiplier_test/"):
    """
    Test adaptive statistical extraction with a range of multiplier values
    
    Args:
        image_path: Path to input image
        multiplier_start: Starting multiplier value
        multiplier_end: Ending multiplier value
        interval: Interval between multiplier values
        output_dir: Output directory for results
    """
    
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
    
    # Generate multiplier values
    multipliers = np.arange(multiplier_start, multiplier_end + interval, interval)
    multipliers = np.round(multipliers, 1)  # Round to avoid floating point errors
    
    print(f"\nTesting {len(multipliers)} multiplier values: {multipliers}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
        mask_filename = f"{base_name}_adaptive_mult_{multiplier:.1f}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, cleaned_mask)
    
    return results, masks, stats

def create_comparison_visualization(image_path: str, results: list, masks: dict, 
                                  output_dir: str, stats: dict):
    """
    Create a comprehensive visualization comparing all multiplier results
    """
    extractor = AdaptiveLightExtractor(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create a large comparison plot
    n_multipliers = len(results)
    n_cols = 5  # 5 columns for better layout
    n_rows = (n_multipliers + n_cols) // n_cols  # +1 row for original
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
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
    
    plt.suptitle(f'Adaptive Statistical Extraction - Multiplier Comparison\n'
                f'Image: {base_name} | Mean: {stats["mean_intensity"]:.1f} | Std: {stats["std_intensity"]:.1f}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.join(output_dir, f"{base_name}_adaptive_multiplier_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to: {comparison_path}")
    plt.show()

def create_statistics_plot(results: list, output_dir: str, image_path: str):
    """
    Create plots showing the relationship between multipliers and results
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    multipliers = [r['multiplier'] for r in results]
    thresholds = [r['threshold'] for r in results]
    percentages = [r['light_percentage'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Threshold vs Multiplier
    ax1.plot(multipliers, thresholds, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Std Multiplier', fontsize=12)
    ax1.set_ylabel('Threshold Value', fontsize=12)
    ax1.set_title('Threshold vs Multiplier', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(multipliers) - 0.1, max(multipliers) + 0.1)
    
    # Plot 2: Light Percentage vs Multiplier
    ax2.plot(multipliers, percentages, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Std Multiplier', fontsize=12)
    ax2.set_ylabel('Light Pixels (%)', fontsize=12)
    ax2.set_title('Light Percentage vs Multiplier', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(multipliers) - 0.1, max(multipliers) + 0.1)
    
    # Add value annotations
    for i, (mult, thresh, perc) in enumerate(zip(multipliers, thresholds, percentages)):
        if i % 3 == 0:  # Annotate every 3rd point to avoid clutter
            ax1.annotate(f'{thresh:.0f}', (mult, thresh), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
            ax2.annotate(f'{perc:.1f}%', (mult, perc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    plt.suptitle(f'Adaptive Statistical Analysis - {base_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save statistics plot
    stats_path = os.path.join(output_dir, f"{base_name}_adaptive_statistics.png")
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    print(f"Statistics plot saved to: {stats_path}")
    plt.show()

def print_results_table(results: list):
    """
    Print a formatted table of results
    """
    print(f"\n{'='*80}")
    print("ADAPTIVE STATISTICAL EXTRACTION RESULTS")
    print(f"{'='*80}")
    print(f"{'Multiplier':<10} {'Threshold':<10} {'Light Pixels':<12} {'Percentage':<12} {'Selectivity'}")
    print(f"{'-'*80}")
    
    for result in results:
        mult = result['multiplier']
        thresh = result['threshold']
        pixels = result['light_pixels']
        perc = result['light_percentage']
        
        # Determine selectivity level
        if perc > 15:
            selectivity = "Very Inclusive"
        elif perc > 8:
            selectivity = "Inclusive"
        elif perc > 3:
            selectivity = "Balanced"
        elif perc > 1:
            selectivity = "Selective"
        else:
            selectivity = "Very Selective"
        
        print(f"{mult:<10.1f} {thresh:<10.1f} {pixels:<12,} {perc:<12.1f} {selectivity}")

def main():
    """Main function"""
    
    # Configuration
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"  # Update this path as needed
    output_dir = "./results/adaptive_multiplier_test/"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please update the image_path in the script.")
        return 1
    
    print("="*80)
    print("ADAPTIVE LIGHT EXTRACTION - MULTIPLIER RANGE TEST")
    print("="*80)
    print(f"Testing multipliers from 0.5 to 2.0 with 0.1 intervals")
    print(f"Using adaptive statistical extraction method")
    print(f"Output directory: {output_dir}")
    
    try:
        # Run the multiplier range test
        results, masks, stats = test_multiplier_range(
            image_path=image_path,
            multiplier_start=0.5,
            multiplier_end=2.0,
            interval=0.1,
            output_dir=output_dir
        )
        
        # Print results table
        print_results_table(results)
        
        # Create visualizations
        print(f"\nGenerating visualizations...")
        create_comparison_visualization(image_path, results, masks, output_dir, stats)
        create_statistics_plot(results, output_dir, image_path)
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {len(results)}")
        print(f"Multiplier range: 0.5 - 2.0 (interval: 0.1)")
        print(f"Output directory: {output_dir}")
        print(f"Individual masks saved: {len(masks)}")
        print(f"Comparison visualization: {os.path.basename(image_path).split('.')[0]}_adaptive_multiplier_comparison.png")
        print(f"Statistics plot: {os.path.basename(image_path).split('.')[0]}_adaptive_statistics.png")
        
        # Recommendations
        best_balanced = min(results, key=lambda x: abs(x['light_percentage'] - 5.0))
        print(f"\nRECOMMENDATIONS:")
        print(f"For ~5% light extraction: multiplier = {best_balanced['multiplier']:.1f} ({best_balanced['light_percentage']:.1f}%)")
        
        conservative = [r for r in results if r['light_percentage'] < 2.0]
        if conservative:
            print(f"Most conservative (< 2%): multiplier = {conservative[-1]['multiplier']:.1f} ({conservative[-1]['light_percentage']:.1f}%)")
        
        inclusive = [r for r in results if r['light_percentage'] > 10.0]
        if inclusive:
            print(f"Most inclusive (> 10%): multiplier = {inclusive[0]['multiplier']:.1f} ({inclusive[0]['light_percentage']:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
