#!/usr/bin/env python3
"""
Analyze Z-Stack Intensity Distribution
Determines which z-plane has the highest fluorescent intensity droplets
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def analyze_zstack_intensity(image_dir: str, output_report: str = "intensity_analysis.txt"):
    """
    Analyze intensity distribution across z-stack planes
    
    Args:
        image_dir: Directory containing z-stack images
        output_report: Output report filename
    """
    image_dir = Path(image_dir)
    
    # Find all z-stack images
    image_files = sorted(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print("=" * 70)
    print("Z-STACK INTENSITY ANALYSIS")
    print("=" * 70)
    print()
    
    # Store statistics for each z-plane
    z_stats = []
    
    print("Analyzing each z-plane...")
    print("-" * 70)
    
    for img_path in image_files:
        # Extract z-index from filename
        filename = img_path.name
        try:
            z_idx = int(filename.split('_z')[1].split('_')[0])
        except:
            print(f"⚠️  Could not parse z-index from {filename}, skipping")
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load {filename}, skipping")
            continue
        
        # Convert to grayscale for intensity analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate intensity statistics
        mean_intensity = np.mean(gray)
        max_intensity = np.max(gray)
        std_intensity = np.std(gray)
        
        # Calculate high-intensity pixel count (bright droplets)
        # Threshold at 200+ for bright fluorescent droplets
        bright_pixels = np.sum(gray >= 200)
        very_bright_pixels = np.sum(gray >= 240)
        
        # Calculate 95th percentile (captures brightest droplets)
        percentile_95 = np.percentile(gray, 95)
        percentile_99 = np.percentile(gray, 99)
        
        # Calculate total intensity (sum of all pixel values)
        total_intensity = np.sum(gray.astype(np.float64))
        
        z_stats.append({
            'z_index': z_idx,
            'filename': filename,
            'mean_intensity': mean_intensity,
            'max_intensity': max_intensity,
            'std_intensity': std_intensity,
            'bright_pixels': bright_pixels,
            'very_bright_pixels': very_bright_pixels,
            'percentile_95': percentile_95,
            'percentile_99': percentile_99,
            'total_intensity': total_intensity
        })
        
        print(f"z{z_idx:02d}: Mean={mean_intensity:6.2f}, Max={max_intensity:3d}, "
              f"Bright={bright_pixels:6d}, VeryBright={very_bright_pixels:5d}, "
              f"P95={percentile_95:6.2f}")
    
    if not z_stats:
        print("❌ No valid z-planes analyzed")
        return
    
    # Sort by z-index
    z_stats.sort(key=lambda x: x['z_index'])
    
    print()
    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print()
    
    # Find peak planes for different metrics
    max_mean_idx = max(z_stats, key=lambda x: x['mean_intensity'])
    max_bright_idx = max(z_stats, key=lambda x: x['bright_pixels'])
    max_very_bright_idx = max(z_stats, key=lambda x: x['very_bright_pixels'])
    max_total_idx = max(z_stats, key=lambda x: x['total_intensity'])
    max_p99_idx = max(z_stats, key=lambda x: x['percentile_99'])
    
    print("🔍 PEAK PLANES BY DIFFERENT METRICS:")
    print("-" * 70)
    print(f"  Highest Mean Intensity:      z{max_mean_idx['z_index']:02d} ({max_mean_idx['mean_intensity']:.2f})")
    print(f"  Most Bright Pixels (≥200):   z{max_bright_idx['z_index']:02d} ({max_bright_idx['bright_pixels']:,} pixels)")
    print(f"  Most Very Bright (≥240):     z{max_very_bright_idx['z_index']:02d} ({max_very_bright_idx['very_bright_pixels']:,} pixels)")
    print(f"  Highest Total Intensity:     z{max_total_idx['z_index']:02d} ({max_total_idx['total_intensity']:.0f})")
    print(f"  Highest 99th Percentile:     z{max_p99_idx['z_index']:02d} ({max_p99_idx['percentile_99']:.2f})")
    print()
    
    # Determine the best plane (using bright pixels as primary metric for droplets)
    best_plane = max_bright_idx
    print("⭐ RECOMMENDED PEAK PLANE FOR DROPLET ANALYSIS:")
    print("-" * 70)
    print(f"  z{best_plane['z_index']:02d} - {best_plane['filename']}")
    print(f"  This plane has the most bright pixels (≥200 intensity)")
    print(f"  Bright pixel count: {best_plane['bright_pixels']:,}")
    print(f"  Very bright pixels: {best_plane['very_bright_pixels']:,}")
    print(f"  Mean intensity: {best_plane['mean_intensity']:.2f}")
    print()
    
    # Create intensity distribution plot
    print("📊 Creating intensity distribution plots...")
    create_intensity_plots(z_stats, image_dir.name)
    
    # Save detailed report
    save_report(z_stats, best_plane, output_report)
    
    print()
    print("=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)
    
    return z_stats, best_plane


def create_intensity_plots(z_stats, series_name):
    """Create visualization plots of intensity distribution"""
    
    z_indices = [s['z_index'] for s in z_stats]
    mean_intensities = [s['mean_intensity'] for s in z_stats]
    bright_pixels = [s['bright_pixels'] for s in z_stats]
    very_bright_pixels = [s['very_bright_pixels'] for s in z_stats]
    percentile_99 = [s['percentile_99'] for s in z_stats]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Z-Stack Intensity Analysis - {series_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Intensity
    axes[0, 0].plot(z_indices, mean_intensities, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Z-Plane Index', fontsize=10)
    axes[0, 0].set_ylabel('Mean Intensity', fontsize=10)
    axes[0, 0].set_title('Mean Intensity Across Z-Stack', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    peak_idx = z_stats[mean_intensities.index(max(mean_intensities))]['z_index']
    axes[0, 0].axvline(x=peak_idx, color='r', linestyle='--', alpha=0.5, label=f'Peak: z{peak_idx:02d}')
    axes[0, 0].legend()
    
    # Plot 2: Bright Pixels Count
    axes[0, 1].plot(z_indices, bright_pixels, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Z-Plane Index', fontsize=10)
    axes[0, 1].set_ylabel('Number of Bright Pixels (≥200)', fontsize=10)
    axes[0, 1].set_title('Bright Droplet Pixels Across Z-Stack', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    peak_idx = z_stats[bright_pixels.index(max(bright_pixels))]['z_index']
    axes[0, 1].axvline(x=peak_idx, color='r', linestyle='--', alpha=0.5, label=f'Peak: z{peak_idx:02d}')
    axes[0, 1].legend()
    
    # Plot 3: Very Bright Pixels
    axes[1, 0].plot(z_indices, very_bright_pixels, 'm-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Z-Plane Index', fontsize=10)
    axes[1, 0].set_ylabel('Number of Very Bright Pixels (≥240)', fontsize=10)
    axes[1, 0].set_title('Very Bright Droplet Pixels Across Z-Stack', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    peak_idx = z_stats[very_bright_pixels.index(max(very_bright_pixels))]['z_index']
    axes[1, 0].axvline(x=peak_idx, color='r', linestyle='--', alpha=0.5, label=f'Peak: z{peak_idx:02d}')
    axes[1, 0].legend()
    
    # Plot 4: 99th Percentile
    axes[1, 1].plot(z_indices, percentile_99, 'r-o', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Z-Plane Index', fontsize=10)
    axes[1, 1].set_ylabel('99th Percentile Intensity', fontsize=10)
    axes[1, 1].set_title('99th Percentile Intensity Across Z-Stack', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    peak_idx = z_stats[percentile_99.index(max(percentile_99))]['z_index']
    axes[1, 1].axvline(x=peak_idx, color='r', linestyle='--', alpha=0.5, label=f'Peak: z{peak_idx:02d}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    output_filename = f'intensity_analysis_{series_name}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot: {output_filename}")
    plt.close()


def save_report(z_stats, best_plane, output_filename):
    """Save detailed text report"""
    
    with open(output_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Z-STACK INTENSITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RECOMMENDED PEAK PLANE FOR DROPLET ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"z{best_plane['z_index']:02d} - {best_plane['filename']}\n")
        f.write(f"  Bright pixel count (≥200): {best_plane['bright_pixels']:,}\n")
        f.write(f"  Very bright pixels (≥240): {best_plane['very_bright_pixels']:,}\n")
        f.write(f"  Mean intensity: {best_plane['mean_intensity']:.2f}\n")
        f.write(f"  99th percentile: {best_plane['percentile_99']:.2f}\n")
        f.write("\n")
        
        f.write("DETAILED STATISTICS FOR ALL Z-PLANES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Z-Plane':<8} {'Mean':>8} {'Max':>5} {'StdDev':>8} {'Bright':>8} {'V.Bright':>9} {'P95':>8} {'P99':>8}\n")
        f.write("-" * 80 + "\n")
        
        for stat in z_stats:
            f.write(f"z{stat['z_index']:02d}      "
                   f"{stat['mean_intensity']:8.2f} "
                   f"{stat['max_intensity']:5d} "
                   f"{stat['std_intensity']:8.2f} "
                   f"{stat['bright_pixels']:8d} "
                   f"{stat['very_bright_pixels']:9d} "
                   f"{stat['percentile_95']:8.2f} "
                   f"{stat['percentile_99']:8.2f}\n")
        
        f.write("\n")
        f.write("NOTES:\n")
        f.write("-" * 80 + "\n")
        f.write("  - Bright pixels: count of pixels with intensity ≥ 200 (typical bright droplets)\n")
        f.write("  - Very bright: count of pixels with intensity ≥ 240 (very intense droplets)\n")
        f.write("  - P95/P99: 95th and 99th percentile intensity values\n")
        f.write("  - The plane with most bright pixels typically has the best droplet focus\n")
    
    print(f"  ✓ Saved report: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze z-stack intensity distribution to find peak droplet plane'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory containing z-stack images'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='intensity_analysis.txt',
        help='Output report filename (default: intensity_analysis.txt)'
    )
    
    args = parser.parse_args()
    
    analyze_zstack_intensity(args.input, args.output)


if __name__ == '__main__':
    main()

