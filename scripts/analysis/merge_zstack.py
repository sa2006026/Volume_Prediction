#!/usr/bin/env python3
"""
Z-Stack Image Merger
Merges z-stack microscopy images using industry-standard methods:
1. Maximum Intensity Projection (MIP) - Best for fluorescence imaging
2. Extended Depth of Field (EDF) - Focus stacking for sharp composite
3. Average Intensity Projection - Smoothed composite
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple
import argparse


class ZStackMerger:
    """Merge z-stack images using various industry-standard methods"""
    
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self.images = []
        self.image_paths = []
        
    def load_zstack_images(self, pattern: str = "*.jpg") -> int:
        """
        Load all z-stack images from directory
        
        Args:
            pattern: File pattern to match (default: *.jpg)
            
        Returns:
            Number of images loaded
        """
        # Get all matching files and sort them by z-index
        image_files = sorted(self.image_dir.glob(pattern))
        
        if not image_files:
            raise ValueError(f"No images found matching pattern {pattern} in {self.image_dir}")
        
        print(f"📂 Found {len(image_files)} z-stack images")
        
        # Load all images
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                self.images.append(img)
                self.image_paths.append(img_path)
                print(f"  ✓ Loaded: {img_path.name} ({img.shape[1]}x{img.shape[0]})")
            else:
                print(f"  ✗ Failed to load: {img_path.name}")
        
        if not self.images:
            raise ValueError("No images could be loaded successfully")
        
        print(f"✅ Successfully loaded {len(self.images)} images\n")
        return len(self.images)
    
    def maximum_intensity_projection(self) -> np.ndarray:
        """
        Maximum Intensity Projection (MIP)
        Takes the maximum pixel value across all z-planes
        
        This is the gold standard for fluorescence microscopy as it preserves
        the brightest features from all focal planes.
        
        Returns:
            Merged image using MIP
        """
        print("🔬 Creating Maximum Intensity Projection (MIP)...")
        
        if not self.images:
            raise ValueError("No images loaded")
        
        # Stack all images along a new axis
        stack = np.stack(self.images, axis=0)
        
        # Take maximum along z-axis (axis 0)
        mip = np.max(stack, axis=0)
        
        print(f"✅ MIP complete: {mip.shape[1]}x{mip.shape[0]}\n")
        return mip
    
    def average_intensity_projection(self) -> np.ndarray:
        """
        Average Intensity Projection
        Averages pixel values across all z-planes
        
        Useful for reducing noise and creating a smoothed composite.
        
        Returns:
            Merged image using average projection
        """
        print("📊 Creating Average Intensity Projection...")
        
        if not self.images:
            raise ValueError("No images loaded")
        
        # Stack all images along a new axis
        stack = np.stack(self.images, axis=0)
        
        # Take average along z-axis (axis 0)
        avg = np.mean(stack, axis=0).astype(np.uint8)
        
        print(f"✅ Average projection complete: {avg.shape[1]}x{avg.shape[0]}\n")
        return avg
    
    def extended_depth_of_field(self, kernel_size: int = 15) -> np.ndarray:
        """
        Extended Depth of Field (EDF) - Focus Stacking
        Creates a composite image where each pixel is taken from the sharpest focal plane
        
        This method is ideal for brightfield microscopy where different regions
        are in focus at different z-levels.
        
        Args:
            kernel_size: Size of the Laplacian kernel for focus measure (default: 15)
            
        Returns:
            Merged image using focus stacking
        """
        print("🎯 Creating Extended Depth of Field (Focus Stacking)...")
        print(f"   Using kernel size: {kernel_size}")
        
        if not self.images:
            raise ValueError("No images loaded")
        
        # Convert all images to grayscale for focus measure
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images]
        
        # Calculate focus measure (Laplacian variance) for each image
        focus_measures = []
        print("   Calculating focus measures for each z-plane...")
        for i, gray_img in enumerate(gray_images):
            # Use Laplacian to measure focus/sharpness
            laplacian = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=kernel_size)
            # Use absolute values to get focus measure map
            focus_map = np.abs(laplacian)
            focus_measures.append(focus_map)
            print(f"     z{i:02d}: mean focus = {focus_map.mean():.2f}")
        
        # Stack focus measures
        focus_stack = np.stack(focus_measures, axis=0)
        
        # Find the z-index with maximum focus for each pixel
        print("   Finding sharpest focal plane for each pixel...")
        best_focus_indices = np.argmax(focus_stack, axis=0)
        
        # Create output image by selecting pixels from the sharpest focal plane
        print("   Compositing final image...")
        height, width = self.images[0].shape[:2]
        channels = self.images[0].shape[2] if len(self.images[0].shape) > 2 else 1
        
        if channels == 3:
            edf_image = np.zeros((height, width, 3), dtype=np.uint8)
            # For each pixel, select from the image with best focus
            for z_idx in range(len(self.images)):
                mask = (best_focus_indices == z_idx)
                for c in range(3):
                    edf_image[:, :, c][mask] = self.images[z_idx][:, :, c][mask]
        else:
            edf_image = np.zeros((height, width), dtype=np.uint8)
            for z_idx in range(len(self.images)):
                mask = (best_focus_indices == z_idx)
                edf_image[mask] = self.images[z_idx][mask]
        
        # Optional: Apply slight smoothing at focus boundaries to reduce artifacts
        edf_image = cv2.bilateralFilter(edf_image, d=5, sigmaColor=10, sigmaSpace=10)
        
        print(f"✅ Extended Depth of Field complete: {edf_image.shape[1]}x{edf_image.shape[0]}\n")
        return edf_image
    
    def weighted_average_projection(self, use_variance: bool = True) -> np.ndarray:
        """
        Weighted Average Projection
        Weights each z-plane by its local sharpness/variance
        
        Args:
            use_variance: Use variance as weight (True) or Laplacian sharpness (False)
            
        Returns:
            Merged image using weighted average
        """
        print("⚖️  Creating Weighted Average Projection...")
        
        if not self.images:
            raise ValueError("No images loaded")
        
        # Calculate weights for each image based on local sharpness
        weights = []
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images]
        
        for i, gray_img in enumerate(gray_images):
            if use_variance:
                # Use local variance as weight
                mean = cv2.blur(gray_img, (9, 9))
                variance = cv2.blur(np.square(gray_img.astype(float) - mean), (9, 9))
                weight = variance
            else:
                # Use Laplacian sharpness as weight
                laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
                weight = np.abs(laplacian)
            
            weights.append(weight)
            print(f"   z{i:02d}: weight range = [{weight.min():.2f}, {weight.max():.2f}]")
        
        # Normalize weights
        weight_stack = np.stack(weights, axis=0)
        weight_sum = np.sum(weight_stack, axis=0)
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero
        normalized_weights = weight_stack / weight_sum[np.newaxis, :, :]
        
        # Apply weighted average
        result = np.zeros_like(self.images[0], dtype=float)
        for i, img in enumerate(self.images):
            # Expand weights to match image channels
            weight_3d = np.repeat(normalized_weights[i][:, :, np.newaxis], 3, axis=2)
            result += img.astype(float) * weight_3d
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        print(f"✅ Weighted average complete: {result.shape[1]}x{result.shape[0]}\n")
        return result
    
    def save_result(self, image: np.ndarray, output_path: str):
        """
        Save the merged image
        
        Args:
            image: Image to save
            output_path: Output file path
        """
        cv2.imwrite(output_path, image)
        print(f"💾 Saved: {output_path}")
        
        # Also save a smaller preview for quick viewing
        preview_path = output_path.replace('.jpg', '_preview.jpg').replace('.png', '_preview.png')
        preview = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        cv2.imwrite(preview_path, preview)
        print(f"💾 Saved preview: {preview_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge z-stack microscopy images using industry-standard methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  mip      : Maximum Intensity Projection (best for fluorescence)
  edf      : Extended Depth of Field / Focus Stacking (best for brightfield)
  average  : Average Intensity Projection (noise reduction)
  weighted : Weighted Average Projection (variance-weighted)
  all      : Generate all methods (default)

Examples:
  python merge_zstack.py --method mip
  python merge_zstack.py --method edf --kernel-size 11
  python merge_zstack.py --input custom_dir/ --output merged.jpg
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='images',
        help='Input directory containing z-stack images (default: images/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='merged_zstack.jpg',
        help='Output file path (default: merged_zstack.jpg)'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        default='all',
        choices=['mip', 'edf', 'average', 'weighted', 'all'],
        help='Merging method to use (default: all)'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.jpg',
        help='File pattern to match (default: *.jpg)'
    )
    
    parser.add_argument(
        '--kernel-size', '-k',
        type=int,
        default=15,
        help='Kernel size for EDF focus measure (default: 15)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Z-Stack Image Merger")
    print("=" * 60)
    print()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input
    
    # Initialize merger
    merger = ZStackMerger(input_dir)
    
    try:
        # Load z-stack images
        num_images = merger.load_zstack_images(args.pattern)
        
        # Determine output base path
        output_base = Path(args.output).stem
        output_ext = Path(args.output).suffix or '.jpg'
        output_dir = Path(args.output).parent or Path('.')
        
        # Process based on selected method
        if args.method == 'mip' or args.method == 'all':
            print("─" * 60)
            mip = merger.maximum_intensity_projection()
            output_path = output_dir / f"{output_base}_MIP{output_ext}"
            merger.save_result(mip, str(output_path))
        
        if args.method == 'edf' or args.method == 'all':
            print("─" * 60)
            edf = merger.extended_depth_of_field(kernel_size=args.kernel_size)
            output_path = output_dir / f"{output_base}_EDF{output_ext}"
            merger.save_result(edf, str(output_path))
        
        if args.method == 'average' or args.method == 'all':
            print("─" * 60)
            avg = merger.average_intensity_projection()
            output_path = output_dir / f"{output_base}_AVG{output_ext}"
            merger.save_result(avg, str(output_path))
        
        if args.method == 'weighted' or args.method == 'all':
            print("─" * 60)
            weighted = merger.weighted_average_projection()
            output_path = output_dir / f"{output_base}_WEIGHTED{output_ext}"
            merger.save_result(weighted, str(output_path))
        
        print()
        print("=" * 60)
        print(f"✅ Z-Stack merge complete! Processed {num_images} images")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

