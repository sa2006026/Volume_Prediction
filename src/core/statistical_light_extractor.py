# python3 statistical_light_extractor.py data/input/overfocus.jpg --method basic --std-multiplier 1.0 

#!/usr/bin/env python3
"""
Statistical Light Intensity Pixel Extractor - Focused Implementation

This script provides a streamlined statistical approach for light intensity pixel extraction
with extensive parameter customization options.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, Optional, Dict, Any


class StatisticalLightExtractor:
    """
    Focused statistical light intensity pixel extractor with customizable parameters
    """
    
    def __init__(self, image_path: str):
        """
        Initialize the extractor with an image
        
        Args:
            image_path: Path to the input image
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.bgr_image = self.original_image.copy()
        self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray_image.shape
        
        # Cache statistics for efficiency
        self._stats = None
        
    def get_image_statistics(self) -> Dict[str, float]:
        """
        Get comprehensive image statistics
        
        Returns:
            Dictionary containing image statistics
        """
        if self._stats is None:
            self._stats = {
                'mean': float(np.mean(self.gray_image)),
                'std': float(np.std(self.gray_image)),
                'median': float(np.median(self.gray_image)),
                'min': float(np.min(self.gray_image)),
                'max': float(np.max(self.gray_image)),
                'q25': float(np.percentile(self.gray_image, 25)),
                'q75': float(np.percentile(self.gray_image, 75)),
                'q90': float(np.percentile(self.gray_image, 90)),
                'q95': float(np.percentile(self.gray_image, 95)),
                'q99': float(np.percentile(self.gray_image, 99)),
                'dynamic_range': float(np.max(self.gray_image) - np.min(self.gray_image)),
                'contrast_ratio': float(np.std(self.gray_image) / np.mean(self.gray_image)) if np.mean(self.gray_image) > 0 else 0.0
            }
        return self._stats
    
    def extract_statistical_basic(self, 
                                std_multiplier: float = 1.5,
                                use_median: bool = False,
                                min_threshold: int = 100,
                                max_threshold: int = 255) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Basic statistical extraction: threshold = mean + (std_multiplier * std)
        
        Args:
            std_multiplier: Multiplier for standard deviation (higher = more selective)
            use_median: Use median instead of mean as base
            min_threshold: Minimum threshold value (clamp lower bound)
            max_threshold: Maximum threshold value (clamp upper bound)
            
        Returns:
            Tuple of (binary mask, extraction info)
        """
        stats = self.get_image_statistics()
        
        base_value = stats['median'] if use_median else stats['mean']
        threshold = base_value + (std_multiplier * stats['std'])
        
        # Clamp threshold to valid range
        threshold = max(min_threshold, min(threshold, max_threshold))
        
        # Apply threshold
        binary_mask = (self.gray_image >= threshold).astype(np.uint8) * 255
        
        # Calculate extraction info
        light_pixels = np.sum(binary_mask > 0)
        percentage = (light_pixels / (self.width * self.height)) * 100
        
        info = {
            'method': 'statistical_basic',
            'parameters': {
                'std_multiplier': std_multiplier,
                'use_median': use_median,
                'min_threshold': min_threshold,
                'max_threshold': max_threshold
            },
            'threshold_value': float(threshold),
            'base_value': float(base_value),
            'std_value': float(stats['std']),
            'light_pixels': int(light_pixels),
            'light_percentage': float(percentage)
        }
        
        return binary_mask, info
    
    def extract_statistical_adaptive(self,
                                   sensitivity: float = 0.5,
                                   contrast_boost: float = 1.0,
                                   percentile_base: float = 75.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptive statistical extraction that adjusts based on image characteristics
        
        Args:
            sensitivity: Overall sensitivity (0.0 = very selective, 1.0 = very inclusive)
            contrast_boost: Boost factor for low-contrast images
            percentile_base: Percentile to use as base instead of mean (50-99)
            
        Returns:
            Tuple of (binary mask, extraction info)
        """
        stats = self.get_image_statistics()
        
        # Adaptive threshold calculation
        base_value = np.percentile(self.gray_image, percentile_base)
        
        # Adjust sensitivity based on contrast
        if stats['contrast_ratio'] < 0.3:  # Low contrast
            effective_sensitivity = sensitivity * contrast_boost
        else:
            effective_sensitivity = sensitivity
        
        # Calculate adaptive multiplier
        std_multiplier = 2.0 * (1.0 - effective_sensitivity)  # Inverse relationship
        
        threshold = base_value + (std_multiplier * stats['std'])
        threshold = max(50, min(threshold, 255))  # Reasonable bounds
        
        # Apply threshold
        binary_mask = (self.gray_image >= threshold).astype(np.uint8) * 255
        
        # Calculate extraction info
        light_pixels = np.sum(binary_mask > 0)
        percentage = (light_pixels / (self.width * self.height)) * 100
        
        info = {
            'method': 'statistical_adaptive',
            'parameters': {
                'sensitivity': sensitivity,
                'contrast_boost': contrast_boost,
                'percentile_base': percentile_base
            },
            'threshold_value': float(threshold),
            'base_value': float(base_value),
            'std_multiplier': float(std_multiplier),
            'effective_sensitivity': float(effective_sensitivity),
            'light_pixels': int(light_pixels),
            'light_percentage': float(percentage)
        }
        
        return binary_mask, info
    
    def extract_statistical_zscore(self,
                                  z_threshold: float = 1.5,
                                  robust: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Z-score based statistical extraction
        
        Args:
            z_threshold: Z-score threshold (standard deviations from mean)
            robust: Use median and MAD instead of mean and std for robustness
            
        Returns:
            Tuple of (binary mask, extraction info)
        """
        stats = self.get_image_statistics()
        
        if robust:
            # Use median and MAD (Median Absolute Deviation) for robustness
            center = stats['median']
            mad = np.median(np.abs(self.gray_image - center))
            scale = mad * 1.4826  # Scale factor to approximate standard deviation
        else:
            # Use mean and standard deviation
            center = stats['mean']
            scale = stats['std']
        
        # Calculate z-scores
        z_scores = np.abs(self.gray_image - center) / scale if scale > 0 else np.zeros_like(self.gray_image)
        
        # Apply threshold (pixels with high positive z-scores are bright)
        threshold_mask = (self.gray_image > center) & (z_scores >= z_threshold)
        binary_mask = threshold_mask.astype(np.uint8) * 255
        
        # Calculate extraction info
        light_pixels = np.sum(binary_mask > 0)
        percentage = (light_pixels / (self.width * self.height)) * 100
        
        info = {
            'method': 'statistical_zscore',
            'parameters': {
                'z_threshold': z_threshold,
                'robust': robust
            },
            'center_value': float(center),
            'scale_value': float(scale),
            'light_pixels': int(light_pixels),
            'light_percentage': float(percentage)
        }
        
        return binary_mask, info
    
    def extract_statistical_iqr(self,
                               iqr_multiplier: float = 1.5,
                               upper_only: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        IQR (Interquartile Range) based statistical extraction
        
        Args:
            iqr_multiplier: Multiplier for IQR to determine outliers
            upper_only: Only consider upper outliers (bright pixels)
            
        Returns:
            Tuple of (binary mask, extraction info)
        """
        stats = self.get_image_statistics()
        
        q25, q75 = stats['q25'], stats['q75']
        iqr = q75 - q25
        
        if upper_only:
            # Only upper outliers (bright pixels)
            upper_threshold = q75 + (iqr_multiplier * iqr)
            binary_mask = (self.gray_image >= upper_threshold).astype(np.uint8) * 255
            threshold_value = upper_threshold
        else:
            # Both upper and lower outliers
            lower_threshold = q25 - (iqr_multiplier * iqr)
            upper_threshold = q75 + (iqr_multiplier * iqr)
            outlier_mask = (self.gray_image <= lower_threshold) | (self.gray_image >= upper_threshold)
            binary_mask = outlier_mask.astype(np.uint8) * 255
            threshold_value = (lower_threshold, upper_threshold)
        
        # Calculate extraction info
        light_pixels = np.sum(binary_mask > 0)
        percentage = (light_pixels / (self.width * self.height)) * 100
        
        info = {
            'method': 'statistical_iqr',
            'parameters': {
                'iqr_multiplier': iqr_multiplier,
                'upper_only': upper_only
            },
            'q25': float(q25),
            'q75': float(q75),
            'iqr': float(iqr),
            'threshold_value': threshold_value if isinstance(threshold_value, tuple) else float(threshold_value),
            'light_pixels': int(light_pixels),
            'light_percentage': float(percentage)
        }
        
        return binary_mask, info
    
    def extract_statistical_custom(self,
                                 base_method: str = "mean",
                                 multiplier: float = 1.5,
                                 offset: float = 0.0,
                                 power: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Custom statistical extraction with flexible formula
        Formula: threshold = (base_value + offset) + multiplier * (std ^ power)
        
        Args:
            base_method: Base value method ("mean", "median", "q75", "q90", "q95")
            multiplier: Multiplier for the spread measure
            offset: Offset to add to base value
            power: Power to raise std to (1.0 = linear, 0.5 = sqrt, 2.0 = quadratic)
            
        Returns:
            Tuple of (binary mask, extraction info)
        """
        stats = self.get_image_statistics()
        
        # Get base value
        base_value = stats.get(base_method, stats['mean'])
        
        # Calculate threshold with custom formula
        std_powered = stats['std'] ** power
        threshold = (base_value + offset) + (multiplier * std_powered)
        threshold = max(0, min(threshold, 255))  # Clamp to valid range
        
        # Apply threshold
        binary_mask = (self.gray_image >= threshold).astype(np.uint8) * 255
        
        # Calculate extraction info
        light_pixels = np.sum(binary_mask > 0)
        percentage = (light_pixels / (self.width * self.height)) * 100
        
        info = {
            'method': 'statistical_custom',
            'parameters': {
                'base_method': base_method,
                'multiplier': multiplier,
                'offset': offset,
                'power': power
            },
            'base_value': float(base_value),
            'std_powered': float(std_powered),
            'threshold_value': float(threshold),
            'light_pixels': int(light_pixels),
            'light_percentage': float(percentage)
        }
        
        return binary_mask, info
    
    def morphological_cleanup(self, binary_mask: np.ndarray, 
                            kernel_size: int = 3, 
                            operations: str = "close") -> np.ndarray:
        """Apply morphological operations to clean up the binary mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operations == "open":
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        elif operations == "close":
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        elif operations == "open_close":
            opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        else:
            cleaned = binary_mask
            
        return cleaned
    
    def parameter_sweep(self, method: str = "basic", param_ranges: Dict = None) -> Dict[str, Any]:
        """
        Perform parameter sweep to find optimal settings
        
        Args:
            method: Method to sweep ("basic", "adaptive", "zscore", "iqr", "custom")
            param_ranges: Dictionary of parameter ranges to test
            
        Returns:
            Dictionary with sweep results
        """
        if param_ranges is None:
            if method == "basic":
                param_ranges = {'std_multiplier': [0.5, 1.0, 1.5, 2.0, 2.5]}
            elif method == "adaptive":
                param_ranges = {'sensitivity': [0.2, 0.4, 0.6, 0.8]}
            elif method == "zscore":
                param_ranges = {'z_threshold': [1.0, 1.5, 2.0, 2.5]}
            elif method == "iqr":
                param_ranges = {'iqr_multiplier': [1.0, 1.5, 2.0, 2.5]}
            elif method == "custom":
                param_ranges = {'multiplier': [0.5, 1.0, 1.5, 2.0]}
        
        results = []
        
        for param_name, param_values in param_ranges.items():
            for param_value in param_values:
                kwargs = {param_name: param_value}
                
                try:
                    if method == "basic":
                        mask, info = self.extract_statistical_basic(**kwargs)
                    elif method == "adaptive":
                        mask, info = self.extract_statistical_adaptive(**kwargs)
                    elif method == "zscore":
                        mask, info = self.extract_statistical_zscore(**kwargs)
                    elif method == "iqr":
                        mask, info = self.extract_statistical_iqr(**kwargs)
                    elif method == "custom":
                        mask, info = self.extract_statistical_custom(**kwargs)
                    
                    results.append({
                        'parameters': info['parameters'],
                        'light_percentage': info['light_percentage'],
                        'light_pixels': info['light_pixels'],
                        'threshold_value': info.get('threshold_value', 'N/A')
                    })
                    
                except Exception as e:
                    print(f"Error with {param_name}={param_value}: {e}")
        
        return {
            'method': method,
            'sweep_results': results,
            'best_result': min(results, key=lambda x: abs(x['light_percentage'] - 5.0)) if results else None  # Target ~5% light pixels
        }
    
    def visualize_comparison(self, methods_and_params: list, save_path: Optional[str] = None):
        """
        Visualize comparison of different parameter settings
        
        Args:
            methods_and_params: List of (method_name, parameters) tuples
            save_path: Optional path to save visualization
        """
        n_methods = len(methods_and_params)
        n_cols = min(4, n_methods + 1)  # +1 for original
        n_rows = (n_methods + 1 + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Show original image
        axes[0].imshow(self.rgb_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Show results for each method/parameter combination
        for i, (method_name, params) in enumerate(methods_and_params, 1):
            if i < len(axes):
                # Extract using specified method and parameters
                if method_name == "basic":
                    mask, info = self.extract_statistical_basic(**params)
                elif method_name == "adaptive":
                    mask, info = self.extract_statistical_adaptive(**params)
                elif method_name == "zscore":
                    mask, info = self.extract_statistical_zscore(**params)
                elif method_name == "iqr":
                    mask, info = self.extract_statistical_iqr(**params)
                elif method_name == "custom":
                    mask, info = self.extract_statistical_custom(**params)
                
                axes[i].imshow(mask, cmap='gray')
                title = f"{method_name}\n{info['light_percentage']:.1f}%"
                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(methods_and_params) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        plt.show()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Statistical Light Intensity Pixel Extractor')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--method', choices=['basic', 'adaptive', 'zscore', 'iqr', 'custom', 'sweep'], 
                       default='basic', help='Statistical method to use')
    parser.add_argument('--std-multiplier', type=float, default=1.5, 
                       help='Standard deviation multiplier for basic method')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                       help='Sensitivity for adaptive method (0.0-1.0)')
    parser.add_argument('--z-threshold', type=float, default=1.5,
                       help='Z-score threshold for zscore method')
    parser.add_argument('--iqr-multiplier', type=float, default=1.5,
                       help='IQR multiplier for iqr method')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--filename', help='Custom output filename (without extension)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    try:
        extractor = StatisticalLightExtractor(args.image_path)
        print(f"Loaded image: {args.image_path} ({extractor.width}x{extractor.height})")
        
        # Show image statistics
        stats = extractor.get_image_statistics()
        print(f"\nImage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        
        if args.method == 'sweep':
            # Parameter sweep
            print(f"\nPerforming parameter sweep...")
            sweep_results = extractor.parameter_sweep('basic')
            print(f"Best parameters: {sweep_results['best_result']['parameters']}")
            print(f"Light percentage: {sweep_results['best_result']['light_percentage']:.1f}%")
        else:
            # Single extraction
            if args.method == 'basic':
                mask, info = extractor.extract_statistical_basic(std_multiplier=args.std_multiplier)
            elif args.method == 'adaptive':
                mask, info = extractor.extract_statistical_adaptive(sensitivity=args.sensitivity)
            elif args.method == 'zscore':
                mask, info = extractor.extract_statistical_zscore(z_threshold=args.z_threshold)
            elif args.method == 'iqr':
                mask, info = extractor.extract_statistical_iqr(iqr_multiplier=args.iqr_multiplier)
            elif args.method == 'custom':
                mask, info = extractor.extract_statistical_custom()
            
            # Clean up mask
            cleaned_mask = extractor.morphological_cleanup(mask)
            
            # Print results
            print(f"\nExtraction Results:")
            print(f"  Method: {info['method']}")
            print(f"  Parameters: {info['parameters']}")
            print(f"  Threshold: {info.get('threshold_value', 'N/A')}")
            print(f"  Light pixels: {info['light_pixels']:,} ({info['light_percentage']:.1f}%)")
            
            # Save results
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                
                # Determine filename
                if args.filename:
                    # Use custom filename
                    filename = f"{args.filename}.png"
                else:
                    # Use default filename pattern
                    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
                    filename = f"{base_name}_statistical_{args.method}.png"
                
                output_path = os.path.join(args.output, filename)
                cv2.imwrite(output_path, cleaned_mask)
                print(f"Saved mask to: {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
