#!/usr/bin/env python3
"""
Parameter Tuning Demo for Statistical Light Extraction

This script demonstrates how to tune parameters for optimal light pixel extraction.
"""

from src.core.statistical_light_extractor import StatisticalLightExtractor
import numpy as np

def demo_basic_parameter_tuning():
    """Demonstrate basic parameter tuning"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print("=== BASIC METHOD PARAMETER TUNING ===")
    print("Testing different std_multiplier values:")
    
    # Test different std_multiplier values
    multipliers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for multiplier in multipliers:
        mask, info = extractor.extract_statistical_basic(std_multiplier=multiplier)
        cleaned_mask = extractor.morphological_cleanup(mask)
        
        print(f"  std_multiplier={multiplier:3.1f}: "
              f"{info['light_percentage']:5.1f}% light pixels "
              f"(threshold: {info['threshold_value']:6.1f})")
    
    print("\nTesting with median base vs mean base:")
    for use_median in [False, True]:
        mask, info = extractor.extract_statistical_basic(
            std_multiplier=1.5, 
            use_median=use_median
        )
        base_type = "median" if use_median else "mean"
        print(f"  {base_type:6} base: {info['light_percentage']:5.1f}% light pixels "
              f"(base: {info['base_value']:6.1f}, threshold: {info['threshold_value']:6.1f})")

def demo_adaptive_parameter_tuning():
    """Demonstrate adaptive method parameter tuning"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print("\n=== ADAPTIVE METHOD PARAMETER TUNING ===")
    print("Testing different sensitivity values:")
    
    # Test different sensitivity values
    sensitivities = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for sensitivity in sensitivities:
        mask, info = extractor.extract_statistical_adaptive(sensitivity=sensitivity)
        
        print(f"  sensitivity={sensitivity:3.1f}: "
              f"{info['light_percentage']:5.1f}% light pixels "
              f"(threshold: {info['threshold_value']:6.1f})")
    
    print("\nTesting different percentile bases:")
    percentiles = [70, 75, 80, 85, 90, 95]
    
    for percentile in percentiles:
        mask, info = extractor.extract_statistical_adaptive(
            sensitivity=0.5,
            percentile_base=percentile
        )
        
        print(f"  percentile={percentile:2d}: "
              f"{info['light_percentage']:5.1f}% light pixels "
              f"(base: {info['base_value']:6.1f})")

def demo_zscore_parameter_tuning():
    """Demonstrate Z-score method parameter tuning"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print("\n=== Z-SCORE METHOD PARAMETER TUNING ===")
    
    # Test different z_threshold values
    z_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print("Standard Z-score method:")
    for z_threshold in z_thresholds:
        mask, info = extractor.extract_statistical_zscore(
            z_threshold=z_threshold, 
            robust=False
        )
        
        print(f"  z_threshold={z_threshold:3.1f}: "
              f"{info['light_percentage']:5.1f}% light pixels")
    
    print("\nRobust Z-score method (using median/MAD):")
    for z_threshold in z_thresholds:
        mask, info = extractor.extract_statistical_zscore(
            z_threshold=z_threshold, 
            robust=True
        )
        
        print(f"  z_threshold={z_threshold:3.1f}: "
              f"{info['light_percentage']:5.1f}% light pixels")

def demo_custom_parameter_tuning():
    """Demonstrate custom method parameter tuning"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print("\n=== CUSTOM METHOD PARAMETER TUNING ===")
    
    # Test different base methods
    base_methods = ["mean", "median", "q75", "q90", "q95"]
    
    print("Testing different base methods:")
    for base_method in base_methods:
        mask, info = extractor.extract_statistical_custom(
            base_method=base_method,
            multiplier=1.5
        )
        
        print(f"  {base_method:6}: "
              f"{info['light_percentage']:5.1f}% light pixels "
              f"(base: {info['base_value']:6.1f}, threshold: {info['threshold_value']:6.1f})")
    
    print("\nTesting different power values (std^power):")
    powers = [0.5, 1.0, 1.5, 2.0]
    
    for power in powers:
        mask, info = extractor.extract_statistical_custom(
            base_method="q75",
            multiplier=1.5,
            power=power
        )
        
        print(f"  power={power:3.1f}: "
              f"{info['light_percentage']:5.1f}% light pixels "
              f"(std^power: {info['std_powered']:6.1f})")

def demo_parameter_sweep():
    """Demonstrate automated parameter sweep"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print("\n=== AUTOMATED PARAMETER SWEEP ===")
    
    # Custom parameter ranges for sweep
    param_ranges = {
        'std_multiplier': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    }
    
    sweep_results = extractor.parameter_sweep('basic', param_ranges)
    
    print("Sweep results (sorted by light percentage):")
    results = sorted(sweep_results['sweep_results'], key=lambda x: x['light_percentage'])
    
    for result in results:
        params = result['parameters']
        print(f"  std_multiplier={params['std_multiplier']:3.1f}: "
              f"{result['light_percentage']:5.1f}% light pixels "
              f"(threshold: {result['threshold_value']:6.1f})")
    
    print(f"\nBest parameters for ~5% target: {sweep_results['best_result']['parameters']}")

def demo_visual_comparison():
    """Demonstrate visual comparison of different parameters"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print("\n=== VISUAL COMPARISON ===")
    
    # Define different parameter combinations to compare
    methods_and_params = [
        ("basic", {"std_multiplier": 1.0}),
        ("basic", {"std_multiplier": 1.5}),
        ("basic", {"std_multiplier": 2.0}),
        ("adaptive", {"sensitivity": 0.3}),
        ("adaptive", {"sensitivity": 0.7}),
        ("zscore", {"z_threshold": 1.5}),
        ("iqr", {"iqr_multiplier": 1.5}),
        ("custom", {"base_method": "q90", "multiplier": 1.0})
    ]
    
    # Save comparison visualization
    output_path = "/home/jimmy/code/2Dto3D_2/output/parameter_comparison.png"
    extractor.visualize_comparison(methods_and_params, output_path)
    
    print(f"Parameter comparison saved to: {output_path}")

def recommend_parameters(target_percentage: float = 5.0):
    """Recommend optimal parameters for a target light percentage"""
    
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    extractor = StatisticalLightExtractor(image_path)
    
    print(f"\n=== PARAMETER RECOMMENDATIONS FOR {target_percentage}% TARGET ===")
    
    # Test multiple methods to find best match
    candidates = []
    
    # Basic method candidates
    for multiplier in np.arange(0.5, 4.0, 0.25):
        mask, info = extractor.extract_statistical_basic(std_multiplier=multiplier)
        diff = abs(info['light_percentage'] - target_percentage)
        candidates.append(("basic", info['parameters'], info['light_percentage'], diff))
    
    # Adaptive method candidates
    for sensitivity in np.arange(0.1, 1.0, 0.1):
        mask, info = extractor.extract_statistical_adaptive(sensitivity=sensitivity)
        diff = abs(info['light_percentage'] - target_percentage)
        candidates.append(("adaptive", info['parameters'], info['light_percentage'], diff))
    
    # Z-score method candidates
    for z_thresh in np.arange(1.0, 3.5, 0.25):
        mask, info = extractor.extract_statistical_zscore(z_threshold=z_thresh)
        diff = abs(info['light_percentage'] - target_percentage)
        candidates.append(("zscore", info['parameters'], info['light_percentage'], diff))
    
    # Sort by difference from target
    candidates.sort(key=lambda x: x[3])
    
    print("Top 5 recommendations:")
    for i, (method, params, percentage, diff) in enumerate(candidates[:5]):
        print(f"  {i+1}. {method}: {params}")
        print(f"     Result: {percentage:.1f}% (diff: {diff:.1f}%)")
        print()

if __name__ == "__main__":
    demo_basic_parameter_tuning()
    demo_adaptive_parameter_tuning()
    demo_zscore_parameter_tuning()
    demo_custom_parameter_tuning()
    demo_parameter_sweep()
    demo_visual_comparison()
    recommend_parameters(5.0)  # Target 5% light pixels
