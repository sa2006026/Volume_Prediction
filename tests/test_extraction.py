#!/usr/bin/env python3
"""
Test script for adaptive light intensity pixel extraction
"""

import os
import sys
from src.core.adaptive_light_extraction import AdaptiveLightExtractor

def main():
    """Test the adaptive light extraction on the overfocus image"""
    
    # Path to the test image
    image_path = "/home/jimmy/code/2Dto3D_2/data/input/overfocus.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return 1
    
    try:
        # Initialize the extractor
        print("Initializing adaptive light extractor...")
        extractor = AdaptiveLightExtractor(image_path)
        
        # Analyze image statistics
        print("\nAnalyzing image statistics...")
        stats = extractor.analyze_image_statistics()
        print(f"Image dimensions: {extractor.width} x {extractor.height}")
        print(f"Mean intensity: {stats['mean_intensity']:.2f}")
        print(f"Standard deviation: {stats['std_intensity']:.2f}")
        print(f"Contrast ratio: {stats['contrast']:.2f}")
        print(f"Dynamic range: {stats['dynamic_range']}")
        
        # Auto-select best method
        recommended_method = extractor.auto_select_method()
        print(f"\nRecommended method: {recommended_method}")
        
        # Extract using all methods for comparison
        print("\nExtracting light pixels using all methods...")
        results = extractor.extract_all_methods()
        
        # Generate comprehensive report
        print("\nGenerating extraction report...")
        report = extractor.generate_report(results)
        
        # Display results
        print(f"\n{'='*60}")
        print("ADAPTIVE LIGHT EXTRACTION RESULTS")
        print(f"{'='*60}")
        
        for method, result in report['method_results'].items():
            print(f"\n{method.upper()} Method:")
            print(f"  Light pixels: {result['light_pixels_count']:,} ({result['light_pixels_percentage']:.1f}%)")
            print(f"  Connected components: {result['connected_components']}")
            if result['largest_component_area'] > 0:
                print(f"  Largest component: {result['largest_component_area']:,} pixels")
        
        # Create output directory
        output_dir = "/home/jimmy/code/2Dto3D_2/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all masks
        print(f"\nSaving results to {output_dir}...")
        import cv2
        for method_name, mask in results.items():
            output_path = os.path.join(output_dir, f"overfocus_{method_name}_mask.png")
            cv2.imwrite(output_path, mask)
            print(f"  Saved {method_name} mask: {output_path}")
        
        # Create visualization
        print("\nGenerating visualization...")
        vis_path = os.path.join(output_dir, "overfocus_comparison.png")
        extractor.visualize_results(results, vis_path)
        
        print(f"\nExtraction complete! Check the {output_dir} directory for results.")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
