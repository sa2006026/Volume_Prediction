#!/usr/bin/env python3
"""
Simple interface for unique droplet analysis
Usage: python3 analyze_unique_droplets.py [csv_directory]
"""

import sys
import os
from find_unique_max_intensity_droplets import UniqueDropletAnalyzer

def main():
    # Default to the csv/1 directory if no argument provided
    if len(sys.argv) > 1:
        csv_directory = sys.argv[1]
    else:
        csv_directory = "/data3/megan_data/Jimmy/Volume_Prediction/csv/1"
    
    # Check if directory exists
    if not os.path.isdir(csv_directory):
        print(f"❌ Error: Directory '{csv_directory}' does not exist!")
        print(f"Usage: python3 {sys.argv[0]} [csv_directory]")
        return
    
    print("🔍 Unique Droplet Maximum Intensity Analysis")
    print("="*55)
    print(f"📁 Directory: {csv_directory}")
    print(f"📏 Location Threshold: 5 pixels")
    print(f"🎯 Goal: Find unique locations with maximum intensity")
    print()
    
    # Run analysis
    analyzer = UniqueDropletAnalyzer(csv_directory, location_threshold=5.0)
    unique_droplets = analyzer.find_unique_max_intensity_droplets()
    
    if unique_droplets:
        # Get statistics
        stats = analyzer.get_statistics()
        
        print(f"📊 SUMMARY:")
        print(f"   Total droplets found: {stats['total_droplets']}")
        print(f"   Unique locations: {stats['unique_locations']}")
        print(f"   Duplicates removed: {stats['duplicates_removed']}")
        print(f"   Reduction: {stats['duplicates_removed']/stats['total_droplets']*100:.1f}%")
        print(f"   Max intensity: {stats['max_intensity']:.2f}")
        print(f"   Locations with multiple droplets: {stats['locations_with_multiple_droplets']}")
        print()
        
        # Show top 15 results
        analyzer.print_results(top_n=15)
        
        # Save results
        output_file = os.path.join(csv_directory, 'unique_max_intensity_droplets.csv')
        analyzer.save_results_to_csv(output_file)
        
        print(f"\n✅ Analysis complete!")
        print(f"💾 Detailed results saved to: {output_file}")
        
    else:
        print("❌ No droplets found to analyze!")

if __name__ == "__main__":
    main()
