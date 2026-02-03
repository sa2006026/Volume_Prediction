#!/usr/bin/env python3
"""
Simple command-line interface for droplet analysis
Usage: python3 analyze_droplets.py [csv_directory]
"""

import sys
import os
from find_max_intensity_droplets import DropletTracker

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
    
    print("🔍 Droplet Maximum Intensity Analysis")
    print("="*50)
    print(f"📁 Directory: {csv_directory}")
    print(f"📏 Threshold: 10 pixels")
    print()
    
    # Run analysis
    tracker = DropletTracker(csv_directory, location_threshold=10.0)
    tracked_droplets = tracker.find_matching_droplets()
    
    if tracked_droplets:
        # Show summary
        multi_z_droplets = [d for d in tracked_droplets if d['total_appearances'] >= 2]
        
        print(f"📊 SUMMARY:")
        print(f"   Total droplet tracks: {len(tracked_droplets)}")
        print(f"   Multi-Z droplets: {len(multi_z_droplets)}")
        print(f"   Max intensity found: {max(d['max_intensity'] for d in tracked_droplets):.2f}")
        print()
        
        # Show top 10 results
        tracker.print_results(min_appearances=2, top_n=10)
        
        # Save results
        output_file = os.path.join(csv_directory, 'droplet_max_intensity_analysis.csv')
        tracker.save_results_to_csv(output_file, min_appearances=1)  # Save all tracks
        
        print(f"\n✅ Analysis complete!")
        print(f"💾 Detailed results saved to: {output_file}")
        
    else:
        print("❌ No droplets found to analyze!")

if __name__ == "__main__":
    main()
