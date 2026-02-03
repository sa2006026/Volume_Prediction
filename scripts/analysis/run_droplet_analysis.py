#!/usr/bin/env python3
"""
Simple script to run droplet analysis on the CSV files
"""

import os
from find_max_intensity_droplets import DropletTracker

def main():
    # Set the CSV directory
    csv_directory = "/data3/megan_data/Jimmy/Volume_Prediction/csv/1"
    
    print("🔍 Starting Droplet Intensity Analysis")
    print(f"📁 CSV Directory: {csv_directory}")
    print(f"📏 Location Threshold: 10 pixels")
    print("="*60)
    
    # Create tracker
    tracker = DropletTracker(csv_directory, location_threshold=10.0)
    
    # Run analysis
    tracked_droplets = tracker.find_matching_droplets()
    
    if tracked_droplets:
        # Print results
        tracker.print_results(min_appearances=2, top_n=15)
        
        # Save results
        output_file = os.path.join(csv_directory, 'max_intensity_droplets.csv')
        tracker.save_results_to_csv(output_file, min_appearances=2)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Found {len(tracked_droplets)} total droplet tracks")
        print(f"💾 Results saved to: {output_file}")
        
    else:
        print("❌ No droplets found to track!")

if __name__ == "__main__":
    main()
