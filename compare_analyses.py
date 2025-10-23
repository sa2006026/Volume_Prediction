#!/usr/bin/env python3
"""
Compare the two different droplet analysis approaches
"""

import sys
import os
from find_max_intensity_droplets import DropletTracker
from find_unique_max_intensity_droplets import UniqueDropletAnalyzer

def main():
    csv_directory = "/data3/megan_data/Jimmy/Volume_Prediction/csv/1"
    
    if len(sys.argv) > 1:
        csv_directory = sys.argv[1]
    
    if not os.path.isdir(csv_directory):
        print(f"❌ Error: Directory '{csv_directory}' does not exist!")
        return
    
    print("🔍 DROPLET ANALYSIS COMPARISON")
    print("="*60)
    print(f"📁 Directory: {csv_directory}")
    print()
    
    # Run tracking analysis (original approach)
    print("📊 APPROACH 1: Droplet Tracking Across Z-levels")
    print("-" * 50)
    tracker = DropletTracker(csv_directory, location_threshold=10.0)
    tracked_droplets = tracker.find_matching_droplets()
    
    if tracked_droplets:
        multi_z = [d for d in tracked_droplets if d['total_appearances'] >= 2]
        print(f"   Starting droplets (z00): {len(tracker.droplet_data['z00'])}")
        print(f"   Tracked across Z-levels: {len(tracked_droplets)}")
        print(f"   Multi-Z droplets: {len(multi_z)}")
        print(f"   Max intensity: {max(d['max_intensity'] for d in tracked_droplets):.2f}")
    
    print()
    
    # Run unique analysis (new approach)
    print("📊 APPROACH 2: Unique Location Analysis")
    print("-" * 50)
    analyzer = UniqueDropletAnalyzer(csv_directory, location_threshold=5.0)
    unique_droplets = analyzer.find_unique_max_intensity_droplets()
    
    if unique_droplets:
        stats = analyzer.get_statistics()
        print(f"   Total droplets found: {stats['total_droplets']}")
        print(f"   Unique locations: {stats['unique_locations']}")
        print(f"   Duplicates removed: {stats['duplicates_removed']}")
        print(f"   Reduction: {stats['duplicates_removed']/stats['total_droplets']*100:.1f}%")
        print(f"   Max intensity: {stats['max_intensity']:.2f}")
    
    print()
    print("🎯 KEY DIFFERENCES:")
    print("-" * 50)
    print("   Approach 1 (Tracking):")
    print("   • Follows specific droplets across ALL Z-levels")
    print("   • Shows intensity progression through Z-stack")
    print("   • Good for understanding droplet behavior")
    print()
    print("   Approach 2 (Unique Locations):")
    print("   • Finds ALL droplets from ALL Z-levels")
    print("   • Removes duplicates at same location")
    print("   • Keeps only highest intensity at each location")
    print("   • Good for comprehensive droplet inventory")
    print()
    
    if tracked_droplets and unique_droplets:
        print("📈 SUMMARY:")
        print("-" * 50)
        print(f"   Tracking approach: {len(tracked_droplets)} droplet tracks")
        print(f"   Unique approach: {len(unique_droplets)} unique locations")
        print(f"   Coverage difference: {len(unique_droplets) - len(tracked_droplets)} additional locations found")

if __name__ == "__main__":
    main()
