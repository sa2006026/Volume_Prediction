#!/usr/bin/env python3
"""
Find Maximum Intensity Droplets Across Z-levels

This script analyzes CSV files from different Z-levels to find droplets that appear
at similar x,y locations (within 10 pixels) and tracks their maximum intensity values.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
import argparse


class DropletTracker:
    """Track droplets across multiple Z-levels and find maximum intensities"""
    
    def __init__(self, csv_directory: str, location_threshold: float = 10.0):
        """
        Initialize the droplet tracker
        
        Args:
            csv_directory: Directory containing CSV files
            location_threshold: Maximum distance in pixels to consider droplets as "same location"
        """
        self.csv_directory = csv_directory
        self.location_threshold = location_threshold
        self.droplet_data = {}  # Will store all droplet data by z-level
        self.tracked_droplets = []  # Will store tracked droplets across z-levels
        
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the directory (only z-level files)"""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        
        # Filter to only z-level files (z00.csv, z01.csv, etc.)
        z_files = [f for f in csv_files if os.path.basename(f).startswith('z') and 
                   os.path.basename(f)[1:3].isdigit()]
        z_files.sort()  # Sort to ensure consistent z-level ordering
        
        print(f"Found {len(z_files)} z-level CSV files:")
        for file in z_files:
            print(f"  - {os.path.basename(file)}")
        
        data = {}
        for csv_file in z_files:
            z_level = os.path.splitext(os.path.basename(csv_file))[0]
            try:
                df = pd.read_csv(csv_file)
                # Verify required columns exist
                required_cols = ['Center_X', 'Center_Y', 'Mean_Intensity', 'Diameter', 'Area', 'Circularity']
                if all(col in df.columns for col in required_cols):
                    data[z_level] = df
                    print(f"Loaded {z_level}: {len(df)} droplets")
                else:
                    print(f"Skipping {z_level}: Missing required columns")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return data
    
    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def find_matching_droplets(self) -> List[Dict]:
        """
        Find droplets that appear at similar locations across different z-levels
        
        Returns:
            List of dictionaries containing tracked droplet information
        """
        self.droplet_data = self.load_csv_files()
        
        if not self.droplet_data:
            print("No CSV data loaded!")
            return []
        
        # Get all z-levels sorted
        z_levels = sorted(self.droplet_data.keys())
        print(f"\nAnalyzing {len(z_levels)} z-levels: {z_levels}")
        
        tracked_droplets = []
        
        # Start with droplets from the first z-level
        first_z = z_levels[0]
        first_df = self.droplet_data[first_z]
        
        for idx, row in first_df.iterrows():
            droplet_track = {
                'track_id': len(tracked_droplets),
                'base_x': row['Center_X'],
                'base_y': row['Center_Y'],
                'z_levels': {},
                'max_intensity': row['Mean_Intensity'],
                'max_intensity_z': first_z,
                'total_appearances': 1
            }
            
            # Store data for first z-level
            droplet_track['z_levels'][first_z] = {
                'mask_id': row['Mask_ID'],
                'center_x': row['Center_X'],
                'center_y': row['Center_Y'],
                'diameter': row['Diameter'],
                'mean_intensity': row['Mean_Intensity'],
                'area': row['Area'],
                'circularity': row['Circularity']
            }
            
            # Look for matching droplets in other z-levels
            for z_level in z_levels[1:]:
                df = self.droplet_data[z_level]
                
                # Find closest droplet within threshold
                min_distance = float('inf')
                best_match = None
                
                for _, candidate_row in df.iterrows():
                    distance = self.calculate_distance(
                        droplet_track['base_x'], droplet_track['base_y'],
                        candidate_row['Center_X'], candidate_row['Center_Y']
                    )
                    
                    if distance <= self.location_threshold and distance < min_distance:
                        min_distance = distance
                        best_match = candidate_row
                
                # If we found a match, add it to the track
                if best_match is not None:
                    droplet_track['z_levels'][z_level] = {
                        'mask_id': best_match['Mask_ID'],
                        'center_x': best_match['Center_X'],
                        'center_y': best_match['Center_Y'],
                        'diameter': best_match['Diameter'],
                        'mean_intensity': best_match['Mean_Intensity'],
                        'area': best_match['Area'],
                        'circularity': best_match['Circularity'],
                        'distance_from_base': min_distance
                    }
                    
                    droplet_track['total_appearances'] += 1
                    
                    # Update maximum intensity
                    if best_match['Mean_Intensity'] > droplet_track['max_intensity']:
                        droplet_track['max_intensity'] = best_match['Mean_Intensity']
                        droplet_track['max_intensity_z'] = z_level
            
            tracked_droplets.append(droplet_track)
        
        # Sort by maximum intensity (descending)
        tracked_droplets.sort(key=lambda x: x['max_intensity'], reverse=True)
        
        self.tracked_droplets = tracked_droplets
        return tracked_droplets
    
    def print_results(self, min_appearances: int = 2, top_n: int = 20):
        """
        Print results of droplet tracking
        
        Args:
            min_appearances: Minimum number of z-levels a droplet must appear in
            top_n: Number of top droplets to display
        """
        if not self.tracked_droplets:
            print("No tracked droplets found!")
            return
        
        # Filter droplets that appear in at least min_appearances z-levels
        filtered_droplets = [d for d in self.tracked_droplets if d['total_appearances'] >= min_appearances]
        
        print(f"\n{'='*80}")
        print(f"DROPLET TRACKING RESULTS")
        print(f"{'='*80}")
        print(f"Total droplets tracked: {len(self.tracked_droplets)}")
        print(f"Droplets appearing in ≥{min_appearances} z-levels: {len(filtered_droplets)}")
        print(f"Location threshold: {self.location_threshold} pixels")
        print(f"\nTop {min(top_n, len(filtered_droplets))} droplets by maximum intensity:")
        print(f"{'-'*80}")
        
        for i, droplet in enumerate(filtered_droplets[:top_n]):
            print(f"\nRank #{i+1} - Track ID: {droplet['track_id']}")
            print(f"  Base Location: ({droplet['base_x']:.1f}, {droplet['base_y']:.1f})")
            print(f"  Max Intensity: {droplet['max_intensity']:.2f} (at {droplet['max_intensity_z']})")
            print(f"  Appearances: {droplet['total_appearances']} z-levels")
            print(f"  Z-levels: {', '.join(sorted(droplet['z_levels'].keys()))}")
            
            # Show intensity progression
            intensities = []
            for z in sorted(droplet['z_levels'].keys()):
                intensity = droplet['z_levels'][z]['mean_intensity']
                intensities.append(f"{z}:{intensity:.1f}")
            print(f"  Intensity progression: {' → '.join(intensities)}")
    
    def save_results_to_csv(self, output_file: str, min_appearances: int = 2):
        """
        Save tracking results to CSV file
        
        Args:
            output_file: Path to output CSV file
            min_appearances: Minimum number of z-levels a droplet must appear in
        """
        if not self.tracked_droplets:
            print("No tracked droplets to save!")
            return
        
        # Filter droplets
        filtered_droplets = [d for d in self.tracked_droplets if d['total_appearances'] >= min_appearances]
        
        # Prepare data for CSV
        csv_data = []
        for droplet in filtered_droplets:
            row = {
                'Track_ID': droplet['track_id'],
                'Base_X': droplet['base_x'],
                'Base_Y': droplet['base_y'],
                'Max_Intensity': droplet['max_intensity'],
                'Max_Intensity_Z_Level': droplet['max_intensity_z'],
                'Total_Appearances': droplet['total_appearances'],
                'Z_Levels': ','.join(sorted(droplet['z_levels'].keys()))
            }
            
            # Add intensity values for each z-level
            for z_level in sorted(droplet['z_levels'].keys()):
                z_data = droplet['z_levels'][z_level]
                row[f'{z_level}_Intensity'] = z_data['mean_intensity']
                row[f'{z_level}_X'] = z_data['center_x']
                row[f'{z_level}_Y'] = z_data['center_y']
                row[f'{z_level}_Diameter'] = z_data['diameter']
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Saved {len(csv_data)} tracked droplets")


def main():
    """Main function to run the droplet tracking analysis"""
    parser = argparse.ArgumentParser(description='Find maximum intensity droplets across Z-levels')
    parser.add_argument('csv_directory', help='Directory containing CSV files')
    parser.add_argument('--threshold', '-t', type=float, default=10.0,
                       help='Location threshold in pixels (default: 10.0)')
    parser.add_argument('--min-appearances', '-m', type=int, default=2,
                       help='Minimum appearances across z-levels (default: 2)')
    parser.add_argument('--top-n', '-n', type=int, default=20,
                       help='Number of top droplets to display (default: 20)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.csv_directory):
        print(f"Error: Directory '{args.csv_directory}' does not exist!")
        return
    
    # Create tracker and run analysis
    tracker = DropletTracker(args.csv_directory, args.threshold)
    tracked_droplets = tracker.find_matching_droplets()
    
    if tracked_droplets:
        # Print results
        tracker.print_results(args.min_appearances, args.top_n)
        
        # Save to CSV if output file specified
        if args.output:
            tracker.save_results_to_csv(args.output, args.min_appearances)
        else:
            # Auto-generate output filename
            output_file = os.path.join(args.csv_directory, 'droplet_tracking_results.csv')
            tracker.save_results_to_csv(output_file, args.min_appearances)
    else:
        print("No droplets found to track!")


if __name__ == "__main__":
    main()
