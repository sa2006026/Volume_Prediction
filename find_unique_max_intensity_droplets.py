#!/usr/bin/env python3
"""
Find Unique Maximum Intensity Droplets Across Z-levels

This script analyzes CSV files from different Z-levels to find all unique droplets.
For droplets at the same location (within 5 pixels), only keeps the one with highest intensity.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
import argparse


class UniqueDropletAnalyzer:
    """Analyze all droplets and keep only unique locations with maximum intensity"""
    
    def __init__(self, csv_directory: str, location_threshold: float = 5.0):
        """
        Initialize the analyzer
        
        Args:
            csv_directory: Directory containing CSV files
            location_threshold: Maximum distance in pixels to consider droplets as "same location"
        """
        self.csv_directory = csv_directory
        self.location_threshold = location_threshold
        self.all_droplets = []  # Will store all droplets from all z-levels
        self.unique_droplets = []  # Will store unique droplets with max intensity
        
    def load_all_droplets(self) -> List[Dict]:
        """Load all droplets from all CSV files"""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        
        # Filter to only z-level files (z00.csv, z01.csv, etc.)
        z_files = [f for f in csv_files if os.path.basename(f).startswith('z') and 
                   os.path.basename(f)[1:3].isdigit()]
        z_files.sort()  # Sort to ensure consistent z-level ordering
        
        print(f"Found {len(z_files)} z-level CSV files:")
        for file in z_files:
            print(f"  - {os.path.basename(file)}")
        
        all_droplets = []
        
        for csv_file in z_files:
            z_level = os.path.splitext(os.path.basename(csv_file))[0]
            try:
                df = pd.read_csv(csv_file)
                # Verify required columns exist
                required_cols = ['Center_X', 'Center_Y', 'Mean_Intensity', 'Diameter', 'Area', 'Circularity']
                if all(col in df.columns for col in required_cols):
                    # Add each droplet with z-level information
                    for idx, row in df.iterrows():
                        droplet = {
                            'z_level': z_level,
                            'mask_id': row['Mask_ID'],
                            'center_x': row['Center_X'],
                            'center_y': row['Center_Y'],
                            'diameter': row['Diameter'],
                            'mean_intensity': row['Mean_Intensity'],
                            'area': row['Area'],
                            'circularity': row['Circularity']
                        }
                        all_droplets.append(droplet)
                    
                    print(f"Loaded {z_level}: {len(df)} droplets")
                else:
                    print(f"Skipping {z_level}: Missing required columns")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        self.all_droplets = all_droplets
        return all_droplets
    
    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def find_unique_max_intensity_droplets(self) -> List[Dict]:
        """
        Find unique droplets by location and keep only the one with maximum intensity
        
        Returns:
            List of unique droplets with maximum intensity at each location
        """
        if not self.all_droplets:
            self.load_all_droplets()
        
        print(f"\nAnalyzing {len(self.all_droplets)} total droplets across all z-levels...")
        print(f"Location threshold: {self.location_threshold} pixels")
        
        unique_droplets = []
        processed_indices = set()
        
        for i, droplet in enumerate(self.all_droplets):
            if i in processed_indices:
                continue
            
            # Find all droplets at the same location (within threshold)
            same_location_droplets = [droplet]
            same_location_indices = [i]
            
            for j, other_droplet in enumerate(self.all_droplets[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                distance = self.calculate_distance(
                    droplet['center_x'], droplet['center_y'],
                    other_droplet['center_x'], other_droplet['center_y']
                )
                
                if distance <= self.location_threshold:
                    same_location_droplets.append(other_droplet)
                    same_location_indices.append(j)
            
            # Find the droplet with maximum intensity at this location
            max_intensity_droplet = max(same_location_droplets, key=lambda d: d['mean_intensity'])
            
            # Add information about how many droplets were at this location
            max_intensity_droplet['droplets_at_location'] = len(same_location_droplets)
            max_intensity_droplet['z_levels_at_location'] = [d['z_level'] for d in same_location_droplets]
            max_intensity_droplet['all_intensities'] = [d['mean_intensity'] for d in same_location_droplets]
            
            unique_droplets.append(max_intensity_droplet)
            
            # Mark all these indices as processed
            processed_indices.update(same_location_indices)
        
        # Sort by intensity (descending)
        unique_droplets.sort(key=lambda x: x['mean_intensity'], reverse=True)
        
        self.unique_droplets = unique_droplets
        return unique_droplets
    
    def print_results(self, top_n: int = 20):
        """
        Print results of unique droplet analysis
        
        Args:
            top_n: Number of top droplets to display
        """
        if not self.unique_droplets:
            print("No unique droplets found!")
            return
        
        print(f"\n{'='*80}")
        print(f"UNIQUE DROPLET ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(f"Total droplets across all z-levels: {len(self.all_droplets)}")
        print(f"Unique locations found: {len(self.unique_droplets)}")
        print(f"Location threshold: {self.location_threshold} pixels")
        print(f"Reduction: {len(self.all_droplets) - len(self.unique_droplets)} duplicate locations removed")
        print(f"\nTop {min(top_n, len(self.unique_droplets))} droplets by maximum intensity:")
        print(f"{'-'*80}")
        
        for i, droplet in enumerate(self.unique_droplets[:top_n]):
            print(f"\nRank #{i+1}")
            print(f"  Location: ({droplet['center_x']:.1f}, {droplet['center_y']:.1f})")
            print(f"  Max Intensity: {droplet['mean_intensity']:.2f} (from {droplet['z_level']})")
            print(f"  Diameter: {droplet['diameter']:.2f}")
            print(f"  Area: {droplet['area']:.2f}")
            print(f"  Circularity: {droplet['circularity']:.3f}")
            print(f"  Droplets at this location: {droplet['droplets_at_location']}")
            print(f"  Z-levels: {', '.join(sorted(droplet['z_levels_at_location']))}")
            
            if droplet['droplets_at_location'] > 1:
                intensities = [f"{z}:{intensity:.1f}" for z, intensity in 
                             zip(droplet['z_levels_at_location'], droplet['all_intensities'])]
                print(f"  All intensities: {', '.join(sorted(intensities))}")
    
    def save_results_to_csv(self, output_file: str):
        """
        Save unique droplet results to CSV file
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.unique_droplets:
            print("No unique droplets to save!")
            return
        
        # Prepare data for CSV
        csv_data = []
        for i, droplet in enumerate(self.unique_droplets):
            row = {
                'Rank': i + 1,
                'Center_X': droplet['center_x'],
                'Center_Y': droplet['center_y'],
                'Max_Intensity': droplet['mean_intensity'],
                'Source_Z_Level': droplet['z_level'],
                'Diameter': droplet['diameter'],
                'Area': droplet['area'],
                'Circularity': droplet['circularity'],
                'Droplets_At_Location': droplet['droplets_at_location'],
                'Z_Levels_At_Location': ','.join(sorted(droplet['z_levels_at_location'])),
                'All_Intensities': ','.join([f"{intensity:.2f}" for intensity in droplet['all_intensities']])
            }
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Saved {len(csv_data)} unique droplets")
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        if not self.unique_droplets:
            return {}
        
        intensities = [d['mean_intensity'] for d in self.unique_droplets]
        droplet_counts = [d['droplets_at_location'] for d in self.unique_droplets]
        
        stats = {
            'total_droplets': len(self.all_droplets),
            'unique_locations': len(self.unique_droplets),
            'duplicates_removed': len(self.all_droplets) - len(self.unique_droplets),
            'max_intensity': max(intensities),
            'min_intensity': min(intensities),
            'mean_intensity': np.mean(intensities),
            'std_intensity': np.std(intensities),
            'max_droplets_at_location': max(droplet_counts),
            'locations_with_multiple_droplets': sum(1 for count in droplet_counts if count > 1)
        }
        
        return stats


def main():
    """Main function to run the unique droplet analysis"""
    parser = argparse.ArgumentParser(description='Find unique maximum intensity droplets across Z-levels')
    parser.add_argument('csv_directory', help='Directory containing CSV files')
    parser.add_argument('--threshold', '-t', type=float, default=5.0,
                       help='Location threshold in pixels (default: 5.0)')
    parser.add_argument('--top-n', '-n', type=int, default=20,
                       help='Number of top droplets to display (default: 20)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.csv_directory):
        print(f"Error: Directory '{args.csv_directory}' does not exist!")
        return
    
    # Create analyzer and run analysis
    analyzer = UniqueDropletAnalyzer(args.csv_directory, args.threshold)
    unique_droplets = analyzer.find_unique_max_intensity_droplets()
    
    if unique_droplets:
        # Print results
        analyzer.print_results(args.top_n)
        
        # Print statistics
        stats = analyzer.get_statistics()
        print(f"\n{'='*80}")
        print(f"ANALYSIS STATISTICS")
        print(f"{'='*80}")
        print(f"Total droplets: {stats['total_droplets']}")
        print(f"Unique locations: {stats['unique_locations']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")
        print(f"Intensity range: {stats['min_intensity']:.2f} - {stats['max_intensity']:.2f}")
        print(f"Mean intensity: {stats['mean_intensity']:.2f} ± {stats['std_intensity']:.2f}")
        print(f"Max droplets at one location: {stats['max_droplets_at_location']}")
        print(f"Locations with multiple droplets: {stats['locations_with_multiple_droplets']}")
        
        # Save to CSV if output file specified
        if args.output:
            analyzer.save_results_to_csv(args.output)
        else:
            # Auto-generate output filename
            output_file = os.path.join(args.csv_directory, 'unique_max_intensity_droplets.csv')
            analyzer.save_results_to_csv(output_file)
    else:
        print("No droplets found to analyze!")


if __name__ == "__main__":
    main()
