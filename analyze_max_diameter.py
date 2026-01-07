#!/usr/bin/env python3
"""
Analyze droplet data from multiple z-slices to find the maximum diameter for each unique droplet.
A droplet is considered unique if its (x,y) location is within a 10-pixel error margin.
"""

import pandas as pd
import glob
import os
from process_droplets import process_all_slides

def main():
    # Directory containing the CSV files
    base_dir = '/home/mib/Jimmy/Volume_Prediction/csv/2025-10-09_ddPCR_pipette_1Series127'
    
    # Get all z*.csv files (z00.csv, z01.csv, etc.)
    all_files = sorted(glob.glob(os.path.join(base_dir, 'z*.csv')))
    
    # Filter to only include files matching the pattern z[0-9][0-9].csv
    # Specifically: z00, z01, z02, z03, z05-z15 (as specified by user)
    file_paths = []
    for fp in all_files:
        base_name = os.path.basename(fp)
        # Match pattern z followed by two digits and .csv
        if base_name.startswith('z') and base_name.endswith('.csv') and len(base_name) == 7:
            # Extract the number part (e.g., "00" from "z00.csv")
            try:
                num_part = base_name[1:3]
                num = int(num_part)
                # Include only the files the user specified: z00, z01, z02, z03, z05-z15
                if num in [0, 1, 2, 3] or (num >= 5 and num <= 15):
                    file_paths.append(fp)
            except ValueError:
                continue
    
    if not file_paths:
        print("No matching z*.csv files found in the specified directory.")
        print("Please check the directory path:")
        print(base_dir)
        return
    
    print(f"Found {len(file_paths)} droplet files:")
    for fp in file_paths:
        print(f"  - {os.path.basename(fp)}")
    print()
    
    # Process all slides to find unique droplets with maximum diameters
    print("Processing droplets...")
    result_df = process_all_slides(file_paths, error_margin=10)
    
    # Export the result to a CSV file
    output_csv_path = '/home/mib/Jimmy/Volume_Prediction/max_diameter_droplets.csv'
    result_df.to_csv(output_csv_path, index=False)
    print(f"\nResults exported to: {output_csv_path}")
    print(f"Total unique droplets found: {len(result_df)}")
    print()
    
    # Calculate and display the quantity of max diameter droplets in each slide
    slide_counts = result_df['slide'].value_counts().sort_index()
    
    print("=" * 60)
    print("Quantity of droplets with max diameter in each slide:")
    print("=" * 60)
    for slide_name, count in slide_counts.items():
        print(f"{slide_name}: {count} droplets have the max diameter across all slides")
    print("=" * 60)
    print()
    
    # Summary statistics
    print("Summary Statistics:")
    print(f"- Total unique droplets: {len(result_df)}")
    print(f"- Average diameter: {result_df['Diameter_μm'].mean():.2f} μm")
    print(f"- Max diameter: {result_df['Diameter_μm'].max():.2f} μm")
    print(f"- Min diameter: {result_df['Diameter_μm'].min():.2f} μm")
    print(f"- Median diameter: {result_df['Diameter_μm'].median():.2f} μm")
    print()
    
    # Show sample of results
    print("Sample of results (first 10 unique droplets):")
    print(result_df.head(10).to_string(index=False))
    print()

if __name__ == "__main__":
    main()

