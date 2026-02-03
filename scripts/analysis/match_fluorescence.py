#!/usr/bin/env python3
"""
Match fluorescent droplets with max diameter droplets and create occupancy CSV.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import argparse
import os

def match_fluorescence_to_droplets(max_diameter_file, fluorescence_file, error_margin=10):
    """
    Match fluorescent droplets to max diameter droplets based on spatial proximity.
    
    Args:
        max_diameter_file: CSV file with all max diameter droplets
        fluorescence_file: CSV file with fluorescent droplets
        error_margin: Maximum distance in pixels for matching (default 10)
        
    Returns:
        DataFrame with diameter and occupancy (0=empty, 1=fluorescent)
    """
    # Read the data files
    max_df = pd.read_csv(max_diameter_file)
    fluor_df = pd.read_csv(fluorescence_file)
    # Standardize column names for flexibility across input CSV variants
    def standardize_columns(df: pd.DataFrame, require_diameter: bool) -> pd.DataFrame:
        rename_map = {}
        # X coordinate
        if 'Center_X_px' not in df.columns:
            for cand in ['Center_X', 'X', 'x', 'center_x', 'centerX']:
                if cand in df.columns:
                    rename_map[cand] = 'Center_X_px'
                    break
        # Y coordinate
        if 'Center_Y_px' not in df.columns:
            for cand in ['Center_Y', 'Y', 'y', 'center_y', 'centerY']:
                if cand in df.columns:
                    rename_map[cand] = 'Center_Y_px'
                    break
        # Diameter (only required for BF/max)
        if require_diameter and 'Diameter_μm' not in df.columns:
            for cand in ['Diameter', 'diameter', 'Diameter_um', 'Diameter (um)']:
                if cand in df.columns:
                    rename_map[cand] = 'Diameter_μm'
                    break
        if rename_map:
            df = df.rename(columns=rename_map)
        # Validate presence
        missing = []
        if 'Center_X_px' not in df.columns:
            missing.append('Center_X_px')
        if 'Center_Y_px' not in df.columns:
            missing.append('Center_Y_px')
        if require_diameter and 'Diameter_μm' not in df.columns:
            missing.append('Diameter_μm')
        if missing:
            raise ValueError(f"Missing required columns after standardization: {missing}. Columns present: {list(df.columns)}")
        # Coerce numeric
        for col in ['Center_X_px', 'Center_Y_px'] + (['Diameter_μm'] if require_diameter else []):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows with NaN coords
        df = df.dropna(subset=['Center_X_px', 'Center_Y_px'])
        return df
    max_df = standardize_columns(max_df, require_diameter=True)
    fluor_df = standardize_columns(fluor_df, require_diameter=False)
    
    print(f"Loaded {len(max_df)} max diameter droplets")
    print(f"Loaded {len(fluor_df)} fluorescent droplets")
    
    # Initialize occupancy as 0 (empty) for all droplets
    max_df['occupancy'] = 0
    
    # Get coordinates for both datasets
    max_coords = max_df[['Center_X_px', 'Center_Y_px']].values
    fluor_coords = fluor_df[['Center_X_px', 'Center_Y_px']].values
    
    # Calculate pairwise distances between all max diameter droplets and fluorescent droplets
    distances = cdist(max_coords, fluor_coords, 'euclidean')
    
    # For each max diameter droplet, check if any fluorescent droplet is within error_margin
    for i in range(len(max_df)):
        if np.any(distances[i, :] <= error_margin):
            max_df.loc[i, 'occupancy'] = 1
    
    # Create result dataframe with diameter and occupancy
    result_df = pd.DataFrame({
        'Diameter_μm': max_df['Diameter_μm'],
        'occupancy': max_df['occupancy'].astype(int)
    })
    
    # Sort by occupancy (0 first, then 1)
    result_df = result_df.sort_values(by='occupancy', ascending=True).reset_index(drop=True)
    
    # Print statistics
    num_fluorescent = (result_df['occupancy'] == 1).sum()
    num_empty = (result_df['occupancy'] == 0).sum()
    
    print(f"\nMatching complete:")
    print(f"  - Empty droplets (occupancy=0): {num_empty}")
    print(f"  - Fluorescent droplets (occupancy=1): {num_fluorescent}")
    print(f"  - Total: {len(result_df)}")
    print(f"  - Fluorescence rate: {num_fluorescent/len(result_df)*100:.1f}%")
    
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match fluorescent droplets to BF droplets and export occupancy CSV.")
    parser.add_argument(
        "--bf",
        dest="bf_file",
        type=str,
        default="/data3/megan_data/Jimmy/max_diameter_droplets.csv",
        help="BF droplets CSV (e.g., max_diameter_droplets.csv or a per-slice BF CSV).",
    )
    parser.add_argument(
        "--fluor",
        dest="fluor_file",
        type=str,
        default="/data3/megan_data/Jimmy/Volume_Prediction/csv/MIB_6_035_2.csv",
        help="Fluorescent droplets CSV.",
    )
    parser.add_argument(
        "--out",
        dest="out_file",
        type=str,
        default="/data3/megan_data/Jimmy/droplet_occupancy.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--error",
        dest="error_margin",
        type=float,
        default=10.0,
        help="Maximum XY distance in pixels for a match (default: 10).",
    )
    args = parser.parse_args()
    max_diameter_file = args.bf_file
    fluorescence_file = args.fluor_file
    output_file = args.out_file
    
    # Match and create occupancy CSV
    result_df = match_fluorescence_to_droplets(max_diameter_file, fluorescence_file, error_margin=args.error_margin)
    
    # Export to CSV
    result_df.to_csv(output_file, index=False)
    print(f"\nResults exported to: {output_file}")
    
    # Show first few rows of each occupancy type
    print("\n--- Sample Empty Droplets (occupancy=0) ---")
    print(result_df[result_df['occupancy'] == 0].head(10).to_string(index=False))
    
    print("\n--- Sample Fluorescent Droplets (occupancy=1) ---")
    print(result_df[result_df['occupancy'] == 1].head(10).to_string(index=False))

