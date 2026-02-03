#!/usr/bin/env python3
"""
Match Fluorescent Droplets to BF Imaging Data

This script reads the unique_max_intensity_droplets.csv file (from fluorescent imaging)
and matches each droplet to its corresponding droplet in the BF imaging data at the
same z-level where maximum intensity was observed.

The matching uses the x,y location with a tolerance of ±5 pixels.
"""

import pandas as pd
import numpy as np
import os
import re
from typing import Dict, List, Tuple, Optional
import argparse


class FluorescentToBFMatcher:
    """Match fluorescent droplets to BF imaging data"""

    def __init__(self, fluorescent_csv: str, bf_directory: str, location_threshold: float = 5.0):
        """
        Initialize the matcher

        Args:
            fluorescent_csv: Path to unique_max_intensity_droplets.csv
            bf_directory: Directory containing BF imaging CSV files
            location_threshold: Maximum distance in pixels to consider droplets as matching
        """
        self.fluorescent_csv = fluorescent_csv
        self.bf_directory = bf_directory
        self.location_threshold = float(location_threshold)
        self.fluorescent_data = None
        self.bf_data_cache = {}  # Cache BF data by z-level

    def load_fluorescent_data(self) -> pd.DataFrame:
        """Load the fluorescent imaging results"""
        print(f"Loading fluorescent data from: {self.fluorescent_csv}")
        df = pd.read_csv(self.fluorescent_csv)
        print(f"  Loaded {len(df)} unique droplets")
        self.fluorescent_data = df
        return df

    def _parse_z_level(self, z_string: str) -> int:
        """Extract numeric z-level from string like 'z_8' or 'z8'"""
        match = re.search(r'z[_-]?(\d+)', z_string, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return -1

    def load_bf_csv(self, z_level: str) -> Optional[pd.DataFrame]:
        """Load BF CSV for a specific z-level"""
        if z_level in self.bf_data_cache:
            return self.bf_data_cache[z_level]

        # Try different filename patterns
        z_num = self._parse_z_level(z_level)
        possible_filenames = [
            f"z_{z_num}.csv",
            f"z{z_num}.csv",
            f"z_{z_num:02d}.csv",
            f"z{z_num:02d}.csv",
        ]

        for filename in possible_filenames:
            filepath = os.path.join(self.bf_directory, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    self.bf_data_cache[z_level] = df
                    print(f"  Loaded BF data for {z_level}: {len(df)} droplets")
                    return df
                except Exception as e:
                    print(f"  Error loading {filepath}: {e}")
                    return None

        print(f"  Warning: BF CSV not found for {z_level}")
        return None

    def find_matching_bf_droplet(self, x: float, y: float, bf_df: pd.DataFrame) -> Optional[Dict]:
        """
        Find a matching droplet in BF data based on x,y location

        Args:
            x: X coordinate from fluorescent data
            y: Y coordinate from fluorescent data
            bf_df: BF imaging dataframe for the z-level

        Returns:
            Dictionary with matching BF droplet data, or None if no match
        """
        # Find column names flexibly
        cols = bf_df.columns.tolist()
        x_col = next((c for c in cols if 'center_x' in c.lower()), None)
        y_col = next((c for c in cols if 'center_y' in c.lower()), None)
        diameter_col = next((c for c in cols if 'diameter' in c.lower()), None)
        intensity_col = next((c for c in cols if 'intensity' in c.lower()), None)
        area_col = next((c for c in cols if 'area' in c.lower()), None)
        circ_col = next((c for c in cols if 'circularity' in c.lower()), None)
        mask_col = next((c for c in cols if 'mask' in c.lower() or c.lower() == 'id'), None)
        
        if not x_col or not y_col:
            return None

        # Calculate distances to all droplets in BF data
        bf_df['distance'] = np.sqrt(
            (bf_df[x_col] - x)**2 + (bf_df[y_col] - y)**2
        )

        # Find droplets within threshold
        matches = bf_df[bf_df['distance'] <= self.location_threshold]

        if len(matches) == 0:
            return None

        # If multiple matches, take the closest one
        best_match = matches.loc[matches['distance'].idxmin()]

        result = {
            'bf_mask_id': best_match[mask_col] if mask_col else None,
            'bf_x': best_match[x_col],
            'bf_y': best_match[y_col],
            'bf_diameter': best_match[diameter_col] if diameter_col else None,
            'bf_intensity': best_match[intensity_col] if intensity_col else None,
            'bf_area': best_match[area_col] if area_col else None,
            'bf_circularity': best_match[circ_col] if circ_col else None,
            'distance': best_match['distance']
        }

        return result

    def match_all_droplets(self) -> pd.DataFrame:
        """
        Match all fluorescent droplets to BF data

        Returns:
            DataFrame with combined fluorescent and BF data
        """
        if self.fluorescent_data is None:
            self.load_fluorescent_data()

        print(f"\nMatching fluorescent droplets to BF imaging data...")
        print(f"Location threshold: {self.location_threshold} pixels\n")

        results = []
        matched_count = 0
        unmatched_count = 0

        for idx, row in self.fluorescent_data.iterrows():
            fluor_x = row['Center_X']
            fluor_y = row['Center_Y']
            z_level = row['Source_Z_Level']

            # Load BF data for this z-level
            bf_df = self.load_bf_csv(z_level)

            if bf_df is None:
                result = {
                    'Rank': row['Rank'],
                    'Fluorescent_X': fluor_x,
                    'Fluorescent_Y': fluor_y,
                    'Fluorescent_Max_Intensity': row['Max_Intensity'],
                    'Fluorescent_Diameter': row['Diameter'],
                    'Fluorescent_Area': row['Area'],
                    'Fluorescent_Circularity': row['Circularity'],
                    'Z_Level': z_level,
                    'Droplets_At_Location': row['Droplets_At_Location'],
                    'BF_Mask_ID': None,
                    'BF_X': None,
                    'BF_Y': None,
                    'BF_Diameter': row['Diameter'],  # fallback to fluorescent diameter
                    'BF_Intensity': None,
                    'BF_Area': None,
                    'BF_Circularity': None,
                    'XY_Distance': None,
                    'Match_Status': 'FALLBACK_FLUORESCENT'
                }
                unmatched_count += 1
            else:
                # Find matching BF droplet
                bf_match = self.find_matching_bf_droplet(fluor_x, fluor_y, bf_df)

                if bf_match:
                    result = {
                        'Rank': row['Rank'],
                        'Fluorescent_X': fluor_x,
                        'Fluorescent_Y': fluor_y,
                        'Fluorescent_Max_Intensity': row['Max_Intensity'],
                        'Fluorescent_Diameter': row['Diameter'],
                        'Fluorescent_Area': row['Area'],
                        'Fluorescent_Circularity': row['Circularity'],
                        'Z_Level': z_level,
                        'Droplets_At_Location': row['Droplets_At_Location'],
                        'BF_Mask_ID': bf_match['bf_mask_id'],
                        'BF_X': bf_match['bf_x'],
                        'BF_Y': bf_match['bf_y'],
                        'BF_Diameter': bf_match['bf_diameter'],
                        'BF_Intensity': bf_match['bf_intensity'],
                        'BF_Area': bf_match['bf_area'],
                        'BF_Circularity': bf_match['bf_circularity'],
                        'XY_Distance': bf_match['distance'],
                        'Match_Status': 'MATCHED'
                    }
                    matched_count += 1
                else:
                    result = {
                        'Rank': row['Rank'],
                        'Fluorescent_X': fluor_x,
                        'Fluorescent_Y': fluor_y,
                        'Fluorescent_Max_Intensity': row['Max_Intensity'],
                        'Fluorescent_Diameter': row['Diameter'],
                        'Fluorescent_Area': row['Area'],
                        'Fluorescent_Circularity': row['Circularity'],
                        'Z_Level': z_level,
                        'Droplets_At_Location': row['Droplets_At_Location'],
                        'BF_Mask_ID': None,
                        'BF_X': None,
                        'BF_Y': None,
                        'BF_Diameter': row['Diameter'],  # fallback to fluorescent diameter
                        'BF_Intensity': None,
                        'BF_Area': None,
                        'BF_Circularity': None,
                        'XY_Distance': None,
                        'Match_Status': 'FALLBACK_FLUORESCENT'
                    }
                    unmatched_count += 1

            results.append(result)

        results_df = pd.DataFrame(results)

        # Print summary
        print(f"\n{'='*80}")
        print(f"MATCHING SUMMARY")
        print(f"{'='*80}")
        print(f"Total fluorescent droplets: {len(self.fluorescent_data)}")
        print(f"Successfully matched: {matched_count}")
        print(f"Unmatched: {unmatched_count}")
        print(f"Match rate: {matched_count/len(self.fluorescent_data)*100:.1f}%")

        # Sort results by Match_Status priority, then by Rank
        status_priority = {
            'MATCHED': 0,
            'FALLBACK_FLUORESCENT': 1,
            'NO_MATCH_FOUND': 2,
            'BF_CSV_NOT_FOUND': 2,
        }
        results_df['__order__'] = results_df['Match_Status'].map(lambda s: status_priority.get(s, 99))
        results_df = (
            results_df
            .sort_values(by=['__order__', 'Rank'])
            .drop(columns=['__order__'])
            .reset_index(drop=True)
        )

        return results_df

    def print_sample_results(self, results_df: pd.DataFrame, n: int = 10):
        """Print sample results"""
        print(f"\n{'='*80}")
        print(f"SAMPLE RESULTS (Top {n} by Fluorescent Intensity)")
        print(f"{'='*80}\n")

        matched = results_df[results_df['Match_Status'] == 'MATCHED'].head(n)

        for idx, row in matched.iterrows():
            print(f"Rank #{int(row['Rank'])}")
            print(f"  Location: ({row['Fluorescent_X']:.1f}, {row['Fluorescent_Y']:.1f}) at {row['Z_Level']}")
            print(f"  Fluorescent Max Intensity: {row['Fluorescent_Max_Intensity']:.2f}")
            print(f"  Fluorescent Diameter: {row['Fluorescent_Diameter']:.2f} μm")
            print(f"  BF Diameter: {row['BF_Diameter']:.2f} μm" if pd.notna(row['BF_Diameter']) else "  BF Diameter: N/A")
            print(f"  Diameter Difference: {abs(row['Fluorescent_Diameter'] - row['BF_Diameter']):.2f} μm" if pd.notna(row['BF_Diameter']) else "")
            print(f"  XY Distance: {row['XY_Distance']:.2f} pixels")
            print()

    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """Save results to CSV"""
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Total records: {len(results_df)}")


def _find_xy_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = df.columns.tolist()
    x_col = next((c for c in cols if 'center_x' in c.lower()), None)
    y_col = next((c for c in cols if 'center_y' in c.lower()), None)
    return x_col, y_col


def direct_match_two_csvs(bf_csv: str, fluor_csv: str, threshold: float = 5.0) -> pd.DataFrame:
    """
    Directly match two CSVs (BF and Fluorescent) by centroid proximity.
    Returns a DataFrame with BF_* and F_* columns plus XY_Distance.
    """
    print(f"Direct mode: Matching BF '{bf_csv}' to Fluorescent '{fluor_csv}'")
    bf_df = pd.read_csv(bf_csv)
    f_df = pd.read_csv(fluor_csv)

    bf_x_col, bf_y_col = _find_xy_cols(bf_df)
    f_x_col, f_y_col = _find_xy_cols(f_df)

    if not bf_x_col or not bf_y_col or not f_x_col or not f_y_col:
        raise ValueError("Could not find Center_X/Center_Y columns in one of the CSVs.")

    # Prepare numpy arrays for vectorized distance computation
    f_x = f_df[f_x_col].to_numpy(dtype=float)
    f_y = f_df[f_y_col].to_numpy(dtype=float)

    results: List[Dict] = []

    for bf_idx, bf_row in bf_df.iterrows():
        bx = float(bf_row[bf_x_col])
        by = float(bf_row[bf_y_col])
        dx = f_x - bx
        dy = f_y - by
        dists = np.sqrt(dx * dx + dy * dy)
        within = dists <= float(threshold)
        if not np.any(within):
            continue
        for f_idx in np.where(within)[0]:
            f_row = f_df.iloc[f_idx]
            combined: Dict = {'XY_Distance': float(dists[f_idx])}
            for col in bf_df.columns:
                combined[f"BF_{col}"] = bf_row[col]
            for col in f_df.columns:
                combined[f"F_{col}"] = f_row[col]
            results.append(combined)

    out_df = pd.DataFrame(results)
    print(f"Direct mode: matched pairs = {len(out_df)}")
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description='Match fluorescent droplets to BF imaging data based on z-level and location'
    )

    # New direct mode arguments (optional). If both provided, script runs in direct mode.
    parser.add_argument(
        '--bf_csv',
        help='Path to a single BF CSV file (direct mode)'
    )
    parser.add_argument(
        '--fluor_csv',
        help='Path to a single fluorescent CSV file (direct mode)'
    )

    # Original by-z mode arguments
    parser.add_argument(
        'fluorescent_csv',
        nargs='?',
        help='Path to unique_max_intensity_droplets.csv from fluorescent imaging (by-z mode)'
    )
    parser.add_argument(
        'bf_directory',
        nargs='?',
        help='Directory containing BF imaging CSV files (by-z mode)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Location matching threshold in pixels (default: 5.0)'
    )
    parser.add_argument(
        '--output',
        help='Output CSV file path (default: auto-generated in same directory as input)'
    )

    args = parser.parse_args()

    # If direct mode args provided, run direct mode and exit.
    if args.bf_csv and args.fluor_csv:
        if not os.path.isfile(args.bf_csv):
            print(f"Error: BF CSV file '{args.bf_csv}' does not exist!")
            return
        if not os.path.isfile(args.fluor_csv):
            print(f"Error: Fluorescent CSV file '{args.fluor_csv}' does not exist!")
            return

        direct_df = direct_match_two_csvs(args.bf_csv, args.fluor_csv, args.threshold)
        if args.output:
            output_path = args.output
        else:
            # Place output alongside BF CSV by default
            out_dir = os.path.dirname(args.bf_csv)
            base_bf = os.path.splitext(os.path.basename(args.bf_csv))[0]
            base_f = os.path.splitext(os.path.basename(args.fluor_csv))[0]
            output_path = os.path.join(out_dir, f"{base_bf}__{base_f}__direct_matched.csv")
        direct_df.to_csv(output_path, index=False)
        print(f"Wrote direct matched CSV: {output_path}")
        return

    # Otherwise proceed with original by-z workflow

    # Validate inputs
    if not os.path.isfile(args.fluorescent_csv):
        print(f"Error: Fluorescent CSV file '{args.fluorescent_csv}' does not exist!")
        return

    if not os.path.isdir(args.bf_directory):
        print(f"Error: BF directory '{args.bf_directory}' does not exist!")
        return

    # Create matcher and run
    matcher = FluorescentToBFMatcher(args.fluorescent_csv, args.bf_directory, args.threshold)
    results_df = matcher.match_all_droplets()

    # Print sample results
    matcher.print_sample_results(results_df, n=15)

    # Calculate and print diameter comparison statistics
    matched = results_df[results_df['Match_Status'] == 'MATCHED']
    if len(matched) > 0:
        print(f"\n{'='*80}")
        print(f"DIAMETER COMPARISON STATISTICS")
        print(f"{'='*80}")
        fluor_diameters = matched['Fluorescent_Diameter'].dropna()
        bf_diameters = matched['BF_Diameter'].dropna()

        if len(fluor_diameters) > 0 and len(bf_diameters) > 0:
            print(f"Fluorescent Diameter: {fluor_diameters.mean():.2f} ± {fluor_diameters.std():.2f} μm")
            print(f"BF Diameter: {bf_diameters.mean():.2f} ± {bf_diameters.std():.2f} μm")

            # Calculate diameter differences
            valid_pairs = matched[matched['BF_Diameter'].notna() & matched['Fluorescent_Diameter'].notna()]
            if len(valid_pairs) > 0:
                diameter_diff = (valid_pairs['Fluorescent_Diameter'] - valid_pairs['BF_Diameter']).abs()
                print(f"Mean Absolute Diameter Difference: {diameter_diff.mean():.2f} ± {diameter_diff.std():.2f} μm")
                print(f"Max Diameter Difference: {diameter_diff.max():.2f} μm")
                print(f"Pairs with <5μm difference: {(diameter_diff < 5).sum()} ({(diameter_diff < 5).sum()/len(valid_pairs)*100:.1f}%)")

        print(f"\nMean XY Distance: {matched['XY_Distance'].mean():.2f} ± {matched['XY_Distance'].std():.2f} pixels")

    # Save results
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output filename in same directory as input
        input_dir = os.path.dirname(args.fluorescent_csv)
        output_path = os.path.join(input_dir, 'fluorescent_to_bf_matched.csv')

    matcher.save_results(results_df, output_path)


if __name__ == "__main__":
    main()

