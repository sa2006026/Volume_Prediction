#!/usr/bin/env python3
"""
Compare a single BF z-level CSV (e.g., z_0.csv) with fluorescent_to_bf_matched.csv

For each droplet in fluorescent_to_bf_matched.csv, attempt to find a neighbor in the
provided BF z-level file within a pixel threshold (default ±5). Report diameter
comparisons:
  - Fluorescent_Diameter vs BF(zX)_Diameter
  - BF_Diameter(from matched source z-level) vs BF(zX)_Diameter, when available

You can optionally restrict the comparison to only rows whose Source Z_Level equals
the BF file's z-level (e.g., only Z_Level == 'z_0').
"""

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd


def parse_z_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = re.search(r"^(z[_-]?\d+)\.csv$", base, re.IGNORECASE)
    return m.group(1) if m else None


def find_column(cols, key_substr: str) -> Optional[str]:
    key = key_substr.lower()
    return next((c for c in cols if key in c.lower()), None)


def load_bf_csv(bf_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(bf_csv_path)
    cols = df.columns.tolist()
    x_col = find_column(cols, 'center_x')
    y_col = find_column(cols, 'center_y')
    d_col = find_column(cols, 'diameter')
    m_col = find_column(cols, 'mask')
    if x_col is None or y_col is None or d_col is None:
        raise ValueError("BF CSV missing required columns (Center_X, Center_Y, Diameter)")
    # Rename to unified keys for ease
    rename_map = {}
    if m_col: rename_map[m_col] = 'BF_Mask_ID'
    rename_map[x_col] = 'BF_X'
    rename_map[y_col] = 'BF_Y'
    rename_map[d_col] = 'BF_Diameter_from_file'
    df = df.rename(columns=rename_map)
    return df


def compare_against_bf_level(
    matched_csv: str,
    bf_csv_path: str,
    threshold: float = 5.0,
    only_same_z: bool = False,
) -> pd.DataFrame:
    matched_df = pd.read_csv(matched_csv)
    bf_df = load_bf_csv(bf_csv_path)

    # Ensure required cols exist in matched_df
    required = [
        'Rank', 'Fluorescent_X', 'Fluorescent_Y', 'Fluorescent_Diameter',
        'BF_Diameter', 'Z_Level', 'Match_Status'
    ]
    for col in required:
        if col not in matched_df.columns:
            raise ValueError(f"Missing column in matched CSV: {col}")

    bf_level = parse_z_from_filename(bf_csv_path)  # e.g., 'z_0'

    # Optionally filter matched rows to same z-level
    comp_df = matched_df.copy()
    if only_same_z and bf_level is not None:
        comp_df = comp_df[comp_df['Z_Level'].str.lower() == bf_level.lower()]

    # Prepare results
    results = []

    # Vectorized search: compute distances to nearest BF neighbor
    # We'll loop on matched rows to keep it clear and handle per-row details
    for _, row in comp_df.iterrows():
        x, y = row['Fluorescent_X'], row['Fluorescent_Y']
        # distance to all BF points
        dists = np.sqrt((bf_df['BF_X'] - x) ** 2 + (bf_df['BF_Y'] - y) ** 2)
        min_idx = int(dists.idxmin())
        min_dist = float(dists.iloc[min_idx])

        if min_dist <= threshold:
            bf_row = bf_df.iloc[min_idx]
            z0_diam = bf_row['BF_Diameter_from_file']
            result = {
                'Rank': row['Rank'],
                'Z_Level_in_matched': row['Z_Level'],
                'BF_Level_File': bf_level if bf_level else '',
                'Fluorescent_X': x,
                'Fluorescent_Y': y,
                'BF_X_from_file': bf_row['BF_X'],
                'BF_Y_from_file': bf_row['BF_Y'],
                'XY_Distance_to_file': min_dist,
                'Match_Status_in_matched': row['Match_Status'],
                'Fluorescent_Diameter': row['Fluorescent_Diameter'],
                'BF_Diameter_from_matched': row['BF_Diameter'] if pd.notna(row['BF_Diameter']) else None,
                'BF_Diameter_from_file': z0_diam,
                'Diff_Fluor_vs_BFfile': (row['Fluorescent_Diameter'] - z0_diam)
                    if pd.notna(row['Fluorescent_Diameter']) and pd.notna(z0_diam) else None,
                'Diff_BFmatched_vs_BFfile': (row['BF_Diameter'] - z0_diam)
                    if pd.notna(row['BF_Diameter']) and pd.notna(z0_diam) else None,
            }
        else:
            result = {
                'Rank': row['Rank'],
                'Z_Level_in_matched': row['Z_Level'],
                'BF_Level_File': bf_level if bf_level else '',
                'Fluorescent_X': x,
                'Fluorescent_Y': y,
                'BF_X_from_file': None,
                'BF_Y_from_file': None,
                'XY_Distance_to_file': None,
                'Match_Status_in_matched': row['Match_Status'],
                'Fluorescent_Diameter': row['Fluorescent_Diameter'],
                'BF_Diameter_from_matched': row['BF_Diameter'] if pd.notna(row['BF_Diameter']) else None,
                'BF_Diameter_from_file': None,
                'Diff_Fluor_vs_BFfile': None,
                'Diff_BFmatched_vs_BFfile': None,
            }

        results.append(result)

    # Build DataFrame, ensuring expected columns exist even if empty
    expected_cols = [
        'Rank', 'Z_Level_in_matched', 'BF_Level_File', 'Fluorescent_X', 'Fluorescent_Y',
        'BF_X_from_file', 'BF_Y_from_file', 'XY_Distance_to_file', 'Match_Status_in_matched',
        'Fluorescent_Diameter', 'BF_Diameter_from_matched', 'BF_Diameter_from_file',
        'Diff_Fluor_vs_BFfile', 'Diff_BFmatched_vs_BFfile',
    ]
    if results:
        out_df = pd.DataFrame(results)
    else:
        out_df = pd.DataFrame(columns=expected_cols)

    # Sort for readability based on original match status priority, then BF-file match, then Rank
    status_priority = {
        'MATCHED': 0,
        'FALLBACK_FLUORESCENT': 1,
        'NO_MATCH_FOUND': 2,
        'BF_CSV_NOT_FOUND': 2,
    }
    out_df['__status_order__'] = out_df['Match_Status_in_matched'].map(lambda s: status_priority.get(s, 99))
    if 'XY_Distance_to_file' in out_df.columns:
        out_df['__group__'] = out_df['XY_Distance_to_file'].apply(lambda v: 0 if pd.notna(v) else 1)
        out_df = (
            out_df
            .sort_values(by=['__status_order__', '__group__', 'Rank'])
            .drop(columns=['__status_order__', '__group__'])
            .reset_index(drop=True)
        )
    else:
        out_df = (
            out_df
            .sort_values(by=['__status_order__', 'Rank'])
            .drop(columns=['__status_order__'])
            .reset_index(drop=True)
        )
    return out_df


def main():
    parser = argparse.ArgumentParser(description='Compare one BF z-level CSV to fluorescent_to_bf_matched.csv')
    parser.add_argument('bf_csv', help='Path to BF z-level CSV (e.g., .../BF imaging/z_0.csv)')
    parser.add_argument('matched_csv', help='Path to fluorescent_to_bf_matched.csv')
    parser.add_argument('--threshold', type=float, default=5.0, help='XY match threshold in pixels (default: 5.0)')
    parser.add_argument('--only-same-z', action='store_true', help='Restrict to rows where Z_Level equals BF file z-level')
    parser.add_argument('--output', help='Output CSV path (default: alongside BF CSV)')

    args = parser.parse_args()

    if not os.path.isfile(args.bf_csv):
        print(f"Error: BF CSV '{args.bf_csv}' not found")
        return
    if not os.path.isfile(args.matched_csv):
        print(f"Error: matched CSV '{args.matched_csv}' not found")
        return

    comp_df = compare_against_bf_level(
        matched_csv=args.matched_csv,
        bf_csv_path=args.bf_csv,
        threshold=args.threshold,
        only_same_z=args.only_same_z,
    )

    # Summary
    matched_mask = comp_df['XY_Distance_to_file'].notna()
    n_found = int(matched_mask.sum())
    n_total = int(len(comp_df))
    print(f"Found {n_found}/{n_total} neighbors in BF file within threshold {args.threshold}.")
    if n_found > 0 and 'Diff_Fluor_vs_BFfile' in comp_df.columns:
        diffs = comp_df.loc[matched_mask, 'Diff_Fluor_vs_BFfile'].dropna()
        if len(diffs) > 0:
            print(f"Fluorescent vs BF(file) | mean abs diff = {diffs.abs().mean():.2f} μm (n={len(diffs)})")

    # Output
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.dirname(args.bf_csv)
        bf_level = parse_z_from_filename(args.bf_csv) or 'zX'
        out_path = os.path.join(out_dir, f'compare_{bf_level}_vs_fluor_matched.csv')

    comp_df.to_csv(out_path, index=False)
    print(f"Saved comparison CSV to: {out_path}")


if __name__ == '__main__':
    main()


