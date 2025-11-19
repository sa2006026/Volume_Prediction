#!/usr/bin/env python3
"""
Compare droplet diameters between the global max set and a specific z-slice (e.g., z_15 BF).
Matches droplets by nearest neighbor within an XY error margin (pixels), then outputs:
- Center_X_px, Center_Y_px (from max set)
- Diameter_max_um
- Diameter_zslice_um
- delta_um = Diameter_zslice_um - Diameter_max_um
- dist_px (match distance)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare droplet diameters between max set and a z-slice BF CSV.")
    parser.add_argument(
        "--max",
        dest="max_csv",
        default="/data3/megan_data/Jimmy/max_diameter_droplets.csv",
        help="Path to max diameter droplets CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--bf",
        dest="bf_csv",
        default="/data3/megan_data/Jimmy/Volume_Prediction/csv/z_15/z_15_bf_2.csv",
        help="Path to BF z-slice CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        dest="out_csv",
        default="/data3/megan_data/Jimmy/diameter_difference_z15.csv",
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--error",
        dest="error_px",
        type=float,
        default=10.0,
        help="XY matching error margin in pixels (default: %(default)s)",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, x_name: str, y_name: str, d_name: str, label: str) -> None:
    missing = [c for c in (x_name, y_name, d_name) if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def compute_nearest_matches(
    ref_xy: np.ndarray, cand_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point in ref_xy, find nearest neighbor index and distance in cand_xy.
    Returns (indices, distances). If cand_xy is empty, returns (-1, inf) for each ref point.
    """
    if cand_xy.size == 0:
        return np.full(ref_xy.shape[0], -1, dtype=int), np.full(ref_xy.shape[0], np.inf, dtype=float)

    nearest_indices = np.empty(ref_xy.shape[0], dtype=int)
    nearest_distances = np.empty(ref_xy.shape[0], dtype=float)

    # Loop over ref points; arrays are modest in size so this is fine
    for i in range(ref_xy.shape[0]):
        dx = cand_xy[:, 0] - ref_xy[i, 0]
        dy = cand_xy[:, 1] - ref_xy[i, 1]
        dists = np.hypot(dx, dy)
        j = int(np.argmin(dists))
        nearest_indices[i] = j
        nearest_distances[i] = float(dists[j])
    return nearest_indices, nearest_distances


def main() -> None:
    args = parse_args()

    # Load data
    max_df = pd.read_csv(args.max_csv)
    bf_df = pd.read_csv(args.bf_csv)

    # Validate columns
    ensure_columns(max_df, "Center_X_px", "Center_Y_px", "Diameter_μm", "max_diameter CSV")
    ensure_columns(bf_df, "Center_X_px", "Center_Y_px", "Diameter_μm", "BF CSV")

    # Drop rows with missing required fields
    max_df = max_df.dropna(subset=["Center_X_px", "Center_Y_px", "Diameter_μm"]).copy()
    bf_df = bf_df.dropna(subset=["Center_X_px", "Center_Y_px", "Diameter_μm"]).copy()

    # Prepare coordinate arrays
    max_xy = max_df[["Center_X_px", "Center_Y_px"]].to_numpy(dtype=float)
    bf_xy = bf_df[["Center_X_px", "Center_Y_px"]].to_numpy(dtype=float)

    # Nearest neighbor from each max droplet to BF z-slice droplet
    nn_idx, nn_dist = compute_nearest_matches(max_xy, bf_xy)

    # Filter by error margin
    within_margin = nn_dist <= float(args.error_px)
    matched_max = max_df.loc[within_margin].reset_index(drop=True)
    matched_bf = bf_df.iloc[nn_idx[within_margin]].reset_index(drop=True)
    matched_dist = nn_dist[within_margin]

    # Build result
    out_df = pd.DataFrame(
        {
            "Center_X_px": matched_max["Center_X_px"],
            "Center_Y_px": matched_max["Center_Y_px"],
            "Diameter_max_um": matched_max["Diameter_μm"],
            "Diameter_zslice_um": matched_bf["Diameter_μm"],
            "delta_um": matched_bf["Diameter_μm"].to_numpy(dtype=float)
            - matched_max["Diameter_μm"].to_numpy(dtype=float),
            "dist_px": matched_dist,
            "max_slide": matched_max.get("slide", pd.Series([None] * len(matched_max))),
        }
    )

    # Write output
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    # Console summary
    total_max = len(max_df)
    matched_count = len(out_df)
    print(f"Total droplets in max set       : {total_max}")
    print(f"Matched to BF within {args.error_px:.1f}px : {matched_count}")
    if matched_count:
        delta = out_df["delta_um"]
        print(f"Delta (zslice - max) um: mean={delta.mean():.2f}, median={delta.median():.2f}, min={delta.min():.2f}, max={delta.max():.2f}")
        print(f"Output: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


