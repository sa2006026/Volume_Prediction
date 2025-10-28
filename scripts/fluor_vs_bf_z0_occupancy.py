#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute occupancy by checking if a location in one z_0.csv exists within a pixel radius "
            "in the other z_0.csv. Outputs x,y,diameter (from anchor), occupancy."
        )
    )
    parser.add_argument(
        "--fluor_z0",
        type=Path,
        default=Path(
            "/data3/megan_data/Jimmy/Volume_Prediction/csv/1/csv_realunit/fluorescent imaging/z_0.csv"
        ),
        help="Path to fluorescent imaging z_0.csv",
    )
    parser.add_argument(
        "--bf_z0",
        type=Path,
        default=Path(
            "/data3/megan_data/Jimmy/Volume_Prediction/csv/1/csv_realunit/BF imaging/z_0.csv"
        ),
        help="Path to brightfield imaging z_0.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/data3/megan_data/Jimmy/Volume_Prediction/csv/1/csv_realunit/fluorescent imaging/z_0_fluor_vs_bf_occupancy.csv"
        ),
        help="Output CSV path for x,y,diameter,occupancy",
    )
    parser.add_argument(
        "--anchor",
        choices=["bf", "fluor"],
        default="bf",
        help=(
            "Which dataset to anchor rows on: 'bf' (rows from BF z_0.csv, default) or 'fluor' "
            "(rows from fluorescent z_0.csv)."
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Maximum Euclidean distance in pixels to consider a location match (default: 5)",
    )
    return parser.parse_args()


def load_z0(z0_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(z0_csv)
    required_cols = ["Center_X_px", "Center_Y_px", "Diameter_μm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{z0_csv} missing required columns: {missing}")
    df["Center_X_px"] = pd.to_numeric(df["Center_X_px"], errors="coerce")
    df["Center_Y_px"] = pd.to_numeric(df["Center_Y_px"], errors="coerce")
    df["Diameter_μm"] = pd.to_numeric(df["Diameter_μm"], errors="coerce")
    df = df.dropna(subset=["Center_X_px", "Center_Y_px", "Diameter_μm"]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()

    fluor_df = load_z0(args.fluor_z0)
    bf_df = load_z0(args.bf_z0)

    if args.anchor == "bf":
        anchor_df = bf_df
        search_x = fluor_df["Center_X_px"].to_numpy(dtype=float)
        search_y = fluor_df["Center_Y_px"].to_numpy(dtype=float)
    else:
        anchor_df = fluor_df
        search_x = bf_df["Center_X_px"].to_numpy(dtype=float)
        search_y = bf_df["Center_Y_px"].to_numpy(dtype=float)

    # Prepare arrays
    anchor_x = anchor_df["Center_X_px"].to_numpy(dtype=float)
    anchor_y = anchor_df["Center_Y_px"].to_numpy(dtype=float)
    occupancy = np.zeros(len(anchor_df), dtype=int)

    # For each anchor location, check any counterpart within radius
    for i, (x, y) in enumerate(zip(anchor_x, anchor_y)):
        dx = search_x - x
        dy = search_y - y
        rect_mask = (np.abs(dx) <= args.radius) & (np.abs(dy) <= args.radius)
        if np.any(rect_mask):
            dists = np.hypot(dx[rect_mask], dy[rect_mask])
            if np.any(dists <= args.radius):
                occupancy[i] = 1

    out_df = pd.DataFrame(
        {
            "x": anchor_df["Center_X_px"],
            "y": anchor_df["Center_Y_px"],
            "diameter": anchor_df["Diameter_μm"],
            "occupancy": occupancy,
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Sort by occupancy (1 first), stable order within groups
    out_df.sort_values(by=["occupancy"], ascending=[False], inplace=True, kind="mergesort")
    out_df.to_csv(args.out, index=False)

    total = len(out_df)
    occ = int(out_df["occupancy"].sum())
    who = "BF" if args.anchor == "bf" else "fluorescent"
    print(
        f"Wrote {args.out} with {occ}/{total} {who} locations having a counterpart match (radius <= {args.radius})."
    )


if __name__ == "__main__":
    main()


