#!/usr/bin/env python3

import argparse
from math import hypot
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace z_0.csv diameters using matched BF_Diameter from fluorescent_to_bf_matched.csv "
            "when a BF location is within a given pixel radius."
        )
    )
    parser.add_argument(
        "--matched",
        type=Path,
        default=Path(
            "/data3/megan_data/Jimmy/Volume_Prediction/csv/1/csv_realunit/fluorescent imaging/fluorescent_to_bf_matched.csv"
        ),
        help="Path to fluorescent_to_bf_matched.csv",
    )
    parser.add_argument(
        "--z0",
        type=Path,
        default=Path(
            "/data3/megan_data/Jimmy/Volume_Prediction/csv/1/csv_realunit/BF imaging/z_0.csv"
        ),
        help="Path to z_0.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/data3/megan_data/Jimmy/Volume_Prediction/csv/1/csv_realunit/BF imaging/z_0_replaced_from_match.csv"
        ),
        help="Output CSV path for x,y,diameter,occupancy",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Maximum Euclidean distance in pixels to consider a BF match (default: 5)",
    )
    return parser.parse_args()


def load_matched_bf_points(matched_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(matched_csv)
    # Keep only rows that truly have BF coordinates and were matched
    df = df.loc[df.get("Match_Status").astype(str) == "MATCHED", ["BF_X", "BF_Y", "BF_Diameter"]]
    # Coerce to numeric, drop rows with any missing numeric data
    for col in ("BF_X", "BF_Y", "BF_Diameter"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["BF_X", "BF_Y", "BF_Diameter"]).reset_index(drop=True)
    return df


def load_z0(z0_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(z0_csv)
    # Ensure required columns exist
    required_cols = ["Center_X_px", "Center_Y_px", "Diameter_μm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"z_0.csv missing required columns: {missing}")
    # Coerce to numeric
    df["Center_X_px"] = pd.to_numeric(df["Center_X_px"], errors="coerce")
    df["Center_Y_px"] = pd.to_numeric(df["Center_Y_px"], errors="coerce")
    df["Diameter_μm"] = pd.to_numeric(df["Diameter_μm"], errors="coerce")
    df = df.dropna(subset=["Center_X_px", "Center_Y_px", "Diameter_μm"]).reset_index(drop=True)
    return df


def find_best_match(
    x: float,
    y: float,
    bf_x: np.ndarray,
    bf_y: np.ndarray,
    bf_diameter: np.ndarray,
    radius: float,
) -> Tuple[bool, float]:
    # Quick rectangular filter then compute Euclidean distances
    dx = bf_x - x
    dy = bf_y - y
    rect_mask = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
    if not np.any(rect_mask):
        return False, float("nan")

    candidates_idx = np.where(rect_mask)[0]
    if candidates_idx.size == 0:
        return False, float("nan")

    # Compute true distances only for candidates
    dists = np.hypot(dx[candidates_idx], dy[candidates_idx])
    within_idx = candidates_idx[dists <= radius]
    if within_idx.size == 0:
        return False, float("nan")

    # Choose the closest match
    best_local = within_idx[np.argmin(np.hypot(bf_x[within_idx] - x, bf_y[within_idx] - y))]
    return True, float(bf_diameter[best_local])


def main() -> None:
    args = parse_args()

    matched_df = load_matched_bf_points(args.matched)
    z0_df = load_z0(args.z0)

    bf_x = matched_df["BF_X"].to_numpy(dtype=float)
    bf_y = matched_df["BF_Y"].to_numpy(dtype=float)
    bf_diameter = matched_df["BF_Diameter"].to_numpy(dtype=float)

    replaced_flags = np.zeros(len(z0_df), dtype=int)
    output_diameters = z0_df["Diameter_μm"].to_numpy(dtype=float).copy()

    for i, (x, y) in enumerate(zip(z0_df["Center_X_px"].to_numpy(), z0_df["Center_Y_px"].to_numpy())):
        matched, new_diameter = find_best_match(x, y, bf_x, bf_y, bf_diameter, args.radius)
        if matched:
            output_diameters[i] = new_diameter
            replaced_flags[i] = 1

    out_df = pd.DataFrame(
        {
            "x": z0_df["Center_X_px"],
            "y": z0_df["Center_Y_px"],
            "diameter": output_diameters,
            "occupancy": replaced_flags,
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Sort by occupancy (replaced first)
    out_df.sort_values(by=["occupancy"], ascending=[False], inplace=True, kind="mergesort")
    out_df.to_csv(args.out, index=False)

    total = len(out_df)
    replaced = int(out_df["occupancy"].sum())
    print(
        f"Wrote {args.out} with {replaced}/{total} locations replaced using matched BF_Diameter (radius <= {args.radius})."
    )


if __name__ == "__main__":
    main()


