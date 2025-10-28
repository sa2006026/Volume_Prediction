#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute occupancy of target (anchor) CSV rows by matching to source CSV within a pixel radius. "
            "Output columns: diameter, occupancy. Sorted by occupancy (1 first)."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to source CSV to match against (e.g., fluorescent z*.csv)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Path to target CSV (anchor, e.g., BF z*.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Max Euclidean distance in pixels to count as a match (default: 5)",
    )
    return parser.parse_args()


def load_z_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["Center_X_px", "Center_Y_px", "Diameter_μm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing}: {csv_path}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()

    src_df = load_z_csv(args.source)
    tgt_df = load_z_csv(args.target)

    src_x = src_df["Center_X_px"].to_numpy(dtype=float)
    src_y = src_df["Center_Y_px"].to_numpy(dtype=float)

    occupancy = np.zeros(len(tgt_df), dtype=int)

    tgt_x = tgt_df["Center_X_px"].to_numpy(dtype=float)
    tgt_y = tgt_df["Center_Y_px"].to_numpy(dtype=float)

    for i, (x, y) in enumerate(zip(tgt_x, tgt_y)):
        dx = src_x - x
        dy = src_y - y
        rect = (np.abs(dx) <= args.radius) & (np.abs(dy) <= args.radius)
        if np.any(rect):
            dists = np.hypot(dx[rect], dy[rect])
            if np.any(dists <= args.radius):
                occupancy[i] = 1

    out_df = pd.DataFrame({
        "diameter": tgt_df["Diameter_μm"],
        "occupancy": occupancy,
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.sort_values(by=["occupancy"], ascending=[False], inplace=True, kind="mergesort")
    out_df.to_csv(args.out, index=False)

    print(f"Wrote {args.out} (kept {int(occupancy.sum())}/{len(occupancy)} matches, radius <= {args.radius}).")


if __name__ == "__main__":
    main()


