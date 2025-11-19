#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match BF detections to Fluorescent detections by coordinates."
    )
    parser.add_argument("--bf", required=True, help="Path to BF CSV file")
    parser.add_argument("--f", required=True, help="Path to Fluorescent CSV file")
    parser.add_argument("--out", required=True, help="Path to output matched CSV")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Positional tolerance in pixels for matching (Euclidean distance). Default exact match.",
    )
    return parser.parse_args()


def read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def coord_key(row: Dict[str, str]) -> Tuple[float, float]:
    # Robust rounding to 2 decimals (inputs often like '395.00')
    x = round(float(row["Center_X_px"]), 2)
    y = round(float(row["Center_Y_px"]), 2)
    return (x, y)


def build_f_index(
    f_rows: List[Dict[str, str]]
) -> Dict[Tuple[float, float], List[Dict[str, str]]]:
    index: Dict[Tuple[float, float], List[Dict[str, str]]] = {}
    for r in f_rows:
        k = coord_key(r)
        index.setdefault(k, []).append(r)
    return index


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def match_rows(
    bf_rows: List[Dict[str, str]],
    f_rows: List[Dict[str, str]],
    tolerance: float,
) -> List[Dict[str, str]]:
    """
    Returns combined rows for all matches.
    If tolerance == 0: exact match by (x,y) after rounding to 2 decimals.
    Else: any F row whose (x,y) is within Euclidean distance <= tolerance.
    """
    combined: List[Dict[str, str]] = []
    if tolerance <= 0:
        f_index = build_f_index(f_rows)
        for bf in bf_rows:
            k = coord_key(bf)
            if k in f_index:
                for fr in f_index[k]:
                    combined.append(build_combined_row(bf, fr))
    else:
        # Quadratic scan, acceptable for few thousand rows
        f_coords = [(coord_key(fr), fr) for fr in f_rows]
        for bf in bf_rows:
            k_bf = coord_key(bf)
            for k_f, fr in f_coords:
                if distance(k_bf, k_f) <= tolerance:
                    combined.append(build_combined_row(bf, fr))
    return combined


def build_combined_row(bf: Dict[str, str], fr: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    # Prefix BF_ and F_ to avoid header collisions and keep provenance clear
    for key, val in bf.items():
        out[f"BF_{key}"] = val
    for key, val in fr.items():
        out[f"F_{key}"] = val
    return out


def write_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        # If no rows, still write a CSV with a minimal header for clarity
        with open(out_path, "w", newline="") as f:
            f.write("BF_Mask_ID,BF_Center_X_px,BF_Center_Y_px,BF_Diameter_μm,BF_Mean_Intensity,BF_Area_μm²,BF_Circularity,F_Mask_ID,F_Center_X_px,F_Center_Y_px,F_Diameter_μm,F_Mean_Intensity,F_Area_μm²,F_Circularity\n")
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bf_rows = read_csv_rows(args.bf)
    f_rows = read_csv_rows(args.f)
    matched = match_rows(bf_rows, f_rows, args.tolerance)
    write_csv(matched, args.out)
    print(f"Matched pairs: {len(matched)}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()


