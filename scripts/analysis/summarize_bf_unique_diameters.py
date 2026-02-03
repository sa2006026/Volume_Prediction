#!/usr/bin/env python3
"""
Summarize unique droplet diameters across BF z-levels (cluster by XY within ±5 px)

Input: Directory containing BF CSV files (z_*.csv)
Output: CSV summarizing each unique droplet cluster with diameter stats per z-level

For each cluster:
 - Representative (avg) XY
 - Observed z-levels
 - Count of observations
 - Per-z diameter values
 - Summary stats (min, max, mean, std) of diameter
"""

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def list_bf_csvs(folder: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        if re.match(r"z[_-]?\d+\.csv$", name, re.IGNORECASE):
            files.append(os.path.join(folder, name))
    return files


def parse_z_from_filename(path: str) -> Optional[int]:
    m = re.search(r"z[_-]?(\d+)\.csv$", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else None


def find_col(cols: List[str], key: str) -> Optional[str]:
    key = key.lower()
    return next((c for c in cols if key in c.lower()), None)


def read_bf_rows(path: str) -> List[Dict]:
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    x_col = find_col(cols, 'center_x')
    y_col = find_col(cols, 'center_y')
    d_col = find_col(cols, 'diameter')
    if x_col is None or y_col is None or d_col is None:
        return []
    z = parse_z_from_filename(path)
    rows: List[Dict] = []
    for _, r in df.iterrows():
        try:
            x = float(r[x_col])
            y = float(r[y_col])
            d = float(r[d_col])
        except Exception:
            continue
        rows.append({
            'x': x,
            'y': y,
            'diameter': d,
            'z': z if z is not None else -1,
            'file': os.path.basename(path),
        })
    return rows


class Cluster:
    def __init__(self, x: float, y: float):
        self.rep_x = x
        self.rep_y = y
        self.points: List[Dict] = []

    def add(self, p: Dict) -> None:
        self.points.append(p)
        n = len(self.points)
        self.rep_x = self.rep_x + (p['x'] - self.rep_x) / n
        self.rep_y = self.rep_y + (p['y'] - self.rep_y) / n

    def within(self, x: float, y: float, thr: float) -> bool:
        return abs(self.rep_x - x) <= thr and abs(self.rep_y - y) <= thr

    def z_levels(self) -> List[int]:
        return sorted({p['z'] for p in self.points})

    def diameter_stats(self) -> Dict:
        ds = [p['diameter'] for p in self.points]
        if not ds:
            return {'min': None, 'max': None, 'mean': None, 'std': None}
        arr = np.array(ds, dtype=float)
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=0)),
        }


def build_clusters(rows: List[Dict], threshold: float) -> List[Cluster]:
    clusters: List[Cluster] = []
    for p in rows:
        matched = False
        for c in clusters:
            if c.within(p['x'], p['y'], threshold):
                c.add(p)
                matched = True
                break
        if not matched:
            c = Cluster(p['x'], p['y'])
            c.add(p)
            clusters.append(c)
    return clusters


def summarize(folder: str, threshold: float) -> pd.DataFrame:
    csvs = list_bf_csvs(folder)
    all_rows: List[Dict] = []
    for path in csvs:
        all_rows.extend(read_bf_rows(path))
    clusters = build_clusters(all_rows, threshold)

    # Collect per-cluster rows
    out_rows: List[Dict] = []
    for idx, c in enumerate(clusters, start=1):
        z_to_diams: Dict[int, List[float]] = {}
        for p in c.points:
            z_to_diams.setdefault(p['z'], []).append(p['diameter'])

        # For each z, record mean diameter (in case multiple per z in proximity)
        per_z = {f"z_{z}": float(np.mean(diams)) for z, diams in sorted(z_to_diams.items())}
        stats = c.diameter_stats()
        row = {
            'cluster_id': idx,
            'rep_x': c.rep_x,
            'rep_y': c.rep_y,
            'num_observations': len(c.points),
            'z_levels': ','.join(f"z_{z}" for z in c.z_levels()),
            'diameter_min': stats['min'],
            'diameter_max': stats['max'],
            'diameter_mean': stats['mean'],
            'diameter_std': stats['std'],
        }
        row.update(per_z)
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    # Sort clusters by diameter_max desc, then observations desc
    if not df.empty:
        df = df.sort_values(by=['diameter_max', 'num_observations'], ascending=[False, False]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description='Summarize unique droplet diameters across BF z-levels')
    parser.add_argument('bf_folder', help='Folder with BF z_*.csv files')
    parser.add_argument('--threshold', type=float, default=5.0, help='XY clustering threshold in pixels (default 5.0)')
    parser.add_argument('--output', help='Output CSV path (default: bf_folder/bf_unique_diameter_summary.csv)')
    args = parser.parse_args()

    if not os.path.isdir(args.bf_folder):
        print(f"Error: Folder not found: {args.bf_folder}")
        return

    df = summarize(args.bf_folder, args.threshold)
    out_path = args.output or os.path.join(args.bf_folder, 'bf_unique_diameter_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved summary to: {out_path}")
    if not df.empty:
        print(f"Clusters: {len(df)} | diameter range: {df['diameter_min'].min():.2f}-{df['diameter_max'].max():.2f} μm")


if __name__ == '__main__':
    main()


