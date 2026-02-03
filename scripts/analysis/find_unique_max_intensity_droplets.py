#!/usr/bin/env python3

"""
Unique droplet analysis across z-level CSVs.

Reads all CSV files in a directory (e.g., z_0.csv, z_1.csv, ...), treats droplets
observed across z-levels as the same droplet if their (x, y) positions are within
±location_threshold in BOTH x and y, and reports the maximum intensity per unique droplet.

CSV format is inferred with flexible column names. Supported columns (case-insensitive):
  - X:       ["x", "center_x", "center_x_px", "centerx", "cx"]
  - Y:       ["y", "center_y", "center_y_px", "centery", "cy"]
  - Intensity: ["mean_intensity", "intensity", "mean", "avg_intensity", "meanintensity"]

Output: unique_max_intensity_droplets.csv in the same directory.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Flexible column name mapping (lowercased)
X_CANDIDATES = ["x", "center_x", "centerx", "center_x_px", "cx"]
Y_CANDIDATES = ["y", "center_y", "centery", "center_y_px", "cy"]
I_CANDIDATES = [
    "mean_intensity",
    "intensity",
    "mean",
    "avg_intensity",
    "meanintensity",
]


def _find_col(header_lower: List[str], candidates: List[str]) -> Optional[int]:
    for cand in candidates:
        if cand in header_lower:
            return header_lower.index(cand)
    return None


def _parse_z_from_filename(filename: str) -> Optional[int]:
    # Expect patterns like z_0.csv, z-12.csv, Z_3.csv, etc.
    # Updated to match 'z' at start of filename or after a directory separator
    m = re.search(r"(?:^|[\\/_])z[_-]?(\d+)\.[Cc][Ss][Vv]$", filename, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


@dataclass
class Droplet:
    x: float
    y: float
    intensity: float
    z: int
    source_file: str


@dataclass
class DropletCluster:
    id: int
    # Representative location (running average)
    rep_x: float
    rep_y: float
    droplets: List[Droplet] = field(default_factory=list)

    def add(self, d: Droplet) -> None:
        self.droplets.append(d)
        # Update representative location as running average for stability
        n = len(self.droplets)
        self.rep_x = self.rep_x + (d.x - self.rep_x) / n
        self.rep_y = self.rep_y + (d.y - self.rep_y) / n

    def within_threshold(self, x: float, y: float, threshold: float) -> bool:
        return abs(self.rep_x - x) <= threshold and abs(self.rep_y - y) <= threshold

    def max_intensity_record(self) -> Tuple[float, int, str]:
        """Returns (max_intensity, z_at_max, source_file_at_max)."""
        if not self.droplets:
            return 0.0, -1, ""
        best = max(self.droplets, key=lambda d: d.intensity)
        return best.intensity, best.z, best.source_file

    def avg_xy(self) -> Tuple[float, float]:
        if not self.droplets:
            return self.rep_x, self.rep_y
        sx = sum(d.x for d in self.droplets)
        sy = sum(d.y for d in self.droplets)
        n = len(self.droplets)
        return sx / n, sy / n

    def z_range(self) -> Tuple[int, int]:
        if not self.droplets:
            return -1, -1
        zs = [d.z for d in self.droplets]
        return min(zs), max(zs)


class UniqueDropletAnalyzer:
    def __init__(self, csv_directory: str, location_threshold: float = 5.0):
        self.csv_directory = csv_directory
        self.location_threshold = float(location_threshold)
        self._droplets: List[Droplet] = []
        self._clusters: List[DropletCluster] = []

    def _list_csv_files(self) -> List[str]:
        files = []
        for name in sorted(os.listdir(self.csv_directory)):
            if name.lower().endswith(".csv"):
                files.append(os.path.join(self.csv_directory, name))
        return files

    def _read_csv_droplets(self, path: str) -> List[Droplet]:
        droplets: List[Droplet] = []
        z_guess = _parse_z_from_filename(path)
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return droplets

            header_lower = [h.strip().lower() for h in header]
            ix = _find_col(header_lower, X_CANDIDATES)
            iy = _find_col(header_lower, Y_CANDIDATES)
            ii = _find_col(header_lower, I_CANDIDATES)
            # If no explicit intensity column, attempt fallback to a single numeric column named 'diameter' etc? skip
            if ix is None or iy is None or ii is None:
                # Try a more permissive approach: if an 'id' column is present, still require x,y,intensity
                return droplets

            for row in reader:
                if len(row) <= max(ix, iy, ii):
                    continue
                try:
                    x = float(row[ix])
                    y = float(row[iy])
                    intensity = float(row[ii])
                except ValueError:
                    continue

                z = z_guess if z_guess is not None else -1
                droplets.append(Droplet(x=x, y=y, intensity=intensity, z=z, source_file=os.path.basename(path)))

        return droplets

    def _cluster_droplets(self) -> None:
        self._clusters = []
        cluster_id = 1
        for d in self._droplets:
            # Try to match an existing cluster within threshold
            assigned = False
            for cl in self._clusters:
                if cl.within_threshold(d.x, d.y, self.location_threshold):
                    cl.add(d)
                    assigned = True
                    break
            if not assigned:
                new_cl = DropletCluster(id=cluster_id, rep_x=d.x, rep_y=d.y, droplets=[d])
                self._clusters.append(new_cl)
                cluster_id += 1

    def find_unique_max_intensity_droplets(self) -> List[Dict[str, object]]:
        """Returns list of dicts with per-unique-droplet max intensity summary."""
        self._droplets = []
        for csv_path in self._list_csv_files():
            self._droplets.extend(self._read_csv_droplets(csv_path))

        if not self._droplets:
            return []

        self._cluster_droplets()

        results: List[Dict[str, object]] = []
        for cl in self._clusters:
            max_int, z_at_max, file_at_max = cl.max_intensity_record()
            avg_x, avg_y = cl.avg_xy()
            zmin, zmax = cl.z_range()
            results.append({
                "cluster_id": cl.id,
                "rep_x": round(cl.rep_x, 2),
                "rep_y": round(cl.rep_y, 2),
                "avg_x": round(avg_x, 2),
                "avg_y": round(avg_y, 2),
                "num_observations": len(cl.droplets),
                "max_intensity": round(max_int, 4),
                "z_at_max": z_at_max,
                "file_at_max": file_at_max,
                "z_min": zmin,
                "z_max": zmax,
            })

        # Sort results by max intensity descending
        results.sort(key=lambda r: r["max_intensity"], reverse=True)
        self._results_cache = results
        return results

    def get_statistics(self) -> Dict[str, object]:
        total = len(self._droplets)
        unique = len(self._clusters)
        duplicates_removed = total - unique
        max_int = max((r["max_intensity"] for r in getattr(self, "_results_cache", [])), default=0.0)
        # Count clusters with more than one observation
        multi = sum(1 for cl in self._clusters if len(cl.droplets) > 1)
        return {
            "total_droplets": total,
            "unique_locations": unique,
            "duplicates_removed": duplicates_removed,
            "max_intensity": max_int,
            "locations_with_multiple_droplets": multi,
        }

    def print_results(self, top_n: int = 15) -> None:
        results = getattr(self, "_results_cache", [])
        if not results:
            print("No results to display.")
            return
        print("Top results (by max intensity):")
        for r in results[:top_n]:
            print(
                f"  id={r['cluster_id']:4d}  maxI={r['max_intensity']:.4f}  "
                f"(x,y)=({r['rep_x']:.2f},{r['rep_y']:.2f})  z@max={r['z_at_max']}  obs={r['num_observations']}  file={r['file_at_max']}"
            )

    def save_results_to_csv(self, output_path: str) -> None:
        results = getattr(self, "_results_cache", [])
        if not results:
            return
        fieldnames = [
            "cluster_id",
            "rep_x",
            "rep_y",
            "avg_x",
            "avg_y",
            "num_observations",
            "max_intensity",
            "z_at_max",
            "file_at_max",
            "z_min",
            "z_max",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

#!/usr/bin/env python3
"""
Find Unique Maximum Intensity Droplets Across Z-levels

This script analyzes CSV files from different Z-levels to find all unique droplets.
For droplets at the same location (within 5 pixels), only keeps the one with highest intensity.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
import argparse


class UniqueDropletAnalyzer:
    """Analyze all droplets and keep only unique locations with maximum intensity"""
    
    def __init__(self, csv_directory: str, location_threshold: float = 5.0):
        """
        Initialize the analyzer
        
        Args:
            csv_directory: Directory containing CSV files
            location_threshold: Maximum distance in pixels to consider droplets as "same location"
        """
        self.csv_directory = csv_directory
        self.location_threshold = location_threshold
        self.all_droplets = []  # Will store all droplets from all z-levels
        self.unique_droplets = []  # Will store unique droplets with max intensity
        
    def load_all_droplets(self) -> List[Dict]:
        """Load all droplets from all CSV files"""
        csv_files = glob.glob(os.path.join(self.csv_directory, "*.csv"))
        
        # Filter to only z-level files (z_0.csv, z_1.csv, z00.csv, z01.csv, etc.)
        import re
        z_files = [f for f in csv_files if re.match(r'z[_-]?\d+\.csv$', os.path.basename(f), re.IGNORECASE)]
        z_files.sort()  # Sort to ensure consistent z-level ordering
        
        print(f"Found {len(z_files)} z-level CSV files:")
        for file in z_files:
            print(f"  - {os.path.basename(file)}")
        
        all_droplets = []
        
        for csv_file in z_files:
            z_level = os.path.splitext(os.path.basename(csv_file))[0]
            try:
                df = pd.read_csv(csv_file)
                # Find column names flexibly (handle _px suffix, μm suffix, etc.)
                cols = df.columns.tolist()
                x_col = next((c for c in cols if 'center_x' in c.lower()), None)
                y_col = next((c for c in cols if 'center_y' in c.lower()), None)
                intensity_col = next((c for c in cols if 'intensity' in c.lower()), None)
                diameter_col = next((c for c in cols if 'diameter' in c.lower()), None)
                area_col = next((c for c in cols if 'area' in c.lower()), None)
                circ_col = next((c for c in cols if 'circularity' in c.lower()), None)
                mask_col = next((c for c in cols if 'mask' in c.lower() or c.lower() == 'id'), None)
                
                if x_col and y_col and intensity_col:
                    # Add each droplet with z-level information
                    for idx, row in df.iterrows():
                        droplet = {
                            'z_level': z_level,
                            'mask_id': row[mask_col] if mask_col else idx,
                            'center_x': row[x_col],
                            'center_y': row[y_col],
                            'diameter': row[diameter_col] if diameter_col else None,
                            'mean_intensity': row[intensity_col],
                            'area': row[area_col] if area_col else None,
                            'circularity': row[circ_col] if circ_col else None
                        }
                        all_droplets.append(droplet)
                    
                    print(f"Loaded {z_level}: {len(df)} droplets")
                else:
                    print(f"Skipping {z_level}: Missing required columns (X, Y, or Intensity)")
                    print(f"  Available columns: {cols}")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        self.all_droplets = all_droplets
        return all_droplets
    
    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def find_unique_max_intensity_droplets(self) -> List[Dict]:
        """
        Find unique droplets by location and keep only the one with maximum intensity
        
        Returns:
            List of unique droplets with maximum intensity at each location
        """
        if not self.all_droplets:
            self.load_all_droplets()
        
        print(f"\nAnalyzing {len(self.all_droplets)} total droplets across all z-levels...")
        print(f"Location threshold: {self.location_threshold} pixels")
        
        unique_droplets = []
        processed_indices = set()
        
        for i, droplet in enumerate(self.all_droplets):
            if i in processed_indices:
                continue
            
            # Find all droplets at the same location (within threshold)
            same_location_droplets = [droplet]
            same_location_indices = [i]
            
            for j, other_droplet in enumerate(self.all_droplets[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                distance = self.calculate_distance(
                    droplet['center_x'], droplet['center_y'],
                    other_droplet['center_x'], other_droplet['center_y']
                )
                
                if distance <= self.location_threshold:
                    same_location_droplets.append(other_droplet)
                    same_location_indices.append(j)
            
            # Find the droplet with maximum intensity at this location
            max_intensity_droplet = max(same_location_droplets, key=lambda d: d['mean_intensity'])
            
            # Add information about how many droplets were at this location
            max_intensity_droplet['droplets_at_location'] = len(same_location_droplets)
            max_intensity_droplet['z_levels_at_location'] = [d['z_level'] for d in same_location_droplets]
            max_intensity_droplet['all_intensities'] = [d['mean_intensity'] for d in same_location_droplets]
            
            unique_droplets.append(max_intensity_droplet)
            
            # Mark all these indices as processed
            processed_indices.update(same_location_indices)
        
        # Sort by intensity (descending)
        unique_droplets.sort(key=lambda x: x['mean_intensity'], reverse=True)
        
        self.unique_droplets = unique_droplets
        return unique_droplets
    
    def print_results(self, top_n: int = 20):
        """
        Print results of unique droplet analysis
        
        Args:
            top_n: Number of top droplets to display
        """
        if not self.unique_droplets:
            print("No unique droplets found!")
            return
        
        print(f"\n{'='*80}")
        print(f"UNIQUE DROPLET ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(f"Total droplets across all z-levels: {len(self.all_droplets)}")
        print(f"Unique locations found: {len(self.unique_droplets)}")
        print(f"Location threshold: {self.location_threshold} pixels")
        print(f"Reduction: {len(self.all_droplets) - len(self.unique_droplets)} duplicate locations removed")
        print(f"\nTop {min(top_n, len(self.unique_droplets))} droplets by maximum intensity:")
        print(f"{'-'*80}")
        
        for i, droplet in enumerate(self.unique_droplets[:top_n]):
            print(f"\nRank #{i+1}")
            print(f"  Location: ({droplet['center_x']:.1f}, {droplet['center_y']:.1f})")
            print(f"  Max Intensity: {droplet['mean_intensity']:.2f} (from {droplet['z_level']})")
            print(f"  Diameter: {droplet['diameter']:.2f}")
            print(f"  Area: {droplet['area']:.2f}")
            print(f"  Circularity: {droplet['circularity']:.3f}")
            print(f"  Droplets at this location: {droplet['droplets_at_location']}")
            print(f"  Z-levels: {', '.join(sorted(droplet['z_levels_at_location']))}")
            
            if droplet['droplets_at_location'] > 1:
                intensities = [f"{z}:{intensity:.1f}" for z, intensity in 
                             zip(droplet['z_levels_at_location'], droplet['all_intensities'])]
                print(f"  All intensities: {', '.join(sorted(intensities))}")
    
    def save_results_to_csv(self, output_file: str):
        """
        Save unique droplet results to CSV file
        
        Args:
            output_file: Path to output CSV file
        """
        if not self.unique_droplets:
            print("No unique droplets to save!")
            return
        
        # Prepare data for CSV
        csv_data = []
        for i, droplet in enumerate(self.unique_droplets):
            row = {
                'Rank': i + 1,
                'Center_X': droplet['center_x'],
                'Center_Y': droplet['center_y'],
                'Max_Intensity': droplet['mean_intensity'],
                'Source_Z_Level': droplet['z_level'],
                'Diameter': droplet['diameter'],
                'Area': droplet['area'],
                'Circularity': droplet['circularity'],
                'Droplets_At_Location': droplet['droplets_at_location'],
                'Z_Levels_At_Location': ','.join(sorted(droplet['z_levels_at_location'])),
                'All_Intensities': ','.join([f"{intensity:.2f}" for intensity in droplet['all_intensities']])
            }
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Saved {len(csv_data)} unique droplets")
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        if not self.unique_droplets:
            return {}
        
        intensities = [d['mean_intensity'] for d in self.unique_droplets]
        droplet_counts = [d['droplets_at_location'] for d in self.unique_droplets]
        
        stats = {
            'total_droplets': len(self.all_droplets),
            'unique_locations': len(self.unique_droplets),
            'duplicates_removed': len(self.all_droplets) - len(self.unique_droplets),
            'max_intensity': max(intensities),
            'min_intensity': min(intensities),
            'mean_intensity': np.mean(intensities),
            'std_intensity': np.std(intensities),
            'max_droplets_at_location': max(droplet_counts),
            'locations_with_multiple_droplets': sum(1 for count in droplet_counts if count > 1)
        }
        
        return stats


def main():
    """Main function to run the unique droplet analysis"""
    parser = argparse.ArgumentParser(description='Find unique maximum intensity droplets across Z-levels')
    parser.add_argument('csv_directory', help='Directory containing CSV files')
    parser.add_argument('--threshold', '-t', type=float, default=5.0,
                       help='Location threshold in pixels (default: 5.0)')
    parser.add_argument('--top-n', '-n', type=int, default=20,
                       help='Number of top droplets to display (default: 20)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.csv_directory):
        print(f"Error: Directory '{args.csv_directory}' does not exist!")
        return
    
    # Create analyzer and run analysis
    analyzer = UniqueDropletAnalyzer(args.csv_directory, args.threshold)
    unique_droplets = analyzer.find_unique_max_intensity_droplets()
    
    if unique_droplets:
        # Print results
        analyzer.print_results(args.top_n)
        
        # Print statistics
        stats = analyzer.get_statistics()
        print(f"\n{'='*80}")
        print(f"ANALYSIS STATISTICS")
        print(f"{'='*80}")
        print(f"Total droplets: {stats['total_droplets']}")
        print(f"Unique locations: {stats['unique_locations']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")
        print(f"Intensity range: {stats['min_intensity']:.2f} - {stats['max_intensity']:.2f}")
        print(f"Mean intensity: {stats['mean_intensity']:.2f} ± {stats['std_intensity']:.2f}")
        print(f"Max droplets at one location: {stats['max_droplets_at_location']}")
        print(f"Locations with multiple droplets: {stats['locations_with_multiple_droplets']}")
        
        # Save to CSV if output file specified
        if args.output:
            analyzer.save_results_to_csv(args.output)
        else:
            # Auto-generate output filename
            output_file = os.path.join(args.csv_directory, 'unique_max_intensity_droplets.csv')
            analyzer.save_results_to_csv(output_file)
    else:
        print("No droplets found to analyze!")


if __name__ == "__main__":
    main()
