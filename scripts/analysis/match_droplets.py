#!/usr/bin/env python3
"""
Match droplets between ground truth CSV and single layer CSV based on x,y coordinates.
Output matched data with diameter, ring width, and ground truth diameter.
"""

import csv
import math
import os
from pathlib import Path


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def match_droplets(ground_truth_file, single_layer_file, output_file, tolerance=10.0):
    """
    Match droplets between ground truth and single layer CSVs.
    
    Args:
        ground_truth_file: Path to ground truth CSV (max_diameter_droplets_*.csv)
        single_layer_file: Path to single layer CSV (z*.csv)
        output_file: Path to output CSV file
        tolerance: Maximum distance in pixels for matching (default: 10.0)
    """
    # Read ground truth data
    ground_truth_data = []
    with open(ground_truth_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth_data.append({
                'x': float(row['Center_X_px']),
                'y': float(row['Center_Y_px']),
                'diameter': float(row['Diameter_μm']),
                'slide': row.get('slide', '')
            })
    
    print(f"📊 Loaded {len(ground_truth_data)} droplets from ground truth file")
    
    # Read single layer data
    single_layer_data = []
    with open(single_layer_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            single_layer_data.append({
                'mask_id': int(row['Mask_ID']),
                'x': float(row['Center_X_px']),
                'y': float(row['Center_Y_px']),
                'diameter': float(row['Diameter_μm']),
                'ring_width': float(row['Ring_Width_μm']),
                'mean_intensity': float(row.get('Mean_Intensity', 0)),
                'area': float(row.get('Area_μm²', 0)),
                'circularity': float(row.get('Circularity', 0)),
                'dark_edge_diameter': float(row.get('Dark_Edge_Diameter_μm', 0)),
                'dark_ratio': float(row.get('Dark_Ratio', 0))
            })
    
    print(f"📊 Loaded {len(single_layer_data)} droplets from single layer file")
    
    # Match droplets
    matched_data = []
    unmatched_single = []
    used_ground_truth = set()
    
    for single in single_layer_data:
        best_match = None
        best_distance = float('inf')
        
        # Find closest ground truth droplet within tolerance
        for idx, gt in enumerate(ground_truth_data):
            if idx in used_ground_truth:
                continue
            
            distance = calculate_distance(
                single['x'], single['y'],
                gt['x'], gt['y']
            )
            
            if distance <= tolerance and distance < best_distance:
                best_distance = distance
                best_match = (idx, gt)
        
        if best_match:
            idx, gt = best_match
            matched_data.append({
                'Mask_ID': single['mask_id'],
                'Center_X_px': single['x'],
                'Center_Y_px': single['y'],
                'Distance_px': round(best_distance, 2),
                'Diameter_μm': single['diameter'],
                'Ring_Width_μm': single['ring_width'],
                'Ground_Truth_Diameter_μm': gt['diameter'],
                'Diameter_Difference_μm': round(single['diameter'] - gt['diameter'], 2),
                'Mean_Intensity': single['mean_intensity'],
                'Area_μm²': single['area'],
                'Circularity': single['circularity'],
                'Dark_Edge_Diameter_μm': single['dark_edge_diameter'],
                'Dark_Ratio': single['dark_ratio'],
                'Slide': gt['slide']
            })
            used_ground_truth.add(idx)
        else:
            unmatched_single.append(single)
    
    # Write matched data to output CSV
    if matched_data:
        fieldnames = [
            'Mask_ID', 'Center_X_px', 'Center_Y_px', 'Distance_px',
            'Diameter_μm', 'Ring_Width_μm', 'Ground_Truth_Diameter_μm',
            'Diameter_Difference_μm', 'Mean_Intensity', 'Area_μm²',
            'Circularity', 'Dark_Edge_Diameter_μm', 'Dark_Ratio', 'Slide'
        ]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matched_data)
        
        print(f"✅ Matched {len(matched_data)} droplets")
        print(f"📝 Output written to: {output_file}")
        print(f"⚠️  {len(unmatched_single)} droplets from single layer file could not be matched")
        print(f"⚠️  {len(ground_truth_data) - len(matched_data)} droplets from ground truth file were not matched")
        
        # Print statistics
        if matched_data:
            diameter_diffs = [abs(m['Diameter_Difference_μm']) for m in matched_data]
            print(f"\n📊 Statistics:")
            print(f"   Average diameter difference: {sum(diameter_diffs) / len(diameter_diffs):.2f} μm")
            print(f"   Max diameter difference: {max(diameter_diffs):.2f} μm")
            print(f"   Min diameter difference: {min(diameter_diffs):.2f} μm")
    else:
        print("❌ No matches found! Check coordinate systems and tolerance.")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Match droplets between ground truth and single layer CSV files'
    )
    parser.add_argument(
        'ground_truth',
        type=str,
        help='Path to ground truth CSV file (max_diameter_droplets_*.csv)'
    )
    parser.add_argument(
        'single_layer',
        type=str,
        help='Path to single layer CSV file (z*.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: matched_droplets_<timestamp>.csv)'
    )
    parser.add_argument(
        '-t', '--tolerance',
        type=float,
        default=10.0,
        help='Maximum distance in pixels for matching (default: 10.0)'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(args.single_layer).stem
        args.output = f"matched_droplets_{base_name}_{timestamp}.csv"
    
    # Check if files exist
    if not os.path.exists(args.ground_truth):
        print(f"❌ Error: Ground truth file not found: {args.ground_truth}")
        return
    
    if not os.path.exists(args.single_layer):
        print(f"❌ Error: Single layer file not found: {args.single_layer}")
        return
    
    # Run matching
    match_droplets(args.ground_truth, args.single_layer, args.output, args.tolerance)


if __name__ == '__main__':
    main()

