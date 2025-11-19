#!/usr/bin/env python3
"""
Visualize Z-Stack Process
Creates a visualization showing how individual z-planes combine into merged images
"""

import cv2
import numpy as np
from pathlib import Path


def create_zstack_montage():
    """Create a montage showing selected z-planes"""
    
    image_dir = Path('images')
    
    # Select representative z-planes (every 4th plane)
    selected_planes = [0, 4, 8, 12, 16, 19]
    
    images = []
    for z_idx in selected_planes:
        pattern = f"*_z{z_idx:02d}_*.jpg"
        files = list(image_dir.glob(pattern))
        if files:
            img = cv2.imread(str(files[0]))
            if img is not None:
                # Resize for montage
                small = cv2.resize(img, (256, 256))
                
                # Add label
                font = cv2.FONT_HERSHEY_SIMPLEX
                label = f"z{z_idx:02d}"
                cv2.putText(small, label, (10, 30), font, 0.8, (255, 255, 255), 2)
                cv2.putText(small, label, (10, 30), font, 0.8, (0, 255, 0), 1)
                
                images.append(small)
                print(f"✓ Added z-plane {z_idx}")
    
    if len(images) != 6:
        print("❌ Could not load all required z-planes")
        return
    
    # Create 2x3 grid
    row1 = np.hstack(images[0:3])
    row2 = np.hstack(images[3:6])
    montage = np.vstack([row1, row2])
    
    # Add title
    title_height = 60
    title_bar = np.zeros((title_height, montage.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (40, 40, 40)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Z-Stack Focal Planes (6 of 20 shown)"
    text_size = cv2.getTextSize(title, font, 1.0, 2)[0]
    text_x = (title_bar.shape[1] - text_size[0]) // 2
    cv2.putText(title_bar, title, (text_x, 40), font, 1.0, (255, 255, 255), 2)
    
    # Combine title with montage
    result = np.vstack([title_bar, montage])
    
    # Add border
    result = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(100, 100, 100))
    
    output_path = 'zstack_focal_planes.jpg'
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\n✅ Z-stack montage saved: {output_path}")
    print(f"   Size: {result.shape[1]}x{result.shape[0]}")


def create_process_diagram():
    """Create a diagram showing the merging process"""
    
    # Load a few z-planes
    image_dir = Path('images')
    selected = [5, 10, 15]  # Low, middle, high focus
    
    planes = []
    for z_idx in selected:
        pattern = f"*_z{z_idx:02d}_*.jpg"
        files = list(image_dir.glob(pattern))
        if files:
            img = cv2.imread(str(files[0]))
            if img is not None:
                small = cv2.resize(img, (200, 200))
                planes.append((z_idx, small))
    
    if len(planes) != 3:
        print("⚠️  Could not create process diagram")
        return
    
    # Create canvas
    canvas_width = 1200
    canvas_height = 400
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Add input planes on left
    y_start = 100
    for i, (z_idx, img) in enumerate(planes):
        y_pos = y_start + i * 100 - 100
        if y_pos < 0:
            y_pos = 20
        if y_pos + 200 > canvas_height:
            y_pos = canvas_height - 220
        
        canvas[y_pos:y_pos+200, 20:220] = img
        label = f"z{z_idx:02d}"
        cv2.putText(canvas, label, (25, y_pos + 30), font, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, label, (25, y_pos + 30), font, 0.6, (0, 100, 255), 1)
    
    # Add arrows
    arrow_color = (0, 0, 0)
    cv2.arrowedLine(canvas, (240, 200), (340, 200), arrow_color, 3, tipLength=0.2)
    
    # Add "+" symbols
    cv2.putText(canvas, "+", (270, 180), font, 1.5, arrow_color, 3)
    
    # Add method labels in center
    methods = [
        ("Maximum Intensity", 360, 80),
        ("Focus Stacking", 360, 200),
        ("Average", 360, 320)
    ]
    
    for method, x, y in methods:
        cv2.putText(canvas, method, (x, y), font, 0.6, (0, 0, 0), 2)
        cv2.putText(canvas, method, (x, y), font, 0.6, (0, 100, 200), 1)
    
    # Add output arrows
    cv2.arrowedLine(canvas, (560, 200), (660, 200), arrow_color, 3, tipLength=0.2)
    
    # Load and add merged result
    mip = cv2.imread('merged_zstack_MIP.jpg')
    if mip is not None:
        result_img = cv2.resize(mip, (300, 300))
        y_pos = (canvas_height - 300) // 2
        canvas[y_pos:y_pos+300, 700:1000] = result_img
        
        label = "Merged Result"
        cv2.putText(canvas, label, (720, y_pos - 10), font, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, label, (720, y_pos - 10), font, 0.8, (0, 150, 0), 1)
        
        stats = "(All 20 planes)"
        cv2.putText(canvas, stats, (750, y_pos + 320), font, 0.5, (0, 0, 0), 1)
    
    # Add title
    title = "Z-Stack Merging Process"
    cv2.putText(canvas, title, (20, 35), font, 1.2, (0, 0, 0), 3)
    cv2.putText(canvas, title, (20, 35), font, 1.2, (0, 100, 200), 2)
    
    output_path = 'zstack_process_diagram.jpg'
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"✅ Process diagram saved: {output_path}")


def create_focus_plot():
    """Create a visualization of focus quality across z-planes"""
    
    # Focus measures from the EDF output (captured during merge)
    focus_data = [
        (0, 30467124.57),
        (1, 34981530.35),
        (2, 40936189.76),
        (3, 50615630.32),
        (4, 61195264.95),
        (5, 71761416.62),
        (6, 80599537.67),
        (7, 86690916.52),
        (8, 90496802.37),
        (9, 91001053.02),  # Peak focus
        (10, 88096389.33),
        (11, 82773883.42),
        (12, 72706176.57),
        (13, 60678113.50),
        (14, 53833447.57),
        (15, 46998777.06),
        (16, 39095564.72),
        (17, 31691619.28),
        (18, 26448977.96),
        (19, 23596773.37)
    ]
    
    # Create plot canvas
    width = 800
    height = 400
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Normalize focus values to canvas height
    focus_values = [f[1] for f in focus_data]
    max_focus = max(focus_values)
    min_focus = min(focus_values)
    
    # Calculate plot area
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    
    # Draw axes
    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)
    cv2.line(canvas, (margin, margin), (margin, height - margin), (0, 0, 0), 2)
    
    # Plot focus curve
    points = []
    for i, (z, focus) in enumerate(focus_data):
        x = margin + int((i / 19) * plot_width)
        normalized_focus = (focus - min_focus) / (max_focus - min_focus)
        y = height - margin - int(normalized_focus * plot_height)
        points.append((x, y))
    
    # Draw curve
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i + 1], (0, 100, 255), 3)
    
    # Draw points
    for i, (x, y) in enumerate(points):
        if i == 9:  # Peak focus at z09
            cv2.circle(canvas, (x, y), 6, (0, 0, 255), -1)
            cv2.circle(canvas, (x, y), 8, (0, 0, 0), 2)
        else:
            cv2.circle(canvas, (x, y), 4, (0, 100, 255), -1)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    title = "Focus Quality Across Z-Stack"
    cv2.putText(canvas, title, (width // 2 - 150, 30), font, 0.8, (0, 0, 0), 2)
    
    # X-axis label
    cv2.putText(canvas, "Z-Plane Index", (width // 2 - 60, height - 10), font, 0.6, (0, 0, 0), 1)
    
    # Y-axis label
    cv2.putText(canvas, "Focus", (5, height // 2), font, 0.6, (0, 0, 0), 1)
    
    # Peak annotation
    peak_x, peak_y = points[9]
    cv2.putText(canvas, "Peak at z09", (peak_x - 40, peak_y - 15), font, 0.5, (255, 0, 0), 1)
    
    # X-axis tick labels
    for i in [0, 5, 10, 15, 19]:
        x = margin + int((i / 19) * plot_width)
        cv2.putText(canvas, f"z{i}", (x - 10, height - margin + 20), font, 0.4, (0, 0, 0), 1)
        cv2.line(canvas, (x, height - margin), (x, height - margin + 5), (0, 0, 0), 1)
    
    output_path = 'zstack_focus_plot.jpg'
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"✅ Focus plot saved: {output_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("Creating Z-Stack Visualizations")
    print("=" * 60)
    print()
    
    create_zstack_montage()
    print()
    create_process_diagram()
    print()
    create_focus_plot()
    
    print()
    print("=" * 60)
    print("✅ All visualizations created!")
    print("=" * 60)

