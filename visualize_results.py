#!/usr/bin/env python3
"""
Visualize the droplet maximum diameter analysis results.
Creates a bar chart showing the distribution of max diameter droplets across slides.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations():
    # Read the results
    df = pd.read_csv('/data3/megan_data/Jimmy/max_diameter_droplets.csv')
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of max diameter droplets by slide
    slide_counts = df['slide'].value_counts().sort_index()
    
    # Extract z-level numbers for proper sorting
    slide_numbers = [int(s.split('_')[1]) for s in slide_counts.index]
    sorted_indices = sorted(range(len(slide_numbers)), key=lambda i: slide_numbers[i])
    sorted_slides = [slide_counts.index[i] for i in sorted_indices]
    sorted_counts = [slide_counts.iloc[i] for i in sorted_indices]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(sorted_slides)), sorted_counts, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Z-Slice', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Droplets with Max Diameter', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Max Diameter Droplets Across Z-Slices', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(sorted_slides)))
    ax1.set_xticklabels(sorted_slides, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Diameter distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(df['Diameter_μm'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Diameter (μm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Maximum Diameters', fontsize=14, fontweight='bold')
    ax2.axvline(df['Diameter_μm'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Diameter_μm"].mean():.2f} μm')
    ax2.axvline(df['Diameter_μm'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Diameter_μm"].median():.2f} μm')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Cumulative percentage by slide
    ax3 = axes[1, 0]
    cumulative_counts = [sorted_counts[0]]
    for i in range(1, len(sorted_counts)):
        cumulative_counts.append(cumulative_counts[-1] + sorted_counts[i])
    cumulative_pct = [c / cumulative_counts[-1] * 100 for c in cumulative_counts]
    
    ax3.plot(range(len(sorted_slides)), cumulative_pct, marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax3.set_xlabel('Z-Slice', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution of Max Diameter Droplets', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(sorted_slides)))
    ax3.set_xticklabels(sorted_slides, rotation=45, ha='right')
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 105])
    
    # Add percentage labels
    for i, pct in enumerate(cumulative_pct):
        ax3.text(i, pct + 2, f'{pct:.1f}%', ha='center', fontsize=8)
    
    # 4. Box plot of diameters by slide
    ax4 = axes[1, 1]
    
    # Prepare data for box plot
    slide_data = []
    slide_labels = []
    for slide in sorted_slides:
        slide_df = df[df['slide'] == slide]
        if len(slide_df) > 0:
            slide_data.append(slide_df['Diameter_μm'].values)
            slide_labels.append(slide)
    
    bp = ax4.boxplot(slide_data, labels=slide_labels, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_edgecolor('black')
    
    ax4.set_xlabel('Z-Slice', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Diameter (μm)', fontsize=12, fontweight='bold')
    ax4.set_title('Diameter Distribution by Z-Slice', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = '/data3/megan_data/Jimmy/droplet_analysis_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total unique droplets: {len(df)}")
    print(f"Average diameter: {df['Diameter_μm'].mean():.2f} μm")
    print(f"Std deviation: {df['Diameter_μm'].std():.2f} μm")
    print(f"Min diameter: {df['Diameter_μm'].min():.2f} μm")
    print(f"25th percentile: {df['Diameter_μm'].quantile(0.25):.2f} μm")
    print(f"Median diameter: {df['Diameter_μm'].median():.2f} μm")
    print(f"75th percentile: {df['Diameter_μm'].quantile(0.75):.2f} μm")
    print(f"Max diameter: {df['Diameter_μm'].max():.2f} μm")
    print("="*60)
    
    # Top 5 slides with most max diameter droplets
    print("\nTop 5 slides with most max diameter droplets:")
    top5 = slide_counts.nlargest(5)
    for slide, count in top5.items():
        pct = (count / len(df)) * 100
        print(f"  {slide}: {count} droplets ({pct:.1f}%)")

if __name__ == "__main__":
    try:
        create_visualizations()
        print("\nVisualization complete!")
    except ImportError as e:
        print(f"Warning: Could not create visualization: {e}")
        print("Please install matplotlib and seaborn if you want visualizations:")
        print("  pip install matplotlib seaborn")

