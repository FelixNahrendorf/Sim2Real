#!/usr/bin/env python3
"""
Histogram Comparison Tool
Compares two overall_statistics.csv files and creates overlay histogram plots
with average differences displayed in the legend.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path


def load_histogram_data(csv_file):
    """Load histogram data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        
        # Check if this is the detailed histogram format
        required_columns = ['metric', 'average', 'bin_center', 'frequency']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None


def extract_histogram_data(df, metric):
    """Extract histogram data for a specific metric"""
    metric_data = df[df['metric'] == metric].copy()
    
    if metric_data.empty:
        return None
    
    # Sort by bin_center to ensure correct order
    metric_data = metric_data.sort_values('bin_center')
    
    return {
        'bin_centers': metric_data['bin_center'].values,
        'frequencies': metric_data['frequency'].values,
        'bin_starts': metric_data['bin_start'].values,
        'bin_ends': metric_data['bin_end'].values,
        'average': metric_data['average'].iloc[0],
        'std': metric_data['std'].iloc[0],
        'total_samples': metric_data['total_samples'].iloc[0]
    }


def create_comparison_plot(data1, data2, metric, label1, label2, output_path):
    """Create comparison histogram plot for a specific metric with dual y-axes"""
    
    # Extract data for this metric
    hist1 = extract_histogram_data(data1, metric)
    hist2 = extract_histogram_data(data2, metric)
    
    if hist1 is None or hist2 is None:
        print(f"Warning: Could not find data for metric '{metric}' in one or both files")
        return False
    
    # Calculate absolute average difference
    avg_diff = abs(hist2['average'] - hist1['average'])
    delta_sign = "Δ"
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax2 = ax1.twinx()  # Create second y-axis
    
    # Calculate bin widths
    bin_widths1 = hist1['bin_ends'] - hist1['bin_starts']
    bin_widths2 = hist2['bin_ends'] - hist2['bin_starts']
    
    # Plot histograms on separate y-axes
    bars1 = ax1.bar(hist1['bin_centers'], hist1['frequencies'], 
                    width=bin_widths1, alpha=0.6, color='red', 
                    edgecolor='darkred', linewidth=1,
                    label=f'{label1} (n={hist1["total_samples"]})')
    
    bars2 = ax2.bar(hist2['bin_centers'], hist2['frequencies'], 
                    width=bin_widths2, alpha=0.6, color='blue', 
                    edgecolor='darkblue', linewidth=1,
                    label=f'{label2} (n={hist2["total_samples"]})')
    
    # Add average lines
    y1_max = max(hist1['frequencies']) if len(hist1['frequencies']) > 0 else 1
    y2_max = max(hist2['frequencies']) if len(hist2['frequencies']) > 0 else 1
    
    ax1.axvline(hist1['average'], color='darkred', linestyle='--', linewidth=3,
                label=f'{label1} Avg: {hist1["average"]:.2f}')
    ax2.axvline(hist2['average'], color='darkblue', linestyle='--', linewidth=3,
                label=f'{label2} Avg: {hist2["average"]:.2f}')
    
    # Format metric title
    metric_title = metric.replace('_', ' ').title()
    
    # Set title and labels
    ax1.set_title(f'{metric_title} Distribution Comparison\n'
                 f'{delta_sign} = {avg_diff:.2f} | {label1} Avg: {hist1["average"]:.2f} | {label2} Avg: {hist2["average"]:.2f}', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel(f'{label1} Frequency', fontsize=12, color='darkred')
    ax2.set_ylabel(f'{label2} Frequency', fontsize=12, color='darkblue')
    
    # Color the y-axis labels
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax2.tick_params(axis='y', labelcolor='darkblue')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Add delta information to legend
    legend_elements = lines1 + lines2 + [
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=0,
                   label=f'{delta_sign} = {avg_diff:.2f}')
    ]
    legend_labels = labels1 + labels2 + [f'{delta_sign} = {avg_diff:.2f}']
    
    ax1.legend(legend_elements, legend_labels, loc='best', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True


def create_comparison_plots(csv_file1, csv_file2, output_dir, label1="Dataset 1", label2="Dataset 2"):
    """Create comparison plots for all metrics"""
    
    # Load data
    print(f"Loading data from {csv_file1}...")
    data1 = load_histogram_data(csv_file1)
    if data1 is None:
        return False
    
    print(f"Loading data from {csv_file2}...")
    data2 = load_histogram_data(csv_file2)
    if data2 is None:
        return False
    
    # Get list of metrics
    metrics1 = set(data1['metric'].unique())
    metrics2 = set(data2['metric'].unique())
    common_metrics = metrics1.intersection(metrics2)
    
    if not common_metrics:
        print("Error: No common metrics found between the two files")
        return False
    
    print(f"Found {len(common_metrics)} common metrics: {sorted(common_metrics)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create plots for each metric
    success_count = 0
    for metric in sorted(common_metrics):
        print(f"Creating comparison plot for {metric}...")
        
        output_file = output_path / f"comparison_{metric}.png"
        
        if create_comparison_plot(data1, data2, metric, label1, label2, output_file):
            print(f"  Saved: {output_file}")
            success_count += 1
        else:
            print(f"  Failed to create plot for {metric}")
    
    # Create combined plot with all metrics
    print("Creating combined comparison plot...")
    create_combined_comparison_plot(data1, data2, label1, label2, 
                                   output_path / "comparison_all_metrics.png")
    
    print(f"\nComparison complete! Created {success_count} individual plots and 1 combined plot.")
    print(f"Results saved to: {output_path}")
    
    return True


def create_combined_comparison_plot(data1, data2, label1, label2, output_path):
    """Create a combined plot with all metrics in subplots using dual y-axes"""
    
    metrics = ['brightness', 'sharpness', 'vignetting', 'compression_artifacts', 'noise', 'chromatic_aberration']
    metric_titles = ['Brightness', 'Sharpness', 'Vignetting', 'Compression Artifacts', 'Noise', 'Chromatic Aberration']
    
    # Filter to only metrics that exist in both datasets
    available_metrics = []
    available_titles = []
    for metric, title in zip(metrics, metric_titles):
        if metric in data1['metric'].values and metric in data2['metric'].values:
            available_metrics.append(metric)
            available_titles.append(title)
    
    if not available_metrics:
        print("No common metrics found for combined plot")
        return
    
    # Create subplots with extra spacing
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))
    fig.suptitle(f'Image Quality Metrics Comparison: {label1} vs {label2}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, (metric, metric_title) in enumerate(zip(available_metrics, available_titles)):
        if i >= len(axes_flat):
            break
            
        ax1 = axes_flat[i]
        ax2 = ax1.twinx()  # Create second y-axis for each subplot
        
        # Extract data
        hist1 = extract_histogram_data(data1, metric)
        hist2 = extract_histogram_data(data2, metric)
        
        if hist1 is None or hist2 is None:
            ax1.text(0.5, 0.5, f'No data for {metric_title}', 
                    ha='center', va='center', transform=ax1.transAxes)
            continue
        
        # Calculate absolute average difference
        avg_diff = abs(hist2['average'] - hist1['average'])
        
        # Calculate bin widths
        bin_widths1 = hist1['bin_ends'] - hist1['bin_starts']
        bin_widths2 = hist2['bin_ends'] - hist2['bin_starts']
        
        # Plot histograms on separate y-axes
        ax1.bar(hist1['bin_centers'], hist1['frequencies'], 
                width=bin_widths1, alpha=0.6, color='red', 
                edgecolor='darkred', linewidth=1)
        
        ax2.bar(hist2['bin_centers'], hist2['frequencies'], 
                width=bin_widths2, alpha=0.6, color='blue', 
                edgecolor='darkblue', linewidth=1)
        
        # Add average lines
        ax1.axvline(hist1['average'], color='darkred', linestyle='--', linewidth=2)
        ax2.axvline(hist2['average'], color='darkblue', linestyle='--', linewidth=2)
        
        # Set title with averages and delta
        title_text = (f'{metric_title}\n'
                     f'Avg: {hist1["average"]:.2f} | {hist2["average"]:.2f}\n'
                     f'Δ = {avg_diff:.2f}')
        ax1.set_title(title_text, fontweight='bold', fontsize=10)
        
        # Set labels
        ax1.set_xlabel('Value', fontsize=9)
        ax1.set_ylabel(f'{label1}', fontsize=9, color='darkred')
        ax2.set_ylabel(f'{label2}', fontsize=9, color='darkblue')
        
        # Color the y-axis labels
        ax1.tick_params(axis='y', labelcolor='darkred', labelsize=8)
        ax2.tick_params(axis='y', labelcolor='darkblue', labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        
        # Add grid
        ax1.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(available_metrics), len(axes_flat)):
        axes_flat[i].remove()
    
    # Create comprehensive legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.6, edgecolor='darkred',
                     label=f'{label1} (Histograms & Left Y-axis)'),
        plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.6, edgecolor='darkblue',
                     label=f'{label2} (Histograms & Right Y-axis)'),
        plt.Line2D([0], [0], color='darkred', linestyle='--', linewidth=2,
                   label=f'{label1} Averages (Red lines)'),
        plt.Line2D([0], [0], color='darkblue', linestyle='--', linewidth=2,
                   label=f'{label2} Averages (Blue lines)'),
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=0,
                   label='Δ = Absolute difference between averages')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
               ncol=3, fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout with extra spacing between title and plots, and between rows
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.10,     # Space for legend at bottom
        top=0.90,        # Space between title and first row of plots
        hspace=0.28,     # Vertical space between rows of plots
        wspace=0.25      # Horizontal space between columns
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved combined plot: {output_path}")
    
    # Print summary statistics
    print(f"\n📊 Summary of Average Differences:")
    for metric in available_metrics:
        hist1 = extract_histogram_data(data1, metric)
        hist2 = extract_histogram_data(data2, metric)
        if hist1 and hist2:
            avg_diff = abs(hist2['average'] - hist1['average'])
            metric_name = metric.replace('_', ' ').title()
            print(f"  {metric_name:20s}: Δ = {avg_diff:6.2f} ({hist1['average']:6.2f} vs {hist2['average']:6.2f})")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare two histogram CSV files and create overlay plots')
    
    parser.add_argument('csv_file1', help='First CSV file (overall_statistics.csv format)')
    parser.add_argument('csv_file2', help='Second CSV file (overall_statistics.csv format)')
    parser.add_argument('output_dir', help='Output directory for comparison plots')
    parser.add_argument('--label1', default='Dataset 1', help='Label for first dataset (default: Dataset 1)')
    parser.add_argument('--label2', default='Dataset 2', help='Label for second dataset (default: Dataset 2)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate input files
    if not Path(args.csv_file1).exists():
        print(f"Error: File {args.csv_file1} does not exist!")
        sys.exit(1)
    
    if not Path(args.csv_file2).exists():
        print(f"Error: File {args.csv_file2} does not exist!")
        sys.exit(1)
    
    print("Histogram Comparison Tool")
    print("=" * 50)
    print(f"Comparing:")
    print(f"  {args.label1}: {args.csv_file1}")
    print(f"  {args.label2}: {args.csv_file2}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Create comparison plots
    success = create_comparison_plots(
        args.csv_file1, 
        args.csv_file2, 
        args.output_dir,
        args.label1,
        args.label2
    )
    
    if success:
        print("\n✅ Comparison completed successfully!")
    else:
        print("\n❌ Comparison failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()