#!/usr/bin/env python3
"""
Camera Parameters Analysis Tool

Analyzes camera intrinsic and extrinsic parameters from JSON files and generates
comprehensive visualizations and statistics.

Usage:
    python3 camera_value_analysis.py <path_pattern> --output-dir <output_directory>
    
Example:
    python3 camera_value_analysis.py "/data/CAM_*.json" --output-dir results
    python3 camera_value_analysis.py "./cameras/CAM_*.json" --output-dir analysis_output
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
from pathlib import Path
import pandas as pd
from scipy import stats
import sys

def load_camera_data(file_path):
    """Load camera data from JSON file and extract parameters."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract coordinates (x, y, z)
        coords = np.array(data['coordinates'])
        x_coords = coords[:, 0]
        y_coords = coords[:, 1] 
        z_coords = coords[:, 2]
        
        # Extract other parameters
        pitches = np.array(data['pitchs'])
        yaws = np.array(data['yaws'])
        fovs = np.array(data['fov'])
        
        return {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords,
            'pitch': pitches,
            'yaw': yaws,
            'fov': fovs
        }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_statistics(data_dict):
    """Compute statistics for each parameter."""
    stats_dict = {}
    for param, values in data_dict.items():
        stats_dict[param] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    return stats_dict

def analyze_camera_files(file_pattern):
    """Analyze camera JSON files matching the pattern."""
    camera_data = {}
    camera_stats = {}
    
    # Find all files matching the pattern
    json_files = glob.glob(file_pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {file_pattern}")
        return None, None
    
    print(f"Found {len(json_files)} files matching pattern")
    
    for file_path in json_files:
        # Get filename without extension and remove "_converted" suffix
        camera_name = Path(file_path).stem
        if camera_name.endswith('_converted'):
            camera_name = camera_name[:-10]  # Remove "_converted"
        
        print(f"Processing: {camera_name}")
        
        data = load_camera_data(file_path)
        if data is not None:
            camera_data[camera_name] = data
            camera_stats[camera_name] = compute_statistics(data)
            print(f"✓ Loaded data for {camera_name}")
        else:
            print(f"✗ Failed to load {camera_name}")
    
    return camera_data, camera_stats

def create_visualization(camera_data, camera_stats, output_dir):
    """Create comprehensive visualization of camera parameters."""
    if not camera_data:
        print("No camera data to visualize")
        return
    
    # Define consistent camera order and colors
    camera_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    color_map = {
        'CAM_FRONT': 'red',
        'CAM_FRONT_LEFT': 'green', 
        'CAM_FRONT_RIGHT': 'blue',
        'CAM_BACK': 'orange',
        'CAM_BACK_LEFT': 'black',
        'CAM_BACK_RIGHT': 'yellow'
    }
    
    # Filter and order cameras based on what's available in the data
    available_cameras = [cam for cam in camera_order if cam in camera_data.keys()]
    
    # Set up the figure with subplots
    plt.style.use('default')  # Ensure consistent styling
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Camera Parameters', fontsize=16, fontweight='bold')
    
    # Parameters to plot
    params = ['x', 'y', 'z', 'pitch', 'yaw', 'fov']
    param_labels = ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 
                   'Pitch (rad)', 'Yaw (rad)', 'FOV (degrees)']
    
    # Plot each parameter
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i//3, i%3]
        
        # Plot distribution for each camera in consistent order
        for camera_name in available_cameras:
            data = camera_data[camera_name]
            values = data[param]
            color = color_map[camera_name]
            
            # Plot histogram
            ax.hist(values, bins=20, alpha=0.6, color=color, 
                   label=f'{camera_name}', density=True)
            
            # Plot mean line
            mean_val = camera_stats[camera_name][param]['mean']
            ax.axvline(mean_val, color=color, linestyle='--', 
                      linewidth=2, alpha=0.8)
        
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'{label} Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    png_path = output_dir / 'camera_parameters_distribution.png'
    svg_path = output_dir / 'camera_parameters_distribution.svg'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"✓ Distribution plot saved as '{png_path}' and '{svg_path}'")
    
    plt.show()

def create_individual_camera_plots(camera_data, camera_stats, output_dir):
    """Create individual plots for each camera showing absolute value distributions."""
    params = ['x', 'y', 'z', 'pitch', 'yaw', 'fov']
    param_labels = ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 
                   'Pitch (rad)', 'Yaw (rad)', 'FOV (degrees)']
    
    # Define consistent camera order and colors
    camera_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    color_map = {
        'CAM_FRONT': 'red',
        'CAM_FRONT_LEFT': 'green', 
        'CAM_FRONT_RIGHT': 'blue',
        'CAM_BACK': 'orange',
        'CAM_BACK_LEFT': 'black',
        'CAM_BACK_RIGHT': 'yellow'
    }
    
    for camera_name, data in camera_data.items():
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{camera_name} - Parameter Distributions (Absolute Values)', 
                    fontsize=14, fontweight='bold')
        
        camera_color = color_map.get(camera_name, 'gray')  # Use gray as fallback
        
        for j, (param, label) in enumerate(zip(params, param_labels)):
            ax = axes[j//3, j%3]
            
            # Get absolute values
            values = np.abs(data[param])
            stats = camera_stats[camera_name][param]
            
            # Create histogram
            n, bins, patches = ax.hist(values, bins=15, alpha=0.7, color=camera_color, 
                                     edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            mean_abs = np.mean(values)
            std_abs = np.std(values)
            
            stats_text = f'Mean: {mean_abs:.4f}\nStd: {std_abs:.4f}\nMin: {np.min(values):.4f}\nMax: {np.max(values):.4f}'
            ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add vertical line for mean
            ax.axvline(mean_abs, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_abs:.4f}')
            
            ax.set_xlabel(f'|{label}|')
            ax.set_ylabel('Frequency')
            ax.set_title(f'|{label}| Distribution')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save individual camera plots
        png_path = output_dir / f'{camera_name}_parameter_distributions.png'
        svg_path = output_dir / f'{camera_name}_parameter_distributions.svg'
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"✓ Individual plot for {camera_name} saved as '{png_path}' and '{svg_path}'")
        
        plt.close()  # Close to save memory

def create_box_plots(camera_data, camera_stats, output_dir):
    """Create box plots for parameter comparison across cameras."""
    params = ['x', 'y', 'z', 'pitch', 'yaw', 'fov']
    param_labels = ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 
                   'Pitch (rad)', 'Yaw (rad)', 'FOV (degrees)']
    
    # Define consistent camera order and colors
    camera_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    color_map = {
        'CAM_FRONT': 'red',
        'CAM_FRONT_LEFT': 'green', 
        'CAM_FRONT_RIGHT': 'blue',
        'CAM_BACK': 'orange',
        'CAM_BACK_LEFT': 'black',
        'CAM_BACK_RIGHT': 'yellow'
    }
    
    # Filter and order cameras based on what's available in the data
    available_cameras = [cam for cam in camera_order if cam in camera_data.keys()]
    
    plt.style.use('default')  # Ensure consistent styling
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Camera Parameters - Box Plots Comparison', fontsize=16, fontweight='bold')
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i//3, i%3]
        
        # Prepare data for box plot in consistent order
        data_for_boxplot = []
        labels = []
        colors = []
        
        for camera_name in available_cameras:
            data_for_boxplot.append(camera_data[camera_name][param])
            labels.append(camera_name)
            colors.append(color_map[camera_name])
        
        # Create box plot
        bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        
        # Color the boxes with consistent colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add statistics annotations
        for j, camera_name in enumerate(available_cameras):
            stats = camera_stats[camera_name][param]
            
            # Position for text (above each box)
            x_pos = j + 1
            y_max = ax.get_ylim()[1]
            y_pos = y_max * 0.95
            
            # Create statistics text
            stats_text = f"Max: {stats['max']:.3f}\nMean: {stats['mean']:.3f}\nMin: {stats['min']:.3f}"
            
            # Add text box with statistics
            ax.text(x_pos, y_pos, stats_text, 
                   horizontalalignment='center',
                   verticalalignment='top',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor=colors[j],
                           alpha=0.8,
                           linewidth=1))
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    png_path = output_dir / 'camera_parameters_boxplots.png'
    svg_path = output_dir / 'camera_parameters_boxplots.svg'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"✓ Box plots saved as '{png_path}' and '{svg_path}'")
    
    plt.show()

def create_statistics_table(camera_stats, output_dir):
    """Create and display statistics table."""
    params = ['x', 'y', 'z', 'pitch', 'yaw', 'fov']
    
    # Create output file
    stats_file = output_dir / 'camera_parameters_statistics.txt'
    
    with open(stats_file, 'w') as f:
        # Write to file and print to console
        header = "="*100 + "\n"
        title = "CAMERA PARAMETERS STATISTICS\n"
        header2 = "="*100 + "\n"
        
        f.write(header + title + header2)
        print("\n" + header.strip())
        print(title.strip())
        print(header2.strip())
        
        for param in params:
            param_header = f"\n{param.upper()} STATISTICS:\n"
            separator = "-" * 80 + "\n"
            table_header = f"{'Camera':<20} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'Median':<12}\n"
            
            f.write(param_header + separator + table_header + separator)
            print(param_header.strip())
            print(separator.strip())
            print(table_header.strip())
            print(separator.strip())
            
            for camera_name, stats in camera_stats.items():
                param_stats = stats[param]
                row = (f"{camera_name:<20} {param_stats['mean']:<12.6f} {param_stats['std']:<12.6f} "
                      f"{param_stats['min']:<12.6f} {param_stats['max']:<12.6f} {param_stats['median']:<12.6f}\n")
                f.write(row)
                print(row.strip())
    
    print(f"✓ Statistics saved to '{stats_file}'")

def save_statistics_csv(camera_stats, output_dir):
    """Save statistics to CSV file."""
    all_stats = []
    for camera_name, stats in camera_stats.items():
        for param, param_stats in stats.items():
            row = {
                'Camera': camera_name,
                'Parameter': param,
                'Mean': param_stats['mean'],
                'Std_Dev': param_stats['std'],
                'Min': param_stats['min'],
                'Max': param_stats['max'],
                'Median': param_stats['median']
            }
            all_stats.append(row)
    
    df = pd.DataFrame(all_stats)
    csv_path = output_dir / 'camera_parameters_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Statistics saved to '{csv_path}'")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze camera parameters from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_value_analysis.py "/path/to/cameras/CAM_*.json" --output-dir results
  python3 camera_value_analysis.py "./data/CAM_*.json" --output-dir analysis
  python3 camera_value_analysis.py "CAM_*.json" --output-dir output
        """
    )
    
    parser.add_argument(
        'file_pattern',
        help='File pattern to match camera JSON files (e.g., "CAM_*.json" or "/path/CAM_*.json")'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for saving results'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Do not display plots (only save to files)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Analyze camera files
    print(f"Searching for files matching: {args.file_pattern}")
    camera_data, camera_stats = analyze_camera_files(args.file_pattern)
    
    if not camera_data or not camera_stats:
        print("No valid camera data found. Please check your file pattern and formats.")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {len(camera_data)} camera files")
    
    # Set matplotlib backend for no-display mode
    if args.no_display:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        print("Running in no-display mode (plots will be saved but not shown)")
    
    # Create visualizations and save results
    print("\nGenerating visualizations...")
    create_visualization(camera_data, camera_stats, output_dir)
    create_box_plots(camera_data, camera_stats, output_dir)
    create_individual_camera_plots(camera_data, camera_stats, output_dir)
    
    print("\nGenerating statistics...")
    create_statistics_table(camera_stats, output_dir)
    save_statistics_csv(camera_stats, output_dir)
    
    print(f"\n✓ Analysis complete! All results saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()