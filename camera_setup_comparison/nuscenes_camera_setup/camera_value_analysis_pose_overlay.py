#!/usr/bin/env python3
"""
Camera Parameters Analysis Tool with CSV Overlay

Analyzes camera intrinsic and extrinsic parameters from JSON files and generates
box plots with optional CSV mean value overlays.

Usage:
    python3 camera_value_analysis_enhanced.py <json_directory> --output-dir <output_directory> [--csv-file <csv_file>]
    
Example:
    python3 camera_value_analysis_enhanced.py "./cameras/" --output-dir results --csv-file reference_means.csv
    python3 camera_value_analysis_enhanced.py "/data/cameras/" --output-dir analysis_output
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
        
        # Extract roll angles if available
        rolls = None
        if 'rolls' in data:
            rolls = np.array(data['rolls'])
        elif 'roll' in data:
            rolls = np.array(data['roll'])
        
        result = {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords,
            'pitch': pitches,
            'yaw': yaws,
            'fov': fovs
        }
        
        # Add roll if available
        if rolls is not None:
            result['roll'] = rolls
        
        return result
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_statistics(data_dict):
    """Compute comprehensive statistics for each parameter."""
    stats_dict = {}
    for param, values in data_dict.items():
        # Original values statistics
        stats_dict[param] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
        
        # Absolute values statistics (for individual camera plots)
        abs_values = np.abs(values)
        stats_dict[param]['abs_mean'] = np.mean(abs_values)
        stats_dict[param]['abs_std'] = np.std(abs_values)
        stats_dict[param]['abs_min'] = np.min(abs_values)
        stats_dict[param]['abs_max'] = np.max(abs_values)
        stats_dict[param]['abs_median'] = np.median(abs_values)
        
        # Store raw data for outlier detection and other computations
        stats_dict[param]['raw_values'] = values
        stats_dict[param]['abs_values'] = abs_values
        
    return stats_dict

def analyze_camera_files(json_directory):
    """Analyze camera JSON files in the specified directory."""
    camera_data = {}
    camera_stats = {}
    
    # Convert to Path object
    json_dir = Path(json_directory)
    
    if not json_dir.exists():
        print(f"Directory does not exist: {json_directory}")
        return None, None
    
    if not json_dir.is_dir():
        print(f"Path is not a directory: {json_directory}")
        return None, None
    
    # Find all JSON files in the directory
    json_files = list(json_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in directory: {json_directory}")
        return None, None
    
    print(f"Found {len(json_files)} JSON files in directory")
    
    for file_path in json_files:
        # Get filename without extension and remove "_converted" suffix
        camera_name = file_path.stem
        if camera_name.endswith('_converted'):
            camera_name = camera_name[:-10]  # Remove "_converted"
        elif camera_name.endswith('_transformed'):
            camera_name = camera_name[:-12]  # Remove "_transformed"
        
        print(f"Processing: {camera_name}")
        
        data = load_camera_data(file_path)
        if data is not None:
            camera_data[camera_name] = data
            camera_stats[camera_name] = compute_statistics(data)
            print(f"✓ Loaded data for {camera_name}")
        else:
            print(f"✗ Failed to load {camera_name}")
    
    return camera_data, camera_stats

def load_csv_means(csv_file_path):
    """Load mean values from CSV file."""
    try:
        df = pd.read_csv(csv_file_path)
        
        # Expected columns: Camera, Parameter, Mean, Std_Dev, Min, Max, Median
        if 'Camera' not in df.columns or 'Parameter' not in df.columns or 'Mean' not in df.columns:
            print(f"Error: CSV file must contain 'Camera', 'Parameter', and 'Mean' columns")
            return None
        
        csv_means = {}
        for _, row in df.iterrows():
            camera = row['Camera']
            parameter = row['Parameter']
            mean_value = row['Mean']
            
            if camera not in csv_means:
                csv_means[camera] = {}
            csv_means[camera][parameter] = mean_value
        
        print(f"✓ Loaded CSV means for {len(csv_means)} cameras")
        return csv_means
        
    except Exception as e:
        print(f"Error loading CSV file {csv_file_path}: {e}")
        return None

def get_available_params(camera_data):
    """Get available parameters from the camera data."""
    all_params = set()
    for data in camera_data.values():
        all_params.update(data.keys())
    
    # Define parameter order preference
    param_order = ['x', 'y', 'z', 'pitch', 'yaw', 'roll', 'fov']
    
    # Return parameters in preferred order
    available_params = [param for param in param_order if param in all_params]
    return available_params

def get_param_labels():
    """Get parameter labels for plotting."""
    return {
        'x': 'X Coordinate',
        'y': 'Y Coordinate', 
        'z': 'Z Coordinate',
        'pitch': 'Pitch (rad)',
        'yaw': 'Yaw (rad)',
        'roll': 'Roll (rad)',
        'fov': 'FOV (degrees)'
    }

def create_box_plots_with_csv_overlay(camera_data, camera_stats, output_dir, csv_means=None):
    """Create box plots for parameter comparison across cameras with optional CSV mean overlays."""
    # Get available parameters and their labels
    params = get_available_params(camera_data)
    
    # Reorder params to match preferred order for visualization
    if len(params) >= 6:
        params_swap = params[5]
        params[5] = params[4]
        params[4] = params[3]
        params[3] = params_swap
    
    param_labels = get_param_labels()
    
    # Calculate number of subplots needed
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    # Define consistent camera order and colors
    camera_order = ['CAM_FRONT','CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    color_map = {
        'CAM_FRONT': 'red',
        'CAM_FRONT_RIGHT': 'blue',
        'CAM_FRONT_LEFT': 'green', 
        'CAM_BACK': 'orange',
        'CAM_BACK_LEFT': 'black',
        'CAM_BACK_RIGHT': 'yellow'
    }
    
    # Filter and order cameras based on what's available in the data
    available_cameras = [cam for cam in camera_order if cam in camera_data.keys()]
    
    plt.style.use('default')  # Ensure consistent styling
    # Use similar sizing as individual camera plots - 5 units height per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Add CSV overlay info to title if CSV data is provided
    title = 'NuScenes Camera Parameters Box Plots'
    if csv_means:
        title += ' (with SEED4D Pose Overlay)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    
    for i, param in enumerate(params):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        label = param_labels[param]
        
        # Prepare data for box plot in consistent order
        data_for_boxplot = []
        labels = []
        colors = []
        
        for camera_name in available_cameras:
            if param in camera_data[camera_name]:
                # Use raw values from pre-computed statistics
                data_for_boxplot.append(camera_stats[camera_name][param]['raw_values'])
                labels.append(camera_name.replace('CAM_', ''))  # Shorter labels
                colors.append(color_map[camera_name])
        
        if data_for_boxplot:  # Only create plot if there's data
            # Create box plot with better styling
            bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(linewidth=2, color='darkred'))
            
            # Color the boxes with consistent colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
            
            # Add CSV mean overlays if provided
            if csv_means:
                for j, camera_name in enumerate([cam for cam in available_cameras if param in camera_data[cam]]):
                    if camera_name in csv_means and param in csv_means[camera_name]:
                        csv_mean = csv_means[camera_name][param]
                        # Add horizontal line at the CSV mean value
                        # Position it at x-coordinate j+1 (box plot positions start at 1)
                        x_pos = j + 1
                        line_width = 0.3  # Width of the horizontal line
                        ax.plot([x_pos - line_width, x_pos + line_width], 
                               [csv_mean, csv_mean], 
                               color='darkmagenta', linewidth=3, solid_capstyle='butt')
            
            # Add statistics text box in upper right corner (like individual plots)
            stats_text_lines = []
            for j, camera_name in enumerate([cam for cam in available_cameras if param in camera_data[cam]]):
                stats = camera_stats[camera_name][param]
                camera_short = camera_name.replace('CAM_', '')
                
                if csv_means and camera_name in csv_means and param in csv_means[camera_name]:
                    csv_mean = csv_means[camera_name][param]
                    # Compute absolute distance between means
                    abs_distance = abs(stats["mean"] - csv_mean)
                    line = f'{camera_short}: μ={stats["mean"]:.3f}, μ(SEED4D)={csv_mean:.3f}, Δ={abs_distance:.3f}'
                else:
                    line = f'{camera_short}: μ={stats["mean"]:.3f}, σ={stats["std"]:.3f}'
                
                stats_text_lines.append(line)
            
            stats_text = '\n'.join(stats_text_lines)
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # Styling similar to individual camera plots
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add some padding to make the spread more visible
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            padding = y_range * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
            
        else:
            # No data available for this parameter
            ax.text(0.5, 0.5, f'No data available for {param}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{label} Comparison')
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    filename_suffix = '_with_csv_overlay' if csv_means else ''
    png_path = output_dir / f'Nuscene_camera_parameters_boxplots{filename_suffix}.png'
    svg_path = output_dir / f'Nuscene_camera_parameters_boxplots{filename_suffix}.svg'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    
    if csv_means:
        print(f"✓ Box plots with CSV overlay saved as '{png_path}' and '{svg_path}'")
    else:
        print(f"✓ Box plots saved as '{png_path}' and '{svg_path}'")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze camera parameters from JSON files with optional CSV mean overlay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_value_analysis_enhanced.py "./cameras/" --output-dir results
  python3 camera_value_analysis_enhanced.py "/path/to/cameras/" --output-dir analysis --csv-file reference_means.csv
  python3 camera_value_analysis_enhanced.py "data/cameras/" --output-dir output --csv-file stats.csv
        """
    )
    
    parser.add_argument(
        'json_directory',
        help='Directory containing camera JSON files'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for saving results'
    )
    
    parser.add_argument(
        '--csv-file',
        help='CSV file containing mean values to overlay on box plots (optional)'
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
    print(f"Searching for JSON files in directory: {args.json_directory}")
    camera_data, camera_stats = analyze_camera_files(args.json_directory)
    
    if not camera_data or not camera_stats:
        print("No valid camera data found. Please check your directory path and file formats.")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {len(camera_data)} camera files")
    
    # Load CSV means if provided
    csv_means = None
    if args.csv_file:
        print(f"\nLoading CSV file: {args.csv_file}")
        csv_means = load_csv_means(args.csv_file)
        if csv_means is None:
            print("Failed to load CSV file. Proceeding without CSV overlay.")
    
    # Set matplotlib backend for no-display mode
    if args.no_display:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        print("Running in no-display mode (plots will be saved but not shown)")
    
    # Create box plots with optional CSV overlay
    print("\nGenerating box plots...")
    create_box_plots_with_csv_overlay(camera_data, camera_stats, output_dir, csv_means)
    
    print(f"\n✓ Analysis complete! Box plots saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()