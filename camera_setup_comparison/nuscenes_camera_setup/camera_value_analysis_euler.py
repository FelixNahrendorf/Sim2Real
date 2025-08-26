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

def detect_outliers(data, method='iqr', factor=1.5):
    """Detect outliers using IQR method."""
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return outliers, lower_bound, upper_bound
    return np.array([]), None, None

def detect_std_outliers(data, factor=1.5):
    """Detect outliers using standard deviation method."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    lower_bound = mean_val - factor * std_val
    upper_bound = mean_val + factor * std_val
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

def create_visualization(camera_data, camera_stats, output_dir):
    """Create comprehensive visualization of camera parameters."""
    if not camera_data:
        print("No camera data to visualize")
        return
    
    # Define consistent camera order and colors
    camera_order = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
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
    
    # Get available parameters and their labels
    params = get_available_params(camera_data)
    param_labels = get_param_labels()

    #reorder params to match preferred order for visualization
    params_swap=params[5]
    params[5]=params[4]
    params[4]=params[3]
    params[3]=params_swap
    
    # Calculate number of subplots needed
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    # Set up the figure with subplots
    plt.style.use('default')  # Ensure consistent styling
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    fig.suptitle('Camera Parameters', fontsize=16, fontweight='bold')
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each parameter
    for i, param in enumerate(params):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        label = param_labels[param]
        
        # Plot distribution for each camera in consistent order
        for camera_name in available_cameras:
            if param in camera_data[camera_name]:
                # Use raw values from pre-computed statistics
                values = camera_stats[camera_name][param]['raw_values']
                color = color_map[camera_name]
                
                # Plot histogram
                ax.hist(values, bins=20, alpha=0.6, color=color, 
                       label=f'{camera_name}', density=True)
                
                # Plot mean line using pre-computed mean
                mean_val = camera_stats[camera_name][param]['mean']
                ax.axvline(mean_val, color=color, linestyle='--', 
                          linewidth=2, alpha=0.8)
        
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'{label} Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    png_path = output_dir / 'Nuscene_camera_parameters_distribution.png'
    svg_path = output_dir / 'Nuscene_camera_parameters_distribution.svg'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"✓ Distribution plot saved as '{png_path}' and '{svg_path}'")
    
    plt.show()

def create_individual_camera_plots(camera_data, camera_stats, output_dir):
    """Create individual plots for each camera showing original value distributions with enhanced outlier detection."""
    # Get available parameters and their labels
    params = get_available_params(camera_data)
    param_labels = get_param_labels()

    #reorder params to match preferred order for visualization
    params_swap=params[5]
    params[5]=params[4]
    params[4]=params[3]
    params[3]=params_swap
    
    # Calculate number of subplots needed
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    # Define consistent camera order and colors
    camera_order = ['CAM_FRONT',  'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    color_map = {
        'CAM_FRONT': 'red',
        'CAM_FRONT_RIGHT': 'blue',
        'CAM_FRONT_LEFT': 'green',
        'CAM_BACK': 'orange',
        'CAM_BACK_LEFT': 'black',
        'CAM_BACK_RIGHT': 'yellow'
    }
    
    for camera_name, data in camera_data.items():
        plt.style.use('default')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))  # Increased width for better visibility
        fig.suptitle(f'{camera_name} - Parameter Distributions (Original Values)', 
                    fontsize=16, fontweight='bold')
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        camera_color = color_map.get(camera_name, 'gray')  # Use gray as fallback
        
        for j, param in enumerate(params):
            row, col = j // n_cols, j % n_cols
            ax = axes[row, col]
            
            label = param_labels[param]
            
            if param in data:
                # Use pre-computed original values and statistics
                param_stats = camera_stats[camera_name][param]
                values = param_stats['raw_values']  # Use pre-computed original values
                
                # Determine optimal number of bins for better precision
                n_data_points = len(values)
                n_bins_sturges = int(np.ceil(np.log2(n_data_points) + 1))
                n_bins_sqrt = int(np.ceil(np.sqrt(n_data_points)))
                n_bins = max(min(n_bins_sturges, n_bins_sqrt), 25)  # Minimum 25 bins for higher precision
                n_bins = min(n_bins, 50)  # Maximum 50 bins to avoid over-binning
                
                # Create histogram with higher precision
                n, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, color=camera_color, 
                                         edgecolor='black', linewidth=0.3)
                
                # Detect std dev outliers on original values (more than 2 std dev from mean)
                std_outliers, std_lower_bound, std_upper_bound = detect_std_outliers(values, factor=1.5)
                
                # Highlight outlier bins with different color
                if len(std_outliers) > 0:
                    # Find which bins contain outliers
                    for i, (bin_left, bin_right) in enumerate(zip(bins[:-1], bins[1:])):
                        # Check if this bin contains outliers
                        bin_outliers = std_outliers[(std_outliers >= bin_left) & (std_outliers < bin_right)]
                        if len(bin_outliers) > 0:
                            patches[i].set_facecolor('red')
                            patches[i].set_alpha(0.9)
                            patches[i].set_edgecolor('darkred')
                            patches[i].set_linewidth(1.0)
                
                # Use pre-computed statistics - no duplicate calculations
                n_std_outliers = len(std_outliers)
                outlier_percentage = (n_std_outliers / len(values)) * 100
                
                stats_text = (f'Mean: {param_stats["mean"]:.4f}\n'
                            f'Std: {param_stats["std"]:.4f}\n'
                            f'Min: {param_stats["min"]:.4f}\n'
                            f'Max: {param_stats["max"]:.4f}\n'
                            f'Data points: {len(values)}\n'
                            f'Outliers (>1.5σ): {n_std_outliers} ({outlier_percentage:.1f}%)\n'
                            f'Bins: {n_bins}')
                
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
                
                # Add vertical lines for mean and std dev bounds
                ax.axvline(param_stats['mean'], color='darkblue', linestyle='--', linewidth=2, alpha=0.8, 
                          label=f'Mean: {param_stats["mean"]:.4f}')
                
                # Add red lines for all individual outliers
                if len(std_outliers) > 0:
                    # Plot vertical lines for each outlier value
                    unique_outliers = np.unique(std_outliers)
                    for outlier_val in unique_outliers:
                        ax.axvline(outlier_val, color='red', linestyle='-', linewidth=1, alpha=0.8)
                    
                    # Add legend entry for outliers
                    ax.axvline(unique_outliers[0], color='red', linestyle='-', linewidth=1, alpha=0.8, 
                              label=f'Outliers (>1.5σ from mean)')
                
                # Annotate outlier bins if they exist
                if len(std_outliers) > 0:
                    for i, freq in enumerate(n):
                        if freq > 0:
                            bin_center = (bins[i] + bins[i + 1]) / 2
                            bin_outliers = std_outliers[(std_outliers >= bins[i]) & (std_outliers < bins[i + 1])]
                            if len(bin_outliers) > 0:
                                ax.annotate(f'Outliers: {int(freq)}', 
                                           xy=(bin_center, freq), 
                                           xytext=(bin_center, freq * 1.2),
                                           arrowprops=dict(arrowstyle='->', color='red', lw=1),
                                           fontsize=8, ha='center', color='red')
                                break  # Only annotate the first outlier bin to avoid clutter
                
                ax.set_xlabel(f'{label}')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{label} Distribution - High Precision')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=8)
                
                # Improve tick density for better precision visualization
                ax.tick_params(axis='both', which='major', labelsize=8)
                
            else:
                # Parameter not available for this camera
                ax.text(0.5, 0.5, f'{param} not available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{label} Distribution')
        
        # Hide unused subplots
        for j in range(n_params, n_rows * n_cols):
            row, col = j // n_cols, j % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save individual camera plots
        png_path = output_dir / f'{camera_name}_parameter_distributions.png'
        svg_path = output_dir / f'{camera_name}_parameter_distributions.svg'
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(svg_path, bbox_inches='tight')
        print(f"✓ Enhanced individual plot for {camera_name} saved as '{png_path}' and '{svg_path}'")
        
        plt.close()  # Close to save memory

def create_box_plots(camera_data, camera_stats, output_dir):
    """Create box plots for parameter comparison across cameras with proper sizing and visibility."""
    # Get available parameters and their labels
    params = get_available_params(camera_data)
    
    #reorder params to match preferred order for visualization
    params_swap=params[5]
    params[5]=params[4]
    params[4]=params[3]
    params[3]=params_swap
    
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
    fig.suptitle('Camera Parameters - Box Plots Comparison', fontsize=14, fontweight='bold')
    
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
            
            # Remove the equal aspect ratio constraint that was causing issues
            # Let matplotlib handle the aspect ratio naturally
            
            # Add statistics text box in upper right corner (like individual plots)
            stats_text_lines = []
            for j, camera_name in enumerate([cam for cam in available_cameras if param in camera_data[cam]]):
                stats = camera_stats[camera_name][param]
                camera_short = camera_name.replace('CAM_', '')
                stats_text_lines.append(f'{camera_short}: μ={stats["mean"]:.3f}, σ={stats["std"]:.3f}')
            
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
    png_path = output_dir / 'Nuscene_camera_parameters_boxplots.png'
    svg_path = output_dir / 'Nuscene_camera_parameters_boxplots.svg'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"✓ Box plots saved as '{png_path}' and '{svg_path}'")
    
    plt.show()

def create_statistics_table(camera_stats, output_dir):
    """Create and display statistics table."""
    # Get available parameters
    all_params = set()
    for stats in camera_stats.values():
        all_params.update(stats.keys())
    
    # Define parameter order preference
    param_order = ['x', 'y', 'z', 'pitch', 'yaw', 'roll', 'fov']
    params = [param for param in param_order if param in all_params]

    #reorder params to match preferred order for visualization
    params_swap=params[5]
    params[5]=params[4]
    params[4]=params[3]
    params[3]=params_swap
    
    # Create output file
    stats_file = output_dir / 'Nuscene_camera_parameters_statistics.txt'
    
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
                if param in stats:
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
    csv_path = output_dir / 'Nuscene_camera_poses.csv'
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