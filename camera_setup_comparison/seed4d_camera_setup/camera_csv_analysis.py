#!/usr/bin/env python3
"""
Camera Parameters Analysis Tool - CSV Version

Analyzes camera intrinsic and extrinsic parameters from CSV files and generates
comprehensive visualizations and statistics.

Usage:
    python3 camera_value_analysis.py <csv_file_path> --output-dir <output_directory>
    
Example:
    python3 camera_value_analysis.py "camera_parameters.csv" --output-dir results
    python3 camera_value_analysis.py "/data/camera_stats.csv" --output-dir analysis_output
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
from pathlib import Path
import pandas as pd
from scipy import stats
import sys

def load_camera_data_from_csv(file_path):
    """Load camera data from CSV file and reorganize by camera."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['Camera', 'Parameter', 'Mean', 'Std_Dev', 'Min', 'Max', 'Median']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing columns in CSV: {missing_columns}")
            return None, None
        
        # Get unique cameras and parameters
        cameras = df['Camera'].unique()
        parameters = df['Parameter'].unique()
        
        print(f"Found cameras: {list(cameras)}")
        print(f"Found parameters: {list(parameters)}")
        
        # Reorganize data by camera
        camera_data = {}
        camera_stats = {}
        
        for camera in cameras:
            camera_df = df[df['Camera'] == camera]
            camera_data[camera] = {}
            camera_stats[camera] = {}
            
            for _, row in camera_df.iterrows():
                param = row['Parameter']
                
                # Generate synthetic data based on statistics
                # This simulates the original data distribution
                mean_val = row['Mean']
                std_val = row['Std_Dev']
                min_val = row['Min']
                max_val = row['Max']
                median_val = row['Median']
                
                # Generate sample data (100 points) that matches the statistics
                # Use a combination of normal distribution and clipping to match min/max
                np.random.seed(hash(f"{camera}_{param}") % 2**32)  # Consistent seed for reproducibility
                
                # Generate more points than needed, then select to match statistics
                n_samples = 1000
                samples = np.random.normal(mean_val, std_val, n_samples)
                
                # Clip to min/max bounds
                samples = np.clip(samples, min_val, max_val)
                
                # Select subset that best matches the median
                samples_sorted = np.sort(samples)
                target_median_idx = len(samples_sorted) // 2
                current_median = samples_sorted[target_median_idx]
                
                # Adjust samples to better match the median if needed
                median_diff = median_val - current_median
                samples = samples + median_diff
                samples = np.clip(samples, min_val, max_val)
                
                # Take a subset for final data (100 points is reasonable)
                final_samples = np.random.choice(samples, size=min(100, len(samples)), replace=False)
                
                camera_data[camera][param] = final_samples
                
                # Store the original statistics
                camera_stats[camera][param] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'median': median_val
                }
        
        return camera_data, camera_stats
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def analyze_camera_csv_files(file_pattern):
    """Analyze camera CSV files matching the pattern."""
    # Find all files matching the pattern
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        # If no glob matches, try treating it as a single file path
        if os.path.exists(file_pattern):
            csv_files = [file_pattern]
        else:
            print(f"No files found matching pattern: {file_pattern}")
            return None, None
    
    print(f"Found {len(csv_files)} files matching pattern")
    
    # For now, we'll process the first CSV file
    # In the future, this could be extended to merge multiple CSV files
    file_path = csv_files[0]
    print(f"Processing: {file_path}")
    
    camera_data, camera_stats = load_camera_data_from_csv(file_path)
    
    if camera_data is not None and camera_stats is not None:
        print(f"✓ Loaded data for {len(camera_data)} cameras")
        return camera_data, camera_stats
    else:
        print(f"✗ Failed to load {file_path}")
        return None, None

def get_available_params(camera_data):
    """Get available parameters from the camera data."""
    all_params = set()
    for data in camera_data.values():
        all_params.update(data.keys())
    
    # Define parameter order preference
    param_order = ['x', 'y', 'z', 'pitch', 'yaw', 'roll', 'fov']
    
    # Return parameters in preferred order, but include any additional ones found
    available_params = [param for param in param_order if param in all_params]
    additional_params = [param for param in all_params if param not in param_order]
    available_params.extend(sorted(additional_params))
    
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
    # Add any additional cameras not in the predefined order
    additional_cameras = [cam for cam in camera_data.keys() if cam not in camera_order]
    available_cameras.extend(additional_cameras)
    
    # Generate colors for additional cameras
    additional_colors = ['purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, cam in enumerate(additional_cameras):
        if cam not in color_map:
            color_map[cam] = additional_colors[i % len(additional_colors)]
    
    # Get available parameters and their labels
    params = get_available_params(camera_data)
    param_labels = get_param_labels()
    
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
        
        label = param_labels.get(param, param.title())
        
        # Plot distribution for each camera in consistent order
        for camera_name in available_cameras:
            if param in camera_data[camera_name]:
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
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
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
    # Get available parameters and their labels
    params = get_available_params(camera_data)
    param_labels = get_param_labels()
    
    # Calculate number of subplots needed
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
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
    
    # Add colors for additional cameras
    additional_colors = ['purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    additional_cameras = [cam for cam in camera_data.keys() if cam not in camera_order]
    for i, cam in enumerate(additional_cameras):
        if cam not in color_map:
            color_map[cam] = additional_colors[i % len(additional_colors)]
    
    for camera_name, data in camera_data.items():
        plt.style.use('default')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle(f'{camera_name} - Parameter Distributions (Absolute Values)', 
                    fontsize=14, fontweight='bold')
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        camera_color = color_map.get(camera_name, 'gray')  # Use gray as fallback
        
        for j, param in enumerate(params):
            row, col = j // n_cols, j % n_cols
            ax = axes[row, col]
            
            label = param_labels.get(param, param.title())
            
            if param in data:
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
            else:
                # Parameter not available for this camera
                ax.text(0.5, 0.5, f'{param} not available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'|{label}| Distribution')
        
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
        print(f"✓ Individual plot for {camera_name} saved as '{png_path}' and '{svg_path}'")
        
        plt.close()  # Close to save memory

def create_box_plots(camera_data, camera_stats, output_dir):
    """Create box plots for parameter comparison across cameras with proper sizing and visibility."""
    # Get available parameters and their labels
    params = get_available_params(camera_data)
    param_labels = get_param_labels()
    
    # Calculate number of subplots needed
    n_params = len(params)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
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
    # Add any additional cameras not in the predefined order
    additional_cameras = [cam for cam in camera_data.keys() if cam not in camera_order]
    available_cameras.extend(additional_cameras)
    
    # Generate colors for additional cameras
    additional_colors = ['purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, cam in enumerate(additional_cameras):
        if cam not in color_map:
            color_map[cam] = additional_colors[i % len(additional_colors)]
    
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
        
        label = param_labels.get(param, param.title())
        
        # Prepare data for box plot in consistent order
        data_for_boxplot = []
        labels = []
        colors = []
        
        for camera_name in available_cameras:
            if param in camera_data[camera_name]:
                data_for_boxplot.append(camera_data[camera_name][param])
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
    png_path = output_dir / 'camera_parameters_boxplots.png'
    svg_path = output_dir / 'camera_parameters_boxplots.svg'
    
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
    additional_params = [param for param in all_params if param not in param_order]
    params.extend(sorted(additional_params))
    
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
    csv_path = output_dir / 'camera_parameters_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Statistics saved to '{csv_path}'")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze camera parameters from CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_value_analysis.py "camera_parameters.csv" --output-dir results
  python3 camera_value_analysis.py "/path/to/data/camera_stats.csv" --output-dir analysis
  python3 camera_value_analysis.py "*.csv" --output-dir output
        """
    )
    
    parser.add_argument(
        'file_pattern',
        help='CSV file path or pattern to match camera parameter files (e.g., "camera_params.csv" or "*.csv")'
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
    camera_data, camera_stats = analyze_camera_csv_files(args.file_pattern)
    
    if not camera_data or not camera_stats:
        print("No valid camera data found. Please check your file pattern and CSV format.")
        print("Expected CSV format: Camera, Parameter, Mean, Std_Dev, Min, Max, Median")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded data for {len(camera_data)} cameras")
    
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