#!/usr/bin/env python3
"""
Camera Parameters Difference Analysis Tool

Analyzes differences in camera parameters between multiple CSV files and generates
comprehensive visualizations and statistics.

Usage:
    python3 camera_diff_analysis.py <csv_file1> <csv_file2> [csv_file3] ... --output-dir <output_directory>
    
Example:
    python3 camera_diff_analysis.py baseline.csv new_config.csv --output-dir diff_results
    python3 camera_diff_analysis.py config1.csv config2.csv config3.csv --output-dir multi_diff_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
from itertools import combinations
import sys

def load_camera_csv(file_path):
    """Load camera statistics from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} records from {file_path}")
        
        # Clean up column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Clean up Camera and Parameter columns
        if 'Camera' in df.columns:
            df['Camera'] = df['Camera'].str.strip()
        if 'Parameter' in df.columns:
            df['Parameter'] = df['Parameter'].str.strip()
        
        print(f"  - Cameras found: {sorted(df['Camera'].unique().tolist())}")
        print(f"  - Parameters found: {sorted(df['Parameter'].unique().tolist())}")
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_differences(df1, df2, filename1, filename2):
    """Compute differences between two camera parameter datasets."""
    print(f"\nComputing differences between {filename1} and {filename2}...")
    
    # Create unique keys for matching
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    df1_copy['match_key'] = df1_copy['Camera'].astype(str) + '|' + df1_copy['Parameter'].astype(str)
    df2_copy['match_key'] = df2_copy['Camera'].astype(str) + '|' + df2_copy['Parameter'].astype(str)
    
    print(f"File 1 ({filename1}) has {len(df1_copy)} records with {len(df1_copy['match_key'].unique())} unique camera-parameter combinations")
    print(f"File 2 ({filename2}) has {len(df2_copy)} records with {len(df2_copy['match_key'].unique())} unique camera-parameter combinations")
    
    # Merge dataframes on Camera and Parameter using inner join to only get matches
    merged = pd.merge(df1_copy, df2_copy, on=['Camera', 'Parameter'], suffixes=('_1', '_2'), how='inner')
    
    if merged.empty:
        print(f"Warning: No matching camera/parameter combinations found between {filename1} and {filename2}")
        
        # Debug: Show what's in each file
        print(f"\nDEBUG - {filename1} combinations:")
        for combo in sorted(df1_copy['match_key'].unique()):
            print(f"  {combo}")
        
        print(f"\nDEBUG - {filename2} combinations:")
        for combo in sorted(df2_copy['match_key'].unique()):
            print(f"  {combo}")
        
        return None
    
    print(f"✓ Found {len(merged)} matching camera-parameter combinations")
    
    # Show matched combinations
    matched_combos = []
    for _, row in merged.iterrows():
        combo = f"{row['Camera']},{row['Parameter']}"
        matched_combos.append(combo)
    print(f"Matched combinations: {sorted(set(matched_combos))}")
    
    # Compute differences for each statistic
    diff_data = []
    stat_columns = ['Mean', 'Std_Dev', 'Min', 'Max', 'Median']
    
    for _, row in merged.iterrows():
        diff_row = {
            'Camera': row['Camera'],
            'Parameter': row['Parameter'],
            'File1': filename1,
            'File2': filename2
        }
        
        for stat in stat_columns:
            col1 = f"{stat}_1"
            col2 = f"{stat}_2"
            if col1 in row and col2 in row and pd.notna(row[col1]) and pd.notna(row[col2]):
                #diff_row[f"{stat}_Diff"] = abs(row[col2]) - abs(row[col1])  # File2 - File1
                diff_row[f"{stat}_Diff"] = np.abs(row[col2] - row[col1])  # File2 - File1
                diff_row[f"{stat}_1"] = row[col1]
                diff_row[f"{stat}_2"] = row[col2]
                
                # Print individual differences for verification
                print(f"  {row['Camera']},{row['Parameter']},{stat}: {row[col1]:.6f} -> {row[col2]:.6f} (diff: {row[col2] - row[col1]:.6f})")
        
        diff_data.append(diff_row)
    
    result_df = pd.DataFrame(diff_data)
    print(f"✓ Computed differences for {len(result_df)} combinations")
    
    return result_df

def get_camera_order_and_colors():
    """Get consistent camera order and color mapping."""
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
    return camera_order, color_map

def get_param_order_and_labels(available_params=None):
    """Get parameter order and labels, auto-detecting from available data if provided."""
    
    # Default parameter mappings for different types of camera data
    default_orders = {
        # Transform matrix parameters (rotation matrix + translation + intrinsics)
        'transform_matrix': ['TX', 'TY', 'TZ', 'R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33', 'fx', 'fy', 'cx', 'cy'],
        # Pose parameters (position + orientation + FOV)
        'pose': ['x', 'y', 'z', 'pitch', 'yaw', 'roll', 'fov'],
        # Intrinsics only
        'intrinsics': ['fx', 'fy', 'cx', 'cy', 'fov'],
        # Extrinsics only  
        'extrinsics': ['x', 'y', 'z', 'pitch', 'yaw', 'roll', 'TX', 'TY', 'TZ'],
        # Rotation matrix only
        'rotation_matrix': ['R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33']
    }
    
    param_labels = {
        # Translation/Position
        'x': 'X Coordinate',
        'y': 'Y Coordinate', 
        'z': 'Z Coordinate',
        'TX': 'Translation X',
        'TY': 'Translation Y',
        'TZ': 'Translation Z',
        
        # Rotation
        'pitch': 'Pitch (rad)',
        'yaw': 'Yaw (rad)',
        'roll': 'Roll (rad)',
        'R11': 'Rotation Matrix R11',
        'R12': 'Rotation Matrix R12',
        'R13': 'Rotation Matrix R13',
        'R21': 'Rotation Matrix R21',
        'R22': 'Rotation Matrix R22',
        'R23': 'Rotation Matrix R23',
        'R31': 'Rotation Matrix R31',
        'R32': 'Rotation Matrix R32',
        'R33': 'Rotation Matrix R33',
        
        # Camera intrinsics
        'fov': 'FOV (degrees)',
        'fx': 'Focal Length X',
        'fy': 'Focal Length Y',
        'cx': 'Principal Point X',
        'cy': 'Principal Point Y'
    }
    
    # Auto-detect parameter type if available_params is provided
    if available_params:
        available_set = set(available_params)
        print(f"Auto-detecting parameter type from available parameters: {sorted(available_params)}")
        
        # Check which parameter set matches best
        best_match = None
        best_overlap = 0
        
        for param_type, param_list in default_orders.items():
            overlap_count = len(available_set.intersection(set(param_list)))
            coverage = overlap_count / len(param_list) if param_list else 0
            print(f"  {param_type}: {overlap_count}/{len(param_list)} parameters match (coverage: {coverage:.1%})")
            
            if overlap_count > best_overlap:
                best_overlap = overlap_count
                best_match = param_type
        
        if best_match:
            print(f"✓ Best match: {best_match} with {best_overlap} overlapping parameters")
            # Filter to only include available parameters, maintaining order
            param_order = [p for p in default_orders[best_match] if p in available_set]
            
            # Add any remaining parameters that weren't in the best match
            remaining_params = [p for p in available_params if p not in param_order]
            if remaining_params:
                print(f"  Adding remaining parameters: {remaining_params}")
                param_order.extend(sorted(remaining_params))
                
            return param_order, param_labels
        
        # If no good match, use all available parameters sorted
        print("No good parameter type match found, using all available parameters")
        param_order = sorted(available_params)
        return param_order, param_labels
    
    # Default fallback to transform matrix type
    print("No parameters provided, using default transform matrix parameter order")
    return default_orders['transform_matrix'], param_labels

def create_difference_boxplots(diff_df, filename1, filename2, output_dir, stat_type='Mean'):
    """Create box plots for parameter differences."""
    camera_order, color_map = get_camera_order_and_colors()
    
    # Get available parameters from the data
    available_params = sorted(diff_df['Parameter'].unique())
    param_order, param_labels = get_param_order_and_labels(available_params)
    
    # Filter available cameras and parameters
    available_cameras = [cam for cam in camera_order if cam in diff_df['Camera'].unique()]
    available_params_ordered = [param for param in param_order if param in diff_df['Parameter'].unique()]
    
    if not available_params_ordered:
        print(f"Warning: No parameters found for {filename1} vs {filename2}")
        return
    
    print(f"Creating {stat_type} box plots for parameters: {available_params_ordered}")
    
    # Calculate subplot layout
    n_params = len(available_params_ordered)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plt.style.use('default')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Parameter Differences - {stat_type}\nDifference between {filename2} and {filename1}', 
                fontsize=14, fontweight='bold')
    
    # Handle single row case
    if n_rows == 1:
        if n_cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.reshape(1, -1)
    elif n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    
    diff_col = f"{stat_type}_Diff"
    
    for i, param in enumerate(available_params_ordered):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        label = param_labels.get(param, param)
        
        # Filter data for this parameter
        param_data = diff_df[diff_df['Parameter'] == param]
        
        if param_data.empty:
            ax.text(0.5, 0.5, f'No data for {param}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{label} Difference')
            continue
        
        # Prepare data for box plot in consistent order
        data_for_boxplot = []
        labels = []
        colors = []
        
        for camera_name in available_cameras:
            camera_data = param_data[param_data['Camera'] == camera_name]
            if not camera_data.empty and diff_col in camera_data.columns:
                diff_values = camera_data[diff_col].values
                if len(diff_values) > 0 and not np.isnan(diff_values).all():
                    data_for_boxplot.append(diff_values)
                    labels.append(camera_name.replace('CAM_', ''))
                    colors.append(color_map[camera_name])
        
        if data_for_boxplot:
            # Create box plot
            bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(linewidth=2, color='darkred'))
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
            
            # Add zero reference line
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
            
            # Add statistics text
            stats_text_lines = []
            for j, camera_name in enumerate([cam for cam in available_cameras if not param_data[param_data['Camera'] == cam].empty]):
                camera_diff_data = param_data[param_data['Camera'] == camera_name]
                if not camera_diff_data.empty and diff_col in camera_diff_data.columns:
                    diff_vals = camera_diff_data[diff_col].values
                    if len(diff_vals) > 0 and not np.isnan(diff_vals).all():
                        camera_short = camera_name.replace('CAM_', '')
                        mean_diff = np.nanmean(diff_vals)
                        stats_text_lines.append(f'{camera_short}: {mean_diff:.6f}')
            
            if stats_text_lines:
                stats_text = '\n'.join(stats_text_lines)
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            ax.set_ylabel(f'{label} Difference')
            ax.set_title(f'{label} Difference')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, f'No data for {param}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{label} Difference')
    
    # Hide unused subplots
    for i in range(n_params, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    safe_filename1 = "".join(c for c in filename1 if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_filename2 = "".join(c for c in filename2 if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    png_path = output_dir / f'diff_{safe_filename1}_vs_{safe_filename2}_{stat_type}_boxplots.png'
    svg_path = output_dir / f'diff_{safe_filename1}_vs_{safe_filename2}_{stat_type}_boxplots.svg'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    print(f"✓ Difference box plots saved as '{png_path}' and '{svg_path}'")
    
    plt.close()

def save_difference_statistics(diff_df, filename1, filename2, output_dir):
    """Save difference statistics to TXT and CSV files."""
    if diff_df.empty:
        print(f"Warning: No difference data to save for {filename1} vs {filename2}")
        return
    
    safe_filename1 = "".join(c for c in filename1 if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_filename2 = "".join(c for c in filename2 if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    # Save CSV
    csv_path = output_dir / f'diff_{safe_filename1}_vs_{safe_filename2}_statistics.csv'
    
    # Prepare CSV data
    csv_data = []
    stat_types = ['Mean', 'Std_Dev', 'Min', 'Max', 'Median']
    
    for _, row in diff_df.iterrows():
        for stat in stat_types:
            diff_col = f"{stat}_Diff"
            if diff_col in row and pd.notna(row[diff_col]):
                csv_row = {
                    'Camera': row['Camera'],
                    'Parameter': row['Parameter'],
                    'Statistic': stat,
                    'File1_Value': row.get(f"{stat}_1", 'N/A'),
                    'File2_Value': row.get(f"{stat}_2", 'N/A'),
                    'Difference': row[diff_col],
                    'File1': filename1,
                    'File2': filename2
                }
                csv_data.append(csv_row)
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False)
        print(f"✓ Difference statistics saved to '{csv_path}'")
    
    # Save TXT
    txt_path = output_dir / f'diff_{safe_filename1}_vs_{safe_filename2}_statistics.txt'
    
    # Get available parameters from the data
    available_params = sorted(diff_df['Parameter'].unique())
    param_order, param_labels = get_param_order_and_labels(available_params)
    
    with open(txt_path, 'w') as f:
        header = "="*100 + "\n"
        title = f"CAMERA PARAMETERS DIFFERENCE ANALYSIS\n"
        subtitle = f"Difference between {filename2} and {filename1}\n"
        header2 = "="*100 + "\n"
        
        f.write(header + title + subtitle + header2)
        print(f"\n{header.strip()}")
        print(title.strip())
        print(subtitle.strip())
        print(header2.strip())
        
        # Group by parameter
        available_params_ordered = [param for param in param_order if param in diff_df['Parameter'].unique()]
        
        for param in available_params_ordered:
            param_data = diff_df[diff_df['Parameter'] == param]
            if param_data.empty:
                continue
                
            param_header = f"\n{param.upper()} DIFFERENCES:\n"
            separator = "-" * 80 + "\n"
            
            f.write(param_header + separator)
            print(param_header.strip())
            print(separator.strip())
            
            # Write statistics for each camera
            camera_order, _ = get_camera_order_and_colors()
            available_cameras = [cam for cam in camera_order if cam in param_data['Camera'].unique()]
            
            for camera in available_cameras:
                camera_data = param_data[param_data['Camera'] == camera]
                if not camera_data.empty:
                    f.write(f"\n{camera}:\n")
                    print(f"\n{camera}:")
                    
                    stat_types = ['Mean', 'Std_Dev', 'Min', 'Max', 'Median']
                    for stat in stat_types:
                        diff_col = f"{stat}_Diff"
                        val1_col = f"{stat}_1"
                        val2_col = f"{stat}_2"
                        
                        if (diff_col in camera_data.columns and 
                            not camera_data[diff_col].isna().all()):
                            
                            diff_val = camera_data[diff_col].iloc[0]
                            val1 = camera_data[val1_col].iloc[0] if val1_col in camera_data.columns else 'N/A'
                            val2 = camera_data[val2_col].iloc[0] if val2_col in camera_data.columns else 'N/A'
                            
                            if pd.notna(diff_val):
                                line = f"  {stat:<10}: {val1:<15.6f} -> {val2:<15.6f} (diff: {diff_val:>15.6f})\n"
                                f.write(line)
                                print(line.strip())
    
    print(f"✓ Difference statistics saved to '{txt_path}'")

def analyze_differences(csv_files, output_dir):
    """Analyze differences between all combinations of CSV files."""
    # Load all CSV files
    dataframes = {}
    filenames = {}
    
    for csv_file in csv_files:
        file_path = Path(csv_file)
        if file_path.exists():
            df = load_camera_csv(file_path)
            if df is not None:
                filename = file_path.stem  # Get filename without extension
                dataframes[csv_file] = df
                filenames[csv_file] = filename
        else:
            print(f"Warning: File {csv_file} does not exist")
    
    if len(dataframes) < 2:
        print("Error: Need at least 2 valid CSV files to compute differences")
        return
    
    print(f"\n✓ Loaded {len(dataframes)} CSV files")
    
    # Compute differences for all combinations
    file_list = list(dataframes.keys())
    combinations_list = list(combinations(file_list, 2))
    
    print(f"Computing differences for {len(combinations_list)} combinations...")
    
    for file1, file2 in combinations_list:
        filename1 = filenames[file1]
        filename2 = filenames[file2]
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {filename1} vs {filename2}")
        print(f"{'='*60}")
        
        # Compute differences
        diff_df = compute_differences(dataframes[file1], dataframes[file2], filename1, filename2)
        
        if diff_df is not None and not diff_df.empty:
            # Create visualizations for each statistic type
            stat_types = ['Mean', 'Std_Dev', 'Min', 'Max', 'Median']
            for stat_type in stat_types:
                if f"{stat_type}_Diff" in diff_df.columns:
                    create_difference_boxplots(diff_df, filename1, filename2, output_dir, stat_type)
            
            # Save statistics
            save_difference_statistics(diff_df, filename1, filename2, output_dir)
        else:
            print(f"Warning: No differences computed for {filename1} vs {filename2}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze differences in camera parameters between CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 camera_diff_analysis.py baseline.csv new_config.csv --output-dir diff_results
  python3 camera_diff_analysis.py config1.csv config2.csv config3.csv --output-dir multi_diff
        """
    )
    
    parser.add_argument(
        'csv_files',
        nargs='+',
        help='CSV files containing camera parameter statistics (minimum 2 files)'
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
    
    if len(args.csv_files) < 2:
        print("Error: Please provide at least 2 CSV files")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Set matplotlib backend for no-display mode
    if args.no_display:
        import matplotlib
        matplotlib.use('Agg')
        print("Running in no-display mode (plots will be saved but not shown)")
    
    # Analyze differences
    analyze_differences(args.csv_files, output_dir)
    
    print(f"\n✓ Analysis complete! All results saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()