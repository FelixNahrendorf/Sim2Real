#!/usr/bin/env python3

import os
import json
import pandas as pd
import re
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

def find_subdirectories(base_dir, pattern_prefix="carla_on_nuscene_0shot"):
    """
    Find subdirectories matching the pattern: carla_on_nuscene_0shot_tdX_rendY
    """
    subdirs = []
    pattern = re.compile(rf"{pattern_prefix}_td(\d+)_rend(\d+)$")
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist!")
        return subdirs
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                td_val = int(match.group(1))
                rend_val = int(match.group(2))
                subdirs.append((item, td_val, rend_val))
    
    # Sort by td value first, then by rend value
    subdirs.sort(key=lambda x: (x[1], x[2]))
    return subdirs

def load_scores(subdir_path):
    """
    Load scores from scores_all_avg.json file in the subdirectory
    """
    subdir_path_mod = os.path.join(subdir_path, "nuscene")
    scores_file = os.path.join(subdir_path_mod, "scores_all_avg.json")
    if os.path.exists(scores_file):
        try:
            with open(scores_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {scores_file}: {e}")
            return None
    else:
        print(f"File not found: {scores_file}")
        return None

def create_row_name(td_val, rend_val):
    """
    Create row name in format: td0_rend0
    """
    return f"td{td_val}_rend{rend_val}"

def add_spacing_rows(df):
    """
    Add empty rows between different td values
    """
    if df.empty:
        return df
    
    # Extract td values from row names
    td_values = []
    for idx in df.index:
        if pd.notna(idx) and idx != '':
            td_match = re.match(r'td(\d+)_', str(idx))
            if td_match:
                td_values.append(int(td_match.group(1)))
            else:
                td_values.append(-1)
        else:
            td_values.append(-1)
    
    # Create new dataframe with spacing
    new_rows = []
    new_indices = []
    
    prev_td = None
    for i, (idx, td_val) in enumerate(zip(df.index, td_values)):
        # Add spacing row if td value changed (but not for the first row)
        if prev_td is not None and td_val != prev_td and td_val != -1:
            new_rows.append(pd.Series([np.nan] * len(df.columns), index=df.columns))
            new_indices.append('')
        
        new_rows.append(df.iloc[i])
        new_indices.append(idx)
        prev_td = td_val if td_val != -1 else prev_td
    
    result_df = pd.DataFrame(new_rows, index=new_indices)
    return result_df

def get_rankings(df, metric_cols):
    """
    Calculate rankings for each metric column
    """
    rankings = {}
    
    for col in metric_cols:
        if col in df.columns:
            # Remove NaN values and non-numeric values for ranking
            valid_values = []
            for val in df[col]:
                # Only include scalar numeric values
                if not isinstance(val, (list, np.ndarray)) and pd.notna(val) and isinstance(val, (int, float)):
                    valid_values.append(val)
            
            if len(valid_values) == 0:
                continue
            
            valid_series = pd.Series(valid_values)
                
            if col.upper() in ['PSNR', 'SSIM']:
                # Higher is better
                sorted_vals = valid_series.sort_values(ascending=False)
            else:  # DRMSE, LPIPS
                # Lower is better
                sorted_vals = valid_series.sort_values(ascending=True)
            
            # Create ranking dictionary (value -> rank)
            col_rankings = {}
            for rank, val in enumerate(sorted_vals.head(3), 1):
                if val not in col_rankings:  # Handle ties by giving same rank to first occurrence
                    col_rankings[val] = rank
            
            rankings[col] = col_rankings
    
    return rankings

def create_color_matrix(df, rankings):
    """
    Create a color matrix for visualization
    """
    color_matrix = np.full(df.shape, 0, dtype=int)  # 0 = white/no highlight
    
    for col_idx, col in enumerate(df.columns):
        if col in rankings:
            for row_idx, val in enumerate(df[col]):
                # Handle different types of values including arrays
                if isinstance(val, (list, np.ndarray)):
                    # Skip array values for ranking
                    continue
                elif pd.notna(val) and val in rankings[col]:
                    rank = rankings[col][val]
                    color_matrix[row_idx, col_idx] = rank  # 1=red, 2=orange, 3=yellow
    
    return color_matrix

def create_visualization(df, rankings, output_prefix):
    """
    Create PNG and SVG visualizations of the table
    """
    # Set up the plot style
    plt.style.use('default')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 1.5), max(8, len(df.index) * 0.5)))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create color matrix
    color_matrix = create_color_matrix(df, rankings)
    
    # Define colors: 0=white, 1=red, 2=orange, 3=yellow
    colors = ['white', 'red', 'orange', 'yellow']
    
    # Create the table
    table_data = []
    for idx, row in df.iterrows():
        formatted_row = []
        for val in row:
            # Handle different types of values including arrays
            if isinstance(val, (list, np.ndarray)):
                # Convert arrays to string representation
                formatted_row.append(str(val))
            elif pd.isna(val):
                formatted_row.append('')
            elif isinstance(val, float):
                formatted_row.append(f'{val:.4f}')
            else:
                formatted_row.append(str(val))
        table_data.append(formatted_row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Apply colors based on rankings
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            cell = table[(i + 1, j)]  # +1 because of header row
            color_idx = color_matrix[i, j]
            
            if color_idx > 0:
                cell.set_facecolor(colors[color_idx])
                if color_idx in [1, 2]:  # red or orange
                    cell.set_text_props(weight='bold', color='white')
                else:  # yellow
                    cell.set_text_props(weight='bold', color='black')
            else:
                cell.set_facecolor('white')
    
    # Style header row
    for j in range(len(df.columns)):
        cell = table[(0, j)]
        cell.set_facecolor('lightgray')
        cell.set_text_props(weight='bold')
    
    # Style row labels
    for i in range(len(df.index)):
        cell = table[(i + 1, -1)]
        if df.index[i] == '':  # Empty spacing row
            cell.set_facecolor('lightblue')
            cell.set_alpha(0.3)
        else:
            cell.set_facecolor('lightgray')
            cell.set_text_props(weight='bold')
    
    # Add title
    plt.title('Score Analysis Table\n(Red=1st, Orange=2nd, Yellow=3rd best values)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='red', label='1st Best'),
        patches.Patch(color='orange', label='2nd Best'),
        patches.Patch(color='yellow', label='3rd Best')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Save as PNG
    plt.savefig(f'{output_prefix}_table.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"PNG visualization saved to '{output_prefix}_table.png'")
    
    # Save as SVG
    plt.savefig(f'{output_prefix}_table.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"SVG visualization saved to '{output_prefix}_table.svg'")
    
    plt.close()

def create_metrics_plot(df, rankings, output_prefix):
    """
    Create additional visualization showing metric trends
    """
    # Filter out spacing rows and get numeric columns
    df_clean = df[df.index != ''].copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for metrics plot")
        return
    
    # Create subplots for each metric
    n_metrics = len(numeric_cols)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_metrics > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Plot the metric values
        x_pos = range(len(df_clean))
        values = df_clean[col].values
        
        # Color bars based on ranking
        bar_colors = ['lightblue'] * len(values)
        if col in rankings:
            for i, val in enumerate(values):
                # Only apply colors to scalar numeric values
                if not isinstance(val, (list, np.ndarray)) and pd.notna(val) and val in rankings[col]:
                    rank = rankings[col][val]
                    if rank == 1:
                        bar_colors[i] = 'red'
                    elif rank == 2:
                        bar_colors[i] = 'orange'
                    elif rank == 3:
                        bar_colors[i] = 'yellow'
        
        bars = ax.bar(x_pos, values, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Customize the plot
        ax.set_title(f'{col}', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_clean.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            # Only add labels for numeric values
            if not isinstance(val, (list, np.ndarray)) and pd.notna(val) and isinstance(val, (int, float)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save metrics plot
    plt.savefig(f'{output_prefix}_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to '{output_prefix}_metrics.png'")
    
    plt.savefig(f'{output_prefix}_metrics.svg', format='svg', bbox_inches='tight')
    print(f"Metrics plot saved to '{output_prefix}_metrics.svg'")
    
    plt.close()

def create_line_plots(df, output_prefix):
    """
    Create line plots for each metric with different colors for each td value
    """
    # Filter out spacing rows
    df_clean = df[df.index != ''].copy()
    
    # Find metric columns
    metric_cols = []
    for col in df_clean.columns:
        col_upper = col.upper()
        if any(metric in col_upper for metric in ['PSNR', 'SSIM', 'DRMSE', 'LPIPS']):
            metric_cols.append(col)
    
    if len(metric_cols) == 0:
        print("No metric columns found for line plots")
        return
    
    # Extract td and rend values from row names
    td_rend_data = []
    for idx in df_clean.index:
        match = re.match(r'td(\d+)_rend(\d+)', idx)
        if match:
            td_val = int(match.group(1))
            rend_val = int(match.group(2))
            td_rend_data.append((idx, td_val, rend_val))
        else:
            td_rend_data.append((idx, -1, -1))
    
    # Get unique td values and sort them
    unique_tds = sorted(list(set([td for _, td, _ in td_rend_data if td >= 0])))
    
    # Define colors for different td values
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_tds)))
    td_color_map = {td: colors[i] for i, td in enumerate(unique_tds)}
    
    # Create subplots for each metric
    n_metrics = len(metric_cols)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_metrics > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for metric_idx, metric_col in enumerate(metric_cols):
        ax = axes[metric_idx]
        
        # Group data by td value
        td_groups = {}
        for i, (row_name, td_val, rend_val) in enumerate(td_rend_data):
            if td_val >= 0:
                if td_val not in td_groups:
                    td_groups[td_val] = {'x': [], 'y': [], 'labels': [], 'missing': []}
                
                td_groups[td_val]['x'].append(i)
                td_groups[td_val]['y'].append(df_clean.iloc[i][metric_col])
                td_groups[td_val]['labels'].append(row_name)
                # Check if value is missing or non-numeric
                val = df_clean.iloc[i][metric_col]
                is_missing = isinstance(val, (list, np.ndarray)) or pd.isna(val) or not isinstance(val, (int, float))
                td_groups[td_val]['missing'].append(is_missing)
        
        # Plot lines for each td value
        for td_val in sorted(td_groups.keys()):
            group = td_groups[td_val]
            x_vals = group['x']
            y_vals = group['y']
            missing = group['missing']
            
            # Separate continuous and missing segments
            continuous_x, continuous_y = [], []
            missing_segments = []
            
            i = 0
            while i < len(x_vals):
                if not missing[i]:
                    # Start of continuous segment
                    seg_x, seg_y = [x_vals[i]], [y_vals[i]]
                    i += 1
                    
                    # Continue continuous segment
                    while i < len(x_vals) and not missing[i]:
                        seg_x.append(x_vals[i])
                        seg_y.append(y_vals[i])
                        i += 1
                    
                    # Plot continuous segment
                    ax.plot(seg_x, seg_y, 
                           color=td_color_map[td_val], 
                           linewidth=2.5, 
                           marker='o', 
                           markersize=6,
                           label=f'td{td_val}' if len(continuous_x) == 0 else "")
                    
                    continuous_x.extend(seg_x)
                    continuous_y.extend(seg_y)
                else:
                    # Handle missing data
                    if i > 0 and i < len(x_vals) - 1:
                        # Find next valid point
                        next_valid = i + 1
                        while next_valid < len(x_vals) and missing[next_valid]:
                            next_valid += 1
                        
                        if next_valid < len(x_vals):
                            # Draw dotted line from previous valid to next valid
                            prev_idx = i - 1
                            while prev_idx >= 0 and missing[prev_idx]:
                                prev_idx -= 1
                            
                            if prev_idx >= 0:
                                ax.plot([x_vals[prev_idx], x_vals[next_valid]], 
                                       [y_vals[prev_idx], y_vals[next_valid]], 
                                       color=td_color_map[td_val], 
                                       linewidth=2, 
                                       linestyle=':', 
                                       alpha=0.7)
                    i += 1
        
        # Customize the plot
        ax.set_title(f'{metric_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel(metric_col, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(len(df_clean)))
        ax.set_xticklabels(df_clean.index, rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='best')
        
        # Improve y-axis formatting
        if metric_col.upper() in ['PSNR']:
            ax.set_ylabel(f'{metric_col} (dB)', fontsize=12)
        elif metric_col.upper() in ['SSIM']:
            ax.set_ylabel(f'{metric_col} (0-1)', fontsize=12)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title and adjust layout
    fig.suptitle('Metric Trends by Configuration\n(Dotted lines indicate missing data)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save line plots
    plt.savefig(f'{output_prefix}_line_plots.png', dpi=300, bbox_inches='tight')
    print(f"Line plots saved to '{output_prefix}_line_plots.png'")
    
    plt.savefig(f'{output_prefix}_line_plots.svg', format='svg', bbox_inches='tight')
    print(f"Line plots saved to '{output_prefix}_line_plots.svg'")
    
    plt.close()
    
    # Create individual plots for each metric (larger, more detailed)
    for metric_col in metric_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group data by td value
        td_groups = {}
        for i, (row_name, td_val, rend_val) in enumerate(td_rend_data):
            if td_val >= 0:
                if td_val not in td_groups:
                    td_groups[td_val] = {'x': [], 'y': [], 'labels': [], 'missing': []}
                
                td_groups[td_val]['x'].append(i)
                td_groups[td_val]['y'].append(df_clean.iloc[i][metric_col])
                td_groups[td_val]['labels'].append(row_name)
                # Check if value is missing or non-numeric
                val = df_clean.iloc[i][metric_col]
                is_missing = isinstance(val, (list, np.ndarray)) or pd.isna(val) or not isinstance(val, (int, float))
                td_groups[td_val]['missing'].append(is_missing)
        
        # Plot lines for each td value
        for td_val in sorted(td_groups.keys()):
            group = td_groups[td_val]
            x_vals = group['x']
            y_vals = group['y']
            missing = group['missing']
            
            # Plot continuous segments
            i = 0
            first_line = True
            while i < len(x_vals):
                if not missing[i]:
                    # Start of continuous segment
                    seg_x, seg_y = [x_vals[i]], [y_vals[i]]
                    i += 1
                    
                    # Continue continuous segment
                    while i < len(x_vals) and not missing[i]:
                        seg_x.append(x_vals[i])
                        seg_y.append(y_vals[i])
                        i += 1
                    
                    # Plot continuous segment
                    ax.plot(seg_x, seg_y, 
                           color=td_color_map[td_val], 
                           linewidth=3, 
                           marker='o', 
                           markersize=8,
                           label=f'td{td_val}' if first_line else "")
                    first_line = False
                else:
                    # Handle missing data with dotted lines
                    if i > 0 and i < len(x_vals) - 1:
                        # Find next valid point
                        next_valid = i + 1
                        while next_valid < len(x_vals) and missing[next_valid]:
                            next_valid += 1
                        
                        if next_valid < len(x_vals):
                            # Find previous valid point
                            prev_idx = i - 1
                            while prev_idx >= 0 and missing[prev_idx]:
                                prev_idx -= 1
                            
                            if prev_idx >= 0:
                                ax.plot([x_vals[prev_idx], x_vals[next_valid]], 
                                       [y_vals[prev_idx], y_vals[next_valid]], 
                                       color=td_color_map[td_val], 
                                       linewidth=2.5, 
                                       linestyle=':', 
                                       alpha=0.8)
                    i += 1
        
        # Customize the individual plot
        ax.set_title(f'{metric_col} by Configuration', fontsize=16, fontweight='bold')
        ax.set_xlabel('Configuration', fontsize=14)
        ax.set_ylabel(f'{metric_col}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(range(len(df_clean)))
        ax.set_xticklabels(df_clean.index, rotation=45, ha='right', fontsize=10)
        
        # Add legend
        ax.legend(loc='best', fontsize=12)
        
        # Add text box with statistics
        # Only calculate statistics for numeric values
        numeric_values = []
        for val in df_clean[metric_col]:
            if not isinstance(val, (list, np.ndarray)) and pd.notna(val) and isinstance(val, (int, float)):
                numeric_values.append(val)
        
        if len(numeric_values) > 0:
            numeric_series = pd.Series(numeric_values)
            stats_text = f'Mean: {numeric_series.mean():.4f}\nStd: {numeric_series.std():.4f}\nMin: {numeric_series.min():.4f}\nMax: {numeric_series.max():.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual metric plot
        safe_metric_name = metric_col.replace('/', '_').replace(' ', '_')
        plt.savefig(f'{output_prefix}_{safe_metric_name}_individual.png', dpi=300, bbox_inches='tight')
        print(f"Individual {metric_col} plot saved to '{output_prefix}_{safe_metric_name}_individual.png'")
        
        plt.savefig(f'{output_prefix}_{safe_metric_name}_individual.svg', format='svg', bbox_inches='tight')
        print(f"Individual {metric_col} plot saved to '{output_prefix}_{safe_metric_name}_individual.svg'")
        
        plt.close()

def main():
    """
    Main function to process directories and create the analysis table
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Analyze score files from subdirectories and create visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 score_analysis.py /path/to/parent/directory
  python3 score_analysis.py ./experiments --output results
        """
    )
    
    parser.add_argument('directory', 
                       help='Path to parent directory to search for subdirectories')
    parser.add_argument('--output', '-o', default='score_analysis',
                       help='Output prefix for generated files (default: score_analysis)')
    parser.add_argument('--pattern', default='carla_on_nuscene_0shot',
                       help='Pattern prefix for subdirectory matching (default: carla_on_nuscene_0shot)')
    
    args = parser.parse_args()
    
    base_directory = args.directory
    output_prefix = args.output
    pattern_prefix = args.pattern
    
    print(f"Searching for subdirectories in: {os.path.abspath(base_directory)}")
    print(f"Pattern: {pattern_prefix}_tdX_rendY")
    print(f"Output prefix: {output_prefix}")
    
    # Find matching subdirectories
    subdirs = find_subdirectories(base_directory, pattern_prefix)
    
    if not subdirs:
        print("No matching subdirectories found!")
        print("Expected pattern: {pattern_prefix}_td<number>_rend<number>")
        return 1
    
    print(f"Found {len(subdirs)} matching subdirectories:")
    for subdir, td, rend in subdirs:
        print(f"  {subdir}")
    
    # Load data from each subdirectory
    data_rows = []
    row_names = []
    
    for subdir_name, td_val, rend_val in subdirs:
        subdir_path = os.path.join(base_directory, subdir_name)
        scores = load_scores(subdir_path)
        
        if scores is not None:
            row_name = create_row_name(td_val, rend_val)
            data_rows.append(scores)
            row_names.append(row_name)
            print(f"Loaded data for {row_name}: {len(scores)} metrics")
        else:
            print(f"Skipping {subdir_name} due to missing or invalid scores file")
    
    if not data_rows:
        print("No valid data found!")
        return 1
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, index=row_names)
    
    # Remove encoder and decoder columns if they exist
    columns_to_remove = ['encoder', 'decoder']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Removed column: {col}")
    
    # Add spacing rows between different td values
    df_with_spacing = add_spacing_rows(df)
    
    print("\nDataFrame created successfully!")
    print(f"Shape: {df_with_spacing.shape}")
    print(f"Columns: {list(df_with_spacing.columns)}")
    
    # Define metric columns to highlight
    metric_cols = []
    for col in df_with_spacing.columns:
        col_upper = col.upper()
        if any(metric in col_upper for metric in ['PSNR', 'SSIM', 'DRMSE', 'LPIPS']):
            metric_cols.append(col)
    
    print(f"Metrics to highlight: {metric_cols}")
    
    # Calculate rankings
    rankings = get_rankings(df_with_spacing, metric_cols)
    
    # Create visualizations
    try:
        create_visualization(df_with_spacing, rankings, output_prefix)
        create_metrics_plot(df_with_spacing, rankings, output_prefix)
        create_line_plots(df_with_spacing, output_prefix)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib pandas")
        import traceback
        traceback.print_exc()
    
    # Save raw data to CSV
    df_with_spacing.to_csv(f"{output_prefix}_raw.csv")
    print(f"Raw data saved to '{output_prefix}_raw.csv'")
    
    # Print the table to console
    print("\n" + "="*100)
    print("RESULTS TABLE")
    print("="*100)
    print(df_with_spacing.to_string())
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    df_clean = df_with_spacing[df_with_spacing.index != ''].copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df_clean[numeric_cols].describe())
    
    print(f"\nAll output files saved with prefix: {output_prefix}")
    return 0

if __name__ == "__main__":
    sys.exit(main())