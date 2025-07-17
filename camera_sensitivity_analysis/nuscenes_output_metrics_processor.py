import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def process_json_files(base_dir="/app/outputs"):
    """
    Process JSON files from model runs and extract metrics
    """
    data = []
    
    # Find all directories that match the pattern
    base_path = Path(base_dir)
    
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith("ego_exo_sensitivity_analysis_"):
            # Extract model run name (everything after "ego_exo_sensitivity_analysis_")
            model_run_name = folder.name.replace("ego_exo_sensitivity_analysis_", "")
            
            # Filter out any model run that contains "1" in the name
            if "1" in model_run_name:
                print(f"Filtered out (contains '1'): {model_run_name}")
                continue
            
            # Look for seed subdirectory and scores_all_avg.json file
            seed_dir = folder / "seed"
            json_file = seed_dir / "scores_all_avg.json"
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        scores = json.load(f)
                    
                    # Extract required metrics
                    row_data = {
                        'model_run': model_run_name,
                        'psnr': scores.get('psnr', None),
                        'ssim': scores.get('ssim', None),
                        'lpips': scores.get('lpips', None),
                        'drmse': scores.get('drmse', None),
                        'compute_time': scores.get('compute_time', None)
                    }
                    
                    data.append(row_data)
                    print(f"Processed: {model_run_name}")
                    
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error processing {json_file}: {e}")
    
    return pd.DataFrame(data)

def add_baseline_differences(df):
    """
    Add columns showing PSNR differences from baseline runs
    """
    # Find baseline runs (assuming they contain 'baseline' in the name)
    baseline_runs = df[df['model_run'].str.contains('baseline', case=False, na=False)]
    
    if len(baseline_runs) == 0:
        print("Warning: No baseline runs found")
        return df
    
    # Add difference columns for each baseline
    for idx, baseline_row in baseline_runs.iterrows():
        baseline_name = baseline_row['model_run']
        baseline_psnr = baseline_row['psnr']
        
        if baseline_psnr is not None:
            col_name = f'psnr_diff_from_{baseline_name}'
            df[col_name] = df['psnr'] - baseline_psnr
    
    return df

def create_table_plot(df, output_path="metrics_table.png"):
    """
    Create a matplotlib table visualization and save as PNG
    """
    # Round numerical values for better display
    df_display = df.copy()
    numeric_columns = ['psnr', 'ssim', 'lpips', 'drmse', 'compute_time']
    
    for col in numeric_columns:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    # Round difference columns
    diff_columns = [col for col in df_display.columns if col.startswith('psnr_diff_from_')]
    for col in diff_columns:
        df_display[col] = df_display[col].round(4)
    
    # Find the baseline row
    baseline_row = None
    baseline_idx = None
    for idx, row in df_display.iterrows():
        if row['model_run'] == 'baseline_nuscene_adjusted':
            baseline_row = row
            baseline_idx = idx
            break
    
    if baseline_row is None:
        print("Warning: baseline_nuscene_adjusted not found in the data")
        # Fallback to any baseline row
        for idx, row in df_display.iterrows():
            if 'baseline' in row['model_run'].lower():
                baseline_row = row
                baseline_idx = idx
                break
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table with custom column widths
    table = ax.table(cellText=df_display.values,
                    colLabels=df_display.columns,
                    cellLoc='center',
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Adjust column widths - make baseline difference columns wider
    for i, col_name in enumerate(df_display.columns):
        if col_name == 'model_run':
            # Make first column (model_run)  wider
            for j in range(len(df_display) + 1):  # +1 for header
                table[(j, i)].set_width(0.195)  # 95% wider than default (0.1)
        elif col_name in ['psnr', 'ssim', 'lpips', 'drmse', 'compute_time']:
            # Make these columns narrower
            for j in range(len(df_display) + 1):  # +1 for header
                table[(j, i)].set_width(0.065)  # 65% of default width (0.1)
        elif col_name.startswith('psnr_diff_from_'):
            # Make baseline difference columns wider
            for j in range(len(df_display) + 1):  # +1 for header
                table[(j, i)].set_width(0.2)  # 100% wider than default (0.1)
    
    # Color the header
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Function to get color for ranking (red to yellow gradient)
    def get_rank_color(rank):
        """Get color based on rank (1=red, 5=yellow)"""
        colors = ['#FF6666', '#FF9933', '#FFC233', '#FFE066', '#FFFF99']  # red to yellow
        return colors[rank - 1] if rank <= 5 else None
    
    def get_rank_color_reversed(rank):
        """Get color based on rank for lpips (1=yellow, 5=red) - reversed"""
        colors = ['#FFFF99', '#FFE066', '#FFC233', '#FF9933', '#FF6666']  # yellow to red
        return colors[rank - 1] if rank <= 5 else None
    
    def get_green_for_best():
        """Get green color for best (lowest) compute_time"""
        return '#90EE90'  # light green

    # Color cells based on ranking for each numeric column
    for col_idx, col_name in enumerate(df_display.columns):
        if col_name in numeric_columns or col_name.startswith('psnr_diff_from_'):
            # Get column values (excluding NaN)
            col_values = df_display[col_name].dropna()
            
            if len(col_values) > 0:
                # Handle different coloring schemes based on column
                if col_name == 'lpips':
                    # For lpips, color the 5 HIGHEST values (worst performance)
                    sorted_values = col_values.sort_values(ascending=False)  # highest first
                    
                    for row_idx, value in enumerate(df_display[col_name]):
                        if pd.notna(value):
                            # Find rank in sorted values (1-based, highest first)
                            rank = (sorted_values == value).idxmax()
                            position = list(sorted_values.index).index(rank) + 1
                            
                            if position <= 5:
                                color = get_rank_color_reversed(position)
                                if color:
                                    table[(row_idx + 1, col_idx)].set_facecolor(color)
                
                elif col_name == 'compute_time':
                    # For compute_time, only color the lowest value with green
                    min_value = col_values.min()
                    
                    for row_idx, value in enumerate(df_display[col_name]):
                        if pd.notna(value) and value == min_value:
                            table[(row_idx + 1, col_idx)].set_facecolor(get_green_for_best())
                
                else:
                    # Default: color the 5 LOWEST values (best performance)
                    sorted_values = col_values.sort_values()  # lowest first
                    
                    for row_idx, value in enumerate(df_display[col_name]):
                        if pd.notna(value):
                            # Find rank in sorted values (1-based)
                            rank = (sorted_values == value).idxmax()
                            position = list(sorted_values.index).index(rank) + 1
                            
                            if position <= 5:
                                color = get_rank_color(position)
                                if color:
                                    table[(row_idx + 1, col_idx)].set_facecolor(color)
    
    # Override baseline row coloring (but keep rank colors for numeric columns)
    for idx, row in df_display.iterrows():
        if 'baseline' in row['model_run'].lower():
            # Only color non-numeric columns for baseline highlighting
            for j, col_name in enumerate(df_display.columns):
                if col_name not in numeric_columns and not col_name.startswith('psnr_diff_from_'):
                    table[(idx + 1, j)].set_facecolor('#f0f0f0')

    # Add blue highlighting for values better than baseline_nuscene_adjusted
    if baseline_row is not None:
        for col_idx, col_name in enumerate(df_display.columns):
            if col_name in numeric_columns:
                # Skip compute_time column for blue highlighting
                if col_name == 'compute_time':
                    continue
                
                baseline_value = baseline_row[col_name]
                
                if pd.notna(baseline_value):
                    for row_idx, value in enumerate(df_display[col_name]):
                        if pd.notna(value) and row_idx != baseline_idx:  # Don't color baseline row
                            should_highlight = False
                            
                            # Special case for drmse: only highlight when < baseline
                            if col_name == 'drmse':
                                should_highlight = value < baseline_value
                            # For lpips, lower is better (highlight when < baseline)
                            elif col_name == 'lpips':
                                should_highlight = value < baseline_value
                            # For all other metrics, higher is better (highlight when > baseline)
                            else:
                                should_highlight = value > baseline_value
                            
                            if should_highlight:
                                table[(row_idx + 1, col_idx)].set_facecolor('#ADD8E6')  # Light blue
    
    plt.title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Table plot saved as: {output_path}")

def main():
    # Process JSON files
    print("Processing JSON files...")
    df = process_json_files()
    
    if df.empty:
        print("No data found. Please check the directory structure and file paths.")
        return
    
    # Add baseline differences
    print("Computing baseline differences...")
    df = add_baseline_differences(df)
    
    # Sort by model run name for better readability
    df = df.sort_values('model_run')
    
    # Save CSV
    csv_output = "metrics_comparison.csv"
    df.to_csv(csv_output, index=False)
    print(f"CSV saved as: {csv_output}")
    
    # Create and save table plot
    print("Creating table visualization...")
    create_table_plot(df)
    
    # Display summary
    print(f"\nProcessed {len(df)} model runs:")
    print(df['model_run'].tolist())
    
    # Display the dataframe
    print("\nMetrics Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()