import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from scipy import stats

def load_camera_data(file_path):
    """Load camera data from JSON file and extract parameters."""
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

def analyze_camera_files(directory_path):
    """Analyze all camera JSON files in the directory."""
    directory = Path(directory_path)
    camera_data = {}
    camera_stats = {}
    
    # Find all JSON files
    json_files = list(directory.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return None, None
    
    for file_path in json_files:
        camera_name = file_path.stem  # Get filename without extension
        try:
            data = load_camera_data(file_path)
            camera_data[camera_name] = data
            camera_stats[camera_name] = compute_statistics(data)
            print(f"Loaded data for {camera_name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return camera_data, camera_stats

def create_visualization(camera_data, camera_stats):
    """Create comprehensive visualization of camera parameters."""
    if not camera_data:
        print("No camera data to visualize")
        return
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Camera Parameters', fontsize=16, fontweight='bold')
    
    # Parameters to plot
    params = ['x', 'y', 'z', 'pitch', 'yaw', 'fov']
    param_labels = ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 
                   'Pitch (rad)', 'Yaw (rad)', 'FOV (degrees)']
    
    # Color palette for different cameras
    colors = plt.cm.Set1(np.linspace(0, 1, len(camera_data)))
    
    # Plot each parameter
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i//3, i%3]
        
        # Plot distribution for each camera
        for j, (camera_name, data) in enumerate(camera_data.items()):
            values = data[param]
            
            # Plot histogram
            ax.hist(values, bins=20, alpha=0.6, color=colors[j], 
                   label=f'{camera_name}', density=True)
            
            # Plot mean line
            mean_val = camera_stats[camera_name][param]['mean']
            ax.axvline(mean_val, color=colors[j], linestyle='--', 
                      linewidth=2, alpha=0.8)
        
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(f'{label} Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('camera_parameters_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('camera_parameters_distribution.svg', bbox_inches='tight')
    print("Distribution plot saved as 'camera_parameters_distribution.png' and 'camera_parameters_distribution.svg'")
    
    plt.show()

def create_statistics_table(camera_stats):
    """Create and display statistics table."""
    params = ['x', 'y', 'z', 'pitch', 'yaw', 'fov']
    
    print("\n" + "="*100)
    print("CAMERA PARAMETERS STATISTICS")
    print("="*100)
    
    for param in params:
        print(f"\n{param.upper()} STATISTICS:")
        print("-" * 80)
        print(f"{'Camera':<20} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
        print("-" * 80)
        
        for camera_name, stats in camera_stats.items():
            param_stats = stats[param]
            print(f"{camera_name:<20} {param_stats['mean']:<12.6f} {param_stats['std']:<12.6f} "
                  f"{param_stats['min']:<12.6f} {param_stats['max']:<12.6f} {param_stats['median']:<12.6f}")

def create_box_plots(camera_data):
    """Create box plots for parameter comparison across cameras."""
    params = ['x', 'y', 'z', 'pitch', 'yaw', 'fov']
    param_labels = ['X Coordinate', 'Y Coordinate', 'Z Coordinate', 
                   'Pitch (rad)', 'Yaw (rad)', 'FOV (degrees)']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Camera Parameters - Box Plots Comparison', fontsize=16, fontweight='bold')
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i//3, i%3]
        
        # Prepare data for box plot
        data_for_boxplot = []
        labels = []
        
        for camera_name, data in camera_data.items():
            data_for_boxplot.append(data[param])
            labels.append(camera_name)
        
        # Create box plot
        bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set1(np.linspace(0, 1, len(camera_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('camera_parameters_boxplots.png', dpi=300, bbox_inches='tight')
    plt.savefig('camera_parameters_boxplots.svg', bbox_inches='tight')
    print("Box plots saved as 'camera_parameters_boxplots.png' and 'camera_parameters_boxplots.svg'")
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Example usage
    directory_path = "."  # Current directory - change this to your data directory
    
    print("Analyzing camera parameter files...")
    camera_data, camera_stats = analyze_camera_files(directory_path)
    
    if camera_data and camera_stats:
        # Create visualizations
        create_visualization(camera_data, camera_stats)
        create_box_plots(camera_data)
        
        # Display statistics table
        create_statistics_table(camera_stats)
        
        # Optional: Save statistics to CSV
        save_to_csv = input("\nSave statistics to CSV? (y/n): ").lower().strip() == 'y'
        if save_to_csv:
            # Create DataFrame for CSV export
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
            df.to_csv('camera_parameters_statistics.csv', index=False)
            print("Statistics saved to 'camera_parameters_statistics.csv'")
    else:
        print("No valid camera data found. Please check your file paths and formats.")
