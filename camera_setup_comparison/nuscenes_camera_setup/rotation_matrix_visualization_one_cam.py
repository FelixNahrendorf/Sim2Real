#!/usr/bin/env python3
"""
Camera Data Analysis and Visualization Tool

Usage:
    python3 rotation_matrix_visualization_one_cam.py "/path/to/cameras/*.json" --output-dir rotation_matrix_visualization_one_cam
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import glob
from pathlib import Path
import sys

def load_and_parse_data(file_paths):
    """
    Load JSON data from multiple files and extract translation and rotation coordinates.
    
    Args:
        file_paths (list): List of paths to JSON files
        
    Returns:
        tuple: (translation_data, rotation_data, file_info) as numpy arrays and list
    """
    all_translations = []
    all_rotations = []
    file_info = []
    
    for file_path in file_paths:
        try:
            print(f"Processing: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both single objects and arrays
            if isinstance(data, dict):
                data = [data]
            
            # Extract translation coordinates (x, y, z)
            translations = [item['translation'] for item in data if 'translation' in item]
            rotations = [item['rotation'] for item in data if 'rotation' in item]
            
            all_translations.extend(translations)
            all_rotations.extend(rotations)
            
            file_info.append({
                'file': Path(file_path).name,
                'samples': len(translations)
            })
            
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    if not all_translations:
        raise ValueError("No valid translation data found in any files")
    
    return np.array(all_translations), np.array(all_rotations), file_info

def compute_statistics(data, labels):
    """
    Compute mean, min, max for each coordinate.
    
    Args:
        data (np.array): Data array with shape (n_samples, n_coordinates)
        labels (list): Labels for each coordinate
        
    Returns:
        pd.DataFrame: Statistics summary
    """
    stats = []
    for i, label in enumerate(labels):
        coordinate_data = data[:, i]
        stats.append({
            'Coordinate': label,
            'Mean': np.mean(coordinate_data),
            'Min': np.min(coordinate_data),
            'Max': np.max(coordinate_data),
            'Std': np.std(coordinate_data),
            'Median': np.median(coordinate_data),
            'Q1': np.percentile(coordinate_data, 25),
            'Q3': np.percentile(coordinate_data, 75)
        })
    
    return pd.DataFrame(stats)

def create_visualizations(translations, rotations, output_dir):
    """
    Create comprehensive visualizations for translation and rotation data.
    
    Args:
        translations (np.array): Translation data
        rotations (np.array): Rotation data
        output_dir (Path): Output directory for PNG files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots for histograms
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Camera Data Distribution Analysis', fontsize=18, fontweight='bold')
    
    # Translation labels
    translation_labels = ['X', 'Y', 'Z']
    
    # Rotation labels (quaternion components)
    rotation_labels = ['W', 'X', 'Y', 'Z']
    
    # Colors for different coordinates
    translation_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    rotation_colors = ['#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    # Plot translation distributions
    for i, (label, color) in enumerate(zip(translation_labels, translation_colors)):
        # Histogram
        axes[0, i].hist(translations[:, i], bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[0, i].set_title(f'Translation {label}', fontweight='bold', fontsize=14)
        axes[0, i].set_xlabel('Value', fontsize=12)
        axes[0, i].set_ylabel('Frequency', fontsize=12)
        axes[0, i].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(translations[:, i])
        std_val = np.std(translations[:, i])
        axes[0, i].axvline(mean_val, color='red', linestyle='--', linewidth=2)
        axes[0, i].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                       transform=axes[0, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add empty subplot for translation summary
    axes[0, 3].axis('off')
    trans_summary = f"Translation Summary\nTotal samples: {len(translations)}\n"
    trans_summary += f"X range: [{np.min(translations[:, 0]):.3f}, {np.max(translations[:, 0]):.3f}]\n"
    trans_summary += f"Y range: [{np.min(translations[:, 1]):.3f}, {np.max(translations[:, 1]):.3f}]\n"
    trans_summary += f"Z range: [{np.min(translations[:, 2]):.3f}, {np.max(translations[:, 2]):.3f}]"
    axes[0, 3].text(0.1, 0.9, trans_summary, transform=axes[0, 3].transAxes,
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot rotation distributions
    for i, (label, color) in enumerate(zip(rotation_labels, rotation_colors)):
        # Histogram
        axes[1, i].hist(rotations[:, i], bins=30, alpha=0.7, color=color, edgecolor='black')
        axes[1, i].set_title(f'Rotation {label}', fontweight='bold', fontsize=14)
        axes[1, i].set_xlabel('Value', fontsize=12)
        axes[1, i].set_ylabel('Frequency', fontsize=12)
        axes[1, i].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(rotations[:, i])
        std_val = np.std(rotations[:, i])
        axes[1, i].axvline(mean_val, color='red', linestyle='--', linewidth=2)
        axes[1, i].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                       transform=axes[1, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    histogram_file = output_dir / 'camera_data_histograms.png'
    plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create boxplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle('Camera Data Boxplot Analysis', fontsize=18, fontweight='bold')
    
    # Translation boxplot
    bp1 = axes[0].boxplot([translations[:, i] for i in range(3)], 
                         labels=translation_labels,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=2),
                         capprops=dict(linewidth=2))
    axes[0].set_title('Translation Coordinates', fontweight='bold', fontsize=16)
    axes[0].set_ylabel('Value', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Rotation boxplot
    bp2 = axes[1].boxplot([rotations[:, i] for i in range(4)], 
                         labels=rotation_labels,
                         patch_artist=True,
                         boxprops=dict(facecolor='lightgreen', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=2),
                         capprops=dict(linewidth=2))
    axes[1].set_title('Rotation Quaternion Components', fontweight='bold', fontsize=16)
    axes[1].set_ylabel('Value', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    boxplot_file = output_dir / 'camera_data_boxplots.png'
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return histogram_file, boxplot_file

def analyze_camera_data(input_pattern, output_dir):
    """
    Main function to analyze camera data and create visualizations.
    
    Args:
        input_pattern (str): Glob pattern for input JSON files
        output_dir (str): Output directory for results
        
    Returns:
        tuple: (translation_stats, rotation_stats, output_files)
    """
    output_path = Path(output_dir)
    
    # Find matching files
    file_paths = glob.glob(input_pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files found matching pattern: {input_pattern}")
    
    print(f"Found {len(file_paths)} files matching pattern")
    
    # Load and parse data
    print("Loading and parsing data...")
    translations, rotations, file_info = load_and_parse_data(file_paths)
    
    total_samples = len(translations)
    print(f"Loaded {total_samples} total data points from {len(file_info)} files")
    
    # Print file information
    print("\nFile processing summary:")
    for info in file_info:
        print(f"  {info['file']}: {info['samples']} samples")
    
    # Compute statistics
    translation_labels = ['Translation_X', 'Translation_Y', 'Translation_Z']
    rotation_labels = ['Rotation_W', 'Rotation_X', 'Rotation_Y', 'Rotation_Z']
    
    translation_stats = compute_statistics(translations, translation_labels)
    rotation_stats = compute_statistics(rotations, rotation_labels)
    
    # Print statistics
    print("\n" + "="*50)
    print("TRANSLATION STATISTICS")
    print("="*50)
    print(translation_stats.round(6).to_string(index=False))
    
    print("\n" + "="*50)
    print("ROTATION STATISTICS")
    print("="*50)
    print(rotation_stats.round(6).to_string(index=False))
    
    # Create visualizations
    print(f"\nCreating visualizations in: {output_path}")
    output_files = create_visualizations(translations, rotations, output_path)
    
    # Save statistics to CSV files
    translation_stats.to_csv(output_path / 'translation_statistics.csv', index=False)
    rotation_stats.to_csv(output_path / 'rotation_statistics.csv', index=False)
    
    # Save file info
    pd.DataFrame(file_info).to_csv(output_path / 'file_processing_summary.csv', index=False)
    
    print(f"\nAnalysis complete!")
    print("Files generated:")
    print(f"  - {output_files[0]}")
    print(f"  - {output_files[1]}")
    print(f"  - {output_path / 'translation_statistics.csv'}")
    print(f"  - {output_path / 'rotation_statistics.csv'}")
    print(f"  - {output_path / 'file_processing_summary.csv'}")
    
    return translation_stats, rotation_stats, output_files

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze camera calibration data and create visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 rotation_matrix_visualization_one_cam.py "/path/to/cameras/*.json" --output-dir results
    python3 rotation_matrix_visualization_one_cam.py "data/camera_*.json" --output-dir analysis_output
    python3 rotation_matrix_visualization_one_cam.py "single_file.json" --output-dir my_results
        """
    )
    
    parser.add_argument('input_pattern', 
                       help='Input file pattern (supports glob patterns like "/path/to/*.json")')
    parser.add_argument('--output-dir', '-o', 
                       required=True,
                       help='Output directory for results and visualizations')
    
    args = parser.parse_args()
    
    try:
        analyze_camera_data(args.input_pattern, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
