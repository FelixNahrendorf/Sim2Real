#!/usr/bin/env python3
"""
Script to filter out duplicate camera poses from nuScenes transformed JSON files.
Each pose is defined by coordinates (x,y,z), roll, pitch, yaw, and fov values.
Creates filtered JSON files and a frequency report.
"""

import json
import sys
import os
import glob
import argparse
from typing import Dict, List, Tuple, Any
from collections import Counter


def read_camera_data(file_path: str) -> Dict[str, List]:
    """
    Read camera data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing coordinates, rolls, pitchs, yaws, and fov lists
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate required keys
        required_keys = ['coordinates', 'rolls', 'pitchs', 'yaws', 'fov']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)


def create_pose_key(coord: List[float], roll: float, pitch: float, yaw: float, 
                    fov: float, precision: int = 8) -> Tuple:
    """
    Create a hashable key for a camera pose by rounding values to specified precision.
    
    Args:
        coord: [x, y, z] coordinates
        roll: Roll angle
        pitch: Pitch angle
        yaw: Yaw angle
        fov: Field of view
        precision: Number of decimal places for rounding
        
    Returns:
        Tuple representing the unique pose
    """
    return (
        round(coord[0], precision),
        round(coord[1], precision),
        round(coord[2], precision),
        round(roll, precision),
        round(pitch, precision),
        round(yaw, precision),
        round(fov, precision)
    )


def format_pose_key(pose_key: Tuple) -> str:
    """
    Format a pose key as a readable string.
    
    Args:
        pose_key: Tuple representing the pose
        
    Returns:
        Formatted string
    """
    return (f"coord=({pose_key[0]:.6f}, {pose_key[1]:.6f}, {pose_key[2]:.6f}), "
            f"roll={pose_key[3]:.6f}, pitch={pose_key[4]:.6f}, "
            f"yaw={pose_key[5]:.6f}, fov={pose_key[6]:.6f}")


def filter_duplicates(data: Dict[str, List], precision: int = 8) -> Tuple[Dict[str, List], Counter, Dict]:
    """
    Filter out duplicate camera poses and count frequencies.
    
    Args:
        data: Dictionary containing camera data
        precision: Number of decimal places for comparison
        
    Returns:
        Tuple of (filtered_data, pose_counter, pose_indices)
        - filtered_data: Dictionary with unique poses only
        - pose_counter: Counter of how many times each pose appears
        - pose_indices: Dictionary mapping pose_key to list of all indices where it appears
    """
    coordinates = data['coordinates']
    rolls = data['rolls']
    pitchs = data['pitchs']
    yaws = data['yaws']
    fovs = data['fov']
    
    # Validate that all arrays have the same length
    lengths = [len(coordinates), len(rolls), len(pitchs), len(yaws), len(fovs)]
    if len(set(lengths)) > 1:
        raise ValueError(f"Array lengths don't match: coordinates={len(coordinates)}, "
                        f"rolls={len(rolls)}, pitchs={len(pitchs)}, yaws={len(yaws)}, "
                        f"fov={len(fovs)}")
    
    pose_counter = Counter()
    pose_indices = {}  # Maps pose_key to list of all indices where it appears
    seen_poses = {}  # Maps pose_key to first occurrence index
    unique_indices = []
    
    for i in range(len(coordinates)):
        pose_key = create_pose_key(coordinates[i], rolls[i], pitchs[i], 
                                   yaws[i], fovs[i], precision)
        
        # Count this pose
        pose_counter[pose_key] += 1
        
        # Track all indices for this pose
        if pose_key not in pose_indices:
            pose_indices[pose_key] = []
        pose_indices[pose_key].append(i)
        
        # Track first occurrence
        if pose_key not in seen_poses:
            seen_poses[pose_key] = i
            unique_indices.append(i)
    
    # Create filtered data
    filtered_data = {
        'coordinates': [coordinates[i] for i in unique_indices],
        'rolls': [rolls[i] for i in unique_indices],
        'pitchs': [pitchs[i] for i in unique_indices],
        'yaws': [yaws[i] for i in unique_indices],
        'fov': [fovs[i] for i in unique_indices]
    }
    
    return filtered_data, pose_counter, pose_indices


def save_filtered_data(data: Dict[str, List], output_path: str) -> None:
    """
    Save filtered data to JSON file.
    
    Args:
        data: Filtered camera data
        output_path: Path to save the output file
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  Error saving file '{output_path}': {e}")
        sys.exit(1)


def print_stats(original_data: Dict[str, List], filtered_data: Dict[str, List]) -> None:
    """
    Print statistics about the filtering process.
    
    Args:
        original_data: Original camera data
        filtered_data: Filtered camera data
    """
    original_count = len(original_data['coordinates'])
    filtered_count = len(filtered_data['coordinates'])
    duplicates_removed = original_count - filtered_count
    
    print(f"  Original poses: {original_count}")
    print(f"  Unique poses: {filtered_count}")
    print(f"  Duplicates removed: {duplicates_removed}")
    if original_count > 0:
        print(f"  Duplicate percentage: {(duplicates_removed / original_count) * 100:.1f}%")


def process_files(input_pattern: str, output_dir: str = None, 
                 precision: int = 8) -> Tuple[Dict[str, Counter], Dict[str, Dict]]:
    """
    Process all JSON files matching the pattern.
    
    Args:
        input_pattern: Glob pattern for input files
        output_dir: Directory to save output files (optional)
        precision: Decimal places for comparison
        
    Returns:
        Tuple of (all_pose_counters, all_pose_indices)
        - all_pose_counters: Dictionary mapping filename to pose counter
        - all_pose_indices: Dictionary mapping filename to pose indices dict
    """
    # Find all matching files
    input_files = sorted(glob.glob(input_pattern))
    
    if not input_files:
        print(f"Error: No files found matching pattern '{input_pattern}'")
        sys.exit(1)
    
    print(f"Found {len(input_files)} file(s) to process:")
    for file in input_files:
        print(f"  - {file}")
    print()
    
    # Process each file
    all_pose_counters = {}
    all_pose_indices = {}
    
    for i, input_file in enumerate(input_files, 1):
        print(f"=== Processing file {i}/{len(input_files)}: {os.path.basename(input_file)} ===")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{base_name}_unique.json"
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, output_filename)
        else:
            input_dir = os.path.dirname(input_file)
            output_file = os.path.join(input_dir, output_filename) if input_dir else output_filename
        
        try:
            # Read and process data
            original_data = read_camera_data(input_file)
            filtered_data, pose_counter, pose_indices = filter_duplicates(original_data, precision)
            
            # Save results
            save_filtered_data(filtered_data, output_file)
            print_stats(original_data, filtered_data)
            
            # Store pose counter and indices for report
            all_pose_counters[os.path.basename(input_file)] = pose_counter
            all_pose_indices[os.path.basename(input_file)] = pose_indices
            
        except Exception as e:
            print(f"  Error processing {input_file}: {e}")
            continue
        
        print()
    
    return all_pose_counters, all_pose_indices


def generate_report(all_pose_counters: Dict[str, Counter], 
                   all_pose_indices: Dict[str, Dict], 
                   output_file: str) -> None:
    """
    Generate a text report showing unique poses, their frequencies, and indices.
    
    Args:
        all_pose_counters: Dictionary mapping filename to pose counter
        all_pose_indices: Dictionary mapping filename to pose indices dict
        output_file: Path to save the report
    """
    try:
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CAMERA POSE FREQUENCY REPORT\n")
            f.write("="*80 + "\n\n")
            
            for filename in sorted(all_pose_counters.keys()):
                pose_counter = all_pose_counters[filename]
                pose_indices = all_pose_indices[filename]
                
                f.write(f"\nFile: {filename}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total unique poses: {len(pose_counter)}\n")
                f.write(f"Total pose instances: {sum(pose_counter.values())}\n\n")
                
                # Sort poses by frequency (descending) and then by pose values
                sorted_poses = sorted(pose_counter.items(), 
                                     key=lambda x: (-x[1], x[0]))
                
                f.write("Unique Poses (sorted by frequency):\n\n")
                for idx, (pose_key, count) in enumerate(sorted_poses, 1):
                    f.write(f"Pose #{idx} (appears {count} times):\n")
                    f.write(f"  {format_pose_key(pose_key)}\n")
                    
                    # Get and format the indices
                    indices = pose_indices[pose_key]
                    f.write(f"  Indices: {format_indices_list(indices)}\n\n")
                
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)


def format_indices_list(indices: List[int], max_per_line: int = 15) -> str:
    """
    Format a list of indices for display, with line wrapping.
    
    Args:
        indices: List of integer indices
        max_per_line: Maximum number of indices to show per line
        
    Returns:
        Formatted string with indices
    """
    if not indices:
        return "[]"
    
    # Convert to strings
    idx_strs = [str(i) for i in sorted(indices)]
    
    # If the list is short, show it all on one line
    if len(idx_strs) <= max_per_line:
        return "[" + ", ".join(idx_strs) + "]"
    
    # Otherwise, format with line breaks
    lines = []
    for i in range(0, len(idx_strs), max_per_line):
        chunk = idx_strs[i:i + max_per_line]
        if i == 0:
            lines.append("[" + ", ".join(chunk) + ",")
        elif i + max_per_line >= len(idx_strs):
            lines.append("           " + ", ".join(chunk) + "]")
        else:
            lines.append("           " + ", ".join(chunk) + ",")
    
    return "\n".join(lines)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter duplicate camera poses from nuScenes transformed JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using directory path (will process all *_transformed.json files):
  python3 filter_camera_duplicates.py /path/to/transformed/
  python3 filter_camera_duplicates.py /path/to/transformed/ --output-dir ./results
  
  # Using directory with custom pattern:
  python3 filter_camera_duplicates.py /path/to/transformed/ --pattern "CAM_FRONT*.json"
  python3 filter_camera_duplicates.py . --pattern "*.json"
  
  # Using glob pattern directly (original behavior):
  python3 filter_camera_duplicates.py "CAM_*_transformed.json"
  python3 filter_camera_duplicates.py "*.json" --output-dir ./filtered
        """
    )
    
    parser.add_argument(
        'input_path',
        help='Directory containing JSON files or glob pattern for input files'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_transformed.json',
        help='File pattern to match when input_path is a directory (default: *_transformed.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for filtered files (default: same as input file directory)'
    )
    
    parser.add_argument(
        '--precision',
        type=int,
        default=8,
        help='Decimal places for comparison (default: 8)'
    )
    
    parser.add_argument(
        '--report-file',
        type=str,
        default='camera_pose_frequency_report.txt',
        help='Output filename for frequency report (default: camera_pose_frequency_report.txt)'
    )
    
    return parser.parse_args()


def main():
    """Main function to process camera data files."""
    args = parse_arguments()
    
    # Validate precision
    if args.precision < 0:
        print("Error: Precision must be non-negative")
        sys.exit(1)
    
    # Determine if input_path is a directory or a glob pattern
    if os.path.isdir(args.input_path):
        # It's a directory - construct the glob pattern
        input_pattern = os.path.join(args.input_path, args.pattern)
        input_type = "directory"
        display_path = args.input_path
    else:
        # It's a glob pattern (or doesn't exist, will be caught by process_files)
        input_pattern = args.input_path
        input_type = "pattern"
        display_path = args.input_path
    
    print("="*80)
    print("CAMERA POSE DUPLICATE FILTER")
    print("="*80)
    if input_type == "directory":
        print(f"Input directory: {display_path}")
        print(f"File pattern: {args.pattern}")
    else:
        print(f"Input pattern: {display_path}")
    print(f"Precision: {args.precision} decimal places")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print(f"Report file: {args.report_file}")
    print()
    
    # Process all files
    all_pose_counters, all_pose_indices = process_files(input_pattern, args.output_dir, args.precision)
    
    # Generate frequency report
    print("="*80)
    print("Generating frequency report...")
    
    # Determine report path
    if args.output_dir:
        report_path = os.path.join(args.output_dir, args.report_file)
    else:
        report_path = args.report_file
    
    generate_report(all_pose_counters, all_pose_indices, report_path)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()



'''QUICK START GUIDE
=================

Your camera pose duplicate filter script is ready to use!

WHAT YOU GET:
-------------
1. filter_camera_duplicates.py - Main script
2. README.md - Complete documentation
3. usage_examples.sh - Example commands
4. test_output/ - Sample output from your CAM_FRONT_transformed.json

TEST RESULTS (from CAM_FRONT_transformed.json):
------------------------------------------------
✓ Original poses: 850
✓ Unique poses: 3
✓ Duplicates removed: 847 (99.6%)
✓ Generated: CAM_FRONT_transformed_unique.json
✓ Generated: camera_pose_frequency_report.txt

UNIQUE POSES FOUND:
-------------------
Pose #1: appears 452 times
  coord=(0.822006, -0.004755, 1.494913)
  roll=0.000000, pitch=0.012230, yaw=4.722650, fov=65.121585
  Indices: [59, 60, 61, 62, 63, ...]

Pose #2: appears 383 times
  coord=(0.800791, -0.015946, 1.510958)
  roll=0.000000, pitch=-0.000805, yaw=4.718074, fov=64.561479
  Indices: [0, 1, 2, 3, 4, 5, ...]

Pose #3: appears 15 times
  coord=(0.771000, 0.026000, 1.536000)
  roll=0.000000, pitch=-0.005585, yaw=4.707677, fov=64.709182
  Indices: [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135]

QUICK USAGE:
------------
# Process all *_transformed.json files in a directory:
python3 filter_camera_duplicates.py /path/to/transformed/

# With custom output directory:
python3 filter_camera_duplicates.py /path/to/transformed/ --output-dir ./unique_poses

# With custom file pattern:
python3 filter_camera_duplicates.py /path/to/transformed/ --pattern "CAM_FRONT*.json"

# Using current directory:
python3 filter_camera_duplicates.py . --output-dir ./results

# For your nuScenes data:
cd /mnt/share/felix/code/Sim2Real/camera_setup_comparison/nuscenes_camera_setup/
python3 /path/to/filter_camera_duplicates.py transformed/ \
    --output-dir transformed/unique_poses \
    --report-file nuscenes_pose_report.txt

OUTPUT FILES:
-------------
For each input file like CAM_FRONT_transformed.json:
  → CAM_FRONT_transformed_unique.json (only unique poses)
  
Plus one frequency report:
  → camera_pose_frequency_report.txt (or custom name)

The report shows:
  - How many unique poses exist per file
  - Each unique pose with full details
  - How many times each pose appears
  - All array indices where each pose occurs in the original file
  - Sorted by frequency (most common first)

WHAT THE SCRIPT DOES:
---------------------
1. Reads JSON files with camera pose data
2. Identifies unique poses based on: coordinates, roll, pitch, yaw, fov
3. Tracks all array indices where each unique pose appears
4. Filters out duplicate poses (keeps first occurrence)
5. Saves filtered JSON files with _unique suffix
6. Generates frequency report showing pose statistics and indices

KEY FEATURES:
-------------
✓ Batch processing with wildcards
✓ Configurable precision (--precision)
✓ Custom output directory (--output-dir)
✓ Detailed frequency reports
✓ Preserves JSON structure
✓ Shows statistics (duplicates removed, percentages)

For more details, see README.md'''