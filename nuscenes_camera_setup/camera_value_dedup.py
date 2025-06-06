#!/usr/bin/env python3
"""
Script to filter out duplicate camera setups from JSON files.
Each setup is defined by coordinates (x,y,z), pitch, yaw, and fov values.
"""

import json
import sys
import os
import glob
import argparse
from typing import Dict, List, Tuple, Any


def read_camera_data(file_path: str) -> Dict[str, List]:
    """
    Read camera data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing coordinates, pitchs, yaws, and fov lists
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate required keys
        required_keys = ['coordinates', 'pitchs', 'yaws', 'fov']
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


def create_setup_key(coord: List[float], pitch: float, yaw: float, fov: float, precision: int = 8) -> Tuple:
    """
    Create a hashable key for a camera setup by rounding values to specified precision.
    
    Args:
        coord: [x, y, z] coordinates
        pitch: Pitch angle
        yaw: Yaw angle
        fov: Field of view
        precision: Number of decimal places for rounding
        
    Returns:
        Tuple representing the unique setup
    """
    return (
        round(coord[0], precision),
        round(coord[1], precision),
        round(coord[2], precision),
        round(pitch, precision),
        round(yaw, precision),
        round(fov, precision)
    )


def filter_duplicates(data: Dict[str, List], precision: int = 8) -> Dict[str, List]:
    """
    Filter out duplicate camera setups.
    
    Args:
        data: Dictionary containing camera data
        precision: Number of decimal places for comparison
        
    Returns:
        Dictionary with duplicates removed
    """
    coordinates = data['coordinates']
    pitchs = data['pitchs']
    yaws = data['yaws']
    fovs = data['fov']
    
    # Validate that all arrays have the same length
    lengths = [len(coordinates), len(pitchs), len(yaws), len(fovs)]
    if len(set(lengths)) > 1:
        raise ValueError(f"Array lengths don't match: coordinates={len(coordinates)}, "
                        f"pitchs={len(pitchs)}, yaws={len(yaws)}, fov={len(fovs)}")
    
    seen_setups = set()
    unique_indices = []
    
    for i in range(len(coordinates)):
        setup_key = create_setup_key(coordinates[i], pitchs[i], yaws[i], fovs[i], precision)
        
        if setup_key not in seen_setups:
            seen_setups.add(setup_key)
            unique_indices.append(i)
    
    # Create filtered data
    filtered_data = {
        'coordinates': [coordinates[i] for i in unique_indices],
        'pitchs': [pitchs[i] for i in unique_indices],
        'yaws': [yaws[i] for i in unique_indices],
        'fov': [fovs[i] for i in unique_indices]
    }
    
    return filtered_data


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
        print(f"Filtered data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file '{output_path}': {e}")
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
    
    print(f"\nFiltering Statistics:")
    print(f"Original setups: {original_count}")
    print(f"Unique setups: {filtered_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    if original_count > 0:
        print(f"Duplicate percentage: {(duplicates_removed / original_count) * 100:.1f}%")


def process_single_file(input_file: str, output_dir: str = None, precision: int = 8) -> Tuple[int, int]:
    """
    Process a single camera data file.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save output file (optional)
        precision: Decimal places for comparison
        
    Returns:
        Tuple of (original_count, filtered_count)
    """
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = f"{base_name}_filtered.json"
    
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_filename)
    else:
        # Save in same directory as input file
        input_dir = os.path.dirname(input_file)
        output_file = os.path.join(input_dir, output_filename) if input_dir else output_filename
    
    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Read and process data
    original_data = read_camera_data(input_file)
    filtered_data = filter_duplicates(original_data, precision)
    
    # Save results
    save_filtered_data(filtered_data, output_file)
    print_stats(original_data, filtered_data)
    
    return len(original_data['coordinates']), len(filtered_data['coordinates'])


def expand_file_patterns(patterns: List[str]) -> List[str]:
    """
    Expand file patterns using glob to get actual file paths.
    
    Args:
        patterns: List of file patterns (may include wildcards)
        
    Returns:
        List of actual file paths
    """
    all_files = []
    for pattern in patterns:
        matched_files = glob.glob(pattern)
        if matched_files:
            all_files.extend(matched_files)
        else:
            # If no glob matches, check if it's a regular file
            if os.path.isfile(pattern):
                all_files.append(pattern)
            else:
                print(f"Warning: No files found matching pattern '{pattern}'")
    
    # Remove duplicates and sort
    return sorted(list(set(all_files)))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter out duplicate camera setups from JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_processor.py CAM_FRONT.json
  python camera_processor.py "CAM_*.json" --output-dir ./processed
  python camera_processor.py CAM_FRONT.json CAM_BACK.json --output-dir ./output
  python camera_processor.py "*.json" --precision 6 --output-dir ./filtered
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input JSON files or glob patterns (e.g., CAM_*.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for filtered files (default: same as input file directory)'
    )
    
    parser.add_argument(
        '--precision',
        type=int,
        default=8,
        help='Decimal places for comparison (default: 8)'
    )
    
    return parser.parse_args()


def main():
    """Main function to process camera data files."""
    args = parse_arguments()
    
    # Validate precision
    if args.precision < 0:
        print("Error: Precision must be non-negative")
        sys.exit(1)
    
    print(f"Precision: {args.precision} decimal places")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print(f"Input patterns: {args.input_files}")
    
    # Expand file patterns
    input_files = expand_file_patterns(args.input_files)
    
    if not input_files:
        print("Error: No files found matching the specified patterns.")
        sys.exit(1)
    
    print(f"Found {len(input_files)} file(s) to process:")
    for file in input_files:
        print(f"  - {file}")
    print()
    
    # Process all files
    total_original = 0
    total_filtered = 0
    
    for i, input_file in enumerate(input_files, 1):
        print(f"=== Processing file {i}/{len(input_files)} ===")
        try:
            original_count, filtered_count = process_single_file(
                input_file, 
                args.output_dir, 
                args.precision
            )
            total_original += original_count
            total_filtered += filtered_count
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
        print()
    
    # Print summary statistics
    if len(input_files) > 1:
        total_duplicates = total_original - total_filtered
        print("=== SUMMARY ===")
        print(f"Total files processed: {len(input_files)}")
        print(f"Total original setups: {total_original}")
        print(f"Total unique setups: {total_filtered}")
        print(f"Total duplicates removed: {total_duplicates}")
        if total_original > 0:
            print(f"Overall duplicate percentage: {(total_duplicates / total_original) * 100:.1f}%")


if __name__ == "__main__":
    main()
