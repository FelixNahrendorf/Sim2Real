#!/usr/bin/env python3
"""
Generate combined camera pose JSON files from frequency report.
Extracts ALL data directly from the report - no need for original JSON files.
"""

import json
import sys
import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# Camera order for output JSON files
CAMERA_ORDER = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]


def parse_pose_values(pose_line: str) -> Tuple[List[float], float, float, float, float]:
    """
    Parse pose values from a line like:
    coord=(0.822006, -0.004755, 1.494913), roll=0.000000, pitch=0.012230, yaw=4.722650, fov=65.121585
    
    Returns:
        Tuple of (coordinates, roll, pitch, yaw, fov)
    """
    # Extract coordinates
    coord_match = re.search(r'coord=\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)', pose_line)
    if not coord_match:
        return None
    
    coordinates = [float(coord_match.group(1)), float(coord_match.group(2)), float(coord_match.group(3))]
    
    # Extract roll, pitch, yaw, fov
    roll_match = re.search(r'roll=([-\d.]+)', pose_line)
    pitch_match = re.search(r'pitch=([-\d.]+)', pose_line)
    yaw_match = re.search(r'yaw=([-\d.]+)', pose_line)
    fov_match = re.search(r'fov=([-\d.]+)', pose_line)
    
    if not all([roll_match, pitch_match, yaw_match, fov_match]):
        return None
    
    roll = float(roll_match.group(1))
    pitch = float(pitch_match.group(1))
    yaw = float(yaw_match.group(1))
    fov = float(fov_match.group(1))
    
    return coordinates, roll, pitch, yaw, fov


def parse_report_file(report_path: str) -> Dict[str, Dict[int, Tuple]]:
    """
    Parse the frequency report to extract camera poses with their values and indices.
    
    Args:
        report_path: Path to the frequency report text file
        
    Returns:
        Dictionary mapping camera name to {pose_id: (pose_values, set of indices)}
        where pose_values = (coordinates, roll, pitch, yaw, fov)
    """
    camera_data = defaultdict(dict)
    
    try:
        with open(report_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Report file '{report_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading report file: {e}")
        sys.exit(1)
    
    # Split by file sections
    file_sections = re.split(r'File: (.*?_transformed.*?\.json)', content)
    
    for i in range(1, len(file_sections), 2):
        camera_filepath = file_sections[i]
        section_content = file_sections[i + 1] if i + 1 < len(file_sections) else ""
        
        # Extract camera name (e.g., CAM_FRONT from CAM_FRONT_transformed.json)
        camera_filename = os.path.basename(camera_filepath)
        camera_name = camera_filename.replace('_transformed_unique.json', '').replace('_transformed.json', '')
        
        # Find all pose sections with values and indices
        # Pattern to match: Pose #N (appears X times):\n  coord=(...), roll=..., pitch=..., yaw=..., fov=...\n  Indices: [...]
        pose_pattern = r'Pose #(\d+) \(appears \d+ times\):\s*([^\n]+)\s*Indices: \[([\d,\s]+)\]'
        pose_matches = re.finditer(pose_pattern, section_content, re.DOTALL)
        
        for match in pose_matches:
            pose_id = int(match.group(1))
            pose_values_line = match.group(2)
            indices_str = match.group(3)
            
            # Parse pose values
            pose_values = parse_pose_values(pose_values_line)
            if pose_values is None:
                print(f"Warning: Could not parse pose values for {camera_name} Pose #{pose_id}")
                continue
            
            # Parse indices
            indices = set()
            for num_str in re.findall(r'\d+', indices_str):
                indices.add(int(num_str))
            
            camera_data[camera_name][pose_id] = (pose_values, indices)
    
    return camera_data


def find_matching_indices(camera_data: Dict[str, Dict[int, Tuple]]) -> Dict[Tuple, Tuple]:
    """
    Find indices that match across all cameras.
    
    Args:
        camera_data: Dictionary mapping camera name to {pose_id: (pose_values, set of indices)}
        
    Returns:
        Dictionary mapping (cam1_pose, cam2_pose, ...) tuple to (pose_values_tuple, set of matching indices)
        where pose_values_tuple = ((cam1_values), (cam2_values), ...)
    """
    # Get all possible index values
    all_indices = set()
    for camera_poses in camera_data.values():
        for pose_values, indices in camera_poses.values():
            all_indices.update(indices)
    
    # For each index, find which pose it belongs to in each camera
    matching_combinations = defaultdict(lambda: ([], set()))
    
    for idx in sorted(all_indices):
        pose_combination = []
        pose_values_list = []
        valid = True
        
        for camera_name in CAMERA_ORDER:
            if camera_name not in camera_data:
                valid = False
                break
            
            # Find which pose this index belongs to in this camera
            found_pose = None
            found_values = None
            for pose_id, (pose_values, indices) in camera_data[camera_name].items():
                if idx in indices:
                    found_pose = pose_id
                    found_values = pose_values
                    break
            
            if found_pose is None:
                valid = False
                break
            
            pose_combination.append(found_pose)
            pose_values_list.append(found_values)
        
        if valid:
            pose_tuple = tuple(pose_combination)
            if not matching_combinations[pose_tuple][0]:  # First time seeing this combination
                matching_combinations[pose_tuple] = (pose_values_list, set())
            matching_combinations[pose_tuple][1].add(idx)
    
    return matching_combinations


def generate_combined_json(
    pose_values_list: List[Tuple],
    output_path: str
) -> bool:
    """
    Generate a combined JSON file for a specific pose combination.
    
    Args:
        pose_values_list: List of (coordinates, roll, pitch, yaw, fov) for each camera
        output_path: Path to save the combined JSON file
        
    Returns:
        True if successful, False otherwise
    """
    combined_data = {
        'coordinates': [],
        'rolls': [],
        'pitchs': [],
        'yaws': [],
        'fov': []
    }
    
    # Extract values for each camera
    for pose_values in pose_values_list:
        coordinates, roll, pitch, yaw, fov = pose_values
        combined_data['coordinates'].append(coordinates)
        combined_data['rolls'].append(roll)
        combined_data['pitchs'].append(pitch)
        combined_data['yaws'].append(yaw)
        combined_data['fov'].append(fov)
    
    # Save the combined JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(combined_data, f, indent=4)
        return True
    except Exception as e:
        print(f"  Error saving file '{output_path}': {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate combined camera pose JSON files from frequency report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_combined_poses.py report.txt
  python3 generate_combined_poses.py report.txt --output-dir ./scenes
  python3 generate_combined_poses.py /path/to/camera_report.txt --output-dir ./output

Note: All data is extracted directly from the report file.
      No need for original JSON files!
        """
    )
    
    parser.add_argument(
        'report_file',
        help='Path to the camera pose frequency report text file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./combined_scenes',
        help='Output directory for combined scene JSON files (default: ./combined_scenes)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    print("="*80)
    print("COMBINED CAMERA POSE GENERATOR")
    print("="*80)
    print(f"Report file: {args.report_file}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Parse the report file (extracts all pose values directly)
    print("Parsing frequency report...")
    camera_data = parse_report_file(args.report_file)
    
    if not camera_data:
        print("Error: No camera data found in report file.")
        sys.exit(1)
    
    print(f"Found data for {len(camera_data)} cameras:")
    for camera_name in sorted(camera_data.keys()):
        num_poses = len(camera_data[camera_name])
        print(f"  {camera_name}: {num_poses} unique poses")
    print()
    
    # Find matching indices across all cameras
    print("Finding matching pose combinations...")
    matching_combinations = find_matching_indices(camera_data)
    
    print(f"Found {len(matching_combinations)} unique pose combinations")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate combined JSON files
    print("Generating combined scene JSON files...")
    successful = 0
    
    for i, (pose_combination, (pose_values_list, indices)) in enumerate(sorted(matching_combinations.items()), 1):
        output_filename = f"scene_{i:04d}.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        pose_str = ", ".join(f"P{p}" for p in pose_combination)
        print(f"Scene {i:04d}: Poses [{pose_str}] - {len(indices)} occurrences", end="")
        
        success = generate_combined_json(pose_values_list, output_path)
        
        if success:
            print(f" ✓ {output_filename}")
            successful += 1
        else:
            print(" ✗ Failed")
    
    print()
    print("="*80)
    print(f"Successfully generated {successful}/{len(matching_combinations)} scene files")
    print(f"Output directory: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()