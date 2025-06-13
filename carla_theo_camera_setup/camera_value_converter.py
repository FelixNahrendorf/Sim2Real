#!/usr/bin/env python3
import json
import math
import argparse
import os
import glob
from typing import Dict, List, Tuple, Any

def extract_position_from_transform(transform_matrix: List[List[float]]) -> List[float]:
    """Extract x, y, z position from 4x4 transformation matrix"""
    return [transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3]]

def compute_pitch_yaw_from_transform(transform_matrix: List[List[float]]) -> Tuple[float, float]:
    """Compute pitch and yaw angles from transformation matrix"""
    # Extract rotation matrix elements (top-left 3x3)
    r00, r01, r02 = transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2]
    r10, r11, r12 = transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2]
    r20, r21, r22 = transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2]
    
    # Extract Euler angles (ZYX convention)
    # Pitch (rotation around X-axis)
    pitch = math.atan2(-r12, r22)
    
    # Yaw (rotation around Z-axis)
    yaw = math.atan2(-r01, r00)
    
    return pitch, yaw

def compute_fov_from_focal_length(fl_x: float, fl_y: float, width: int, height: int) -> float:
    """Compute field of view from focal length and image dimensions"""
    # Compute horizontal FOV in degrees
    fov_x = 2 * math.atan(width / (2 * fl_x)) * 180 / math.pi
    
    # Return the horizontal FOV (commonly used)
    return fov_x

def convert_single_file(input_file: str, output_dir: str) -> bool:
    """Convert a single transforms.json file to target format"""
    
    try:
        # Read input file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        pitchs = []
        yaws = []
        fovs = []
        
        # Get camera parameters - check if they're at top level or frame level
        if 'fl_x' in data:
            # Camera parameters are at top level (new format)
            global_fl_x = data['fl_x']
            global_fl_y = data['fl_y']
            global_w = data['w']
            global_h = data['h']
            use_global_params = True
        else:
            use_global_params = False
        
        # Process each frame
        for frame in data['frames']:
            # Extract position (x, y, z)
            position = extract_position_from_transform(frame['transform_matrix'])
            coordinates.append(position)
            
            # Compute pitch and yaw
            pitch, yaw = compute_pitch_yaw_from_transform(frame['transform_matrix'])
            pitchs.append(pitch)
            yaws.append(yaw)
            
            # Compute FOV - use global or frame-specific parameters
            if use_global_params:
                fov = compute_fov_from_focal_length(
                    global_fl_x, global_fl_y, 
                    global_w, global_h
                )
            else:
                # Frame-specific parameters (old format)
                fov = compute_fov_from_focal_length(
                    frame['fl_x'], frame['fl_y'], 
                    frame['w'], frame['h']
                )
            fovs.append(fov)
        
        # Create output data structure
        output_data = {
            "coordinates": coordinates,
            "pitchs": pitchs,
            "yaws": yaws,
            "fov": fovs
        }
        
        # Create output filename
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{input_filename}_converted.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write output file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"✓ Converted: {input_file} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_file}: {str(e)}")
        return False

def main() -> int:
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Convert camera transform JSON files to target format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 camera_value_converter.py "CAM_*.json" --output-dir ./processed
  python3 camera_value_converter.py transforms.json --output-dir ./output
  python3 camera_value_converter.py "data/*.json" --output-dir ./results
        '''
    )
    
    parser.add_argument(
        'input_pattern',
        help='Input file pattern (supports glob patterns like "CAM_*.json" or single file)'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for converted files'
    )
    
    args = parser.parse_args()
    
    # Find matching files using glob pattern
    input_files = glob.glob(args.input_pattern)
    
    if not input_files:
        print(f"No files found matching pattern: {args.input_pattern}")
        return 1
    
    print(f"Found {len(input_files)} file(s) to process:")
    for file in input_files:
        print(f"  - {file}")
    print()
    
    # Process each file
    success_count = 0
    for input_file in input_files:
        if convert_single_file(input_file, args.output_dir):
            success_count += 1
    
    # Summary
    print(f"\nProcessing complete:")
    print(f"  Successful: {success_count}/{len(input_files)}")
    print(f"  Output directory: {args.output_dir}")
    
    return 0 if success_count == len(input_files) else 1

if __name__ == "__main__":
    exit(main())
