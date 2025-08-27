#!/usr/bin/env python3
"""
Transforms Ego JSON to CSV Extractor

Extracts transformation matrices and camera intrinsics from a single transforms_ego.json file
and outputs a CSV with mean values for TX, TY, TZ, R11-R33, fx, fy, cx, cy for each camera.

Usage:
    python3 transforms_ego_to_csv.py <transforms_ego.json> --output-file <output.csv>
    
Example:
    python3 transforms_ego_to_csv.py transforms_ego.json --output-file camera_parameters.csv
"""

import json
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
import sys

def load_transforms_ego_data(file_path):
    """Load transforms_ego data from JSON file and extract transformation matrix components and intrinsics."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract all frames
        frames = data['frames']
        
        # Define camera names in order from first to 6th frame
        camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                       'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        # Initialize dictionary to store camera data
        camera_data = {}
        
        # Process each frame (assuming 6 cameras)
        for i, frame in enumerate(frames[:6]):  # Limit to first 6 frames
            if i < len(camera_names):
                camera_name = camera_names[i]
                
                # Extract transformation matrix
                transform_matrix = np.array(frame['transform_matrix'])
                
                # Extract all parameters
                camera_params = {
                    'TX': transform_matrix[0, 3],
                    'TY': transform_matrix[1, 3],
                    'TZ': transform_matrix[2, 3],
                    'R11': transform_matrix[0, 0],
                    'R12': transform_matrix[0, 1],
                    'R13': transform_matrix[0, 2],
                    'R21': transform_matrix[1, 0],
                    'R22': transform_matrix[1, 1],
                    'R23': transform_matrix[1, 2],
                    'R31': transform_matrix[2, 0],
                    'R32': transform_matrix[2, 1],
                    'R33': transform_matrix[2, 2],
                    'fx': frame['fl_x'],
                    'fy': frame['fl_y'],
                    'cx': frame['cx'],
                    'cy': frame['cy']
                }
                
                camera_data[camera_name] = camera_params
        
        return camera_data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_csv_output(camera_data, output_file):
    """Create CSV file with camera parameters."""
    
    # Define parameter order
    param_order = ['TX', 'TY', 'TZ', 'R11', 'R12', 'R13', 'R21', 'R22', 'R23', 'R31', 'R32', 'R33', 'fx', 'fy', 'cx', 'cy']
    
    # Prepare data for CSV
    csv_data = []
    
    for camera_name, params in camera_data.items():
        for param in param_order:
            if param in params:
                row = {
                    'Camera': camera_name,
                    'Parameter': param,
                    'Mean': params[param],  # Using the single value as mean
                    'Std_Dev': 0.0,         # No variation since it's a single value
                    'Min': params[param],   # Same as mean
                    'Max': params[param],   # Same as mean
                    'Median': params[param] # Same as mean
                }
                csv_data.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"✓ Camera parameters saved to '{output_file}'")
    
    # Print summary
    print(f"\n✓ Processed {len(camera_data)} cameras:")
    for camera_name in camera_data.keys():
        print(f"  - {camera_name}")
    print(f"✓ Extracted {len(param_order)} parameters per camera")
    print(f"✓ Total rows in CSV: {len(csv_data)}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract camera parameters from transforms_ego.json to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 transforms_ego_to_csv.py transforms_ego.json --output-file camera_params.csv
  python3 transforms_ego_to_csv.py /path/to/transforms_ego.json --output-file /path/to/output.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to transforms_ego.json file'
    )
    
    parser.add_argument(
        '--output-file',
        required=True,
        help='Output CSV file path'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {args.input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input file: {input_path.absolute()}")
    print(f"Output file: {output_path.absolute()}")
    
    # Load and process the transforms_ego data
    print(f"\nLoading transforms_ego data from: {args.input_file}")
    camera_data = load_transforms_ego_data(args.input_file)
    
    if not camera_data:
        print("No valid camera data found. Please check your file format.")
        sys.exit(1)
    
    # Create CSV output
    print("\nGenerating CSV output...")
    create_csv_output(camera_data, args.output_file)
    
    print(f"\n✓ Processing complete!")

if __name__ == "__main__":
    main()