#!/usr/bin/env python3

import json
import numpy as np
import argparse
import os
from pathlib import Path
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion


def apply_coordinate_transformation(position):
    """
    Apply coordinate transformation:
    x_new = -z_old
    y_new = x_old
    z_new = -y_old
    """
    x_old, y_old, z_old = position
    return np.array([-z_old, x_old, -y_old])


def apply_rotation_transformation(quaternion):
    """
    Apply rotation transformation to match the coordinate system change
    """
    # Convert to rotation matrix
    rotation_matrix = quaternion.rotation_matrix
    
    # Transformation matrix
    T = np.array([
        [ 0,  0, -1],
        [ 1,  0,  0],
        [ 0, -1,  0]
    ])
    
    # Apply transformation: R_new = T * R_old
    transformed_rotation_matrix = T @ rotation_matrix
    
    # Convert back to quaternion
    transformed_quaternion = Quaternion(matrix=transformed_rotation_matrix)
    
    return transformed_quaternion


def quaternion_to_transform_matrix(quaternion, translation):
    """
    Convert quaternion and translation to 4x4 transformation matrix
    """
    rotation_matrix = quaternion.rotation_matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix


def transform_camera_poses(camera_input_data):
    """
    Transform camera pose data and return the transformed transforms JSON structure.
    
    Args:
        camera_input_data: Dictionary with camera data containing:
            - For each camera: {
                'translation': [x, y, z],
                'rotation': [w, x, y, z] or Quaternion object,
                'camera_intrinsic': 3x3 matrix or list,
                'image_width': int (optional, default 1600),
                'image_height': int (optional, default 900)
              }
    
    Returns:
        dict: Transformed transforms JSON structure
    """
    
    # Process each camera
    camera_data = {}
    for camera_name, data in camera_input_data.items():
        # Get original position and rotation
        original_translation = np.array(data['translation'])
        
        # Handle rotation input (could be quaternion object or list)
        if isinstance(data['rotation'], Quaternion):
            original_quaternion = data['rotation']
        else:
            # Assume [w, x, y, z] format
            rot = data['rotation']
            original_quaternion = Quaternion(w=rot[0], x=rot[1], y=rot[2], z=rot[3])
        
        # Apply x-axis flip
        x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
        original_quaternion = original_quaternion * x_axis_flip
        
        # Apply coordinate transformation
        transformed_translation = apply_coordinate_transformation(original_translation)
        transformed_quaternion = apply_rotation_transformation(original_quaternion)
        
        # Create transformation matrix
        transformed_transform_matrix = quaternion_to_transform_matrix(
            transformed_quaternion, transformed_translation
        )
        
        # Get camera intrinsics
        camera_intrinsic = np.array(data['camera_intrinsic'])
        fl_x = camera_intrinsic[0, 0]
        fl_y = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        
        # Get image dimensions (with defaults)
        w = data.get('image_width', 1600)
        h = data.get('image_height', 900)
        
        camera_data[camera_name] = {
            'transform_matrix': transformed_transform_matrix,
            'fl_x': float(fl_x),
            'fl_y': float(fl_y),
            'cx': float(cx),
            'cy': float(cy),
            'w': w,
            'h': h
        }
    
    # Create transformed transforms JSON structure
    transformed_json = {
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }
    
    # Process each camera in the data
    for idx, (camera_name, data) in enumerate(camera_data.items()):
        frame = {
            "file_path": f"../sensors/{idx}_rgb.png",
            "depth_file_path": f"../sensors/{idx}_depth.png",
            "semantic_segmentation_file_path": f"../sensors/{idx}_semantic_segmentation.png",
            "instance_segmentation_file_path": f"../sensors/{idx}_instance_segmentation.png",
            "transform_matrix": data['transform_matrix'].tolist(),
            "fl_x": data['fl_x'],
            "fl_y": data['fl_y'],
            "cx": data['cx'],
            "cy": data['cy'],
            "w": data['w'],
            "h": data['h'],
            "camera_name": camera_name
        }
        
        transformed_json["frames"].append(frame)
    
    return transformed_json


def process_single_camera_file(input_file, output_dir=None):
    """Process a single camera JSON file and create transformed output."""
    print(f"Processing {input_file}...")
    
    # Load input data
    with open(input_file, 'r') as f:
        camera_list = json.load(f)
    
    # Convert list format to dict format expected by transform_camera_poses
    camera_input_data = {}
    for i, camera_data in enumerate(camera_list):
        camera_name = f"camera_{i:03d}"
        camera_input_data[camera_name] = {
            'translation': camera_data['translation'],
            'rotation': camera_data['rotation'],  # [w, x, y, z] format
            'camera_intrinsic': camera_data['camera_intrinsic'],
            'image_width': 1600,  # Default value
            'image_height': 900   # Default value
        }
    
    # Transform camera poses
    transformed_json = transform_camera_poses(camera_input_data)
    
    # Generate output filename
    input_path = Path(input_file)
    output_filename = f"{input_path.stem}_ego_transform_matrix.json"
    
    if output_dir:
        output_path = Path(output_dir) / output_filename
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_path = input_path.parent / output_filename
    
    # Save output
    with open(output_path, 'w') as f:
        json.dump(transformed_json, f, indent=4)
    
    print(f"Saved transformed data to {output_path}")
    print(f"Processed {len(camera_list)} camera poses")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Transform individual camera pose data files')
    parser.add_argument('input_path', 
                       help='Input directory path containing CAM_*.json files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for transformed files (default: same as input directory)')
    
    args = parser.parse_args()
    
    # Check if input path is a directory
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path does not exist: {args.input_path}")
        return
    
    if not input_path.is_dir():
        print(f"Input path is not a directory: {args.input_path}")
        return
    
    # Find all CAM_*.json files in the directory
    input_files = list(input_path.glob("CAM_*.json"))
    
    if not input_files:
        print(f"No CAM_*.json files found in {args.input_path}")
        return
    
    print(f"Found {len(input_files)} CAM_*.json files in {args.input_path}")
    
    # Set output directory (default to input directory if not specified)
    output_dir = args.output_dir if args.output_dir else str(input_path)
    
    # Process each file individually
    processed_files = []
    for input_file in input_files:
        try:
            output_file = process_single_camera_file(str(input_file), output_dir)
            processed_files.append(output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {len(processed_files)} files:")
    for file_path in processed_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()