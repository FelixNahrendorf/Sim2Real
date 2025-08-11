#!/usr/bin/env python3

import json
import numpy as np
import argparse
import glob
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


def quaternion_to_euler(quaternion):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians using scipy for stability."""
    # Convert pyquaternion to scipy format (x, y, z, w)
    if isinstance(quaternion, Quaternion):
        quat_scipy = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    else:
        # Assume [w, x, y, z] format, convert to [x, y, z, w]
        quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    
    # Use scipy for numerically stable conversion
    r = Rotation.from_quat(quat_scipy)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    
    return roll, pitch, yaw


def compute_horizontal_fov(camera_intrinsic, image_width=1600):
    """Compute horizontal field of view from camera intrinsics using most stable approach."""
    # Extract focal length in x direction
    fl_x = camera_intrinsic[0][0]
    
    # Use atan2 for numerical stability
    half_fov = np.arctan2(image_width / 2.0, fl_x)
    horizontal_fov_rad = 2 * half_fov
    
    # Convert to degrees
    horizontal_fov_deg = np.degrees(horizontal_fov_rad)
    
    return horizontal_fov_deg


def extract_transform_data(transformed_json):
    """Extract coordinates, Euler angles, and FOV from transformed JSON."""
    coordinates = []
    rolls = []
    pitches = []
    yaws = []
    fovs = []
    
    for frame in transformed_json["frames"]:
        # Extract translation from transform matrix
        transform_matrix = np.array(frame["transform_matrix"])
        translation = transform_matrix[:3, 3]
        coordinates.append(translation.tolist())
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = transform_matrix[:3, :3]
        r = Rotation.from_matrix(rotation_matrix)
        quat_scipy = r.as_quat()  # Returns [x, y, z, w]
        
        # Convert to Euler angles
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
        
        # Compute FOV from intrinsics
        # Reconstruct camera intrinsic matrix
        camera_intrinsic = [
            [frame["fl_x"], 0.0, frame["cx"]],
            [0.0, frame["fl_y"], frame["cy"]],
            [0.0, 0.0, 1.0]
        ]
        fov = compute_horizontal_fov(camera_intrinsic, frame["w"])
        fovs.append(fov)
    
    return {
        "coordinates": coordinates,
        "rolls": rolls,
        "pitchs": pitches,  # Keep original spelling from mock file
        "yaws": yaws,
        "fov": fovs
    }


def process_camera_file(input_file, output_dir=None):
    """Process a single camera JSON file."""
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
    
    # Extract coordinates, Euler angles, and FOV
    output_data = extract_transform_data(transformed_json)
    
    # Generate output filename
    input_path = Path(input_file)
    output_filename = f"{input_path.stem}_transformed.json"
    
    if output_dir:
        output_path = Path(output_dir) / output_filename
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_path = input_path.parent / output_filename
    
    # Save output
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Saved transformed data to {output_path}")
    print(f"Processed {len(camera_list)} camera poses")


def main():
    parser = argparse.ArgumentParser(description='Transform camera pose data')
    parser.add_argument('input_path', 
                       help='Input folder path containing CAM_*.json files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for transformed files (default: same as input folder)')
    
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
    
    # Process each file
    for input_file in input_files:
        try:
            process_camera_file(str(input_file), output_dir)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":
    main()