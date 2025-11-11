"""
Extract the very first nuScenes scene from TEST dataset with:
- 6 camera images
- Original intrinsics/extrinsics in JSON format (matching nuscene_first_cameras_original_format.json)
- Transformed intrinsics/extrinsics in JSON format (matching nuscene_transform_matrix_format.json)
All outputs saved in a new directory
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion.quaternion import Quaternion
import torch

# Update these paths according to your setup
NUSCENE_DATA_DIR = "/app/datasets/nuscenes_full/"

# Camera sensor names (as defined in the original code)
desired_sensor_names = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT', 
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]

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

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    New transformation to match the format pixelsplat was trained on (format of seed4d/carla generator)
    """
    # Apply x-axis flip
    x_axis_flip = Quaternion(axis=[1, 0, 0], angle=np.pi)
    original_quaternion = rotation * x_axis_flip
        
    # Apply coordinate transformation
    transformed_translation = apply_coordinate_transformation(translation)
    transformed_quaternion = apply_rotation_transformation(original_quaternion)
        
    # Create transformation matrix
    transformed_transform_matrix = quaternion_to_transform_matrix(
        transformed_quaternion, transformed_translation
    )

    transformed_transform_matrix[0,1] *= -1  
    transformed_transform_matrix[0,2] *= -1
    transformed_transform_matrix[1,1] *= -1
    transformed_transform_matrix[1,2] *= -1
    transformed_transform_matrix[2,1] *= -1
    transformed_transform_matrix[2,2] *= -1

    return transformed_transform_matrix

def load_nuscene_frame_data(nusc, sample_token: str):
    """
    Load nuScenes frame data including images, intrinsics, and extrinsics.
    This is the _load_nuscene_frame_data method from the original code.
    """
    frame_information = []
    nuscene_frame_info = nusc.get("sample", sample_token)
    sample_frame_info = nuscene_frame_info["data"]
    
    for sensor in desired_sensor_names:
        sensor_data = nusc.get("sample_data", sample_frame_info[sensor])
        
        # Get ego pose
        ego_pose_info = nusc.get(table_name="ego_pose", token=sensor_data["ego_pose_token"])
        ego_pose_rotation = Quaternion(ego_pose_info["rotation"])
        ego_pose_translation = np.array(ego_pose_info["translation"])
        
        # Get sensor calibration
        sensor_pose_info = nusc.get(table_name="calibrated_sensor", 
                                    token=sensor_data["calibrated_sensor_token"])
        sensor_intrinsic = np.array(sensor_pose_info["camera_intrinsic"])
        sensor_pose_rotation = Quaternion(sensor_pose_info["rotation"])
        sensor_pose_translation = np.array(sensor_pose_info["translation"])
        
        # Compute transformation matrices
        sensor_transform = transform_matrix(sensor_pose_translation, sensor_pose_rotation)
        
        sensor_file_path = NUSCENE_DATA_DIR + sensor_data["filename"]
        
        # Normalize intrinsics
        intrinsic_normal = np.zeros((3, 3))
        intrinsic_normal[0, 0] = sensor_intrinsic[0, 0] / sensor_data["width"]
        intrinsic_normal[1, 1] = sensor_intrinsic[1, 1] / sensor_data["height"]
        intrinsic_normal[2, 2] = 1
        intrinsic_normal[0, 2] = sensor_intrinsic[0, 2] / sensor_data["width"]
        intrinsic_normal[1, 2] = sensor_intrinsic[1, 2] / sensor_data["height"]
        
        # Store original (non-transformed) data as well
        frame_information.append({
            'sensor_name': sensor,
            'image_path': sensor_file_path,
            'sample_data_token': sensor_data['token'],
            'calibrated_sensor_token': sensor_pose_info['token'],
            'original_intrinsic': sensor_intrinsic,
            'original_extrinsic_rotation': sensor_pose_rotation,
            'original_extrinsic_translation': sensor_pose_translation,
            'original_ego_rotation': ego_pose_rotation,
            'original_ego_translation': ego_pose_translation,
            'width': sensor_data["width"],
            'height': sensor_data["height"],
            'transformed_intrinsic': intrinsic_normal,
            'transformed_extrinsic': sensor_transform
        })
    
    return frame_information

def main():
    print("Initializing nuScenes TEST dataset...")
    version = 'v1.0-test'
    nusc = NuScenes(version=version, dataroot=NUSCENE_DATA_DIR)
    
    print("Getting test scenes...")
    all_splits = create_splits_scenes()
    test_scenes = all_splits['test']
    
    print(f"Found {len(test_scenes)} test scenes")
    print(f"First test scene name: {test_scenes[0]}")
    
    # Get the first sample token from the first test scene
    first_scene = None
    for scene in nusc.scene:
        if scene["name"] == test_scenes[0]:
            first_scene = scene
            break
    
    if first_scene is None:
        print("Error: Could not find the first test scene!")
        return
    
    first_sample_token = first_scene["first_sample_token"]
    print(f"First sample token: {first_sample_token}")
    
    # Load the frame data
    print("Loading frame data...")
    frame_data = load_nuscene_frame_data(nusc, first_sample_token)
    
    # Create output directory
    output_dir = Path("/app/outputs/first_nuscene_test_scene_output")
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Create subdirectory for images
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Save data
    print("Saving camera images...")
    
    for idx, cam_data in enumerate(frame_data):
        sensor_name = cam_data['sensor_name']
        print(f"\nProcessing {sensor_name}...")
        
        # Copy image
        image_src = cam_data['image_path']
        image_dst = images_dir / f"{idx:02d}_{sensor_name}.jpg"
        
        if os.path.exists(image_src):
            shutil.copy2(image_src, image_dst)
            print(f"  Copied image to {image_dst}")
        else:
            print(f"  Warning: Image not found at {image_src}")
    
    # Save original intrinsics/extrinsics in the format of nuscene_first_cameras_original_format.json
    print("\nSaving original intrinsics and extrinsics (JSON format)...")
    original_data = []
    for idx, cam_data in enumerate(frame_data):
        original_entry = {
            "token": cam_data['sample_data_token'],
            "sensor_token": cam_data['calibrated_sensor_token'],
            "translation": cam_data['original_extrinsic_translation'].tolist(),
            "rotation": [
                cam_data['original_extrinsic_rotation'].w,
                cam_data['original_extrinsic_rotation'].x,
                cam_data['original_extrinsic_rotation'].y,
                cam_data['original_extrinsic_rotation'].z
            ],
            "camera_intrinsic": cam_data['original_intrinsic'].tolist()
        }
        original_data.append(original_entry)
    
    with open(output_dir / "original_camera_params.json", 'w') as f:
        json.dump(original_data, f, indent=2)
    
    # Save transformed intrinsics/extrinsics in the format of nuscene_transform_matrix_format.json
    print("Saving transformed intrinsics and extrinsics (JSON format)...")
    transformed_data = {
        "camera_model": "OPENCV",
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }
    
    for idx, cam_data in enumerate(frame_data):
        sensor_name = cam_data['sensor_name']
        
        # Get the intrinsic parameters from the original (non-normalized) intrinsic matrix
        fl_x = cam_data['original_intrinsic'][0, 0]
        fl_y = cam_data['original_intrinsic'][1, 1]
        cx = cam_data['original_intrinsic'][0, 2]
        cy = cam_data['original_intrinsic'][1, 2]
        
        frame_entry = {
            "file_path": f"../images/{idx:02d}_{sensor_name}.jpg",
            "transform_matrix": cam_data['transformed_extrinsic'].tolist(),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": cam_data['width'],
            "h": cam_data['height'],
            "camera_name": sensor_name
        }
        transformed_data["frames"].append(frame_entry)
    
    with open(output_dir / "transformed_camera_params.json", 'w') as f:
        json.dump(transformed_data, f, indent=2)
    
    # Save summary info
    print("Saving summary...")
    scene_summary = {
        "dataset_version": "v1.0-test",
        "scene_name": first_scene['name'],
        "scene_token": first_scene['token'],
        "first_sample_token": first_sample_token,
        "number_of_cameras": len(frame_data),
        "camera_names": [d['sensor_name'] for d in frame_data],
        "output_directory": str(output_dir),
        "files": {
            "images": "images/",
            "original_parameters": "original_camera_params.json",
            "transformed_parameters": "transformed_camera_params.json"
        }
    }
    
    with open(output_dir / "scene_info.json", 'w') as f:
        json.dump(scene_summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - {len(frame_data)} camera images saved in: images/")
    print(f"  - Original intrinsics/extrinsics: original_camera_params.json")
    print(f"  - Transformed intrinsics/extrinsics: transformed_camera_params.json")
    print(f"  - Scene information: scene_info.json")
    print("=" * 80)

if __name__ == "__main__":
    main()